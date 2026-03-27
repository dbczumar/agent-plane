"""MCP server connections with tool discovery and caching.

Manages connections to MCP servers (stdio and HTTP transports),
discovers tools via the MCP ``tools/list`` protocol, and caches
discovery results with a configurable TTL to avoid repeated
round-trips on every workflow execution.

Each ``McpServerConnection`` wraps a single MCP server. The
``ToolManager`` creates one per ``MCPServerConfig`` in the agent
spec, calls ``connect()`` at workflow start, and ``close()`` in
the finally block.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import random
import threading
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from cachetools import TTLCache
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.shared.message import SessionMessage
from mcp.types import CONNECTION_CLOSED, CallToolResult, ContentBlock, TextContent
from mcp.types import Tool as McpToolDef

from agent_plane.spec.types import MCPServerConfig, RetryConfig
from agent_plane.tools.base import Tool

_T = TypeVar("_T")

# Type aliases for the (read, write) stream pair returned by MCP
# transports. Uses anyio's concrete stream types parameterized
# over the MCP session message type.
_ReadStream = MemoryObjectReceiveStream[SessionMessage | Exception]
_WriteStream = MemoryObjectSendStream[SessionMessage]

_logger = logging.getLogger(__name__)

# Maximum time to wait for the background event loop thread
# to stop during shutdown, in seconds.
_LOOP_STOP_TIMEOUT_SECONDS = 5

# Default retry policy for MCP connection-level reconnection
# (transport died, server crashed). Separate from the tool-level
# retry in workflow.py, which handles call timeouts. These
# defaults give a flaky server ~6s to restart (1s + 2s + 4s
# backoff with jitter across 3 reconnect attempts).
_MCP_RECONNECT_DEFAULTS = RetryConfig(
    max_attempts=3,
    backoff_base=1.0,
    backoff_max=10.0,
)


class EventLoopThread:
    """
    Persistent event loop running in a daemon thread.

    MCP sessions are bound to the event loop they were created
    on — using a fresh ``asyncio.run()`` per call would close the
    loop and kill the session between ``connect()`` and
    ``call_tool()``. This class keeps a single loop alive so
    that ``connect()``, ``call_tool()``, and ``close()`` all
    execute on the same loop.

    Owned by ``ToolManager``: created at ``start()``, stopped at
    ``shutdown()``.
    """

    def __init__(self) -> None:
        """
        Create and start the background event loop thread.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
        )
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """
        Submit a coroutine to the persistent loop and block
        until it completes.

        :param coro: An awaitable object, e.g.
            ``conn.call_tool("name", {})``.
        :returns: The coroutine's return value.
        """
        future: concurrent.futures.Future[_T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        """
        Stop the event loop and join the background thread.

        Safe to call multiple times.
        """
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=_LOOP_STOP_TIMEOUT_SECONDS)
        if not self._loop.is_closed():
            self._loop.close()


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """
    Run an async coroutine from synchronous code (one-shot).

    Creates a temporary event loop, runs the coroutine, and
    closes the loop. Suitable for isolated async calls that
    don't need a persistent session (e.g. tests). For MCP
    tool invocation, use :class:`EventLoopThread` instead —
    MCP sessions are bound to the loop they were created on.

    If an event loop is already running on the current thread
    (e.g. inside ``pytest-asyncio``), spawns a background
    thread with its own loop.

    :param coro: An awaitable object.
    :returns: The coroutine's return value.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running — safe to create one directly.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # An event loop is already running on this thread.
    # Run the coroutine in a separate thread with its own loop.
    return _run_in_thread(coro)


def _run_in_thread(coro: Coroutine[Any, Any, _T]) -> _T:
    """
    Run an async coroutine in a separate thread with its own
    event loop.

    Used by ``_run_async()`` when the calling thread already
    has a running event loop.

    :param coro: An awaitable object.
    :returns: The coroutine's return value.
    """

    def _target() -> _T:
        """
        Thread target that creates a fresh event loop.

        :returns: The coroutine's return value.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_target)
        return future.result()


# Default discovery cache TTL in seconds (5 minutes).
_DEFAULT_CACHE_TTL_SECONDS = 300

# Maximum number of MCP server discovery results to cache.
# Each entry is lightweight (a list of tool definitions), so 64
# is generous for any realistic deployment.
_DEFAULT_CACHE_MAX_SIZE = 64

# Module-level discovery cache: bounded LRU with TTL expiration
# (via cachetools.TTLCache). Keyed by a stable server identity
# string (see _cache_key). Survives across ToolManager instances
# so sequential workflow executions against the same agent avoid
# redundant tools/list round-trips.
_discovery_cache: TTLCache[str, list[McpToolDef]] = TTLCache(
    maxsize=_DEFAULT_CACHE_MAX_SIZE,
    ttl=_DEFAULT_CACHE_TTL_SECONDS,
)


def _cache_key(config: MCPServerConfig) -> str:
    """
    Build a stable cache key for an MCP server config.

    Uses the server name + transport-specific identity (command+args
    for stdio, url for http) so that two configs pointing at the
    same server share the cache entry.

    :param config: The MCP server configuration.
    :returns: A string suitable as a dict key.
    """
    if config.transport == "stdio":
        return f"stdio:{config.name}:{config.command}:{config.args}"
    # http transport — url is the identity
    return f"http:{config.name}:{config.url}"


def clear_discovery_cache() -> None:
    """
    Clear all cached MCP tool discovery results.

    Useful in tests to ensure a clean state.
    """
    _discovery_cache.clear()


@dataclass
class McpServerConnection:
    """
    Manages the lifecycle of a single MCP server connection.

    Handles both stdio and HTTP transports. On ``connect()``,
    establishes the transport, initializes the MCP session, and
    discovers tools (from cache if fresh, otherwise via
    ``tools/list``). On ``close()``, tears down the session and
    transport.

    :param config: The MCP server configuration from the agent
        spec, e.g. ``MCPServerConfig(name="github",
        transport="stdio", command="npx", ...)``.
    :param work_dir: Working directory for stdio subprocess
        execution. When set, the subprocess ``cwd`` is set to
        this path so that relative command paths (e.g.
        ``"tools/mcp/server.py"``) resolve correctly. ``None``
        inherits the parent process CWD.
    """

    config: MCPServerConfig
    work_dir: Path | None = None
    _session: ClientSession | None = field(default=None, init=False, repr=False)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _discovered_tools: list[McpToolDef] = field(default_factory=list, init=False, repr=False)

    async def connect(self) -> list[McpToolDef]:
        """
        Establish the MCP connection and discover tools.

        Always opens a live transport and session so that
        ``call_tool()`` works after ``connect()`` returns. If
        the discovery cache is fresh, skips the ``tools/list``
        round-trip and returns cached tool definitions. Otherwise
        performs a live ``tools/list`` call and updates the cache.

        :returns: List of MCP tool definitions exposed by this
            server.
        :raises ValueError: If the transport type is not
            ``"stdio"`` or ``"http"``.
        """
        await self._open_session()
        return await self._discover_or_use_cache()

    async def call_tool(
        self,
        name: str,
        # Values are Any because MCP tool arguments are JSON
        # objects with heterogeneous value types (str, int,
        # bool, nested dicts, etc.). Matches the MCP SDK's
        # own ClientSession.call_tool() signature.
        arguments: dict[str, Any],
    ) -> str:
        """
        Invoke a tool on this MCP server.

        If the call fails due to a dead connection (server
        process crashed, transport dropped), reconnects with
        exponential backoff and retries. The retry policy is
        taken from ``config.retry`` (falling back to
        ``_MCP_RECONNECT_DEFAULTS``). Permanent errors (invalid
        args, tool not found) are not retried.

        :param name: The tool name as returned by discovery.
        :param arguments: The tool arguments dict (already parsed
            from the LLM's JSON string).
        :returns: The tool result as a string. For multi-content
            results, text blocks are joined with newlines.
        :raises RuntimeError: If ``connect()`` has not been called.
        """
        if self._session is None:
            raise RuntimeError(
                f"MCP server {self.config.name!r} has no live "
                f"session — call connect() before call_tool()"
            )
        retry = self.config.retry or _MCP_RECONNECT_DEFAULTS
        return await _call_tool_with_reconnect(
            conn=self,
            name=name,
            arguments=arguments,
            retry=retry,
        )

    async def _invoke_tool(
        self,
        name: str,
        arguments: dict[str, Any],  # JSON values — see call_tool
    ) -> str:
        """
        Send a single ``tools/call`` request to the MCP session.

        :param name: The tool name.
        :param arguments: The tool arguments dict.
        :returns: The formatted tool result string.
        """
        assert self._session is not None
        result = await self._session.call_tool(name=name, arguments=arguments)
        return _format_call_result(result)

    async def _reconnect(self) -> None:
        """
        Tear down the dead session and open a fresh one.

        Called by ``call_tool()`` after detecting a connection
        error. Does not re-discover tools — the tool list from
        the original ``connect()`` is still valid.
        """
        await self.close()
        await self._open_session()

    async def _open_session(self) -> None:
        """
        Open transport and initialize the MCP session.

        Always creates a live session regardless of cache state,
        so that ``call_tool()`` works after ``connect()``.
        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        read_stream, write_stream = await self._open_transport()
        # Session-level read timeout applies to initialize(),
        # list_tools(), and any call_tool() that doesn't pass
        # its own per-call timeout. Falls back to the MCP SDK
        # default (no timeout) when config.timeout is None.
        session_timeout = (
            timedelta(seconds=self.config.timeout) if self.config.timeout is not None else None
        )
        self._session = ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=session_timeout,
        )
        # Enter the session's async context via the exit stack
        # so it gets cleaned up on close().
        await self._exit_stack.enter_async_context(self._session)
        await self._session.initialize()

    async def _discover_or_use_cache(
        self,
    ) -> list[McpToolDef]:
        """
        Return tool definitions from cache or live discovery.

        If the cache is fresh, returns cached definitions without
        calling ``tools/list``. Otherwise performs a live
        ``tools/list`` call and updates the cache.

        Must be called after ``_open_session()`` so that
        ``self._session`` is live.

        :returns: List of MCP tool definitions.
        """
        cached = self._check_cache()
        if cached is not None:
            self._discovered_tools = cached
            _logger.debug(
                "MCP server %r: using cached discovery (%d tools)",
                self.config.name,
                len(cached),
            )
            return cached

        # Guaranteed by _open_session() which runs before this.
        assert self._session is not None
        tools_result = await self._session.list_tools()
        self._discovered_tools = tools_result.tools
        self._update_cache(tools_result.tools)
        _logger.info(
            "MCP server %r: discovered %d tool(s)",
            self.config.name,
            len(tools_result.tools),
        )
        return tools_result.tools

    async def close(self) -> None:
        """
        Tear down the MCP session and transport.

        Safe to call multiple times or if ``connect()`` was never
        called.
        """
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None

    @property
    def discovered_tools(self) -> list[McpToolDef]:
        """
        Tools discovered on the last ``connect()`` call.

        :returns: The list of MCP tool definitions. Empty if
            ``connect()`` has not been called.
        """
        return self._discovered_tools

    def _check_cache(self) -> list[McpToolDef] | None:
        """
        Return cached discovery if still fresh, else ``None``.

        TTLCache handles expiry internally — ``get()`` returns
        ``None`` for expired or absent entries.

        :returns: Cached tool list or ``None`` if expired or
            absent.
        """
        key = _cache_key(self.config)
        # TTLCache lacks type stubs, so .get() returns Any.
        result: list[McpToolDef] | None = _discovery_cache.get(key)
        return result

    def _update_cache(self, tools: list[McpToolDef]) -> None:
        """
        Store discovery results in the module-level cache.

        TTLCache tracks insertion time internally and evicts
        entries after the configured TTL.

        :param tools: The freshly discovered tool definitions.
        """
        key = _cache_key(self.config)
        _discovery_cache[key] = tools

    async def _open_transport(
        self,
    ) -> tuple[_ReadStream, _WriteStream]:
        """
        Open the MCP transport and return (read, write) streams.

        The transport's async context is registered on the exit
        stack so it gets torn down on ``close()``.

        :returns: A ``(read_stream, write_stream)`` tuple of
            anyio memory object streams parameterized over
            ``SessionMessage``.
        :raises ValueError: If the transport type is unsupported.
        """
        if self.config.transport == "stdio":
            return await self._open_stdio()
        if self.config.transport == "http":
            return await self._open_http()
        raise ValueError(f"Unsupported MCP transport: {self.config.transport!r}")

    async def _open_stdio(
        self,
    ) -> tuple[_ReadStream, _WriteStream]:
        """
        Open a stdio MCP transport.

        :returns: A ``(read_stream, write_stream)`` tuple from
            the stdio client context manager.
        """
        assert self._exit_stack is not None
        assert self.config.command is not None
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
            # Set cwd so relative command paths (e.g.
            # "tools/mcp/server.py") resolve from the agent's
            # extracted working directory.
            cwd=self.work_dir,
        )
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        return read_stream, write_stream

    async def _open_http(
        self,
    ) -> tuple[_ReadStream, _WriteStream]:
        """
        Open an HTTP (SSE) MCP transport.

        :returns: A ``(read_stream, write_stream)`` tuple from
            the SSE client context manager.
        """
        assert self._exit_stack is not None
        assert self.config.url is not None
        # sse_client timeout controls the initial HTTP connection
        # handshake (default 5s). sse_read_timeout controls how
        # long to wait for each SSE event (default 300s). We
        # apply the per-server timeout to both when configured.
        timeout = self.config.timeout
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(
                url=self.config.url,
                headers=self.config.headers or None,
                # MCP SDK default: 5s for initial HTTP connection handshake.
                timeout=float(timeout) if timeout is not None else 5,
                # MCP SDK default: 300s (5 min) for SSE event read.
                sse_read_timeout=float(timeout) if timeout is not None else 300,
            )
        )
        return read_stream, write_stream


class McpTool(Tool):
    """
    Proxy tool that delegates invocation to an MCP server session.

    Created by ``ToolManager`` during MCP discovery — one
    ``McpTool`` per tool exposed by each MCP server. The tool's
    schema is derived from the MCP ``Tool`` definition returned
    by ``tools/list``.

    :param tool_def: The MCP tool definition from discovery.
    :param connection: The ``McpServerConnection`` that owns
        the session for invoking this tool.
    :param run_sync: Callable that bridges async → sync,
        e.g. ``EventLoopThread.run``. Must execute coroutines
        on the same event loop that the connection was created on.
    """

    def __init__(
        self,
        tool_def: McpToolDef,
        connection: McpServerConnection,
        run_sync: Callable[[Coroutine[Any, Any, str]], str],
    ) -> None:
        """
        Initialize the MCP tool proxy.

        :param tool_def: The MCP tool definition from discovery,
            containing name, description, and input schema.
        :param connection: The ``McpServerConnection`` to use
            for invocation.
        :param run_sync: Callable that runs an awaitable on the
            persistent event loop, e.g. ``EventLoopThread.run``.
        """
        self._tool_def = tool_def
        self._connection = connection
        self._run_sync = run_sync

    @property
    def name(self) -> str:
        """
        Unique tool name from the MCP server.

        :returns: The tool name, e.g. ``"github_list_issues"``.
        """
        return self._tool_def.name

    # Returns dict[str, Any] — defined by the Tool ABC. OpenAI
    # tool schemas are inherently heterogeneous (nested dicts,
    # strings, lists) so Any is the narrowest safe value type.
    def get_schema(self) -> dict[str, Any]:
        """
        Return OpenAI Chat Completions tool schema.

        Converts the MCP tool definition's ``inputSchema`` to
        the OpenAI format expected by litellm.

        :returns: An OpenAI-format tool schema dict.
        """
        return {
            "type": "function",
            "function": {
                "name": self._tool_def.name,
                # OpenAI tool schema requires description to be
                # a string, not None.
                "description": (self._tool_def.description or ""),
                "parameters": (self._tool_def.inputSchema),
            },
        }

    def invoke(self, arguments: str) -> str:
        """
        Invoke the MCP tool via the server session.

        Parses the JSON arguments string and delegates to the
        ``McpServerConnection.call_tool()`` async method via
        the persistent event loop provided at construction.

        :param arguments: JSON-encoded arguments string from
            the LLM.
        :returns: The tool result as a string.
        """
        parsed = json.loads(arguments) if arguments else {}
        return self._run_sync(self._connection.call_tool(self.name, parsed))


# Exception types that indicate a dead/broken connection
# rather than a legitimate tool error. These are worth
# retrying after a reconnect.
_CONNECTION_ERROR_TYPES = (
    EOFError,
    BrokenPipeError,
    ConnectionError,
    OSError,
)


def _is_connection_error(exc: BaseException) -> bool:
    """
    Determine if an exception indicates a dead MCP connection.

    Returns ``True`` for transport-level failures (broken pipe,
    EOF, connection reset) and MCP-level connection-closed
    errors. Returns ``False`` for tool-level errors (invalid
    args, tool not found) which should not trigger a reconnect.

    :param exc: The exception to classify.
    :returns: ``True`` if the error is connection-related.
    """
    if isinstance(exc, _CONNECTION_ERROR_TYPES):
        return True
    if isinstance(exc, McpError):
        return exc.error.code == CONNECTION_CLOSED
    return False


def _backoff_delay(attempt: int, retry: RetryConfig) -> float:
    """
    Compute the backoff delay for a reconnect attempt.

    Uses exponential backoff with jitter (50–100% of the computed
    delay) capped at ``retry.backoff_max``.

    :param attempt: Zero-based retry index (0 = first retry).
    :param retry: Retry config with ``backoff_base`` and
        ``backoff_max``.
    :returns: Sleep duration in seconds.
    """
    delay = min(
        retry.backoff_base ** (attempt + 1),
        retry.backoff_max,
    )
    return delay * random.uniform(0.5, 1.0)


async def _call_tool_with_reconnect(
    conn: McpServerConnection,
    name: str,
    arguments: dict[str, Any],  # JSON values — see call_tool
    retry: RetryConfig,
) -> str:
    """
    Invoke a tool, reconnecting with backoff on connection errors.

    On a connection-level failure (dead transport, server crash),
    reconnects and retries up to ``retry.max_attempts - 1`` times
    with exponential backoff. Permanent errors (invalid args, tool
    not found) are raised immediately without retrying.

    :param conn: The MCP server connection to invoke on.
    :param name: The tool name as returned by discovery.
    :param arguments: The tool arguments dict.
    :param retry: Retry policy controlling max attempts, backoff
        base, and backoff cap.
    :returns: The formatted tool result string.
    """
    last_exc: Exception | None = None

    for attempt in range(retry.max_attempts):
        try:
            return await conn._invoke_tool(name, arguments)
        except Exception as exc:
            if not _is_connection_error(exc):
                raise
            last_exc = exc
            # Last attempt — don't reconnect, just raise.
            if attempt + 1 >= retry.max_attempts:
                break
            delay = _backoff_delay(attempt, retry)
            _logger.warning(
                "MCP server %r: connection lost during tool call "
                "%r (attempt %d/%d), reconnecting in %.1fs",
                conn.config.name,
                name,
                attempt + 1,
                retry.max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            await conn._reconnect()

    # All attempts exhausted — re-raise the last connection error.
    assert last_exc is not None
    raise last_exc


def _format_call_result(result: CallToolResult) -> str:
    """
    Convert an MCP ``CallToolResult`` to a plain string.

    Extracts text content blocks and joins them. If the result
    indicates an error, prefixes the output with ``"Error: "``.

    :param result: The ``CallToolResult`` from
        ``session.call_tool()``.
    :returns: A string representation of the tool result.
        Returns ``"(empty response)"`` when the server sends no
        content blocks.
    """
    parts: list[str] = []
    for block in result.content:
        parts.append(_format_content_block(block))
    joined = "\n".join(parts)
    if not joined:
        joined = "(empty response)"
    if result.isError:
        return f"Error: {joined}"
    return joined


def _format_content_block(block: ContentBlock) -> str:
    """
    Convert a single MCP content block to a string.

    Returns ``.text`` for ``TextContent`` (the most common case).
    For non-text types (``ImageContent``, ``AudioContent``,
    ``EmbeddedResource``, ``ResourceLink``), serializes the full
    Pydantic model to JSON.

    :param block: A content block from ``CallToolResult.content``,
        e.g. ``TextContent(type="text", text="hello")``.
    :returns: A string representation of the block.
    """
    if isinstance(block, TextContent):
        return block.text
    # All ContentBlock variants are Pydantic BaseModels.
    return json.dumps(block.model_dump())
