"""Tests for agent_plane.tools.mcp (MCP connections and tools)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cachetools import TTLCache
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, ErrorData, ImageContent, TextContent

from agent_plane.spec.types import MCPServerConfig
from agent_plane.tools.mcp import (
    McpServerConnection,
    McpTool,
    _cache_key,
    _discovery_cache,
    _format_call_result,
    _is_connection_error,
    _run_async,
    clear_discovery_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    """
    Clear the module-level discovery cache before each test.
    """
    clear_discovery_cache()


def _make_stdio_config(
    name: str = "test-server",
) -> MCPServerConfig:
    """
    Create a minimal stdio MCP server config.

    :param name: Server name identifier.
    :returns: An ``MCPServerConfig`` for stdio transport.
    """
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command="echo",
        args=["hello"],
    )


def _make_http_config(
    name: str = "test-http",
) -> MCPServerConfig:
    """
    Create a minimal HTTP MCP server config.

    :param name: Server name identifier.
    :returns: An ``MCPServerConfig`` for HTTP transport.
    """
    return MCPServerConfig(
        name=name,
        transport="http",
        url="http://localhost:9000/mcp",
    )


def _make_mcp_tool_def(
    name: str = "test_tool",
    description: str = "A test tool.",
) -> MagicMock:
    """
    Create a mock MCP tool definition matching ``mcp.types.Tool``.

    Uses a MagicMock because we only read ``.name``,
    ``.description``, and ``.inputSchema`` — these are plain
    attribute reads, not isinstance checks.

    :param name: Tool name.
    :param description: Tool description.
    :returns: A mock with name, description, and inputSchema.
    """
    tool_def = MagicMock()
    tool_def.name = name
    tool_def.description = description
    tool_def.inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }
    return tool_def


@contextmanager
def _mock_mcp_transport(
    tools: list[MagicMock] | None = None,
) -> Iterator[AsyncMock]:
    """
    Mock the MCP transport and session for ``connect()`` tests.

    Patches ``stdio_client`` and ``ClientSession`` so that
    ``McpServerConnection.connect()`` can run without a real
    MCP server. The mock session's ``list_tools()`` returns
    the provided tool definitions.

    :param tools: Mock tool definitions to return from
        ``list_tools()``. Defaults to an empty list.
    :yields: The mock ``ClientSession`` instance.
    """
    mock_session = AsyncMock()
    mock_tools_result = MagicMock()
    mock_tools_result.tools = tools or []
    mock_session.list_tools.return_value = mock_tools_result
    mock_session.initialize = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "agent_plane.tools.mcp.stdio_client",
        return_value=mock_ctx,
    ):
        with patch(
            "agent_plane.tools.mcp.ClientSession",
            return_value=mock_session,
        ):
            yield mock_session


# ── _cache_key ───────────────────────────────────────────


def test_cache_key_stdio_includes_command_and_args() -> None:
    """
    Cache key for stdio config includes name, command, and args.
    """
    config = _make_stdio_config()
    key = _cache_key(config)
    assert "stdio" in key
    assert "test-server" in key
    assert "echo" in key


def test_cache_key_http_includes_url() -> None:
    """
    Cache key for HTTP config includes name and url.
    """
    config = _make_http_config()
    key = _cache_key(config)
    assert "http" in key
    assert "localhost:9000" in key


def test_cache_key_different_configs_differ() -> None:
    """
    Different server configs produce different cache keys.
    """
    key1 = _cache_key(_make_stdio_config("server-a"))
    key2 = _cache_key(_make_stdio_config("server-b"))
    assert key1 != key2


# ── McpServerConnection caching ──────────────────────────


@pytest.mark.asyncio()
async def test_connect_skips_list_tools_when_cache_fresh() -> None:
    """
    ``connect()`` skips the ``list_tools()`` round-trip when
    the cache is fresh, but still opens a live session so
    ``call_tool()`` works.
    """
    config = _make_stdio_config()
    tool_def = _make_mcp_tool_def()

    # Pre-populate the cache — TTLCache uses dict assignment.
    _discovery_cache[_cache_key(config)] = [tool_def]

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        tools = await conn.connect()

    assert len(tools) == 1
    assert tools[0].name == "test_tool"
    # Session was created (for invocation), but list_tools
    # was NOT called (served from cache).
    mock_session.initialize.assert_awaited_once()
    mock_session.list_tools.assert_not_awaited()

    await conn.close()


@pytest.mark.asyncio()
async def test_cached_connect_has_live_session() -> None:
    """
    When discovery is served from cache, the connection still
    has a live session that can invoke tools.
    """
    config = _make_stdio_config()
    _discovery_cache[_cache_key(config)] = [_make_mcp_tool_def()]

    with _mock_mcp_transport() as mock_session:
        # Set up call_tool to return a mock result.
        mock_result = MagicMock()
        mock_result.content = [TextContent(type="text", text="cached ok")]
        mock_result.isError = False
        mock_session.call_tool.return_value = mock_result

        conn = McpServerConnection(config=config)
        await conn.connect()
        result = await conn.call_tool("test_tool", {"query": "hi"})

    assert result == "cached ok"
    mock_session.call_tool.assert_awaited_once()

    await conn.close()


@pytest.mark.asyncio()
async def test_connect_skips_expired_cache() -> None:
    """
    ``connect()`` ignores cache entries older than the TTL and
    performs a live discovery via ``list_tools()``.
    """
    config = _make_stdio_config()

    # Use a TTLCache with a controllable timer so we can
    # simulate expiry without sleeping. Start at t=0, insert
    # the entry, then advance past the TTL.
    current_time = [0.0]
    expired_cache: TTLCache[str, list[MagicMock]] = TTLCache(
        maxsize=64,
        ttl=300,
        timer=lambda: current_time[0],
    )
    expired_cache[_cache_key(config)] = [_make_mcp_tool_def()]
    # Advance time past the 300s TTL.
    current_time[0] = 1000.0

    fresh_tool = _make_mcp_tool_def("fresh_tool")
    with _mock_mcp_transport([fresh_tool]) as mock_session:
        with patch("agent_plane.tools.mcp._discovery_cache", expired_cache):
            conn = McpServerConnection(config=config)
            tools = await conn.connect()

    assert len(tools) == 1
    assert tools[0].name == "fresh_tool"
    mock_session.initialize.assert_awaited_once()
    mock_session.list_tools.assert_awaited_once()

    await conn.close()


@pytest.mark.asyncio()
async def test_connect_populates_cache() -> None:
    """
    A successful live ``connect()`` stores results in the
    module-level cache.
    """
    config = _make_stdio_config()
    tool_def = _make_mcp_tool_def("cached_tool")

    with _mock_mcp_transport([tool_def]):
        conn = McpServerConnection(config=config)
        await conn.connect()

    key = _cache_key(config)
    assert key in _discovery_cache
    cached = _discovery_cache.get(key)
    assert cached is not None
    assert len(cached) == 1
    assert cached[0].name == "cached_tool"

    await conn.close()


# ── McpServerConnection.call_tool ────────────────────────


@pytest.mark.asyncio()
async def test_call_tool_raises_without_connect() -> None:
    """
    ``call_tool()`` raises RuntimeError when ``connect()``
    was never called.
    """
    conn = McpServerConnection(config=_make_stdio_config())

    with pytest.raises(RuntimeError, match="no live session"):
        await conn.call_tool("test_tool", {"query": "hi"})


# ── McpServerConnection.close ────────────────────────────


@pytest.mark.asyncio()
async def test_close_is_safe_when_never_connected() -> None:
    """
    ``close()`` does not raise if ``connect()`` was never called.
    """
    conn = McpServerConnection(config=_make_stdio_config())
    await conn.close()


# ── McpTool ──────────────────────────────────────────────


def test_mcp_tool_name() -> None:
    """
    McpTool.name returns the MCP tool definition's name.
    """
    tool_def = _make_mcp_tool_def("my_tool")
    conn = McpServerConnection(config=_make_stdio_config())
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        # Stub run_sync — not exercised in this test.
        run_sync=MagicMock(),
    )
    assert tool.name == "my_tool"


def test_mcp_tool_schema_openai_format() -> None:
    """
    McpTool.get_schema returns an OpenAI-format tool schema.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_stdio_config())
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        # Stub run_sync — not exercised in this test.
        run_sync=MagicMock(),
    )
    schema = tool.get_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "A test tool."
    assert "properties" in schema["function"]["parameters"]


def test_mcp_tool_invoke_delegates_to_run_sync() -> None:
    """
    McpTool.invoke parses JSON args and calls the connection's
    call_tool method via the run_sync callable.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_stdio_config())

    mock_run_sync = MagicMock(return_value="result text")
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        run_sync=mock_run_sync,
    )
    result = tool.invoke(json.dumps({"query": "hello"}))

    assert result == "result text"
    mock_run_sync.assert_called_once()


# ── _format_call_result ──────────────────────────────────


def test_format_call_result_text_content() -> None:
    """
    Text content blocks are extracted and joined.
    """
    block = TextContent(type="text", text="Hello world")
    result = MagicMock()
    result.content = [block]
    result.isError = False

    assert _format_call_result(result) == "Hello world"


def test_format_call_result_multiple_blocks() -> None:
    """
    Multiple text blocks are joined with newlines.
    """
    block1 = TextContent(type="text", text="Line 1")
    block2 = TextContent(type="text", text="Line 2")
    result = MagicMock()
    result.content = [block1, block2]
    result.isError = False

    assert _format_call_result(result) == "Line 1\nLine 2"


def test_format_call_result_error_prefix() -> None:
    """
    Error results are prefixed with "Error: ".
    """
    block = TextContent(type="text", text="something went wrong")
    result = MagicMock()
    result.content = [block]
    result.isError = True

    formatted = _format_call_result(result)
    assert formatted.startswith("Error: ")
    assert "something went wrong" in formatted


def test_format_call_result_non_text_content() -> None:
    """
    Non-text content (e.g. images) is serialized as JSON via
    ``model_dump()``.
    """
    block = ImageContent(type="image", data="base64data", mimeType="image/png")
    result = MagicMock()
    result.content = [block]
    result.isError = False

    formatted = _format_call_result(result)
    parsed = json.loads(formatted)
    assert parsed["type"] == "image"
    assert parsed["data"] == "base64data"


def test_format_call_result_empty_content() -> None:
    """
    An empty content list returns ``"(empty response)"``
    instead of a blank string.
    """
    result = MagicMock()
    result.content = []
    result.isError = False

    assert _format_call_result(result) == "(empty response)"


def test_format_call_result_empty_content_with_error() -> None:
    """
    An empty content list with ``isError=True`` returns
    ``"Error: (empty response)"``.
    """
    result = MagicMock()
    result.content = []
    result.isError = True

    assert _format_call_result(result) == "Error: (empty response)"


# ── clear_discovery_cache ────────────────────────────────


def test_clear_discovery_cache() -> None:
    """
    ``clear_discovery_cache()`` empties the module-level cache.
    """
    config = _make_stdio_config()
    _discovery_cache[_cache_key(config)] = []
    assert len(_discovery_cache) > 0

    clear_discovery_cache()
    assert len(_discovery_cache) == 0


def test_discovery_cache_evicts_lru_when_full() -> None:
    """
    The discovery cache evicts the least-recently-used entry
    when it reaches ``maxsize``.

    Uses a small TTLCache (maxsize=2) to verify that inserting
    a third entry evicts the oldest one.
    """
    small_cache: TTLCache[str, list[MagicMock]] = TTLCache(
        maxsize=2,
        ttl=300,
    )
    small_cache["server-a"] = [_make_mcp_tool_def("tool_a")]
    small_cache["server-b"] = [_make_mcp_tool_def("tool_b")]

    # Inserting a third entry should evict the LRU (server-a).
    small_cache["server-c"] = [_make_mcp_tool_def("tool_c")]

    assert "server-a" not in small_cache
    assert "server-b" in small_cache
    assert "server-c" in small_cache
    assert len(small_cache) == 2


# ── _open_transport validation ───────────────────────────


@pytest.mark.asyncio()
async def test_connect_raises_on_unsupported_transport() -> None:
    """
    ``connect()`` raises ValueError for unknown transport types.
    """
    config = MCPServerConfig(
        name="bad",
        transport="grpc",
    )
    conn = McpServerConnection(config=config)
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        await conn.connect()


# ── _run_async ───────────────────────────────────────────


def test_run_async_from_sync_context() -> None:
    """
    ``_run_async()`` runs a coroutine from synchronous code.
    """

    async def _add(a: int, b: int) -> int:
        """
        Trivial async function for testing.

        :param a: First operand.
        :param b: Second operand.
        :returns: The sum.
        """
        return a + b

    assert _run_async(_add(2, 3)) == 5


@pytest.mark.asyncio()
async def test_run_async_from_async_context() -> None:
    """
    ``_run_async()`` works even when called from an async
    context (e.g. pytest-asyncio), unlike ``asyncio.run()``
    which would raise RuntimeError.
    """

    async def _greet(name: str) -> str:
        """
        Trivial async function for testing.

        :param name: Name to greet.
        :returns: A greeting string.
        """
        return f"hello {name}"

    # This runs inside an async test (event loop is running).
    # asyncio.run() would fail here; _run_async() must succeed.
    result = _run_async(_greet("world"))
    assert result == "hello world"


# ── _is_connection_error ─────────────────────────────────


def test_is_connection_error_eof() -> None:
    """
    EOFError is classified as a connection error.
    """
    assert _is_connection_error(EOFError()) is True


def test_is_connection_error_broken_pipe() -> None:
    """
    BrokenPipeError is classified as a connection error.
    """
    assert _is_connection_error(BrokenPipeError()) is True


def test_is_connection_error_connection_reset() -> None:
    """
    ConnectionResetError (subclass of ConnectionError) is
    classified as a connection error.
    """
    assert _is_connection_error(ConnectionResetError()) is True


def test_is_connection_error_mcp_connection_closed() -> None:
    """
    McpError with CONNECTION_CLOSED code is classified as a
    connection error.
    """
    exc = McpError(
        ErrorData(
            code=CONNECTION_CLOSED,
            message="Connection closed",
        )
    )
    assert _is_connection_error(exc) is True


def test_is_connection_error_mcp_other_code() -> None:
    """
    McpError with a non-connection code (e.g. INVALID_PARAMS)
    is NOT classified as a connection error.
    """
    exc = McpError(
        ErrorData(
            code=-32602,  # INVALID_PARAMS
            message="Invalid params",
        )
    )
    assert _is_connection_error(exc) is False


def test_is_connection_error_value_error() -> None:
    """
    ValueError is NOT classified as a connection error.
    """
    assert _is_connection_error(ValueError("bad")) is False


# ── Reconnection on server death ─────────────────────────


@pytest.mark.asyncio()
async def test_call_tool_reconnects_on_connection_error() -> None:
    """
    When a tool call fails with a connection error, the
    connection is torn down, re-established, and the call
    is retried exactly once.
    """
    config = _make_stdio_config()

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        # First call_tool raises a connection error (server died).
        # Second call_tool (after reconnect) succeeds.
        ok_result = MagicMock()
        ok_result.content = [TextContent(type="text", text="recovered")]
        ok_result.isError = False

        mock_session.call_tool.side_effect = [
            EOFError("server died"),
            ok_result,
        ]

        # Patch _reconnect to re-establish the mock session
        # (in production this opens a new transport).
        with patch.object(conn, "_reconnect", new_callable=AsyncMock) as mock_reconnect:
            result = await conn.call_tool("test_tool", {"query": "hi"})

        assert result == "recovered"
        mock_reconnect.assert_awaited_once()

    await conn.close()


@pytest.mark.asyncio()
async def test_call_tool_does_not_reconnect_on_tool_error() -> None:
    """
    When a tool call fails with a non-connection error (e.g.
    McpError for invalid params), no reconnect is attempted.
    """
    config = _make_stdio_config()

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        mock_session.call_tool.side_effect = McpError(
            ErrorData(
                code=-32602,
                message="Invalid params",
            )
        )

        with pytest.raises(McpError):
            await conn.call_tool("test_tool", {"query": "hi"})

    await conn.close()


@pytest.mark.asyncio()
async def test_call_tool_propagates_if_retry_fails() -> None:
    """
    If the retry after reconnect also fails, the error is
    propagated to the caller.
    """
    config = _make_stdio_config()

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        # Both calls fail with connection errors.
        mock_session.call_tool.side_effect = [
            EOFError("server died"),
            EOFError("server died again"),
        ]

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with pytest.raises(EOFError, match="died again"):
                await conn.call_tool("test_tool", {"query": "hi"})


# ── Timeout propagation ──────────────────────────────────


@pytest.mark.asyncio()
async def test_connect_passes_timeout_to_client_session() -> None:
    """
    When ``MCPServerConfig.timeout`` is set, ``connect()`` must
    pass ``read_timeout_seconds=timedelta(seconds=timeout)`` to
    ``ClientSession``.
    """
    config = MCPServerConfig(
        name="test-timeout",
        transport="stdio",
        command="echo",
        args=["hello"],
        timeout=60,
    )

    captured_kwargs: dict[str, Any] = {}

    def _capturing_session(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncMock:
        """
        Fake ``ClientSession`` constructor that records kwargs.

        :param args: Positional args (read_stream, write_stream).
        :param kwargs: Keyword args including read_timeout_seconds.
        :returns: A mock session with working initialize/list_tools.
        """
        captured_kwargs.update(kwargs)
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools = MagicMock()
        mock_tools.tools = []
        mock_session.list_tools.return_value = mock_tools
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock()),
    )
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "agent_plane.tools.mcp.ClientSession",
        side_effect=_capturing_session,
    ):
        with patch(
            "agent_plane.tools.mcp.stdio_client",
            return_value=mock_ctx,
        ):
            conn = McpServerConnection(config=config)
            await conn.connect()

    # timeout=60 must be converted to timedelta(seconds=60) for the
    # MCP SDK's ClientSession read_timeout_seconds parameter.
    assert captured_kwargs.get("read_timeout_seconds") == timedelta(seconds=60), (
        "ClientSession must receive read_timeout_seconds as a "
        "timedelta matching the config timeout"
    )

    await conn.close()


@pytest.mark.asyncio()
async def test_connect_passes_none_timeout_to_client_session() -> None:
    """
    When ``MCPServerConfig.timeout`` is ``None`` (default),
    ``connect()`` must pass ``read_timeout_seconds=None`` so the
    MCP SDK uses its built-in default.
    """
    config = MCPServerConfig(
        name="test-no-timeout",
        transport="stdio",
        command="echo",
        args=["hello"],
        # timeout defaults to None
    )

    captured_kwargs: dict[str, Any] = {}

    def _capturing_session(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncMock:
        """
        Fake ``ClientSession`` constructor that records kwargs.

        :param args: Positional args (read_stream, write_stream).
        :param kwargs: Keyword args including read_timeout_seconds.
        :returns: A mock session with working initialize/list_tools.
        """
        captured_kwargs.update(kwargs)
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools = MagicMock()
        mock_tools.tools = []
        mock_session.list_tools.return_value = mock_tools
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock()),
    )
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "agent_plane.tools.mcp.ClientSession",
        side_effect=_capturing_session,
    ):
        with patch(
            "agent_plane.tools.mcp.stdio_client",
            return_value=mock_ctx,
        ):
            conn = McpServerConnection(config=config)
            await conn.connect()

    # When timeout is None, read_timeout_seconds must be None so the
    # MCP SDK falls back to its own default (no timeout).
    assert captured_kwargs.get("read_timeout_seconds") is None, (
        "ClientSession must receive read_timeout_seconds=None when config timeout is unset"
    )

    await conn.close()


@pytest.mark.asyncio()
async def test_connect_http_passes_timeout_to_sse_client() -> None:
    """
    When ``MCPServerConfig(transport="http", timeout=60)``,
    ``connect()`` must pass ``timeout=60.0`` and
    ``sse_read_timeout=60.0`` to ``sse_client``.
    """
    config = MCPServerConfig(
        name="test-http-timeout",
        transport="http",
        url="http://localhost:9000/mcp",
        timeout=60,
    )

    captured_sse_kwargs: dict[str, Any] = {}

    mock_sse_ctx = AsyncMock()
    mock_sse_ctx.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock()),
    )
    mock_sse_ctx.__aexit__ = AsyncMock(return_value=False)

    def _capturing_sse_client(**kwargs: Any) -> AsyncMock:
        """
        Fake ``sse_client`` that records kwargs.

        :param kwargs: Keyword args including timeout and
            sse_read_timeout.
        :returns: An async context manager yielding mock streams.
        """
        captured_sse_kwargs.update(kwargs)
        return mock_sse_ctx

    def _capturing_session(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncMock:
        """
        Fake ``ClientSession`` constructor for HTTP transport.

        :param args: Positional args (read_stream, write_stream).
        :param kwargs: Keyword args.
        :returns: A mock session with working initialize/list_tools.
        """
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools = MagicMock()
        mock_tools.tools = []
        mock_session.list_tools.return_value = mock_tools
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    with patch(
        "agent_plane.tools.mcp.sse_client",
        side_effect=_capturing_sse_client,
    ):
        with patch(
            "agent_plane.tools.mcp.ClientSession",
            side_effect=_capturing_session,
        ):
            conn = McpServerConnection(config=config)
            await conn.connect()

    # Both timeout (HTTP handshake) and sse_read_timeout (SSE event
    # wait) must be set to the config timeout as a float.
    assert captured_sse_kwargs["timeout"] == 60.0, (
        "sse_client timeout must equal the config timeout as float"
    )
    assert captured_sse_kwargs["sse_read_timeout"] == 60.0, (
        "sse_client sse_read_timeout must equal the config timeout as float"
    )

    await conn.close()


@pytest.mark.asyncio()
async def test_connect_http_uses_default_timeouts_when_none() -> None:
    """
    When ``MCPServerConfig(transport="http", timeout=None)``,
    ``connect()`` must pass the MCP SDK defaults: ``timeout=5``
    and ``sse_read_timeout=300``.
    """
    config = MCPServerConfig(
        name="test-http-default",
        transport="http",
        url="http://localhost:9000/mcp",
        # timeout defaults to None
    )

    captured_sse_kwargs: dict[str, Any] = {}

    mock_sse_ctx = AsyncMock()
    mock_sse_ctx.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock()),
    )
    mock_sse_ctx.__aexit__ = AsyncMock(return_value=False)

    def _capturing_sse_client(**kwargs: Any) -> AsyncMock:
        """
        Fake ``sse_client`` that records kwargs.

        :param kwargs: Keyword args including timeout and
            sse_read_timeout.
        :returns: An async context manager yielding mock streams.
        """
        captured_sse_kwargs.update(kwargs)
        return mock_sse_ctx

    def _capturing_session(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncMock:
        """
        Fake ``ClientSession`` constructor for HTTP transport.

        :param args: Positional args (read_stream, write_stream).
        :param kwargs: Keyword args.
        :returns: A mock session with working initialize/list_tools.
        """
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools = MagicMock()
        mock_tools.tools = []
        mock_session.list_tools.return_value = mock_tools
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    with patch(
        "agent_plane.tools.mcp.sse_client",
        side_effect=_capturing_sse_client,
    ):
        with patch(
            "agent_plane.tools.mcp.ClientSession",
            side_effect=_capturing_session,
        ):
            conn = McpServerConnection(config=config)
            await conn.connect()

    # SDK default: 5s for initial HTTP connection handshake.
    assert captured_sse_kwargs["timeout"] == 5, (
        "sse_client timeout must default to 5 when config timeout is None"
    )
    # SDK default: 300s (5 min) for SSE event read.
    assert captured_sse_kwargs["sse_read_timeout"] == 300, (
        "sse_client sse_read_timeout must default to 300 when config timeout is None"
    )


# ── HTTP transport: connection, headers, discovery ────────


@dataclass
class CapturedHttpArgs:
    """
    Container for kwargs captured from ``sse_client`` and
    ``ClientSession`` during HTTP transport tests.

    :param sse_kwargs: Keyword arguments passed to ``sse_client``,
        e.g. ``{"url": "...", "headers": {...}, "timeout": 5}``.
    :param session_kwargs: Keyword arguments passed to
        ``ClientSession``, e.g. ``{"read_timeout_seconds": ...}``.
    :param mock_session: The mock ``ClientSession`` instance for
        setting up ``call_tool()`` side effects.
    """

    sse_kwargs: dict[str, Any] = field(default_factory=dict)
    session_kwargs: dict[str, Any] = field(default_factory=dict)
    mock_session: AsyncMock = field(default_factory=AsyncMock)


@contextmanager
def _mock_http_transport(
    tools: list[MagicMock] | None = None,
) -> Iterator[CapturedHttpArgs]:
    """
    Mock the HTTP (SSE) transport and session for HTTP tests.

    Patches ``sse_client`` and ``ClientSession`` so that
    ``McpServerConnection.connect()`` can run without a real
    HTTP server. Captures the kwargs passed to both so tests
    can verify URL, headers, timeout, and other arguments.

    :param tools: Mock tool definitions to return from
        ``list_tools()``. Defaults to an empty list.
    :yields: A :class:`CapturedHttpArgs` with captured kwargs
        and the mock session.
    """
    captured = CapturedHttpArgs()

    mock_sse_ctx = AsyncMock()
    mock_sse_ctx.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock()),
    )
    mock_sse_ctx.__aexit__ = AsyncMock(return_value=False)

    def _capturing_sse_client(**kwargs: Any) -> AsyncMock:
        """
        Fake ``sse_client`` that records kwargs.

        :param kwargs: Keyword args including url, headers,
            timeout, and sse_read_timeout.
        :returns: An async context manager yielding mock streams.
        """
        captured.sse_kwargs.update(kwargs)
        return mock_sse_ctx

    def _capturing_session(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncMock:
        """
        Fake ``ClientSession`` constructor that records kwargs.

        :param args: Positional args (read_stream, write_stream).
        :param kwargs: Keyword args including read_timeout_seconds.
        :returns: A mock session with working initialize/list_tools.
        """
        captured.session_kwargs.update(kwargs)
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = tools or []
        mock_session.list_tools.return_value = mock_tools_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        captured.mock_session = mock_session
        return mock_session

    with patch(
        "agent_plane.tools.mcp.sse_client",
        side_effect=_capturing_sse_client,
    ):
        with patch(
            "agent_plane.tools.mcp.ClientSession",
            side_effect=_capturing_session,
        ):
            yield captured


@pytest.mark.asyncio()
async def test_http_connect_passes_url_to_sse_client() -> None:
    """
    HTTP ``connect()`` passes the config URL to ``sse_client``.
    """
    config = MCPServerConfig(
        name="test-http",
        transport="http",
        url="https://mcp.example.com/sse",
    )

    with _mock_http_transport() as captured:
        conn = McpServerConnection(config=config)
        await conn.connect()

    assert captured.sse_kwargs["url"] == "https://mcp.example.com/sse"

    await conn.close()


@pytest.mark.asyncio()
async def test_http_connect_passes_headers_to_sse_client() -> None:
    """
    HTTP ``connect()`` propagates auth headers from the config
    to ``sse_client``.
    """
    config = MCPServerConfig(
        name="test-http-headers",
        transport="http",
        url="http://localhost:9000/mcp",
        headers={
            "Authorization": "Bearer tok_xyz",
            "X-Custom": "value",
        },
    )

    with _mock_http_transport() as captured:
        conn = McpServerConnection(config=config)
        await conn.connect()

    assert captured.sse_kwargs["headers"] == {
        "Authorization": "Bearer tok_xyz",
        "X-Custom": "value",
    }

    await conn.close()


@pytest.mark.asyncio()
async def test_http_connect_passes_none_headers_when_empty() -> None:
    """
    HTTP ``connect()`` passes ``headers=None`` when the config
    has no headers, so ``sse_client`` uses its default.
    """
    config = MCPServerConfig(
        name="test-http-no-headers",
        transport="http",
        url="http://localhost:9000/mcp",
        # headers defaults to empty dict
    )

    with _mock_http_transport() as captured:
        conn = McpServerConnection(config=config)
        await conn.connect()

    # Empty dict is converted to None via `or None`.
    assert captured.sse_kwargs["headers"] is None

    await conn.close()


@pytest.mark.asyncio()
async def test_http_connect_discovers_tools() -> None:
    """
    HTTP ``connect()`` discovers tools via ``list_tools()`` and
    returns them, just like stdio transport.
    """
    config = MCPServerConfig(
        name="test-http-discovery",
        transport="http",
        url="http://localhost:9000/mcp",
    )
    tool_def = _make_mcp_tool_def("http_tool")

    with _mock_http_transport([tool_def]) as captured:
        conn = McpServerConnection(config=config)
        tools = await conn.connect()

    assert len(tools) == 1
    assert tools[0].name == "http_tool"
    captured.mock_session.initialize.assert_awaited_once()
    captured.mock_session.list_tools.assert_awaited_once()

    await conn.close()


@pytest.mark.asyncio()
async def test_http_call_tool_invokes_session() -> None:
    """
    ``call_tool()`` on an HTTP connection delegates to the
    session's ``call_tool`` method and returns formatted results.
    """
    config = MCPServerConfig(
        name="test-http-invoke",
        transport="http",
        url="http://localhost:9000/mcp",
    )

    with _mock_http_transport([_make_mcp_tool_def("http_tool")]) as captured:
        conn = McpServerConnection(config=config)
        await conn.connect()

        # Set up call_tool to return a mock result.
        mock_result = MagicMock()
        mock_result.content = [
            TextContent(type="text", text="HTTP result"),
        ]
        mock_result.isError = False
        captured.mock_session.call_tool.return_value = mock_result

        result = await conn.call_tool("http_tool", {"query": "test"})

    assert result == "HTTP result"
    captured.mock_session.call_tool.assert_awaited_once_with(
        name="http_tool",
        arguments={"query": "test"},
    )

    await conn.close()


@pytest.mark.asyncio()
async def test_http_reconnect_on_connection_error() -> None:
    """
    HTTP ``call_tool()`` reconnects and retries on a connection
    error, just like stdio transport.
    """
    config = MCPServerConfig(
        name="test-http-reconnect",
        transport="http",
        url="http://localhost:9000/mcp",
    )

    with _mock_http_transport([_make_mcp_tool_def("http_tool")]) as captured:
        conn = McpServerConnection(config=config)
        await conn.connect()

        ok_result = MagicMock()
        ok_result.content = [
            TextContent(type="text", text="recovered via HTTP"),
        ]
        ok_result.isError = False

        captured.mock_session.call_tool.side_effect = [
            ConnectionError("HTTP connection lost"),
            ok_result,
        ]

        with patch.object(conn, "_reconnect", new_callable=AsyncMock) as mock_reconnect:
            result = await conn.call_tool("http_tool", {"query": "retry"})

    assert result == "recovered via HTTP"
    mock_reconnect.assert_awaited_once()

    await conn.close()


@pytest.mark.asyncio()
async def test_http_connect_uses_cache() -> None:
    """
    HTTP ``connect()`` uses the discovery cache when fresh,
    skipping ``list_tools()`` while still opening a live session.
    """
    config = MCPServerConfig(
        name="test-http-cached",
        transport="http",
        url="http://localhost:9000/mcp",
    )
    tool_def = _make_mcp_tool_def("cached_http_tool")
    _discovery_cache[_cache_key(config)] = [tool_def]

    with _mock_http_transport() as captured:
        conn = McpServerConnection(config=config)
        tools = await conn.connect()

    assert len(tools) == 1
    assert tools[0].name == "cached_http_tool"
    # Session opened (for invocation), but list_tools skipped.
    captured.mock_session.initialize.assert_awaited_once()
    captured.mock_session.list_tools.assert_not_awaited()

    await conn.close()
