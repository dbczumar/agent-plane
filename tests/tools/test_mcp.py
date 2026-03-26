"""Tests for agent_plane.tools.mcp (MCP connections and tools)."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_plane.spec.types import MCPServerConfig
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, ErrorData

from agent_plane.tools.mcp import (
    McpServerConnection,
    McpTool,
    _CachedDiscovery,
    _cache_key,
    _discovery_cache,
    _format_call_result,
    _format_content_block,
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
    mock_session.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock())
    )
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

    # Pre-populate the cache.
    _discovery_cache[_cache_key(config)] = _CachedDiscovery(
        tools=[tool_def],
        fetched_at=time.monotonic(),
    )

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
    _discovery_cache[_cache_key(config)] = _CachedDiscovery(
        tools=[_make_mcp_tool_def()],
        fetched_at=time.monotonic(),
    )

    with _mock_mcp_transport() as mock_session:
        # Set up call_tool to return a mock result.
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="cached ok")]
        mock_result.isError = False
        mock_session.call_tool.return_value = mock_result

        conn = McpServerConnection(config=config)
        await conn.connect()
        result = await conn.call_tool(
            "test_tool", {"query": "hi"}
        )

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

    # Pre-populate with an expired entry.
    _discovery_cache[_cache_key(config)] = _CachedDiscovery(
        tools=[_make_mcp_tool_def()],
        # Expired: fetched 1000 seconds ago.
        fetched_at=time.monotonic() - 1000,
    )

    fresh_tool = _make_mcp_tool_def("fresh_tool")
    with _mock_mcp_transport([fresh_tool]) as mock_session:
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
    assert len(_discovery_cache[key].tools) == 1
    assert _discovery_cache[key].tools[0].name == "cached_tool"

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
    tool = McpTool(tool_def=tool_def, connection=conn)
    assert tool.name == "my_tool"


def test_mcp_tool_schema_openai_format() -> None:
    """
    McpTool.get_schema returns an OpenAI-format tool schema.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_stdio_config())
    tool = McpTool(tool_def=tool_def, connection=conn)
    schema = tool.get_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "A test tool."
    assert "properties" in schema["function"]["parameters"]


def test_mcp_tool_invoke_delegates_to_connection() -> None:
    """
    McpTool.invoke parses JSON args and calls the connection's
    call_tool method via _run_async.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_stdio_config())

    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="result text")]
    mock_result.isError = False

    with patch.object(
        conn,
        "call_tool",
        new_callable=AsyncMock,
        return_value="result text",
    ):
        with patch(
            "agent_plane.tools.mcp._run_async",
            return_value="result text",
        ) as mock_run:
            tool = McpTool(
                tool_def=tool_def, connection=conn
            )
            result = tool.invoke(
                json.dumps({"query": "hello"})
            )

    assert result == "result text"
    mock_run.assert_called_once()


# ── _format_call_result ──────────────────────────────────


def test_format_call_result_text_content() -> None:
    """
    Text content blocks are extracted and joined.
    """
    block = MagicMock()
    block.text = "Hello world"
    result = MagicMock()
    result.content = [block]
    result.isError = False

    assert _format_call_result(result) == "Hello world"


def test_format_call_result_multiple_blocks() -> None:
    """
    Multiple text blocks are joined with newlines.
    """
    block1 = MagicMock()
    block1.text = "Line 1"
    block2 = MagicMock()
    block2.text = "Line 2"
    result = MagicMock()
    result.content = [block1, block2]
    result.isError = False

    assert _format_call_result(result) == "Line 1\nLine 2"


def test_format_call_result_error_prefix() -> None:
    """
    Error results are prefixed with "Error: ".
    """
    block = MagicMock()
    block.text = "something went wrong"
    result = MagicMock()
    result.content = [block]
    result.isError = True

    formatted = _format_call_result(result)
    assert formatted.startswith("Error: ")
    assert "something went wrong" in formatted


def test_format_call_result_non_text_content() -> None:
    """
    Non-text content (e.g. images) is serialized as JSON.
    """
    # spec=[] disables auto-attribute creation so getattr(.text)
    # returns None, simulating non-text content (e.g. ImageContent).
    block = MagicMock(spec=[])
    block.model_dump = MagicMock(
        return_value={"type": "image", "data": "base64..."}
    )
    result = MagicMock()
    result.content = [block]
    result.isError = False

    formatted = _format_call_result(result)
    parsed = json.loads(formatted)
    assert parsed["type"] == "image"


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


def test_format_content_block_unknown_type_falls_back_to_str() -> None:
    """
    A content block with no ``.text`` and no ``.model_dump()``
    falls back to ``str(block)``.
    """

    class _UnknownBlock:
        """
        Simulates a hypothetical future MCP content type that
        is neither TextContent nor a Pydantic BaseModel.
        """

        def __str__(self) -> str:
            return "CustomBlock(data=42)"

    assert _format_content_block(_UnknownBlock()) == "CustomBlock(data=42)"


# ── clear_discovery_cache ────────────────────────────────


def test_clear_discovery_cache() -> None:
    """
    ``clear_discovery_cache()`` empties the module-level cache.
    """
    config = _make_stdio_config()
    _discovery_cache[_cache_key(config)] = _CachedDiscovery(
        tools=[],
        fetched_at=time.monotonic(),
    )
    assert len(_discovery_cache) > 0

    clear_discovery_cache()
    assert len(_discovery_cache) == 0


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
    assert _is_connection_error(
        ConnectionResetError()
    ) is True


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
        ok_result.content = [MagicMock(text="recovered")]
        ok_result.isError = False

        mock_session.call_tool.side_effect = [
            EOFError("server died"),
            ok_result,
        ]

        # Patch _reconnect to re-establish the mock session
        # (in production this opens a new transport).
        with patch.object(
            conn, "_reconnect", new_callable=AsyncMock
        ) as mock_reconnect:
            result = await conn.call_tool(
                "test_tool", {"query": "hi"}
            )

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
            await conn.call_tool(
                "test_tool", {"query": "hi"}
            )

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

        with patch.object(
            conn, "_reconnect", new_callable=AsyncMock
        ):
            with pytest.raises(EOFError, match="died again"):
                await conn.call_tool(
                    "test_tool", {"query": "hi"}
                )
