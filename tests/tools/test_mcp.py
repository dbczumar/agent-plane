"""Tests for agent_plane.tools.mcp (MCP connections and tools)."""

from __future__ import annotations

import json
import logging
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

from agent_plane.spec.types import MCPServerConfig, RetryConfig
from agent_plane.tools.base import ToolContext
from agent_plane.tools.mcp import (
    _CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    _CIRCUIT_BREAKER_THRESHOLD,
    _MCP_RECONNECT_DEFAULTS,
    EventLoopThread,
    McpServerConnection,
    McpServerDisabledError,
    McpTool,
    _backoff_delay,
    _cache_key,
    _CircuitBreaker,
    _collect_problematic_keywords,
    _discovery_cache,
    _format_call_result,
    _is_connection_error,
    _normalize_input_schema,
    _run_async,
    clear_discovery_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    """
    Clear the module-level discovery cache before each test.
    """
    clear_discovery_cache()


def _make_http_config(
    name: str = "test-server",
    url: str = "http://localhost:9000/mcp",
) -> MCPServerConfig:
    """
    Create a minimal HTTP MCP server config.

    :param name: Server name identifier.
    :param url: Server endpoint URL.
    :returns: An ``MCPServerConfig`` for HTTP transport.
    """
    return MCPServerConfig(
        name=name,
        url=url,
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

    Patches ``sse_client`` and ``ClientSession`` so that
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
        "agent_plane.tools.mcp.sse_client",
        return_value=mock_ctx,
    ):
        with patch(
            "agent_plane.tools.mcp.ClientSession",
            return_value=mock_session,
        ):
            yield mock_session


# ── _cache_key ───────────────────────────────────────────


def test_cache_key_includes_name_and_url() -> None:
    """
    Cache key includes the server name and URL.
    """
    config = _make_http_config()
    key = _cache_key(config)
    assert "http" in key
    assert "test-server" in key
    assert "localhost:9000" in key


def test_cache_key_different_configs_differ() -> None:
    """
    Different server configs produce different cache keys.
    """
    key1 = _cache_key(_make_http_config("server-a"))
    key2 = _cache_key(_make_http_config("server-b"))
    assert key1 != key2


# ── McpServerConnection caching ──────────────────────────


@pytest.mark.asyncio()
async def test_connect_skips_list_tools_when_cache_fresh() -> None:
    """
    ``connect()`` skips the ``list_tools()`` round-trip when
    the cache is fresh, but still opens a live session so
    ``call_tool()`` works.
    """
    config = _make_http_config()
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
    config = _make_http_config()
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
    config = _make_http_config()

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
    config = _make_http_config()
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
    conn = McpServerConnection(config=_make_http_config())

    with pytest.raises(RuntimeError, match="no live session"):
        await conn.call_tool("test_tool", {"query": "hi"})


# ── McpServerConnection.close ────────────────────────────


@pytest.mark.asyncio()
async def test_close_is_safe_when_never_connected() -> None:
    """
    ``close()`` does not raise if ``connect()`` was never called.
    """
    conn = McpServerConnection(config=_make_http_config())
    await conn.close()


# ── McpTool ──────────────────────────────────────────────


def test_mcp_tool_name() -> None:
    """
    McpTool.name() returns the MCP tool definition's name.
    """
    tool_def = _make_mcp_tool_def("my_tool")
    conn = McpServerConnection(config=_make_http_config())
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        # Stub run_sync — not exercised in this test.
        run_sync=MagicMock(),
    )
    assert tool.name() == "my_tool"


def test_mcp_tool_schema_openai_format() -> None:
    """
    McpTool.get_schema returns an OpenAI-format tool schema.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_http_config())
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


def test_mcp_tool_invoke_delegates_to_run_sync(tool_ctx: ToolContext) -> None:
    """
    McpTool.invoke parses JSON args and calls the connection's
    call_tool method via the run_sync callable.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_http_config())

    mock_run_sync = MagicMock(return_value="result text")
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        run_sync=mock_run_sync,
    )
    result = tool.invoke(json.dumps({"query": "hello"}), tool_ctx)

    assert result == "result text"
    mock_run_sync.assert_called_once()


def test_mcp_tool_invoke_returns_error_on_invalid_json(tool_ctx: ToolContext) -> None:
    """
    McpTool.invoke returns an error string (not raises) when the
    LLM sends malformed JSON arguments.

    If this crashed with JSONDecodeError instead of returning an
    error string, the workflow would abort instead of letting the
    LLM retry with corrected arguments.
    """
    tool_def = _make_mcp_tool_def("search")
    conn = McpServerConnection(config=_make_http_config())
    mock_run_sync = MagicMock(return_value="should not be called")
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        run_sync=mock_run_sync,
    )

    result = tool.invoke("not valid json {{{", tool_ctx)

    # Error message returned to the LLM, not an exception.
    # If this were a JSONDecodeError instead of a string, the
    # workflow would crash instead of letting the LLM retry.
    assert "Invalid JSON arguments" in result
    # run_sync must not be called — error caught before dispatch
    mock_run_sync.assert_not_called()


def test_mcp_tool_invoke_empty_string_parses_as_empty_dict(tool_ctx: ToolContext) -> None:
    """
    Empty-string arguments parse as ``{}`` so tools with no
    required parameters can be called without arguments.
    """
    tool_def = _make_mcp_tool_def("list_directory")
    conn = McpServerConnection(config=_make_http_config())
    mock_run_sync = MagicMock(return_value="ok")
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        run_sync=mock_run_sync,
    )

    result = tool.invoke("", tool_ctx)

    assert result == "ok"
    # Exactly one dispatch — empty string parsed as {} successfully.
    # If parsing failed, run_sync would not be called.
    mock_run_sync.assert_called_once()


def test_mcp_server_config_repr_redacts_headers() -> None:
    """
    MCPServerConfig.__repr__ replaces header values with
    ``[REDACTED]`` so credentials don't leak in logs or
    exception tracebacks.

    If the real Authorization token appeared in repr, it would
    be captured by ``_logger.exception()`` in manager.py when
    MCP connections fail.
    """
    config = MCPServerConfig(
        name="secret-svc",
        url="http://example.com/sse",
        headers={
            "Authorization": "Bearer sk-SUPER-SECRET-TOKEN",
            "X-Custom": "also-secret",
        },
    )
    r = repr(config)

    # Header keys are visible (useful for debugging which headers are set)
    assert "Authorization" in r
    assert "X-Custom" in r
    # Actual secret values must NOT appear
    assert "sk-SUPER-SECRET-TOKEN" not in r
    assert "also-secret" not in r
    # Redaction marker is present
    assert "[REDACTED]" in r
    # Non-sensitive fields are still visible
    assert "secret-svc" in r
    assert "http://example.com/sse" in r


def test_mcp_server_config_repr_empty_headers() -> None:
    """
    Repr works correctly when there are no headers — no crash,
    no ``[REDACTED]`` in the output.
    """
    config = MCPServerConfig(name="plain", url="http://localhost/sse")
    r = repr(config)

    assert "plain" in r
    assert "http://localhost/sse" in r
    assert "[REDACTED]" not in r


# ── _normalize_input_schema ───────────────────────────────


def test_normalize_none_schema_returns_empty_object() -> None:
    """
    ``None`` inputSchema (tool has no parameters) is normalized
    to a valid empty object schema.
    """
    result = _normalize_input_schema(None, "no_args_tool")
    assert result == {"type": "object", "properties": {}}


def test_normalize_missing_properties_injects_empty() -> None:
    """
    A schema with ``type: object`` but no ``properties`` key
    gets ``properties: {}`` injected. OpenAI rejects schemas
    without this key (openai/openai-agents-python#449).
    """
    schema = {"type": "object"}
    result = _normalize_input_schema(schema, "bare_object_tool")
    assert result["properties"] == {}
    assert result["type"] == "object"


def test_normalize_preserves_existing_properties() -> None:
    """
    A schema that already has ``properties`` is returned as-is
    (no double-injection).
    """
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    result = _normalize_input_schema(schema, "normal_tool")
    assert result["properties"] == {"query": {"type": "string"}}
    assert result["required"] == ["query"]


def test_normalize_does_not_mutate_original_schema() -> None:
    """
    ``_normalize_input_schema`` returns a new dict when
    modifying — it does not mutate the original schema dict.
    """
    original = {"type": "object"}
    result = _normalize_input_schema(original, "test")
    # Result has properties injected.
    assert "properties" in result
    # Original is untouched.
    assert "properties" not in original


def test_normalize_non_object_schema_unchanged() -> None:
    """
    A schema with a non-object type (e.g. ``array``) is not
    modified — ``properties`` injection only applies to objects.
    """
    schema = {"type": "array", "items": {"type": "string"}}
    result = _normalize_input_schema(schema, "array_tool")
    assert result == schema
    assert "properties" not in result


def test_normalize_warns_on_ref(caplog: pytest.LogCaptureFixture) -> None:
    """
    A schema containing ``$ref`` triggers a warning log.
    """
    schema = {
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Item"},
        },
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
    }
    _normalize_input_schema(schema, "ref_tool")
    assert any("$ref" in msg for msg in caplog.messages)


def test_normalize_warns_on_oneof(caplog: pytest.LogCaptureFixture) -> None:
    """
    A schema containing ``oneOf`` triggers a warning log.
    """
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
            },
        },
    }
    _normalize_input_schema(schema, "oneof_tool")
    assert any("oneOf" in msg for msg in caplog.messages)


def test_normalize_no_warning_for_clean_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A clean schema with no problematic keywords produces no
    warnings.
    """
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    _normalize_input_schema(schema, "clean_tool")
    assert not any("reject" in msg or "inconsistent" in msg for msg in caplog.messages)


# ── _collect_problematic_keywords ─────────────────────────


def test_collect_finds_ref_in_properties() -> None:
    """
    ``$ref`` nested inside a property is detected.
    """
    schema = {
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Item"},
        },
    }
    assert "$ref" in _collect_problematic_keywords(schema)


def test_collect_finds_oneof_in_nested_property() -> None:
    """
    ``oneOf`` inside a nested property is detected.
    """
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
            },
        },
    }
    assert "oneOf" in _collect_problematic_keywords(schema)


def test_collect_finds_keywords_in_array_items() -> None:
    """
    Problematic keywords inside ``items`` of an array are
    detected.
    """
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "x": {"$ref": "#/$defs/X"},
            },
        },
    }
    assert "$ref" in _collect_problematic_keywords(schema)


def test_collect_finds_keywords_in_defs() -> None:
    """
    Problematic keywords inside ``$defs`` are detected.
    """
    schema = {
        "type": "object",
        "properties": {},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {"$ref": "#/$defs/Node"},
                },
            },
        },
    }
    assert "$ref" in _collect_problematic_keywords(schema)


def test_collect_returns_empty_for_clean_schema() -> None:
    """
    A clean schema returns an empty set.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    assert _collect_problematic_keywords(schema) == set()


def test_mcp_tool_schema_normalizes_missing_properties() -> None:
    """
    ``McpTool.get_schema()`` normalizes a schema missing
    ``properties`` — the LLM receives a valid schema.
    """
    tool_def = MagicMock()
    tool_def.name = "bare_tool"
    tool_def.description = "A tool with no properties."
    # MCP allows {"type": "object"} with no properties.
    tool_def.inputSchema = {"type": "object"}

    conn = McpServerConnection(config=_make_http_config())
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        run_sync=MagicMock(),
    )
    schema = tool.get_schema()

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}


def test_mcp_tool_schema_normalizes_none_input_schema() -> None:
    """
    ``McpTool.get_schema()`` handles ``inputSchema=None``
    (no parameters) by returning a valid empty object schema.
    """
    tool_def = MagicMock()
    tool_def.name = "no_params_tool"
    tool_def.description = "Tool with no inputSchema."
    tool_def.inputSchema = None

    conn = McpServerConnection(config=_make_http_config())
    tool = McpTool(
        tool_def=tool_def,
        connection=conn,
        run_sync=MagicMock(),
    )
    schema = tool.get_schema()

    params = schema["function"]["parameters"]
    assert params == {"type": "object", "properties": {}}


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
    config = _make_http_config()
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


# ── _backoff_delay ────────────────────────────────────────


def test_backoff_delay_increases_with_attempt() -> None:
    """
    ``_backoff_delay`` increases with each attempt (exponential
    backoff) and applies jitter (0.5–1.0x).
    """
    retry = RetryConfig(backoff_base=2.0, backoff_max=30.0)
    # Attempt 0: base^1 = 2.0, with jitter [1.0, 2.0]
    delay_0 = _backoff_delay(0, retry)
    assert 1.0 <= delay_0 <= 2.0

    # Attempt 1: base^2 = 4.0, with jitter [2.0, 4.0]
    delay_1 = _backoff_delay(1, retry)
    assert 2.0 <= delay_1 <= 4.0

    # Attempt 2: base^3 = 8.0, with jitter [4.0, 8.0]
    delay_2 = _backoff_delay(2, retry)
    assert 4.0 <= delay_2 <= 8.0


def test_backoff_delay_capped_at_max() -> None:
    """
    ``_backoff_delay`` never exceeds ``backoff_max`` (before
    jitter).
    """
    retry = RetryConfig(backoff_base=10.0, backoff_max=5.0)
    # 10^(0+1) = 10, capped to 5.0, jitter [2.5, 5.0]
    delay = _backoff_delay(0, retry)
    assert delay <= 5.0


# ── Reconnection on server death ─────────────────────────


@pytest.mark.asyncio()
async def test_call_tool_reconnects_on_connection_error() -> None:
    """
    When a tool call fails with a connection error, the
    connection reconnects with backoff and retries.
    """
    config = _make_http_config()

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
        # Patch asyncio.sleep to avoid real delays in tests.
        with patch.object(conn, "_reconnect", new_callable=AsyncMock) as mock_reconnect:
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
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
    config = _make_http_config()

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
async def test_call_tool_exhausts_all_retries_then_raises() -> None:
    """
    When all reconnect attempts fail, the last connection
    error is propagated to the caller.
    """
    # Use a config with exactly 3 max_attempts so we
    # expect 3 invoke calls total (1 initial + 2 retries).
    config = MCPServerConfig(
        name="test-retry-exhaust",
        url="http://localhost:9000/mcp",
        retry=RetryConfig(
            max_attempts=3,
            backoff_base=1.0,
            backoff_max=10.0,
        ),
    )

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        mock_session.call_tool.side_effect = [
            EOFError("attempt 1"),
            EOFError("attempt 2"),
            EOFError("attempt 3"),
        ]

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EOFError, match="attempt 3"):
                    await conn.call_tool("test_tool", {"query": "hi"})

        # 3 invoke calls, 2 reconnects (no reconnect after last failure).
        assert mock_session.call_tool.await_count == 3


@pytest.mark.asyncio()
async def test_call_tool_uses_config_retry_policy() -> None:
    """
    ``call_tool()`` uses the per-server ``config.retry`` when
    set, rather than the module-level default.
    """
    # Only 2 attempts — should fail after 2 invoke calls.
    config = MCPServerConfig(
        name="test-custom-retry",
        url="http://localhost:9000/mcp",
        retry=RetryConfig(
            max_attempts=2,
            backoff_base=0.5,
            backoff_max=5.0,
        ),
    )

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        mock_session.call_tool.side_effect = [
            EOFError("attempt 1"),
            EOFError("attempt 2"),
        ]

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EOFError, match="attempt 2"):
                    await conn.call_tool("test_tool", {"query": "hi"})

        assert mock_session.call_tool.await_count == 2


@pytest.mark.asyncio()
async def test_call_tool_sleeps_between_retries() -> None:
    """
    ``call_tool()`` sleeps with backoff between reconnect
    attempts. Verifies that ``asyncio.sleep`` is called with
    increasing delays.
    """
    config = MCPServerConfig(
        name="test-backoff",
        url="http://localhost:9000/mcp",
        retry=RetryConfig(
            max_attempts=3,
            backoff_base=2.0,
            backoff_max=30.0,
        ),
    )

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        ok_result = MagicMock()
        ok_result.content = [TextContent(type="text", text="ok")]
        ok_result.isError = False

        mock_session.call_tool.side_effect = [
            EOFError("attempt 1"),
            EOFError("attempt 2"),
            ok_result,
        ]

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with patch(
                "agent_plane.tools.mcp.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep:
                result = await conn.call_tool("test_tool", {"query": "hi"})

        assert result == "ok"
        # Two sleeps: before retry 1 and before retry 2.
        assert mock_sleep.await_count == 2
        # Delays should increase (backoff_base=2.0: 2^1, 2^2).
        # Jitter applies 0.5–1.0x, so delay 1 is in [1.0, 2.0],
        # delay 2 is in [2.0, 4.0].
        delay_1 = mock_sleep.await_args_list[0].args[0]
        delay_2 = mock_sleep.await_args_list[1].args[0]
        assert 1.0 <= delay_1 <= 2.0
        assert 2.0 <= delay_2 <= 4.0


@pytest.mark.asyncio()
async def test_call_tool_default_retry_has_three_attempts() -> None:
    """
    When ``config.retry`` is ``None``, ``call_tool()`` falls
    back to ``_MCP_RECONNECT_DEFAULTS`` which allows 3 attempts.
    """
    # No retry config — should use default (3 attempts).
    config = MCPServerConfig(
        name="test-default-retry",
        url="http://localhost:9000/mcp",
        # retry defaults to None
    )

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()

        mock_session.call_tool.side_effect = [
            EOFError("attempt 1"),
            EOFError("attempt 2"),
            EOFError("attempt 3"),
        ]

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EOFError, match="attempt 3"):
                    await conn.call_tool("test_tool", {"query": "hi"})

        assert mock_session.call_tool.await_count == _MCP_RECONNECT_DEFAULTS.max_attempts


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
        url="http://localhost:9000/mcp",
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
            "agent_plane.tools.mcp.sse_client",
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
        url="http://localhost:9000/mcp",
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
            "agent_plane.tools.mcp.sse_client",
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
    When ``MCPServerConfig(timeout=60)``,
    ``connect()`` must pass ``timeout=60.0`` and
    ``sse_read_timeout=60.0`` to ``sse_client``.
    """
    config = MCPServerConfig(
        name="test-http-timeout",
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
    When ``MCPServerConfig(timeout=None)``,
    ``connect()`` must pass the MCP SDK defaults: ``timeout=5``
    and ``sse_read_timeout=300``.
    """
    config = MCPServerConfig(
        name="test-http-default",
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
    returns them.
    """
    config = MCPServerConfig(
        name="test-http-discovery",
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
    error.
    """
    config = MCPServerConfig(
        name="test-http-reconnect",
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
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
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


# ── Circuit breaker ──────────────────────────────────────────


def test_circuit_breaker_allows_calls_when_closed() -> None:
    """
    A fresh breaker in CLOSED state allows calls without raising.
    """
    breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=10.0)
    # Should not raise.
    breaker.pre_call("test-server")


def test_circuit_breaker_trips_after_threshold_failures() -> None:
    """
    The breaker trips after ``failure_threshold`` consecutive
    failures and blocks subsequent calls.
    """
    breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
    for _ in range(3):
        breaker.record_failure("test-server")

    assert breaker.is_tripped is True
    with pytest.raises(McpServerDisabledError) as exc_info:
        breaker.pre_call("my-server")
    assert exc_info.value.server_name == "my-server"
    assert exc_info.value.consecutive_failures == 3


def test_circuit_breaker_does_not_trip_below_threshold() -> None:
    """
    Fewer failures than the threshold do not trip the breaker.
    """
    breaker = _CircuitBreaker(failure_threshold=5, cooldown_seconds=10.0)
    for _ in range(4):
        breaker.record_failure("test-server")

    assert breaker.is_tripped is False
    # Should not raise.
    breaker.pre_call("test-server")


def test_circuit_breaker_resets_on_success() -> None:
    """
    A successful call resets the failure counter and un-trips
    the breaker.
    """
    breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
    for _ in range(3):
        breaker.record_failure("test-server")
    assert breaker.is_tripped is True

    breaker.record_success()
    assert breaker.is_tripped is False
    assert breaker.consecutive_failures == 0
    # Should not raise after reset.
    breaker.pre_call("test-server")


def test_circuit_breaker_half_open_after_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    After the cooldown period elapses, the breaker enters
    half-open state and allows one probe call.
    """
    import time as time_module

    breaker = _CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    breaker.record_failure("test-server")
    breaker.record_failure("test-server")
    assert breaker.is_tripped is True

    # Advance time past the cooldown.
    original_monotonic = time_module.monotonic
    monkeypatch.setattr(
        time_module,
        "monotonic",
        lambda: original_monotonic() + 15.0,
    )

    # Cooldown elapsed — half-open state allows one probe.
    assert breaker.is_tripped is False
    breaker.pre_call("test-server")  # Should not raise.


def test_circuit_breaker_re_trips_on_half_open_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If the half-open probe fails, the breaker re-trips.
    """
    import time as time_module

    breaker = _CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    breaker.record_failure("test-server")
    breaker.record_failure("test-server")

    # Advance time past the cooldown.
    original_monotonic = time_module.monotonic
    monkeypatch.setattr(
        time_module,
        "monotonic",
        lambda: original_monotonic() + 15.0,
    )

    # Half-open probe allowed.
    breaker.pre_call("test-server")
    # Probe fails — re-trip.
    breaker.record_failure("test-server")
    assert breaker.is_tripped is True


def test_circuit_breaker_cooldown_remaining_in_error() -> None:
    """
    The ``McpServerDisabledError`` includes the approximate
    cooldown remaining.
    """
    breaker = _CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("test-server")

    with pytest.raises(McpServerDisabledError) as exc_info:
        breaker.pre_call("test-server")
    # Cooldown just started, so remaining should be close to 30s.
    assert exc_info.value.cooldown_remaining > 25.0


def test_circuit_breaker_failure_count_resets_on_success() -> None:
    """
    Interspersed successes prevent the breaker from tripping.
    """
    breaker = _CircuitBreaker(failure_threshold=3, cooldown_seconds=10.0)
    breaker.record_failure("test-server")
    breaker.record_failure("test-server")
    # Success resets the counter.
    breaker.record_success()
    breaker.record_failure("test-server")
    breaker.record_failure("test-server")
    # Only 2 consecutive failures — not at threshold.
    assert breaker.is_tripped is False


def test_circuit_breaker_trip_log_includes_server_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    When the breaker trips, the warning log message includes the
    server name so operators can identify which MCP server failed.

    :param caplog: Pytest fixture that captures log records.
    """
    breaker = _CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    with caplog.at_level(logging.WARNING, logger="agent_plane.tools.mcp"):
        breaker.record_failure("my-flaky-server")
        breaker.record_failure("my-flaky-server")

    assert breaker.is_tripped is True
    # The trip log must name the specific server.
    trip_messages = [r.message for r in caplog.records if "tripped" in r.message]
    assert len(trip_messages) == 1, f"Expected 1 trip log, got {len(trip_messages)}"
    assert "my-flaky-server" in trip_messages[0]


def test_circuit_breaker_default_constants() -> None:
    """
    Module-level circuit breaker constants have expected values.
    """
    assert _CIRCUIT_BREAKER_THRESHOLD == 5
    assert _CIRCUIT_BREAKER_COOLDOWN_SECONDS == 30.0


@pytest.mark.asyncio
async def test_call_tool_trips_breaker_after_repeated_failures() -> None:
    """
    ``McpServerConnection.call_tool()`` records failures in the
    circuit breaker. After ``failure_threshold`` exhausted
    invocations, subsequent calls raise ``McpServerDisabledError``.
    """
    config = _make_http_config()

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()
        # Override breaker threshold to 2 for a quick test.
        conn._breaker = _CircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=60.0,
        )

        mock_session.call_tool = AsyncMock(side_effect=EOFError("dead"))

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
                # First call: exhausts retries, records failure.
                with pytest.raises(EOFError):
                    await conn.call_tool("my_tool", {"x": 1})

                # Second call: exhausts retries, trips breaker.
                with pytest.raises(EOFError):
                    await conn.call_tool("my_tool", {"x": 1})

        # Third call: breaker is tripped, fails immediately.
        with pytest.raises(McpServerDisabledError) as exc_info:
            await conn.call_tool("my_tool", {"x": 1})
        assert exc_info.value.server_name == config.name
        assert exc_info.value.consecutive_failures == 2

    await conn.close()


@pytest.mark.asyncio
async def test_call_tool_resets_breaker_on_success() -> None:
    """
    A successful ``call_tool()`` resets the circuit breaker so
    that prior failures don't accumulate across successes.
    """
    config = _make_http_config()

    with _mock_mcp_transport() as mock_session:
        conn = McpServerConnection(config=config)
        await conn.connect()
        conn._breaker = _CircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=60.0,
        )

        ok_result = MagicMock()
        ok_result.content = [TextContent(type="text", text="ok")]
        ok_result.isError = False

        # 3 fails then 1 success (reconnect retries 3 per call).
        mock_session.call_tool = AsyncMock(
            side_effect=[
                EOFError("dead"),
                EOFError("dead"),
                EOFError("dead"),
                EOFError("dead"),
                EOFError("dead"),
                ok_result,
            ]
        )

        with patch.object(conn, "_reconnect", new_callable=AsyncMock):
            with patch("agent_plane.tools.mcp.asyncio.sleep", new_callable=AsyncMock):
                # First invocation: all 3 retries fail → failure (count=1).
                with pytest.raises(EOFError):
                    await conn.call_tool("my_tool", {})

                # Second invocation: 3rd retry succeeds → reset.
                result = await conn.call_tool("my_tool", {})
                assert result == "ok"

        # Breaker should be reset — no accumulated failures.
        assert conn._breaker.consecutive_failures == 0

    await conn.close()


# ── Circuit breaker half-open atomic gate ─────────────────


def test_circuit_breaker_half_open_clears_tripped_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Entering half-open state clears ``_tripped_at`` so that
    concurrent callers see CLOSED (not half-open) and don't
    also enter the probe path.

    :param monkeypatch: Pytest monkeypatch for time.monotonic.
    """
    import time as time_module

    breaker = _CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    breaker.record_failure("test-server")
    breaker.record_failure("test-server")
    assert breaker.is_tripped is True

    # Advance time past the cooldown.
    original_monotonic = time_module.monotonic
    monkeypatch.setattr(
        time_module,
        "monotonic",
        lambda: original_monotonic() + 15.0,
    )

    # First pre_call enters half-open and clears the tripped state.
    breaker.pre_call("test-server")
    # Breaker should no longer report as tripped — a concurrent
    # caller sees CLOSED, not half-open.
    assert breaker.is_tripped is False
    # Second pre_call should not raise (sees CLOSED state).
    breaker.pre_call("test-server")


# ── EventLoopThread ──────────────────────────────────────


def test_event_loop_thread_stop_logs_warning_on_join_timeout(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If the background thread does not exit within the join
    timeout, ``stop()`` logs a warning and skips ``loop.close()``
    to avoid undefined behavior on a still-running loop.

    :param caplog: Pytest fixture that captures log records.
    :param monkeypatch: Pytest monkeypatch for the timeout constant.
    """
    # Use a short timeout so the test doesn't block.
    monkeypatch.setattr("agent_plane.tools.mcp._LOOP_STOP_TIMEOUT_SECONDS", 0.1)

    elt = EventLoopThread()
    # Replace the thread's join to simulate a stuck thread: after
    # join() returns, is_alive() still reports True.
    original_join = elt._thread.join

    def _fake_join(timeout: float | None = None) -> None:
        """
        Simulate a thread that ignores the join timeout.

        :param timeout: Ignored — the thread stays alive.
        """
        original_join(timeout=0.01)

    monkeypatch.setattr(elt._thread, "join", _fake_join)
    monkeypatch.setattr(elt._thread, "is_alive", lambda: True)

    with caplog.at_level(logging.WARNING, logger="agent_plane.tools.mcp"):
        elt.stop()

    # Warning should mention the timeout.
    warning_msgs = [r.message for r in caplog.records if "did not stop" in r.message]
    assert len(warning_msgs) == 1, f"Expected 1 warning, got {len(warning_msgs)}"
    # loop.close() should NOT have been called (skipped for stuck thread).
    assert not elt._loop.is_closed()

    # Cleanup: actually stop the loop so the daemon thread exits.
    if elt._loop.is_running():
        elt._loop.call_soon_threadsafe(elt._loop.stop)


def test_event_loop_thread_stop_closes_loop_on_clean_exit() -> None:
    """
    Normal ``stop()`` closes the event loop when the thread
    exits cleanly within the timeout.
    """
    elt = EventLoopThread()
    elt.stop()
    assert elt._loop.is_closed()
