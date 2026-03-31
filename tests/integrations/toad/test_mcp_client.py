"""Tests for the MCP client module used by the Toad ACP adapter."""

from __future__ import annotations

from typing import Any

from mcp.client.stdio import StdioServerParameters

from integrations.toad.mcp_client import (
    ToolSchema,
    _extract_text,
    _mcp_tool_to_schema,
    parse_mcp_server_params,
)


def test_parse_mcp_server_params_basic() -> None:
    """Parses command, args, env, cwd from raw ACP dict."""
    raw: dict[str, object] = {
        "command": "npx",
        "args": ["-y", "@mcp/server-fs", "/tmp"],
        "env": {"NODE_ENV": "production"},
        "name": "filesystem",
    }
    params = parse_mcp_server_params(raw, cwd="/home/user")
    assert params.command == "npx"
    assert params.args == ["-y", "@mcp/server-fs", "/tmp"]
    assert params.env == {"NODE_ENV": "production"}
    assert params.cwd == "/home/user"


def test_parse_mcp_server_params_minimal() -> None:
    """Minimal config with just command works."""
    raw: dict[str, object] = {"command": "my-server"}
    params = parse_mcp_server_params(raw)
    assert params.command == "my-server"
    assert params.args == []
    assert params.env is None
    assert params.cwd is None


def test_mcp_tool_to_schema_full() -> None:
    """MCP tool with description and schema converts to OpenAI format."""

    class FakeTool:
        """Mimics MCP Tool fields."""

        name = "read_file"
        description = "Read a file from disk"
        inputSchema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        }

    schema = _mcp_tool_to_schema(FakeTool())
    assert schema.name == "read_file"
    assert schema.schema["type"] == "function"
    func = schema.schema["function"]
    assert func["name"] == "read_file"
    assert func["description"] == "Read a file from disk"
    assert "path" in func["parameters"]["properties"]


def test_mcp_tool_to_schema_no_description() -> None:
    """MCP tool with no description gets empty string."""

    class FakeTool:
        """Mimics MCP Tool fields."""

        name = "do_thing"
        description = None
        inputSchema: dict[str, Any] | None = None

    schema = _mcp_tool_to_schema(FakeTool())
    assert schema.schema["function"]["description"] == ""
    # Should have a default empty parameters schema
    assert schema.schema["function"]["parameters"]["type"] == "object"


def test_extract_text_single_block() -> None:
    """Single TextContent block extracts cleanly."""
    from mcp.types import TextContent

    class FakeResult:
        """Mimics CallToolResult."""

        content = [TextContent(type="text", text="hello world")]

    assert _extract_text(FakeResult()) == "hello world"


def test_extract_text_multiple_blocks() -> None:
    """Multiple blocks are joined with newlines."""
    from mcp.types import TextContent

    class FakeResult:
        """Mimics CallToolResult."""

        content = [
            TextContent(type="text", text="line 1"),
            TextContent(type="text", text="line 2"),
        ]

    assert _extract_text(FakeResult()) == "line 1\nline 2"
