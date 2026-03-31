"""Lightweight MCP client for connecting to Toad-provided MCP servers.

Handles stdio-transport MCP servers (the common case for Toad),
discovers tools via ``tools/list``, and executes them via
``tools/call``. Each :class:`McpConnection` wraps one server.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent

log = logging.getLogger(__name__)


@dataclass
class ToolSchema:
    """An OpenAI-format tool schema discovered from an MCP server.

    :param name: Tool function name, e.g. ``"read_file"``.
    :param schema: Full OpenAI function tool dict suitable for
        the ``tools`` field in ``POST /v1/responses``.
    """

    name: str
    schema: dict[str, Any]


@dataclass
class McpConnection:
    """A connection to a single MCP server.

    Manages the lifecycle of a stdio subprocess, discovers tools,
    and routes ``call_tool`` requests.

    :param server_params: Parameters for launching the MCP server
        subprocess (command, args, env, cwd).
    :param server_name: Human-readable name for logging.
    """

    server_params: StdioServerParameters
    server_name: str
    _session: ClientSession | None = field(
        default=None, repr=False
    )
    _exit_stack: AsyncExitStack | None = field(
        default=None, repr=False
    )
    _tools: list[ToolSchema] = field(
        default_factory=list, repr=False
    )

    async def connect(self) -> list[ToolSchema]:
        """Start the MCP server subprocess and discover tools.

        :returns: List of discovered tool schemas in OpenAI format.
        """
        self._exit_stack = AsyncExitStack()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(self.server_params, errlog=sys.stderr)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools = [
            _mcp_tool_to_schema(tool) for tool in result.tools
        ]
        log.info(
            "MCP server %s: discovered %d tools",
            self.server_name,
            len(self._tools),
        )
        return self._tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool call on this MCP server.

        :param name: Tool name to invoke.
        :param arguments: Tool arguments as a dict.
        :returns: The tool's text output as a string.
        :raises RuntimeError: If not connected.
        """
        if self._session is None:
            raise RuntimeError(
                f"MCP server {self.server_name!r} not connected"
            )
        result: CallToolResult = await self._session.call_tool(
            name, arguments
        )
        return _extract_text(result)

    async def close(self) -> None:
        """Shut down the MCP server subprocess.

        Safe to call multiple times or if never connected.
        """
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        """Previously discovered tool schemas.

        :returns: List of :class:`ToolSchema` from the last
            :meth:`connect` call.
        """
        return self._tools


def _mcp_tool_to_schema(tool: Any) -> ToolSchema:
    """Convert an MCP Tool definition to OpenAI function schema.

    :param tool: An MCP ``Tool`` object with ``name``,
        ``description``, and ``inputSchema`` fields.
    :returns: A :class:`ToolSchema` with the OpenAI-format dict.
    """
    return ToolSchema(
        name=tool.name,
        schema={
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    )


def _extract_text(result: CallToolResult) -> str:
    """Extract text content from an MCP CallToolResult.

    Concatenates all ``TextContent`` blocks. Non-text content
    (images, resources) is represented as ``[type:...]``.

    :param result: The MCP tool call result.
    :returns: A string representation of the result.
    """
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            # Non-text content — include type indicator
            parts.append(f"[{block.type}]")
    return "\n".join(parts)


def parse_mcp_server_params(
    raw: dict[str, object],
    cwd: str | None = None,
) -> StdioServerParameters:
    """Parse an ACP ``mcpServers`` entry into MCP parameters.

    ACP sends MCP server configs as::

        {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
            "env": {"KEY": "value"},
            "name": "filesystem"
        }

    :param raw: The raw MCP server dict from ACP ``session/new``.
    :param cwd: Working directory for the subprocess, from the
        ACP session's ``cwd`` field.
    :returns: A :class:`StdioServerParameters` for ``stdio_client``.
    """
    command = str(raw["command"])
    args_raw = raw.get("args", [])
    args = [str(a) for a in args_raw] if isinstance(args_raw, list) else []
    env_raw = raw.get("env")
    env: dict[str, str] | None = None
    if isinstance(env_raw, dict):
        env = {str(k): str(v) for k, v in env_raw.items()}
    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
    )
