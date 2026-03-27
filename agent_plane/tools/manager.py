"""Registry-based tool manager for agent execution.

Each workflow creates its own ToolManager, connects MCP servers at
start, and tears them down in finally. MCP discovery results are
cached across workflow executions to avoid repeated round-trips.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.types import Tool as McpToolDef

from agent_plane.spec import AgentSpec
from agent_plane.tools.base import Tool
from agent_plane.tools.builtins import (
    LoadSkillTool,
    ReadSkillFileTool,
    any_skill_has_resources,
)
from agent_plane.tools.client_specified import CallbackTool, CallbackToolSpec
from agent_plane.tools.mcp import (
    EventLoopThread,
    McpServerConnection,
    McpTool,
)

_logger = logging.getLogger(__name__)


class ToolManager:
    """
    Registry-based tool manager for a single workflow execution.

    Tools are registered at init time (built-in skill tools and
    client-specified callback tools) and at ``start()`` time (MCP
    tools discovered from configured servers). Dispatch is via
    ``self._tools[name].invoke(arguments)`` — no hardcoded if/elif
    chains.

    Owns a persistent :class:`EventLoopThread` for MCP operations.
    MCP sessions are bound to the event loop they were created on,
    so ``connect()``, ``call_tool()``, and ``close()`` must all
    run on the same loop.

    Registers at init:
    - ``load_skill`` (if the agent has skills)
    - ``read_skill_file`` (if any skill has bundled resources)
    - One :class:`CallbackTool` per entry in ``client_tool_specs``

    Registers at ``start()``:
    - MCP tools discovered from ``mcp_servers`` in the agent spec

    Not yet implemented:
    - Local tool execution (Python / TypeScript)
    """

    def __init__(
        self,
        spec: AgentSpec,
        work_dir: Path,
        client_tool_specs: list[CallbackToolSpec] | None = None,
    ) -> None:
        """
        Initialize the tool manager and register built-in and
        client-specified tools.

        MCP tools are not registered until ``start()`` is called.

        :param spec: The parsed AgentSpec defining which tools
            (skills, MCP servers) are available.
        :param work_dir: Path to the extracted agent image
            directory on disk, used as the working directory
            for local tool execution.
        :param client_tool_specs: Optional list of
            :class:`CallbackToolSpec` objects supplied by the API
            caller at request time, e.g.
            ``[CallbackToolSpec(name="get_weather", ...)]``.
            ``None`` and ``[]`` are equivalent (no client tools).
        """
        self._spec = spec
        self._work_dir = work_dir
        self._started = False
        self._tools: dict[str, Tool] = {}
        self._mcp_connections: list[McpServerConnection] = []
        self._loop_thread: EventLoopThread | None = None
        self._register_skill_tools()
        self._register_client_tools(client_tool_specs or [])

    def _register_skill_tools(self) -> None:
        """
        Register built-in skill tools based on the agent spec.

        Adds ``load_skill`` if the agent has skills, and
        ``read_skill_file`` if any skill has bundled resources.
        """
        if not self._spec.skills:
            return
        load_tool = LoadSkillTool(self._spec.skills)
        self._tools[load_tool.name] = load_tool
        if any_skill_has_resources(self._spec.skills):
            read_tool = ReadSkillFileTool(self._spec.skills)
            self._tools[read_tool.name] = read_tool

    def _register_client_tools(
        self,
        specs: list[CallbackToolSpec],
    ) -> None:
        """
        Register client-specified callback tools.

        If a client tool name collides with an already-registered
        tool (e.g. a built-in skill tool), the client tool wins
        and a warning is logged.

        :param specs: List of :class:`CallbackToolSpec` objects to
            register, e.g.
            ``[CallbackToolSpec(name="get_weather", ...)]``.
        """
        for spec in specs:
            if spec.name in self._tools:
                _logger.warning(
                    "Client-specified tool %r shadows existing tool — overwriting",
                    spec.name,
                )
            self._tools[spec.name] = CallbackTool(spec)

    def start(self) -> None:
        """
        Connect to MCP servers and discover their tools.

        Creates a persistent event loop thread for MCP
        operations, connects to each configured server, and
        registers their tools. Duplicate tool names across
        servers are logged as warnings — the last server wins.
        """
        if self._spec.mcp_servers:
            self._loop_thread = EventLoopThread()
            self._loop_thread.run(self._connect_mcp_servers())
        self._started = True

    def shutdown(self) -> None:
        """
        Disconnect from all MCP servers and stop the event loop.

        Safe to call even if ``start()`` was never called.
        """
        if self._mcp_connections and self._loop_thread is not None:
            self._loop_thread.run(self._close_mcp_servers())
        if self._loop_thread is not None:
            self._loop_thread.stop()
            self._loop_thread = None
        self._started = False

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Return OpenAI-format tool schemas for all registered
        tools.

        :returns: A list of OpenAI tool schema dicts, each
            with ``"type": "function"`` and a ``"function"``
            sub-dict describing the tool.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def call_tool(self, name: str, arguments: str) -> str:
        """
        Dispatch a tool call to the registered handler.

        :param name: The tool function name, e.g.
            ``"load_skill"`` or ``"github_list_issues"``.
        :param arguments: JSON-encoded arguments string from
            the LLM, e.g. ``'{"name": "summarize"}'``.
        :returns: The tool's string result, or an error
            message if the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool {name!r} not found. Registered tools: {list(self._tools.keys())}"
        return tool.invoke(arguments)

    async def _connect_mcp_servers(self) -> None:
        """
        Connect to all configured MCP servers and register
        their tools.

        Each server is connected sequentially. If a server
        fails to connect, it is logged and skipped — other
        servers still proceed.
        """
        for config in self._spec.mcp_servers:
            conn = McpServerConnection(
                config=config,
                work_dir=self._work_dir,
            )
            try:
                tools = await conn.connect()
            except Exception:
                _logger.exception(
                    "Failed to connect to MCP server %r",
                    config.name,
                )
                continue
            self._mcp_connections.append(conn)
            self._register_mcp_tools(conn, tools)

    def _register_mcp_tools(
        self,
        connection: McpServerConnection,
        tools: list[McpToolDef],
    ) -> None:
        """
        Register discovered MCP tools in the tool registry.

        :param connection: The MCP server connection that
            owns these tools.
        :param tools: List of MCP tool definitions from
            ``tools/list``.
        """
        assert self._loop_thread is not None
        for tool_def in tools:
            if tool_def.name in self._tools:
                _logger.warning(
                    "MCP tool %r from server %r shadows existing tool — overwriting",
                    tool_def.name,
                    connection.config.name,
                )
            mcp_tool = McpTool(
                tool_def=tool_def,
                connection=connection,
                run_sync=self._loop_thread.run,
            )
            self._tools[mcp_tool.name] = mcp_tool

    async def _close_mcp_servers(self) -> None:
        """
        Close all active MCP server connections.

        Errors during close are logged but not raised.
        """
        for conn in self._mcp_connections:
            try:
                await conn.close()
            except Exception:
                _logger.exception(
                    "Error closing MCP server %r",
                    conn.config.name,
                )
        self._mcp_connections.clear()
