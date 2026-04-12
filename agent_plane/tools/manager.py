"""Registry-based tool manager for agent execution.

Each workflow creates its own ToolManager, connects MCP servers at
start, and tears them down in finally. MCP discovery results are
cached across workflow executions to avoid repeated round-trips.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from mcp.types import Tool as McpToolDef

from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.spec import AgentSpec
from agent_plane.tools.base import Tool, ToolContext, is_valid_tool_name
from agent_plane.tools.builtins import (
    CancelSubAgentTool,
    CheckSubAgentsTool,
    LoadSkillTool,
    ReadSkillFileTool,
    SpawnTool,
    any_skill_has_resources,
    get_builtin_tool,
)
from agent_plane.tools.client_specified import ClientSideTool, ClientSideToolSpec
from agent_plane.tools.local import load_local_python_tools
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
    client-specified tools) and at ``start()`` time (MCP tools
    discovered from configured servers). Dispatch is via
    ``self._tools[name].invoke(arguments)`` — no hardcoded if/elif
    chains.

    Owns a persistent :class:`EventLoopThread` for MCP operations.
    MCP sessions are bound to the event loop they were created on,
    so ``connect()``, ``call_tool()``, and ``close()`` must all
    run on the same loop.

    Registers at init:
    - ``load_skill`` (if the agent has skills)
    - ``read_skill_file`` (if any skill has bundled resources)
    - Built-in tools from ``tools.builtins`` (e.g. ``web_search``)
    - One :class:`LocalPythonTool` per ``tools/python/*.py`` file
    - One :class:`ClientSideTool` per entry in ``client_tool_specs``

    Registers at ``start()``:
    - MCP tools discovered from ``mcp_servers`` in the agent spec
    """

    def __init__(
        self,
        spec: AgentSpec,
        client_tool_specs: list[ClientSideToolSpec] | None = None,
        workdir: Path | None = None,
        sandbox_enabled: bool = True,
    ) -> None:
        """
        Initialize the tool manager and register built-in,
        client-specified, and local tools.

        MCP tools are not registered until ``start()`` is called.

        :param spec: The parsed AgentSpec defining which tools
            (skills, MCP servers) are available.
        :param client_tool_specs: Optional list of
            :class:`ClientSideToolSpec` objects supplied by the API
            caller at request time, e.g.
            ``[ClientSideToolSpec(name="get_weather", ...)]``.
            ``None`` and ``[]`` are equivalent (no client tools).
        :param workdir: The extracted agent image directory on disk.
            Required for local tool loading. ``None`` skips local
            tool registration, e.g. ``Path("/tmp/cache/ag_abc123")``.
        :param sandbox_enabled: Runtime policy for ``srt`` sandboxing.
            ``True`` enables sandboxing when ``srt`` is on PATH.
            This is a deployment decision from ``RuntimeCaps``, not
            an agent config setting.
        """
        self._spec = spec
        self._sandbox_enabled = sandbox_enabled
        self._started = False
        self._tools: dict[str, Tool] = {}
        self._mcp_connections: list[McpServerConnection] = []
        self._loop_thread: EventLoopThread | None = None
        self._srt_available = shutil.which("srt") is not None
        self._uv_available = shutil.which("uv") is not None
        self._register_skill_tools()
        self._register_builtin_tools()
        self._register_sub_agent_tools()
        self._register_local_tools(workdir)
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
        self._tools[load_tool.name()] = load_tool
        if any_skill_has_resources(self._spec.skills):
            read_tool = ReadSkillFileTool(self._spec.skills)
            self._tools[read_tool.name()] = read_tool

    def _register_builtin_tools(self) -> None:
        """
        Register built-in tools declared in ``tools.builtins``.

        Most tools are looked up in the built-in registry and
        instantiated with spec-level config. ``code_sandbox`` and
        ``upload_file`` are handled specially — they need sandbox
        capability flags from ToolManager.
        """
        for entry in self._spec.tools.builtins:
            tool = self._create_builtin(entry.name, entry.config)
            if tool is None:
                _logger.warning(
                    "Unknown built-in tool %r — skipping. "
                    "Available: web_search, code_sandbox, upload_file, "
                    "search_conversations, list_files, download_file",
                    entry.name,
                )
                continue
            self._tools[tool.name()] = tool

    def _create_builtin(
        self,
        name: str,
        config: dict[str, str] | None,
    ) -> Tool | None:
        """
        Instantiate a built-in tool by name.

        :param name: The builtin name from the spec.
        :param config: Optional spec-level config dict.
        :returns: A :class:`Tool` instance, or ``None``.
        """
        if name == "web_search":
            return self._create_web_search(config)
        if name == "web_fetch":
            return self._create_web_fetch()
        if name == "code_sandbox":
            return self._create_code_sandbox()
        if name == "upload_file":
            from agent_plane.tools.builtins.upload_file import UploadFileTool

            return UploadFileTool()
        return get_builtin_tool(name, config=config)

    def _create_web_search(self, config: dict[str, str] | None) -> Tool:
        """
        Build a WebSearchTool with the parent's LLM provider.

        Uses ``parse_model_string`` to resolve the provider from
        the model string (e.g. ``"gpt-5.4"`` → ``"openai"``).

        :param config: Spec-level tool config.
        :returns: A configured WebSearchTool.
        """
        from agent_plane.tools.builtins.web_search import WebSearchTool

        llm_provider = None
        if self._spec.llm is not None and self._spec.llm.model:
            from agent_plane.llms.routing import parse_model_string

            llm_provider = parse_model_string(self._spec.llm.model).provider
        return WebSearchTool(config=config, llm_provider=llm_provider)

    def _create_web_fetch(self) -> Tool:
        """
        Build a WebFetchTool with the parent's spec.

        :returns: A WebFetchTool that inherits the parent's LLM config.
        """
        from agent_plane.tools.builtins.web_fetch import WebFetchTool

        return WebFetchTool(parent_spec=self._spec)

    def _create_code_sandbox(self) -> Tool:
        """
        Build a CodeSandboxTool with runtime capability flags.

        :returns: A CodeSandboxTool with srt/sandbox settings.
        """
        from agent_plane.tools.builtins.code_sandbox import CodeSandboxTool

        return CodeSandboxTool(
            srt_available=self._srt_available,
            sandbox_enabled=self._sandbox_enabled,
        )

    def _register_sub_agent_tools(self) -> None:
        """
        Register spawn/collect tools when the agent has sub-agents
        declared in ``tools.agents``.

        Builds a name-to-spec lookup from the agent's
        ``sub_agents`` list and registers :class:`SpawnTool` and
        :class:`CheckSubAgentsTool`.
        """
        if not self._spec.tools.agents:
            return

        sub_specs = {sa.name: sa for sa in self._spec.sub_agents if sa.name is not None}
        spawn = SpawnTool(sub_specs=sub_specs)
        self._tools[SpawnTool.name()] = spawn
        self._tools[CheckSubAgentsTool.name()] = CheckSubAgentsTool()
        self._tools[CancelSubAgentTool.name()] = CancelSubAgentTool()

    def _register_local_tools(self, workdir: Path | None) -> None:
        """
        Load and register local Python tools from the agent image.

        Tools with invalid names are skipped with a warning.
        If ``workdir`` is ``None`` or the spec has no local tools,
        this is a no-op.

        :param workdir: The agent image directory, or ``None``.
        """
        if workdir is None or not self._spec.local_tools:
            return
        for tool in load_local_python_tools(
            self._spec.local_tools,
            workdir,
            sandbox_config=self._spec.tools.sandbox,
            srt_available=self._srt_available,
            uv_available=self._uv_available,
            sandbox_enabled=self._sandbox_enabled,
        ):
            if not is_valid_tool_name(tool.name()):
                _logger.warning(
                    "Local tool %r has invalid name — skipping",
                    tool.name(),
                )
                continue
            if tool.name() in self._tools:
                _logger.warning(
                    "Local tool %r shadows existing tool — overwriting",
                    tool.name(),
                )
            self._tools[tool.name()] = tool

    def _register_client_tools(
        self,
        specs: list[ClientSideToolSpec],
    ) -> None:
        """
        Register client-specified tools.

        Raises :class:`AgentPlaneError` if a tool name violates the
        OpenAI function-calling constraint
        (``^[a-zA-Z0-9_-]{1,256}$``). If a client tool name collides
        with an already-registered tool (e.g. a built-in skill tool),
        the client tool wins and a warning is logged.

        :param specs: List of :class:`ClientSideToolSpec` objects to
            register, e.g.
            ``[ClientSideToolSpec(name="get_weather", ...)]``.
        :raises AgentPlaneError: If any tool name is invalid.
        """
        for spec in specs:
            if not is_valid_tool_name(spec.name):
                raise AgentPlaneError(
                    f"Invalid client tool name {spec.name!r}: must match [a-zA-Z0-9_-]{{1,256}}",
                    code=ErrorCode.INVALID_INPUT,
                )
            if spec.name in self._tools:
                _logger.warning(
                    "Client-specified tool %r shadows existing tool — overwriting",
                    spec.name,
                )
            self._tools[spec.name] = ClientSideTool(spec)

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

    def get_tool_names(self) -> list[str]:
        """
        Return the names of all registered tools.

        :returns: Tool names, e.g. ``["spawn_sub_agents",
            "load_skill", "web_search"]``.
        """
        return list(self._tools.keys())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Return OpenAI-format tool schemas for all registered
        tools.

        :returns: A list of OpenAI tool schema dicts, each
            with ``"type": "function"`` and a ``"function"``
            sub-dict describing the tool.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def get_tool(self, name: str) -> Tool | None:
        """
        Look up a registered tool by name.

        :param name: The tool function name, e.g. ``"load_skill"``.
        :returns: The :class:`Tool` instance, or ``None`` if not
            registered.
        """
        return self._tools.get(name)

    def call_tool(
        self,
        name: str,
        arguments: str,
        ctx: ToolContext,
    ) -> str:
        """
        Dispatch a tool call to the registered handler.

        :param name: The tool function name, e.g.
            ``"load_skill"`` or ``"github_list_issues"``.
        :param arguments: JSON-encoded arguments string from
            the LLM, e.g. ``'{"name": "summarize"}'``.
        :param ctx: Server-side execution context with task
            and agent identity.
        :returns: The tool's string result, or an error
            message if the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool {name!r} not found. Registered tools: {list(self._tools.keys())}"
        return tool.invoke(arguments, ctx)

    def get_client_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Return the raw OpenAI-format schemas for all registered
        client-side tools.

        Used by :class:`SpawnTool` to propagate client tools to
        sub-agent workflows — the sub-agent's LLM needs the schemas
        so it knows which tools are available.

        :returns: List of tool schema dicts, e.g.
            ``[{"type": "function", "function": {"name": "Read", ...}}]``.
            Empty list if no client tools are registered.
        """
        return [
            tool.get_schema() for tool in self._tools.values() if isinstance(tool, ClientSideTool)
        ]

    def is_client_side_tool(self, name: str) -> bool:
        """
        Return ``True`` if the named tool is a :class:`ClientSideTool`.

        Used by the agent loop to detect when the LLM has invoked a
        client-side tool. On detection, the workflow persists the
        ``function_call`` items, streams them to the caller, and
        completes the response without executing any tools server-side.

        :param name: The tool function name, e.g. ``"get_weather"``.
        :returns: ``True`` if the tool is a :class:`ClientSideTool`,
            ``False`` if the tool is not registered or is a different
            tool type.
        """
        return isinstance(self._tools.get(name), ClientSideTool)

    async def _connect_mcp_servers(self) -> None:
        """
        Connect to all configured MCP servers and register
        their tools.

        Each server is connected sequentially. If a server
        fails to connect, it is logged and skipped — other
        servers still proceed.
        """
        for config in self._spec.mcp_servers:
            conn = McpServerConnection(config=config)
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

        Tools with names that violate the OpenAI function-calling
        constraint (``^[a-zA-Z0-9_-]{1,256}$``) are skipped with a
        warning — MCP servers may expose tools whose names contain
        characters that LLM providers reject.

        :param connection: The MCP server connection that
            owns these tools.
        :param tools: List of MCP tool definitions from
            ``tools/list``.
        """
        if self._loop_thread is None:
            raise RuntimeError("EventLoopThread not initialized — call start() first")
        for tool_def in tools:
            if not is_valid_tool_name(tool_def.name):
                _logger.warning(
                    "MCP tool %r from server %r has an invalid name "
                    "(must match [a-zA-Z0-9_-]{1,256}) — skipping",
                    tool_def.name,
                    connection.config.name,
                )
                continue
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
            self._tools[mcp_tool.name()] = mcp_tool

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
