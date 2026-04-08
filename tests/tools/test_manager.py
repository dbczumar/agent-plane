"""Tests for agent_plane.tools.manager (ToolManager)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_plane.errors import AgentPlaneError
from agent_plane.spec.types import (
    AgentSpec,
    LocalToolInfo,
    MCPServerConfig,
    SkillSpec,
)
from agent_plane.tools import ToolManager
from agent_plane.tools.base import ToolContext
from agent_plane.tools.client_specified import ClientSideToolSpec
from agent_plane.tools.mcp import clear_discovery_cache

_TEST_CTX = ToolContext(task_id="task_test", agent_id="agent_test")


@pytest.fixture()
def skill_with_resources(tmp_path: Path) -> SkillSpec:
    """
    A skill with a ``references/`` directory containing a
    file, for testing ``read_skill_file`` registration.

    :returns: A ``SkillSpec`` pointing at a real directory
        with a reference file.
    """
    skill_dir = tmp_path / "skills" / "code-review"
    skill_dir.mkdir(parents=True)
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "style-guide.md").write_text("# Style Guide\n\nUse snake_case.")
    return SkillSpec(
        name="code-review",
        description="Reviews code.",
        content="Review the code.",
        skill_dir=skill_dir,
    )


@pytest.fixture()
def skill_no_resources() -> SkillSpec:
    """
    A skill with no ``skill_dir`` (in-memory only).

    :returns: A ``SkillSpec`` with ``skill_dir=None``.
    """
    return SkillSpec(
        name="summarize",
        description="Summarizes text.",
        content="Summarize the input concisely.",
    )


@pytest.fixture(autouse=True)
def _clean_mcp_cache() -> None:
    """
    Clear the MCP discovery cache before each test.
    """
    clear_discovery_cache()


def _make_spec(
    skills: list[SkillSpec] | None = None,
    mcp_servers: list[MCPServerConfig] | None = None,
    local_tools: list[LocalToolInfo] | None = None,
) -> AgentSpec:
    """
    Create a minimal ``AgentSpec`` with the given skills,
    MCP servers, and local tools.

    :param skills: Skills to include, or ``None`` for no
        skills.
    :param mcp_servers: MCP server configs, or ``None`` for
        no MCP servers.
    :param local_tools: Local tool infos, or ``None`` for
        no local tools.
    :returns: An ``AgentSpec`` with ``spec_version=1``.
    """
    return AgentSpec(
        spec_version=1,
        skills=skills or [],
        mcp_servers=mcp_servers or [],
        local_tools=local_tools or [],
    )


# ── Registry dispatch ─────────────────────────────────


def test_registry_dispatches_to_load_skill(
    skill_no_resources: SkillSpec,
) -> None:
    """
    ToolManager.call_tool dispatches to LoadSkillTool via
    the registry.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
    )
    result = mgr.call_tool(
        "load_skill",
        json.dumps({"name": "summarize"}),
        _TEST_CTX,
    )
    assert result == "Summarize the input concisely."


def test_registry_dispatches_to_read_skill_file(
    skill_with_resources: SkillSpec,
) -> None:
    """
    ToolManager.call_tool dispatches to ReadSkillFileTool
    via the registry.
    """
    mgr = ToolManager(
        _make_spec([skill_with_resources]),
    )
    result = mgr.call_tool(
        "read_skill_file",
        json.dumps(
            {
                "skill_name": "code-review",
                "path": "references/style-guide.md",
            }
        ),
        _TEST_CTX,
    )
    assert "# Style Guide" in result


def test_registry_unknown_tool_returns_error(
    skill_no_resources: SkillSpec,
) -> None:
    """
    ToolManager.call_tool returns error for unregistered tools.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
    )
    result = mgr.call_tool("nonexistent", json.dumps({}), _TEST_CTX)
    assert "not found" in result
    assert "load_skill" in result


# ── get_tool_schemas ──────────────────────────────────


def test_schemas_include_load_skill_when_skills_exist(
    skill_no_resources: SkillSpec,
) -> None:
    """
    get_tool_schemas includes load_skill when the agent has
    skills.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
    )
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "load_skill" in names


def test_schemas_include_read_skill_file_with_resources(
    skill_with_resources: SkillSpec,
) -> None:
    """
    get_tool_schemas includes read_skill_file when a skill
    has bundled resource files.
    """
    mgr = ToolManager(
        _make_spec([skill_with_resources]),
    )
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "read_skill_file" in names


def test_schemas_exclude_read_skill_file_without_resources(
    skill_no_resources: SkillSpec,
) -> None:
    """
    get_tool_schemas does NOT include read_skill_file when
    no skill has bundled resources.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
    )
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "read_skill_file" not in names


def test_schemas_empty_when_no_skills() -> None:
    """
    get_tool_schemas returns empty when agent has no skills.
    """
    mgr = ToolManager(_make_spec([]))
    assert mgr.get_tool_schemas() == []


# ── MCP integration ──────────────────────────────────────


def _make_mock_mcp_tool(
    name: str = "mcp_tool",
    description: str = "An MCP tool.",
) -> MagicMock:
    """
    Create a mock MCP tool definition.

    :param name: Tool name.
    :param description: Tool description.
    :returns: A mock matching ``mcp.types.Tool`` attribute API.
    """
    tool_def = MagicMock()
    tool_def.name = name
    tool_def.description = description
    tool_def.inputSchema = {
        "type": "object",
        "properties": {},
    }
    return tool_def


def _patch_mcp_connect(
    tools: list[MagicMock],
) -> patch:
    """
    Patch ``McpServerConnection.connect`` to return mock tools
    without opening a real transport.

    :param tools: List of mock MCP tool definitions to return.
    :returns: A ``patch`` context manager.
    """
    return patch(
        "agent_plane.tools.manager.McpServerConnection.connect",
        new_callable=AsyncMock,
        return_value=tools,
    )


def test_start_discovers_mcp_tools() -> None:
    """
    ``start()`` discovers MCP tools and registers them so they
    appear in ``get_tool_schemas()``.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec)

    tool_def = _make_mock_mcp_tool("github_search")

    with _patch_mcp_connect([tool_def]):
        mgr.start()

    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "github_search" in names

    mgr.shutdown()


def test_start_mcp_tools_callable() -> None:
    """
    MCP tools registered by ``start()`` can be dispatched via
    ``call_tool()``.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec)

    tool_def = _make_mock_mcp_tool("do_thing")

    with _patch_mcp_connect([tool_def]):
        mgr.start()

    # McpTool.invoke delegates to its _run_sync callable,
    # which is EventLoopThread.run. Patch the registered
    # tool's _run_sync to return a canned result without
    # hitting the real event loop.
    mcp_tool = mgr._tools["do_thing"]
    mcp_tool._run_sync = MagicMock(return_value="tool result")

    result = mgr.call_tool(
        "do_thing",
        json.dumps({"param": "value"}),
        _TEST_CTX,
    )

    assert result == "tool result"

    mgr.shutdown()


def test_start_mcp_failure_does_not_block_other_tools(
    skill_no_resources: SkillSpec,
) -> None:
    """
    If an MCP server fails to connect, skill tools still work.
    """
    mcp_config = MCPServerConfig(
        name="broken-mcp",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(
        skills=[skill_no_resources],
        mcp_servers=[mcp_config],
    )
    mgr = ToolManager(spec)

    with patch(
        "agent_plane.tools.manager.McpServerConnection.connect",
        new_callable=AsyncMock,
        side_effect=ConnectionError("failed"),
    ):
        # Should not raise — connection failure is logged.
        mgr.start()

    # Skill tools still work.
    result = mgr.call_tool(
        "load_skill",
        json.dumps({"name": "summarize"}),
        _TEST_CTX,
    )
    assert "Summarize" in result

    mgr.shutdown()


def test_shutdown_safe_without_start() -> None:
    """
    ``shutdown()`` is safe to call without ``start()``.
    """
    spec = _make_spec()
    mgr = ToolManager(spec)
    mgr.shutdown()


# ── Client-specified tools ────────────────────────────────


def _make_client_side_spec(name: str) -> ClientSideToolSpec:
    """
    Build a minimal :class:`ClientSideToolSpec` for use in manager tests.

    :param name: Tool function name, e.g. ``"get_weather"``.
    :returns: A :class:`ClientSideToolSpec` with a minimal schema.
    """
    return ClientSideToolSpec(
        name=name,
        schema={
            "type": "function",
            "function": {"name": name, "description": "A test tool.", "parameters": {}},
        },
    )


def test_client_tools_registered_in_schemas() -> None:
    """
    Client-specified tools appear in get_tool_schemas() alongside
    built-in tools without calling start().

    A failure here means the LLM never sees client tools — the
    client_tool_specs constructor arg is not being wired up.
    """
    spec = _make_spec()
    mgr = ToolManager(
        spec,
        client_tool_specs=[
            _make_client_side_spec("get_weather"),
            _make_client_side_spec("send_email"),
        ],
    )

    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]

    # Both client tools appear in schemas — 2 registered, 2 returned
    assert len(schemas) == 2, (
        f"Expected 2 schemas (2 client tools), got {len(schemas)}. "
        "If 0, client_tool_specs are not being registered."
    )
    assert "get_weather" in names
    assert "send_email" in names


def test_is_client_side_tool_returns_true_for_registered_client_tools() -> None:
    """
    is_client_side_tool returns True for registered ClientSideTool
    entries and False for built-in tools and unknown names.

    The agent loop uses this to detect when to complete the response
    instead of executing tools server-side. A failure here would
    cause client-side tools to be dispatched through call_tool,
    triggering RuntimeError from ClientSideTool.invoke.
    """
    spec = _make_spec(skills=[SkillSpec(name="summarize", description=".", content=".")])
    mgr = ToolManager(
        spec,
        client_tool_specs=[
            _make_client_side_spec("get_weather"),
            _make_client_side_spec("send_email"),
        ],
    )

    # Client tools are detected as client-side
    assert mgr.is_client_side_tool("get_weather") is True, (
        "Expected True for registered ClientSideTool 'get_weather'. "
        "If False, is_client_side_tool is not checking isinstance(tool, ClientSideTool)."
    )
    assert mgr.is_client_side_tool("send_email") is True

    # Built-in tool is not client-side
    assert mgr.is_client_side_tool("load_skill") is False, (
        "Expected False for built-in 'load_skill'. "
        "If True, is_client_side_tool is not type-checking correctly."
    )

    # Unregistered tool is not client-side
    assert mgr.is_client_side_tool("nonexistent") is False


def test_client_tool_shadows_skill_tool(
    skill_no_resources: SkillSpec,
) -> None:
    """
    A client tool with the same name as a skill tool overwrites the
    skill tool (last registered wins, with a warning).

    This ensures the override behavior is intentional — clients can
    replace spec-defined tools at request time.
    """
    spec = _make_spec(skills=[skill_no_resources])
    mgr = ToolManager(
        spec,
        # 'load_skill' is the built-in skill tool name
        client_tool_specs=[_make_client_side_spec("load_skill")],
    )

    schemas = mgr.get_tool_schemas()
    # Only one 'load_skill' — client version overwrote built-in
    names = [s["function"]["name"] for s in schemas]
    assert names.count("load_skill") == 1, (
        f"Expected exactly one 'load_skill' (client overwrite), got {names.count('load_skill')}."
    )

    # The registered tool is the client's ClientSideTool, not LoadSkillTool
    from agent_plane.tools.client_specified import ClientSideTool

    assert isinstance(mgr._tools["load_skill"], ClientSideTool), (
        "Expected ClientSideTool after client override, "
        f"got {type(mgr._tools['load_skill']).__name__}."
    )


def test_client_tools_none_equivalent_to_empty() -> None:
    """
    Passing client_tool_specs=None and client_tool_specs=[] produce
    the same result: no client tools registered.
    """
    spec = _make_spec()
    mgr_none = ToolManager(spec, client_tool_specs=None)
    mgr_empty = ToolManager(spec, client_tool_specs=[])

    assert mgr_none.get_tool_schemas() == []
    assert mgr_empty.get_tool_schemas() == []


def test_mcp_duplicate_tool_name_last_wins() -> None:
    """
    When two MCP servers expose a tool with the same name, the
    last server's tool wins (with a warning log).
    """
    config_a = MCPServerConfig(
        name="server-a",
        url="http://localhost:9000/mcp",
    )
    config_b = MCPServerConfig(
        name="server-b",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(mcp_servers=[config_a, config_b])
    mgr = ToolManager(spec)

    tool_a = _make_mock_mcp_tool("shared_tool")
    tool_b = _make_mock_mcp_tool("shared_tool")

    call_count = 0

    async def fake_connect(self: MagicMock) -> list[MagicMock]:
        """
        Return different tool mocks for each server.

        :param self: The McpServerConnection instance
            (injected by the patch).
        :returns: A list with one mock tool definition.
        """
        nonlocal call_count
        call_count += 1
        # First call returns tool_a, second returns tool_b.
        return [tool_a] if call_count == 1 else [tool_b]

    with patch(
        "agent_plane.tools.manager.McpServerConnection.connect",
        new=fake_connect,
    ):
        mgr.start()

    # Both registered under same name — last wins.
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names.count("shared_tool") == 1

    mgr.shutdown()


# ── Tool name validation ─────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "tool with spaces",
        "tool:colon",
        "tool.dot",
        "",
        "a" * 65,  # exceeds 64-char limit
        "tool/slash",
        "ns::tool",
    ],
    ids=[
        "spaces",
        "colon",
        "dot",
        "empty",
        "too_long",
        "slash",
        "double_colon",
    ],
)
def test_mcp_tool_invalid_name_skipped(
    name: str,
) -> None:
    """
    MCP tools with names violating the OpenAI constraint
    (``[a-zA-Z0-9_-]{1,64}``) are skipped at registration
    and do not appear in ``get_tool_schemas()``.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec)

    tool_def = _make_mock_mcp_tool(name)

    with _patch_mcp_connect([tool_def]):
        mgr.start()

    assert mgr.get_tool_schemas() == []

    mgr.shutdown()


def test_mcp_tool_valid_names_registered() -> None:
    """
    MCP tools with valid names (alphanumeric, underscore,
    hyphen, up to 64 chars) are registered normally.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec)

    tools = [
        _make_mock_mcp_tool("simple"),
        _make_mock_mcp_tool("with_underscore"),
        _make_mock_mcp_tool("with-hyphen"),
        _make_mock_mcp_tool("MixedCase123"),
    ]

    with _patch_mcp_connect(tools):
        mgr.start()

    schemas = mgr.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"simple", "with_underscore", "with-hyphen", "MixedCase123"}

    mgr.shutdown()


def test_mcp_tool_mixed_valid_and_invalid() -> None:
    """
    When an MCP server returns a mix of valid and invalid tool
    names, only valid tools are registered.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        url="http://localhost:9000/mcp",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec)

    tools = [
        _make_mock_mcp_tool("valid_tool"),
        _make_mock_mcp_tool("invalid tool"),  # space
        _make_mock_mcp_tool("also_valid"),
    ]

    with _patch_mcp_connect(tools):
        mgr.start()

    schemas = mgr.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"valid_tool", "also_valid"}

    mgr.shutdown()


@pytest.mark.parametrize(
    "name",
    [
        "tool with spaces",
        "tool:colon",
        "",
        "a" * 65,
    ],
    ids=[
        "spaces",
        "colon",
        "empty",
        "too_long",
    ],
)
def test_client_tool_invalid_name_raises(
    name: str,
) -> None:
    """
    Client-specified tools with invalid names raise
    ``AgentPlaneError`` at registration time.
    """
    spec = _make_spec()
    with pytest.raises(AgentPlaneError, match="Invalid client tool name"):
        ToolManager(
            spec,
            client_tool_specs=[_make_client_side_spec(name)],
        )


# ── Local tool registration ──────────────────────────


def _write_local_tool(
    workdir: Path,
    filename: str,
    schema_name: str,
) -> None:
    """
    Write a minimal local Python tool file to
    ``workdir/tools/python/<filename>``.

    :param workdir: Agent image root directory.
    :param filename: File name, e.g. ``"web_fetch.py"``.
    :param schema_name: The ``name`` field in SCHEMA.
    """
    py_dir = workdir / "tools" / "python"
    py_dir.mkdir(parents=True, exist_ok=True)
    code = f'''
"""Test tool."""
from typing import Any
SCHEMA: dict[str, Any] = {{
    "type": "function",
    "function": {{
        "name": "{schema_name}",
        "description": "A test tool.",
        "parameters": {{"type": "object", "properties": {{}}}},
    }},
}}
async def run(arguments: dict[str, Any]) -> str:
    """Execute."""
    return "local_tool_result"
'''
    (py_dir / filename).write_text(code)


def test_local_tools_registered_and_callable(
    tmp_path: Path,
) -> None:
    """
    ToolManager registers local Python tools from the workdir
    and dispatches calls to them.
    """
    _write_local_tool(tmp_path, "echo_tool.py", "echo_tool")
    info = LocalToolInfo(
        name="echo_tool",
        path="tools/python/echo_tool.py",
        language="python",
    )
    spec = _make_spec(local_tools=[info])
    mgr = ToolManager(spec, workdir=tmp_path)
    schemas = mgr.get_tool_schemas()
    # Local tool appears in the schema list.
    names = [s["function"]["name"] for s in schemas]
    assert "echo_tool" in names, (
        f"Expected 'echo_tool' in schemas, got {names}. "
        f"If missing, _register_local_tools did not register the tool."
    )
    # Dispatching works through call_tool.
    result = mgr.call_tool("echo_tool", json.dumps({}), _TEST_CTX)
    assert result == "local_tool_result", (
        f"Expected 'local_tool_result', got {result!r}. "
        f"If 'Error: tool not found', the tool was not registered."
    )


def test_local_tools_skipped_without_workdir() -> None:
    """
    ToolManager with workdir=None skips local tool registration
    without error, even if spec has local_tools.
    """
    info = LocalToolInfo(
        name="some_tool",
        path="tools/python/some_tool.py",
        language="python",
    )
    spec = _make_spec(local_tools=[info])
    # workdir=None (default) — should not raise.
    mgr = ToolManager(spec)
    assert mgr.get_tool_schemas() == [], "No tools should be registered when workdir is None."
