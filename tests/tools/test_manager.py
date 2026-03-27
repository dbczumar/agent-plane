"""Tests for agent_plane.tools.manager (ToolManager)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_plane.spec.types import (
    AgentSpec,
    MCPServerConfig,
    SkillSpec,
)
from agent_plane.tools import ToolManager
from agent_plane.tools.mcp import clear_discovery_cache


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """
    Temporary working directory for the ToolManager.

    :returns: A ``Path`` to a temp directory.
    """
    return tmp_path / "work"


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
) -> AgentSpec:
    """
    Create a minimal ``AgentSpec`` with the given skills and
    MCP servers.

    :param skills: Skills to include, or ``None`` for no
        skills.
    :param mcp_servers: MCP server configs, or ``None`` for
        no MCP servers.
    :returns: An ``AgentSpec`` with ``spec_version=1``.
    """
    return AgentSpec(
        spec_version=1,
        skills=skills or [],
        mcp_servers=mcp_servers or [],
    )


# ── Registry dispatch ─────────────────────────────────


def test_registry_dispatches_to_load_skill(
    work_dir: Path,
    skill_no_resources: SkillSpec,
) -> None:
    """
    ToolManager.call_tool dispatches to LoadSkillTool via
    the registry.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
        work_dir,
    )
    result = mgr.call_tool(
        "load_skill",
        json.dumps({"name": "summarize"}),
    )
    assert result == "Summarize the input concisely."


def test_registry_dispatches_to_read_skill_file(
    work_dir: Path,
    skill_with_resources: SkillSpec,
) -> None:
    """
    ToolManager.call_tool dispatches to ReadSkillFileTool
    via the registry.
    """
    mgr = ToolManager(
        _make_spec([skill_with_resources]),
        work_dir,
    )
    result = mgr.call_tool(
        "read_skill_file",
        json.dumps(
            {
                "skill_name": "code-review",
                "path": "references/style-guide.md",
            }
        ),
    )
    assert "# Style Guide" in result


def test_registry_unknown_tool_returns_error(
    work_dir: Path,
    skill_no_resources: SkillSpec,
) -> None:
    """
    ToolManager.call_tool returns error for unregistered tools.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
        work_dir,
    )
    result = mgr.call_tool("nonexistent", json.dumps({}))
    assert "not found" in result
    assert "load_skill" in result


# ── get_tool_schemas ──────────────────────────────────


def test_schemas_include_load_skill_when_skills_exist(
    work_dir: Path,
    skill_no_resources: SkillSpec,
) -> None:
    """
    get_tool_schemas includes load_skill when the agent has
    skills.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
        work_dir,
    )
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "load_skill" in names


def test_schemas_include_read_skill_file_with_resources(
    work_dir: Path,
    skill_with_resources: SkillSpec,
) -> None:
    """
    get_tool_schemas includes read_skill_file when a skill
    has bundled resource files.
    """
    mgr = ToolManager(
        _make_spec([skill_with_resources]),
        work_dir,
    )
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "read_skill_file" in names


def test_schemas_exclude_read_skill_file_without_resources(
    work_dir: Path,
    skill_no_resources: SkillSpec,
) -> None:
    """
    get_tool_schemas does NOT include read_skill_file when
    no skill has bundled resources.
    """
    mgr = ToolManager(
        _make_spec([skill_no_resources]),
        work_dir,
    )
    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "read_skill_file" not in names


def test_schemas_empty_when_no_skills(
    work_dir: Path,
) -> None:
    """
    get_tool_schemas returns empty when agent has no skills.
    """
    mgr = ToolManager(_make_spec([]), work_dir)
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


def test_start_discovers_mcp_tools(
    work_dir: Path,
) -> None:
    """
    ``start()`` discovers MCP tools and registers them so they
    appear in ``get_tool_schemas()``.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        transport="stdio",
        command="echo",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec, work_dir)

    tool_def = _make_mock_mcp_tool("github_search")

    with _patch_mcp_connect([tool_def]):
        mgr.start()

    schemas = mgr.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "github_search" in names

    mgr.shutdown()


def test_start_mcp_tools_callable(
    work_dir: Path,
) -> None:
    """
    MCP tools registered by ``start()`` can be dispatched via
    ``call_tool()``.
    """
    mcp_config = MCPServerConfig(
        name="test-mcp",
        transport="stdio",
        command="echo",
    )
    spec = _make_spec(mcp_servers=[mcp_config])
    mgr = ToolManager(spec, work_dir)

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
    )

    assert result == "tool result"

    mgr.shutdown()


def test_start_mcp_failure_does_not_block_other_tools(
    work_dir: Path,
    skill_no_resources: SkillSpec,
) -> None:
    """
    If an MCP server fails to connect, skill tools still work.
    """
    mcp_config = MCPServerConfig(
        name="broken-mcp",
        transport="stdio",
        command="nonexistent",
    )
    spec = _make_spec(
        skills=[skill_no_resources],
        mcp_servers=[mcp_config],
    )
    mgr = ToolManager(spec, work_dir)

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
    )
    assert "Summarize" in result

    mgr.shutdown()


def test_shutdown_safe_without_start(
    work_dir: Path,
) -> None:
    """
    ``shutdown()`` is safe to call without ``start()``.
    """
    spec = _make_spec()
    mgr = ToolManager(spec, work_dir)
    mgr.shutdown()


def test_mcp_duplicate_tool_name_last_wins(
    work_dir: Path,
) -> None:
    """
    When two MCP servers expose a tool with the same name, the
    last server's tool wins (with a warning log).
    """
    config_a = MCPServerConfig(
        name="server-a",
        transport="stdio",
        command="echo",
    )
    config_b = MCPServerConfig(
        name="server-b",
        transport="stdio",
        command="echo",
    )
    spec = _make_spec(mcp_servers=[config_a, config_b])
    mgr = ToolManager(spec, work_dir)

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
