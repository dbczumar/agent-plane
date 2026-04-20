"""Tests for the ``introspect`` built-in tool."""

from __future__ import annotations

from agent_plane.spec.types import (
    AgentSpec,
    BuiltinToolConfig,
    ExecutorSpec,
    InteractionConfig,
    LLMConfig,
    LocalToolInfo,
    ModalityConfig,
    SkillSpec,
    ToolsConfig,
)
from agent_plane.tools.builtins.introspect import IntrospectTool


def _make_spec() -> AgentSpec:
    """
    Build a rich AgentSpec for testing all introspect sections.

    :returns: An AgentSpec with name, model, tools, skills, sub-agents.
    """
    child = AgentSpec(
        spec_version=1,
        name="researcher",
        description="Searches the web for info.",
        llm=LLMConfig(model="openai/gpt-5.4"),
        instructions="You are a researcher. Search the web.",
        skills=[
            SkillSpec(
                name="deep-research",
                description="In-depth research skill.",
                content="When researching:\n1. Search broadly\n2. Cross-reference",
            ),
        ],
    )
    return AgentSpec(
        spec_version=1,
        name="archer",
        description="A research assistant.",
        llm=LLMConfig(model="openai/gpt-5.4"),
        executor=ExecutorSpec(type="llm"),
        interaction=InteractionConfig(
            modalities=ModalityConfig(input=["text", "image"], output=["text"]),
        ),
        tools=ToolsConfig(
            builtins=[
                BuiltinToolConfig(name="web_search"),
                BuiltinToolConfig(name="terminal_run"),
            ],
            agents=["researcher"],
        ),
        instructions="You are archer. Investigate topics.",
        skills=[
            SkillSpec(
                name="explain",
                description="Explain concepts clearly.",
                content="When explaining:\n1. Use simple language\n2. Give examples",
            ),
        ],
        local_tools=[
            LocalToolInfo(name="word_count", language="python", path="tools/python/word_count.py"),
        ],
        sub_agents=[child],
    )


# ── Summary (no section) ────────────────────────────


def test_summary_contains_agent_name() -> None:
    """Summary must include the agent name."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "archer" in result


def test_summary_contains_model() -> None:
    """Summary must include the LLM model."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "openai/gpt-5.4" in result


def test_summary_lists_builtins() -> None:
    """Summary must list builtin tool names."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "web_search" in result
    assert "terminal_run" in result


def test_summary_lists_skills() -> None:
    """Summary must list skill names."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "explain" in result


def test_summary_lists_sub_agents() -> None:
    """Summary must list sub-agent names."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "researcher" in result


def test_summary_lists_local_tools() -> None:
    """Summary must list local tool names."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "word_count" in result


def test_summary_shows_drill_hint() -> None:
    """Summary must hint at section drilling."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke("{}", None)
    assert "introspect(section=" in result


# ── Instructions section ─────────────────────────────


def test_instructions_returns_content() -> None:
    """instructions section returns the agent's AGENTS.md content."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "instructions"}', None)
    assert "You are archer" in result


# ── Skills section ───────────────────────────────────


def test_skills_lists_all() -> None:
    """skills section lists all skills with descriptions."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "skills"}', None)
    assert "explain" in result
    assert "Explain concepts" in result


def test_skills_drill_returns_content() -> None:
    """skills/<name> returns that skill's full content."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "skills/explain"}', None)
    # Skill content includes the markdown body.
    assert "Use simple language" in result


def test_skills_unknown_returns_error() -> None:
    """skills/<unknown> returns error with available names."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "skills/nonexistent"}', None)
    assert "not found" in result.lower()
    assert "explain" in result


# ── Tools section ────────────────────────────────────


def test_tools_section_lists_builtins() -> None:
    """tools section lists builtin tools."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "tools"}', None)
    assert "web_search" in result
    assert "terminal_run" in result


def test_tools_section_lists_local() -> None:
    """tools section lists local tools."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "tools"}', None)
    assert "word_count" in result


# ── Sub-agents section ───────────────────────────────


def test_sub_agents_lists_all() -> None:
    """sub_agents section lists all sub-agents."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "sub_agents"}', None)
    assert "researcher" in result


def test_sub_agents_drill_returns_summary() -> None:
    """sub_agents/<name> returns that sub-agent's summary."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "sub_agents/researcher"}', None)
    # Should be a summary of the researcher sub-agent.
    assert "researcher" in result
    assert "openai/gpt-5.4" in result


def test_sub_agents_drill_instructions() -> None:
    """sub_agents/<name>/instructions returns the sub-agent's instructions."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "sub_agents/researcher/instructions"}', None)
    assert "You are a researcher" in result


def test_sub_agents_drill_skills() -> None:
    """sub_agents/<name>/skills lists the sub-agent's skills."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "sub_agents/researcher/skills"}', None)
    assert "deep-research" in result


def test_sub_agents_drill_skill_content() -> None:
    """sub_agents/<name>/skills/<skill> returns skill content."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "sub_agents/researcher/skills/deep-research"}', None)
    assert "Search broadly" in result


def test_sub_agents_unknown_returns_error() -> None:
    """sub_agents/<unknown> returns error with available names."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "sub_agents/nonexistent"}', None)
    assert "not found" in result.lower()
    assert "researcher" in result


# ── Config section ───────────────────────────────────


def test_config_section_returns_spec_fields() -> None:
    """config section returns key spec fields."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "config"}', None)
    assert "spec_version: 1" in result
    assert "archer" in result


# ── Unknown section ──────────────────────────────────


def test_unknown_section_returns_error() -> None:
    """Unknown root section returns error with valid options."""
    tool = IntrospectTool(spec=_make_spec())
    result = tool.invoke('{"section": "bogus"}', None)
    assert "Unknown section" in result
    assert "instructions" in result
    assert "skills" in result


# ── Tool name ────────────────────────────────────────


def test_tool_name() -> None:
    """Tool name is 'introspect'."""
    assert IntrospectTool.name() == "introspect"
