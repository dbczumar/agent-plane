"""Tests for the ``web_fetch`` built-in tool."""

from __future__ import annotations

import pytest

from agent_plane.spec.types import (
    AgentSpec,
    ExecutorSpec,
    LLMConfig,
)
from agent_plane.tools.builtins.web_fetch import (
    RESEARCHER_NAME,
    WebFetchTool,
    build_researcher_spec,
)

# ── Helpers ──────────────────────────────────────────


def _make_parent_spec(
    model: str = "openai/gpt-5.4",
    executor_type: str | None = None,
) -> AgentSpec:
    """
    Build a minimal parent AgentSpec for testing.

    :param model: The LLM model string.
    :param executor_type: Executor type override, or ``None``
        for default (llm).
    :returns: An AgentSpec suitable for constructing WebFetchTool.
    """
    executor = ExecutorSpec()
    if executor_type is not None:
        executor = ExecutorSpec(type=executor_type)
    return AgentSpec(
        spec_version=1,
        name="test-parent",
        llm=LLMConfig(model=model),
        executor=executor,
    )


# ── Schema ───────────────────────────────────────────


def test_web_fetch_schema_is_function() -> None:
    """Schema is a standard function schema with query + url params."""
    parent = _make_parent_spec()
    tool = WebFetchTool(parent_spec=parent)
    schema = tool.get_schema()
    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "web_fetch"
    # query is required, url is optional.
    assert "query" in func["parameters"]["required"]
    assert "url" in func["parameters"]["properties"]
    assert "url" not in func["parameters"]["required"]


def test_web_fetch_name() -> None:
    """Tool name is 'web_fetch'."""
    assert WebFetchTool.name() == "web_fetch"


# ── Researcher spec ──────────────────────────────────


def test_researcher_inherits_parent_model() -> None:
    """
    The __web_researcher sub-agent must use the parent's LLM config.
    If it used a different model, the web_fetch tool would fail for
    agents using non-default providers (e.g. anthropic).
    """
    parent = _make_parent_spec(model="anthropic/claude-sonnet-4-20250514")
    tool = WebFetchTool(parent_spec=parent)
    researcher = tool.researcher_spec
    assert researcher.llm is not None, (
        "Researcher spec must have an llm block — "
        "without it, the workflow fails with 'no LLM configuration'."
    )
    assert researcher.llm.model == "anthropic/claude-sonnet-4-20250514", (
        f"Researcher should inherit parent model, got {researcher.llm.model!r}."
    )


def test_researcher_has_code_sandbox() -> None:
    """
    The researcher must have code_sandbox as its tool — without it,
    the sub-agent can't execute scripts to fetch web content.
    """
    parent = _make_parent_spec()
    tool = WebFetchTool(parent_spec=parent)
    researcher = tool.researcher_spec
    builtin_names = [b.name for b in researcher.tools.builtins]
    assert "code_sandbox" in builtin_names, (
        f"Researcher must have code_sandbox, got {builtin_names}."
    )


def test_researcher_name_is_internal() -> None:
    """
    The researcher name must use __ prefix to avoid collision
    with user-declared sub-agent names.
    """
    parent = _make_parent_spec()
    tool = WebFetchTool(parent_spec=parent)
    assert tool.researcher_spec.name == RESEARCHER_NAME
    assert RESEARCHER_NAME.startswith("__"), (
        f"Internal sub-agent name should start with __, got {RESEARCHER_NAME!r}."
    )


def test_researcher_appended_to_parent_sub_agents() -> None:
    """
    After construction, the researcher spec must be in the parent's
    sub_agents list so _resolve_agent_spec_for_task can find it.
    """
    parent = _make_parent_spec()
    # sub_agents starts empty.
    assert len(parent.sub_agents) == 0
    WebFetchTool(parent_spec=parent)
    # Now it should have the researcher.
    names = [s.name for s in parent.sub_agents]
    assert RESEARCHER_NAME in names, f"Researcher should be in parent's sub_agents, got {names}."


def test_researcher_not_conversational() -> None:
    """
    The researcher should be non-conversational (one-shot task).
    """
    parent = _make_parent_spec()
    tool = WebFetchTool(parent_spec=parent)
    assert tool.researcher_spec.interaction.conversational is False


def test_researcher_has_instructions() -> None:
    """
    The researcher must have non-empty instructions that mention
    web research.
    """
    parent = _make_parent_spec()
    tool = WebFetchTool(parent_spec=parent)
    instructions = tool.researcher_spec.instructions
    assert instructions is not None
    # 100 chars minimum ensures non-trivial instructions. If shorter,
    # the researcher won't have enough guidance to know how to search
    # the web and extract content.
    assert len(instructions) > 100, (
        f"Researcher instructions too short ({len(instructions)} chars). "
        f"If < 100, the sub-agent won't have enough context to perform "
        f"web research effectively."
    )
    assert "web" in instructions.lower()


# ── Executor guard ───────────────────────────────────


def test_non_llm_executor_returns_error() -> None:
    """
    web_fetch must return an error for non-llm executors since
    they don't support sub-agents.
    """
    from agent_plane.tools.base import ToolContext

    parent = _make_parent_spec(executor_type="agents_sdk")
    tool = WebFetchTool(parent_spec=parent)
    ctx = ToolContext(task_id="t1", agent_id="a1")
    result = tool.invoke('{"query": "test"}', ctx)
    assert "not available" in result.lower(), (
        f"Expected error about executor compatibility, got: {result}"
    )
    assert "agents_sdk" in result


def test_llm_executor_does_not_error_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    web_fetch with default llm executor should NOT trigger the
    executor guard. It will raise RuntimeError (no runtime
    initialized) when it tries to spawn — but that's past the
    guard, proving it wasn't blocked.
    """
    from agent_plane.runtime import _globals
    from agent_plane.tools.base import ToolContext

    # Other tests (e.g. tests/stores/test_task_store.py) initialize
    # the global task_store and leave it set, which causes this
    # test's `runtime not initialized` assertion to fail when run
    # after them. Force the globals to None for this test so the
    # uninitialized-runtime branch is reachable regardless of order.
    monkeypatch.setattr(_globals, "_task_store", None)

    parent = _make_parent_spec(executor_type="llm")
    tool = WebFetchTool(parent_spec=parent)
    ctx = ToolContext(task_id="t1", agent_id="a1")
    # RuntimeError from task_store access proves we got past
    # the executor guard (which returns a string, not raises).
    with pytest.raises(RuntimeError, match="runtime not initialized"):
        tool.invoke('{"query": "test"}', ctx)


def test_missing_query_returns_error() -> None:
    """Tool returns clear error when query is missing."""
    from agent_plane.tools.base import ToolContext

    parent = _make_parent_spec()
    tool = WebFetchTool(parent_spec=parent)
    ctx = ToolContext(task_id="t1", agent_id="a1")
    result = tool.invoke("{}", ctx)
    assert "query" in result.lower()


# ── build_researcher_spec standalone ────────────────


def testbuild_researcher_spec_copies_llm() -> None:
    """
    build_researcher_spec must copy the parent's LLM config
    exactly — same model string, same object reference for
    connection details.
    """
    llm = LLMConfig(
        model="groq/llama-4-scout",
        connection={"api_key": "test-key"},
    )
    parent = AgentSpec(spec_version=1, llm=llm)
    researcher = build_researcher_spec(parent)
    # Same LLM config object (reference copy, not deep copy —
    # the researcher doesn't modify it).
    assert researcher.llm is parent.llm
    assert researcher.llm.model == "groq/llama-4-scout"


def testbuild_researcher_spec_default_executor() -> None:
    """Researcher should use default executor (llm)."""
    parent = _make_parent_spec()
    researcher = build_researcher_spec(parent)
    assert researcher.executor.type == "llm"
