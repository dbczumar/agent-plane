"""
Workflow-layer enforcement helper tests.

The engine-level behavior of all three decisions (ALLOW / ASK
/ DENY) is covered extensively in
:mod:`tests.runtime.policies.test_engine_skeleton` and
siblings. These tests cover the thin Phase 6 wrappers in
:mod:`agent_plane.runtime.workflow` that adapt the engine's
:class:`PolicyResult` to the workflow's sentinel / result
conventions:

- :func:`_enforce_tool_call_policy` — ``None`` on ALLOW, a
  ``[Denied by policy: ...]`` sentinel string on DENY, and
  the same sentinel on refused ASK.
- :func:`_enforce_tool_result_policy` — returns the original
  tool result on ALLOW, sentinel on DENY / refused ASK.
- :func:`_enforce_output_policy` — returns the original
  assistant text on ALLOW, sentinel on DENY / refused ASK.

The ASK path is exercised with a stub :func:`_handle_policy_ask`
to avoid wiring a real DBOS + task_store stack — the stub
returns True / False directly. This keeps the tests focused
on the wrapper logic rather than re-testing the ASK seam
(that is covered in :mod:`tests.runtime.policies.test_approval`).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.policies.label import LabelPolicy
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.workflow import (
    _build_deny_sentinel,
    _content_preview,
    _enforce_output_policy,
    _enforce_tool_call_policy,
    _enforce_tool_result_policy,
)
from agent_plane.spec.types import (
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _make_engine_with_policy(
    spec: LabelPolicySpec,
    store: SqlAlchemyConversationStore,
) -> PolicyEngine:
    """
    Build a minimal PolicyEngine around a single LabelPolicy.

    :param spec: The LabelPolicySpec driving the decision.
    :param store: Real persistence layer the engine writes
        through on ALLOW label writes.
    :returns: An engine the enforcement helpers can call.
    """
    return PolicyEngine(
        policies=[LabelPolicy(spec)],
        label_defs={},
        ask_timeout=30,
        conversation_id="conv_test",
        initial_labels={},
        conversation_store=store,
    )


def _noop_engine(store: SqlAlchemyConversationStore) -> PolicyEngine:
    """
    Build a zero-policy engine — mirrors what
    :func:`build_policy_engine` produces for agents that declare
    no ``guardrails:`` block. Always returns ALLOW.
    """
    return PolicyEngine(
        policies=[],
        label_defs={},
        ask_timeout=30,
        conversation_id="conv_test",
        initial_labels={},
        conversation_store=store,
    )


def _deny_tool_policy(tool_name: str, phase: Phase) -> LabelPolicySpec:
    """Spec: always-DENY label policy gated on a specific tool."""
    return LabelPolicySpec(
        name=f"deny_{tool_name}_{phase.value}",
        on=[PhaseSelector(phase=phase, tool_name=tool_name)],
        action=PolicyAction.DENY,
        reason=f"no {tool_name} for you",
    )


def _allow_tool_policy(tool_name: str, phase: Phase) -> LabelPolicySpec:
    """Spec: always-ALLOW label policy gated on a specific tool."""
    return LabelPolicySpec(
        name=f"allow_{tool_name}_{phase.value}",
        on=[PhaseSelector(phase=phase, tool_name=tool_name)],
        action=PolicyAction.ALLOW,
    )


def _ask_tool_policy(tool_name: str, phase: Phase) -> LabelPolicySpec:
    """Spec: always-ASK label policy gated on a specific tool."""
    return LabelPolicySpec(
        name=f"ask_{tool_name}_{phase.value}",
        on=[PhaseSelector(phase=phase, tool_name=tool_name)],
        action=PolicyAction.ASK,
        reason="approve?",
    )


# ── _build_deny_sentinel ──────────────────────────────────


def test_deny_sentinel_with_reason_includes_reason() -> None:
    """The sentinel carries the policy's reason inline — e2e
    tests grep for ``[Denied by policy`` across all four sites,
    so the shape must stay stable."""
    assert _build_deny_sentinel("block the banana") == "[Denied by policy: block the banana]"


def test_deny_sentinel_without_reason_uses_bare_form() -> None:
    """When a policy DENYs without a reason, fall back to
    ``[Denied by policy]`` so the string is still
    grep-matchable."""
    assert _build_deny_sentinel(None) == "[Denied by policy]"


def test_deny_sentinel_empty_reason_uses_bare_form() -> None:
    """Empty string reason → same bare form as None. Prevents
    a weird ``[Denied by policy: ]`` rendering on policies
    that emit an empty reason string."""
    assert _build_deny_sentinel("") == "[Denied by policy]"


# ── _content_preview ──────────────────────────────────────


def test_content_preview_passes_strings_through() -> None:
    """INPUT/OUTPUT content is a plain string — it should pass
    through the preview builder unchanged so the approval UI
    shows the real text."""
    assert _content_preview("hello world") == "hello world"


def test_content_preview_json_dumps_dicts() -> None:
    """TOOL_CALL/TOOL_RESULT content is a dict — dump as JSON
    so the approval UI renders a structured preview instead
    of a Python repr."""
    rendered = _content_preview({"tool": "search", "args": {"q": "x"}})
    assert json.loads(rendered) == {"tool": "search", "args": {"q": "x"}}


def test_content_preview_repr_fallback_on_unknown_shape() -> None:
    """Any unexpected content type falls back to repr so the
    classifier / UI sees *something*, not a stringified object
    identity."""

    class _Weird:
        def __repr__(self) -> str:
            return "Weird()"

    assert _content_preview(_Weird()) == "Weird()"


# ── _enforce_tool_call_policy ─────────────────────────────


@pytest.mark.asyncio
async def test_tool_call_allow_returns_none(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ALLOW at TOOL_CALL → helper returns ``None`` so the
    dispatcher proceeds with the tool normally."""
    engine = _make_engine_with_policy(
        _allow_tool_policy("search", Phase.TOOL_CALL),
        conversation_store,
    )
    task_store = MagicMock()
    sentinel = await _enforce_tool_call_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=task_store,
        tool_name="search",
        arguments='{"q": "hello"}',
    )
    assert sentinel is None
    # ALLOW never parks, so the task_store must not be touched.
    assert not task_store.method_calls


@pytest.mark.asyncio
async def test_tool_call_deny_returns_sentinel(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """DENY at TOOL_CALL → helper returns the blocked sentinel
    so the dispatcher skips the call entirely."""
    engine = _make_engine_with_policy(
        _deny_tool_policy("search", Phase.TOOL_CALL),
        conversation_store,
    )
    sentinel = await _enforce_tool_call_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="search",
        arguments='{"q": "hello"}',
    )
    assert sentinel == "[Denied by policy: no search for you]"


@pytest.mark.asyncio
async def test_tool_call_selector_miss_allows_other_tools(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A policy scoped to ``tool_call:search`` must NOT fire on
    a call to ``weather``. This matches POLICIES.md §4 selector
    semantics — the helper must respect them."""
    engine = _make_engine_with_policy(
        _deny_tool_policy("search", Phase.TOOL_CALL),
        conversation_store,
    )
    sentinel = await _enforce_tool_call_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="weather",
        arguments='{"city": "Tokyo"}',
    )
    assert sentinel is None


@pytest.mark.asyncio
async def test_tool_call_malformed_arguments_do_not_crash(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """The LLM can emit malformed JSON as ``arguments``. The
    helper must not crash — the policy still gets to gate on
    the tool name, and malformed content falls back to the
    raw string so a content-inspecting policy sees *something*."""
    engine = _make_engine_with_policy(
        _deny_tool_policy("search", Phase.TOOL_CALL),
        conversation_store,
    )
    sentinel = await _enforce_tool_call_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="search",
        arguments="{not json",
    )
    # Denied because selector matches; the broken JSON didn't
    # prevent the policy from firing.
    assert sentinel is not None
    assert "search" in sentinel


@pytest.mark.asyncio
async def test_tool_call_ask_approved_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ASK at TOOL_CALL → if the user approves, the helper
    returns ``None`` so the dispatcher proceeds. This exercises
    the ASK → approve → ALLOW path the integration guide
    documents."""
    engine = _make_engine_with_policy(
        _ask_tool_policy("search", Phase.TOOL_CALL),
        conversation_store,
    )

    async def _fake_ask(**kwargs: Any) -> bool:
        """Stub approval — returns True (approved)."""
        return True

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._handle_policy_ask",
        _fake_ask,
    )
    sentinel = await _enforce_tool_call_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="search",
        arguments='{"q": "x"}',
    )
    assert sentinel is None


@pytest.mark.asyncio
async def test_tool_call_ask_refused_returns_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ASK at TOOL_CALL → if the user refuses / timeout /
    cancel, the helper returns the blocked sentinel. This is
    the critical fail-closed path per POLICIES.md §7.2."""
    engine = _make_engine_with_policy(
        _ask_tool_policy("search", Phase.TOOL_CALL),
        conversation_store,
    )

    async def _fake_ask(**kwargs: Any) -> bool:
        """Stub refusal — returns False."""
        return False

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._handle_policy_ask",
        _fake_ask,
    )
    sentinel = await _enforce_tool_call_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="search",
        arguments='{"q": "x"}',
    )
    assert sentinel is not None
    assert "[Denied by policy" in sentinel


# ── _enforce_tool_result_policy ───────────────────────────


@pytest.mark.asyncio
async def test_tool_result_allow_returns_original_text(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ALLOW at TOOL_RESULT → helper returns the raw tool
    output unchanged so the LLM sees what the tool produced."""
    engine = _make_engine_with_policy(
        _allow_tool_policy("search", Phase.TOOL_RESULT),
        conversation_store,
    )
    result = await _enforce_tool_result_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="search",
        result_text="3 matches found",
    )
    assert result == "3 matches found"


@pytest.mark.asyncio
async def test_tool_result_deny_substitutes_sentinel(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """DENY at TOOL_RESULT → the raw tool output is replaced
    with a sentinel BEFORE reaching ``function_call_output``,
    so the LLM cannot see blocked content on the next turn."""
    engine = _make_engine_with_policy(
        _deny_tool_policy("search", Phase.TOOL_RESULT),
        conversation_store,
    )
    result = await _enforce_tool_result_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="search",
        result_text="classified intel: ...",
    )
    assert result == "[Denied by policy: no search for you]"
    # The sentinel MUST NOT contain the raw blocked text —
    # that's the whole point of replacing it pre-persist.
    assert "classified intel" not in result


@pytest.mark.asyncio
async def test_tool_result_selector_miss_allows_other_tools(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A ``tool_result:search`` selector does not match a
    ``weather`` result. Phase 6 wiring must pass through the
    correct ``tool_name`` so selectors evaluate right."""
    engine = _make_engine_with_policy(
        _deny_tool_policy("search", Phase.TOOL_RESULT),
        conversation_store,
    )
    result = await _enforce_tool_result_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        tool_name="weather",
        result_text="sunny",
    )
    assert result == "sunny"


# ── _enforce_output_policy ────────────────────────────────


@pytest.mark.asyncio
async def test_output_allow_returns_original_text(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ALLOW at OUTPUT → helper returns the LLM's final text
    unchanged so the assistant reply lands verbatim in
    conversation_items."""
    spec = LabelPolicySpec(
        name="allow_all",
        on=[PhaseSelector(phase=Phase.OUTPUT, tool_name=None)],
        action=PolicyAction.ALLOW,
    )
    engine = _make_engine_with_policy(spec, conversation_store)
    result = await _enforce_output_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        text="Hello world",
    )
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_output_deny_substitutes_sentinel_pre_persist(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """DENY at OUTPUT → the raw assistant text is replaced
    with a sentinel BEFORE persistence (POLICIES.md §11.4 —
    load-bearing ordering). The caller persists the returned
    string, so this check proves the blocked text never makes
    it to ``conversation_items``."""
    spec = LabelPolicySpec(
        name="deny_output",
        on=[PhaseSelector(phase=Phase.OUTPUT, tool_name=None)],
        action=PolicyAction.DENY,
        reason="output contains secrets",
    )
    engine = _make_engine_with_policy(spec, conversation_store)
    result = await _enforce_output_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        text="SECRET: the answer is 42",
    )
    assert result == "[Denied by policy: output contains secrets]"
    assert "SECRET" not in result
    assert "42" not in result


@pytest.mark.asyncio
async def test_output_noop_engine_always_allows(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """An engine with zero policies (the no-op engine built
    for agents without a ``guardrails:`` block) always returns
    ALLOW. The OUTPUT helper must pass the text through
    unchanged — this is the zero-overhead path every non-policy
    agent walks."""
    engine = _noop_engine(conversation_store)
    result = await _enforce_output_policy(
        engine=engine,
        task_id="task_1",
        root_task_id=None,
        task_store=MagicMock(),
        text="Hello from an unguardrailed agent",
    )
    assert result == "Hello from an unguardrailed agent"
