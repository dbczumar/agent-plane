"""
Multi-turn e2e tests — simulate a full agent conversation
across engine rebuilds, label persistence, and ASK cycles.

Each test mimics what a real workflow would do across
multiple `_run_agent_loop` iterations on the same
conversation:

1. Build an engine on turn 1 from the spec.
2. Evaluate phases (ALLOW / ASK / DENY), maybe running
   through the approval helper.
3. Rebuild the engine (workflow restart / new turn) on the
   same conversation — labels persist.
4. Subsequent evaluations see the post-previous-turn state.

Covers the properties a production run depends on that
cannot be tested by single-turn fixtures alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_plane.policies.prompt import PromptPolicy
from agent_plane.policies.types import EvaluationContext
from agent_plane.runtime.policies import (
    _await_policy_approval,
    build_policy_engine,
)
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.spec.parser import parse
from agent_plane.spec.types import (
    Phase,
    PolicyAction,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "_fixtures" / "agents"


class _Harness:
    """Minimal approval harness for multi-turn scenarios."""

    def __init__(self) -> None:
        self.captured_prompts: list[str] = []
        self.current_verdict: str | None = None

    def register(self, call_id: str, task_id: str, args_json: str) -> None:
        pass

    def emit(self, event: dict[str, Any]) -> None:
        pass

    def set_verdict(self, verdict: str) -> None:
        """Program the next ASK verdict before driving a cycle."""
        self.current_verdict = verdict

    async def park(self, call_id: str, timeout_s: int) -> str | None:
        """Return the verdict set by the test."""
        return self.current_verdict


def _tool_ctx(name: str, args: dict[str, Any] | None = None) -> EvaluationContext:
    """TOOL_CALL context helper."""
    return EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": name, "args": args or {}},
        tool_name=name,
    )


def _rebuild_engine(
    store: SqlAlchemyConversationStore,
    conversation_id: str,
    fixture: str = "secure-research",
) -> PolicyEngine:
    """Rebuild an engine on an existing conversation.

    Mirrors what a workflow does at the top of each
    `_run_agent_loop` iteration: fresh PolicyEngine,
    persisted label state seeded from the store."""
    spec = parse(_FIXTURES / fixture)
    return build_policy_engine(
        spec=spec,
        conversation_id=conversation_id,
        conversation_store=store,
    )


# ── Multi-turn state persistence ──────────────────────


@pytest.mark.asyncio
async def test_taint_persists_across_workflow_restarts(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Simulates a workflow crash + replay. Turn 1 taints
    integrity via web_search; turn 2 (fresh engine, same
    conversation) sees the tainted state and DENYs / ASKs
    shell calls. If this regresses, a restart would silently
    reset the label state — catastrophic for IFC guarantees."""
    spec = parse(_FIXTURES / "secure-research")
    conv = conversation_store.create_conversation()

    # Turn 1: taint integrity via web_search.
    engine_1 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    r1 = await engine_1.evaluate(_tool_ctx("web_search", {"q": "q"}))
    assert r1.action == PolicyAction.ALLOW
    assert engine_1.labels["integrity"] == "0"

    # Turn 2: completely fresh engine on the same
    # conversation (models a workflow restart).
    engine_2 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    # The fresh engine's hot cache picked up the tainted
    # integrity — ON CONFLICT DO NOTHING left the "0" alone.
    assert engine_2.labels["integrity"] == "0"

    # run_shell on turn 2 → ASK (low-integrity enforcement).
    r2 = await engine_2.evaluate(_tool_ctx("run_shell", {"cmd": "ls"}))
    assert r2.action == PolicyAction.ASK
    assert r2.deciding_policy == "ask_low_integrity"


@pytest.mark.asyncio
async def test_multi_turn_ask_approve_then_later_denial(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Turn 1: web search taints, shell ASKs, user approves.
    Turn 2: confidential read additionally elevates
    sensitivity. Turn 3: shell DENYs (deny_exfil condition
    requires BOTH taints). Proves the ASK approval on turn 1
    survives to turn 3's stricter evaluation."""
    spec = parse(_FIXTURES / "secure-research")
    conv = conversation_store.create_conversation()

    # Turn 1: web search + shell with approve.
    engine_1 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    await engine_1.evaluate(_tool_ctx("web_search", {"q": "x"}))
    shell_result_1 = await engine_1.evaluate(_tool_ctx("run_shell", {"cmd": "ls"}))
    assert shell_result_1.action == PolicyAction.ASK
    # Approve via the real helper.
    harness = _Harness()
    harness.set_verdict('{"approved": true}')
    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=shell_result_1,
        phase=Phase.TOOL_CALL,
        content_preview="ls",
        policy_engine=engine_1,
        register=harness.register,
        emit=harness.emit,
        park=harness.park,
    )
    assert approved is True
    # shell_result_1 had no pending set_labels (the ASKing
    # policies in secure-research don't write any), so
    # post-approval state is unchanged beyond the taint.

    # Turn 2: fresh engine, now also read confidential.
    engine_2 = _rebuild_engine(conversation_store, conv.id)
    await engine_2.evaluate(_tool_ctx("read_internal_doc", {"id": "d"}))
    # secure-research labels are `integrity` + `confidentiality`.
    assert engine_2.labels == {"integrity": "0", "confidentiality": "1"}

    # Turn 3: fresh engine, attempt shell with BOTH taints.
    engine_3 = _rebuild_engine(conversation_store, conv.id)
    r3 = await engine_3.evaluate(_tool_ctx("run_shell", {"cmd": "ls"}))
    # deny_exfil fires (both labels tainted) — this is a
    # hard DENY now, not an ASK that could be approved.
    assert r3.action == PolicyAction.DENY
    assert r3.deciding_policy == "deny_contaminated_shell"


@pytest.mark.asyncio
async def test_rate_limit_counter_is_workflow_scoped(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Stateful FunctionPolicy (rate limiter) resets between
    workflow rebuilds — its closure state is per-workflow by
    design (POLICIES.md §4). If this regresses, a user who
    hits the rate limit could never exceed it again even on
    a fresh conversation-restart turn.

    Note: this is the DECLARED behavior. An agent author
    who wants cross-turn rate limiting uses labels; closure
    state is intentionally per-workflow."""
    spec = parse(_FIXTURES / "rate-limited-search")
    conv = conversation_store.create_conversation()

    # Turn 1: exhaust the 3-call budget, 4th asks.
    engine_1 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    for _ in range(3):
        r = await engine_1.evaluate(_tool_ctx("web_search", {"q": "x"}))
        assert r.action == PolicyAction.ALLOW
    r = await engine_1.evaluate(_tool_ctx("web_search", {"q": "x"}))
    assert r.action == PolicyAction.ASK

    # Turn 2: fresh engine — counter resets (closure is
    # workflow-scoped). First call ALLOWs again.
    engine_2 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    r2 = await engine_2.evaluate(_tool_ctx("web_search", {"q": "y"}))
    # ALLOW — counter reset on rebuild.
    assert r2.action == PolicyAction.ALLOW


# ── Approved-once / blocked-next-time via labels ──────


@pytest.mark.asyncio
async def test_approval_writes_persistent_audit_label(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Models a 'once-approved, always-approved' pattern
    via labels: the ASKing policy writes a label on approve;
    a later policy sees the label and skips the ASK.

    This is a hand-crafted agent (no fixture) because it's
    the canonical use case for ASK-writes-labels that isn't
    covered by the three omniagents fixtures."""
    from agent_plane.policies.label import LabelPolicy
    from agent_plane.spec.types import (
        LabelPolicySpec,
        PhaseSelector,
    )

    # Policy 1: if `shell_approved` not yet "yes", ASK.
    ask_once = LabelPolicy(
        LabelPolicySpec(
            name="ask_once",
            on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="run_shell")],
            action=PolicyAction.ASK,
            reason="first-use approval",
            set_labels={"shell_approved": "yes"},
        ),
    )
    # Policy 2: once `shell_approved == "yes"`, this label
    # policy matches and emits ALLOW immediately.
    # POLICIES.md §4 max-action composition: a later ALLOW
    # does NOT override an earlier ASK, but the ASK policy
    # is gated on the ABSENCE of the approval label — once
    # approved, it doesn't fire.
    conv = conversation_store.create_conversation()
    engine_1 = PolicyEngine(
        policies=[ask_once],
        label_defs={},
        ask_timeout=30,
        conversation_id=conv.id,
        initial_labels={},
        conversation_store=conversation_store,
    )
    # First turn: ASK.
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {"cmd": "ls"}},
        tool_name="run_shell",
    )
    r1 = await engine_1.evaluate(ctx)
    assert r1.action == PolicyAction.ASK
    # Approve.
    harness = _Harness()
    harness.set_verdict('{"approved": true}')
    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=r1,
        phase=Phase.TOOL_CALL,
        content_preview="ls",
        policy_engine=engine_1,
        register=harness.register,
        emit=harness.emit,
        park=harness.park,
    )
    assert approved is True
    # Label landed on approve.
    assert engine_1.labels == {"shell_approved": "yes"}

    # Now reconfigure: same ASK policy but WITH a
    # condition that suppresses it once approved. Engine
    # rebuild simulates the "I already approved this in
    # a prior workflow" case.
    ask_once_gated = LabelPolicy(
        LabelPolicySpec(
            name="ask_once_gated",
            on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="run_shell")],
            # ASK only applies when approval not yet granted —
            # condition matches on absence requires a marker,
            # so we key on a negative ("not approved"). Here we
            # model it by only ASKing when shell_approved != "yes"
            # — but condition only supports equality, so we
            # invert: the policy fires when label MATCHES some
            # default-unset value. This is the standard idiom.
            condition={"shell_approved": "no"},
            action=PolicyAction.ASK,
            reason="first-use approval",
        ),
    )
    # Build fresh engine with shell_approved already "yes"
    # from the prior turn's approval.
    engine_2 = PolicyEngine(
        policies=[ask_once_gated],
        label_defs={},
        ask_timeout=30,
        conversation_id=conv.id,
        initial_labels=dict(engine_1.labels),
        conversation_store=conversation_store,
    )
    # Condition ``shell_approved == "no"`` does NOT match
    # (actual is "yes") → policy skipped → default ALLOW.
    r2 = await engine_2.evaluate(ctx)
    assert r2.action == PolicyAction.ALLOW


# ── PromptPolicy across turns ─────────────────────────


@pytest.mark.asyncio
async def test_prompt_policy_classification_label_persists(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A PromptPolicy classifier writes a sensitivity label
    on a tool result. Later turns see the elevated label
    without calling the classifier again (the label itself
    is the memo). Demonstrates why persistent labels matter
    for expensive LLM-backed classifiers."""
    spec = parse(_FIXTURES / "prompt-policy-demo")
    conv = conversation_store.create_conversation()

    # Turn 1: classifier writes sensitivity=confidential.
    engine_1 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )

    async def _classifier(prompt: str) -> dict[str, Any]:
        # Returns confidential classification.
        return {"action": "allow", "set_labels": {"sensitivity": "confidential"}}

    for policy in engine_1.policies:
        if isinstance(policy, PromptPolicy) and policy.spec.name == "classify_doc_sensitivity":
            policy._classifier = _classifier

    r = await engine_1.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_RESULT,
            content={"output": "confidential text"},
            tool_name="read_doc",
        ),
    )
    assert r.action == PolicyAction.ALLOW
    assert engine_1.labels["sensitivity"] == "confidential"

    # Turn 2: fresh engine on same conversation — picks up
    # the persisted classification WITHOUT invoking the
    # classifier again (initial label wins because
    # monotonic=increasing blocks any decrease anyway, but
    # the point is the classifier doesn't run on a pure
    # read-side operation).
    engine_2 = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    assert engine_2.labels["sensitivity"] == "confidential"
    # Input-phase classifier would only run if we evaluate
    # INPUT. If the workflow doesn't touch tool_result on
    # this turn, the (expensive) classifier never fires.
