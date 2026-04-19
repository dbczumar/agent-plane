"""
End-to-end ASK cycle tests — engine + approval helper
composed in the same sequence the workflow will use.

Each test runs the canonical cycle:

1. ``engine.evaluate(ctx)`` → ASK result with accumulated
   label writes.
2. Caller hands the result to
   :func:`_await_policy_approval` with stub
   register / emit / park callbacks.
3. Verdict drives labels-apply-or-drop per §7.2.
4. Next ``engine.evaluate(ctx)`` sees the post-approval
   state.

This is the complete ASK-cycle contract — if Phase 6 wires
these two pieces together in `_run_agent_loop` correctly,
production behavior will match these assertions.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_plane.runtime.policies import _await_policy_approval
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.policies.label import LabelPolicy
from agent_plane.spec.types import (
    EvaluationContext,
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
    PolicyResult,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)

# ── Harness ────────────────────────────────────────────


class _ApprovalHarness:
    """
    Bundle the register/emit/park seams so tests read cleanly.
    """

    def __init__(self, verdict: str | None) -> None:
        self._verdict = verdict
        self.registered_call_ids: list[str] = []
        self.emitted_items: list[dict[str, Any]] = []

    def register(self, call_id: str, task_id: str, args_json: str) -> None:
        """Record the pending call_id so the test can later
        correlate verdict routing."""
        self.registered_call_ids.append(call_id)

    def emit(self, event: dict[str, Any]) -> None:
        """Record the SSE event item — tests inspect the
        function_call shape for spec parity."""
        self.emitted_items.append(event["item"])

    async def park(self, call_id: str, timeout_s: int) -> str | None:
        """Return the pre-configured verdict string, or
        raise TimeoutError when verdict is ``TIMEOUT``."""
        if self._verdict == "TIMEOUT":
            raise TimeoutError(f"no verdict within {timeout_s}s")
        return self._verdict


async def _run_ask_cycle(
    engine: PolicyEngine,
    ctx: EvaluationContext,
    harness: _ApprovalHarness,
) -> tuple[PolicyResult, bool]:
    """
    Drive one full ASK cycle through the engine + approval
    helper. Returns the composed result + final approval
    outcome.
    """
    result = await engine.evaluate(ctx)
    assert result.action == PolicyAction.ASK, (
        f"Harness expects ASK from evaluate(); got {result.action}"
    )
    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=ctx.phase,
        content_preview=str(ctx.content),
        policy_engine=engine,
        register=harness.register,
        emit=harness.emit,
        park=harness.park,
    )
    return result, approved


def _ask_policy(
    name: str,
    *,
    phase: Phase = Phase.TOOL_CALL,
    tool_name: str | None = "run_shell",
    condition: dict[str, str | list[str]] | None = None,
    set_labels: dict[str, str] | None = None,
    reason: str = "approval required",
) -> LabelPolicy:
    """Build an ASKing LabelPolicy — the typical ASK source."""
    return LabelPolicy(
        LabelPolicySpec(
            name=name,
            on=[PhaseSelector(phase=phase, tool_name=tool_name)],
            condition=condition,
            action=PolicyAction.ASK,
            reason=reason,
            set_labels=set_labels,
        ),
    )


def _build_engine(
    store: SqlAlchemyConversationStore,
    policies: list,
    *,
    initial_labels: dict[str, str] | None = None,
) -> PolicyEngine:
    """Build engine + fresh conversation."""
    conv = store.create_conversation()
    return PolicyEngine(
        policies=policies,
        label_defs={},
        ask_timeout=30,
        conversation_id=conv.id,
        initial_labels=initial_labels or {},
        conversation_store=store,
    )


# ── Happy path: ASK → approve → labels land ──────────


@pytest.mark.asyncio
async def test_ask_cycle_approve_lands_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """End-to-end: engine ASKs with pending label writes;
    caller approves; labels reach the store AND the hot
    cache. Next evaluation sees the new state."""
    policy = _ask_policy(
        "confirm_dangerous",
        set_labels={"approved_once": "true"},
    )
    engine = _build_engine(conversation_store, [policy])
    harness = _ApprovalHarness('{"approved": true}')
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {"cmd": "ls"}},
        tool_name="run_shell",
    )

    result, approved = await _run_ask_cycle(engine, ctx, harness)
    assert approved is True
    # Engine-composed result carried the pending writes.
    assert result.set_labels == {"approved_once": "true"}
    # Post-approval hot cache reflects the write.
    assert engine.labels == {"approved_once": "true"}
    # Persisted — next workflow replay sees the same state.
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {"approved_once": "true"}


@pytest.mark.asyncio
async def test_ask_cycle_refuse_drops_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ASK → refuse → labels DROPPED. Load-bearing §7.2
    invariant: a denied ASK must leave no trace. If this
    regresses, users could effectively approve operations
    by denying them."""
    policy = _ask_policy(
        "confirm_dangerous",
        set_labels={"approved_once": "true"},
    )
    engine = _build_engine(conversation_store, [policy])
    harness = _ApprovalHarness('{"approved": false}')
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {"cmd": "ls"}},
        tool_name="run_shell",
    )

    result, approved = await _run_ask_cycle(engine, ctx, harness)
    assert approved is False
    # set_labels returned on the result (caller would know
    # what was SUPPOSED to land), but NOT applied to the store.
    assert result.set_labels == {"approved_once": "true"}
    assert engine.labels == {}
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {}


@pytest.mark.asyncio
async def test_ask_cycle_timeout_drops_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ASK → timeout → labels DROPPED. Timeout path yields
    same side-effect-free outcome as explicit refuse."""
    policy = _ask_policy(
        "gate",
        set_labels={"integrity": "0"},
    )
    engine = _build_engine(conversation_store, [policy])
    harness = _ApprovalHarness("TIMEOUT")
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {}},
        tool_name="run_shell",
    )

    _, approved = await _run_ask_cycle(engine, ctx, harness)
    assert approved is False
    assert engine.labels == {}
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {}


# ── Multi-policy ASK composition cycle ────────────────


@pytest.mark.asyncio
async def test_ask_cycle_multiple_askers_combined_approval(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """When multiple policies ASK on the same phase, one
    combined approval resolves them all. On approve, every
    ASKing policy's set_labels lands. Proves §4 ASK
    composition + §7.2 single-approval-per-phase."""
    p1 = _ask_policy("first", set_labels={"a": "1"})
    p2 = _ask_policy("second", set_labels={"b": "2"})
    p3 = _ask_policy("third", set_labels={"c": "3"})
    engine = _build_engine(conversation_store, [p1, p2, p3])
    harness = _ApprovalHarness('{"approved": true}')
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {}},
        tool_name="run_shell",
    )

    result, approved = await _run_ask_cycle(engine, ctx, harness)
    assert approved is True
    # First-ASKer-in-YAML wins deciding_policy.
    assert result.deciding_policy == "first"
    # Combined reason mentions all three policies.
    assert "first:" in result.reason
    assert "second:" in result.reason
    assert "third:" in result.reason
    # All three policies' set_labels landed — single
    # approval authorized every write.
    assert engine.labels == {"a": "1", "b": "2", "c": "3"}


@pytest.mark.asyncio
async def test_ask_cycle_multiple_askers_combined_refuse(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Same multi-policy scenario with a refuse. NONE of
    the labels land — all-or-nothing semantics."""
    p1 = _ask_policy("first", set_labels={"a": "1"})
    p2 = _ask_policy("second", set_labels={"b": "2"})
    engine = _build_engine(conversation_store, [p1, p2])
    harness = _ApprovalHarness('{"approved": false}')
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {}},
        tool_name="run_shell",
    )

    _, approved = await _run_ask_cycle(engine, ctx, harness)
    assert approved is False
    assert engine.labels == {}


# ── State flows across ASK cycles ─────────────────────


@pytest.mark.asyncio
async def test_approved_labels_visible_in_next_evaluation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """After an approval applies `integrity: 0`, a later
    condition-gated policy can read that state and fire
    accordingly. Demonstrates ASK → label → condition-driven
    downstream behavior — the core IFC loop through ASK."""
    # First policy ASKs, writes integrity=0 on approve.
    taint = _ask_policy(
        "confirm_taint",
        set_labels={"integrity": "0"},
    )
    # Second policy fires UNCONDITIONALLY on run_shell (no
    # tool narrowing on our selector → matches every
    # run_shell invocation) with a condition-gate on
    # integrity=0. Only enforces after taint is established.
    shell_guard = LabelPolicy(
        LabelPolicySpec(
            name="shell_guard",
            on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="run_shell")],
            condition={"integrity": "0"},
            action=PolicyAction.DENY,
            reason="tainted; shell disallowed",
        ),
    )
    engine = _build_engine(conversation_store, [taint, shell_guard])
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {"cmd": "ls"}},
        tool_name="run_shell",
    )

    # First cycle: ASK then approve.
    harness1 = _ApprovalHarness('{"approved": true}')
    _, approved = await _run_ask_cycle(engine, ctx, harness1)
    assert approved is True
    assert engine.labels["integrity"] == "0"

    # Second cycle: same ctx, now shell_guard's condition
    # matches → DENY short-circuits before the ASKing
    # policy fires.
    result2 = await engine.evaluate(ctx)
    assert result2.action == PolicyAction.DENY
    assert result2.deciding_policy == "shell_guard"


@pytest.mark.asyncio
async def test_refused_ask_does_not_poison_next_evaluation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """After a REFUSED ASK, the label state must stay
    clean — a subsequent re-evaluation sees the original
    context and can ASK again (or ALLOW)."""
    policy = _ask_policy("retry_gate", set_labels={"dangerous": "1"})
    engine = _build_engine(conversation_store, [policy])
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "run_shell", "args": {}},
        tool_name="run_shell",
    )

    # First cycle: refuse.
    harness1 = _ApprovalHarness('{"approved": false}')
    _, approved1 = await _run_ask_cycle(engine, ctx, harness1)
    assert approved1 is False
    # State unchanged.
    assert engine.labels == {}

    # Second cycle: re-evaluation sees the same clean state.
    # The policy ASKs AGAIN (not stuck post-refuse).
    harness2 = _ApprovalHarness('{"approved": true}')
    _, approved2 = await _run_ask_cycle(engine, ctx, harness2)
    assert approved2 is True
    assert engine.labels == {"dangerous": "1"}


# ── Emitted function_call shape verification ──────────


@pytest.mark.asyncio
async def test_emitted_function_call_matches_spec(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """The synthetic function_call the ASK flow emits
    matches POLICIES.md §7.1: name=request_approval,
    status=action_required, args JSON carries the four
    ApprovalRequest fields. Clients that recognize
    ``request_approval`` render an approval UI; tests must
    guarantee this contract hasn't drifted."""
    policy = _ask_policy(
        "confirm_write",
        phase=Phase.TOOL_CALL,
        tool_name="write_file",
        reason="writes require review",
    )
    engine = _build_engine(conversation_store, [policy])
    harness = _ApprovalHarness('{"approved": true}')
    ctx = EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": "write_file", "args": {"path": "secrets.txt"}},
        tool_name="write_file",
    )

    await _run_ask_cycle(engine, ctx, harness)
    # Exactly one function_call emitted.
    assert len(harness.emitted_items) == 1
    item = harness.emitted_items[0]
    assert item["type"] == "function_call"
    assert item["name"] == "request_approval"
    assert item["status"] == "action_required"
    # call_id consistency between registration + emission.
    assert item["call_id"] == harness.registered_call_ids[0]
    # Arguments payload carries the four fields a client
    # renders in its approval UI.
    import json

    args = json.loads(item["arguments"])
    # policy_name is the FIRST ASKer in YAML order — which
    # is also the deciding_policy on the composed result.
    assert args["policy_name"] == "confirm_write"
    assert args["phase"] == "tool_call"
    # reason includes the combined form that the engine
    # produces when compositing.
    assert "confirm_write: writes require review" in args["reason"]
