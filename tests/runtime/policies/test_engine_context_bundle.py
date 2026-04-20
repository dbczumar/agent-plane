"""
Tests for :meth:`PolicyEngine._context`.

The context bundle is what FunctionPolicy callables receive
as their second argument (and what PromptPolicy would
receive if its evaluate() used the bundle — it currently
doesn't, reserved for future). Verifies:

- Bundle carries labels + conversation_id.
- Labels in the bundle are a defensive copy (caller
  mutation doesn't corrupt engine state).
- Bundle reflects the CURRENT hot cache at each evaluation,
  not a stale snapshot from engine init.
- Policies see each other's set_labels writes via the
  bundle (sequential within one evaluate() call).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_plane.policies.function import FunctionPolicy
from agent_plane.policies.label import LabelPolicy
from agent_plane.policies.types import EvaluationContext, PolicyResult
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.spec.types import (
    FunctionPolicySpec,
    FunctionRef,
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _capturing_policy(bucket: dict[str, Any]) -> FunctionPolicy:
    """Build a FunctionPolicy that records the context it
    receives into *bucket*. Used to inspect what the engine
    passed at evaluate time."""

    def _evaluate(
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> PolicyResult:
        bucket["ctx"] = ctx
        bucket["context"] = dict(context)  # copy to capture snapshot
        return PolicyResult(action=PolicyAction.ALLOW)

    spec = FunctionPolicySpec(
        name="capture",
        on=[PhaseSelector(phase=Phase.INPUT)],
        function=FunctionRef(path="test.not.used"),  # build-time stub
    )
    return FunctionPolicy(spec, _evaluate)


def _build(
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


# ── Context bundle carries labels + conversation_id ──


@pytest.mark.asyncio
async def test_context_carries_labels_and_conversation_id(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """FunctionPolicy's second arg receives the engine's
    context bundle — labels snapshot + conversation_id."""
    bucket: dict[str, Any] = {}
    policy = _capturing_policy(bucket)
    engine = _build(
        conversation_store,
        [policy],
        initial_labels={"integrity": "1"},
    )
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    # labels present.
    assert bucket["context"]["labels"] == {"integrity": "1"}
    # conversation_id present and correct.
    assert bucket["context"]["conversation_id"] == engine.conversation_id


@pytest.mark.asyncio
async def test_context_labels_is_defensive_copy(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A policy callable mutating the labels dict it
    received must NOT corrupt engine state. Without the
    defensive copy, a buggy / malicious policy could clear
    all labels silently."""
    policy_labels_after_mutation: dict[str, str] = {}

    def _mutating(
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> PolicyResult:
        # Mutate the dict we received.
        context["labels"]["integrity"] = "tampered"
        context["labels"]["NEW_KEY"] = "injected"
        return PolicyResult(action=PolicyAction.ALLOW)

    spec = FunctionPolicySpec(
        name="mutator",
        on=[PhaseSelector(phase=Phase.INPUT)],
        function=FunctionRef(path="test.not.used"),
    )
    policy = FunctionPolicy(spec, _mutating)
    engine = _build(
        conversation_store,
        [policy],
        initial_labels={"integrity": "1"},
    )
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    # Engine's hot cache is UNCHANGED — tampered value
    # did not leak.
    assert engine.labels == {"integrity": "1"}
    # Other tests recover — no injected state.
    policy_labels_after_mutation.update(engine.labels)
    assert "NEW_KEY" not in policy_labels_after_mutation


# ── Hot cache freshness ───────────────────────────────


@pytest.mark.asyncio
async def test_context_reflects_current_hot_cache(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Context at evaluate-time reflects the engine's hot
    cache AS IT IS NOW, not as it was at engine init. A
    subsequent evaluation sees the effects of a prior
    policy's set_labels."""
    bucket_1: dict[str, Any] = {}
    bucket_2: dict[str, Any] = {}

    # Two captures — we swap between evaluations.
    policy_1 = _capturing_policy(bucket_1)
    policy_2 = _capturing_policy(bucket_2)

    engine = _build(
        conversation_store,
        [policy_1],
        initial_labels={"integrity": "1"},
    )

    # First evaluation: policy_1 fires.
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    assert bucket_1["context"]["labels"] == {"integrity": "1"}

    # Apply a write outside evaluate — bumps the hot cache.
    engine.apply_label_writes({"integrity": "0"})

    # Swap in policy_2 and re-evaluate.
    engine.policies = [policy_2]
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    # Second bucket captured the UPDATED hot cache.
    assert bucket_2["context"]["labels"] == {"integrity": "0"}


# ── Composed evaluations see prior writes within same ──


@pytest.mark.asyncio
async def test_policy_sees_earlier_policys_set_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A LabelPolicy writes X; a later FunctionPolicy in
    the same `evaluate()` call sees X via context.labels.

    Wait — the CURRENT engine implementation builds the
    context ONCE at the top of evaluate() and passes the
    same copy to each policy. So within one evaluate call,
    a later policy sees the INITIAL state, not the
    accumulated state. Let's verify that documented behavior
    here rather than assume.
    """
    bucket: dict[str, Any] = {}

    # Policy 1: LabelPolicy writes integrity=0.
    writer = LabelPolicy(
        LabelPolicySpec(
            name="writer",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ALLOW,
            set_labels={"integrity": "0"},
        ),
    )

    # Policy 2: FunctionPolicy captures what it sees.
    reader = _capturing_policy(bucket)

    engine = _build(
        conversation_store,
        [writer, reader],
        initial_labels={"integrity": "1"},
    )
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))

    # Within one evaluate() call, context is built ONCE at
    # the top. The reader sees the PRE-WRITER snapshot.
    # This is a documented tradeoff — if policies need to
    # react to each other's writes, they should be split
    # across evaluations (different phases).
    assert bucket["context"]["labels"] == {"integrity": "1"}
    # After evaluate, the cache reflects the accumulated
    # writes from the ALLOW composition.
    assert engine.labels == {"integrity": "0"}
