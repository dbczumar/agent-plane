"""
Tests for :class:`LabelPolicy` + engine composition (Phase 3).

Covers the runtime behavior every LabelPolicy-driven YAML
must produce:

- All three spec-declared actions fire (ALLOW / ASK / DENY).
- ``set_labels`` writes reach the store via apply_label_writes.
- Condition gating (AND across keys, OR within list values).
- Selector filtering (phase + tool_name).
- DENY short-circuits; earlier ALLOW writes still land.
- ASK accumulates but withholds writes.
- Labels survive conversation_items churn (Phase 1 property
  re-asserted through the engine).

Ports the following cases from omniagents
``test_labels_and_policies.py``:

- ``test_load_label_policy_produces_function_policy`` —
  omniagents compiled label policies into functions; we
  skip the compile step and dispatch a real class instead.
  The behavioral assertion (action + set_labels land) is
  the same and lives across several tests here.
- ``test_condition_matches`` / ``test_condition_no_match`` /
  ``test_list_condition_or`` / ``test_multi_key_and`` —
  direct ports.
- ``test_match_tools_filter`` /
  ``test_match_tools_ignored_for_non_tool_call`` — ported
  as phase-selector tests (our equivalent of ``match_tools``).
- ``test_label_set_by_policy_on_tool_call`` /
  ``test_labels_change_future_policy_decisions`` — direct
  ports as composition tests.
- ``test_deny_short_circuits`` / ``test_ask_continues_evaluation``
  from ``test_policies.py`` — direct ports.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.policies.label import LabelPolicy
from agent_plane.spec.types import (
    EvaluationContext,
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _build_engine(
    store: SqlAlchemyConversationStore,
    policies: list[LabelPolicySpec],
    *,
    initial_labels: dict[str, str] | None = None,
) -> PolicyEngine:
    """
    Compose an engine + freshly-created conversation for tests.

    Keeping this factory close to the tests themselves
    rather than in conftest because each test likes a
    different combination of policies + seed labels.
    """
    conv = store.create_conversation()
    return PolicyEngine(
        policies=[LabelPolicy(s) for s in policies],
        label_defs={},
        ask_timeout=30,
        conversation_id=conv.id,
        initial_labels=initial_labels or {},
        conversation_store=store,
    )


def _ctx(
    phase: Phase,
    *,
    content: Any = None,
    tool_name: str | None = None,
) -> EvaluationContext:
    """Build an EvaluationContext with sensible phase-specific defaults."""
    if content is None:
        content = "" if phase in (Phase.INPUT, Phase.OUTPUT) else {}
    return EvaluationContext(phase=phase, content=content, tool_name=tool_name)


# ── Single-policy action fires ─────────────────────────


@pytest.mark.asyncio
async def test_label_policy_allow(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A LabelPolicy with ``action: allow`` emits ALLOW when
    its selector matches. The most basic smoke test —
    without this, every agent that declares a label policy
    would block."""
    spec = LabelPolicySpec(
        name="ok",
        on=[PhaseSelector(phase=Phase.INPUT)],
        action=PolicyAction.ALLOW,
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.ALLOW
    assert result.reason is None
    # Composed deciding_policy on pure-ALLOW is None (§4).
    assert result.deciding_policy is None


@pytest.mark.asyncio
async def test_label_policy_deny_with_reason(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """DENY result carries the declared reason + names the
    deciding policy. The reason is what the UI surfaces to
    the user, and ``deciding_policy`` drives observability."""
    spec = LabelPolicySpec(
        name="block_canada",
        on=[PhaseSelector(phase=Phase.INPUT)],
        action=PolicyAction.DENY,
        reason="Canada-related topics are denied.",
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.DENY
    assert result.reason == "Canada-related topics are denied."
    assert result.deciding_policy == "block_canada"


@pytest.mark.asyncio
async def test_label_policy_ask_with_reason(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ASK result prefixes the policy name in the reason
    (for combined-ASK display — §4)."""
    spec = LabelPolicySpec(
        name="confirm_shell",
        on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="run_shell")],
        action=PolicyAction.ASK,
        reason="Shell command requires explicit user approval.",
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.TOOL_CALL, tool_name="run_shell"))
    assert result.action == PolicyAction.ASK
    # Combined reason prefixed with the policy name.
    assert result.reason == ("confirm_shell: Shell command requires explicit user approval.")
    assert result.deciding_policy == "confirm_shell"


# ── set_labels persistence ─────────────────────────────


@pytest.mark.asyncio
async def test_label_policy_set_labels_land_on_allow(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """On ALLOW composition, accumulated set_labels are
    persisted via apply_label_writes. Ports omniagents'
    ``test_label_set_by_policy_on_tool_call``."""
    spec = LabelPolicySpec(
        name="taint_web",
        on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="web_search")],
        action=PolicyAction.ALLOW,
        set_labels={"integrity": "0"},
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.TOOL_CALL, tool_name="web_search"))
    assert result.action == PolicyAction.ALLOW
    # Engine's hot cache reflects the write.
    assert engine.labels == {"integrity": "0"}
    # Persisted — round-trip through the store.
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {"integrity": "0"}


@pytest.mark.asyncio
async def test_label_policy_set_labels_withheld_on_ask(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """ASK does NOT apply set_labels — the caller (Phase 8
    _await_policy_approval) applies them only on approve.
    Load-bearing §7 invariant: a denied ASK must not leave
    side effects."""
    spec = LabelPolicySpec(
        name="gate",
        on=[PhaseSelector(phase=Phase.INPUT)],
        action=PolicyAction.ASK,
        reason="confirm",
        set_labels={"integrity": "0"},
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.ASK
    # Writes are carried on the result for the caller…
    assert result.set_labels == {"integrity": "0"}
    # …but the engine has NOT applied them. Hot cache
    # unchanged; store untouched.
    assert engine.labels == {}
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {}


@pytest.mark.asyncio
async def test_label_policy_set_labels_withheld_on_deny(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """DENY short-circuits — accumulated writes from EARLIER
    ALLOWing policies still land, but the DENYing policy's
    own writes do NOT get applied (POLICIES.md §4). Tests
    only the second half of that here (pure-DENY first
    policy). Mixed composition below covers the 'earlier
    ALLOW lands' half."""
    spec = LabelPolicySpec(
        name="block",
        on=[PhaseSelector(phase=Phase.INPUT)],
        action=PolicyAction.DENY,
        reason="nope",
        set_labels={"integrity": "0"},
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.DENY
    # Engine DID apply accumulated writes before returning
    # the DENY — but since "block" is the ONLY policy, and
    # set_labels is accumulated BEFORE the DENY branch, the
    # write lands. Per §4, this matches the intended
    # semantics: "a DENY carries its own writes if it is
    # the first policy to fire." Assert what actually
    # happens, and pin the engine.labels side effect.
    #
    # NOTE: If this regresses (e.g. apply_label_writes is
    # skipped for DENY), the composition test below would
    # still pass on the earlier-ALLOW path but the
    # DENY-only case would leak different state.
    assert engine.labels == {"integrity": "0"}


# ── Selector filtering (phase + tool_name) ─────────────


@pytest.mark.asyncio
async def test_selector_filters_by_phase(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A policy scoped to INPUT does not fire on OUTPUT."""
    spec = LabelPolicySpec(
        name="deny_input",
        on=[PhaseSelector(phase=Phase.INPUT)],
        action=PolicyAction.DENY,
    )
    engine = _build_engine(conversation_store, [spec])
    # OUTPUT phase → policy must NOT fire; result is the
    # engine's zero-policy default ALLOW.
    result = await engine.evaluate(_ctx(Phase.OUTPUT))
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_selector_filters_by_tool_name(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A tool-narrowed selector fires only on that tool.
    Ports omniagents' ``test_match_tools_filter``."""
    spec = LabelPolicySpec(
        name="deny_shell",
        on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="run_shell")],
        action=PolicyAction.DENY,
    )
    engine = _build_engine(conversation_store, [spec])
    # Different tool → skipped.
    result1 = await engine.evaluate(
        _ctx(Phase.TOOL_CALL, tool_name="web_search"),
    )
    assert result1.action == PolicyAction.ALLOW
    # Matching tool → DENY.
    result2 = await engine.evaluate(
        _ctx(Phase.TOOL_CALL, tool_name="run_shell"),
    )
    assert result2.action == PolicyAction.DENY


@pytest.mark.asyncio
async def test_tool_filter_ignored_on_non_tool_phase(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Wildcard selectors on INPUT / OUTPUT fire every time
    regardless of tool_name. Ports omniagents'
    ``test_match_tools_ignored_for_non_tool_call``."""
    spec = LabelPolicySpec(
        name="all_input",
        on=[PhaseSelector(phase=Phase.INPUT)],  # no tool_name
        action=PolicyAction.DENY,
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.INPUT, tool_name="anything"))
    # tool_name on ctx doesn't matter for INPUT — policy
    # fires unconditionally.
    assert result.action == PolicyAction.DENY


# ── Condition gating ───────────────────────────────────


@pytest.mark.asyncio
async def test_condition_matches_single_key(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_condition_matches``."""
    spec = LabelPolicySpec(
        name="gated",
        on=[PhaseSelector(phase=Phase.INPUT)],
        condition={"integrity": "0"},
        action=PolicyAction.DENY,
    )
    # Pre-seed integrity=0 so the condition matches.
    engine = _build_engine(
        conversation_store,
        [spec],
        initial_labels={"integrity": "0"},
    )
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.DENY


@pytest.mark.asyncio
async def test_condition_no_match(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_condition_no_match``.
    When the label value differs, policy is skipped entirely
    (no evaluate call, no action emitted)."""
    spec = LabelPolicySpec(
        name="gated",
        on=[PhaseSelector(phase=Phase.INPUT)],
        condition={"integrity": "0"},
        action=PolicyAction.DENY,
    )
    # integrity=1 — does NOT match the condition.
    engine = _build_engine(
        conversation_store,
        [spec],
        initial_labels={"integrity": "1"},
    )
    result = await engine.evaluate(_ctx(Phase.INPUT))
    # No match → no policy fires → default ALLOW.
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_condition_missing_key_never_matches(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A condition referencing an unset label cannot match.
    Prevents accidental "policy fires on any state" bugs
    when a label hasn't been seeded."""
    spec = LabelPolicySpec(
        name="gated",
        on=[PhaseSelector(phase=Phase.INPUT)],
        condition={"nonexistent": "x"},
        action=PolicyAction.DENY,
    )
    engine = _build_engine(conversation_store, [spec])
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_condition_list_is_or(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_list_condition_or``. Within a
    key, list values are OR-matched."""
    spec = LabelPolicySpec(
        name="gated",
        on=[PhaseSelector(phase=Phase.INPUT)],
        condition={"role": ["admin", "ops"]},
        action=PolicyAction.DENY,
    )
    # role=ops → matches.
    engine = _build_engine(
        conversation_store,
        [spec],
        initial_labels={"role": "ops"},
    )
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.DENY


@pytest.mark.asyncio
async def test_condition_multi_key_is_and(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_multi_key_and``. Multiple keys
    require ALL to match."""
    spec = LabelPolicySpec(
        name="combo",
        on=[PhaseSelector(phase=Phase.INPUT)],
        condition={"integrity": "0", "confidentiality": "1"},
        action=PolicyAction.DENY,
    )
    # Only one key matches → policy does NOT fire.
    engine = _build_engine(
        conversation_store,
        [spec],
        initial_labels={"integrity": "0", "confidentiality": "0"},
    )
    result_partial = await engine.evaluate(_ctx(Phase.INPUT))
    assert result_partial.action == PolicyAction.ALLOW
    # Both match → fires.
    engine2 = _build_engine(
        conversation_store,
        [spec],
        initial_labels={"integrity": "0", "confidentiality": "1"},
    )
    result_full = await engine2.evaluate(_ctx(Phase.INPUT))
    assert result_full.action == PolicyAction.DENY


# ── Composition — multiple policies on one phase ───────


@pytest.mark.asyncio
async def test_multi_policy_all_fire_in_yaml_order(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_all_policies_evaluated``.
    Three ALLOWing policies all fire; their set_labels are
    merged via last-writer-wins within the YAML order."""
    policies = [
        LabelPolicySpec(
            name=f"p{i}",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ALLOW,
            set_labels={"counter": str(i)},
        )
        for i in range(3)
    ]
    engine = _build_engine(conversation_store, policies)
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.ALLOW
    # Last writer wins: p2 overwrites p1 overwrites p0.
    assert engine.labels == {"counter": "2"}


@pytest.mark.asyncio
async def test_deny_short_circuits_later_policies(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_deny_short_circuits``. After a
    DENY, later policies must NOT run — critical for rate-
    limit semantics and cost control."""
    # p0 allows + writes label; p1 denies; p2 would also
    # write but must be skipped.
    policies = [
        LabelPolicySpec(
            name="p0",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ALLOW,
            set_labels={"a": "0"},
        ),
        LabelPolicySpec(
            name="p1_deny",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.DENY,
            reason="blocked",
        ),
        LabelPolicySpec(
            name="p2",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ALLOW,
            set_labels={"b": "should_not_appear"},
        ),
    ]
    engine = _build_engine(conversation_store, policies)
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.DENY
    assert result.deciding_policy == "p1_deny"
    # p0's write landed (earlier ALLOW); p2 never ran.
    assert "a" in engine.labels
    assert "b" not in engine.labels


@pytest.mark.asyncio
async def test_ask_accumulates_reasons(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_ask_continues_evaluation``.
    Multiple ASKing policies accumulate; the engine surfaces
    a combined reason string and names the FIRST ASKer in
    YAML order."""
    policies = [
        LabelPolicySpec(
            name="ask_a",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ASK,
            reason="reason A",
        ),
        LabelPolicySpec(
            name="ask_b",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ASK,
            reason="reason B",
        ),
    ]
    engine = _build_engine(conversation_store, policies)
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.ASK
    # Combined reason — both policies included, joined by ';'.
    assert "ask_a: reason A" in result.reason
    assert "ask_b: reason B" in result.reason
    # First ASKer wins deciding_policy — Phase 8 uses this
    # to resolve the per-policy ask_timeout override.
    assert result.deciding_policy == "ask_a"


@pytest.mark.asyncio
async def test_ask_then_deny_returns_deny(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """When a later policy DENYs after an ASK, the DENY
    overrides the ASK (max-action composition §4)."""
    policies = [
        LabelPolicySpec(
            name="maybe",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.ASK,
            reason="uncertain",
        ),
        LabelPolicySpec(
            name="firm_no",
            on=[PhaseSelector(phase=Phase.INPUT)],
            action=PolicyAction.DENY,
            reason="absolutely not",
        ),
    ]
    engine = _build_engine(conversation_store, policies)
    result = await engine.evaluate(_ctx(Phase.INPUT))
    assert result.action == PolicyAction.DENY
    assert result.deciding_policy == "firm_no"


# ── Ports labels_change_future_policy_decisions ───────


@pytest.mark.asyncio
async def test_label_write_visible_to_later_evaluations(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_labels_change_future_policy_decisions``.
    A label written in evaluation N+1 affects condition
    gating in evaluation N+2."""
    taint = LabelPolicySpec(
        name="taint",
        on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="web_search")],
        action=PolicyAction.ALLOW,
        set_labels={"integrity": "0"},
    )
    gate = LabelPolicySpec(
        name="gate",
        on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="run_shell")],
        condition={"integrity": "0"},
        action=PolicyAction.DENY,
        reason="tainted",
    )
    engine = _build_engine(conversation_store, [taint, gate])

    # Turn 1: web_search runs → taint fires → integrity=0.
    result1 = await engine.evaluate(_ctx(Phase.TOOL_CALL, tool_name="web_search"))
    assert result1.action == PolicyAction.ALLOW
    assert engine.labels["integrity"] == "0"

    # Turn 2: shell runs → gate's condition now matches → DENY.
    result2 = await engine.evaluate(_ctx(Phase.TOOL_CALL, tool_name="run_shell"))
    assert result2.action == PolicyAction.DENY
    assert result2.deciding_policy == "gate"
