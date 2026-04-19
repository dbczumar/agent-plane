"""
Tests for :func:`build_policy_engine` (Phase 2).

Covers:

- Zero-guardrails path: ``spec.guardrails is None`` → no-op
  engine with empty policies and labels.
- Empty-guardrails path: ``guardrails: {}`` → no-op engine
  but with spec-declared ask_timeout.
- Declared policies round-trip from spec to engine in YAML
  order.
- Initial label seeding via UPSERT: writes only for keys
  missing from the persisted state, idempotent across two
  successive builds.
- Hot cache is built from the post-seed snapshot (not the
  pre-seed read).
- Existing labels are NOT clobbered when the spec's initial
  differs from the persisted value.
"""

from __future__ import annotations

from pathlib import Path

from agent_plane.runtime.policies.builder import build_policy_engine
from agent_plane.spec.parser import parse
from agent_plane.spec.types import (
    AgentSpec,
    GuardrailsSpec,
    LabelDef,
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _write_spec(
    tmp_path: Path,
    config_yaml: str,
) -> Path:
    """Write a config.yaml to a fresh agent-dir fixture."""
    (tmp_path / "config.yaml").write_text(config_yaml)
    return tmp_path


# ── Zero-guardrails (engine stays alive but is a no-op) ─


def test_build_without_guardrails_returns_noop_engine(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A spec with no `guardrails:` block still builds an
    engine. The enforcement sites (Phase 5+) call through it
    unconditionally — if this raised, we'd have to guard
    every call site with `if engine is not None`, which
    POLICIES.md §10 explicitly avoids."""
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: no-guardrails
""",
    )
    spec = parse(agent_dir)
    assert spec.guardrails is None

    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    # Every observable field on the engine matches the "no
    # guardrails declared" state.
    assert engine.policies == []
    assert engine.label_defs == {}
    assert engine.labels == {}


def test_build_with_empty_guardrails_block(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """`guardrails: {}` explicitly declared — engine has no
    policies, no labels, default ask_timeout. Distinguishable
    from the None case only in that ask_timeout is present,
    but functionally identical for evaluate()."""
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: empty-guardrails
guardrails: {}
""",
    )
    spec = parse(agent_dir)
    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    assert engine.policies == []
    assert engine.label_defs == {}
    assert engine.labels == {}


# ── Declared policies + label seeding ──────────────────


def test_build_propagates_declared_policies_in_yaml_order(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Policies land on the engine in their YAML declaration
    order. The engine's evaluate() loop (Phase 3+) depends on
    this for DENY short-circuit semantics and first-ASK
    selection."""
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: ordered
guardrails:
  policies:
    alpha:
      type: label
      on: [input]
      action: allow
    bravo:
      type: label
      on: [input]
      action: allow
    charlie:
      type: label
      on: [input]
      action: allow
""",
    )
    spec = parse(agent_dir)
    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    # Names in YAML order — regression would reorder alphabet
    # or reverse direction.
    assert [p.name for p in engine.policies] == ["alpha", "bravo", "charlie"]


def test_build_seeds_initial_labels(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """`LabelDef.initial` values with no persisted row get
    written through set_labels. Verified via the store's
    round-trip."""
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: seeded
guardrails:
  labels:
    integrity: "1"
    sensitivity:
      initial: public
      values: [public, internal, confidential]
""",
    )
    spec = parse(agent_dir)
    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    # Hot cache reflects the seeded values.
    assert engine.labels == {"integrity": "1", "sensitivity": "public"}
    # Persisted too — not just in memory.
    conv_refetched = conversation_store.get_conversation(conv.id)
    assert conv_refetched is not None
    assert conv_refetched.labels == {"integrity": "1", "sensitivity": "public"}


def test_build_skips_labels_without_initial(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Labels declared with no `initial` (unset-until-written
    pattern) do not produce seed rows. Without this,
    policies that gate on "label absent" would incorrectly
    fire after build."""
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: partial
guardrails:
  labels:
    has_initial: "1"
    no_initial:
      values: [a, b]
""",
    )
    spec = parse(agent_dir)
    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    # Only `has_initial` lands; `no_initial` is absent.
    assert engine.labels == {"has_initial": "1"}


def test_build_is_idempotent_on_existing_labels(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Building twice on the same conversation does not
    overwrite existing labels — the ON-CONFLICT-DO-NOTHING
    semantic per POLICIES.md §10. A policy may have written
    a value; a second workflow build must not revert it.

    If this regresses, the seeding path is doing UPSERT-always
    instead of UPSERT-if-missing, and ongoing label state
    would reset every time a workflow starts.
    """
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: idempotent
guardrails:
  labels:
    integrity: "1"
""",
    )
    spec = parse(agent_dir)
    conv = conversation_store.create_conversation()

    # First build: seeds integrity="1" as declared.
    first = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    assert first.labels == {"integrity": "1"}

    # Simulate a policy writing integrity="0" mid-conversation.
    first.apply_label_writes({"integrity": "0"})
    conv_after_policy = conversation_store.get_conversation(conv.id)
    assert conv_after_policy is not None
    assert conv_after_policy.labels == {"integrity": "0"}

    # Second build: MUST NOT revert integrity to "1" —
    # the declared initial is an "if missing" seed, not an
    # "every build" reset.
    second = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    # If this reads "1", the seeding clobbered the policy's
    # write — a serious IFC safety bug (taint would silently
    # reset to clean on any workflow restart).
    assert second.labels == {"integrity": "0"}


def test_build_preserves_ask_timeout_override(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Spec-level `ask_timeout` overrides the default on the
    engine. Later phases read this for ASK routing."""
    agent_dir = _write_spec(
        tmp_path,
        """
spec_version: 1
name: long-review
guardrails:
  ask_timeout: 600
""",
    )
    spec = parse(agent_dir)
    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    assert engine.ask_timeout == 600


# ── Programmatic API (non-YAML) parity ─────────────────


def test_build_from_programmatic_spec(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Building from an in-memory AgentSpec works too —
    tests that don't want to round-trip through YAML should
    be able to construct an engine directly. Critical for
    Phase 3+ unit tests that build fine-grained specs."""
    spec = AgentSpec(
        spec_version=1,
        name="programmatic",
        guardrails=GuardrailsSpec(
            labels={"integrity": LabelDef(initial="1")},
            policies=[
                LabelPolicySpec(
                    name="taint_web",
                    on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="web")],
                    action=PolicyAction.ALLOW,
                    set_labels={"integrity": "0"},
                ),
            ],
            ask_timeout=45,
        ),
    )
    conv = conversation_store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=conversation_store,
    )
    assert engine.ask_timeout == 45
    assert len(engine.policies) == 1
    assert engine.policies[0].name == "taint_web"
    assert engine.labels == {"integrity": "1"}
