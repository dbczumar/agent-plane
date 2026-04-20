"""
Tests for :meth:`PolicyEngine.apply_label_writes` schema
validation (POLICIES.md §10 / §13).

Silent-drop semantics:

- Key not in ``LabelDef.values`` → dropped.
- Key violates ``monotonic`` direction → dropped.
- Unknown key (no LabelDef) → set freely.
- Valid write → persisted via the store.

The drop path is silent by design (matches omniagents) —
a runtime validation failure does NOT raise. The surviving
writes still land atomically.

Ports omniagents ``test_labels_and_policies.py``:
- test_engine_enforces_root_label_schema_monotonicity
- test_invalid_initial_label_value_rejected_by_schema
  (handled at spec-load in parser tests — this file covers
  the runtime post-seed write path)
"""

from __future__ import annotations

from agent_plane.runtime.policies.engine import (
    PolicyEngine,
    _monotonic_ok,
)
from agent_plane.spec.types import LabelDef
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)

# ── _monotonic_ok unit tests ──────────────────────────


def test_monotonic_ok_unset_label_allows_any_value() -> None:
    """Seeding from None passes — nothing to compare yet."""
    ldef = LabelDef(values=["0", "1"], monotonic="increasing")
    assert _monotonic_ok(ldef, None, "0") is True
    assert _monotonic_ok(ldef, None, "1") is True


def test_monotonic_ok_increasing_accepts_equal_or_greater() -> None:
    """Increasing: new index >= current index."""
    ldef = LabelDef(values=["0", "1", "2"], monotonic="increasing")
    # 0 → 1 ok
    assert _monotonic_ok(ldef, "0", "1") is True
    # 0 → 0 ok (equal)
    assert _monotonic_ok(ldef, "0", "0") is True
    # 2 → 0 rejected (decrease)
    assert _monotonic_ok(ldef, "2", "0") is False
    # 1 → 0 rejected
    assert _monotonic_ok(ldef, "1", "0") is False


def test_monotonic_ok_decreasing_accepts_equal_or_less() -> None:
    """Decreasing: new index <= current index."""
    ldef = LabelDef(values=["0", "1"], monotonic="decreasing")
    # 1 → 0 ok
    assert _monotonic_ok(ldef, "1", "0") is True
    # 1 → 1 ok
    assert _monotonic_ok(ldef, "1", "1") is True
    # 0 → 1 rejected
    assert _monotonic_ok(ldef, "0", "1") is False


# ── Engine-level filtering ────────────────────────────


def _build_engine_with_defs(
    store: SqlAlchemyConversationStore,
    label_defs: dict[str, LabelDef],
    *,
    initial_labels: dict[str, str] | None = None,
) -> PolicyEngine:
    """Build an engine with specific label_defs."""
    conv = store.create_conversation()
    return PolicyEngine(
        policies=[],
        label_defs=label_defs,
        ask_timeout=30,
        conversation_id=conv.id,
        initial_labels=initial_labels or {},
        conversation_store=store,
    )


def test_apply_label_writes_drops_value_outside_enum(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A value not in ``LabelDef.values`` is silently
    dropped. Prevents a policy (or a PromptPolicy
    classifier) from injecting an arbitrary string into an
    enumerated label."""
    engine = _build_engine_with_defs(
        conversation_store,
        {"integrity": LabelDef(values=["0", "1"])},
    )
    # "2" is not in values → dropped. "integrity": "1" is
    # valid → lands.
    engine.apply_label_writes({"integrity": "1", "other": "x"})
    # Hot cache has the valid write + the unknown-key
    # write (unknown keys pass through per POLICIES.md §10
    # schemaless-set-freely rule).
    assert engine.labels == {"integrity": "1", "other": "x"}

    # Now try to set an out-of-enum value.
    engine.apply_label_writes({"integrity": "2"})
    # Dropped — cache still shows "1".
    assert engine.labels["integrity"] == "1"


def test_apply_label_writes_drops_monotonic_violation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Violating monotonic direction → silent drop. The
    taint-clearing safety property for IFC: once integrity
    drops to "0" (decreasing monotonic), attempts to set it
    back to "1" are rejected.

    If this regresses, a malicious or broken policy could
    silently untaint the session — defeating the entire
    IFC design."""
    engine = _build_engine_with_defs(
        conversation_store,
        {"integrity": LabelDef(values=["0", "1"], monotonic="decreasing")},
        initial_labels={"integrity": "1"},
    )
    # Legal: 1 → 0 (decreasing allowed).
    engine.apply_label_writes({"integrity": "0"})
    assert engine.labels["integrity"] == "0"

    # Illegal: 0 → 1 (attempted INCREASE on decreasing
    # monotonic). Dropped.
    engine.apply_label_writes({"integrity": "1"})
    # Still "0" — the "1" attempt was rejected.
    assert engine.labels["integrity"] == "0"

    # Persisted state reflects the drop.
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels["integrity"] == "0"


def test_apply_label_writes_drops_increasing_violation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Symmetric case: increasing monotonic rejects
    decreases."""
    engine = _build_engine_with_defs(
        conversation_store,
        {
            "sensitivity": LabelDef(
                values=["public", "internal", "confidential"],
                monotonic="increasing",
            ),
        },
        initial_labels={"sensitivity": "internal"},
    )
    # internal → confidential: increase → allowed.
    engine.apply_label_writes({"sensitivity": "confidential"})
    assert engine.labels["sensitivity"] == "confidential"

    # confidential → public: decrease → dropped.
    engine.apply_label_writes({"sensitivity": "public"})
    # Still confidential.
    assert engine.labels["sensitivity"] == "confidential"


def test_apply_label_writes_partial_batch_survives(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """One key in a multi-key batch violates the schema;
    OTHER keys still land. Silent-drop is per-key, not
    all-or-nothing."""
    engine = _build_engine_with_defs(
        conversation_store,
        {
            "integrity": LabelDef(values=["0", "1"], monotonic="decreasing"),
            "other": LabelDef(values=["a", "b"]),
        },
        initial_labels={"integrity": "0"},
    )
    # integrity 0→1 violates decreasing (drop); other
    # "a" is valid (land).
    engine.apply_label_writes({"integrity": "1", "other": "a"})
    # Only `other` landed; integrity unchanged.
    assert engine.labels == {"integrity": "0", "other": "a"}


def test_apply_label_writes_schemaless_keys_pass_freely(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Keys with no LabelDef are set freely — the
    omniagents-parity behavior that lets policies write
    ad-hoc labels without declaring a schema first
    (POLICIES.md §10)."""
    engine = _build_engine_with_defs(
        conversation_store,
        {},  # no label_defs at all
    )
    engine.apply_label_writes({"any": "value", "anything": "123"})
    # Both landed — no schema to enforce.
    assert engine.labels == {"any": "value", "anything": "123"}


def test_apply_label_writes_values_only_no_monotonic(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """`values` declared without `monotonic` → enum check
    only, transitions between declared values are free."""
    engine = _build_engine_with_defs(
        conversation_store,
        {"role": LabelDef(values=["admin", "user", "guest"])},
        initial_labels={"role": "user"},
    )
    # Free transitions within the enum.
    engine.apply_label_writes({"role": "admin"})
    assert engine.labels["role"] == "admin"
    engine.apply_label_writes({"role": "guest"})
    assert engine.labels["role"] == "guest"
    # Out-of-enum still rejected.
    engine.apply_label_writes({"role": "root"})
    assert engine.labels["role"] == "guest"
