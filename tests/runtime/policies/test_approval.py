"""
Tests for :func:`_await_policy_approval` and the verdict
parser (Phase 8).

Ports these omniagents ``test_labels_and_policies.py`` cases:

- ``test_label_policy_ask_approve`` — approve round-trip
  applies set_labels
- ``test_label_policy_ask_handler_receives_tool_args`` —
  approval request carries the reason + preview (our shape
  is ``ApprovalRequest`` rather than raw tool_args)
- ``test_label_policy_ask_deny`` — refuse leaves no writes
- ``test_ask_timeout`` — timeout → refuse path
- ``test_ask_user_denies_not_timeout_message`` — refuse
  reason distinguishable from timeout
- ``test_no_handler_denies`` — missing verdict row → DENY

Plus Phase 8-specific coverage:

- Strict verdict parsing (only ``{"approved": true}``
  returns True)
- Per-policy ask_timeout override via
  ``result.deciding_policy`` lookup
- Labels apply on approve, NOT on refuse / timeout /
  malformed (load-bearing §7.2 invariant)
- Synthetic function_call shape matches spec (name =
  "request_approval", status = "action_required")
- Content preview truncated to 1024 chars
- Cancel-during-ASK semantics (via park returning None
  with cancelled status)
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_plane.runtime.policies.approval import (
    ApprovalRequest,
    _await_policy_approval,
    _parse_verdict,
    _truncate,
)
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.policies.label import LabelPolicy
from agent_plane.spec.types import (
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
    PolicyResult,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)

# ── Fixtures / helpers ────────────────────────────────


def _engine_with_policies(
    store: SqlAlchemyConversationStore,
    policies: list,
    ask_timeout: int = 30,
) -> PolicyEngine:
    """Build engine for tests that need `spec_for` to resolve."""
    conv = store.create_conversation()
    return PolicyEngine(
        policies=policies,
        label_defs={},
        ask_timeout=ask_timeout,
        conversation_id=conv.id,
        initial_labels={},
        conversation_store=store,
    )


def _ask_policy(
    name: str,
    *,
    ask_timeout: int | None = None,
    set_labels: dict[str, str] | None = None,
) -> LabelPolicy:
    """Build an ASKing LabelPolicy — the typical ASK source."""
    return LabelPolicy(
        LabelPolicySpec(
            name=name,
            on=[PhaseSelector(phase=Phase.INPUT)],
            ask_timeout=ask_timeout,
            action=PolicyAction.ASK,
            reason="review needed",
            set_labels=set_labels,
        ),
    )


def _composed_ask(
    *,
    deciding_policy: str,
    reason: str = "please approve",
    set_labels: dict[str, str] | None = None,
) -> PolicyResult:
    """Fabricate an engine-composed ASK result."""
    return PolicyResult(
        action=PolicyAction.ASK,
        reason=reason,
        set_labels=set_labels,
        deciding_policy=deciding_policy,
    )


class _Recorder:
    """
    Test recorder for the register / emit callbacks.

    Makes it trivial to assert on what the approval helper
    published without touching a real SSE stream or store.
    """

    def __init__(self) -> None:
        self.registered: list[tuple[str, str, str]] = []
        self.emitted: list[dict[str, Any]] = []

    def register(self, call_id: str, task_id: str, args_json: str) -> None:
        self.registered.append((call_id, task_id, args_json))

    def emit(self, event: dict[str, Any]) -> None:
        self.emitted.append(event)


def _approving_park(verdict: str) -> Any:
    """Park callback that instantly returns the given verdict string."""

    async def _park(call_id: str, timeout_s: int) -> str:
        return verdict

    return _park


def _timing_out_park() -> Any:
    """Park callback that always raises TimeoutError."""

    async def _park(call_id: str, timeout_s: int) -> str:
        raise TimeoutError(f"no verdict within {timeout_s}s")

    return _park


def _returns_none_park() -> Any:
    """Park callback that returns None — cancelled or missing row."""

    async def _park(call_id: str, timeout_s: int) -> str | None:
        return None

    return _park


# ── _parse_verdict ─────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"approved": true}', True),
        ('{"approved": false}', False),
        ('{"approved": "true"}', False),  # string not bool → reject
        ('{"approved": 1}', False),  # truthy int not bool → reject
        ("{}", False),
        ("not json", False),
        ("", False),
        (None, False),
        ('[{"approved": true}]', False),  # non-dict root
        ('{"something_else": true}', False),
    ],
)
def test_parse_verdict_strict(raw: str | None, expected: bool) -> None:
    """Strict verdict parser: only exact
    ``{"approved": true}`` returns True. Everything else
    (malformed, non-bool, non-dict, missing field, explicit
    false) returns False — fail-closed per POLICIES.md §13.
    If this regresses, rogue / malformed PATCH bodies could
    silently approve restricted operations."""
    assert _parse_verdict(raw) is expected


# ── _truncate ──────────────────────────────────────────


def test_truncate_short_passes() -> None:
    """Under-limit text returns unchanged."""
    assert _truncate("hi", limit=10) == "hi"


def test_truncate_long_clips_with_marker() -> None:
    """Over-limit text is clipped with an explicit marker
    so viewers can see truncation happened."""
    clipped = _truncate("x" * 100, limit=20)
    # First 20 chars of x, then the marker.
    assert clipped == "x" * 20 + " [truncated]"


# ── ApprovalRequest serialization ─────────────────────


def test_approval_request_serializes_all_fields() -> None:
    """Every field round-trips through JSON — the client's
    approval handler relies on this shape."""
    req = ApprovalRequest(
        phase="tool_call",
        reason="needs review",
        policy_name="confirm_shell",
        content_preview="ls -la",
    )
    import json

    data = json.loads(req.to_arguments_json())
    assert data == {
        "phase": "tool_call",
        "reason": "needs review",
        "policy_name": "confirm_shell",
        "content_preview": "ls -la",
    }


# ── _await_policy_approval — happy paths ──────────────


@pytest.mark.asyncio
async def test_approval_approve_applies_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents
    ``test_label_policy_ask_approve``. On approve, the
    ASK-accumulated set_labels reach the store."""
    policy = _ask_policy("gate", set_labels={"integrity": "0"})
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(
        deciding_policy="gate",
        set_labels={"integrity": "0"},
    )

    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="hello",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_approving_park('{"approved": true}'),
    )
    assert approved is True
    # Labels landed — both hot cache and persisted.
    assert engine.labels == {"integrity": "0"}
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {"integrity": "0"}


@pytest.mark.asyncio
async def test_approval_refuse_does_not_apply_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_label_policy_ask_deny``.
    On explicit refuse, labels are NOT applied — the
    load-bearing §7.2 invariant that a denied ASK leaves
    no side effects."""
    policy = _ask_policy("gate", set_labels={"integrity": "0"})
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(
        deciding_policy="gate",
        set_labels={"integrity": "0"},
    )

    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="hello",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_approving_park('{"approved": false}'),
    )
    assert approved is False
    # No labels landed — hot cache empty, store empty.
    assert engine.labels == {}
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {}


@pytest.mark.asyncio
async def test_approval_timeout_does_not_apply_labels(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_ask_timeout``. Park raises
    TimeoutError → helper returns False without applying
    labels."""
    policy = _ask_policy("gate", set_labels={"integrity": "0"})
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(
        deciding_policy="gate",
        set_labels={"integrity": "0"},
    )

    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="hello",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_timing_out_park(),
    )
    assert approved is False
    # Labels not applied on timeout.
    assert engine.labels == {}


@pytest.mark.asyncio
async def test_approval_missing_verdict_row_denies(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Ports omniagents ``test_no_handler_denies``. Park
    returns None (cancelled / missing row) → helper returns
    False. Covers the cancel-during-ASK path where the
    pending row was advanced to ``cancelled`` by the cancel
    handler (POLICIES.md §12)."""
    policy = _ask_policy("gate", set_labels={"integrity": "0"})
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="gate", set_labels={"integrity": "0"})

    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="hello",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_returns_none_park(),
    )
    assert approved is False
    assert engine.labels == {}


@pytest.mark.asyncio
async def test_approval_malformed_verdict_denies(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A PATCH body with garbage ``output`` → helper returns
    False. The route stays a dumb pipe; verdict parsing
    fail-closes here (POLICIES.md §13 malformed-verdict
    rule)."""
    policy = _ask_policy("gate")
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="gate")

    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="hello",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        # Garbage JSON → strict parser returns False.
        park=_approving_park("banana garbage"),
    )
    assert approved is False


# ── Register + emit payloads ──────────────────────────


@pytest.mark.asyncio
async def test_approval_registers_pending_row(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """The register callback receives the generated
    call_id, the task_id, and the arguments JSON. These
    three fields are what the PATCH route uses to route a
    verdict back to the parked workflow."""
    policy = _ask_policy("gate")
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="gate", reason="please")

    await _await_policy_approval(
        task_id="task_abc",
        root_task_id="task_abc",
        result=result,
        phase=Phase.TOOL_CALL,
        content_preview="ls -la",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_approving_park('{"approved": true}'),
    )
    # Exactly one row registered.
    assert len(recorder.registered) == 1
    call_id, task_id, args_json = recorder.registered[0]
    # Call_id has the standard prefix.
    assert call_id.startswith("call_")
    # Task_id matches the parked workflow.
    assert task_id == "task_abc"
    # Arguments carry all four ApprovalRequest fields.
    import json

    args = json.loads(args_json)
    assert args["phase"] == "tool_call"
    assert args["reason"] == "please"
    assert args["policy_name"] == "gate"
    assert args["content_preview"] == "ls -la"


@pytest.mark.asyncio
async def test_approval_emits_synthetic_function_call(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """The emit callback receives a
    ``response.output_item.done`` event with a
    function_call item named ``request_approval``. This is
    what the client's reserved-name handler dispatches on
    (POLICIES.md §7.1)."""
    policy = _ask_policy("gate")
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="gate")

    await _await_policy_approval(
        task_id="task_abc",
        root_task_id="task_abc",
        result=result,
        phase=Phase.INPUT,
        content_preview="x",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_approving_park('{"approved": true}'),
    )
    assert len(recorder.emitted) == 1
    event = recorder.emitted[0]
    assert event["type"] == "response.output_item.done"
    item = event["item"]
    assert item["type"] == "function_call"
    assert item["name"] == "request_approval"
    # status='action_required' mirrors client-side tool
    # tunneling — the client knows to pause and prompt.
    assert item["status"] == "action_required"
    # call_id is consistent between register and emit.
    registered_call_id = recorder.registered[0][0]
    assert item["call_id"] == registered_call_id


# ── Per-policy ask_timeout override ───────────────────


@pytest.mark.asyncio
async def test_per_policy_ask_timeout_override_wins(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """When the deciding policy has its own ask_timeout,
    that value is passed to the park callback — not the
    engine's default. Enables long-review policies (e.g.
    50 KB documents) without bumping the global default."""
    captured: dict[str, int] = {}

    async def _capturing_park(call_id: str, timeout_s: int) -> str:
        captured["timeout_s"] = timeout_s
        return '{"approved": true}'

    # Policy declares its own 300s timeout.
    policy = _ask_policy("long_review", ask_timeout=300)
    engine = _engine_with_policies(
        conversation_store,
        [policy],
        ask_timeout=30,
    )
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="long_review")

    await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="x",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_capturing_park,
    )
    # Per-policy 300 beat engine's 30.
    assert captured["timeout_s"] == 300


@pytest.mark.asyncio
async def test_engine_ask_timeout_default_when_no_override(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Without a per-policy override, the engine's spec-level
    default applies."""
    captured: dict[str, int] = {}

    async def _capturing_park(call_id: str, timeout_s: int) -> str:
        captured["timeout_s"] = timeout_s
        return '{"approved": true}'

    policy = _ask_policy("gate", ask_timeout=None)
    engine = _engine_with_policies(
        conversation_store,
        [policy],
        ask_timeout=45,
    )
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="gate")

    await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="x",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_capturing_park,
    )
    # Engine's 45 used because policy didn't override.
    assert captured["timeout_s"] == 45


@pytest.mark.asyncio
async def test_unknown_deciding_policy_falls_back_to_engine_timeout(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """If deciding_policy is set to a name the engine
    doesn't know (shouldn't happen in production but
    defensive), fallback to the engine default."""
    captured: dict[str, int] = {}

    async def _capturing_park(call_id: str, timeout_s: int) -> str:
        captured["timeout_s"] = timeout_s
        return '{"approved": true}'

    engine = _engine_with_policies(
        conversation_store,
        [],
        ask_timeout=60,
    )
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="nonexistent")

    await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="x",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_capturing_park,
    )
    # Unknown name → engine default.
    assert captured["timeout_s"] == 60


# ── Content preview truncation ────────────────────────


@pytest.mark.asyncio
async def test_content_preview_truncated_to_1024(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Long content previews are clipped so the UI is not
    swamped. 1024 is the chosen limit (POLICIES.md §7.2)."""
    policy = _ask_policy("gate")
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    result = _composed_ask(deciding_policy="gate")

    await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="A" * 2000,
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_approving_park('{"approved": true}'),
    )
    import json

    args = json.loads(recorder.registered[0][2])
    preview = args["content_preview"]
    # Exactly 1024 chars + " [truncated]" suffix.
    assert preview.startswith("A" * 1024)
    assert preview.endswith(" [truncated]")


# ── No set_labels on result ───────────────────────────


@pytest.mark.asyncio
async def test_approve_with_no_set_labels_is_noop(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """An ASK result carrying no set_labels (empty/None) on
    approve does not touch the store — no pointless empty
    apply_label_writes call."""
    policy = _ask_policy("gate")
    engine = _engine_with_policies(conversation_store, [policy])
    recorder = _Recorder()
    # Result has no set_labels — a policy that just wants
    # approval without writing state.
    result = _composed_ask(deciding_policy="gate", set_labels=None)

    approved = await _await_policy_approval(
        task_id="task_1",
        root_task_id="task_1",
        result=result,
        phase=Phase.INPUT,
        content_preview="x",
        policy_engine=engine,
        register=recorder.register,
        emit=recorder.emit,
        park=_approving_park('{"approved": true}'),
    )
    assert approved is True
    # Store unchanged — no spurious empty writes.
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {}
