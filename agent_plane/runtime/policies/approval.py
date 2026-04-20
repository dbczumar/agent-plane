"""
ASK flow — ``_await_policy_approval`` helper.

When the engine composes an ASK result, the caller (the
workflow enforcement site) hands the result + identity to
this helper. The helper:

1. Registers a ``request_approval`` row in
   ``pending_tool_calls`` (same table client-side tool calls
   use).
2. Emits a synthetic ``function_call`` SSE event on the root
   task's stream — the client's reserved-name handler
   renders an approval UI.
3. Parks on the existing ``tool_result`` DBOS topic,
   respecting the per-policy / spec-level ASK timeout.
4. On wake, parses the verdict strictly: only
   ``{"approved": true}`` returns True. Anything else —
   malformed JSON, missing field, wrong type, refuse,
   timeout — returns False (caller maps to DENY).
5. Applies the ASK-accumulated ``set_labels`` **only on
   approve** (POLICIES.md §7.2 invariant: a denied / timed-out
   ASK leaves no side effects).

Phase 8 scope: helper + verdict-parsing logic + tests with
injectable park/wake stubs. The production wiring of the
DBOS parking mechanism + SSE publishing lands in Phase 6
when the workflow integration picks up the approval path.
See POLICIES.md §7, §13.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agent_plane.policies.types import ApprovalRequest, PolicyResult
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.spec.types import Phase

# Parking callback contract — the workflow will bind a real
# DBOS-parking implementation (`dbos_recv_async(topic="tool_result")`
# with the per-policy timeout) at Phase 6 wiring time. Tests
# inject canned awaitables that return a verdict string or
# raise TimeoutError.
_ParkCallback = Callable[[str, int], Awaitable[str | None]]

# SSE publisher contract — the workflow binds a real
# `_write_output` variant that emits the synthetic
# function_call to the root task's stream. Tests can pass a
# no-op or a recorder.
_EmitCallback = Callable[[dict[str, Any]], None]

# Pending-call persister contract — registers the synthetic
# call_id ↔ task_id ↔ tool_name mapping so the PATCH route
# can wake this workflow when the verdict arrives. Tests
# typically pass a no-op.
_RegisterCallback = Callable[[str, str, str], None]


async def _await_policy_approval(
    *,
    task_id: str,
    root_task_id: str,
    result: PolicyResult,
    phase: Phase,
    content_preview: str,
    policy_engine: PolicyEngine,
    register: _RegisterCallback,
    emit: _EmitCallback,
    park: _ParkCallback,
) -> bool:
    """
    Drive one ASK round-trip, return True if approved.

    Wired into the workflow enforcement sites at Phase 6. The
    three callback parameters (``register``, ``emit``,
    ``park``) are the seams that let the helper work in tests
    without requiring a full DBOS + SSE stack.

    On approve: applies the ASK-accumulated ``set_labels``
    from the engine's composed result. On refuse / timeout /
    malformed verdict: returns ``False`` and applies nothing,
    preserving POLICIES.md §7.2 "no side effects on denied
    ASK".

    :param task_id: The sub-agent's task ID (the parked
        workflow).
    :param root_task_id: The root task whose SSE stream
        receives the synthetic function_call.
    :param result: Composed :class:`PolicyResult` — carries
        the combined reason, deciding_policy, and withheld
        set_labels.
    :param phase: Which enforcement point produced the ASK.
    :param content_preview: Truncated content snapshot for
        the UI.
    :param policy_engine: Engine — used to resolve the
        per-policy ``ask_timeout`` override off the deciding
        policy's spec, and to apply label writes on approve.
    :param register: Seam: register the pending call row.
    :param emit: Seam: publish the synthetic function_call
        on the root task's stream.
    :param park: Seam: block until the PATCH route delivers
        a verdict or the timeout elapses. Raises on timeout.
    :returns: ``True`` when the verdict is exactly
        ``{"approved": true}``; ``False`` otherwise.
    """
    call_id = f"call_{uuid.uuid4().hex}"
    approval_request = ApprovalRequest(
        phase=phase.value,
        reason=result.reason or "",
        policy_name=result.deciding_policy or "",
        content_preview=_truncate(content_preview, limit=1024),
    )
    args_json = approval_request.to_arguments_json()
    register(call_id, task_id, args_json)
    emit(_synthetic_function_call(call_id, args_json))

    effective_timeout = _resolve_ask_timeout(policy_engine, result)
    try:
        raw_verdict = await park(call_id, effective_timeout)
    except TimeoutError:
        return False

    approved = _parse_verdict(raw_verdict)
    if approved and result.set_labels:
        # POLICIES.md §7.2: writes accumulated by ASKing
        # policies land only on approve. On refuse / timeout /
        # malformed verdict we drop them — a denied ASK must
        # leave no trace.
        policy_engine.apply_label_writes(result.set_labels)
    return approved


def _synthetic_function_call(call_id: str, args_json: str) -> dict[str, Any]:
    """
    Build the SSE ``response.output_item.done`` payload for
    the synthetic request_approval function_call.

    The shape matches what a real client-side tool call
    looks like — recognized by the reserved name
    (``request_approval``) rather than any new event type
    (POLICIES.md §7.1).

    :param call_id: Generated call_id for this approval.
    :param args_json: Already-serialized ApprovalRequest.
    :returns: Dict the workflow emits onto the root task's
        stream verbatim.
    """
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": "request_approval",
            "status": "action_required",
            "arguments": args_json,
        },
    }


def _resolve_ask_timeout(
    engine: PolicyEngine,
    result: PolicyResult,
) -> int:
    """
    Pick the effective timeout for this ASK.

    Per-policy override wins over the spec-level default:
    ``result.deciding_policy`` names the first ASKing
    policy in YAML order; we read its spec's
    ``ask_timeout`` field and fall back to the engine's
    spec-level ``ask_timeout`` if absent.

    :param engine: The workflow's engine.
    :param result: Composed ASK result — carries
        deciding_policy.
    :returns: Timeout in seconds. Always > 0 (spec-load
        rejects ``<= 0`` at parse time).
    """
    deciding_spec = engine.spec_for(result.deciding_policy)
    if deciding_spec is not None and deciding_spec.ask_timeout is not None:
        return deciding_spec.ask_timeout
    return engine.ask_timeout


def _parse_verdict(raw: str | None) -> bool:
    """
    Strict verdict parser — returns True ONLY for
    ``{"approved": true}`` (exact bool True).

    Fail-closed rule from POLICIES.md §13: anything else
    (missing field, wrong type, unparseable JSON, non-dict
    root, explicit ``false``) returns False. The PATCH
    route stays a dumb pipe — all verdict semantics live
    here, which keeps the server route generic.

    :param raw: The verdict string delivered via the park
        callback. ``None`` when no row was present on wake
        (race with cancel or malformed PATCH). Also returns
        False.
    :returns: ``True`` only on exact ``{"approved": true}``.
    """
    if raw is None:
        return False
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    # `is True` is deliberate — rejects truthy-but-not-True
    # values (1, "yes", etc.) so the wire contract stays
    # strict.
    return parsed.get("approved") is True


def _truncate(text: str, *, limit: int) -> str:
    """
    Truncate content for the approval UI preview.

    :param text: Raw content string.
    :param limit: Maximum characters. 1024 keeps the UI
        readable without overwhelming a paginated viewer.
    :returns: Truncated string with a ``" [truncated]"``
        marker appended when clipping occurred.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + " [truncated]"


__all__ = ["_await_policy_approval"]
