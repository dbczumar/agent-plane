"""
Runtime evaluation contracts for the policy system.

These are the shapes that cross the ``Policy.evaluate()``
boundary and the engine-to-approval-helper boundary. They are
NOT spec types — they do not appear in any config.yaml the user
writes. Spec types (what the parser consumes and emits) live in
:mod:`agent_plane.spec.types`; runtime evaluation types live
here.

Three types live in this module:

- :class:`EvaluationContext` — what the caller hands to the
  engine on each enforcement call (phase + content +
  resolved tool_name).
- :class:`PolicyResult` — what a single policy returns and what
  the engine composes across policies.
- :class:`ApprovalRequest` — the wire contract for the synthetic
  ``request_approval`` function_call args that carry an ASK
  to the client's approval handler.

Agent-author Python callables import :class:`EvaluationContext`
and :class:`PolicyResult` from here (or from the
:mod:`agent_plane.policies` package entry point).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_plane.spec.types import Phase, PolicyAction


@dataclass(frozen=True)
class EvaluationContext:
    """
    Everything the engine needs to evaluate one phase.

    Filled by the caller (workflow or executor hook) BEFORE
    calling ``engine.evaluate(ctx)``. The engine never has to
    introspect ``content`` to answer "which tool was this?" —
    the caller resolves ``tool_name`` because only it has the
    local state to do so cheaply (on ``TOOL_RESULT`` the
    ``function_call_output`` payload carries ``call_id`` but no
    ``name``; the caller knows the name from the earlier
    dispatch).

    :param phase: The enforcement point.
    :param content: Phase-specific payload — shape depends on
        ``phase``: ``INPUT`` / ``OUTPUT`` carry ``str`` (raw
        user / assistant text); ``TOOL_CALL`` / ``TOOL_RESULT``
        carry ``dict[str, Any]`` (the function_call or
        function_call_output dict, which includes
        ``call_id``). Policies know which shape to expect from
        their declared ``on:`` phases — the engine never
        introspects this field itself.
    :param tool_name: Resolved tool name. Populated on
        ``TOOL_CALL`` and ``TOOL_RESULT``; ``None`` on
        ``INPUT`` and ``OUTPUT``.
    """

    phase: Phase
    content: Any
    tool_name: str | None = None


@dataclass(frozen=True)
class PolicyResult:
    """
    One policy's decision (or the engine's composed decision).

    Returned by ``Policy.evaluate()`` and by
    ``PolicyEngine.evaluate()``. The same shape is used at
    both layers: individual policies return a single-policy
    decision, the engine composes them and returns the
    aggregate.

    :param action: The decision (``ALLOW``, ``ASK``, or
        ``DENY``), e.g. ``PolicyAction.DENY``.
    :param reason: Human-readable reason string. Shown to the
        user on ASK, included in logs / spans on DENY, ``None``
        on ALLOW, e.g. ``"Canada-related topics are denied."``.
    :param set_labels: Labels the policy wants to write. For
        a single-policy result: the raw writes the policy
        requested (before whitelist filtering). For an
        engine-composed result: the writes the engine has
        accumulated and intends to apply on this decision
        (filtering already done). ``None`` when the policy
        wrote no labels, e.g. ``{"integrity": "0"}``.
    :param deciding_policy: Name of the policy whose action
        drove the composed result. Engine-set only —
        single-policy results leave it ``None``. On DENY: the
        first short-circuiting policy. On ASK: the first
        ASKing policy in YAML order. On ALLOW: ``None``.
        Powers the ``deciding_policy`` outer-span attribute
        (POLICIES.md §11.5) and the per-policy ``ask_timeout``
        lookup (§7.2).
    """

    action: PolicyAction
    reason: str | None = None
    set_labels: dict[str, str] | None = None
    deciding_policy: str | None = None


@dataclass(frozen=True)
class ApprovalRequest:
    """
    The payload surfaced to the client's approval handler.

    Mirrors the arguments JSON the synthetic
    ``request_approval`` function_call carries. Kept as a
    dataclass (not an ad-hoc dict) so the contract is
    explicit at the seam between workflow and approval
    helper, and so tests assert against named fields
    instead of dict keys.

    :param phase: Which enforcement point produced the ASK,
        e.g. ``"tool_call"``. String form for JSON-friendliness
        (the wire format is opaque to middleware).
    :param reason: Combined reason string from all ASKing
        policies, joined with ``"; "`` per §4. Shown to the
        user in the approval UI.
    :param policy_name: Name of the deciding (first-in-YAML-
        order) ASKing policy. Drives per-policy ask_timeout
        lookup and observability.
    :param content_preview: Truncated snapshot of the content
        being gated. Lets a human reviewer see what they're
        approving without overwhelming the UI on a 50 KB
        payload.
    """

    phase: str
    reason: str
    policy_name: str
    content_preview: str

    def to_arguments_json(self) -> str:
        """
        Serialize to the arguments-string format the client
        sees on the synthetic function_call.

        :returns: JSON string — the ``arguments`` field of
            the emitted function_call item.
        """
        return json.dumps(
            {
                "phase": self.phase,
                "reason": self.reason,
                "policy_name": self.policy_name,
                "content_preview": self.content_preview,
            }
        )


__all__ = [
    "ApprovalRequest",
    "EvaluationContext",
    "PolicyResult",
]
