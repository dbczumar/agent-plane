"""
:class:`LabelPolicy` — YAML-declarative policy with no Python
or LLM callable.

When the selector + condition gates match the current
context, emits the fixed action and ``set_labels`` declared
in the spec. See POLICIES.md §9.3 for the runtime model.
"""

from __future__ import annotations

from typing import Any

from agent_plane.policies.base import Policy
from agent_plane.policies.types import EvaluationContext, PolicyResult
from agent_plane.spec.types import LabelPolicySpec


class LabelPolicy(Policy):
    """
    A policy driven entirely by its YAML declaration.

    Fires :attr:`spec.action` (one of ``ALLOW``, ``ASK``,
    ``DENY``) and emits :attr:`spec.set_labels` (a fixed
    ``dict[str, str]``) whenever the engine dispatches to it.
    No branching on content, no inspection of labels — those
    concerns belong to :class:`FunctionPolicy` and
    :class:`PromptPolicy`.

    :param spec: The :class:`LabelPolicySpec` this policy
        was built from.
    """

    spec: LabelPolicySpec

    def __init__(self, spec: LabelPolicySpec) -> None:
        """
        Bind the spec to this runtime instance.

        :param spec: Declarative spec with pre-validated
            ``action`` and ``set_labels`` (the parser
            rejects malformed combinations at spec load).
        """
        self.spec = spec

    async def evaluate(
        self,
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> PolicyResult:
        """
        Emit the declared action + set_labels.

        Phase + tool-name filtering is already done by the
        engine's PhaseSelector pass; condition gating also
        done by the engine before dispatch. If ``evaluate``
        is being called, the policy has been told to fire.

        :param ctx: Current evaluation context (unused —
            LabelPolicy's decision is content-independent).
        :param context: Engine-provided context bundle
            (unused — see ``ctx``).
        :returns: ``PolicyResult(action=spec.action,
            reason=spec.reason,
            set_labels=copy of spec.set_labels)``.
            ``deciding_policy`` is left ``None``; the engine
            sets it on the composed result.
        """
        # Explicitly reference both arguments so static
        # analysis does not complain about unused parameters.
        # The semantics of LabelPolicy are to ignore content
        # and labels — the decision is driven entirely by the
        # spec declaration.
        del ctx, context
        # The action on the spec may be None if the parser
        # built an incomplete instance. That would be a
        # parser bug — fail loud with a clear message rather
        # than silently emitting some default.
        if self.spec.action is None:
            raise ValueError(
                f"LabelPolicy {self.spec.name!r} has no action declared; "
                f"parser must never build a LabelPolicySpec without one.",
            )
        return PolicyResult(
            action=self.spec.action,
            reason=self.spec.reason,
            # Defensive copy so a policy whose decision
            # accumulates with others cannot be mutated by
            # the engine composition step.
            set_labels=dict(self.spec.set_labels) if self.spec.set_labels else None,
        )
