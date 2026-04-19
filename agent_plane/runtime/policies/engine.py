"""
``PolicyEngine`` — per-workflow owner of policies + label state.

The engine is a plain local constructed at the top of
``_run_agent_loop`` and passed explicitly to the enforcement
sites. No ContextVar, no container class (see POLICIES.md §4
for the rationale).

Phase 2 scope: skeleton only. ``evaluate(ctx)`` returns ALLOW
with empty set_labels for every call because no policy
subclasses are registered yet. Phases 3/4/7 layer in the
concrete policy types; this class's shape does not need to
change when they do.
"""

from __future__ import annotations

from typing import Any

from agent_plane.spec.types import (
    DEFAULT_ASK_TIMEOUT,
    EvaluationContext,
    LabelDef,
    PolicyAction,
    PolicyResult,
    PolicySpec,
)
from agent_plane.stores.conversation_store import ConversationStore


class PolicyEngine:
    """
    Owns policies + label state for one workflow execution.

    Constructed once at the top of ``_run_agent_loop`` via
    :func:`build_policy_engine` and passed explicitly to the
    four enforcement sites (§5). Labels are hot-cached on the
    engine for the life of the workflow and written through to
    ``conversation_labels`` via the conversation store on every
    ``apply_label_writes`` call.

    :param policies: Per-workflow policy instances in YAML
        declaration order. Phase 2 accepts ``list[PolicySpec]``
        directly because concrete Policy subclasses don't
        exist yet; later phases swap this for instantiated
        Policy objects.
    :param label_defs: Per-key ``LabelDef`` schemas from the
        agent spec. Used by ``apply_label_writes`` to validate
        ``values`` + ``monotonic`` constraints. Empty dict
        when no labels were declared.
    :param ask_timeout: Spec-wide default approval timeout in
        seconds (POLICIES.md §7). Per-policy overrides live on
        :class:`PolicySpec` and are looked up via
        :meth:`spec_for`.
    :param conversation_id: The conversation this engine owns
        label state for.
    :param initial_labels: Labels already persisted for the
        conversation at workflow-start (the hot cache seed).
    :param conversation_store: Write-through target for label
        mutations. Held by reference so every
        ``apply_label_writes`` call goes to the same backing
        store as the spec-declared conversation.
    """

    def __init__(
        self,
        *,
        policies: list[PolicySpec],
        label_defs: dict[str, LabelDef],
        ask_timeout: int,
        conversation_id: str,
        initial_labels: dict[str, str],
        conversation_store: ConversationStore,
    ) -> None:
        self.policies = policies
        self.label_defs = label_defs
        self.ask_timeout = ask_timeout
        self._conversation_id = conversation_id
        self._labels = dict(initial_labels)
        self._store = conversation_store

    @property
    def labels(self) -> dict[str, str]:
        """
        Read-only snapshot of the hot label cache.

        Returns a defensive copy so callers that mutate the
        dict do not corrupt engine state. Policies read labels
        through the ``context`` passed into their ``evaluate``
        method; this property is for introspection (tests, UI).

        :returns: Mapping from label key to value.
        """
        return dict(self._labels)

    @property
    def conversation_id(self) -> str:
        """:returns: The conversation this engine owns."""
        return self._conversation_id

    async def evaluate(self, ctx: EvaluationContext) -> PolicyResult:
        """
        Evaluate the composed policy decision for one phase.

        Phase 2 behavior: with zero concrete policy subclasses
        wired in, every call returns ``ALLOW`` with an empty
        ``set_labels``. The evaluate loop, action validation,
        and composition arrive in Phases 3–7.

        :param ctx: The current evaluation context
            (phase + content + resolved tool_name).
        :returns: A composed :class:`PolicyResult`. Phase 2
            always returns ALLOW; later phases return the real
            composed decision.
        """
        # Intentional no-op for Phase 2 — the four enforcement
        # sites (wired in Phase 5+) still call evaluate() and
        # branch on the result, so returning ALLOW keeps the
        # contract stable across phases.
        _ = ctx
        return PolicyResult(
            action=PolicyAction.ALLOW,
            reason=None,
            set_labels=None,
            deciding_policy=None,
        )

    def apply_label_writes(self, set_labels: dict[str, str]) -> None:
        """
        Validate and persist label writes.

        Phase 2 behavior: writes pass through to the store
        unchanged. Schema validation (``values`` /
        ``monotonic`` per :class:`LabelDef`) is enforced in
        Phase 3 when ``LabelPolicy`` lands — the separate path
        keeps Phase 2 shippable on its own.

        :param set_labels: Mapping of label key to value. No-op
            on empty dict. Writes update both the hot cache on
            this engine and the persistent row in
            ``conversation_labels`` in a single UPSERT
            transaction (POLICIES.md §6.3).
        """
        if not set_labels:
            return
        self._store.set_labels(self._conversation_id, set_labels)
        self._labels.update(set_labels)

    def spec_for(self, policy_name: str | None) -> PolicySpec | None:
        """
        Look up a :class:`PolicySpec` by name.

        Used by ``_await_policy_approval`` (Phase 8) to resolve
        the per-policy ``ask_timeout`` override off the
        deciding policy's spec. ``None`` input returns ``None``
        to keep the caller's null-handling terse.

        :param policy_name: Name of the policy to look up,
            e.g. ``"block_canada_input"``. ``None`` returns
            ``None`` directly.
        :returns: The matching spec, or ``None`` when no policy
            with that name exists (or *policy_name* was
            ``None``).
        """
        if policy_name is None:
            return None
        for spec in self.policies:
            if spec.name == policy_name:
                return spec
        return None

    def _context(self) -> dict[str, Any]:
        """
        Build the context bundle passed to each Policy.evaluate().

        Exposes a read-only snapshot of the hot label cache
        plus identity fields. Used by Phase 3+ when concrete
        Policy subclasses need to inspect labels for
        condition evaluation. Phase 2 never calls this because
        ``evaluate`` returns early; it's defined here so the
        API is stable across phases.

        :returns: Context dict with keys ``labels``
            (defensive copy) and ``conversation_id``.
        """
        return {
            "labels": dict(self._labels),
            "conversation_id": self._conversation_id,
        }


# Re-export the defaults for callers that need them without
# importing from spec.types directly.
__all__ = ["PolicyEngine", "DEFAULT_ASK_TIMEOUT"]
