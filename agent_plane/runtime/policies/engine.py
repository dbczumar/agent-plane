"""
``PolicyEngine`` — per-workflow owner of policies + label state.

The engine is a plain local constructed at the top of
``_run_agent_loop`` and passed explicitly to the enforcement
sites. No ContextVar, no container class (see POLICIES.md §4
for the rationale).

Phase 3 ships the full evaluate() loop that dispatches to
registered :class:`Policy` instances (``LabelPolicy`` at
this phase). Phases 4/7 add :class:`FunctionPolicy` /
:class:`PromptPolicy` without changing the engine shape —
the orchestration here already handles the composition.
"""

from __future__ import annotations

from typing import Any

from agent_plane.runtime.policies.base import Policy
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

    :param policies: Per-workflow :class:`Policy` instances
        in YAML declaration order. The engine iterates this
        list in order on every ``evaluate`` call; DENY
        short-circuits, ASK accumulates, ALLOW continues.
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
        policies: list[Policy],
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

        Runs the pipeline from POLICIES.md §4:

        1. For each policy in YAML order:
           a. Skip if no :class:`PhaseSelector` matches.
           b. Skip if the policy's ``condition`` label-gate
              does not match the current hot-cache snapshot.
           c. Dispatch to ``policy.evaluate``.
           d. (Phase 4/7 will layer in action-list validation
              and the classifier-only carve-out for
              FunctionPolicy and PromptPolicy — LabelPolicy
              emits pre-validated actions so step (d) is a
              no-op at this phase.)
           e. Accumulate ``set_labels`` writes.
        2. On DENY: short-circuit. Apply accumulated writes
           from any ALLOWing predecessors, then return the
           DENY result (with ``deciding_policy`` set).
        3. After the loop, if any policy ASKed: return an ASK
           result carrying accumulated (but unapplied)
           writes — the caller applies them only on approve
           (POLICIES.md §7.2).
        4. Otherwise: apply writes, return ALLOW.

        :param ctx: The current evaluation context
            (phase + content + resolved tool_name).
        :returns: The composed :class:`PolicyResult`. Single-
            policy results are always wrapped into a composed
            result here — callers receive ALLOW / ASK / DENY
            directly.
        """
        accumulated: dict[str, str] = {}
        ask_reasons: list[str] = []
        deciding_ask_policy: str | None = None
        context = self._context()

        for policy in self.policies:
            if not self._should_fire(policy.spec, ctx):
                continue
            result = await policy.evaluate(ctx, context)
            if result.set_labels:
                accumulated.update(result.set_labels)
            if result.action == PolicyAction.DENY:
                return self._compose_deny(policy.spec.name, result.reason, accumulated)
            if result.action == PolicyAction.ASK:
                ask_reasons.append(
                    f"{policy.spec.name}: {result.reason or 'approval required'}",
                )
                if deciding_ask_policy is None:
                    deciding_ask_policy = policy.spec.name

        if ask_reasons:
            return PolicyResult(
                action=PolicyAction.ASK,
                reason="; ".join(ask_reasons),
                set_labels=dict(accumulated) if accumulated else None,
                deciding_policy=deciding_ask_policy,
            )
        self.apply_label_writes(accumulated)
        return PolicyResult(
            action=PolicyAction.ALLOW,
            reason=None,
            set_labels=dict(accumulated) if accumulated else None,
            deciding_policy=None,
        )

    def _compose_deny(
        self,
        deciding_policy: str,
        reason: str | None,
        accumulated: dict[str, str],
    ) -> PolicyResult:
        """
        Build the DENY short-circuit result.

        Applies accumulated writes from earlier ALLOWing
        policies (plus the DENYing policy's own writes that
        already landed in ``accumulated``) before returning —
        per POLICIES.md §4. Extracted from ``evaluate`` to
        keep that method under the 40-line limit.

        :param deciding_policy: Name of the policy whose DENY
            short-circuited the chain.
        :param reason: Reason carried on the DENYing result.
        :param accumulated: Label writes gathered across
            every policy up to and including the DENYing
            one.
        :returns: Composed DENY :class:`PolicyResult`.
        """
        self.apply_label_writes(accumulated)
        return PolicyResult(
            action=PolicyAction.DENY,
            reason=reason,
            set_labels=dict(accumulated) if accumulated else None,
            deciding_policy=deciding_policy,
        )

    def _should_fire(
        self,
        spec: PolicySpec,
        ctx: EvaluationContext,
    ) -> bool:
        """
        Check whether a policy's selector + condition gates
        pass for the current context.

        Two stages, short-circuited in order per §4 key
        semantics:

        1. :class:`PhaseSelector` match — cheap, no label
           reads.
        2. ``condition`` label-gate — AND across keys; list
           values = OR within the key.

        :param spec: The policy's spec.
        :param ctx: The current evaluation context.
        :returns: ``True`` when the engine should dispatch to
            ``policy.evaluate``; ``False`` when the policy is
            skipped entirely for this context.
        """
        if not any(sel.matches(ctx) for sel in spec.on):
            return False
        if spec.condition is not None and not _condition_matches(
            spec.condition,
            self._labels,
        ):
            return False
        return True

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
        for policy in self.policies:
            if policy.spec.name == policy_name:
                return policy.spec
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


def _condition_matches(
    condition: dict[str, str | list[str]],
    labels: dict[str, str],
) -> bool:
    """
    Evaluate a policy's ``condition:`` block against the
    current label snapshot.

    Semantics (POLICIES.md §4, §10):

    - AND across keys: every key in *condition* must match
      for the policy to fire.
    - Within a key, a scalar value is an equality check; a
      list is an OR — the stored value must appear in the
      list.
    - A key present in *condition* but absent from *labels*
      never matches (the policy did not set that label, so
      the gate stays closed).

    :param condition: Declarative condition from the spec.
        Values are already string-coerced at spec load.
    :param labels: Current hot-cache snapshot.
    :returns: ``True`` if every key's check passes.
    """
    for key, expected in condition.items():
        actual = labels.get(key)
        if actual is None:
            return False
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


# Re-export the defaults for callers that need them without
# importing from spec.types directly.
__all__ = ["PolicyEngine", "DEFAULT_ASK_TIMEOUT"]
