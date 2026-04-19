"""
``build_policy_engine`` — construct a :class:`PolicyEngine` for a
workflow.

Called at the top of ``_run_agent_loop``. Seeds any
``LabelDef.initial`` values that are not already present in
``conversation_labels`` using an
``INSERT ... ON CONFLICT DO NOTHING`` semantic so that two
concurrent workflows on the same conversation (the v2 case
tracked in POLICIES.md Open Q #6) never clobber each other's
view of a label's first value.

Phase 2 scope: zero-policy and declared-policy paths both work;
concrete Policy subclasses land in Phases 3+, and this builder
will start instantiating them as those phases ship.
"""

from __future__ import annotations

from agent_plane.runtime.policies.base import Policy
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.policies.function import resolve_function_policy
from agent_plane.runtime.policies.label import LabelPolicy
from agent_plane.spec.types import (
    DEFAULT_ASK_TIMEOUT,
    AgentSpec,
    FunctionPolicySpec,
    LabelDef,
    LabelPolicySpec,
    PolicySpec,
)
from agent_plane.stores.conversation_store import ConversationStore


def build_policy_engine(
    *,
    spec: AgentSpec,
    conversation_id: str,
    conversation_store: ConversationStore,
) -> PolicyEngine:
    """
    Construct the :class:`PolicyEngine` for one workflow.

    When ``spec.guardrails`` is ``None`` (no guardrails
    declared), returns a no-op engine with empty policies and
    labels — the four enforcement sites still call through,
    they just always ALLOW.

    When declared labels have an ``initial`` value and no row
    exists yet in ``conversation_labels``, seeds via
    ``ConversationStore.set_labels`` — but only for keys not
    already persisted, so existing label state is never
    clobbered. The hot cache is built from the freshly seeded
    snapshot.

    :param spec: The parsed agent spec.
    :param conversation_id: The conversation this workflow is
        running on, e.g. ``"conv_abc123"``.
    :param conversation_store: The store used for label reads
        and writes. Held by the engine for the life of the
        workflow.
    :returns: A :class:`PolicyEngine` ready for evaluation.
    """
    guardrails = spec.guardrails
    if guardrails is None:
        return _build_noop_engine(
            conversation_id=conversation_id,
            conversation_store=conversation_store,
        )
    label_defs = guardrails.labels or {}
    initial_labels = _seed_and_load_labels(
        conversation_id=conversation_id,
        label_defs=label_defs,
        conversation_store=conversation_store,
    )
    return PolicyEngine(
        policies=[_instantiate_policy(s) for s in (guardrails.policies or [])],
        label_defs=label_defs,
        ask_timeout=guardrails.ask_timeout,
        conversation_id=conversation_id,
        initial_labels=initial_labels,
        conversation_store=conversation_store,
    )


def _instantiate_policy(spec: PolicySpec) -> Policy:
    """
    Dispatch a :class:`PolicySpec` to the matching runtime
    :class:`Policy` subclass.

    Phase 3: only :class:`LabelPolicySpec` is supported;
    :class:`FunctionPolicySpec` and :class:`PromptPolicySpec`
    raise a clear NotImplementedError until Phases 4 and 7
    land. The error message names the missing phase so the
    agent author can see why their spec's declaration was
    rejected.

    :param spec: The declarative spec.
    :returns: A :class:`Policy` subclass instance bound to
        the spec.
    :raises NotImplementedError: For policy types whose
        runtime class has not yet shipped.
    """
    if isinstance(spec, LabelPolicySpec):
        return LabelPolicy(spec)
    if isinstance(spec, FunctionPolicySpec):
        return resolve_function_policy(spec)
    raise NotImplementedError(
        f"Policy type {type(spec).__name__} for {spec.name!r} is not "
        f"yet implemented in this phase. ``PromptPolicy`` lands in "
        f"Phase 7 (see designs/POLICIES_IMPLEMENTATION_PLAN.md).",
    )


def _build_noop_engine(
    *,
    conversation_id: str,
    conversation_store: ConversationStore,
) -> PolicyEngine:
    """
    Build an engine for an agent with no guardrails declared.

    Kept as a named helper rather than inlined so the
    zero-policy path is grep-able ("why is every phase
    returning ALLOW?" → search for ``_build_noop_engine``).

    :param conversation_id: The conversation for the workflow.
    :param conversation_store: Writes from this engine still
        go through the store — useful if a later turn of the
        same conversation runs under an updated spec that
        does declare guardrails.
    :returns: An engine with zero policies and an empty label
        cache.
    """
    # We still read the persisted labels (if any) so an engine
    # upgrade mid-conversation sees state its predecessor
    # wrote.
    existing = _load_existing_labels(conversation_id, conversation_store)
    return PolicyEngine(
        policies=[],
        label_defs={},
        ask_timeout=DEFAULT_ASK_TIMEOUT,
        conversation_id=conversation_id,
        initial_labels=existing,
        conversation_store=conversation_store,
    )


def _seed_and_load_labels(
    *,
    conversation_id: str,
    label_defs: dict[str, LabelDef],
    conversation_store: ConversationStore,
) -> dict[str, str]:
    """
    Seed declared initial values and return the current snapshot.

    Race-safe across concurrent workflows: only writes keys
    that are missing from the persisted state. If two
    workflows seed simultaneously, the dialect-specific UPSERT
    guarantees one writer wins per (conversation, key) pair
    and the other no-ops.

    :param conversation_id: The conversation to seed.
    :param label_defs: Per-key declarations from the spec.
        Keys with ``initial is None`` are skipped (those
        labels start unset until a policy writes them).
    :param conversation_store: Target for both the read and
        the seed UPSERT.
    :returns: Full post-seed snapshot of the conversation's
        labels.
    """
    existing = _load_existing_labels(conversation_id, conversation_store)
    to_seed = {
        key: ldef.initial
        for key, ldef in label_defs.items()
        if ldef.initial is not None and key not in existing
    }
    if to_seed:
        conversation_store.set_labels(conversation_id, to_seed)
        # Re-read to pick up the freshly seeded values plus any
        # writes that landed concurrently from another workflow.
        existing = _load_existing_labels(conversation_id, conversation_store)
    return existing


def _load_existing_labels(
    conversation_id: str,
    conversation_store: ConversationStore,
) -> dict[str, str]:
    """
    Load the current persisted label state.

    Empty dict when the conversation has no labels yet (or
    when the conversation itself does not exist yet — the
    caller is responsible for ordering conversation creation
    before engine build).

    :param conversation_id: Conversation to load.
    :param conversation_store: Store to read from.
    :returns: ``{key: value}`` map. Empty when nothing
        persisted.
    """
    conv = conversation_store.get_conversation(conversation_id)
    if conv is None:
        return {}
    return dict(conv.labels)


__all__ = ["build_policy_engine"]
