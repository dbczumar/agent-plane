"""
Integration tests for :class:`PromptPolicy` through the full
parse → build → stub-classifier pipeline.

Unlike ``test_prompt_policy.py`` (which constructs PromptPolicy
instances directly), this file loads the ``prompt-policy-demo``
fixture via the real ``parse()`` + ``build_policy_engine()``
code path, then swaps in test classifiers by mutating the
policy instances the builder produced.

That mirrors the pattern production will use once
``_default_classifier`` is wired to the real LLM — tests
provide canned responses, production provides real ones.

Ports these omniagents cases at the integration level (the
non-integration versions live in ``test_prompt_policy.py``):

- ``test_prompt_policy_denies_content`` — input classifier DENY
- ``test_prompt_policy_can_set_labels_when_enabled`` —
  tool_result classifier writes a label
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_plane.runtime.policies import build_policy_engine
from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.policies.prompt import PromptPolicy
from agent_plane.spec.parser import parse
from agent_plane.spec.types import (
    EvaluationContext,
    Phase,
    PolicyAction,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "_fixtures" / "agents" / "prompt-policy-demo"


def _build_with_stubs(
    store: SqlAlchemyConversationStore,
    classifiers_by_policy_name: dict[str, Any],
) -> PolicyEngine:
    """
    Load the fixture and inject stub classifiers by policy name.

    After `build_policy_engine` constructs the default-
    classifier PromptPolicy instances, we reach into each
    one and swap its classifier attr — the simplest hook
    for integration tests that don't want the production
    NotImplementedError placeholder.

    :param store: Backing conversation store.
    :param classifiers_by_policy_name: Mapping of policy
        name → async classifier stub.
    :returns: PolicyEngine ready to evaluate.
    """
    spec = parse(_FIXTURE)
    conv = store.create_conversation()
    engine = build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=store,
    )
    for policy in engine.policies:
        if isinstance(policy, PromptPolicy) and policy.spec.name in classifiers_by_policy_name:
            # Direct attr swap — mirrors what Phase 9's
            # production classifier injection will do.
            policy._classifier = classifiers_by_policy_name[policy.spec.name]
    return engine


def _canned(result_dict: dict[str, Any]) -> Any:
    """Build an async classifier returning a fixed dict."""

    async def _classifier(prompt: str) -> dict[str, Any]:
        return result_dict

    return _classifier


# ── Input-phase PromptPolicy ──────────────────────────


@pytest.mark.asyncio
async def test_input_classifier_allows_clean_request(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A clean user message passes the block_canada_input
    classifier → ALLOW at the input phase."""
    engine = _build_with_stubs(
        conversation_store,
        {
            "block_canada_input": _canned({"action": "allow", "reason": ""}),
        },
    )
    result = await engine.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="Tell me about Brazil."),
    )
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_input_classifier_denies_canada(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A Canada-mentioning message trips block_canada_input."""
    engine = _build_with_stubs(
        conversation_store,
        {
            "block_canada_input": _canned(
                {"action": "deny", "reason": "Canada-related topic"},
            ),
        },
    )
    result = await engine.evaluate(
        EvaluationContext(
            phase=Phase.INPUT,
            content="What is the airport code for Toronto?",
        ),
    )
    assert result.action == PolicyAction.DENY
    assert result.reason == "Canada-related topic"
    assert result.deciding_policy == "block_canada_input"


# ── Tool-result PromptPolicy that writes labels ───────


@pytest.mark.asyncio
async def test_tool_result_classifier_writes_label(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """classify_doc_sensitivity is a classifier-only policy
    (action:[allow]) that writes a label. Verify the label
    lands in both the hot cache and the store."""
    engine = _build_with_stubs(
        conversation_store,
        {
            "classify_doc_sensitivity": _canned(
                {
                    "action": "allow",
                    "reason": "",
                    "set_labels": {"sensitivity": "confidential"},
                },
            ),
        },
    )
    r = await engine.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_RESULT,
            content={"output": "Internal memo about acquisitions."},
            tool_name="read_doc",
        ),
    )
    assert r.action == PolicyAction.ALLOW
    # Hot cache reflects the classifier's write.
    assert engine.labels["sensitivity"] == "confidential"
    # Persisted — the ALLOW path's accumulated writes went
    # through apply_label_writes.
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels["sensitivity"] == "confidential"


@pytest.mark.asyncio
async def test_tool_result_classifier_label_whitelist_enforced(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """The classifier writes a key NOT in set_labels:
    [sensitivity] — engine filters it out. Prevents a
    classifier from exfiltrating policy state into unrelated
    label keys."""
    engine = _build_with_stubs(
        conversation_store,
        {
            "classify_doc_sensitivity": _canned(
                {
                    "action": "allow",
                    "set_labels": {
                        "sensitivity": "internal",
                        "arbitrary_key": "hacker",
                    },
                },
            ),
        },
    )
    await engine.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_RESULT,
            content={"output": "x"},
            tool_name="read_doc",
        ),
    )
    # Whitelisted key landed; rogue key dropped.
    assert engine.labels == {"sensitivity": "internal"}


# ── Multi-policy composition with PromptPolicy ────────


@pytest.mark.asyncio
async def test_both_classifiers_fire_across_phases(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A full-turn sequence: INPUT classifier ALLOWs; a
    read_doc tool_result runs → sensitivity classifier
    writes. Proves the two PromptPolicy phases coexist in
    one engine."""
    engine = _build_with_stubs(
        conversation_store,
        {
            "block_canada_input": _canned({"action": "allow"}),
            "classify_doc_sensitivity": _canned(
                {
                    "action": "allow",
                    "set_labels": {"sensitivity": "internal"},
                },
            ),
        },
    )

    r_input = await engine.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="benign question"),
    )
    assert r_input.action == PolicyAction.ALLOW

    r_tool = await engine.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_RESULT,
            content={"output": "confidential data"},
            tool_name="read_doc",
        ),
    )
    assert r_tool.action == PolicyAction.ALLOW
    assert engine.labels["sensitivity"] == "internal"


@pytest.mark.asyncio
async def test_input_phase_only_runs_input_policy(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """On INPUT phase, only the input PromptPolicy fires —
    the tool_result classifier is never invoked. Proves the
    selector filter is doing its job even for PromptPolicy
    (which is the most expensive policy type, so mis-firing
    wastes classifier calls)."""
    call_count = {"sensitivity": 0}

    async def _sensitivity_stub(prompt: str) -> dict[str, Any]:
        call_count["sensitivity"] += 1
        return {"action": "allow"}

    engine = _build_with_stubs(
        conversation_store,
        {
            "block_canada_input": _canned({"action": "allow"}),
            "classify_doc_sensitivity": _sensitivity_stub,
        },
    )
    # Evaluate only the INPUT phase.
    await engine.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
    )
    # sensitivity classifier must NOT have been called.
    assert call_count["sensitivity"] == 0
