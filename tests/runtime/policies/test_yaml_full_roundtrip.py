"""
YAML → engine full-roundtrip tests.

Verifies every YAML shape from POLICIES.md §3.1 loads via
the real parser, builds a PolicyEngine, and evaluates to
the expected decision. The closest approximation to "an
agent author wrote this YAML and shipped it" without the
live LLM + workflow wiring.

These tests deliberately do NOT use the three pre-built
fixture directories — they construct spec YAML inline, so
each test demonstrates exactly which YAML shape produces
which engine behavior. A future onboarding doc could port
these test bodies verbatim as "copy-paste-ready examples".
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


def _write_and_build(
    tmp_path: Path,
    store: SqlAlchemyConversationStore,
    yaml_text: str,
) -> PolicyEngine:
    """Write a config.yaml to tmp_path and build the engine."""
    (tmp_path / "config.yaml").write_text(yaml_text)
    spec = parse(tmp_path)
    conv = store.create_conversation()
    return build_policy_engine(
        spec=spec,
        conversation_id=conv.id,
        conversation_store=store,
    )


def _input_ctx(text: str) -> EvaluationContext:
    return EvaluationContext(phase=Phase.INPUT, content=text, tool_name=None)


def _tool_ctx(name: str, args: dict[str, Any] | None = None) -> EvaluationContext:
    return EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": name, "args": args or {}},
        tool_name=name,
    )


# ── Label YAML shapes ─────────────────────────────────


@pytest.mark.asyncio
async def test_yaml_bare_string_label_shorthand(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """``integrity: "1"`` — bare-string shorthand for
    initial value. Parser produces a LabelDef with only
    `initial` set; no schema constraints apply."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: bare-string-labels
guardrails:
  labels:
    integrity: "1"
    role: "admin"
""",
    )
    assert engine.labels == {"integrity": "1", "role": "admin"}


@pytest.mark.asyncio
async def test_yaml_full_schema_label(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Full `{initial, values, monotonic}` declaration
    parses + builds correctly. Monotonic enforces at apply
    time."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: schema-label
guardrails:
  labels:
    sensitivity:
      initial: public
      values: [public, internal, confidential]
      monotonic: increasing
  policies:
    promote:
      type: label
      on: [input]
      action: allow
      set_labels:
        sensitivity: confidential
""",
    )
    # Seeded to "public".
    assert engine.labels == {"sensitivity": "public"}

    # Promote via policy.
    r = await engine.evaluate(_input_ctx("go"))
    assert r.action == PolicyAction.ALLOW
    assert engine.labels["sensitivity"] == "confidential"

    # Demote attempt → blocked by monotonic increasing.
    engine.apply_label_writes({"sensitivity": "public"})
    # Still confidential.
    assert engine.labels["sensitivity"] == "confidential"


# ── Policy YAML shapes ────────────────────────────────


@pytest.mark.asyncio
async def test_yaml_label_policy_deny(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """YAML: `type: label, on: [input], action: deny` →
    DENY on input phase."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: label-deny
guardrails:
  policies:
    block_input:
      type: label
      on: [input]
      action: deny
      reason: "nope"
""",
    )
    r = await engine.evaluate(_input_ctx("anything"))
    assert r.action == PolicyAction.DENY
    assert r.reason == "nope"


@pytest.mark.asyncio
async def test_yaml_function_policy_short_form(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """YAML: `type: function, function: dotted.path` →
    FunctionPolicy using the path directly as evaluator."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: function-short
guardrails:
  policies:
    observer:
      type: function
      on: [input]
      function: tests._fixtures.agents.combined_policies.observe_all
""",
    )
    r = await engine.evaluate(_input_ctx("x"))
    assert r.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_yaml_function_policy_factory_form(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """YAML: dict-form `function: {path, arguments}` →
    factory called with arguments at build time."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: function-factory
guardrails:
  policies:
    limiter:
      type: function
      on: [tool_call:web_search]
      function:
        path: tests._fixtures.agents.rate_limit_policies.rate_limit_search
        arguments:
          limit: 1
""",
    )
    # First call ALLOWs (within budget=1).
    r1 = await engine.evaluate(_tool_ctx("web_search", {"q": "x"}))
    assert r1.action == PolicyAction.ALLOW
    # Second call asks (budget exhausted).
    r2 = await engine.evaluate(_tool_ctx("web_search", {"q": "y"}))
    assert r2.action == PolicyAction.ASK


@pytest.mark.asyncio
async def test_yaml_prompt_policy_with_stub(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """YAML PromptPolicy builds; test stubs the classifier
    to avoid hitting a real LLM."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: prompt-demo
llm:
  model: openai/gpt-4o
guardrails:
  policies:
    check:
      type: prompt
      on: [input]
      action: [allow, deny]
      prompt: "Deny if mentions Canada."
""",
    )

    # Inject classifier.
    async def _deny_classifier(prompt: str) -> dict[str, Any]:
        return {"action": "deny", "reason": "Canada detected"}

    for p in engine.policies:
        if isinstance(p, PromptPolicy):
            p._classifier = _deny_classifier

    r = await engine.evaluate(_input_ctx("Tell me about Toronto"))
    assert r.action == PolicyAction.DENY
    assert r.reason == "Canada detected"


# ── `on:` YAML 1.1 trap regression ────────────────────


@pytest.mark.asyncio
async def test_yaml_on_key_stays_string(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """YAML 1.1 parses `on:` as boolean True by default.
    agent-plane's custom loader keeps it as a string. If
    this regresses, every policy's `on:` key disappears
    and all policies silently stop firing.

    Most critical regression guard in the parser suite —
    duplicated here at the full-roundtrip level so a
    workflow run would fail if the loader ever reverted."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: on-trap-guard
guardrails:
  policies:
    p:
      type: label
      on: [input]
      action: deny
""",
    )
    # If on: got parsed as True, there'd be no policies
    # with a matching selector → default ALLOW. DENY here
    # proves the selector was preserved.
    r = await engine.evaluate(_input_ctx("x"))
    assert r.action == PolicyAction.DENY


# ── Combined: multiple types in one YAML ──────────────


@pytest.mark.asyncio
async def test_yaml_all_three_types_in_one_spec(
    tmp_path: Path,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """YAML declaring LabelPolicy + FunctionPolicy +
    PromptPolicy on different phases. All three build
    correctly and fire on their respective phases."""
    engine = _write_and_build(
        tmp_path,
        conversation_store,
        """
spec_version: 1
name: three-types
llm:
  model: openai/gpt-4o
guardrails:
  labels:
    integrity: "1"
  policies:
    label_taint:
      type: label
      on: [tool_call:web]
      action: allow
      set_labels:
        integrity: "0"
    function_rate:
      type: function
      on: [tool_call:search]
      function: tests._fixtures.agents.combined_policies.observe_all
    prompt_check:
      type: prompt
      on: [input]
      action: [allow, deny]
      prompt: "check"
""",
    )
    # Three policies, one of each type.
    assert len(engine.policies) == 3
    names = [p.spec.name for p in engine.policies]
    assert names == ["label_taint", "function_rate", "prompt_check"]

    # Inject a stub for the PromptPolicy.
    for p in engine.policies:
        if isinstance(p, PromptPolicy):
            p._classifier = lambda _prompt: {"action": "allow"}  # type: ignore[assignment]  # noqa: E501

    # LabelPolicy fires on tool_call:web → ALLOW + write.
    r1 = await engine.evaluate(_tool_ctx("web"))
    assert r1.action == PolicyAction.ALLOW
    assert engine.labels["integrity"] == "0"

    # FunctionPolicy fires on tool_call:search → ALLOW.
    r2 = await engine.evaluate(_tool_ctx("search"))
    assert r2.action == PolicyAction.ALLOW
