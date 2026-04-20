"""
Unit tests for the production PromptPolicy classifier.

Covers the pure helpers (JSON parsing, text extraction, LLM
config resolution) without spinning up a real LLM. The
end-to-end "real LLM call" path is exercised in
``tests/e2e/test_policies_e2e.py::test_prompt_policy_*``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from agent_plane.llms.types import MessageOutput, OutputText, Response
from agent_plane.runtime.policies.prompt_classifier import (
    _extract_assistant_text,
    _parse_classifier_json,
    make_default_classifier,
)
from agent_plane.spec.types import (
    LLMConfig,
    Phase,
    PhaseSelector,
    PolicyAction,
    PromptPolicySpec,
)

# ── _parse_classifier_json ────────────────────────────────


def test_parse_plain_json_object() -> None:
    """
    Plain JSON without fences parses to a dict. This is the
    ideal path — the framework prompt instructs the LLM to
    emit raw JSON.
    """
    raw = '{"action": "allow", "reason": ""}'
    assert _parse_classifier_json(raw) == {"action": "allow", "reason": ""}


def test_parse_json_wrapped_in_code_fences() -> None:
    """
    LLMs often wrap JSON in triple-backtick fences even when
    told otherwise. The helper strips them so the parse
    succeeds. Regression guard: a correct envelope that the
    model decorates must not fail-closed DENY.
    """
    raw = '```json\n{"action": "deny", "reason": "blocked"}\n```'
    assert _parse_classifier_json(raw) == {"action": "deny", "reason": "blocked"}


def test_parse_json_wrapped_in_plain_fences() -> None:
    r"""
    Some LLMs use \`\`\`\n{...}\n\`\`\` (no ``json`` tag).
    Same stripping applies — the fence, not the language
    marker, is what matters.
    """
    raw = '```\n{"action": "ask"}\n```'
    assert _parse_classifier_json(raw) == {"action": "ask"}


def test_parse_surrounding_whitespace_ignored() -> None:
    """Leading/trailing whitespace is stripped before parsing."""
    raw = '   \n  {"action": "allow"}  \n  '
    assert _parse_classifier_json(raw) == {"action": "allow"}


def test_parse_invalid_json_raises_value_error() -> None:
    """
    Invalid JSON → ValueError with a helpful message. The
    engine's safety net catches this and fails closed
    (DENY or substituted ALLOW per POLICIES.md §13).
    """
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_classifier_json("not json at all {")


def test_parse_non_object_root_raises_value_error() -> None:
    """
    A JSON array or scalar can't carry the expected
    ``{"action": ..., "reason": ...}`` shape. Raise rather
    than silently returning — the engine doesn't know how
    to use a list.
    """
    with pytest.raises(ValueError, match="must be an object"):
        _parse_classifier_json('["allow"]')


# ── _extract_assistant_text ───────────────────────────────


def _make_response(texts: list[str]) -> Response:
    """
    Build a minimal :class:`Response` with the given text
    parts for testing extraction. Uses real SDK types
    (MessageOutput + OutputText) — MagicMock would silently
    bypass the isinstance checks in the helper.
    """
    return Response(
        output=[
            MessageOutput(content=[OutputText(text=t) for t in texts]),
        ],
        model="test-model",
    )


def test_extract_single_text_part() -> None:
    """A response with one text part returns it verbatim."""
    resp = _make_response(["hello world"])
    assert _extract_assistant_text(resp) == "hello world"


def test_extract_multiple_text_parts_concatenated() -> None:
    """
    Multiple text parts (rare but possible on some providers)
    concatenate without separator — matches OpenAI's
    response-rendering semantics.
    """
    resp = _make_response(['{"action":', ' "allow"}'])
    assert _extract_assistant_text(resp) == '{"action": "allow"}'


def test_extract_empty_response_raises() -> None:
    """
    A response with no assistant message content is an error
    path — the engine's safety net DENYs / substitutes ALLOW
    depending on the policy's action whitelist.
    """
    resp = Response(output=[], model="test-model")
    with pytest.raises(ValueError, match="no assistant text"):
        _extract_assistant_text(resp)


# ── make_default_classifier: config resolution ────────────


def _prompt_spec(
    *,
    llm: LLMConfig | None = None,
    actions: list[PolicyAction] | None = None,
) -> PromptPolicySpec:
    """Build a PromptPolicySpec for config-resolution tests."""
    return PromptPolicySpec(
        name="test_policy",
        on=[PhaseSelector(phase=Phase.INPUT, tool_name=None)],
        prompt="Deny canada-related requests.",
        llm=llm,
        action=actions or [PolicyAction.ALLOW, PolicyAction.DENY],
    )


def test_raises_when_neither_policy_nor_agent_has_llm() -> None:
    """
    A PromptPolicy with no policy-level llm AND no
    agent-level llm is a configuration error. Fail-loud
    rather than returning a classifier that would crash on
    first call.
    """
    spec = _prompt_spec(llm=None)
    with pytest.raises(ValueError, match="needs an LLM config"):
        make_default_classifier(spec, agent_llm=None)


def test_policy_llm_override_wins() -> None:
    """
    When both are set, the policy-level ``llm:`` override is
    used — classifier models can differ from the generation
    model (often cheaper / faster). Agent-level is fallback
    only.
    """
    policy_llm = LLMConfig(model="openai/gpt-4o-mini", request_timeout=15)
    agent_llm = LLMConfig(model="anthropic/claude-sonnet", request_timeout=300)
    spec = _prompt_spec(llm=policy_llm)

    captured: dict[str, Any] = {}

    class _FakeResponsesNs:
        """Mock responses namespace that records the call."""

        async def create(self, **kwargs: Any) -> Response:
            """Record kwargs and return a valid stub response."""
            captured.update(kwargs)
            return _make_response(['{"action": "allow"}'])

    class _FakeClient:
        """Mock Client that surfaces a fake responses namespace."""

        def __init__(self) -> None:
            """Initialize with the fake namespace."""
            self.responses = _FakeResponsesNs()

    with patch(
        "agent_plane.runtime.policies.prompt_classifier.Client",
        _FakeClient,
    ):
        classifier = make_default_classifier(spec, agent_llm=agent_llm)
        asyncio.run(classifier("test prompt"))

    # The policy-level model was selected, not the agent's.
    assert captured["model"] == "openai/gpt-4o-mini"
    # And its timeout — 15, not the agent's 300.
    assert captured["timeout"] == 15


def test_agent_llm_used_when_policy_override_absent() -> None:
    """
    No policy-level llm → fall back to agent LLM. Proves the
    default path for agents that just declare ``llm:`` at the
    top level and let policies share it.
    """
    agent_llm = LLMConfig(model="anthropic/claude-sonnet", request_timeout=42)
    spec = _prompt_spec(llm=None)

    captured: dict[str, Any] = {}

    class _FakeResponsesNs:
        """Mock responses namespace that records the call."""

        async def create(self, **kwargs: Any) -> Response:
            """Record kwargs and return a stub response."""
            captured.update(kwargs)
            return _make_response(['{"action": "allow"}'])

    class _FakeClient:
        """Mock Client."""

        def __init__(self) -> None:
            """Initialize."""
            self.responses = _FakeResponsesNs()

    with patch(
        "agent_plane.runtime.policies.prompt_classifier.Client",
        _FakeClient,
    ):
        classifier = make_default_classifier(spec, agent_llm=agent_llm)
        asyncio.run(classifier("test prompt"))

    assert captured["model"] == "anthropic/claude-sonnet"
    assert captured["timeout"] == 42


def test_classifier_parses_real_response_json() -> None:
    """
    End-to-end of the classifier callable (minus real
    network): fake LLM returns ``{"action": "deny", "reason":
    "canada"}`` → classifier returns the parsed dict. This
    is the happy path the engine consumes.
    """
    agent_llm = LLMConfig(model="openai/gpt-4o")
    spec = _prompt_spec(llm=None)

    class _FakeResponsesNs:
        """Returns a canned deny verdict."""

        async def create(self, **kwargs: Any) -> Response:
            """Return a fixed deny response."""
            return _make_response(['{"action": "deny", "reason": "canada"}'])

    class _FakeClient:
        """Mock Client."""

        def __init__(self) -> None:
            """Initialize."""
            self.responses = _FakeResponsesNs()

    with patch(
        "agent_plane.runtime.policies.prompt_classifier.Client",
        _FakeClient,
    ):
        classifier = make_default_classifier(spec, agent_llm=agent_llm)
        result = asyncio.run(classifier("Is Canada okay?"))

    assert result == {"action": "deny", "reason": "canada"}
