"""
Production classifier for :class:`PromptPolicy`.

:class:`PromptPolicy` is a pure evaluator in
:mod:`agent_plane.policies.prompt`: it takes a callable
``classifier(prompt) -> dict`` that knows how to turn a
prompt string into the JSON-shaped verdict the policy
expects. Before Phase 9 the production classifier raised
:class:`NotImplementedError` — agents that declared a
``prompt``-type policy could not run without a test stub.

This module ships the real classifier. The agent's top-level
``llm:`` config is the default backend; individual policies
can override it via ``spec.llm``. The classifier's timeout
comes from ``spec.llm.request_timeout`` when declared,
falling back to :data:`DEFAULT_POLICY_CLASSIFIER_TIMEOUT`
(30 s per POLICIES.md §9.2) — explicitly NOT the agent's
generation timeout, which is tuned for generation not
classification.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_plane.llms.client import Client
from agent_plane.llms.types import MessageOutput, Response
from agent_plane.spec.types import (
    DEFAULT_POLICY_CLASSIFIER_TIMEOUT,
    LLMConfig,
    PromptPolicySpec,
)

_logger = logging.getLogger(__name__)

# Classifier stub signature — matches what
# :class:`PromptPolicy` expects (see
# :mod:`agent_plane.policies.prompt`).
_ClassifierCallable = Callable[[str], Awaitable[dict[str, Any]]]


def make_default_classifier(
    policy_spec: PromptPolicySpec,
    agent_llm: LLMConfig | None,
) -> _ClassifierCallable:
    """
    Build the production classifier for one PromptPolicy.

    Resolves the effective LLM config at build time — policy
    override first, agent-level LLM second. Fails loud when
    NEITHER is set: a ``prompt`` policy on an agent without
    an LLM config is a configuration error, and silently
    substituting ALLOW via the carve-out would be worse
    than refusing to start.

    The returned callable:

    1. Sends a single user-role message carrying the framework
       envelope + author prompt (already assembled by
       :meth:`PromptPolicy._build_prompt`).
    2. Calls the LLM through the shared multi-provider
       :class:`~agent_plane.llms.client.Client`.
    3. Extracts the assistant text from the ``Response``.
    4. Parses it as JSON and returns the dict.

    Any failure along the way raises; the engine's
    :func:`~agent_plane.runtime.policies.engine._dispatch_policy`
    safety net catches it and applies the right fail-closed
    policy (DENY for specs that can DENY, substituted-ALLOW
    for classifier-only specs).

    :param policy_spec: The :class:`PromptPolicySpec`. Its
        ``llm`` override wins over ``agent_llm`` when set;
        its ``name`` is carried through for log messages.
    :param agent_llm: The agent-level LLM config (the
        ``llm:`` block at the top of config.yaml). Used when
        the policy didn't declare its own.
    :returns: An async callable matching the classifier
        contract :class:`PromptPolicy` expects.
    :raises ValueError: When no LLM config is available —
        neither on the policy nor on the agent.
    """
    effective_llm = policy_spec.llm or agent_llm
    if effective_llm is None:
        raise ValueError(
            f"PromptPolicy {policy_spec.name!r} needs an LLM config "
            f"(via ``policy.llm`` or the agent-level ``llm:`` block).",
        )
    # Explicit ``is None`` check — ``or`` would demote a
    # legitimate ``request_timeout=0`` (disable-timeout
    # convention on some adapters) to the default.
    if effective_llm.request_timeout is not None:
        timeout = effective_llm.request_timeout
    else:
        timeout = DEFAULT_POLICY_CLASSIFIER_TIMEOUT
    # Single shared client; constructing is cheap and the
    # underlying adapters manage their own HTTP pooling.
    client = Client()

    async def _classifier(prompt: str) -> dict[str, Any]:
        """
        Thin bound closure — delegates to ``_call_classifier``
        which does the actual LLM call + parse. See the
        module-level function for the real work.

        :param prompt: Full prompt assembled by
            :class:`PromptPolicy`.
        :returns: Parsed verdict dict.
        """
        return await _call_classifier(
            client=client,
            policy_name=policy_spec.name,
            model=effective_llm.model,
            connection=effective_llm.connection,
            timeout=timeout,
            prompt=prompt,
        )

    return _classifier


async def _call_classifier(
    *,
    client: Client,
    policy_name: str,
    model: str,
    connection: dict[str, str] | None,
    timeout: int,
    prompt: str,
) -> dict[str, Any]:
    """
    Make the actual LLM call and parse the verdict.

    Extracted from :func:`make_default_classifier` so the
    outer function stays under the 40-line limit and the
    closure's dependencies are explicit (no hidden state
    capture). Every argument the LLM call needs is
    positional here.

    :param client: Shared :class:`~agent_plane.llms.client.Client`.
    :param policy_name: Policy name for log / error messages.
    :param model: Provider-prefixed model id (e.g.
        ``"openai/gpt-4o"``).
    :param connection: Per-provider connection overrides
        (api_key, base_url, etc.) from the effective LLM
        config. ``None`` falls back to adapter defaults.
    :param timeout: Request timeout in seconds — already
        resolved to the policy-classifier default (30s) when
        the config didn't declare one.
    :param prompt: Full prompt string to send.
    :returns: Parsed verdict dict.
    :raises ValueError: On non-JSON responses or empty
        assistant text.
    """
    resp = await client.responses.create(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        model=model,
        connection_params=connection,
        timeout=timeout,
    )
    # ``stream=False`` is the default on ``responses.create``;
    # callers of this module always get a :class:`Response`.
    # Asserting rather than branching keeps the function tight
    # and turns a caller regression into a loud crash.
    assert isinstance(resp, Response), f"classifier expected Response, got {type(resp).__name__}"
    text = _extract_assistant_text(resp)
    try:
        return _parse_classifier_json(text)
    except ValueError as exc:
        _logger.warning(
            "PromptPolicy %r classifier returned non-JSON output: %s",
            policy_name,
            exc,
        )
        raise


def _extract_assistant_text(resp: Response) -> str:
    """
    Pull the assistant text out of a non-streaming Response.

    A well-formed classifier response is a single
    :class:`MessageOutput` whose content is a list of
    ``OutputText`` parts. We concatenate parts (usually one)
    and return their combined text. Anything else — no
    message, no text parts — raises so the engine's fail-
    closed path can convert it.

    :param resp: The ``Response`` from ``responses.create``.
    :returns: Concatenated assistant text.
    :raises ValueError: When the response contains no
        assistant text.
    """
    parts: list[str] = []
    for item in resp.output:
        if not isinstance(item, MessageOutput):
            continue
        for content_part in item.content:
            parts.append(content_part.text)
    if not parts:
        raise ValueError("classifier response contained no assistant text")
    return "".join(parts)


def _parse_classifier_json(text: str) -> dict[str, Any]:
    """
    Parse the classifier's assistant text as JSON.

    LLMs frequently wrap JSON in Markdown code fences (`````` ``json
    {...}`` ``````) even when instructed otherwise; this helper
    strips those fences so the parse succeeds. Any remaining
    JSON error is bubbled up to the caller.

    :param text: Raw assistant text.
    :returns: Parsed dict.
    :raises ValueError: When the text isn't valid JSON, or
        parses to something other than a dict.
    """
    stripped = text.strip()
    # Strip ```json or ``` fences if present — common LLM
    # output shape despite system-prompt instructions.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence line and, if present, the
        # closing one.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"classifier output is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"classifier JSON root must be an object, got {type(parsed).__name__}",
        )
    return parsed


__all__ = ["make_default_classifier"]
