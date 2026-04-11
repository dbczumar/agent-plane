"""E2E test: OpenAI Agents SDK executor with openai-coder agent.

Verifies that the AgentsSdkExecutor runs single-turn and
multi-turn conversations with a real LLM. Uses the openai-coder
agent which has sub-agents, skills, and web search.

Usage::

    pytest tests/e2e/test_agents_sdk_basic.py \
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body from
        GET /v1/responses/{id}.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def test_agents_sdk_single_turn_completes(
    http_client: httpx.Client,
    openai_coder_agent: str,
) -> None:
    """
    Basic smoke test: the Agents SDK executor runs a single
    turn and produces a completed response with correct text.

    **What breaks if wrong:**

    - If ``_ensure_sdk()`` fails, ``from_spec`` raises
      ``ImportError`` and the task fails immediately.
    - If ``_build_model_settings`` maps config incorrectly,
      the LLM rejects the parameters (400 error).
    - If ``_map_event`` doesn't map ``TextChunk`` correctly,
      no text appears in the response output.
    - If ``TurnComplete`` is never yielded, the response
      stays in ``in_progress`` forever and the poll times out.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": openai_coder_agent,
            "input": ("What is 2 + 2? Reply with just the number."),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(
        http_client,
        response_id,
        timeout=60,
    )
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}."
    )

    text = _extract_all_text(body)
    assert "4" in text, f"Expected '4' in response: {text[:300]}"


def test_agents_sdk_multi_turn_remembers(
    http_client: httpx.Client,
    openai_coder_agent: str,
) -> None:
    """
    Two-turn conversation: the agent remembers turn 1 content
    in turn 2 via history replay.

    Turn 1: state a fact. Turn 2: ask about it.

    **What breaks if wrong:**

    - If ``_messages_to_input`` doesn't pass history correctly,
      the SDK sees no prior context and can't answer.
    - If the workflow doesn't load prior items into ``messages``,
      the executor receives an empty history.
    """
    # Turn 1: state a fact.
    resp_1 = http_client.post(
        "/v1/responses",
        json={
            "model": openai_coder_agent,
            "input": ("My name is Zephyr and I live in Portland."),
            "background": True,
        },
    )
    resp_1.raise_for_status()
    id_1 = resp_1.json()["id"]
    body_1 = poll_until_terminal(
        http_client,
        id_1,
        timeout=60,
    )
    assert body_1["status"] == "completed", f"Turn 1 failed: {body_1.get('error')}"

    # Turn 2: ask about the fact.
    resp_2 = http_client.post(
        "/v1/responses",
        json={
            "model": openai_coder_agent,
            "input": ("What is my name and where do I live?"),
            "background": True,
            "previous_response_id": id_1,
        },
    )
    resp_2.raise_for_status()
    id_2 = resp_2.json()["id"]
    body_2 = poll_until_terminal(
        http_client,
        id_2,
        timeout=60,
    )
    assert body_2["status"] == "completed", f"Turn 2 failed: {body_2.get('error')}"

    text = _extract_all_text(body_2).lower()
    assert "zephyr" in text, f"Expected 'zephyr' in turn 2 response: {text[:300]}"
    assert "portland" in text, f"Expected 'portland' in turn 2 response: {text[:300]}"
