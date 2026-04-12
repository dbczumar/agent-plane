"""E2E test: archer agent web search with OpenAI passthrough.

Verifies that archer's ``web_search`` tool works as an OpenAI
native passthrough (no ``search_provider`` config needed) when
the model is ``gpt-5.4`` (no provider prefix). Uses an LLM judge
to evaluate whether the agent returned real, current web results.

This test catches the bug where ``model.split("/")[0]`` on
``"gpt-5.4"`` returned ``"gpt-5.4"`` instead of ``"openai"``,
causing the function-tool path instead of the passthrough.

Usage::

    pytest tests/e2e/test_archer_web_search.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body.
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


def _has_web_search_call(body: dict[str, Any]) -> bool:
    """
    Check if the response contains a native web_search_call event.

    This proves the OpenAI passthrough was used (not the function
    tool fallback which would show as function_call).

    :param body: The terminal response body.
    :returns: True if a web_search_call item exists.
    """
    return any(item.get("type") == "web_search_call" for item in body.get("output", []))


def test_archer_web_search_france_news(
    http_client: httpx.Client,
    archer_agent: str,
    llm_api_key: str,
) -> None:
    """
    Archer uses native OpenAI web search to find current France news.

    Sends a request asking archer to search the web for the latest
    news from France. Verifies:
    1. The response completed successfully.
    2. A native ``web_search_call`` was used (OpenAI passthrough),
       NOT a ``function_call`` to ``web_search`` (which would mean
       the passthrough failed and the function-tool fallback ran).
    3. An LLM judge confirms the response contains real, current
       news about France (not hallucinated or generic).

    **What breaks if this fails:**
    - ``parse_model_string`` not used → ``"gpt-5.4"`` parsed as
      provider ``"gpt-5.4"`` instead of ``"openai"`` → passthrough
      not activated → function tool returns "requires configuration"
    - OpenAI web search API broken → no results
    - Response assembly drops web_search_call items → missing output
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Do a quick web search for today's top news headline "
                "from France. Just one headline is fine — be brief."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    # Archer may spawn sub-agents (fact_checker, summarizer) which
    # adds latency on top of the web search itself.
    body = poll_until_terminal(http_client, response_id, timeout=300)

    # Agent should complete, not fail.
    assert body["status"] == "completed", (
        f"Response status is {body['status']!r}. "
        f"If 'failed', check if web_search returned the "
        f"'requires configuration' error instead of using the "
        f"OpenAI passthrough. Output: {body.get('output', [])}"
    )

    # Must have used the native OpenAI web_search_call passthrough,
    # NOT a function_call to web_search. If we see function_call
    # instead, the provider detection is broken.
    assert _has_web_search_call(body), (
        "Expected native web_search_call (OpenAI passthrough) but "
        "none found. This means web_search fell back to the function "
        "tool path, likely because the model string 'gpt-5.4' was "
        "not recognized as an OpenAI model. Output types: "
        f"{[item.get('type') for item in body.get('output', [])]}"
    )

    full_text = _extract_all_text(body)
    assert len(full_text) > 100, (
        f"Response too short ({len(full_text)} chars) for a news summary. Got: {full_text[:200]}"
    )

    # ── LLM judge: real, current news about France? ────────
    os.environ["OPENAI_API_KEY"] = llm_api_key

    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name="france_news",
        instructions=(
            "You are evaluating whether an AI assistant successfully "
            "searched the web and returned real, current news about "
            "France.\n\n"
            "The assistant's response is:\n"
            "{{ outputs }}\n\n"
            "Evaluate:\n"
            "1. Does the response contain specific news items or "
            "events happening in France (not generic facts about "
            "the country)?\n"
            "2. Does it mention dates, people, or events that suggest "
            "the information is recent (not historical)?\n"
            "3. Does it cite or reference web sources?\n\n"
            "Return True if the response contains real, specific, "
            "apparently current news from France. Return False if "
            "it gives generic information, historical facts, or "
            "appears to be hallucinated without web search results."
        ),
        feedback_value_type=bool,
    )

    feedback = judge(outputs=full_text)
    assert feedback.value is True, (
        f"LLM judge ruled the response does NOT contain real "
        f"current news about France.\n"
        f"Judge rationale: {feedback.rationale}\n"
        f"Response: {full_text[:500]}"
    )
