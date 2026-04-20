"""E2E test: native tool items persist across agent loop iterations.

Verifies that provider-native tool results (e.g. web_search_call)
are persisted to the conversation store and replayed to the LLM on
subsequent iterations. Without this, the LLM re-requests the same
searches in a loop.

Usage::

    pytest tests/e2e/test_native_tool_persistence.py \
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


def _count_web_search_calls(body: dict[str, Any]) -> int:
    """
    Count ``web_search_call`` items in the response output.

    :param body: The terminal response body.
    :returns: Number of web_search_call items.
    """
    return sum(1 for item in body.get("output", []) if item.get("type") == "web_search_call")


def test_web_search_results_not_repeated(
    http_client: httpx.Client,
    archer_agent: str,
    llm_api_key: str,
) -> None:
    """
    The archer agent uses web search to look up GitHub stars for
    multiple repos. The search results must persist across agent
    loop iterations so the LLM doesn't re-request them.

    **What breaks if native tool items aren't persisted:**

    The LLM calls web_search for each repo. If the results are
    lost on the next iteration (e.g. because the LLM also calls
    a regular tool like terminal_run), the LLM sees no search
    results in its history and re-requests the same searches.
    This loops dozens of times, wasting tokens and time.

    **Expected behavior:**

    Each unique search query should appear roughly once. The total
    number of web_search_call items should be proportional to the
    number of repos (not 10x that due to repetition).
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Get the number of GitHub stars for these repos: "
                "mlflow, langfuse, braintrust. "
                "Report the numbers."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=120)
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    text = _extract_all_text(body)
    assert len(text) > 30, f"Expected output, got: {text!r}"

    # Count web_search_call items. With 3 repos, we expect roughly
    # 3-9 searches (some retries are OK). Without persistence, we'd
    # see 30+ due to the repetition loop.
    search_count = _count_web_search_calls(body)
    assert search_count <= 15, (
        f"Too many web_search_call items ({search_count}). "
        f"Native tool results are likely not persisted, causing "
        f"the LLM to re-request the same searches in a loop."
    )

    # Use LLM judge to verify actual star counts were found.
    from mlflow.genai.judges import make_judge

    os.environ["OPENAI_API_KEY"] = llm_api_key

    judge = make_judge(
        name="github_stars_found",
        instructions=(
            "You are evaluating whether an AI assistant successfully "
            "found GitHub star counts for three repositories.\n\n"
            "The assistant was asked for star counts for: mlflow, "
            "langfuse, and braintrust.\n\n"
            "The assistant's response is:\n"
            "{{ outputs }}\n\n"
            "Does the response include numeric star counts for at "
            "least 2 of the 3 repositories? The numbers should be "
            "plausible (thousands to tens of thousands).\n\n"
            "Return True if star counts are present. Return False "
            "if the assistant failed to find them or only has "
            "placeholder text."
        ),
        feedback_value_type=bool,
    )

    feedback = judge(outputs=text)
    assert feedback.value is True, (
        f"LLM judge: star counts not found.\nRationale: {feedback.rationale}\nOutput: {text[:500]}"
    )
