"""E2E test: Claude SDK executor auto-collects sub-agent results.

Verifies that when the Claude SDK executor spawns a sub-agent,
the workflow auto-collects the results before the parent task
completes. The user sends a single message and gets back the
sub-agent's output — no second message or manual polling needed.

Usage::

    pytest tests/e2e/test_claude_coder_auto_collect.py \
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


def test_single_message_subagent_auto_collect(
    http_client: httpx.Client,
    claude_coder_agent: str,
    llm_api_key: str,
) -> None:
    """
    One message triggers spawn + auto-collect. The response includes
    the sub-agent's review output.

    The user sends a single request asking claude-coder to spawn its
    reviewer sub-agent. The workflow auto-collects the sub-agent
    before the parent task completes. The final response must contain
    the reviewer's actual feedback — not just "I spawned it, check
    back later."

    **What breaks if auto-collect is missing:**

    - Without ``_track_spawn_collect`` after observed tool calls,
      ``spawned_ids`` stays empty → the workflow skips auto-collect
      → the parent completes immediately with "sub-agent is running"
      but no actual review content.
    - The user would need to send a second message to poll for
      results, which defeats the purpose.
    """
    # Single message — ask to spawn reviewer and review a well-known file.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "Use spawn_sub_agents to spawn the 'reviewer' sub-agent "
                "and ask it to review /etc/hosts."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    # Generous timeout: parent spawn + sub-agent execution + auto-collect.
    body = poll_until_terminal(http_client, response_id, timeout=240)
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    text = _extract_all_text(body)

    # Use LLM judge to verify the response contains actual sub-agent
    # review content — not just "I spawned it."
    from mlflow.genai.judges import make_judge

    os.environ["OPENAI_API_KEY"] = llm_api_key

    judge = make_judge(
        name="auto_collect_completeness",
        instructions=(
            "You are evaluating whether an AI assistant received "
            "and presented results from a sub-agent it spawned.\n\n"
            "The assistant was asked to spawn a 'reviewer' sub-agent "
            "to review the /etc/hosts file.\n\n"
            "The assistant's response is:\n"
            "{{ outputs }}\n\n"
            "A PASSING response must:\n"
            "1. Show that spawn_sub_agents was called\n"
            "2. Show that check_sub_agents returned a COMPLETED "
            "status (not 'in_progress')\n"
            "3. Include specific observations about the /etc/hosts "
            "file that came from the reviewer\n\n"
            "A FAILING response:\n"
            "- Says 'the reviewer is still working' or 'in progress'\n"
            "- Only shows the assistant's own analysis (not the "
            "sub-agent's)\n"
            "- Says 'check back later'\n\n"
            "Return True ONLY if the sub-agent completed and its "
            "results are included. Return False otherwise."
        ),
        feedback_value_type=bool,
    )

    feedback = judge(outputs=text)
    assert feedback.value is True, (
        f"LLM judge: response did not contain sub-agent results.\n"
        f"This likely means auto-collect did not wait for the "
        f"sub-agent to finish before the parent completed.\n"
        f"Rationale: {feedback.rationale}\n"
        f"Output: {text[:1000]}"
    )
