"""E2E test: Claude SDK executor spawning a sub-agent.

Verifies that the Claude SDK executor can call ``spawn_sub_agents``
(a server-side agent-plane tool) through the unified ``call_tool``
callback, and that the sub-agent executes and returns results.

Usage::

    pytest tests/e2e/test_claude_coder_subagent.py \
        --llm-api-key $(cat /tmp/mykey) -v
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


def _has_tool_call_named(body: dict[str, Any], name: str) -> bool:
    """
    Check if the output contains a function_call with a matching name.

    :param body: The terminal response body.
    :param name: Tool name to search for (exact match).
    :returns: True if found.
    """
    for item in body.get("output", []):
        if item.get("type") == "function_call" and item.get("name") == name:
            return True
    return False


def test_claude_coder_spawns_reviewer(
    http_client: httpx.Client,
    claude_coder_agent: str,
    llm_api_key: str,
) -> None:
    """
    The Claude SDK executor spawns a reviewer sub-agent via
    ``spawn_sub_agents`` and collects the result.

    This tests the full ``call_tool`` routing path: the SDK calls
    ``spawn_sub_agents`` through MCP → the unified ``call_tool``
    callback routes to ``ToolManager`` (server-side) → ``SpawnTool``
    creates a child DBOS workflow → the reviewer runs and produces
    output → the parent auto-collects and includes the result.

    **What breaks if the feature is wrong:**

    - If ``call_tool`` doesn't route ``spawn_sub_agents`` to
      ``ToolManager``, the SDK tries to tunnel it to the client
      (which doesn't know how to spawn agents) → hangs or errors.
    - If the OpenAI wrapper schema extraction is broken, the tool
      is registered with an empty name → SDK can't call it.
    - If the sub-agent spec isn't found in the spec tree, the
      child workflow fails with ``LookupError``.
    """
    # Create a small file for the reviewer to review.
    resp_setup = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "Create a file /tmp/review_target.py with this content:\n\n"
                "def divide(a, b):\n"
                "    return a / b\n\n"
                "def process(items):\n"
                "    for i in range(len(items)):\n"
                "        print(items[i])\n"
            ),
            "background": True,
        },
    )
    resp_setup.raise_for_status()
    setup_id = resp_setup.json()["id"]
    body_setup = poll_until_terminal(http_client, setup_id, timeout=60)
    assert body_setup["status"] == "completed", (
        f"Setup failed: {body_setup.get('error')}"
    )

    # Now ask claude-coder to delegate a review to the reviewer sub-agent.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "You have a sub-agent called 'reviewer'. Use the "
                "spawn_sub_agents tool to spawn it and ask it to "
                "review /tmp/review_target.py. Then collect the "
                "results with check_sub_agents."
            ),
            "background": True,
            "previous_response_id": setup_id,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=180)
    assert body["status"] == "completed", (
        f"Sub-agent task failed: {body.get('error')}"
    )

    text = _extract_all_text(body)
    assert len(text) > 50, (
        f"Expected substantial review output, got: {text!r}"
    )

    # Verify spawn_sub_agents was called. The tool appears in output
    # as an MCP tool call (mcp__agent_plane__spawn_sub_agents) or
    # as a ToolCallObserved (spawn_sub_agents). Check both.
    spawned = (
        _has_tool_call_named(body, "spawn_sub_agents")
        or _has_tool_call_named(body, "mcp__agent_plane__spawn_sub_agents")
    )
    assert spawned, (
        "Expected spawn_sub_agents tool call in output. "
        "Claude may have reviewed directly instead of delegating. "
        f"Tool calls found: {[i.get('name') for i in body.get('output', []) if i.get('type') == 'function_call']}"
    )

    # Use LLM judge to verify the review quality.
    from mlflow.genai.judges import make_judge

    os.environ["OPENAI_API_KEY"] = llm_api_key

    judge = make_judge(
        name="subagent_review_quality",
        instructions=(
            "You are evaluating whether an AI coding assistant "
            "successfully delegated a code review to a sub-agent "
            "and returned meaningful results.\n\n"
            "The assistant was asked to use its 'reviewer' "
            "sub-agent to review a Python file containing a "
            "divide function (no zero-check) and a process "
            "function (using range(len()) antipattern).\n\n"
            "The assistant's response is:\n"
            "{{ outputs }}\n\n"
            "Does the response contain actual code review "
            "feedback that identifies issues in the code? "
            "The review should mention at least one real "
            "issue (division by zero risk, range(len) "
            "antipattern, or similar).\n\n"
            "Return True if the response contains substantive "
            "code review feedback, False if it's generic, "
            "empty, or just says it couldn't complete the task."
        ),
        feedback_value_type=bool,
    )

    feedback = judge(outputs=text)
    assert feedback.value is True, (
        f"LLM judge: review output was not substantive.\n"
        f"Rationale: {feedback.rationale}\n"
        f"Output: {text[:500]}"
    )
