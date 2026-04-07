"""E2E test: Claude SDK executor multi-turn tool call awareness.

Verifies that the Claude SDK subprocess persists across tasks and
retains awareness of tool calls from prior turns. The test sends
two turns to the claude-coder agent:

1. Ask it to fetch MLflow's GitHub star count (triggers a Bash tool call).
2. Ask it what tool it used (requires context from turn 1).

An MLflow LLM judge evaluates the second turn's response to confirm
that Claude correctly identified the tool(s) used.

Usage::

    pytest tests/e2e/test_claude_coder_multi_turn.py \
        --llm-api-key $(cat /tmp/openai_key) \
        --anthropic-api-key $(cat /tmp/anthropic_key) -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body from GET /v1/responses/{id}.
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


def _has_tool_call_named(body: dict[str, Any], substring: str) -> bool:
    """
    Check if the response output contains a function_call with
    a name containing ``substring`` (case-insensitive).

    :param body: The terminal response body.
    :param substring: Substring to match against tool names,
        e.g. ``"bash"`` or ``"Bash"``.
    :returns: True if any matching function_call found.
    """
    lower = substring.lower()
    for item in body.get("output", []):
        if item.get("type") == "function_call":
            name = item.get("name", "")
            if lower in name.lower():
                return True
    return False


def test_claude_coder_remembers_tool_calls(
    http_client: httpx.Client,
    claude_coder_agent: str,
    llm_api_key: str,
) -> None:
    """
    Multi-turn test: Claude SDK subprocess retains tool call context.

    Turn 1: "How many GitHub stars does MLflow have?" — expects
    a Bash tool call (e.g. ``gh repo view``) and a numeric answer.

    Turn 2: "What tools did you use to find that out?" — expects
    Claude to name the tool(s) it used. An MLflow LLM judge
    evaluates whether the response demonstrates genuine awareness
    of its own tool usage (not a generic or hallucinated answer).

    **What breaks if the feature is wrong:**

    - If the SDK subprocess is not persistent across tasks, Claude
      loses all context and either hallucinates or says "I didn't
      use any tools."
    - If ``_build_history_prompt`` garbles function_call items,
      the subprocess sees corrupted context and produces no output.
    - If the per-conversation event loop is not reused, the SDK
      client is orphaned and the second turn hangs or errors.
    """
    # ── Turn 1: fetch GitHub stars ──────────────────────────
    resp_1 = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "How many GitHub stars does the mlflow/mlflow "
                "repository have? Use the gh CLI to find out."
            ),
            "background": True,
        },
    )
    resp_1.raise_for_status()
    response_1_id = resp_1.json()["id"]

    body_1 = poll_until_terminal(http_client, response_1_id, timeout=120)
    assert body_1["status"] == "completed", f"Turn 1 failed: {body_1.get('error', 'unknown')}"

    # Sanity: turn 1 should have used a Bash tool call.
    text_1 = _extract_all_text(body_1)
    assert _has_tool_call_named(body_1, "bash") or "star" in text_1.lower(), (
        f"Turn 1 didn't seem to fetch stars. Output: {text_1[:500]}"
    )

    # ── Turn 2: ask about tools used ────────────────────────
    resp_2 = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": "What tools did you use to find that out?",
            "background": True,
            "previous_response_id": response_1_id,
        },
    )
    resp_2.raise_for_status()
    response_2_id = resp_2.json()["id"]

    body_2 = poll_until_terminal(http_client, response_2_id, timeout=120)
    assert body_2["status"] == "completed", f"Turn 2 failed: {body_2.get('error', 'unknown')}"

    text_2 = _extract_all_text(body_2)
    assert len(text_2) > 10, f"Turn 2 produced no meaningful output. Text: {text_2!r}"

    # ── LLM judge: did Claude articulate the tools used? ────
    #
    # The judge uses gpt-4.1-mini via OpenAI. Set the key in this
    # process (the server subprocess already has it, but the judge
    # runs in the test process).
    import os

    from mlflow.genai.judges import make_judge

    os.environ["OPENAI_API_KEY"] = llm_api_key

    judge = make_judge(
        name="tool_awareness",
        instructions=(
            "You are evaluating whether an AI assistant correctly "
            "identified the tools it used in a previous conversation "
            "turn.\n\n"
            "The assistant was asked to fetch the GitHub star count "
            "for the mlflow/mlflow repository. It used the Bash tool "
            "to run a command like `gh repo view mlflow/mlflow`. "
            "Then the user asked 'What tools did you use?'\n\n"
            "The assistant's response is:\n"
            "{{ outputs }}\n\n"
            "Does the response demonstrate that the assistant knows "
            "it used a tool (e.g. Bash, shell, command line, gh CLI, "
            "or similar)? The assistant does NOT need to use the exact "
            "word 'Bash' — any indication that it ran a command or "
            "used a tool to fetch the data counts as awareness.\n\n"
            "Return True if the assistant shows genuine tool "
            "awareness, False if it denies using tools, gives a "
            "generic answer, or hallucinates."
        ),
        feedback_value_type=bool,
    )

    feedback = judge(outputs=text_2)
    assert feedback.value is True, (
        f"LLM judge ruled that Claude did NOT demonstrate tool "
        f"awareness.\n"
        f"Judge rationale: {feedback.rationale}\n"
        f"Claude's response: {text_2}"
    )
