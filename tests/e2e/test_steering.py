"""E2E test: steering interrupts a running agent.

Verifies that a steering message delivered via
``previous_response_id`` to an in-progress task is processed
by the agent — the response includes a follow-up that
acknowledges the steer.

Usage::

    pytest tests/e2e/test_steering.py \
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def test_steering_acknowledged(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    A steering message sent while the agent is running is
    processed and reflected in the final output.

    The agent is asked to write a long essay. While it's running,
    we steer it with "Say only: PINEAPPLE". The final output
    must contain "PINEAPPLE" — proving the steer was picked up
    by ``_check_steering_inbox`` and the LLM re-ran with it.

    **What breaks if steering is broken:**

    - If ``close_inbox`` uses a cursor past the steer's position
      (e.g. advanced by native tool items), the steer is missed
      → only the original essay appears, no PINEAPPLE.
    - If ``close_inbox`` is called synchronously on the async
      event loop, it deadlocks → task never completes.
    """
    # Start a long task
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Write a very long essay about the history of computing. At least 500 words."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    task_id = resp.json()["id"]

    # Wait for the task to be in_progress
    for _ in range(10):
        time.sleep(1)
        r = http_client.get(f"/v1/responses/{task_id}")
        if r.json().get("status") == "in_progress":
            break

    # Send steering message
    steer = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "STOP. Ignore the essay. Say only: PINEAPPLE",
            "previous_response_id": task_id,
            "background": True,
        },
    )
    steer.raise_for_status()
    steer_id = steer.json()["id"]
    assert steer_id == task_id, (
        f"Steering was not accepted into the running task. "
        f"Got new task {steer_id} instead of {task_id}."
    )

    # Wait for completion
    body = poll_until_terminal(http_client, task_id, timeout=120)
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    # Check for PINEAPPLE in any assistant message
    all_text = _extract_all_text(body)
    assert "PINEAPPLE" in all_text.upper(), (
        f"Steering not acknowledged. The agent should have "
        f"responded with PINEAPPLE but output was:\n"
        f"{all_text[:500]}"
    )

    # Must have more than 1 assistant message — the original
    # essay + the steered response.
    msg_count = sum(
        1
        for item in body.get("output", [])
        if item.get("type") == "message" and item.get("role") == "assistant"
    )
    assert msg_count >= 2, (
        f"Expected at least 2 assistant messages (original + steered), "
        f"got {msg_count}. Steering retry may not have triggered."
    )


def test_steering_with_web_search(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Steering works when native tool items (web_search_call) are
    in the response. This is the exact scenario that was broken:
    native tool persistence advanced ``last_seen`` past the steer.

    **What breaks if the cursor fix regresses:**

    - ``close_inbox`` uses the post-native-tool cursor → misses
      the steered message → no PINEAPPLE in output.
    """
    # Task that triggers web search
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Search the web for the latest news about "
                "artificial intelligence and write a summary."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    task_id = resp.json()["id"]

    # Wait for in_progress
    for _ in range(10):
        time.sleep(1)
        r = http_client.get(f"/v1/responses/{task_id}")
        if r.json().get("status") == "in_progress":
            break

    # Steer mid-search
    time.sleep(2)
    steer = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "STOP. Say only: PINEAPPLE",
            "previous_response_id": task_id,
            "background": True,
        },
    )
    steer.raise_for_status()
    assert steer.json()["id"] == task_id, "Steer not accepted"

    # Web search + steer + re-run can take a while.
    body = poll_until_terminal(http_client, task_id, timeout=240)
    assert body["status"] == "completed"

    all_text = _extract_all_text(body)
    assert "PINEAPPLE" in all_text.upper(), (
        f"Steering with web search not acknowledged: {all_text[:300]}"
    )


def test_steering_after_completed_starts_new_turn(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    A message sent after the task completes creates a new turn,
    not a steer. Verifies that ``_response_terminal`` detection
    works.
    """
    # Quick task
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "Say hello.",
            "background": True,
        },
    )
    resp.raise_for_status()
    task_id = resp.json()["id"]

    body = poll_until_terminal(http_client, task_id, timeout=30)
    assert body["status"] == "completed"

    # Send follow-up AFTER completion
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "What is 2+2?",
            "previous_response_id": task_id,
            "background": True,
        },
    )
    resp2.raise_for_status()
    task2_id = resp2.json()["id"]

    # Should be a NEW task (not steered into the old one)
    assert task2_id != task_id, (
        "Follow-up after completion should create a new task, not steer into the completed one."
    )

    body2 = poll_until_terminal(http_client, task2_id, timeout=30)
    assert body2["status"] == "completed"
    text = _extract_all_text(body2)
    assert "4" in text, f"Expected answer to 2+2, got: {text[:100]}"


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all assistant output_text blocks.

    :param body: The terminal response body.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)
