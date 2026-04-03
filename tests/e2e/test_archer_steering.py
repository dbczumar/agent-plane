"""End-to-end test for steering during auto-collect.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_archer_steering.py \
        --llm-api-key $(cat /tmp/mykey) -v

Exercises:
- Sub-agent spawning with real LLM (archer → summarizer)
- Steering message delivered while auto-collect polls
- Agent processes steering in subsequent LLM turn
"""

from __future__ import annotations

import time

import httpx

from tests.e2e.conftest import poll_until_terminal


def _wait_for_spawn(
    client: httpx.Client,
    response_id: str,
    timeout: float = 120,
) -> None:
    """
    Poll until ``spawn_sub_agents`` appears in the response output.

    This proves the parent agent spawned a sub-agent. At this
    point the sub-agent is running and auto-collect will engage
    once the parent's LLM turn finishes.

    :param client: HTTP client.
    :param response_id: The response ID to poll.
    :param timeout: Max seconds to wait.
    :raises AssertionError: If no spawn call within timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        for item in body.get("output", []):
            if item.get("type") == "function_call" and item.get("name") == "spawn_sub_agents":
                return
        if body["status"] in ("completed", "failed"):
            raise AssertionError(
                f"Response completed without spawning a sub-agent. Output: {body.get('output')}"
            )
        time.sleep(0.5)
    raise AssertionError(f"spawn_sub_agents not found in output within {timeout}s")


def test_steering_during_auto_collect(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Steer the archer agent while it auto-collects a sub-agent.

    Flow:
    1. Ask archer to spawn summarizer for a broad topic.
    2. Wait for the spawn_sub_agents call in output.
    3. Send a steering message ("Actually, just say PINEAPPLE").
    4. Wait for completion.
    5. Assert: the steering was accepted (same response ID) and
       the word "PINEAPPLE" appears in the final output — proving
       the agent processed the steering in a subsequent LLM turn.

    Before the fix, auto-collect blocked indefinitely in a
    polling loop, so the steering message was either rejected
    (inbox closed) or silently skipped (cursor advancement bug).

    Note: with real LLMs, timing is non-deterministic. If the
    sub-agent finishes before the steering arrives, auto-collect
    exits normally and the steering becomes a post-completion
    turn (delivered to a new task). The test handles both paths:
    it only asserts on the PINEAPPLE content when steering was
    accepted into the running task (same response ID).
    """
    # Step 1: spawn a sub-agent with a broad topic to give
    # auto-collect time to engage.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use spawn_sub_agents to spawn the summarizer. "
                "Tell it to summarize the complete history of the "
                "Roman Empire from founding to fall. "
                "Wait for it to finish before replying."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]

    # Step 2: wait for spawn to happen.
    _wait_for_spawn(http_client, response_id, timeout=120)

    # Step 3: steer — ask for a specific, easily verifiable word.
    steer_resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Actually, STOP. Ignore the summarizer's output. "
                "Just reply with the single word PINEAPPLE and "
                "nothing else."
            ),
            "previous_response_id": response_id,
        },
    )
    steer_resp.raise_for_status()
    steer_body = steer_resp.json()
    steered_into_running = steer_body["id"] == response_id

    # Step 4: wait for the task to complete (whichever ID).
    final_id = steer_body["id"]
    final = poll_until_terminal(http_client, final_id, timeout=300)

    # Step 5: assert.
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )

    if steered_into_running:
        # Steering was accepted into the running task — the
        # agent must have processed it. Check that PINEAPPLE
        # appears in the output text.
        text_items = [item for item in final["output"] if item.get("type") == "message"]
        all_text = " ".join(
            c.get("text", "") for item in text_items for c in item.get("content", [])
        ).upper()
        # The LLM should have produced "PINEAPPLE" in at least
        # one of its output messages. If it didn't, either the
        # steering was silently dropped (the original bug) or
        # the LLM disobeyed the instruction.
        assert "PINEAPPLE" in all_text, (
            "Steering was accepted (same response ID) but "
            "PINEAPPLE never appeared in the output. The "
            "auto-collect poll loop may have skipped the "
            f"steering message. Output text: {all_text[:500]}"
        )
