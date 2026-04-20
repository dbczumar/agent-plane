"""
E2E for steering responsiveness during the end-of-turn async drain.

Reproduces the ``ap chat``-visible bug where a user typing
mid-flight (while the parent workflow is blocked on
``_drain_async_completions(block_for_one=True)`` waiting for
async client tools to finish) saw no response until all the
tasks completed. The drain's ``DBOS.recv`` had no signal for
steering messages — they landed in the conversation via
``try_deliver`` but the workflow wasn't polling.

Fix (committed separately): the blocking drain now polls the
conversation store every ``_STEERING_POLL_INTERVAL_S`` (1 s)
alongside the DBOS recv; if steering is detected, the drain
returns early and the outer loop iterates so the LLM sees the
new message.

Test strategy:
- Use the ``client-tool-cancellation-message-test`` fixture
  (parent agent that calls one async_compute then waits).
- Start the parent with a query.
- Once the handle FCO appears (proves the client_tool was
  dispatched and the parent has entered the drain wait),
  wall-clock-record a timestamp and POST a steering message.
- Assert: a NEW assistant message (not the null-text
  placeholder the agent emits alongside the tool call) appears
  in the parent's conversation within ``_STEERING_MAX_LATENCY_S``.
  Without the fix this would take ~1 h (the server's client_tool
  holder workflow timeout); with the fix it lands within the
  steering-poll cadence (~1-2 s).
- The client_tool task is never PATCHed, so it stays
  ``in_progress`` throughout. We clean up by cancelling the
  parent at the end.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_steering_during_async_drain_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_FIXTURE = _FIXTURES_DIR / "client-tool-cancellation-message-test"

# Bound on how long the steering message may sit unseen by the
# agent after being POSTed. The server's steering poll runs
# every _STEERING_POLL_INTERVAL_S (=1 s) and the LLM needs
# another round-trip to emit the acknowledgement, so 15 s is
# a comfortable ceiling that still demonstrates the fix works
# (without it the drain would wait up to 1 h for the
# client_tool holder workflow to time out).
_STEERING_MAX_LATENCY_S = 15.0


_ASYNC_CLIENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "async_compute",
        "description": (
            "Long-running client-side computation. Always call with "
            "synchronous=false. The result is delivered later as a "
            "system message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "Echo this string back as the result.",
                },
                "synchronous": {
                    "type": "boolean",
                    "description": (
                        "MUST be set to false. Dispatches as a "
                        "background task and returns a handle."
                    ),
                },
            },
            "required": ["value", "synchronous"],
        },
    },
}


def _upload(http_client: httpx.Client, agent_dir: Path) -> str:
    """
    Upload an agent bundle from a directory tree.

    :param http_client: HTTP client pointed at the live server.
    :param agent_dir: Directory containing config.yaml.
    :returns: The agent's name.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(agent_dir), arcname=".")
        bundle_path = tmp.name
    try:
        with open(bundle_path, "rb") as f:
            resp = http_client.post(
                "/api/agents",
                files={
                    "bundle": (
                        "agent.tar.gz",
                        f,
                        "application/gzip",
                    ),
                },
            )
        if resp.status_code == 409:
            return agent_dir.name
        resp.raise_for_status()
        return resp.json()["name"]
    finally:
        Path(bundle_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def steering_drain_test_agent(http_client: httpx.Client) -> str:
    """Upload the shared async-client-tool fixture."""
    return _upload(http_client, _FIXTURE)


def _items(http_client: httpx.Client, conv_id: str) -> list[dict[str, Any]]:
    """Fetch all conversation items in store order."""
    resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    resp.raise_for_status()
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


def _wait_for_handle(
    http_client: httpx.Client,
    conv_id: str,
    tool_name: str,
    timeout_s: float = 60.0,
) -> str:
    """
    Poll until the async client-tool handle FCO appears.

    The handle shows up immediately after the workflow
    dispatches the tool and is the signal that the parent has
    now moved on to its next iteration and will shortly hit
    the blocking drain — the scenario we want to probe.

    :param http_client: HTTP client.
    :param conv_id: Conversation id to scan.
    :param tool_name: Name on the preceding function_call.
    :param timeout_s: Max seconds to wait for the handle.
    :returns: The handle's ``task_id``.
    :raises AssertionError: On timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        items = _items(http_client, conv_id)
        last_call_name: str | None = None
        for item in items:
            if item.get("type") == "function_call":
                last_call_name = item.get("name")
            elif item.get("type") == "function_call_output":
                if last_call_name != tool_name:
                    continue
                try:
                    handle = json.loads(item.get("output") or "")
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(handle, dict)
                    and handle.get("kind") == "client_tool"
                    and handle.get("task_id")
                ):
                    return str(handle["task_id"])
        time.sleep(0.25)
    raise AssertionError(
        f"No async client-tool handle appeared in conv {conv_id} within {timeout_s}s"
    )


def _count_assistant_text_items(items: list[dict[str, Any]]) -> int:
    """
    Count assistant ``output_text`` items with non-empty text.

    Null/empty-text items are placeholders the LLM emits
    alongside tool calls — they don't count as a real
    "response" to anything. A new non-empty assistant
    message is the signal the agent has reacted to the
    steering.

    :param items: Raw items list from ``list_items``.
    :returns: Number of non-empty assistant text items.
    """
    count = 0
    for item in items:
        if item.get("role") != "assistant":
            continue
        content = item.get("content") or []
        if not isinstance(content, list) or not content:
            continue
        first = content[0]
        if not isinstance(first, dict) or first.get("type") != "output_text":
            continue
        text = first.get("text") or ""
        if text.strip():
            count += 1
    return count


def test_steering_breaks_blocked_async_drain(
    http_client: httpx.Client,
    steering_drain_test_agent: str,
) -> None:
    """
    The agent must react to a steering message within
    ``_STEERING_MAX_LATENCY_S`` seconds, even while the
    end-of-turn async-drain is blocked waiting for a
    client_tool task the test never PATCHes.

    Specifically tests:
    - ``_drain_async_completions(block_for_one=True)``'s
      heartbeat loop polls the conversation store every
      ``_STEERING_POLL_INTERVAL_S`` for new items past the
      pre-drain cursor.
    - When a steering ``POST /v1/responses`` (with
      ``previous_response_id``) appends a new message, the
      drain returns early.
    - The outer ``_run_agent_loop`` picks up the new message
      via its next-iteration ``_sync_history`` and sends the
      LLM another round-trip.

    Failure modes this test catches:
    - Drain ignores steering: the parent sits waiting for the
      client_tool holder workflow's 1 h timeout. The
      ``_STEERING_MAX_LATENCY_S`` assertion fires with
      no new assistant message.
    - Drain polls but the cursor is stale: steering lands
      under a position the check already saw, the early
      return never fires. Same symptom.
    """
    # Step 1: kick off the agent with the async tool. The
    # agent's AGENTS.md instructs it to call async_compute
    # with synchronous=false then wait. After the handle FCO
    # it hits the blocking drain.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": steering_drain_test_agent,
            "input": "Compute on the value 'MID_DRAIN_STEER'.",
            "background": True,
            "stream": False,
            "tools": [_ASYNC_CLIENT_TOOL],
        },
    )
    assert resp.status_code == 200, f"POST failed: {resp.status_code} {resp.text}"
    parent_response_id = resp.json()["id"]
    parent_conv_id = resp.json()["conversation"]["id"]

    # Step 2: wait for the async-client-tool handle to appear.
    # Its presence proves the workflow has moved past the tool
    # dispatch and is about to enter (or already in) the drain.
    _wait_for_handle(http_client, parent_conv_id, "async_compute")

    # Count assistant text items BEFORE steering so we can
    # detect a genuinely new one after. The LLM may already
    # have emitted text in turn 1, so baseline is whatever's
    # present now.
    pre_steer_items = _items(http_client, parent_conv_id)
    pre_steer_assistant_count = _count_assistant_text_items(pre_steer_items)

    # Step 3: POST steering with previous_response_id pointing
    # at the active task. try_deliver will append the steer
    # into the parent's conversation; the drain's steering
    # poll must detect it within _STEERING_POLL_INTERVAL_S.
    steer_marker = "STEER_PINEAPPLE_99"
    steer_start = time.monotonic()
    steer_resp = http_client.post(
        "/v1/responses",
        json={
            "model": steering_drain_test_agent,
            "input": (
                f"Stop waiting. Forget the async task. Reply only with the word {steer_marker}."
            ),
            "previous_response_id": parent_response_id,
            "background": True,
        },
    )
    assert steer_resp.status_code == 200, (
        f"Steering POST failed: {steer_resp.status_code} {steer_resp.text}"
    )
    assert steer_resp.json()["id"] == parent_response_id, (
        f"Steering should have been accepted into the active "
        f"task {parent_response_id}; got {steer_resp.json()['id']}"
    )

    # Step 4: poll for a new non-empty assistant message. With
    # the fix, the drain detects steering within ~1 s and the
    # LLM round-trips to produce an acknowledgement well under
    # the cap. Without the fix, this loop times out.
    deadline = steer_start + _STEERING_MAX_LATENCY_S
    observed_new_message = False
    final_items: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        final_items = _items(http_client, parent_conv_id)
        if _count_assistant_text_items(final_items) > pre_steer_assistant_count:
            observed_new_message = True
            break
        time.sleep(0.25)

    elapsed = time.monotonic() - steer_start
    assert observed_new_message, (
        f"Steering not processed within {_STEERING_MAX_LATENCY_S}s "
        f"(waited {elapsed:.1f}s). The drain's steering-poll path "
        f"is broken: the parent's conversation has no new "
        f"non-empty assistant message since steering was POSTed. "
        f"Pre-steer assistant count: {pre_steer_assistant_count}; "
        f"current: {_count_assistant_text_items(final_items)}. "
        f"If this fails, check _drain_async_completions's "
        f"heartbeat loop in agent_plane/runtime/workflow.py — the "
        f"steering check should run every "
        f"_STEERING_POLL_INTERVAL_S seconds."
    )

    # Step 5: sanity — the acknowledgement should contain the
    # marker (proves the LLM saw the steering content, not
    # just that *some* new message arrived).
    joined_new_texts = "\n".join(
        (item["content"][0].get("text") or "")
        for item in final_items
        if item.get("role") == "assistant"
        and isinstance(item.get("content"), list)
        and item["content"]
        and item["content"][0].get("type") == "output_text"
    )
    assert steer_marker in joined_new_texts, (
        f"The LLM's new reply should acknowledge the steering "
        f"content (look for marker {steer_marker!r}). This "
        f"catches a regression where the drain breaks on some "
        f"unrelated conversation item but the steering content "
        f"isn't actually delivered to the LLM's next call. "
        f"Joined assistant texts: {joined_new_texts[:800]!r}"
    )

    # Cleanup: cancel the parent so the client_tool holder
    # workflow stops waiting on its 1 h timeout.
    http_client.post(f"/v1/responses/{parent_response_id}/cancel")
