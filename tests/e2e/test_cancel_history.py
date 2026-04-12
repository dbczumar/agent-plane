"""End-to-end test for cancellation history markers.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_cancel_history.py \
        --llm-api-key $LLM_API_KEY -v

Exercises:
- Cancelling an in-progress response via the cancel endpoint
- Verifying a cancellation marker is appended to the conversation
- Verifying a follow-up turn sees the cancellation context
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _wait_for_in_progress(
    client: httpx.Client,
    response_id: str,
    timeout: float = 60,
) -> None:
    """
    Poll until the response transitions to ``in_progress``.

    :param client: HTTP client.
    :param response_id: The response ID to poll.
    :param timeout: Max seconds to wait.
    :raises AssertionError: If not in_progress within timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] == "in_progress":
            return
        if body["status"] in ("completed", "failed", "cancelled"):
            raise AssertionError(
                f"Response reached terminal state {body['status']} before in_progress"
            )
        time.sleep(0.3)
    raise AssertionError(f"Response {response_id} didn't reach in_progress within {timeout}s")


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


def test_cancel_appends_history_marker_and_followup_sees_it(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Cancel an archer response and verify the follow-up sees the
    cancellation in conversation history.

    Flow:
    1. Send a broad question to archer (background) so it takes
       a while.
    2. Wait for ``in_progress``, then cancel.
    3. Verify the conversation has a cancellation marker item.
    4. Send a follow-up asking whether the previous response was
       cancelled.
    5. Assert the follow-up's output mentions the cancellation.

    **What breaks if wrong:**

    - If ``_append_cancellation_item`` is not called after cancel,
      the conversation has no marker and the follow-up agent has
      no awareness of the interruption.
    - If the marker text is missing or malformed, the follow-up
      LLM won't know a cancellation happened.
    """
    # Step 1: send a broad question that will take time.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Write a detailed 2000-word essay about the history "
                "of the Byzantine Empire, covering all major emperors "
                "and key events from 330 AD to 1453 AD."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]
    conv_id = body["conversation"]["id"]

    # Step 2: wait for it to start, then cancel.
    _wait_for_in_progress(http_client, response_id, timeout=60)
    cancel_resp = http_client.post(f"/v1/responses/{response_id}/cancel")
    cancel_resp.raise_for_status()
    assert cancel_resp.json()["status"] == "cancelled"

    # Step 3: verify the conversation has the cancellation marker.
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"order": "desc", "limit": 5},
    )
    items_resp.raise_for_status()
    items = items_resp.json()["data"]
    # Find the cancellation marker — a user message with
    # "interrupted" in its text.
    cancellation_items = [
        item
        for item in items
        if item.get("type") == "message"
        and item.get("role") == "user"
        and any("interrupted" in c.get("text", "") for c in item.get("content", []))
    ]
    assert len(cancellation_items) == 1, (
        f"Expected exactly 1 cancellation marker, found {len(cancellation_items)}. Items: {items}"
    )

    # Step 4: send a follow-up asking about the cancellation.
    followup_resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Was the previous assistant response cancelled or "
                "interrupted? Answer YES or NO, followed by a brief "
                "explanation of how you know."
            ),
            "previous_response_id": response_id,
            "background": True,
        },
    )
    followup_resp.raise_for_status()
    followup_id = followup_resp.json()["id"]

    # Step 5: wait for the follow-up to complete.
    followup_body = poll_until_terminal(
        http_client,
        followup_id,
        timeout=120,
    )
    assert followup_body["status"] == "completed", (
        f"Follow-up failed: {followup_body.get('error')}"
    )

    # The follow-up should acknowledge the cancellation.
    text = _extract_all_text(followup_body).upper()
    assert "YES" in text, (
        f"Expected the follow-up to acknowledge the cancellation with 'YES'. Got: {text[:500]}"
    )
