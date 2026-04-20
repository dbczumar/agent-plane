"""
E2E for audit fix #6 (option d) — cancelled client_tool tasks
must leave a ``[System: task X (client_tool) cancelled]``
message in the parent's conversation so the LLM sees the
cancellation on replay (``previous_response_id``).

Setup (intentionally simple — no sub-agent layer so the test
focuses purely on the cancellation-persistence invariant):

- Parent agent calls the request-supplied client tool
  ``async_compute`` with ``synchronous: false`` — server
  dispatches a ``kind="client_tool"`` task and persists the
  handle FCO with ``status: "in_progress"``.
- The test does NOT PATCH ``async_tool_results``.
- The test ``POST /v1/responses/{parent}/cancel`` to cancel
  the parent. ``cancel_pending_child_tools`` reaps the
  client_tool, the holder workflow's except block fires (sends
  ``async_work_complete`` to the parent's drain — but the
  parent is already terminating so the drain payload is
  stranded), and audit fix #6's persist-direct path writes a
  ``[System: ...]`` user-role message into the parent's
  conversation.

Today (without the fix):
- The drain message is the only signal; it gets stranded;
  the parent's conversation is left with the
  ``"in_progress"`` handle FCO and nothing else. The LLM
  resuming the conversation has no way to know the task was
  cancelled.

After the fix:
- The parent's conversation contains
  ``[System: task <id> (client_tool) cancelled]``. The LLM
  reads it on the next ``previous_response_id`` resumption
  and knows the dispatch is dead.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_client_tool_cancellation_message_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import upload_agent

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_FIXTURE = _FIXTURES_DIR / "client-tool-cancellation-message-test"


# Schema declares ``synchronous`` in properties so the server's
# ``_wants_async_dispatch`` honors the per-call flag. ``required``
# forces the LLM to set it (otherwise the tool would be invoked
# via the legacy sync path — wrong code path).
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
                        "background task and returns a "
                        "{task_id, kind: 'client_tool'} handle "
                        "immediately."
                    ),
                },
            },
            "required": ["value", "synchronous"],
        },
    },
}


@pytest.fixture(scope="session")
def cancellation_message_test_agent(http_client: httpx.Client) -> str:
    """Upload the cancellation-message E2E fixture."""
    return upload_agent(http_client, _FIXTURE)


def _items(http_client: httpx.Client, conv_id: str) -> list[dict[str, Any]]:
    """
    Fetch all items in a conversation in store order.

    :param http_client: HTTP client.
    :param conv_id: Conversation id.
    :returns: List of conversation item dicts.
    """
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
    timeout_s: float = 90.0,
) -> str:
    """
    Poll a conversation for the async client-tool handle FCO.

    :param http_client: HTTP client.
    :param conv_id: Conversation id to scan.
    :param tool_name: Tool name on the preceding function_call.
    :param timeout_s: Max seconds to wait.
    :returns: The handle's ``task_id``.
    :raises AssertionError: On timeout — the LLM never made the
        async call (probably misread the AGENTS.md instructions).
    """
    deadline = time.monotonic() + timeout_s
    last_summary: list[str] = []
    while time.monotonic() < deadline:
        items = _items(http_client, conv_id)
        last_call_name: str | None = None
        last_summary = [f"{item.get('type')!r} name={item.get('name', '-')!r}" for item in items]
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
        time.sleep(0.5)
    raise AssertionError(
        f"No async client-tool handle for {tool_name!r} appeared in "
        f"conversation {conv_id} within {timeout_s}s. Items so far: " + ", ".join(last_summary)
    )


def _wait_for_terminal(
    http_client: httpx.Client,
    response_id: str,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """
    Poll a response until it reaches a terminal status.

    :param http_client: HTTP client.
    :param response_id: Response id to poll.
    :param timeout_s: Max seconds to wait.
    :returns: The terminal response body.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = http_client.get(f"/v1/responses/{response_id}").json()
        if body["status"] in ("completed", "failed", "cancelled", "incomplete"):
            return body
        time.sleep(0.5)
    raise AssertionError(f"Response {response_id} did not reach terminal within {timeout_s}s")


def test_cancelled_client_tool_persists_system_message_in_conversation(
    http_client: httpx.Client,
    cancellation_message_test_agent: str,
) -> None:
    """
    Audit fix #6 (d): when a client_tool task is cancelled
    (parent /cancel here), the parent's conversation must
    include a ``[System: task X (client_tool) cancelled]``
    user-role message so the LLM sees the cancellation on
    replay via ``previous_response_id``.

    Failure modes this test catches:

    - Persist-direct path missing entirely: no ``[System: ...]``
      message in the parent's conv. The handle FCO's stale
      ``"status": "in_progress"`` is the only trace; the LLM
      resuming the conversation can't tell the task is dead.
    - Wrong status: persist runs but writes ``completed`` /
      ``failed`` instead of ``cancelled`` — the LLM would think
      the task succeeded with empty output.
    - Wrong target conversation: the message lands somewhere
      other than the parent's conv (e.g., the client_tool
      task's own conv) — the LLM never sees it on replay.
    """
    # Step 1 — POST: parent calls async_compute(synchronous=false).
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": cancellation_message_test_agent,
            "input": "Compute on the value 'CANCEL_ME'.",
            "background": True,
            "stream": False,
            "tools": [_ASYNC_CLIENT_TOOL],
        },
    )
    assert resp.status_code == 200, f"POST failed: {resp.status_code} {resp.text}"
    parent_response_id = resp.json()["id"]
    parent_conv_id = resp.json()["conversation"]["id"]

    # Step 2 — wait for the handle FCO to appear.
    client_tool_task_id = _wait_for_handle(http_client, parent_conv_id, "async_compute")

    # Step 3 — DO NOT PATCH. Cancel the parent instead. This
    # invokes the route's cancel handler →
    # cancel_pending_child_tools(parent) → cancels the
    # client_tool's holder workflow → audit fix #6 persists
    # the [System: ...] cancelled message in the parent's conv.
    cancel_resp = http_client.post(f"/v1/responses/{parent_response_id}/cancel")
    assert cancel_resp.status_code == 200, (
        f"Cancel failed: {cancel_resp.status_code} {cancel_resp.text}"
    )

    # Step 4 — wait for the parent to reach terminal.
    parent_body = _wait_for_terminal(http_client, parent_response_id)
    assert parent_body["status"] == "cancelled", (
        f"Parent should be cancelled; got status={parent_body['status']!r}"
    )

    # Step 5 — load-bearing assertion: the parent's conv now
    # contains a [System: task <client_tool_task_id>
    # (client_tool) cancelled] user-role message.
    items = _items(http_client, parent_conv_id)
    system_msgs = [
        (item["content"][0].get("text") or "")
        for item in items
        if item.get("role") == "user"
        and isinstance(item.get("content"), list)
        and item["content"]
        and item["content"][0].get("type") == "input_text"
        and (item["content"][0].get("text") or "").startswith("[System: task ")
    ]
    matching = [
        msg
        for msg in system_msgs
        if client_tool_task_id in msg and "(client_tool)" in msg and "cancelled" in msg
    ]
    assert len(matching) == 1, (
        f"AUDIT FIX #6 (d): expected exactly one "
        f"[System: task {client_tool_task_id} (client_tool) cancelled] "
        f"message in the parent's conversation; got {len(matching)} "
        f"matching out of {len(system_msgs)} system messages total. "
        f"Without this, the LLM resuming via previous_response_id "
        f"would only see the stale 'in_progress' handle FCO and have "
        f"no way to know the task was cancelled. "
        f"All system messages: {system_msgs!r}"
    )

    # Step 6 — sanity: the original handle FCO is still
    # present (we don't mutate history; the system message is
    # additive). Catches a regression where someone implements
    # option (a) instead of (d) — overwriting the FCO would
    # lose the dispatch-time record.
    handles = [
        item.get("output", "") for item in items if item.get("type") == "function_call_output"
    ]
    handle_with_id = [h for h in handles if isinstance(h, str) and client_tool_task_id in h]
    assert handle_with_id, (
        f"Original handle FCO must still be present (history is "
        f"append-only). Found FCOs: {handles!r}"
    )
    parsed = json.loads(handle_with_id[0])
    assert parsed.get("status") == "in_progress", (
        f"Audit fix #6 (d) is the additive-message option — the "
        f"original handle's dispatch-time status should remain "
        f"'in_progress' (history is append-only). If this is now "
        f"'cancelled', someone switched to option (a) "
        f"(mutate-in-place); update the test accordingly. "
        f"Got: {parsed!r}"
    )
