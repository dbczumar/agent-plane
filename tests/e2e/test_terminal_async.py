"""End-to-end tests for ``terminal_run(synchronous=False)``.

Drives the async terminal flow through a real LLM on the dedicated
``terminal_test`` fixture agent. Complements
``tests/server/integration/test_terminal_async_integration.py``
(which uses a mocked LLM) by proving the LLM-facing UX works:
the LLM gets a handle back, can poll via ``check_task``, can
cancel via ``cancel_task``, and sees the auto-delivered result.

Requires ``--llm-api-key``. Excluded from default pytest runs.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _get_output_items(
    body: dict[str, Any],
    item_type: str,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter response.output by type + optional tool name.

    :param body: Response body from GET /v1/responses/{id}.
    :param item_type: Item type to keep, e.g. ``"function_call"``.
    :param name: Optional tool name filter.
    :returns: Matching items, original order.
    """
    items = body.get("output", [])
    filtered = [i for i in items if i.get("type") == item_type]
    if name is not None:
        filtered = [i for i in filtered if i.get("name") == name]
    return filtered


def test_async_terminal_returns_handle_and_delivers_result(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """A real LLM can fire ``terminal_run(synchronous=false)``, get a
    ``task_id`` back, and eventually see the stdout auto-delivered.

    The agent is asked to run a quick command asynchronously and
    then acknowledge the completion. If async dispatch is broken:
    - Tool returns stdout inline (not a handle) → no task_id.
    - Auto-delivery broken → agent never sees the result.
    - Kind="terminal" not recognized → drain skips the payload.

    The final response must (a) include a ``terminal_run`` call
    with the async path taken (result payload has ``task_id`` and
    ``status="in_progress"``), (b) include the stdout token
    eventually (in the auto-delivered system message or the
    final assistant text).
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Run the command 'echo async-e2e-token-zzz' in the "
                "background using terminal_run with "
                "synchronous=false. Then wait for it to complete "
                "automatically and confirm you saw the token "
                "'async-e2e-token-zzz' in the result."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=180)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # Verify the tool call was the async variant (result is a
    # handle, not inline stdout).
    fc_items = _get_output_items(final, "function_call", "terminal_run")
    assert len(fc_items) >= 1, (
        f"Expected terminal_run call, got: "
        f"{[i.get('name') for i in _get_output_items(final, 'function_call')]}"
    )
    fco_items = _get_output_items(final, "function_call_output")
    # The first function_call_output corresponding to our
    # terminal_run should be a handle payload, not a stdout dict.
    # Parse the outputs and check at least one has in_progress
    # status with a task_id (the async handle shape).
    has_async_handle = False
    for item in fco_items:
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if parsed.get("status") == "in_progress" and parsed.get("task_id"):
            has_async_handle = True
            break
    assert has_async_handle, (
        f"Expected at least one async handle (status=in_progress + "
        f"task_id) in the function_call_outputs. Got: "
        f"{[i.get('output', '')[:150] for i in fco_items]}"
    )

    # The stdout token must appear somewhere in the conversation
    # (auto-delivered system message, check_task result, or the
    # assistant's final acknowledgement).
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 200},
    )
    items = items_resp.json()["data"]
    all_text_blob = json.dumps(items)
    assert "async-e2e-token-zzz" in all_text_blob, (
        f"Expected the stdout token 'async-e2e-token-zzz' to appear "
        f"somewhere in the conversation (auto-delivered system "
        f"message, check_task result, or assistant text). "
        f"Conversation length: {len(items)} items."
    )


def test_async_terminal_cancel_stops_sleep(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """The LLM can cancel a running async terminal command.

    Agent is instructed to start a long sleep in the background,
    then immediately cancel it. If cancel_task's terminal-kind
    branch is broken, the sleep runs to natural completion and the
    whole task takes ~60s. With cancel working, task finishes in
    a few seconds.

    Verified observationally (total task time) AND by looking for
    the "(terminal) cancelled" system message.
    """
    import time as _time

    start = _time.monotonic()
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Start 'sleep 60' in the background using "
                "terminal_run with synchronous=false. "
                "IMMEDIATELY after that, call cancel_task on the "
                "task_id returned from terminal_run. Then "
                "acknowledge that you cancelled it."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=120)
    elapsed = _time.monotonic() - start
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"
    # If cancel worked, total elapsed should be well under the sleep
    # duration. Conservative bound: 45s (leaves room for LLM latency
    # + auto-delivery). Without cancel working, we'd see ~60s +
    # overhead = 75s+.
    assert elapsed < 45, (
        f"Task took {elapsed:.1f}s — cancel likely didn't interrupt "
        f"the sleep (expected <45s with working cancel; sleep 60 "
        f"naturally is 60+)."
    )

    # Verify the cancelled-terminal system message appears.
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 200},
    )
    items = items_resp.json()["data"]
    system_messages: list[str] = []
    for item in items:
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        for block in item.get("content", []):
            text = block.get("text") or ""
            if text.startswith("[System: task"):
                system_messages.append(text)
    assert any("(terminal) cancelled" in m for m in system_messages), (
        f"Expected a '[System: task X (terminal) cancelled]' system "
        f"message in the conversation, got {system_messages!r}. "
        f"If 'completed' instead, cancel didn't propagate to the "
        f"workflow's status translation."
    )
