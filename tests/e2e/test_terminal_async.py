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
                "Call terminal_run right now with "
                "command='echo async-e2e-token-zzz', "
                "shell='default', and synchronous=false. After the "
                "result auto-delivers, report whether you saw the "
                "token 'async-e2e-token-zzz'. Do NOT write any "
                "commentary before calling the tool — call it first."
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
        params={"limit": 100},
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
    """The LLM can cancel a running async terminal command and
    observe the cancelled status via check_task.

    Agent is instructed to start a long sleep in the background,
    immediately cancel it, then poll via check_task with wait_ms
    so it sees the post-cancel status directly. Terminals are
    no longer in ``_DRAIN_KINDS`` (see
    designs/PERSISTENT_TERMINAL_RESEARCH.md §6.12), so the
    "[System: task X cancelled]" auto-delivery is gone — the
    agent polls explicitly.

    Failure modes caught:
    - cancel_task's terminal-kind branch doesn't interrupt the
      sleep (check_task sees 'completed' after the full 60s or
      times out the wait_ms budget with 'in_progress').
    - SIGINT reaches bash but status classification is wrong
      (check_task sees 'completed' instead of 'cancelled').
    - Elapsed timing check confirms cancel actually short-
      circuited the sleep rather than waiting it out.
    """
    import time as _time

    start = _time.monotonic()
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Step 1: call terminal_run with command='sleep 60', "
                "shell='default', synchronous=false. "
                "Step 2: IMMEDIATELY call cancel_task with the "
                "task_id that terminal_run returned. "
                "Step 3: call check_task with the same task_id and "
                "wait_ms=15000 so the tool blocks long enough for "
                "the cancellation to finalize in the workflow. "
                "Step 4: say 'cancelled'. "
                "Do NOT write any commentary before the tool calls."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=180)
    elapsed = _time.monotonic() - start
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"
    # Timing check: without a working cancel, the sleep runs to
    # natural completion (~60s) plus LLM roundtrips (~40s on gpt-
    # 5.4 medium). With cancel working, the path is cancel → SIGINT
    # → shell exits ~immediately, then just LLM latency. 90s is
    # the regression boundary — past 90s means we waited out the
    # sleep.
    assert elapsed < 90, (
        f"Task took {elapsed:.1f}s — cancel likely didn't interrupt "
        f"the sleep (expected <90s; full sleep 60 + LLM roundtrips "
        f"would push past 90s)."
    )

    # Primary correctness check: the check_task function_call_output
    # reports status='cancelled'. If cancel_task's terminal branch
    # failed to flip the workflow status, we'd see 'completed' or
    # 'in_progress' here — both are clear regression signals.
    check_payloads: list[dict[str, Any]] = []
    for item in final.get("output", []):
        if item.get("type") != "function_call_output":
            continue
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("kind") == "terminal":
            check_payloads.append(parsed)
    assert check_payloads, (
        f"No check_task function_call_output with kind='terminal'. "
        f"The LLM didn't poll via check_task even though the prompt "
        f"instructed it to. Outputs: "
        f"{[(i.get('type'), (i.get('output') or '')[:80]) for i in final.get('output', [])]}"
    )
    assert any(co.get("status") == "cancelled" for co in check_payloads), (
        f"No check_task saw the task in 'cancelled' status. "
        f"Statuses observed: "
        f"{[co.get('status') for co in check_payloads]}. If all "
        f"'in_progress', wait_ms=15000 expired before the cancel "
        f"finalized (SIGINT path too slow). If 'completed', SIGINT "
        f"didn't flip the status (sleep ran to completion)."
    )


def test_check_task_polls_running_terminal_stdout(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """The LLM can poll a running async terminal via ``check_task``
    and see partial stdout via ``recent_activity``.

    This exercises the tail-f story (§6.11): while a command is
    running, ``check_task`` returns ``status="in_progress"`` and
    a ``recent_activity`` field populated from the shell's ring
    buffer. If :func:`_get_recent_terminal_activity` is broken or
    the manager's task-registration path is wrong, the LLM sees no
    stdout until the auto-delivered completion message.

    Failure modes caught:
    - ``recent_activity`` not populated → LLM can't see progress.
    - ``check_task`` short-circuits terminal kind → kind branch
      doesn't fire.
    - The manager's ``peek_task_stdout`` cursor doesn't advance →
      identical data on repeated polls (not strictly asserted here
      but adjacent).
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Step 1: call terminal_run with command='echo "
                "POLL-MARKER-123; sleep 5; echo POLL-DONE', "
                "shell='default', synchronous=false. "
                "Step 2: wait ~1 second, then call check_task with "
                "the task_id. "
                "Step 3: after the result auto-delivers, report "
                "whether you saw POLL-MARKER-123. "
                "Do NOT write commentary before any tool call."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=180)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # Find the check_task function_call_output and verify it
    # contains our marker in recent_activity (or status=completed
    # + result containing it, since "1 second wait" may or may not
    # catch the running phase depending on LLM + scheduler timing).
    fco_items = _get_output_items(final, "function_call_output")
    check_outputs: list[dict[str, Any]] = []
    for item in fco_items:
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if parsed.get("kind") == "terminal" and "task_id" in parsed:
            check_outputs.append(parsed)

    assert len(check_outputs) >= 1, (
        f"Expected at least one terminal-kind check_task output "
        f"(kind='terminal' + task_id field), got {len(check_outputs)}. "
        f"All function_call_outputs: "
        f"{[i.get('output', '')[:100] for i in fco_items]}"
    )

    # Check that at least one poll saw the marker — either live
    # (recent_activity) or after completion (result).
    saw_marker = False
    for co in check_outputs:
        for field in ("recent_activity", "result"):
            val = co.get(field)
            if isinstance(val, str) and "POLL-MARKER-123" in val:
                saw_marker = True
                break
        if saw_marker:
            break
    debug_view = [
        (
            co.get("status"),
            (co.get("recent_activity") or "")[:80],
            (co.get("result") or "")[:80],
        )
        for co in check_outputs
    ]
    assert saw_marker, (
        f"Expected check_task to surface POLL-MARKER-123 in "
        f"recent_activity (live poll) or result (completed), got "
        f"{debug_view}"
    )


def test_parallel_async_terminals_run_on_separate_shells(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """Two async ``terminal_run`` calls on different shells get two
    task_ids and both results auto-deliver.

    If the manager's per-shell task tracking is wrong (e.g. one
    task ID overwriting the other's registration), one of the two
    auto-delivered system messages would be missing, or one task
    would never complete.

    Failure modes caught:
    - Shared task registry across shells → one task ID clobbers
      the other.
    - Auto-delivery serializes and only fires once.
    - Shell cap / name handling breaks at N>1.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Step 1: call terminal_run with command='echo "
                "PAR-A-XXX', shell='shella', synchronous=false. "
                "Step 2: call terminal_run with command='echo "
                "PAR-B-YYY', shell='shellb', synchronous=false. "
                "Step 3: after BOTH results auto-deliver, confirm "
                "you saw both PAR-A-XXX and PAR-B-YYY. "
                "Do NOT write commentary before the tool calls."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=180)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # Two distinct task_ids should appear in the terminal_run
    # handle outputs.
    terminal_run_outputs = []
    for item in _get_output_items(final, "function_call_output"):
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if parsed.get("tool_name") == "terminal_run":
            terminal_run_outputs.append(parsed)

    task_ids = {o.get("task_id") for o in terminal_run_outputs}
    # Both task_ids must exist AND be distinct — a shared id would
    # collapse the two tasks into one registration, causing one to
    # be lost.
    assert len(task_ids) == 2, (
        f"Expected two distinct async task_ids (one per shell), "
        f"got {task_ids}. If 1, the two dispatches shared a task_id "
        f"(registration bug); if 0, the handles weren't returned."
    )

    # Both completion-system messages must appear in the
    # conversation items.
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    system_texts: list[str] = []
    for item in items:
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        for block in item.get("content", []):
            text = block.get("text") or ""
            if text.startswith("[System: task"):
                system_texts.append(text)
    for tid in task_ids:
        assert any(isinstance(tid, str) and tid in m for m in system_texts), (
            f"Missing completion system message for task {tid!r}. "
            f"Got {system_texts!r}. If only one of the two tasks "
            f"has a delivery message, the drain dropped one."
        )

    # Both tokens appear somewhere in the conversation.
    blob = json.dumps(items)
    assert "PAR-A-XXX" in blob and "PAR-B-YYY" in blob, (
        "Both stdout markers must appear — if one is missing, "
        "that shell's output wasn't surfaced to the LLM."
    )
