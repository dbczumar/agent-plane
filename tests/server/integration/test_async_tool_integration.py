"""
Server-integration tests for ``@tool(synchronous=False)`` end-to-end.

These tests exercise the full Phase 2 async-tool pipeline:

1. The LLM invokes an async tool;
2. ``_call_tool`` dispatches it via ``_dispatch_async_tool`` and
   returns a JSON handle (NOT the inline result);
3. The ``background_tool_workflow`` runs the function in a
   subprocess via the fd-3 protocol;
4. On completion it sends an ``async_work_complete`` payload to
   the parent on the documented topic;
5. The parent's between-iteration drain (D4) picks it up and
   persists a ``[System: task ... completed]`` user message;
6. The LLM sees the system message on the next iteration and
   produces a final response.

This is the only place all six steps run together. Unit tests
cover the helpers individually; this is the integration glue.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response

pytestmark = [pytest.mark.asyncio]

_AGENT_NAME = "async-tool-test-agent"

# A deliberately-trivial async tool — the production breakage we
# care about is in the dispatch + drain plumbing, not in the tool
# body. Returning a fixed string makes the assertion at the end
# unambiguous.
_ASYNC_TOOL_SOURCE = '''\
"""Test async tools — fixed markers so assertions are deterministic."""
from agent_plane_client import tool


@tool(synchronous=False)
def slow_compute() -> str:
    """Pretend to do background work and return a marker.

    Args:
        (no arguments)
    """
    return "ASYNC_TOOL_DONE_MARKER"


@tool(synchronous=False)
def slow_echo(label: str) -> str:
    """Echo the label so parallel-spawn tests can prove ordering.

    Args:
        label: Text to echo back.
    """
    return f"ECHO[{label}]"


@tool(synchronous=False)
def boom() -> str:
    """Always raises so the failure path is exercised.

    Args:
        (no arguments)
    """
    raise RuntimeError("intentional boom from async tool")


@tool(synchronous=False)
def long_sleep() -> str:
    """Block long enough that tests can cancel mid-execution.

    The test cancels the parent response within a couple of
    seconds; this 60s sleep guarantees the tool is still
    running when the cancel arrives.

    Args:
        (no arguments)
    """
    import time

    time.sleep(60)
    return "should-never-return"
'''


def _build_async_tool_agent_bundle() -> bytes:
    """
    Build an agent bundle exporting one ``synchronous=False`` tool.

    :returns: tar.gz bytes containing config.yaml plus
        ``tools/python/async_tools.py``. The single file exports
        every async tool the test suite uses (slow_compute,
        slow_echo, boom) so the bundle stays small.
    """
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {
            "model": _AGENT_NAME,
            "connection": {"api_key": "test-key"},
        },
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        cfg_bytes = yaml.dump(config).encode()
        cfg_info = tarfile.TarInfo(name="config.yaml")
        cfg_info.size = len(cfg_bytes)
        tf.addfile(cfg_info, io.BytesIO(cfg_bytes))

        src_bytes = _ASYNC_TOOL_SOURCE.encode()
        src_info = tarfile.TarInfo(name="tools/python/async_tools.py")
        src_info.size = len(src_bytes)
        tf.addfile(src_info, io.BytesIO(src_bytes))
    return buf.getvalue()


async def _create_async_tool_agent(client: httpx.AsyncClient) -> dict[str, Any]:
    """
    Upload the async-tool test bundle.

    :param client: HTTP client for the test server.
    :returns: Agent creation response JSON.
    """
    bundle = _build_async_tool_agent_bundle()
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent upload failed: {resp.status_code} {resp.text}"
    return resp.json()


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status or fail loud.

    :param client: HTTP client.
    :param response_id: The response/task ID to poll.
    :returns: Terminal response body.
    :raises AssertionError: If the response never reaches terminal
        within ~20 seconds (200 polls × 0.1s).
    """
    for _ in range(200):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"Response {response_id} did not reach terminal status",
    )


async def _get_items(
    client: httpx.AsyncClient,
    conv_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all conversation items.

    :param client: HTTP client.
    :param conv_id: The conversation ID.
    :returns: List of item dicts in store order.
    """
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


# ─── Tests ───────────────────────────────────────────────────


async def test_async_tool_dispatch_returns_handle_and_auto_delivers_result(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    End-to-end: async tool dispatch → handle → background work →
    drain → system message → LLM final response.

    Sequence:
    * LLM call 1 returns a tool_call to ``slow_compute``.
    * ``_call_tool`` dispatches via ``_dispatch_async_tool``;
      returns a JSON handle dict to the LLM. The
      ``function_call_output`` for that turn carries the JSON
      handle string (not the eventual ``ASYNC_TOOL_DONE_MARKER``).
    * ``background_tool_workflow`` runs in the background and
      signals ``async_work_complete``.
    * The parent's between-iteration drain wakes on the next
      iteration, persists a ``[System: ...]`` user message
      containing the marker.
    * LLM call 2 receives the system message in its history and
      produces a final response.

    What breaks if wrong (each anchors a specific assertion):
    * Dispatch missed: ``function_call_output`` would carry the
      tool's actual return value (the marker) — not a JSON handle.
    * Drain skipped: the ``[System: ... completed]`` message would
      be absent from the conversation; LLM call 2 wouldn't see it
      in its input.
    * Wrong handle shape: the parsed JSON would be missing fields
      (``task_id``, ``status="in_progress"``, ``message``).
    """
    await _create_async_tool_agent(client)

    # LLM call 1: invoke the async tool. The runtime intercepts
    # this in _call_tool because the tool's metadata.synchronous
    # is False — the LLM never gets the marker inline.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_1",
                "name": "slow_compute",
                "arguments": "{}",
            },
        ],
    )
    # LLM call 2: the LLM has the handle in history but the
    # background tool may not have completed yet. The LLM
    # responds with placeholder text — since pending_tool_tasks
    # is non-empty, the runtime will block on the drain instead
    # of treating this as the final response.
    mock_llm.add_call(text="Working on it…")
    # LLM call 3: by now the auto-delivered system message is in
    # history. We capture received_kwargs on this call to assert
    # the marker reached the prompt.
    call_3 = mock_llm.add_call(text="Got the async result.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Run the async compute, please.",
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    # Completion path proves the workflow finished — without the
    # drain, the end-of-turn auto-collect would block forever
    # waiting on a topic signal that never resulted in a
    # persisted system message.
    assert completed["status"] == "completed", (
        f"Expected completed; got {completed['status']}. "
        f"Error: {completed.get('error')}"
    )

    items = await _get_items(client, conv_id)
    types_in_order = [i["type"] for i in items]

    # ── 1. dispatch — function_call_output carries the JSON handle
    fco_items = [
        i
        for i in items
        if i.get("type") == "function_call_output"
        and i.get("call_id") == "call_async_1"
    ]
    assert len(fco_items) == 1, (
        f"Expected 1 function_call_output for the async tool; "
        f"got {len(fco_items)}. items={types_in_order}"
    )
    handle_payload = json.loads(fco_items[0]["output"])
    # If dispatch fell back to the sync path, this string would
    # be the marker ASYNC_TOOL_DONE_MARKER, not a JSON dict.
    assert handle_payload["tool_name"] == "slow_compute"
    assert handle_payload["status"] == "in_progress"
    # The handle's task_id must be a string and non-empty — the
    # LLM relies on it for check_task / cancel_task lookups.
    assert isinstance(handle_payload["task_id"], str)
    assert handle_payload["task_id"], "Empty task_id in handle"
    # The message field must name the literal task_id so the LLM
    # can see what to pass to check_task. If absent, the LLM has
    # to guess.
    assert handle_payload["task_id"] in handle_payload["message"]

    # ── 2. drain — a [System: task ... completed] user message
    # must appear in the conversation, and it must contain the
    # tool's actual return value (the marker).
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [
        t for t in user_texts if t.startswith("[System: task ")
    ]
    assert len(completion_messages) == 1, (
        f"Expected exactly 1 auto-delivered completion message; "
        f"got {len(completion_messages)}. user_texts={user_texts}"
    )
    completion_text = completion_messages[0]
    assert "completed" in completion_text, (
        f"Auto-delivered message must mark status; got {completion_text!r}"
    )
    assert "ASYNC_TOOL_DONE_MARKER" in completion_text, (
        f"Tool's actual return value missing from auto-delivered "
        f"message — drain may have dropped the payload body. "
        f"Got {completion_text!r}"
    )

    # ── 3. LLM saw the system message
    # Without this, the test would pass even if the system message
    # were persisted but never reached the prompt (the half-broken
    # case where drain runs but _sync_history doesn't pick it up).
    assert call_3.received_kwargs is not None, (
        "Third LLM call was never made — workflow ended before "
        "the LLM could respond to the auto-delivered result."
    )
    llm_input_text = json.dumps(call_3.received_kwargs["input"])
    assert "ASYNC_TOOL_DONE_MARKER" in llm_input_text, (
        "The marker must appear in LLM call 2's input. If missing, "
        "the auto-delivered system message was persisted but not "
        "synced into the in-memory history."
    )


async def test_async_tool_failure_surfaces_truncated_traceback(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A raising ``@tool(synchronous=False)`` must:
    * surface ``status="failed"`` in the auto-delivered message;
    * include the exception class + message so the LLM can adjust
      its plan;
    * NOT cause the parent response to fail (the failure is the
      *tool's*, not the agent's).

    Without this coverage, an exception in a background tool
    could vanish silently — the parent's drain might never wake
    if the failure path skipped ``_send_payload`` (it mustn't,
    per G86).
    """
    await _create_async_tool_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_boom_1",
                "name": "boom",
                "arguments": "{}",
            },
        ],
    )
    mock_llm.add_call(text="Working on it…")
    call_3 = mock_llm.add_call(text="Got the failure.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Run the boom tool, please.",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    # The PARENT response completes successfully — only the tool
    # task itself is "failed". A wrong implementation would
    # propagate the tool exception to the agent loop and fail
    # the whole turn.
    assert completed["status"] == "completed", (
        f"Parent must complete even when an async tool fails. "
        f"Got {completed['status']}; error={completed.get('error')}"
    )

    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    failure_messages = [t for t in user_texts if "[System: task " in t and "failed" in t]
    assert len(failure_messages) == 1, (
        f"Expected one [System: ... failed] message; got {failure_messages}. "
        f"If empty, the failure path skipped _send_payload — "
        f"the parent's drain has nothing to wake on (G86 violated)."
    )
    failure_text = failure_messages[0]
    # The exception class and message must both appear so the LLM
    # can decide whether to retry, change strategy, or apologize
    # to the user.
    assert "RuntimeError" in failure_text
    assert "intentional boom from async tool" in failure_text

    # The follow-up LLM call also sees the failure in its prompt
    # — proves the system message wasn't lost between persist and
    # _sync_history.
    assert call_3.received_kwargs is not None
    llm_input_text = json.dumps(call_3.received_kwargs["input"])
    assert "RuntimeError" in llm_input_text


async def test_parallel_async_tool_spawns_all_auto_deliver(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Three parallel ``@tool(synchronous=False)`` calls in a single
    LLM turn must all dispatch, all run, and all auto-deliver
    distinct results.

    What this guards against:
    * Dispatch serializing async calls (would still pass but be
      pointlessly slow — covered by the next-iteration count).
    * Drain only consuming the first signal (would surface as
      missing markers in the conversation).
    * Handle collisions (same task_id reused for multiple
      dispatches — would surface as identical task_ids in the
      function_call_outputs).
    """
    await _create_async_tool_agent(client)

    labels = ["alpha", "beta", "gamma"]
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": f"call_par_{i}",
                "name": "slow_echo",
                "arguments": json.dumps({"label": label}),
            }
            for i, label in enumerate(labels)
        ],
    )
    # Placeholder turn — pending_tool_tasks non-empty triggers
    # end-of-turn wait.
    mock_llm.add_call(text="Working…")
    # Final response — by now all three signals have drained.
    call_final = mock_llm.add_call(text="All three done.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Echo three labels in parallel.",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    assert completed["status"] == "completed", (
        f"Got {completed['status']}; error={completed.get('error')}"
    )

    items = await _get_items(client, conv_id)
    # Three distinct function_call_outputs (one per dispatched
    # async call) — confirms each got its own handle.
    fco_items = [i for i in items if i.get("type") == "function_call_output"]
    assert len(fco_items) == 3, (
        f"Expected 3 function_call_outputs (one per parallel "
        f"dispatch); got {len(fco_items)}. items="
        f"{[i['type'] for i in items]}"
    )
    handle_task_ids = {json.loads(i["output"])["task_id"] for i in fco_items}
    # Set length 3 proves task_ids are unique — a regression
    # where _dispatch_async_tool reused IDs would collapse this.
    assert len(handle_task_ids) == 3, (
        f"Handle task_ids must be unique across dispatches; "
        f"got {handle_task_ids}"
    )

    # Three distinct system messages — one per completion.
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    assert len(completion_messages) == 3, (
        f"Expected 3 auto-delivered completion messages; "
        f"got {len(completion_messages)}. user_texts={user_texts}"
    )
    completion_blob = "\n".join(completion_messages)
    for label in labels:
        # Each label's marker must appear somewhere — proves the
        # drain delivered every payload, not just the first.
        assert f"ECHO[{label}]" in completion_blob, (
            f"Marker for label {label!r} missing from auto-delivered "
            f"messages — drain may have stopped after the first signal."
        )

    # The final LLM call must see all three markers in its
    # prompt — proves _sync_history pulled in the system messages
    # before the next LLM call.
    assert call_final.received_kwargs is not None
    final_input = json.dumps(call_final.received_kwargs["input"])
    for label in labels:
        assert f"ECHO[{label}]" in final_input, (
            f"Marker for label {label!r} missing from final LLM input."
        )


async def test_parent_cancel_propagates_to_async_tool_task(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Cancelling the parent response while an async tool task is
    still running must:

    * Reach a terminal state for the parent (cancelled).
    * Cause the child tool task row to also reach a terminal
      status (cancelled or failed) — not stay in_progress
      forever (D9 / G86).

    Without DBOS workflow-tree cancellation propagating, the
    background_tool_workflow would keep its 60s sleep going and
    the child task row would be stuck in ``in_progress`` for
    a minute past the cancel.
    """
    from agent_plane.runtime.durability import get_workflow_status_async

    _ = get_workflow_status_async  # may use later; primary check is via task_store

    await _create_async_tool_agent(client)

    # Tool 1: long-running async tool.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_long_1",
                "name": "long_sleep",
                "arguments": "{}",
            },
        ],
    )
    # The runtime won't actually call the LLM again before the
    # cancel arrives — but queue a dummy response in case timing
    # races and the loop gets to the next iteration first.
    mock_llm.add_call(text="should not appear")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Run long_sleep",
    )
    response_id = result.body["id"]

    # Wait until the child tool task row exists — proves dispatch
    # ran and the background workflow has started. Without this
    # the cancel might race ahead of dispatch and miss the
    # propagation entirely.
    child_task_id: str | None = None
    for _ in range(60):
        items_resp = await client.get(
            f"/v1/conversations/{result.body['conversation']['id']}/items",
            params={"limit": 100},
        )
        items = items_resp.json()["data"]
        fcos = [i for i in items if i.get("type") == "function_call_output"]
        if fcos:
            handle = json.loads(fcos[0]["output"])
            child_task_id = handle["task_id"]
            break
        await asyncio.sleep(0.1)
    assert child_task_id is not None, (
        "Async dispatch never produced a function_call_output — "
        "the runtime didn't reach _dispatch_async_tool."
    )

    # Cancel the parent. The route returns 200 + the cancelled
    # response body once the task row is marked.
    cancel_resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert cancel_resp.status_code == 200, (
        f"Cancel failed: {cancel_resp.status_code} {cancel_resp.text}"
    )

    # Parent must reach terminal — without proper propagation it
    # would sit in_progress while the LLM call blocks.
    completed = await _wait_for_completion(client, response_id)
    assert completed["status"] == "cancelled", (
        f"Parent must be cancelled; got {completed['status']}"
    )

    # The child task row must transition to a terminal status —
    # cancel_workflow_async marks the DBOS row CANCELLED, and our
    # store mapper translates that to a terminal status string.
    # Poll up to 5 s.
    final_dbos_status: str | None = None
    last_seen_status: str | None = None
    for _ in range(50):
        status = await get_workflow_status_async(child_task_id)
        last_seen_status = status.status if status is not None else "NULL"
        if status is not None and status.status in {
            "CANCELLED",
            "ERROR",
            "SUCCESS",
        }:
            final_dbos_status = status.status
            break
        await asyncio.sleep(0.1)
    # Without _cancel_pending_child_tools wiring, this assertion
    # times out: the child workflow stays in PENDING/ENQUEUED
    # forever (the time.sleep(60) in the subprocess keeps the
    # worker thread busy, but DBOS still marks the row on cancel).
    assert final_dbos_status == "CANCELLED", (
        f"Child workflow {child_task_id} did not transition to "
        f"CANCELLED after parent cancellation; "
        f"final_dbos_status={final_dbos_status!r}, "
        f"last_seen_status={last_seen_status!r}"
    )
