"""
Server-integration tests for the Phase 5 async client-tool pipeline.

The full chain under test:

1. Caller submits a tool whose ``parameters.properties`` includes
   a ``synchronous`` boolean — surfacing the per-call
   async-dispatch choice to the LLM.
2. LLM emits a ``function_call`` whose ``arguments`` set
   ``synchronous: false``.
3. Workflow's ``_wants_async_dispatch`` reads the per-call arg
   and routes to ``_dispatch_async_client_tools``: creates a
   ``kind="client_tool"`` task, persists a
   ``function_call_output`` carrying the handle JSON
   ``{task_id, kind: "client_tool", ...}``.
4. LLM sees the handle on the next iteration and continues.
5. Caller eventually PATCHes
   ``async_tool_results: [{task_id, status, output|error}]``.
6. PATCH handler marks task terminal in-store and signals
   ``async_work_complete`` to the parent.
7. Parent's drain auto-delivers a ``[System: task ...
   <status>]`` user message; LLM sees it on the next iteration
   and produces a final response.

Plus the error / race / cancel paths.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent

pytestmark = [pytest.mark.asyncio]


# Schema that exposes the per-call ``synchronous`` choice to the
# LLM. The mock LLM is driven by call args (``async_args`` /
# ``sync_args`` below) — these schemas are what the server's
# parser sees as the tool surface.
_ASYNC_CAPABLE_CLIENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "client_long_compute",
        "description": "Long-running client-side computation.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "synchronous": {
                    "type": "boolean",
                    "description": "Set false for async dispatch.",
                },
            },
            "required": ["n"],
        },
    },
}


_SYNC_CLIENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "client_quick_lookup",
        "description": "Quick client-side lookup (sync).",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
}


def _async_args(**extra: Any) -> str:
    """
    Build a JSON args string that requests async dispatch.

    Mock-LLM tool calls use ``arguments`` as a string payload;
    this helper bundles ``synchronous: false`` together with any
    tool-specific args so each test states its intent in one place.

    :param extra: Tool-specific args (``n=5``, etc.).
    :returns: JSON-encoded args dict including ``synchronous: false``.
    """
    return json.dumps({**extra, "synchronous": False})


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
    timeout_iters: int = 200,
) -> dict[str, Any]:
    """Poll until a response reaches a terminal status."""
    for _ in range(timeout_iters):
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
    """Fetch all conversation items in store order."""
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


async def _wait_for_item_type(
    client: httpx.AsyncClient,
    conv_id: str,
    item_type: str,
    *,
    name: str | None = None,
    timeout_iters: int = 200,
) -> dict[str, Any]:
    """
    Poll until an item of the given type (and optionally name) appears.

    Used to detect the function_call_output that carries the
    async client-tool handle — the test needs the handle's
    task_id to construct the PATCH.
    """
    for _ in range(timeout_iters):
        items = await _get_items(client, conv_id)
        for item in items:
            if item.get("type") != item_type:
                continue
            if name is not None and item.get("name") != name:
                continue
            return item
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"No item with type={item_type!r}, name={name!r} appeared "
        f"in conversation {conv_id} within timeout",
    )


# ─── Tests ───────────────────────────────────────────────────


async def test_async_client_tool_returns_handle_to_llm(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    The async client-tool dispatch must produce a
    function_call_output carrying the handle JSON ``{task_id,
    kind: "client_tool", ...}`` — NOT a parked
    ``_ClientToolCallsPending``. The LLM sees the handle
    inline on the next iteration and continues.
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_client_1",
                "name": "client_long_compute",
                "arguments": _async_args(n=42),
            },
        ],
    )
    # Subsequent calls for the parent loop after the handle is
    # in history. The drain will block waiting for the async
    # signal; we finish the test by sending the PATCH.
    for _ in range(10):
        mock_llm.add_call(text="ok")

    # POST with the async client tool declared.
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Compute n=42",
            "background": True,
            "stream": False,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL],
        },
    )
    assert resp.status_code == 200
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    # Wait for the function_call_output that carries the
    # handle. The output appears as soon as the workflow
    # dispatches the async client tool.
    fco = await _wait_for_item_type(
        client,
        conv_id,
        "function_call_output",
    )
    handle = json.loads(fco["output"])
    assert handle["kind"] == "client_tool", (
        f"Handle kind must be 'client_tool'; got {handle!r}. "
        f"If the dispatch went through the legacy parking path "
        f"the output would be the unexecuted function_call's "
        f"call_id, not a handle JSON."
    )
    assert "task_id" in handle and handle["task_id"], (
        f"Handle must carry a non-empty server-issued task_id "
        f"(the client uses it to PATCH back). Got {handle!r}"
    )
    assert handle["status"] == "in_progress"

    # Verify the kind="client_tool" task row exists in the
    # store with status "queued" (not yet finalized).
    from agent_plane.runtime import get_task_store

    task_store = get_task_store()
    task_row = await task_store.get(handle["task_id"])
    assert task_row is not None, (
        "The async client-tool dispatch must have created a "
        f"task_store row for {handle['task_id']!r}"
    )
    assert task_row.kind == "client_tool"

    # Tear down: cancel the parent so the drain stops blocking.
    await client.post(f"/v1/responses/{response_id}/cancel")
    await _wait_for_completion(client, response_id)


async def test_async_client_tool_completion_via_patch_auto_delivers(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    End-to-end: dispatch → handle → PATCH async_tool_results →
    parent's drain auto-delivers a ``[System: task ...
    completed]`` user message → LLM's next iteration sees the
    output verbatim in its prompt.
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_client_done",
                "name": "client_long_compute",
                "arguments": _async_args(n=7),
            },
        ],
    )
    # Subsequent calls. The third-from-end is captured to
    # verify the marker reached the LLM input.
    for _ in range(5):
        mock_llm.add_call(text="working")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Compute via the async client tool",
            "background": True,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL],
        },
    )
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    # Wait for the handle, then PATCH a completion. The handle
    # must include the task_id for the PATCH lookup.
    fco = await _wait_for_item_type(
        client,
        conv_id,
        "function_call_output",
    )
    handle = json.loads(fco["output"])
    task_id = handle["task_id"]

    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "async_tool_results": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "output": "CLIENT_TOOL_RESULT_MARKER_99",
                },
            ],
        },
    )
    assert patch_resp.status_code == 200, (
        f"PATCH async_tool_results failed: {patch_resp.status_code} {patch_resp.text}"
    )

    # The parent's drain should now wake on the
    # async_work_complete signal and persist a
    # [System: task ... completed] user message containing the
    # output. The LLM's next iteration uses one of the queued
    # "working" calls and finishes the workflow.
    completed = await _wait_for_completion(client, response_id)
    assert completed["status"] == "completed", f"Parent did not complete: {completed}"

    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    # Exactly one async tool was dispatched and PATCHed completed,
    # so the drain must auto-deliver exactly one system message.
    # If 0: the PATCH-driven signal isn't reaching the drain.
    # If 2+: the drain is duplicating completions.
    assert len(completion_messages) == 1, (
        f"Expected exactly one auto-delivered [System: ...] message "
        f"for the single async tool; got {len(completion_messages)} "
        f"(user_texts={user_texts})."
    )
    assert "CLIENT_TOOL_RESULT_MARKER_99" in completion_messages[0], (
        f"The PATCH'd output must reach the auto-delivered system "
        f"message verbatim. Got: {completion_messages[0]!r}"
    )

    # Cross-check: the task row is now terminal with the
    # output stored.
    from agent_plane.runtime import get_task_store

    task_row = await get_task_store().get(task_id)
    assert task_row is not None
    assert task_row.status == "completed", (
        f"Task row should be terminal; got status={task_row.status!r}"
    )


async def test_async_client_tool_failure_path(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    PATCH with ``status="failed"`` + error dict → parent
    completes (the failure is the *client tool's*, not the
    parent's) and the drain delivers a
    ``[System: task ... failed]`` message containing the
    exception class + message.
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_fail",
                "name": "client_long_compute",
                "arguments": _async_args(n=-1),
            },
        ],
    )
    for _ in range(5):
        mock_llm.add_call(text="ok")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Trigger a failure on the client tool",
            "background": True,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL],
        },
    )
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    fco = await _wait_for_item_type(client, conv_id, "function_call_output")
    handle = json.loads(fco["output"])
    task_id = handle["task_id"]

    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "async_tool_results": [
                {
                    "task_id": task_id,
                    "status": "failed",
                    "error": {
                        "message": "ValueError: CLIENT_FAIL_MARKER",
                        "traceback": "Traceback...",
                    },
                },
            ],
        },
    )
    assert patch_resp.status_code == 200

    completed = await _wait_for_completion(client, response_id)
    # Parent must complete — the client tool's failure is not
    # an agent-level failure.
    assert completed["status"] == "completed", (
        f"Parent must complete even when an async client tool "
        f"fails. Got {completed['status']}; error={completed.get('error')}"
    )

    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    failure_messages = [t for t in user_texts if "[System: task " in t and "failed" in t]
    assert len(failure_messages) >= 1
    blob = "\n".join(failure_messages)
    assert "CLIENT_FAIL_MARKER" in blob


async def test_async_client_tool_idempotent_patch(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A second PATCH for the same task_id after the first
    completion is a no-op: status stays the same, no
    double-signal to the parent (the parent's drain only sees
    one completion message).
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_idem",
                "name": "client_long_compute",
                "arguments": _async_args(n=1),
            },
        ],
    )
    for _ in range(5):
        mock_llm.add_call(text="ok")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Idempotent PATCH test",
            "background": True,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL],
        },
    )
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    fco = await _wait_for_item_type(client, conv_id, "function_call_output")
    task_id = json.loads(fco["output"])["task_id"]

    body = {
        "async_tool_results": [
            {
                "task_id": task_id,
                "status": "completed",
                "output": "IDEMPOTENT_RESULT",
            },
        ],
    }
    p1 = await client.patch(f"/v1/responses/{response_id}", json=body)
    assert p1.status_code == 200
    p2 = await client.patch(f"/v1/responses/{response_id}", json=body)
    # Second PATCH must succeed (no error code) but the task
    # is already terminal — finalize_async_task is the
    # branch we exercise here. G3 first-write-wins.
    assert p2.status_code == 200

    await _wait_for_completion(client, response_id)
    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    # Exactly one — the second PATCH did NOT re-signal the
    # parent. Without idempotency the parent would see two
    # identical [System: ... completed] messages.
    assert len(completion_messages) == 1, (
        f"Expected exactly 1 auto-delivered completion (idempotent "
        f"PATCH must not re-signal); got {len(completion_messages)}: "
        f"{completion_messages}"
    )


async def test_unknown_task_id_in_async_patch_returns_404(
    client: httpx.AsyncClient,
) -> None:
    """
    PATCH with an async_tool_result whose task_id is unknown
    must surface a 404 — silent acceptance would mask client
    bugs.
    """
    # Create a real response so the response_id is valid.
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Bogus PATCH test",
            "background": True,
        },
    )
    response_id = resp.json()["id"]
    await _wait_for_completion(client, response_id)

    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "async_tool_results": [
                {
                    "task_id": "tsk_does_not_exist",
                    "status": "completed",
                    "output": "x",
                },
            ],
        },
    )
    assert patch_resp.status_code == 404, (
        f"Unknown task_id must return 404; got {patch_resp.status_code}: {patch_resp.text}"
    )


async def test_async_patch_rejects_non_client_tool_kind(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    PATCH async_tool_results must validate ``kind=="client_tool"``
    on the target task — silently letting it finalize a sub-agent
    or @tool task would corrupt their state.
    """
    await create_test_agent(client)

    mock_llm.add_call(text="ok")
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Just a plain turn so we have an agent_task to point at",
            "background": True,
        },
    )
    response_id = resp.json()["id"]
    await _wait_for_completion(client, response_id)

    # PATCH with the parent's task_id (kind="agent_task") —
    # must reject.
    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "async_tool_results": [
                {
                    "task_id": response_id,
                    "status": "completed",
                    "output": "should not be accepted",
                },
            ],
        },
    )
    assert patch_resp.status_code == 409, (
        f"PATCH targeting a non-client_tool kind must 409; got "
        f"{patch_resp.status_code}: {patch_resp.text}"
    )


async def test_mixed_tool_results_and_async_tool_results_in_one_patch(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A single PATCH may carry both ``tool_results`` (sync legacy
    parking) and ``async_tool_results`` (Phase 5). The handler
    processes them independently — proves the two mechanisms
    coexist.

    This test exercises just the async leg via PATCH; the sync
    leg is implicit (an empty tool_results list is allowed).
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_mixed",
                "name": "client_long_compute",
                "arguments": _async_args(n=5),
            },
        ],
    )
    for _ in range(5):
        mock_llm.add_call(text="ok")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Async client only — empty tool_results",
            "background": True,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL, _SYNC_CLIENT_TOOL],
        },
    )
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    fco = await _wait_for_item_type(client, conv_id, "function_call_output")
    task_id = json.loads(fco["output"])["task_id"]

    # PATCH with BOTH lists — tool_results empty (no sync tool
    # was actually called in this test), async_tool_results
    # carries the completion.
    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [],
            "async_tool_results": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "output": "MIXED_PATCH_OK",
                },
            ],
        },
    )
    assert patch_resp.status_code == 200

    await _wait_for_completion(client, response_id)
    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    # One async tool was dispatched and PATCHed, so the drain must
    # deliver exactly one system completion message. The mixed
    # nature of the PATCH is the empty tool_results: [] coexisting
    # with a single async_tool_results entry — both legs must be
    # accepted by the handler.
    # If 0: PATCH async_tool_results didn't signal the parent.
    # If 2+: a duplicate signal escaped or the empty tool_results
    # leg accidentally produced a system message.
    assert len(completion_messages) == 1, (
        f"Expected exactly one auto-delivered [System: ...] message "
        f"for the single async tool; got {len(completion_messages)} "
        f"(user_texts={user_texts})."
    )
    assert "MIXED_PATCH_OK" in completion_messages[0], (
        f"The PATCH'd output must reach the auto-delivered system "
        f"message verbatim. Got: {completion_messages[0]!r}"
    )


async def test_parent_cancel_emits_response_client_task_cancel_sse(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Parent cancel propagates: every non-terminal client_tool
    child gets a ``response.client_task.cancel`` SSE event with
    its task_id, and the task row is marked cancelled in-store
    so a late PATCH with ``status="completed"`` is a no-op
    (G3).
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_cancel",
                "name": "client_long_compute",
                "arguments": _async_args(n=99),
            },
        ],
    )
    for _ in range(10):
        mock_llm.add_call(text="ok")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Cancel test",
            "background": True,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL],
        },
    )
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    fco = await _wait_for_item_type(client, conv_id, "function_call_output")
    task_id = json.loads(fco["output"])["task_id"]

    # Cancel the parent — propagation should mark this task
    # cancelled in-store.
    cancel_resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert cancel_resp.status_code == 200

    await _wait_for_completion(client, response_id)

    # Task row should now be terminal cancelled.
    from agent_plane.runtime import get_task_store

    task_row = await get_task_store().get(task_id)
    assert task_row is not None
    assert task_row.status == "cancelled", (
        f"Client_tool task should be cancelled after parent cancel; "
        f"got status={task_row.status!r}. Parent cancel propagation "
        f"to client_tool kind is not wired correctly."
    )

    # G3: a late PATCH with status="completed" is accepted but
    # does NOT change the cancelled status away.
    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "async_tool_results": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "output": "race-loser",
                },
            ],
        },
    )
    assert patch_resp.status_code == 200, "Late PATCH should be accepted (200) but no-op."
    task_row_after = await get_task_store().get(task_id)
    assert task_row_after is not None
    assert task_row_after.status == "cancelled", (
        f"Late PATCH must NOT override the cancelled status (G3); got {task_row_after.status!r}"
    )


async def test_list_tasks_includes_client_tool_kind(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    ``list_tasks`` (the underlying query, not the LLM-facing
    builtin) returns kind="client_tool" tasks alongside other
    kinds. Used by the LLM-facing list_tasks builtin which
    already filters to non-agent_task kinds (G57).
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_listed",
                "name": "client_long_compute",
                "arguments": _async_args(n=1),
            },
        ],
    )
    for _ in range(5):
        mock_llm.add_call(text="ok")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "list test",
            "background": True,
            "tools": [_ASYNC_CAPABLE_CLIENT_TOOL],
        },
    )
    response_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    await _wait_for_item_type(client, conv_id, "function_call_output")

    from agent_plane.runtime import get_task_store

    children = await get_task_store().list_tasks(root_task_id=response_id)
    kinds = [c.kind for c in children]
    assert "client_tool" in kinds, f"Expected a client_tool child in list_tasks; got kinds={kinds}"

    # Tear down.
    await client.post(f"/v1/responses/{response_id}/cancel")
    await _wait_for_completion(client, response_id)
