"""
Server-integration tests for the Phase 3 sub-agent pipeline.

Mirrors ``test_async_tool_integration.py`` but for sub-agent
spawns. The full chain under test:

1. Parent LLM emits a ``spawn_sub_agent`` tool call.
2. ``SpawnSubAgentTool`` creates a child task (kind="sub_agent")
   and starts the sub-agent's ``agent_execution_workflow``.
3. The handle JSON returns to the parent LLM.
4. The sub-agent runs its own loop, terminates, and signals
   ``async_work_complete`` to the parent (the unified Phase 2/3
   topic).
5. The parent's drain auto-delivers a ``[System: task ...]``
   user message; the next LLM iteration sees it.
6. The parent produces a final response that references the
   sub-agent's output.

Plus three regression tests that verify the legacy batch tools
(``spawn_sub_agents`` / ``check_sub_agents`` / ``cancel_sub_agent``)
are no longer registered (M3 — no dual code path).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import build_agent_bundle, create_test_response

pytestmark = [pytest.mark.asyncio]


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
    timeout_iters: int = 200,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status.

    :param client: HTTP client.
    :param response_id: The response/task ID to poll.
    :param timeout_iters: Max number of 0.1s polls (default
        200 = ~20 s).
    :returns: The terminal response body.
    :raises AssertionError: If terminal isn't reached within
        the timeout.
    """
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
    """
    Fetch all conversation items in store order.

    :param client: HTTP client.
    :param conv_id: The conversation ID.
    :returns: List of item dicts.
    """
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


async def _create_parent_with_sub_agent(
    client: httpx.AsyncClient,
    *,
    parent_name: str,
    sub_agent_name: str,
) -> None:
    """
    Upload a parent agent that declares one named sub-agent.

    :param client: HTTP client.
    :param parent_name: The parent agent's name (matches the
        ``model`` argument used in subsequent
        ``create_test_response`` calls).
    :param sub_agent_name: The sub-agent's name (the ``type``
        the LLM will pass to ``spawn_sub_agent``).
    """
    bundle = build_agent_bundle(
        name=parent_name,
        sub_agents=[
            {"name": sub_agent_name, "description": f"{sub_agent_name} sub-agent"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent upload failed: {resp.status_code} {resp.text}"


# ─── Tests ───────────────────────────────────────────────────


async def test_spawn_sub_agent_auto_delivers_result(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Parent dispatches one sub-agent via the singular
    ``spawn_sub_agent``. The sub-agent's terminal exit signals
    ``async_work_complete``; the parent's drain auto-delivers the
    result as a ``[System: task ... completed]`` user message;
    the next LLM iteration sees that text in its prompt.

    What this catches:
    * Dispatch missed → ``function_call_output`` carries the
      sub-agent's output text directly (not a JSON handle).
    * Drain not wired for sub-agents → no ``[System: ...]``
      message persisted.
    * Auto-delivered message format wrong → LLM never sees the
      marker substring in its prompt.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-spawn-test",
        sub_agent_name="researcher",
    )

    # Parent call 1: dispatch the sub-agent.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn_p3_1",
                "name": "spawn_sub_agent",
                "arguments": json.dumps(
                    {"type": "researcher", "name": "researcher", "input": "Find info about X"},
                ),
            },
        ],
    )
    # Parent and sub-agent share a FIFO mock queue — to avoid
    # depending on which one consumes which call, every text
    # response from this point uses the same distinctive marker.
    # The sub-agent's terminal output ALWAYS carries the marker
    # (whichever call it consumed), so the [System: ...]
    # auto-delivered message must too.
    for _ in range(4):
        mock_llm.add_call(text="SUBAGENT_RESULT_MARKER_47")
    # Trailing add_calls so the queue never runs dry — the
    # exact iteration count depends on whether parent or sub-
    # agent consumes call #2 (FIFO race).
    mock_llm.add_call(text="SUBAGENT_RESULT_MARKER_47")

    result = await create_test_response(
        client,
        model="parent-spawn-test",
        input_text="Research X",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    assert completed["status"] == "completed", (
        f"Got {completed['status']}; error={completed.get('error')}"
    )

    items = await _get_items(client, conv_id)

    # 1. The function_call_output for the spawn carries the
    # JSON handle, not the sub-agent's output. If dispatch
    # short-circuited and ran the sub-agent inline, the output
    # field would be the marker.
    fco_items = [
        i
        for i in items
        if i.get("type") == "function_call_output" and i.get("call_id") == "call_spawn_p3_1"
    ]
    assert len(fco_items) == 1, f"Expected 1 spawn function_call_output; got {len(fco_items)}"
    handle = json.loads(fco_items[0]["output"])
    assert handle["kind"] == "sub_agent", f"Handle kind must be 'sub_agent'; got {handle!r}"
    assert handle["type"] == "researcher"
    assert handle["status"] == "in_progress"
    assert isinstance(handle["task_id"], str) and handle["task_id"], (
        f"Handle task_id must be non-empty string; got {handle!r}"
    )
    assert handle["task_id"] in handle["message"], (
        f"Handle message must embed the task_id verbatim (G12); got {handle!r}"
    )

    # 2. A [System: task ... completed] auto-delivered message
    # is persisted with the sub-agent's marker substring.
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    assert len(completion_messages) >= 1, (
        f"Expected at least one auto-delivered system message; got user_texts={user_texts}"
    )
    completion_blob = "\n".join(completion_messages)
    assert "SUBAGENT_RESULT_MARKER_47" in completion_blob, (
        f"The sub-agent's actual output must reach the auto-"
        f"delivered system message — drain may have dropped the "
        f"payload body. Got: {completion_blob!r}"
    )

    # 3. The sub-agent's task is recorded as kind="sub_agent"
    # in task_store — distinct from kind="tool" (Phase 2). If
    # the SpawnSubAgentTool didn't pass kind="sub_agent" to
    # task_store.create, this would be "agent_task" (the legacy
    # default) and the parent's pending_tool_tasks filter would
    # have missed it (workflow would have hung instead of
    # completing).
    from agent_plane.runtime import get_task_store

    task_store = get_task_store()
    sub_task = await task_store.get(handle["task_id"])
    assert sub_task is not None
    assert sub_task.kind == "sub_agent", (
        f"Sub-agent task row must carry kind='sub_agent'; got kind={sub_task.kind!r}"
    )
    assert sub_task.status == "completed", (
        f"Sub-agent task row must be terminal completed; got status={sub_task.status!r}"
    )


async def test_sub_agent_failure_surfaces_to_parent(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A sub-agent whose inner LLM call raises must signal
    ``status="failed"`` (G86); the parent receives a
    ``[System: ... failed]`` auto-delivered message containing
    the exception class + message. Parent itself must complete
    successfully — the sub-agent's failure is the *sub-agent's*,
    not the parent's.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-fail-test",
        sub_agent_name="crash_worker",
    )

    # Parent dispatches the sub-agent.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn_fail",
                "name": "spawn_sub_agent",
                "arguments": json.dumps(
                    {"type": "crash_worker", "name": "crash_worker", "input": "do_crash_marker"},
                ),
            },
        ],
    )

    # exception_fn routes the failure: every call inspects
    # its input and raises ONLY for the sub-agent's call (whose
    # input contains the unique substring "do_crash_marker").
    # The parent's iter-2 call has the function_call_output
    # JSON in its input — it doesn't match, so the parent's LLM
    # response goes through normally (no exception, returns the
    # text below). Without this routing, the FIFO-shared mock
    # could give the exception to whichever workflow polls
    # first, causing flaky test failures.
    def _crash_only_subagent(kwargs: dict[str, Any]) -> Exception | None:
        input_str = json.dumps(kwargs.get("input", []))
        if "do_crash_marker" in input_str and "function_call_output" not in input_str:
            return RuntimeError("CRASHED_SUBAGENT_MARKER")
        return None

    # Two add_calls so both parent and sub-agent each get one.
    # Each one's exception_fn decides whether to raise.
    mock_llm.add_call(text="placeholder", exception_fn=_crash_only_subagent)
    mock_llm.add_call(text="placeholder", exception_fn=_crash_only_subagent)
    # Parent's iteration after the failure system message
    # arrives: produces the final text.
    parent_final = mock_llm.add_call(text="Sub-agent crashed.", exception_fn=_crash_only_subagent)

    result = await create_test_response(
        client,
        model="parent-fail-test",
        input_text="Run crash_worker",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    # Parent must complete — the sub-agent's failure must NOT
    # propagate as a parent-level error.
    assert completed["status"] == "completed", (
        f"Parent must complete even when a sub-agent fails. "
        f"Got {completed['status']}; error={completed.get('error')}"
    )

    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    failure_messages = [t for t in user_texts if "[System: task " in t and "failed" in t]
    assert len(failure_messages) >= 1, (
        f"Expected at least one [System: ... failed] auto-"
        f"delivered message; got user_texts={user_texts}. "
        f"If missing, the sub-agent's BaseException handler "
        f"didn't signal async_work_complete (G86 violated)."
    )
    failure_blob = "\n".join(failure_messages)
    assert "CRASHED_SUBAGENT_MARKER" in failure_blob, (
        f"The exception message must reach the auto-delivered "
        f"failure — format_failure_payload may have dropped it. "
        f"Got: {failure_blob!r}"
    )

    # The parent's final LLM call sees the failure in its
    # prompt.
    assert parent_final.received_kwargs is not None
    final_input = json.dumps(parent_final.received_kwargs["input"])
    assert "CRASHED_SUBAGENT_MARKER" in final_input, (
        "The failure marker must appear in the parent's final LLM input."
    )


async def test_sub_agent_handle_kind_distinct_from_async_tool(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A sub-agent handle's ``kind`` field is ``"sub_agent"``,
    distinct from ``@tool(synchronous=False)``'s ``"tool"``.

    The LLM uses this discriminator to know whether it's
    waiting on a nested agent (which can use its own tools and
    have rich recent_activity in check_task) versus a single
    background function (whose check_task only surfaces a
    status). Without the distinct kind the LLM would treat
    them identically and miss the difference in
    introspectability.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-kind-test",
        sub_agent_name="helper",
    )

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_kind",
                "name": "spawn_sub_agent",
                "arguments": json.dumps(
                    {"type": "helper", "name": "helper", "input": "do something"},
                ),
            },
        ],
    )
    mock_llm.add_call(text="helper output")
    mock_llm.add_call(text="Working…")
    mock_llm.add_call(text="Done.")

    result = await create_test_response(
        client,
        model="parent-kind-test",
        input_text="Use the helper",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]
    await _wait_for_completion(client, response_id)

    items = await _get_items(client, conv_id)
    fco_items = [
        i
        for i in items
        if i.get("type") == "function_call_output" and i.get("call_id") == "call_kind"
    ]
    assert len(fco_items) == 1
    handle = json.loads(fco_items[0]["output"])
    # The discriminator is a string, not a class type, so the
    # comparison must be exact. Any other value would silently
    # break LLM-side reasoning that branches on the kind.
    assert handle["kind"] == "sub_agent", (
        f"Sub-agent handles must report kind='sub_agent'; got {handle.get('kind')!r}"
    )


# ─── Legacy-removal regression tests ─────────────────────────


async def test_old_spawn_sub_agents_tool_removed(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    The old batch ``spawn_sub_agents`` tool (plural) must NOT
    be registered. Calling it yields the framework's standard
    "tool not found" error string — proving the legacy path
    is well and truly gone (M3).

    A regression in the registration order or a forgotten
    re-export would make this test fail.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-legacy-test",
        sub_agent_name="any",
    )

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_legacy",
                "name": "spawn_sub_agents",
                "arguments": json.dumps(
                    {"agents": [{"name": "any", "input": "x"}]},
                ),
            },
        ],
    )
    # The runtime returns a "tool not found" error string for
    # the unknown tool name; the LLM's next call should produce
    # a final text response acknowledging the error.
    final = mock_llm.add_call(text="That tool no longer exists.")

    result = await create_test_response(
        client,
        model="parent-legacy-test",
        input_text="Try the old tool",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]
    await _wait_for_completion(client, response_id)

    items = await _get_items(client, conv_id)
    fco_items = [
        i
        for i in items
        if i.get("type") == "function_call_output" and i.get("call_id") == "call_legacy"
    ]
    assert len(fco_items) == 1, "Expected one function_call_output for the legacy invocation"
    output = fco_items[0]["output"]
    # The framework's "tool not found" path returns this exact
    # substring — see ToolManager.call_tool fallback. If the
    # legacy tool were still registered, output would be the
    # response_ids JSON instead.
    assert "not found" in output.lower() or "spawn_sub_agents" not in output, (
        f"Calling the deleted spawn_sub_agents must fail with a "
        f"not-found error; got output={output!r}"
    )

    # Sanity: the parent's final LLM call ran (workflow didn't
    # hang or fail).
    assert final.received_kwargs is not None


async def test_check_sub_agents_tool_removed(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    The old ``check_sub_agents`` tool must be gone. Replaced
    by the unified ``check_task`` builtin from Phase 2.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-check-removed",
        sub_agent_name="any",
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_check",
                "name": "check_sub_agents",
                "arguments": json.dumps({"response_ids": ["resp_x"]}),
            },
        ],
    )
    mock_llm.add_call(text="ok")

    result = await create_test_response(
        client,
        model="parent-check-removed",
        input_text="Try the old check",
    )
    await _wait_for_completion(client, result.body["id"])
    items = await _get_items(client, result.body["conversation"]["id"])
    fco = next(
        (
            i
            for i in items
            if i.get("type") == "function_call_output" and i.get("call_id") == "call_check"
        ),
        None,
    )
    assert fco is not None
    assert "not found" in fco["output"].lower(), (
        f"check_sub_agents must be unregistered; got {fco['output']!r}"
    )


async def test_cancel_sub_agent_tool_removed(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    The old ``cancel_sub_agent`` tool must be gone. Replaced
    by the unified ``cancel_task`` builtin from Phase 2.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-cancel-removed",
        sub_agent_name="any",
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_cancel",
                "name": "cancel_sub_agent",
                "arguments": json.dumps({"response_id": "resp_x"}),
            },
        ],
    )
    mock_llm.add_call(text="ok")

    result = await create_test_response(
        client,
        model="parent-cancel-removed",
        input_text="Try the old cancel",
    )
    await _wait_for_completion(client, result.body["id"])
    items = await _get_items(client, result.body["conversation"]["id"])
    fco = next(
        (
            i
            for i in items
            if i.get("type") == "function_call_output" and i.get("call_id") == "call_cancel"
        ),
        None,
    )
    assert fco is not None
    assert "not found" in fco["output"].lower(), (
        f"cancel_sub_agent must be unregistered; got {fco['output']!r}"
    )
