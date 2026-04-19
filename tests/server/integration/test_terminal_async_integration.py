"""Integration tests for ``terminal_run(synchronous=False)``.

Exercises the full async-terminal pipeline through mocked LLM calls:
parent workflow → ``TerminalRunTool.dispatch_async`` →
``background_terminal_workflow`` → ``async_work_complete`` drain →
parent's next iteration.

Separate file from ``test_terminal_integration.py`` (which covers
the sync path: shell_busy, 10-shell cap, crash recovery, idle reaper)
because the async tests need the real runtime init (task store,
DBOS) for the background-workflow dispatch — and the full mock-LLM
fixture that drives the parent loop through multiple iterations.
"""

from __future__ import annotations

import io
import json
import tarfile
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response
from tests.server.integration.test_local_tool_integration import (
    _wait_for_completion,
)

pytestmark = [pytest.mark.asyncio]

_AGENT_NAME = "terminal-async-agent"


def _build_async_terminal_agent_bundle() -> bytes:
    """Build an agent bundle with the three terminal builtins plus
    the task lifecycle builtins (which the manager auto-registers
    on any agent with ``terminal_run`` — exercised via the full
    bundle path).

    :returns: Raw tar.gz bytes for an agent whose tools are
        ``terminal_run``, ``terminal_list``, ``terminal_close``.
    """
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {
            "model": _AGENT_NAME,
            "connection": {"api_key": "test-key"},
        },
        "tools": {
            "builtins": ["terminal_run", "terminal_list", "terminal_close"],
        },
    }
    config_bytes = yaml.dump(config).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


async def _create_async_terminal_agent(client: httpx.AsyncClient) -> None:
    """Upload the async-terminal agent bundle.

    :param client: HTTP client pointed at the test server.
    """
    bundle = _build_async_terminal_agent_bundle()
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent creation failed: {resp.status_code} {resp.text}"


def _tool_outputs(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and JSON-parse every function_call_output.

    Fails loud on un-parseable JSON or non-dict payloads.

    :param body: The response body from ``GET /v1/responses/{id}``.
    :returns: Parsed tool-output dicts with ``_call_id`` spliced in.
    :raises AssertionError: On un-parseable or non-dict outputs.
    """
    parsed: list[dict[str, Any]] = []
    for item in body.get("output", []):
        if item.get("type") != "function_call_output":
            continue
        raw = item.get("output") or ""
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise AssertionError(
                f"function_call_output is not JSON-parseable: "
                f"call_id={item.get('call_id')!r}, raw={raw[:200]!r} ({exc})"
            ) from exc
        assert isinstance(obj, dict), (
            f"function_call_output was parseable but not a dict: "
            f"{obj!r} (call_id={item.get('call_id')!r})"
        )
        obj["_call_id"] = item.get("call_id")
        parsed.append(obj)
    return parsed


@pytest.fixture(autouse=True)
def _force_unsandboxed_terminal(
    monkeypatch: pytest.MonkeyPatch,
    task_store: Any,
) -> None:
    """Force the terminal registry to spawn unsandboxed shells.

    Same rationale as the sync-path integration tests: this file
    asserts manager / workflow behavior that's independent of srt
    wrapping, and avoids a hard dep on srt+node being on PATH in
    every test environment.

    :param monkeypatch: Pytest's monkeypatch fixture.
    :param task_store: Ordering dependency. The ``task_store``
        fixture calls ``init_runtime()``, which installs a fresh
        sandbox-enabled ``TerminalManagerRegistry`` into
        ``_globals``. This fixture must run *after* that so the
        unsandboxed monkeypatch is not overwritten.
    """
    from agent_plane.runtime import _globals
    from agent_plane.terminals import TerminalManagerRegistry

    monkeypatch.setattr(
        _globals,
        "_terminal_registry",
        TerminalManagerRegistry(sandbox_enabled=False),
    )


# ── Dispatch returns an _AsyncToolHandle with a task_id ───────


async def test_terminal_run_async_returns_handle_immediately(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """``terminal_run(synchronous=False)`` returns an _AsyncToolHandle.

    First LLM call issues one ``terminal_run`` with
    ``synchronous=false`` and a fast ``echo`` command. The parent's
    function_call_output must contain a ``task_id`` and an
    ``in_progress`` status — proving the workflow dispatched a
    child instead of blocking on the command.

    Second LLM call lets the parent finalize with a text response
    after the drain delivers the child's completion.

    Failure modes caught:
    - is_async not firing on synchronous=false → tool goes through
      _call_tool synchronously, result contains stdout instead of
      handle.
    - dispatch_async raises or returns wrong shape → parent errors
      or LLM sees malformed output.
    """
    await _create_async_terminal_agent(client)

    # Call 1: the async terminal_run.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_echo",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {
                        "command": "echo async-handle-smoke",
                        "shell": "default",
                        "synchronous": False,
                    }
                ),
            },
        ],
    )
    # Call 2: parent finalizes after child completes. The mock's
    # default behavior for unqueued calls is to return an empty
    # text; we want a non-empty one so the workflow can finalize.
    mock_llm.add_call(text="background started.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Start a background echo.",
    )
    assert result.status_code == 200
    body = await _wait_for_completion(client, result.body["id"])
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    outputs = _tool_outputs(body)
    assert len(outputs) == 1, (
        f"Expected exactly one function_call_output (the async "
        f"handle), got {len(outputs)}: "
        f"{[o.get('task_id') or o.get('status') for o in outputs]}"
    )
    handle = outputs[0]
    # The _AsyncToolHandle serialization: top-level task_id +
    # status="in_progress" + a message telling the LLM to use
    # check_task / cancel_task.
    assert isinstance(handle.get("task_id"), str) and handle["task_id"], (
        f"Expected a non-empty task_id in the handle, got {handle!r}."
    )
    assert handle.get("status") == "in_progress", (
        f"Expected status='in_progress' on the handle (async "
        f"dispatch hallmark), got {handle.get('status')!r}. If "
        f"'completed', the tool ran synchronously — is_async "
        f"isn't detecting synchronous=false."
    )
    assert "check_task" in (handle.get("message") or ""), (
        f"Handle message should name check_task so the LLM knows "
        f"how to poll. Got: {handle.get('message')!r}"
    )


# ── Completion auto-delivers as a system message ──────────────


async def test_completed_async_terminal_command_auto_delivers(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """After the async terminal command completes, the result
    appears in the parent's conversation as a ``[System: task ...]``
    user message.

    Proves the end-to-end drain path:
    1. Parent dispatches async terminal_run.
    2. background_terminal_workflow runs the command, sends
       async_work_complete.
    3. Parent's drain wakes, formats payload, persists as user
       message.
    4. LLM's next iteration sees the message.

    Failure modes caught:
    - Drain not picking up the signal → task hangs forever.
    - Payload not formatted → system message missing or malformed.
    - Terminal kind not recognized by formatter → garbage text.
    """
    await _create_async_terminal_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_completion",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {
                        "command": "echo async-complete-token",
                        "shell": "default",
                        "synchronous": False,
                    }
                ),
            },
        ],
    )
    # Second LLM call: the drain injects the completion and re-calls
    # the LLM. We return a final text.
    mock_llm.add_call(text="final: command completed.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Run a quick async echo.",
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed"

    # Fetch the conversation's items and assert a "[System: task ...
    # (terminal) completed]" message exists with the echo's stdout.
    conv_id = body["conversation"]["id"]
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    system_messages = []
    for item in items:
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        for block in item.get("content", []):
            text = block.get("text") or ""
            if text.startswith("[System: task"):
                system_messages.append(text)
    assert any("(terminal) completed" in m for m in system_messages), (
        f"Expected a '[System: task X (terminal) completed]' user "
        f"message after the async drain ran. Got system messages: "
        f"{system_messages}"
    )
    # And the stdout token is in the auto-delivered body — proves
    # the formatter embedded the terminal result (not just a header).
    assert any("async-complete-token" in m for m in system_messages), (
        f"Stdout token missing from system messages — the result "
        f"payload wasn't formatted with the terminal output. "
        f"Messages: {system_messages}"
    )


# ── Multiple parallel async terminal dispatches ───────────────


async def test_parallel_async_terminals_each_get_own_handle(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Two parallel terminal_run(synchronous=false) calls produce
    two distinct handles.

    Proves the dispatch path doesn't share state across parallel
    calls in one LLM response. Each child gets its own task_id,
    its own shell (they use different names here), and both auto-
    deliver.
    """
    await _create_async_terminal_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_dev",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {
                        "command": "echo from-dev-shell",
                        "shell": "dev",
                        "synchronous": False,
                    }
                ),
            },
            {
                "call_id": "call_test",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {
                        "command": "echo from-test-shell",
                        "shell": "test",
                        "synchronous": False,
                    }
                ),
            },
        ],
    )
    mock_llm.add_call(text="both started.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Start two background echoes.",
    )
    body = await _wait_for_completion(client, result.body["id"])
    assert body["status"] == "completed"

    outputs = _tool_outputs(body)
    assert len(outputs) == 2, (
        f"Expected 2 handles (one per parallel dispatch), got {len(outputs)}."
    )
    task_ids = {o.get("task_id") for o in outputs}
    # Distinct task_ids prove the dispatch path didn't return the
    # same handle twice (e.g. due to a shared state bug). Using a
    # set comprehension + len check catches both "missing task_id"
    # (None would show up once) and "duplicate" (set shrinks).
    assert len(task_ids) == 2 and None not in task_ids, (
        f"Expected two distinct task_ids, got {task_ids!r}."
    )


# ── Kind is recorded on the child task row ───────────────────


async def test_async_terminal_child_task_has_terminal_kind(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """The child task row created by ``TerminalRunTool.dispatch_async``
    has ``kind="terminal"``.

    Matters because ``check_task`` / ``cancel_task`` dispatch
    terminal-specific behavior (stdout deltas, SIGINT) based on
    this kind. If kind were accidentally set to "tool" or
    "sub_agent", those features would silently do the wrong thing.
    """
    from agent_plane.runtime import get_task_store

    await _create_async_terminal_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_kind_check",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {
                        "command": "echo kind-check",
                        "shell": "default",
                        "synchronous": False,
                    }
                ),
            },
        ],
    )
    mock_llm.add_call(text="done.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Kind check.",
    )
    body = await _wait_for_completion(client, result.body["id"])

    outputs = _tool_outputs(body)
    assert len(outputs) == 1
    child_task_id = outputs[0]["task_id"]

    # Inspect the child task row directly via the task store.
    # Proves the ``kind`` field was set correctly regardless of
    # how check_task/list_tasks later surface it.
    task_store = get_task_store()
    # task_store.get_sync reads from DBOS which requires the async
    # variant when inside an event loop; use get_task instead.
    child_task = await task_store.get(child_task_id)
    assert child_task is not None, f"Child task {child_task_id!r} not found in task store."
    assert child_task.kind == "terminal", (
        f"Expected child task kind='terminal' (so check/cancel "
        f"dispatch terminal-specific logic), got {child_task.kind!r}."
    )


# ── Kind="terminal" auto-enables task_lifecycle tools ─────────


async def test_terminal_agent_has_task_lifecycle_tools_registered(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """An agent declaring ``terminal_run`` automatically gets
    ``check_task`` / ``cancel_task`` / ``list_tasks`` available.

    Exercises the auto-registration in
    ``ToolManager._register_task_lifecycle_tools``. Without it, an
    agent with only terminal_run would have NO way to poll or
    cancel an async command — defeating the whole synchronous=False
    surface.

    Proof path: start a tiny synchronous call so the workflow
    actually runs and builds a ToolManager, then have a second
    LLM call that issues ``list_tasks({})``. If list_tasks isn't
    registered, the mock LLM's call gets rejected as an unknown
    tool (surfaces as a function_call_output with an error).
    """
    await _create_async_terminal_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_sync_warmup",
                "name": "terminal_run",
                "arguments": json.dumps({"command": "echo warmup", "shell": "default"}),
            },
        ],
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_list_tasks",
                "name": "list_tasks",
                "arguments": "{}",
            },
        ],
    )
    mock_llm.add_call(text="listed.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Warmup then list.",
    )
    body = await _wait_for_completion(client, result.body["id"])
    assert body["status"] == "completed"

    # Verify list_tasks was actually dispatched (i.e. the tool was
    # recognized by ToolManager). If auto-enable didn't work, the
    # tool would be "unknown" and the output would be an error
    # string from the tool manager's dispatch, not a JSON shape
    # matching list_tasks.
    outputs = _tool_outputs(body)
    # The list_tasks call output: `{"tasks": [...]}` per its schema.
    list_outputs = [o for o in outputs if o.get("_call_id") == "call_list_tasks"]
    assert len(list_outputs) == 1, (
        f"Expected exactly one output from call_list_tasks, got "
        f"{len(list_outputs)}. If 0, list_tasks wasn't registered."
    )
    assert "tasks" in list_outputs[0], (
        f"list_tasks output should contain a 'tasks' key, got "
        f"{list_outputs[0]!r}. Key missing = tool not recognized = "
        f"auto-registration broken."
    )


# ── cancel_task for kind="terminal" interrupts the running shell ─


async def test_cancel_task_interrupts_async_terminal(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """``cancel_task`` on a running terminal task SIGINTs the shell
    and the eventual drain delivers status="cancelled".

    The flow:
    1. Mock LLM kicks off an async ``sleep 30`` via terminal_run.
    2. Mock LLM's next call issues ``cancel_task(task_id)`` using
       the handle from step 1.
    3. Cancel sends SIGINT → bash kills sleep → run_sync returns
       status="killed" → background workflow sends
       async_work_complete(status="cancelled").
    4. Parent drain delivers a "[System: task X (terminal) cancelled]"
       message; final LLM call acknowledges.

    Failure modes caught:
    - cancel_task's terminal-kind branch doesn't interrupt (blocks
      for 30s, test times out).
    - SIGINT reaches bash but status classification is wrong
      ("completed" instead of "cancelled").
    - Drain doesn't pick up the cancellation signal.
    """
    from agent_plane.runtime.background_tool_workflow import (
        AsyncWorkCompletePayload,  # noqa: F401 — used via check below
    )

    await _create_async_terminal_agent(client)

    # First LLM call: start the async sleep.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_start_sleep",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {
                        "command": "sleep 30",
                        "shell": "default",
                        "synchronous": False,
                    }
                ),
            },
        ],
    )

    # The second LLM call needs the handle's task_id to cancel.
    # We use ``tool_calls_fn`` so the arguments are computed from
    # the prior call's kwargs (the function_call_outputs are in
    # the messages passed to this create() call).
    def _cancel_fn(create_kwargs: dict[str, Any]) -> list[dict[str, str]]:
        """Extract the prior task_id from the messages and emit
        a cancel_task call against it.

        :param create_kwargs: A dict of the litellm create() kwargs
            including ``input`` messages; we scan for the
            function_call_output JSON.
        :returns: A single cancel_task tool call targeting the
            handle's task_id.
        """
        # ``input`` is a list of dicts in OpenAI Responses shape.
        input_items: list[dict[str, Any]] = create_kwargs.get("input", [])
        handle_task_id: str | None = None
        for item in input_items:
            if item.get("type") != "function_call_output":
                continue
            try:
                parsed = json.loads(item.get("output") or "")
            except (ValueError, TypeError):
                continue
            tid = parsed.get("task_id")
            if isinstance(tid, str) and tid:
                handle_task_id = tid
                break
        assert handle_task_id is not None, (
            f"Expected a function_call_output containing a task_id "
            f"in the LLM input, but found none. Input items: "
            f"{[i.get('type') for i in input_items]}"
        )
        return [
            {
                "call_id": "call_cancel",
                "name": "cancel_task",
                "arguments": json.dumps({"task_id": handle_task_id}),
            },
        ]

    mock_llm.add_call(tool_calls_fn=_cancel_fn)
    # Third LLM call: after cancel_task output arrives, the parent
    # doesn't immediately finalize — pending async work (the child
    # terminal workflow) still exists, so the drain blocks here.
    # Return a bridge text that the parent will persist while it
    # waits for the drain to deliver the cancellation signal.
    mock_llm.add_call(text="cancelling now...")
    # Fourth LLM call: after the drain delivers the "[System: task
    # ... cancelled]" message, the parent re-invokes the LLM. We
    # return the final acknowledgement so the workflow finalizes.
    mock_llm.add_call(text="cancelled.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Start sleep then cancel it.",
    )
    assert result.status_code == 200
    response_id = result.body["id"]

    # Extended poll — the default 20s isn't enough for this test
    # because the sleep 30 is running in parallel with the cancel
    # path. Even with cancel working, the chain (call 1 → dispatch
    # → call 2 → cancel → SIGINT → child sends signal → drain →
    # call 3/4 → finalize) needs a few seconds on a warm test env.
    async def _wait_long(response_id: str) -> dict[str, Any]:
        """Poll up to 60s for terminal status."""
        import asyncio as _asyncio

        for _ in range(600):
            resp = await client.get(f"/v1/responses/{response_id}")
            body_ = resp.json()
            if body_["status"] in ("completed", "failed", "cancelled"):
                return body_
            await _asyncio.sleep(0.1)
        raise AssertionError(f"Response {response_id} did not reach terminal status in 60s")

    body = await _wait_long(response_id)
    assert body["status"] == "completed", (
        f"Parent task didn't reach completed (likely cancel_task "
        f"failed to interrupt and we waited for the sleep). "
        f"Error: {body.get('error')}"
    )

    # Check conversation items for the cancelled-terminal system
    # message. Without the SIGINT path, the workflow would either
    # time out or deliver a different status.
    conv_id = body["conversation"]["id"]
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
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
        f"Expected a '[System: task X (terminal) cancelled]' "
        f"message, got {system_messages!r}. If the message says "
        f"'completed' instead, the SIGINT path didn't flip status "
        f"to killed; if no message at all, the drain missed the "
        f"signal."
    )
