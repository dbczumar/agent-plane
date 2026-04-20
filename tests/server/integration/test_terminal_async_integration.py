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


async def test_completed_async_terminal_surfaces_via_check_task(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """After an async terminal command completes, ``check_task`` on
    the handle surfaces ``status="completed"`` and the result.

    Since terminals were removed from ``_DRAIN_KINDS`` (turn
    finalization must not block on sessions that might never end),
    the agent polls via ``check_task`` rather than receiving an
    auto-delivered system message. This test exercises that
    polling contract end-to-end: dispatch → child workflow runs →
    DBOS persists the result → next LLM turn calls check_task →
    sees completed + output.

    Failure modes caught:
    - Background terminal workflow not persisting status to DBOS
      (check_task would see in_progress forever).
    - Task-store enrichment from DBOS broken for terminal kind
      (check_task would see agent_task-like defaults).
    - check_task's terminal-kind branch dropping ``result`` field
      on completion (LLM sees status but no output).
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

    # Second LLM call: the LLM polls check_task after the first
    # turn's terminal_run returned a handle. We pass wait_ms to
    # let the tool block server-side for the short echo to
    # complete rather than returning stale in_progress.
    def _check_task_fn(create_kwargs: dict[str, Any]) -> list[dict[str, str]]:
        handle_task_id = _extract_terminal_run_task_id(create_kwargs)
        return [
            {
                "call_id": "call_check_terminal",
                "name": "check_task",
                "arguments": json.dumps({"task_id": handle_task_id, "wait_ms": 5000}),
            }
        ]

    mock_llm.add_call(tool_calls_fn=_check_task_fn)
    # Third LLM call: final ack after we've inspected the result.
    mock_llm.add_call(text="final: command completed.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Run a quick async echo and check on it.",
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed"

    # The check_task function_call_output should contain status=
    # "completed" and the echo's stdout. This is the load-bearing
    # assertion: if the background workflow didn't persist its
    # result to DBOS, check_task sees in_progress and the test
    # fails loudly.
    check_outputs = []
    for item in body.get("output", []):
        if item.get("type") != "function_call_output":
            continue
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if parsed.get("kind") == "terminal":
            check_outputs.append(parsed)
    assert check_outputs, (
        f"Expected at least one check_task function_call_output "
        f"with kind='terminal', got none. Items: "
        f"{[i.get('type') for i in body.get('output', [])]}"
    )
    completed = [co for co in check_outputs if co.get("status") == "completed"]
    assert completed, (
        f"No check_task saw the terminal task in 'completed' "
        f"status. Observed statuses: "
        f"{[co.get('status') for co in check_outputs]}. "
        f"Likely means background_terminal_workflow didn't persist "
        f"the completion to DBOS, or wait_ms=5000 wasn't long "
        f"enough for the echo to finish (unlikely — echo is ~1ms)."
    )
    result_field = completed[-1].get("result") or ""
    assert "async-complete-token" in result_field, (
        f"check_task returned status=completed but the terminal "
        f"result doesn't contain the echo's stdout token. Got "
        f"result={result_field!r}. The task_store's DBOS-result "
        f"enrichment didn't surface the payload, or the formatter "
        f"dropped it."
    )


def _extract_terminal_run_task_id(create_kwargs: dict[str, Any]) -> str:
    """Pull the task_id out of a terminal_run's prior
    function_call_output so the next turn can reference it.

    :param create_kwargs: The litellm-shaped ``create()`` kwargs
        containing the ``input`` list the mock was called with.
    :returns: The task_id from the most recent terminal_run handle.
    :raises AssertionError: If no such handle is in the input.
    """
    input_items: list[dict[str, Any]] = create_kwargs.get("input", [])
    for item in reversed(input_items):
        if item.get("type") != "function_call_output":
            continue
        try:
            parsed = json.loads(item.get("output") or "")
        except (ValueError, TypeError):
            continue
        tid = parsed.get("task_id")
        if isinstance(tid, str) and tid and parsed.get("tool_name") == "terminal_run":
            return tid
    raise AssertionError(
        "No terminal_run handle in LLM input; test setup invariant "
        "broken — the prior turn should have produced one."
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
    and ``check_task`` subsequently sees ``status="cancelled"``.

    The flow:
    1. Mock LLM kicks off an async ``sleep 30`` via terminal_run.
    2. Mock LLM's next call issues ``cancel_task(task_id)`` using
       the handle from step 1.
    3. Cancel sends SIGINT → bash kills sleep → run_sync returns
       status="killed" → background workflow writes
       status="cancelled" into DBOS.
    4. Mock LLM's third call polls ``check_task(task_id,
       wait_ms=…)`` and the tool sees ``status="cancelled"``.

    Terminals are no longer in ``_DRAIN_KINDS`` so the turn does
    not block on auto-delivery; the agent polls explicitly. See
    designs/PERSISTENT_TERMINAL_RESEARCH.md §6.12 for rationale.

    Failure modes caught:
    - cancel_task's terminal-kind branch doesn't interrupt
      (check_task would see running or completed, not cancelled).
    - SIGINT reaches bash but status classification is wrong
      ("completed" instead of "cancelled").
    - Background terminal workflow never persists the cancelled
      status to DBOS (check_task hangs or shows in_progress).
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

    # Third LLM call: after cancel_task's output arrives, poll
    # check_task with wait_ms so the tool itself blocks briefly
    # for the child workflow to finalize its cancelled status in
    # DBOS. Without the drain (terminals aren't in _DRAIN_KINDS
    # anymore) the parent turn doesn't wait for the child, so the
    # LLM has to explicitly observe the post-cancel state.
    def _check_after_cancel_fn(create_kwargs: dict[str, Any]) -> list[dict[str, str]]:
        """Emit a check_task call pointed at the handle, with a
        wait_ms budget big enough to cover SIGINT → child workflow
        finalize → DBOS status update.

        :param create_kwargs: Litellm create() kwargs.
        :returns: One check_task call.
        """
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
            if isinstance(tid, str) and tid and parsed.get("tool_name") == "terminal_run":
                handle_task_id = tid
                break
        assert handle_task_id is not None, (
            f"No terminal_run handle in input; setup invariant broken. "
            f"Input items: {[i.get('type') for i in input_items]}"
        )
        return [
            {
                "call_id": "call_check_after_cancel",
                "name": "check_task",
                "arguments": json.dumps({"task_id": handle_task_id, "wait_ms": 5000}),
            }
        ]

    mock_llm.add_call(tool_calls_fn=_check_after_cancel_fn)
    # Fourth LLM call: final ack.
    mock_llm.add_call(text="cancelled.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Start sleep then cancel it.",
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed", (
        f"Parent task didn't reach completed: {body.get('error')}"
    )

    # The check_task function_call_output must report
    # status="cancelled" for the terminal handle. Without a working
    # SIGINT path, status would be "in_progress" (sleep still
    # running) or "completed" (sleep finished). Either is a
    # regression.
    check_outputs = []
    for item in body.get("output", []):
        if item.get("type") != "function_call_output":
            continue
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if parsed.get("kind") == "terminal":
            check_outputs.append(parsed)
    assert check_outputs, (
        "No check_task output observed for the terminal task. "
        "The mock LLM's third call didn't produce a visible "
        "check_task call, or the check_task tool errored before "
        "reaching the payload-build stage."
    )
    cancelled = [co for co in check_outputs if co.get("status") == "cancelled"]
    assert cancelled, (
        f"No check_task saw the terminal task as 'cancelled'. "
        f"Observed statuses: {[co.get('status') for co in check_outputs]}. "
        f"If 'in_progress', the 5000 ms wait_ms expired before the "
        f"child workflow finalized — the SIGINT path is too slow "
        f"or broken. If 'completed', the SIGINT path didn't flip "
        f"status to killed (sleep ran to completion)."
    )
