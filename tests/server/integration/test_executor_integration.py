"""Integration tests for executor storage lifecycle and call_tool.

Tests use a custom executor injected via monkeypatch to exercise the
real workflow paths (restore → use → persist → cleanup) and the
call_tool callback (register → publish → poll → return) end-to-end
through the HTTP server.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_plane.runtime.executor import (
    Executor,
    ExecutorContext,
    ToolCallObserved,
    ToolCallRequested,
    TurnComplete,
)
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

# ── Test executor stubs ──────────────────────────────────
#
# These are real Executor subclasses — not MagicMock — so the
# workflow's isinstance checks and lifecycle calls work correctly.


_STORAGE_MARKER_FILE = "marker.txt"
_STORAGE_MARKER_CONTENT = "executor_was_here"


@dataclass
class _StorageProbe:
    """
    Shared state between the test and the executor to verify
    storage persistence without relying on MagicMock.

    :param marker_found_on_restore: Set to True by the executor
        if the marker file was present when storage_dir was
        restored (second task on the same conversation).
    :param on_task_start_called: Signals that on_task_start has
        run, so the test knows the executor has inspected the dir.
    """

    marker_found_on_restore: bool = False
    on_task_start_called: threading.Event = field(
        default_factory=threading.Event,
    )


class _StorageTestExecutor(Executor):
    """
    Executor that writes a marker file to storage_dir on first
    run and checks for it on subsequent runs.

    Used to verify that ``_persist_executor_storage`` and
    ``_restore_executor_storage`` round-trip files across tasks.

    :param probe: Shared probe for communicating results back
        to the test.
    """

    def __init__(self, probe: _StorageProbe) -> None:
        self._probe = probe

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> _StorageTestExecutor:
        """
        Not used — instances are constructed directly by the test.

        :param spec: Ignored.
        :returns: Never called.
        """
        raise NotImplementedError

    def on_task_start(self, context: ExecutorContext) -> None:
        """
        Check for marker file, then write one if absent.

        :param context: Executor context with storage_dir.
        """
        marker = context.storage_dir / _STORAGE_MARKER_FILE
        self._probe.marker_found_on_restore = marker.exists()
        if not marker.exists():
            marker.write_text(_STORAGE_MARKER_CONTENT)
        self._probe.on_task_start_called.set()

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[TurnComplete]:
        """
        Return a simple text response.

        :param messages: Ignored.
        :param tools: Ignored.
        :param system_prompt: Ignored.
        :param llm_config: Ignored.
        :param context: Ignored.
        """
        found = self._probe.marker_found_on_restore
        yield TurnComplete(
            text="RESTORED" if found else "FRESH",
        )

    def max_context_tokens(self) -> int | None:
        """
        Return None so the workflow skips @step and compaction.

        :returns: None.
        """
        return None


@dataclass
class _AwaitToolProbe:
    """
    Shared state for the call_tool integration test.

    :param tool_result_received: The tool result content
        returned by the call_tool callback. None until
        the callback returns.
    :param callback_returned: Signals that the executor's
        run_turn completed after receiving the tool result.
    """

    tool_result_received: str | None = None
    callback_returned: threading.Event = field(
        default_factory=threading.Event,
    )


class _AwaitToolExecutor(Executor):
    """
    Executor that calls ``context.call_tool()`` during
    ``run_turn()`` to exercise the client-side tool bridging path.

    :param probe: Shared probe for communicating results.
    """

    def __init__(self, probe: _AwaitToolProbe) -> None:
        self._probe = probe

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> _AwaitToolExecutor:
        """
        Not used — instances are constructed directly by the test.

        :param spec: Ignored.
        :returns: Never called.
        """
        raise NotImplementedError

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[TurnComplete]:
        """
        Call ``call_tool`` for a client-side tool, then
        yield the result as text.

        :param messages: Ignored.
        :param tools: Ignored.
        :param system_prompt: Ignored.
        :param llm_config: Ignored.
        :param context: Provides the ``call_tool`` callback.
        """
        call = ToolCallRequested(
            call_id=f"call_integ_{int(time.monotonic() * 1000)}",
            name="Read",
            arguments={"file_path": "/tmp/test.txt"},
        )
        result = context.call_tool(call)
        self._probe.tool_result_received = result.content
        self._probe.callback_returned.set()
        yield TurnComplete(text=f"Tool returned: {result.content}")

    def max_context_tokens(self) -> int | None:
        """
        Return None so the workflow skips @step and compaction.

        :returns: None.
        """
        return None


# ── Helpers ──────────────────────────────────────────────


async def _poll_until_terminal(
    client: httpx.AsyncClient,
    response_id: str,
    # 200 attempts × 0.1s sleep = 20s max wait.
    # Workflow tasks with stub executors complete in < 2s;
    # 20s provides margin for CI slowness.
    max_attempts: int = 200,
) -> dict[str, Any]:
    """
    Poll GET /v1/responses/{id} until the response reaches a
    terminal state.

    :param client: The HTTP test client.
    :param response_id: The response ID to poll.
    :param max_attempts: Maximum poll iterations before failing.
    :returns: The terminal response body.
    """
    for _ in range(max_attempts):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.1)
    raise AssertionError(f"Response {response_id} never reached terminal state")


async def _poll_for_pending(
    client: httpx.AsyncClient,
    response_id: str,
    # 100 attempts × 0.1s sleep = 10s max wait.
    # The executor registers the pending call within the first
    # LLM turn, so it appears quickly; 10s is generous.
    max_attempts: int = 100,
) -> list[dict[str, Any]]:
    """
    Poll GET /v1/responses/{id} until ``action_required``
    function_call items appear in the output.

    :param client: The HTTP test client.
    :param response_id: The response ID to poll.
    :param max_attempts: Maximum poll iterations.
    :returns: List of pending function_call output items.
    """
    for _ in range(max_attempts):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        pending = [
            item
            for item in body.get("output", [])
            if (item.get("type") == "function_call" and item.get("status") == "action_required")
        ]
        if pending:
            return pending
        if body["status"] in ("completed", "failed"):
            return []
        await asyncio.sleep(0.1)
    return []


def _extract_output_texts(body: dict[str, Any]) -> list[str]:
    """
    Pull all text strings from ``message`` output items in a
    response body.

    :param body: The JSON response body from GET /v1/responses/{id}.
    :returns: A flat list of text strings from output_text content
        blocks across all message items.
    """
    return [
        block["text"]
        for item in body.get("output", [])
        if item.get("type") == "message"
        for block in item.get("content", [])
    ]


# ── Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_executor_storage_persists_across_tasks(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Files written to ``storage_dir`` by the executor during task 1
    are restored from the artifact store and available during task 2
    on the same conversation.

    Exercises the full lifecycle: ``_get_or_restore_executor_storage`` →
    executor writes marker → ``_persist_executor_storage`` →
    next task finds marker on disk (or restores from artifact store).

    **What breaks if the feature is removed:**

    - If ``_persist_executor_storage`` is deleted, the artifact
      snapshot is lost after server restart → storage is empty.
    - If ``_get_or_restore_executor_storage`` is deleted, the
      stable directory is never created → executor has no storage.
    """
    await create_test_agent(client)

    # Use tmp_path so tests don't write to ~/.agent-plane.
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._EXECUTOR_STORAGE_BASE",
        tmp_path / "exec_storage",
    )

    # Task 1: executor writes marker, workflow persists it.
    probe_1 = _StorageProbe()
    executor_1 = _StorageTestExecutor(probe_1)
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor_1,
    )

    resp_1 = await create_test_response(
        client,
        input_text="First turn",
    )
    response_1_id = resp_1.body["id"]
    conv_id = resp_1.body["conversation"]["id"]

    body_1 = await _poll_until_terminal(client, response_1_id)
    assert body_1["status"] == "completed"

    # First task: no prior snapshot → marker NOT found.
    assert probe_1.marker_found_on_restore is False, (
        "First task should start with an empty storage_dir. "
        "If True, a stale snapshot leaked from a prior test."
    )

    # Verify the artifact store has the snapshot.
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    artifact_key = f"executor_storage/{conv_id}/test-agent.tar.gz"
    assert artifact_store.exists(artifact_key), (
        "Artifact snapshot not found after task 1. "
        "_persist_executor_storage did not run or the key "
        f"is wrong. Expected key: {artifact_key}"
    )

    # Task 2: same conversation → executor should find the marker.
    probe_2 = _StorageProbe()
    executor_2 = _StorageTestExecutor(probe_2)
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor_2,
    )

    resp_2 = await create_test_response(
        client,
        input_text="Second turn",
        previous_response_id=response_1_id,
    )
    response_2_id = resp_2.body["id"]

    body_2 = await _poll_until_terminal(client, response_2_id)
    assert body_2["status"] == "completed"

    # Second task: marker WAS found → storage round-tripped.
    assert probe_2.marker_found_on_restore is True, (
        "Second task did not find the marker file. Either "
        "_persist_executor_storage failed to snapshot or "
        "_restore_executor_storage failed to extract."
    )

    # Verify the response text confirms the executor saw the marker.
    output_texts = _extract_output_texts(body_2)
    assert any("RESTORED" in t for t in output_texts), (
        f"Expected 'RESTORED' in output, got {output_texts}. "
        f"The executor's run_turn did not see the marker."
    )


@pytest.mark.asyncio
async def test_call_tool_park_patch_resume(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An executor that calls ``context.call_tool()`` parks
    the tool call, the client discovers it via GET, PATCHes the
    result, and the executor receives it to complete the task.

    Exercises the full path: ``_register_client_tool_call`` →
    ``_publish_client_tool_call`` (live SSE) →
    ``_poll_for_tool_result`` → PATCH handler completes the row →
    poll finds the result → executor returns.

    **What breaks if the feature is removed:**

    - If ``_register_client_tool_call`` is deleted, the PATCH
      handler returns 404 (call_id not found) → task hangs.
    - If ``_publish_client_tool_call`` is deleted, the client
      never sees the function_call in GET output → no PATCH.
    - If ``_poll_for_tool_result`` is deleted, the executor
      never gets the result → task hangs forever.
    """
    await create_test_agent(client)

    probe = _AwaitToolProbe()
    executor = _AwaitToolExecutor(probe)
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor,
    )

    resp = await create_test_response(
        client,
        input_text="Read that file",
        background=True,
    )
    response_id = resp.body["id"]
    assert resp.body["status"] == "queued"

    # Poll until the pending function_call appears in GET output.
    # This exercises _register_client_tool_call (store row) and
    # _publish_client_tool_call (live SSE → GET output rebuild).
    pending = await _poll_for_pending(client, response_id)
    assert len(pending) >= 1, (
        "No action_required function_call appeared in GET output. "
        "Either _register_client_tool_call didn't create the row "
        "or the GET endpoint doesn't surface pending tool calls."
    )

    pending_call = pending[0]
    call_id = pending_call["call_id"]
    # Verify the pending item has the expected tool metadata.
    assert pending_call["name"] == "Read", (
        f"Expected tool name 'Read', got {pending_call['name']!r}."
    )

    # PATCH the tool result — exercises complete_pending_tool_call
    # in the store, which _poll_for_tool_result detects.
    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {"call_id": call_id, "output": "file contents here"},
            ],
        },
    )
    assert patch_resp.status_code == 200, (
        f"PATCH failed with {patch_resp.status_code}: "
        f"{patch_resp.text}. The pending tool call row may not "
        f"have been created by _register_client_tool_call."
    )

    # Wait for the executor to complete.
    body = await _poll_until_terminal(client, response_id)
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. "
        f"The executor may not have received the tool result "
        f"from _poll_for_tool_result."
    )

    # Verify the tool result made it through the full pipeline:
    # PATCH → store → _poll_for_tool_result → executor → output.
    output_texts = _extract_output_texts(body)
    assert any("file contents here" in t for t in output_texts), (
        f"Expected 'file contents here' in output, got "
        f"{output_texts}. The tool result did not traverse "
        f"from the PATCH through _poll_for_tool_result to "
        f"the executor's TurnComplete text."
    )

    # Verify the probe confirms the executor received the result.
    assert probe.tool_result_received == "file contents here", (
        f"Probe shows executor got {probe.tool_result_received!r}. "
        f"_poll_for_tool_result returned wrong content."
    )


# ── ToolCallObserved executor stub ──────────────────────


class _ObservedToolExecutor(Executor):
    """
    Executor that yields ``ToolCallObserved`` events to simulate
    an executor (e.g. Claude SDK) running tools internally.

    Returns ``None`` from ``max_context_tokens()`` so the workflow
    uses the executor-managed (live-only) path.
    """

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> _ObservedToolExecutor:
        """
        Not used — instances are constructed directly by the test.

        :param spec: Ignored.
        :returns: Never called.
        """
        raise NotImplementedError

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[ToolCallObserved | TurnComplete]:
        """
        Yield two ``ToolCallObserved`` events and a ``TurnComplete``.

        The observations simulate an executor that ran ``Read`` and
        ``Bash`` tools internally before producing a final response.

        :param messages: Ignored.
        :param tools: Ignored.
        :param system_prompt: Ignored.
        :param llm_config: Ignored.
        :param context: Ignored.
        """
        yield ToolCallObserved(
            call_id="obs_read_1",
            name="Read",
            arguments={"file_path": "/tmp/test.txt"},
            result="file contents here",
            status="completed",
            duration_ms=42.0,
        )
        yield ToolCallObserved(
            call_id="obs_bash_1",
            name="Bash",
            arguments={"command": "echo hello"},
            result="hello\n",
            status="completed",
            duration_ms=15.0,
        )
        yield TurnComplete(text="Done reading and running.")

    def max_context_tokens(self) -> int | None:
        """
        Return None so the workflow uses the executor-managed path.

        :returns: None.
        """
        return None


# ── ToolCallObserved test ───────────────────────────────


@pytest.mark.asyncio
async def test_observed_tool_calls_persisted_and_in_output(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``ToolCallObserved`` events from an executor-managed executor
    are persisted to the conversation store as ``function_call`` +
    ``function_call_output`` pairs and appear in the GET response
    output.

    Exercises the full path: executor yields ``ToolCallObserved`` →
    ``_emit_executor_live_only`` streams SSE → ``_events_to_response_dict``
    collects into ``"observed_tool_calls"`` → ``_persist_observed_tool_calls``
    writes to conversation store → GET rebuilds output from store.

    **What breaks if the feature is removed:**

    - If ``_emit_executor_live_only`` ignores ``ToolCallObserved``,
      SSE clients never see tool call events during the turn.
    - If ``_events_to_response_dict`` drops ``ToolCallObserved``,
      ``_persist_observed_tool_calls`` has nothing to persist →
      the GET response output is missing function_call items.
    - If ``_persist_observed_tool_calls`` is deleted, the GET
      response output lacks function_call items even though SSE
      streamed them → assertion on output items fails.
    """
    await create_test_agent(client)

    executor = _ObservedToolExecutor()
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor,
    )

    resp = await create_test_response(
        client,
        input_text="Read the file and run the command",
        background=True,
    )
    response_id = resp.body["id"]
    assert resp.body["status"] == "queued"

    body = await _poll_until_terminal(client, response_id)
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. The executor may have raised an error."
    )

    # Verify the final text made it through.
    output_texts = _extract_output_texts(body)
    assert any("Done reading and running." in t for t in output_texts), (
        f"Expected 'Done reading and running.' in output, got "
        f"{output_texts}. TurnComplete text was lost."
    )

    # Verify function_call items appear in the GET response output.
    # 2 ToolCallObserved → 2 function_call + 2 function_call_output items.
    fc_items = [item for item in body.get("output", []) if item.get("type") == "function_call"]
    fc_output_items = [
        item for item in body.get("output", []) if item.get("type") == "function_call_output"
    ]
    # 2 function_call items: one for Read (obs_read_1), one for Bash (obs_bash_1).
    assert len(fc_items) == 2, (
        f"Expected 2 function_call items (Read + Bash), got "
        f"{len(fc_items)}. If 0, _persist_observed_tool_calls "
        f"did not run or _events_to_response_dict dropped the events."
    )
    # 2 function_call_output items: one for each observation.
    assert len(fc_output_items) == 2, (
        f"Expected 2 function_call_output items, got "
        f"{len(fc_output_items)}. Observations were partially persisted."
    )

    # Verify the function_call items have the correct tool names
    # and call_ids — proves the data traversed from executor event
    # through _persist_observed_tool_calls to the conversation store.
    fc_names = {item["name"] for item in fc_items}
    assert fc_names == {"Read", "Bash"}, (
        f"Expected tool names {{'Read', 'Bash'}}, got {fc_names}. "
        f"Tool metadata was lost during persistence."
    )
    fc_call_ids = {item["call_id"] for item in fc_items}
    assert fc_call_ids == {"obs_read_1", "obs_bash_1"}, (
        f"Expected call_ids {{'obs_read_1', 'obs_bash_1'}}, got "
        f"{fc_call_ids}. Call IDs were mutated during persistence."
    )

    # Verify the function_call_output items have the correct results.
    fc_output_by_call_id = {item["call_id"]: item["output"] for item in fc_output_items}
    assert fc_output_by_call_id["obs_read_1"] == "file contents here", (
        f"Read tool output mismatch: {fc_output_by_call_id.get('obs_read_1')!r}."
    )
    assert fc_output_by_call_id["obs_bash_1"] == "hello\n", (
        f"Bash tool output mismatch: {fc_output_by_call_id.get('obs_bash_1')!r}."
    )

    # Verify persistence: items exist in the conversation store.
    conv_id = body["conversation"]["id"]
    conv_store = SqlAlchemyConversationStore(db_uri)
    fc_store_items = conv_store.list_items(
        conv_id,
        type="function_call",
    )
    fc_output_store_items = conv_store.list_items(
        conv_id,
        type="function_call_output",
    )
    # 2 function_call rows persisted to the store.
    assert len(fc_store_items.data) == 2, (
        f"Expected 2 function_call rows in store, got "
        f"{len(fc_store_items.data)}. _persist_observed_tool_calls "
        f"did not write to the conversation store."
    )
    # 2 function_call_output rows persisted to the store.
    assert len(fc_output_store_items.data) == 2, (
        f"Expected 2 function_call_output rows in store, got "
        f"{len(fc_output_store_items.data)}. Observations were "
        f"partially persisted."
    )


class _TextOnlyExecutor(Executor):
    """
    Executor that yields only a ``TurnComplete`` with fixed text.

    Returns ``None`` from ``max_context_tokens()`` so the workflow
    uses the executor-managed path.

    :param text: The response text to return.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> _TextOnlyExecutor:
        """
        Not used — instances are constructed directly by the test.

        :param spec: Ignored.
        :returns: Never called.
        """
        raise NotImplementedError

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[TurnComplete]:
        """
        Yield a single ``TurnComplete`` with the configured text.

        :param messages: Ignored.
        :param tools: Ignored.
        :param system_prompt: Ignored.
        :param llm_config: Ignored.
        :param context: Ignored.
        """
        yield TurnComplete(text=self._text)

    def max_context_tokens(self) -> int | None:
        """
        Return None so the workflow uses the executor-managed path.

        :returns: None.
        """
        return None


@pytest.mark.asyncio
async def test_second_turn_works_after_observed_tool_calls(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A second turn on the same conversation succeeds when the first
    turn persisted ``ToolCallObserved`` items (function_call +
    function_call_output).

    This catches regressions where persisted observed tool call
    items in history break the second turn — e.g. the steering
    inbox misinterpreting them as late messages, the executor
    choking on function_call items in the message list, or
    ``_handle_final_response`` failing because ``write_stream``
    is called outside a step context.

    **What breaks if the feature is wrong:**

    - If ``_persist_observed_tool_calls`` doesn't advance
      ``last_seen``, ``_check_steering_inbox`` sees the observed
      items as steered messages → infinite loop (1000 iterations).
    - If ``_handle_final_response`` / ``_persist_and_stream``
      crashes on the executor-managed path, the second turn
      fails with no output.
    """
    await create_test_agent(client)

    # Turn 1: executor produces observed tool calls.
    executor_1 = _ObservedToolExecutor()
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor_1,
    )

    resp_1 = await create_test_response(
        client,
        input_text="Read the file",
        background=True,
    )
    response_1_id = resp_1.body["id"]
    body_1 = await _poll_until_terminal(client, response_1_id)
    # First turn must complete for the test to be meaningful.
    assert body_1["status"] == "completed", f"Turn 1 failed with status {body_1['status']}."

    # Turn 2: same conversation, text-only response.
    executor_2 = _TextOnlyExecutor("Second turn reply.")
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor_2,
    )

    resp_2 = await create_test_response(
        client,
        input_text="Follow-up question",
        previous_response_id=response_1_id,
        background=True,
    )
    response_2_id = resp_2.body["id"]
    body_2 = await _poll_until_terminal(client, response_2_id)

    # Second turn must complete — not hang or fail.
    assert body_2["status"] == "completed", (
        f"Turn 2 failed with status {body_2['status']}. "
        f"Observed tool call items from turn 1 may have broken "
        f"the second turn (steering inbox, write_stream context, "
        f"or message conversion)."
    )

    # Verify the response text proves the executor ran.
    output_texts = _extract_output_texts(body_2)
    assert any("Second turn reply." in t for t in output_texts), (
        f"Expected 'Second turn reply.' in output, got "
        f"{output_texts}. The second turn executor did not "
        f"produce output."
    )
