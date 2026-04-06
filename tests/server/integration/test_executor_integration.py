"""Integration tests for executor storage lifecycle and await_tool_output.

Tests use a custom executor injected via monkeypatch to exercise the
real workflow paths (restore → use → persist → cleanup) and the
await_tool_output callback (register → publish → poll → return) end-to-end
through the HTTP server.
"""

from __future__ import annotations

import asyncio
import json
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
    TextChunk,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
)
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig
from agent_plane.stores.artifact_store.local import LocalArtifactStore
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
    Shared state for the await_tool_output integration test.

    :param tool_result_received: The tool result content
        returned by the await_tool_output callback. None until
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
    Executor that calls ``context.await_tool_output()`` during
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
        Call ``await_tool_output`` for a client-side tool, then
        yield the result as text.

        :param messages: Ignored.
        :param tools: Ignored.
        :param system_prompt: Ignored.
        :param llm_config: Ignored.
        :param context: Provides the ``await_tool_output`` callback.
        """
        call = ToolCallRequested(
            call_id=f"call_integ_{int(time.monotonic() * 1000)}",
            name="Read",
            arguments={"file_path": "/tmp/test.txt"},
        )
        result = context.await_tool_output(call)
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
    raise AssertionError(
        f"Response {response_id} never reached terminal state"
    )


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
            if (
                item.get("type") == "function_call"
                and item.get("status") == "action_required"
            )
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
        block.get("text", "")
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

    Exercises the full lifecycle: ``_restore_executor_storage`` →
    executor writes marker → ``_persist_executor_storage`` →
    ``_cleanup_executor_storage`` → next task restores marker.

    **What breaks if the feature is removed:**

    - If ``_persist_executor_storage`` is deleted, the marker file
      is lost after task 1 ends → ``marker_found_on_restore`` is
      False on task 2 → assertion fails.
    - If ``_restore_executor_storage`` is deleted, the artifact
      snapshot is never extracted → same failure.
    - If ``_cleanup_executor_storage`` leaves the temp dir around
      but the artifact store is empty, a fresh temp dir on
      task 2 won't have the marker.
    """
    await create_test_agent(client)

    # Task 1: executor writes marker, workflow persists it.
    probe_1 = _StorageProbe()
    executor_1 = _StorageTestExecutor(probe_1)
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._create_executor",
        lambda _spec: executor_1,
    )

    resp_1 = await create_test_response(
        client, input_text="First turn",
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
    artifact_key = f"executor_storage/{conv_id}.tar.gz"
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
async def test_await_tool_output_park_patch_resume(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An executor that calls ``context.await_tool_output()`` parks
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
