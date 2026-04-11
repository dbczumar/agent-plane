"""Helpers for durability integration tests.

Provides server lifecycle, polling, and assertion utilities used
by ``test_durability.py``. Separated so the test file contains
only fixtures and test functions.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from agent_plane.runtime import init as init_runtime
from agent_plane.runtime.agent_cache import AgentCache
from agent_plane.runtime.durability import destroy_dbos
from agent_plane.server.app import create_app
from agent_plane.stores.agent_store.sqlalchemy_store import (
    SqlAlchemyAgentStore,
)
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.file_store.sqlalchemy_store import (
    SqlAlchemyFileStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import (
    SqlAlchemyTaskStore,
)
from agent_plane.tools import ToolManager
from agent_plane.tools.base import ToolContext
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

# ── Dataclasses ──────────────────────────────────────


@dataclass
class ToolGate:
    """
    Synchronization gate for blocking tool execution in tests.

    :param should_block: When ``True``, the tracking wrapper
        blocks on ``release`` before executing the real tool.
        Set to ``False`` to let tool calls pass through.
    :param entered: Set by the tracking wrapper when a tool
        call starts blocking. Tests ``wait()`` on this to
        confirm the tool was reached, e.g.
        ``gate.entered.wait(timeout=10)``.
    :param release: Set by the test to unblock a waiting tool
        call, e.g. ``gate.release.set()``.
    """

    should_block: bool = False
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)


@dataclass
class ToolTracking:
    """
    Result of setting up tool call tracking.

    :param invocations: Shared list recording tool names that
        were actually executed (not replayed from DBOS cache),
        e.g. ``["load_skill"]``.
    :param patch_ctx: Context manager that patches
        ``ToolManager.call_tool``. Use with
        ``with tracking.patch_ctx:``.
    :param gate: Optional gate for blocking tool execution.
    """

    invocations: list[str]
    # AbstractContextManager because unittest.mock._patch is
    # a private type; only the context-manager protocol matters.
    patch_ctx: AbstractContextManager[object]
    gate: ToolGate | None = None


@dataclass
class WorkflowIds:
    """
    Identifiers returned after starting a workflow.

    :param response_id: The response ID, e.g.
        ``"resp_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    """

    response_id: str
    conversation_id: str


@dataclass
class RecoveryResult:
    """
    Result of polling a recovered workflow to terminal state.

    :param terminal_body: The terminal response body dict
        from the API.
    :param client: The HTTP client for the recovered server,
        usable for further assertions.
    """

    # Any: JSON response bodies are inherently heterogeneous.
    terminal_body: dict[str, Any]
    client: httpx.AsyncClient


# ── Constants ────────────────────────────────────────

# Tool call spec used to trigger a checkpointed @step before
# the crash. Uses load_skill with a nonexistent skill — the
# tool result doesn't matter, only that _call_tool executes
# and DBOS checkpoints it.
TOOL_CALL_SPEC = [
    {
        "call_id": "call_durability_1",
        "name": "load_skill",
        "arguments": '{"name": "nonexistent"}',
    },
]


# ── Server lifecycle ─────────────────────────────────


def build_server(
    db_uri: str,
    tmp_path: Path,
    mock_llm: ControllableMockClient,
) -> httpx.AsyncClient:
    """
    Build a complete server stack (stores + runtime + app +
    HTTP client) on the given database.

    Mirrors the fixture chain in ``conftest.py`` but is callable
    multiple times in a single test to simulate server restarts.

    :param db_uri: SQLite database URI, e.g.
        ``"sqlite:///path/to/test.db"``.
    :param tmp_path: Temp directory for artifact and cache
        storage.
    :param mock_llm: The mock LLM client to patch into the
        workflow.
    :returns: An ``httpx.AsyncClient`` wired to the new app.
    """
    agent_store = SqlAlchemyAgentStore(db_uri)
    conversation_store = SqlAlchemyConversationStore(db_uri)
    artifact_store = LocalArtifactStore(
        str(tmp_path / "artifacts"),
    )
    agent_cache = AgentCache(
        artifact_store=artifact_store,
        cache_dir=tmp_path / ".cache",
    )
    task_store = SqlAlchemyTaskStore(db_uri)
    init_runtime(
        conversation_store=conversation_store,
        task_store=task_store,
        agent_store=agent_store,
        agent_cache=agent_cache,
    )

    # Patch the LLM client in both workflow and executor modules
    # so the mock is used everywhere.
    import agent_plane.runtime.executors.default as exec_mod
    import agent_plane.runtime.workflow as wf_mod

    wf_mod._get_llm_client = lambda: mock_llm  # type: ignore[assignment]
    exec_mod._get_llm_client = lambda: mock_llm  # type: ignore[assignment]

    # Force a known context window so the executor uses
    # _checkpointed_turn (@step) for LLM calls. Without this,
    # unknown test models return None from max_context_tokens(),
    # causing _consume_executor_live (no @step) to be used.
    # Durability tests require the @step path so DBOS can
    # checkpoint and replay LLM calls on crash recovery.
    exec_mod._get_model_context_window = lambda model: 128000  # type: ignore[assignment]

    app = create_app(
        agent_store=agent_store,
        file_store=SqlAlchemyFileStore(db_uri),
        task_store=task_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    )


# ── Polling ──────────────────────────────────────────


async def poll_until_terminal(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll a response until it reaches a terminal state.

    :param client: The HTTP client to poll with.
    :param response_id: The response ID to poll, e.g.
        ``"resp_abc123"``.
    :returns: The terminal response body dict.
    :raises AssertionError: If the response never reaches a
        terminal state within 200 iterations.
    """
    for _ in range(200):
        resp = await client.get(
            f"/v1/responses/{response_id}",
        )
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.1)

    raise AssertionError(
        f"Response {response_id} never reached terminal state",
    )


# ── Tool tracking ────────────────────────────────────


def setup_tool_tracking(
    gate: ToolGate | None = None,
) -> ToolTracking:
    """
    Create a ``ToolManager.call_tool`` wrapper that records
    each invocation to a shared list.

    The list survives server rebuilds because it lives in the
    test process, not in the server.

    :param gate: Optional gate for blocking tool calls. When
        provided and ``gate.should_block`` is ``True``, the
        wrapper blocks until ``gate.release`` is set.
    :returns: A :class:`ToolTracking` with the invocation list,
        patch context manager, and optional gate.
    """
    invocations: list[str] = []
    original_call_tool = ToolManager.call_tool

    def _tracking_call_tool(
        self: ToolManager,
        name: str,
        arguments: str,
        ctx: ToolContext,
    ) -> str:
        """
        Wrapper that records each real tool execution.

        :param self: The ToolManager instance.
        :param name: Tool name, e.g. ``"load_skill"``.
        :param arguments: JSON arguments string.
        :param ctx: Server-side execution context.
        :returns: The original ``call_tool`` result.
        """
        invocations.append(name)
        if gate is not None and gate.should_block:
            gate.entered.set()
            gate.release.wait()
        return original_call_tool(self, name, arguments, ctx)

    patch_ctx = patch.object(
        ToolManager,
        "call_tool",
        _tracking_call_tool,
    )
    return ToolTracking(
        invocations=invocations,
        patch_ctx=patch_ctx,
        gate=gate,
    )


# ── Server phases: crash mid-LLM (test 1) ───────────


async def run_server_1(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
    tracking: ToolTracking,
) -> WorkflowIds:
    """
    Build server 1, start a workflow with a tool call, then
    crash mid-LLM.

    :param mock_llm: The shared mock LLM client.
    :param db_uri: SQLite database URI.
    :param tmp_path: Temp directory for artifacts/cache.
    :param tracking: Tool tracking — asserted to have exactly
        1 invocation before crash.
    :returns: :class:`WorkflowIds` with response and
        conversation IDs.
    """
    # Pre-crash agent loop has 2 LLM calls:
    #   Call 1: returns tool call -> tool executes (checkpointed)
    #   Call 2: next loop iteration, blocks mid-response (crash)
    mock_llm.add_call(tool_calls=TOOL_CALL_SPEC)
    call_block = mock_llm.add_call(
        text="This text should never appear",
        block=True,
    )

    client_1 = build_server(db_uri, tmp_path, mock_llm)
    await create_test_agent(client_1)

    created = await create_test_response(
        client_1,
        input_text="Durable request",
    )
    response_id = created.body["id"]
    conv_id = created.body["conversation"]["id"]
    assert created.body["status"] == "queued"

    # Gate: workflow passed the tool call and entered
    # LLM call 2 (the blocked one).  Use asyncio.to_thread so the
    # test event loop stays free — DBOS runs the async workflow on
    # this loop, so a blocking wait() would deadlock it.
    reached = await asyncio.to_thread(call_block.call_event.wait, 10)
    assert reached, "LLM call 2 was never reached (workflow stalled)"

    assert len(tracking.invocations) == 1, (
        f"Expected 1 tool invocation pre-crash, got {len(tracking.invocations)}"
    )

    # Crash: do NOT release — simulates process death
    await client_1.aclose()
    destroy_dbos()

    return WorkflowIds(
        response_id=response_id,
        conversation_id=conv_id,
    )


async def run_server_2(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
    response_id: str,
) -> RecoveryResult:
    """
    Build server 2 on the same DB and wait for DBOS to
    recover the workflow.

    Recovery replays all completed ``@step`` functions from
    cache: LLM call 1 (tool call response) and ``_call_tool``
    are both returned from DBOS cache without re-executing.
    Only LLM call 2 (which was interrupted mid-flight and
    never checkpointed) re-executes with a fresh mock
    response.

    :param mock_llm: The shared mock LLM client.
    :param db_uri: SQLite database URI (same DB as server 1).
    :param tmp_path: Temp directory for artifacts/cache.
    :param response_id: The response ID to poll, e.g.
        ``"resp_abc123"``.
    :returns: :class:`RecoveryResult` with terminal body and
        HTTP client.
    """
    # Only one mock call needed: call 2 (interrupted
    # mid-flight, never checkpointed). Call 1 and _call_tool
    # both replay from DBOS step cache.
    mock_llm.add_call(text="Recovered after server restart")

    client_2 = build_server(db_uri, tmp_path, mock_llm)
    terminal_body = await poll_until_terminal(
        client_2,
        response_id,
    )
    return RecoveryResult(
        terminal_body=terminal_body,
        client=client_2,
    )


# ── Server phases: crash mid-tool (test 2) ──────────


async def run_server_1_crash_mid_tool(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
    gate: ToolGate,
) -> WorkflowIds:
    """
    Build server 1, start a workflow, and crash while a tool
    call is mid-execution.

    The agent loop calls LLM -> gets tool call -> enters
    ``_call_tool`` @step -> the tracking wrapper blocks via
    ``gate``. Since the @step never completes, DBOS has no
    checkpoint for it.

    :param mock_llm: The shared mock LLM client.
    :param db_uri: SQLite database URI.
    :param tmp_path: Temp directory for artifacts/cache.
    :param gate: Tool gate with ``should_block=True``. The
        test waits on ``gate.entered`` then crashes without
        setting ``gate.release``.
    :returns: :class:`WorkflowIds` with response and
        conversation IDs.
    """
    # LLM call 1 returns a tool call. The tool blocks via gate.
    mock_llm.add_call(tool_calls=TOOL_CALL_SPEC)

    client_1 = build_server(db_uri, tmp_path, mock_llm)
    await create_test_agent(client_1)

    created = await create_test_response(
        client_1,
        input_text="Tool crash request",
    )
    response_id = created.body["id"]
    conv_id = created.body["conversation"]["id"]

    # Wait for tool to be entered (blocked via gate).
    # Use asyncio.to_thread so the test event loop stays free —
    # DBOS runs the async workflow on this loop.
    entered = await asyncio.to_thread(gate.entered.wait, 10)
    assert entered, "Tool gate was never entered (workflow stalled)"

    # Crash: do NOT release gate — simulates process death
    # while tool is mid-execution. The _call_tool @step never
    # completes, so DBOS has no checkpoint for it.
    await client_1.aclose()
    destroy_dbos()

    return WorkflowIds(
        response_id=response_id,
        conversation_id=conv_id,
    )


async def run_server_2_after_tool_crash(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
    response_id: str,
) -> RecoveryResult:
    """
    Build server 2 after a mid-tool crash and wait for
    recovery.

    On recovery, DBOS replays LLM call 1 from cache (it
    completed and was checkpointed). ``_call_tool`` has no
    checkpoint (crashed mid-execution) so it re-executes.
    After the tool completes, the workflow continues to
    LLM call 2 which needs a fresh mock response.

    :param mock_llm: The shared mock LLM client.
    :param db_uri: SQLite database URI (same DB as server 1).
    :param tmp_path: Temp directory for artifacts/cache.
    :param response_id: The response ID to poll, e.g.
        ``"resp_abc123"``.
    :returns: :class:`RecoveryResult` with terminal body and
        HTTP client.
    """
    # Recovery needs 1 mock call: LLM call 2 (the post-tool
    # response that never happened pre-crash). LLM call 1
    # replays from cache. _call_tool re-executes (no cache).
    mock_llm.add_call(text="Recovered after tool crash")

    client_2 = build_server(db_uri, tmp_path, mock_llm)
    terminal_body = await poll_until_terminal(
        client_2,
        response_id,
    )
    return RecoveryResult(
        terminal_body=terminal_body,
        client=client_2,
    )


# ── Server phases: steering survives crash (test 3) ──


async def run_server_1_with_steering(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
) -> WorkflowIds:
    """
    Build server 1, start a workflow, inject a steered
    message, then crash.

    ``try_deliver`` writes the steered message in its own
    SQLAlchemy transaction, independent of the DBOS workflow.
    The message should survive the crash.

    :param mock_llm: The shared mock LLM client.
    :param db_uri: SQLite database URI.
    :param tmp_path: Temp directory for artifacts/cache.
    :returns: :class:`WorkflowIds` with response and
        conversation IDs.
    """
    # Block the LLM call so the workflow is mid-execution
    # when we inject the steered message.
    call_block = mock_llm.add_call(
        text="This text should never appear",
        block=True,
    )

    client_1 = build_server(db_uri, tmp_path, mock_llm)
    await create_test_agent(client_1)

    created = await create_test_response(
        client_1,
        input_text="First message",
    )
    response_id = created.body["id"]
    conv_id = created.body["conversation"]["id"]

    # Wait for LLM to be entered (workflow is running).
    # Use asyncio.to_thread so the test event loop stays free —
    # DBOS runs the async workflow on this loop.
    reached = await asyncio.to_thread(call_block.call_event.wait, 10)
    assert reached, "LLM call was never reached (workflow stalled)"

    # Inject steered message via the API. try_deliver writes
    # to conversation_items in its own transaction, independent
    # of the DBOS workflow transaction.
    steered = await create_test_response(
        client_1,
        input_text="Steered message",
        previous_response_id=response_id,
    )
    # Steering should succeed: task is active, inbox is open.
    assert steered.status_code == 200, (
        f"Steering failed with status {steered.status_code}: {steered.body}"
    )

    # Crash: do NOT release LLM — simulates process death.
    await client_1.aclose()
    destroy_dbos()

    return WorkflowIds(
        response_id=response_id,
        conversation_id=conv_id,
    )


async def run_server_2_after_steering_crash(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
    response_id: str,
) -> RecoveryResult:
    """
    Build server 2 after a crash with a steered message
    pending in the database.

    On recovery, the LLM call re-executes (it was blocked
    and never checkpointed). ``_sync_history`` discovers the
    steered message in the conversation store.

    :param mock_llm: The shared mock LLM client.
    :param db_uri: SQLite database URI (same DB as server 1).
    :param tmp_path: Temp directory for artifacts/cache.
    :param response_id: The response ID to poll, e.g.
        ``"resp_abc123"``.
    :returns: :class:`RecoveryResult` with terminal body and
        HTTP client.
    """
    # Recovery needs 1 mock call: the LLM call that was
    # blocked pre-crash re-executes with the steered message
    # now visible in history.
    mock_llm.add_call(text="Recovery after steering")

    client_2 = build_server(db_uri, tmp_path, mock_llm)
    terminal_body = await poll_until_terminal(
        client_2,
        response_id,
    )
    return RecoveryResult(
        terminal_body=terminal_body,
        client=client_2,
    )


# ── Assertions ───────────────────────────────────────


async def assert_recovery_output(
    terminal_body: dict[str, Any],
    recovery_text: str,
) -> None:
    """
    Assert the recovered workflow produced the expected output.

    :param terminal_body: The terminal response body from the
        API.
    :param recovery_text: The expected assistant text, e.g.
        ``"Recovered after server restart"``.
    """
    assert terminal_body["status"] == "completed", (
        f"Expected completed, got {terminal_body['status']}: {terminal_body.get('error')}"
    )
    output = terminal_body["output"]
    assistant_outputs = [o for o in output if o.get("role") == "assistant"]
    assert len(assistant_outputs) >= 1, "No assistant output after recovery"
    assert assistant_outputs[0]["content"][0]["text"] == recovery_text


async def assert_conversation_persisted(
    client: httpx.AsyncClient,
    conv_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """
    Assert conversation items survived the crash.

    :param client: The HTTP client for the recovered server.
    :param conv_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param user_text: Expected user message text, e.g.
        ``"Durable request"``.
    :param assistant_text: Expected assistant response text,
        e.g. ``"Recovered after server restart"``.
    """
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    user_items = [i for i in items if i.get("role") == "user"]
    assistant_items = [i for i in items if i.get("role") == "assistant"]

    assert len(user_items) >= 1, "User message not persisted"
    assert user_items[0]["content"][0]["text"] == user_text

    assert len(assistant_items) >= 1, "Assistant response not persisted after recovery"
    assert assistant_items[0]["content"][0]["text"] == assistant_text


async def assert_steering_persisted(
    client: httpx.AsyncClient,
    conv_id: str,
) -> None:
    """
    Assert that a steered message survived the crash and
    appears in the conversation alongside the original user
    message and the recovery assistant response.

    :param client: The HTTP client for the recovered server.
    :param conv_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    """
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    user_items = [i for i in items if i.get("role") == "user"]
    assistant_items = [i for i in items if i.get("role") == "assistant"]

    # Both the original message and the steered message
    # should be in the conversation.
    user_texts = [i["content"][0]["text"] for i in user_items if i.get("content")]
    assert "First message" in user_texts, f"Original user message not found in {user_texts}"
    assert "Steered message" in user_texts, f"Steered message not found in {user_texts}"

    # The recovery assistant response should be present.
    assert len(assistant_items) >= 1, "No assistant response after recovery"
    assert assistant_items[0]["content"][0]["text"] == "Recovery after steering"


def assert_step_cache_replay(
    tracking: ToolTracking,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Assert that completed ``@step`` functions were replayed
    from DBOS cache on recovery, not re-executed.

    Checks two things:

    1. ``_call_tool`` ran exactly once (pre-crash). On
       recovery DBOS returns its cached output without
       calling ``ToolManager.call_tool`` again.

    2. The mock LLM was called exactly 3 times total:
       - Pre-crash: 2 calls (call 1 returns tool call,
         call 2 blocks — the crash point)
       - Recovery: 1 call (call 2 re-executes because it
         never completed and has no checkpoint; call 1
         replays from DBOS step cache)
       If the count is 4, call 1 was re-executed on
       recovery instead of replayed from cache.

    :param tracking: The :class:`ToolTracking` with the
        invocations list.
    :param mock_llm: The shared mock LLM client whose
        ``call_count`` tracks total invocations across both
        server instances.
    """
    assert len(tracking.invocations) == 1, (
        f"Tool should have executed exactly once "
        f"(DBOS cache replay), but ran "
        f"{len(tracking.invocations)} time(s). "
        f"If 2, the step was re-executed instead of "
        f"replayed from cache."
    )
    # 3 = 2 pre-crash + 1 recovery (only the interrupted
    # LLM call re-executes; the completed one replays)
    assert mock_llm.call_count == 3, (
        f"Expected 3 LLM calls (2 pre-crash + 1 recovery), "
        f"got {mock_llm.call_count}. If 4, the first LLM "
        f"call was re-executed instead of replayed from "
        f"DBOS step cache."
    )


def assert_incomplete_step_reexecuted(
    tracking: ToolTracking,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Assert that an incomplete ``_call_tool`` step re-executed
    on recovery (no DBOS cache entry) while the completed LLM
    step replayed from cache.

    :param tracking: The :class:`ToolTracking` with the
        invocations list.
    :param mock_llm: The shared mock LLM client.
    """
    # Tool ran twice: once pre-crash (tracking wrapper
    # recorded it before blocking, but the @step never
    # completed), once on recovery (re-executed because
    # DBOS had no cached output for it).
    assert len(tracking.invocations) == 2, (
        f"Tool should have executed exactly twice "
        f"(once pre-crash, once on recovery), but ran "
        f"{len(tracking.invocations)} time(s)."
    )
    # LLM mock calls consumed:
    #   Pre-crash: 1 (call 1 returns tool call, checkpointed)
    #   Recovery:  0 for call 1 (replayed from DBOS cache)
    #            + 1 for call 2 (post-tool response, fresh)
    #   Total: 2
    assert mock_llm.call_count == 2, (
        f"Expected 2 LLM mock calls (1 pre-crash + "
        f"1 recovery), got {mock_llm.call_count}. If 3, "
        f"LLM call 1 was re-executed instead of replayed "
        f"from DBOS step cache."
    )
