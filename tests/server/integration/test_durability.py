"""Durability integration tests — crash recovery via DBOS.

Each test simulates a full server crash: tear down the HTTP client,
FastAPI app, all stores, and the DBOS singleton while a workflow is
mid-execution. Then rebuild everything from scratch on the same
database and verify DBOS recovers the pending workflow.

This mirrors production crash recovery: process dies, process
restarts, DBOS.launch() finds pending workflows and re-enqueues
them.

IMPORTANT: All tests in this file must use the
``pinned_dbos_version`` fixture (autouse). DBOS only recovers
workflows whose ``application_version`` matches the current
instance. Without a pinned version, the restarted DBOS computes a
different hash and silently skips recovery.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_plane.runtime.durability import destroy_dbos, ensure_dbos
from tests.server.conftest import ControllableMockClient
from tests.server.integration.durability_helpers import (
    ToolGate,
    assert_conversation_persisted,
    assert_incomplete_step_reexecuted,
    assert_recovery_output,
    assert_steering_persisted,
    assert_step_cache_replay,
    run_server_1,
    run_server_1_crash_mid_tool,
    run_server_1_with_steering,
    run_server_2,
    run_server_2_after_steering_crash,
    run_server_2_after_tool_crash,
    setup_tool_tracking,
)

pytestmark = pytest.mark.asyncio

# Fixed version string used for both the initial DBOS init
# and the post-crash reinit. Without this, DBOS assigns different
# auto-computed hashes and the restarted instance refuses to
# recover "foreign" workflows.
_DBOS_VERSION = "test-durability-v1"


@pytest.fixture(autouse=True)
def pinned_dbos_version() -> Iterator[None]:
    """
    Patch ``ensure_dbos`` so every call uses a fixed
    ``application_version``. Applied before the ``task_store``
    fixture (which calls ``ensure_dbos`` during store init)
    and remains active for the rebuilt server inside the test.
    """
    original = ensure_dbos

    def _pinned(
        uri: str,
        *,
        application_version: str | None = None,
    ) -> None:
        """
        Wrapper that forces a fixed application_version.

        :param uri: Database URI forwarded to the real
            ``ensure_dbos``.
        :param application_version: Accepted for signature
            compatibility but overridden with
            ``_DBOS_VERSION``.
        """
        original(uri, application_version=_DBOS_VERSION)

    with patch(
        "agent_plane.runtime.durability.ensure_dbos",
        side_effect=_pinned,
    ):
        with patch(
            "agent_plane.stores.task_store.sqlalchemy_store.ensure_dbos",
            side_effect=_pinned,
        ):
            yield


async def test_workflow_recovers_after_server_restart(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """
    A workflow interrupted mid-LLM-call completes after a full
    server restart on the same database — and DBOS replays
    completed ``@step`` functions from cache instead of
    re-executing them (partial replay).

    Lifecycle:
    1. Build server instance 1 (stores + app + DBOS)
    2. LLM call 1 (``@step``) returns a tool call ->
       ``_call_tool`` (also ``@step``) executes. Both are
       checkpointed by DBOS.
    3. LLM call 2 (``@step``) blocks (simulating crash
       mid-LLM — this step never completes, so no checkpoint)
    4. Tear down the ENTIRE server (client, app, DBOS)
    5. Build server instance 2 on the same DB
    6. DBOS recovers: LLM call 1 and ``_call_tool`` both
       replay from cache (bodies NOT called). LLM call 2
       re-executes (no checkpoint existed) with recovery text.
    7. Assert recovery text in output and conversation
    8. Assert tool body ran exactly once (proves partial
       replay from cache, not full re-execution)

    Breakage this catches:
    - Workflow silently disappears after crash (no recovery)
    - Recovered workflow returns stale/empty output
    - Task status stuck in in_progress forever
    - Conversation items not persisted by recovered workflow
    - Store reconstruction loses data
    - DBOS step cache not persisted across restart (LLM or
      tool re-executes instead of replaying from cache)
    """
    tracking = setup_tool_tracking()

    with tracking.patch_ctx:
        ids = await run_server_1(
            mock_llm,
            db_uri,
            tmp_path,
            tracking,
        )
        result = await run_server_2(
            mock_llm,
            db_uri,
            tmp_path,
            ids.response_id,
        )

    recovery_text = "Recovered after server restart"
    await assert_recovery_output(result.terminal_body, recovery_text)
    await assert_conversation_persisted(
        result.client,
        ids.conversation_id,
        "Durable request",
        recovery_text,
    )
    assert_step_cache_replay(tracking, mock_llm)

    await result.client.aclose()
    destroy_dbos()


async def test_incomplete_step_reexecutes_after_crash(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """
    A ``_call_tool`` step interrupted mid-execution has no
    DBOS checkpoint and re-executes on recovery — proving
    that incomplete steps are not skipped.

    This is the complement of
    ``test_workflow_recovers_after_server_restart``: that test
    proves completed steps replay from cache; this test proves
    incomplete steps re-run from scratch.

    Lifecycle:
    1. LLM call 1 returns tool call (checkpointed by DBOS)
    2. ``_call_tool`` @step starts, blocks via gate (never
       completes, no checkpoint written)
    3. Crash (tear down server + DBOS)
    4. Recovery: LLM call 1 replays from cache,
       ``_call_tool`` re-executes (no cache), LLM call 2
       runs fresh
    5. Assert tool ran twice (once pre-crash, once recovery)
    6. Assert LLM call 1 replayed from cache (2 total mock
       calls, not 3)
    """
    gate = ToolGate(should_block=True)
    tracking = setup_tool_tracking(gate=gate)

    try:
        with tracking.patch_ctx:
            ids = await run_server_1_crash_mid_tool(
                mock_llm,
                db_uri,
                tmp_path,
                gate,
            )
            # Do NOT release gate here. Releasing between
            # server 1 destroy and server 2 create lets the
            # pre-crash thread's finally block record
            # close_stream via server 2's DBOS, corrupting
            # the step sequence for recovery.
            gate.should_block = False

            result = await run_server_2_after_tool_crash(
                mock_llm,
                db_uri,
                tmp_path,
                ids.response_id,
            )

        await assert_recovery_output(
            result.terminal_body,
            "Recovered after tool crash",
        )
        assert_incomplete_step_reexecuted(tracking, mock_llm)

        await result.client.aclose()
        destroy_dbos()
    finally:
        # Release gate after BOTH DBOS instances are destroyed
        # so the pre-crash thread's close_stream fails cleanly
        # ("System database accessed before DBOS was launched").
        gate.release.set()


async def test_steered_messages_survive_crash(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """
    A message injected via ``try_deliver`` (which writes in
    its own SQLAlchemy transaction, independent of the DBOS
    workflow) survives a crash and is visible after recovery.

    ``try_deliver`` is the server-side half of the steering
    handshake. It inserts the steered message into
    ``conversation_items`` in its own session — NOT inside
    the DBOS workflow transaction. This means steered messages
    are durable even if the workflow crashes.

    Lifecycle:
    1. Start workflow, LLM call blocks (mid-execution)
    2. POST with ``previous_response_id`` triggers
       ``try_deliver`` which writes steered message to DB
       in its own transaction
    3. Crash (tear down server + DBOS)
    4. Recovery: LLM call re-executes, ``_sync_history``
       discovers the steered message in the conversation
       store, workflow completes
    5. Assert steered message and original user message both
       appear in conversation items
    """
    ids = await run_server_1_with_steering(
        mock_llm,
        db_uri,
        tmp_path,
    )
    result = await run_server_2_after_steering_crash(
        mock_llm,
        db_uri,
        tmp_path,
        ids.response_id,
    )

    await assert_recovery_output(
        result.terminal_body,
        "Recovery after steering",
    )
    await assert_steering_persisted(
        result.client,
        ids.conversation_id,
    )

    await result.client.aclose()
    destroy_dbos()
