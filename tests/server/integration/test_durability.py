"""Durability integration tests — crash recovery via DBOS.

Each test simulates a server crash by destroying the DBOS singleton
while a workflow is mid-execution, then reinitializes DBOS on the
same database. DBOS.launch() recovers pending workflows
automatically, replaying from the last checkpoint.

Synchronization uses the same ControllableMockClient gates as the
concurrency tests — no ``time.sleep``.

IMPORTANT: All tests in this file must use the
``pinned_dbos_version`` fixture (autouse). DBOS only recovers
workflows whose ``application_version`` matches the current
instance. Without a pinned version, the restarted DBOS computes a
different hash and silently skips recovery.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from agent_plane.runtime.durability import destroy_dbos, ensure_dbos
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio

# Fixed version string used for both the initial DBOS init
# (via task_store fixture) and the post-crash reinit. Without
# this, DBOS assigns different auto-computed hashes and the
# restarted instance refuses to recover "foreign" workflows.
_DBOS_VERSION = "test-durability-v1"


@pytest.fixture(autouse=True)
def pinned_dbos_version() -> None:
    """
    Patch ``ensure_dbos`` so every call uses a fixed
    ``application_version``. Applied before the ``task_store``
    fixture (which calls ``ensure_dbos`` during store init)
    and remains active for the explicit ``ensure_dbos`` call
    inside the test body.
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
        :param application_version: Ignored — overridden
            with ``_DBOS_VERSION``.
        """
        original(uri, application_version=_DBOS_VERSION)

    with patch(
        "agent_plane.runtime.durability.ensure_dbos",
        side_effect=_pinned,
    ):
        # Also patch the import used by the task store
        with patch(
            "agent_plane.stores.task_store.sqlalchemy_store.ensure_dbos",
            side_effect=_pinned,
        ):
            yield


async def test_workflow_recovers_after_crash(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    db_uri: str,
) -> None:
    """
    A workflow interrupted mid-LLM-call completes after DBOS
    restart on the same database.

    Lifecycle:
    1. Create agent + response (queues workflow)
    2. Block workflow inside the LLM call
    3. Destroy DBOS (simulates server crash)
    4. Queue a fresh mock LLM response for the recovered run
    5. Reinitialize DBOS on the same DB (triggers recovery)
    6. Poll until the response reaches terminal state
    7. Assert the response completed with the recovery-era
       mock text, proving the workflow re-executed the
       uncheckpointed LLM step.

    Breakage this catches:
    - Workflow silently disappears after crash (no recovery)
    - Recovered workflow returns stale/empty output
    - Task status stuck in in_progress forever
    - Conversation items not persisted by recovered workflow
    """
    await create_test_agent(client)

    # Phase 1: start workflow, block inside LLM call
    call_1 = mock_llm.add_call(
        text="This text should never appear", block=True,
    )
    created = await create_test_response(
        client, input_text="Durable request",
    )
    response_id = created.body["id"]
    conv_id = created.body["conversation"]["id"]
    assert created.body["status"] == "queued"

    # Gate: workflow has entered the LLM call
    call_1.call_event.wait(timeout=10)

    # Phase 2: simulate crash — kill DBOS while workflow is
    # blocked inside the LLM step (step not yet checkpointed).
    # Do NOT release the mock call — the thread stays blocked,
    # simulating a real crash where the process dies mid-LLM.
    # release_all() in fixture teardown prevents the orphaned
    # thread from hanging the test runner.
    destroy_dbos()

    # Phase 3: prepare for recovery — queue a new mock response
    # for the re-executed LLM step. After DBOS recovery, the
    # workflow re-runs from the last checkpoint. Since the LLM
    # step was in-flight (not checkpointed), it will be called
    # again with this new mock response.
    recovery_text = "Recovered after server restart"
    mock_llm.add_call(text=recovery_text)

    # Phase 4: restart DBOS — this scans the system database for
    # pending workflows and re-enqueues them automatically.
    # The pinned_dbos_version fixture ensures the same
    # application_version so DBOS recognizes the pending workflow.
    ensure_dbos(db_uri, application_version=_DBOS_VERSION)

    # Phase 5: poll until the workflow completes (or times out)
    terminal_body: dict | None = None
    for _ in range(200):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            terminal_body = body
            break
        # Yield to the event loop so DBOS recovery threads can
        # make progress (no time.sleep — just async yielding).
        await asyncio.sleep(0.1)

    assert terminal_body is not None, (
        f"Response {response_id} never reached terminal state"
    )
    assert terminal_body["status"] == "completed", (
        f"Expected completed, got {terminal_body['status']}: "
        f"{terminal_body.get('error')}"
    )

    # Phase 6: verify output contains the recovery-era text,
    # proving the workflow re-executed the LLM step after crash
    output = terminal_body["output"]
    assert len(output) >= 1, "No output items after recovery"
    assistant_msg = output[0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0]["text"] == recovery_text

    # Phase 7: verify conversation items were persisted —
    # the recovered workflow must persist both the user input
    # (from Phase 1) and the assistant response.
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    user_items = [i for i in items if i["role"] == "user"]
    assistant_items = [
        i for i in items if i["role"] == "assistant"
    ]

    assert len(user_items) >= 1, "User message not persisted"
    assert (
        user_items[0]["content"][0]["text"] == "Durable request"
    )

    assert len(assistant_items) >= 1, (
        "Assistant response not persisted after recovery"
    )
    assert (
        assistant_items[0]["content"][0]["text"] == recovery_text
    )
