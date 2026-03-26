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

import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from agent_plane.runtime import init as init_runtime
from agent_plane.runtime.agent_cache import AgentCache
from agent_plane.runtime.durability import destroy_dbos, ensure_dbos
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
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

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


def _build_server(
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
    :param tmp_path: Temp directory for artifact and cache storage.
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

    # Patch the LLM client at the module level so the real
    # workflow uses our mock (same as the conftest fixture)
    import agent_plane.runtime.workflow as wf_mod

    wf_mod._get_llm_client = lambda: mock_llm  # type: ignore[assignment]

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


async def test_workflow_recovers_after_server_restart(
    mock_llm: ControllableMockClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """
    A workflow interrupted mid-LLM-call completes after a full
    server restart on the same database.

    Lifecycle:
    1. Build server instance 1 (stores + app + DBOS)
    2. Create agent + response, block workflow in LLM call
    3. Tear down the ENTIRE server (client, app, DBOS)
    4. Build server instance 2 on the same DB
    5. Poll the new server until the response completes
    6. Assert recovery-era text in output and conversation

    This is more realistic than only restarting DBOS — it
    rebuilds all stores and the FastAPI app from scratch,
    just like a real process restart.

    Breakage this catches:
    - Workflow silently disappears after crash (no recovery)
    - Recovered workflow returns stale/empty output
    - Task status stuck in in_progress forever
    - Conversation items not persisted by recovered workflow
    - Store reconstruction loses data
    """
    # ── Server instance 1 ─────────────────────────────
    client_1 = _build_server(db_uri, tmp_path, mock_llm)

    await create_test_agent(client_1)

    call_1 = mock_llm.add_call(
        text="This text should never appear",
        block=True,
    )
    created = await create_test_response(
        client_1,
        input_text="Durable request",
    )
    response_id = created.body["id"]
    conv_id = created.body["conversation"]["id"]
    assert created.body["status"] == "queued"

    # Gate: workflow has entered the LLM call
    call_1.call_event.wait(timeout=10)

    # ── Crash: tear down everything ───────────────────
    # Do NOT release the mock call — the thread stays blocked,
    # simulating a real crash where the process dies mid-LLM.
    # mock_llm.release_all() in fixture teardown frees the
    # orphaned thread so the test runner doesn't hang.
    await client_1.aclose()
    destroy_dbos()

    # ── Server instance 2 ─────────────────────────────
    # Queue a recovery mock response. The original call
    # (index 0) was consumed by server 1's workflow thread.
    # The recovered workflow re-executes the uncheckpointed
    # LLM step and gets this one (index 1).
    recovery_text = "Recovered after server restart"
    mock_llm.add_call(text=recovery_text)

    client_2 = _build_server(db_uri, tmp_path, mock_llm)

    # Poll the NEW server until the workflow completes
    terminal_body: dict | None = None
    for _ in range(200):
        resp = await client_2.get(
            f"/v1/responses/{response_id}",
        )
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            terminal_body = body
            break
        await asyncio.sleep(0.1)

    assert terminal_body is not None, f"Response {response_id} never reached terminal state"
    assert terminal_body["status"] == "completed", (
        f"Expected completed, got {terminal_body['status']}: {terminal_body.get('error')}"
    )

    # Verify output contains recovery-era text
    output = terminal_body["output"]
    assert len(output) >= 1, "No output items after recovery"
    assert output[0]["role"] == "assistant"
    assert output[0]["content"][0]["text"] == recovery_text

    # Verify conversation items persisted through the crash
    items_resp = await client_2.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    user_items = [i for i in items if i["role"] == "user"]
    assistant_items = [i for i in items if i["role"] == "assistant"]

    assert len(user_items) >= 1, "User message not persisted"
    assert user_items[0]["content"][0]["text"] == "Durable request"

    assert len(assistant_items) >= 1, "Assistant response not persisted after recovery"
    assert assistant_items[0]["content"][0]["text"] == recovery_text

    await client_2.aclose()
    destroy_dbos()
