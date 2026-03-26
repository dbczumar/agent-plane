"""Concurrency integration tests.

In-process tests use ControllableMockClient's blocking gates to
create deterministic race windows within one server. The
cross-server test launches real ``ap server`` subprocesses
sharing a database and uses a mock LLM HTTP server as the gate.

No ``time.sleep`` — synchronization is purely event-driven
via ``MockCall.call_event`` / ``MockCall.release()`` (in-process)
or ``/gate/pending`` / ``/gate/release`` (cross-server).
"""

from __future__ import annotations

import asyncio
import io
import socket
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import (
    create_test_agent,
    create_test_response,
)

pytestmark = pytest.mark.asyncio


# ── Steering Races ───────────────────────────────────────


async def test_steering_delivers_to_running_workflow(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message into an active workflow via try_deliver.

    Race window: workflow is blocked in the LLM call. The
    HTTP request checks prev_task.status (sees IN_PROGRESS),
    calls try_deliver() which atomically checks inbox_closed
    (False) and appends the steered message.

    Breakage this catches:
    - try_deliver fails to insert the message
    - Steering returns a different response ID (new task)
    - Steered message not persisted in conversation items
    - Workflow makes an extra LLM call on steered input
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="First response", block=True)

    first = await create_test_response(client, input_text="Hello")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]
    assert first.body["status"] == "queued"

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action: steer while workflow is blocked
    second = await create_test_response(
        client,
        input_text="Change direction",
        previous_response_id=first_id,
    )
    # Steering returns the SAME response (not a new task)
    assert second.body["id"] == first_id

    # Steered message is persisted in the conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts
    assert "Change direction" in user_texts

    # Release: workflow completes normally
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break

    assert resp.json()["status"] == "completed"
    # Steering does NOT trigger a second LLM call — steered
    # messages have response_id == task_id, so the workflow's
    # filter (ci.response_id != task_id) excludes them.
    assert mock_llm.call_count == 1


async def test_steering_preserves_position_order(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steered message gets the correct position between the
    original user message and the assistant response.

    Race window: try_deliver inserts at MAX(position)+1
    while the workflow is blocked before persisting its
    own assistant message. After release, the assistant
    message gets MAX(position)+1 which is after the
    steered message.

    Breakage this catches:
    - Position collision (steered and assistant at same pos)
    - Wrong ordering (assistant before steered message)
    - Steered message lost (not in items at all)
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="The answer", block=True)

    first = await create_test_response(
        client,
        input_text="Question 1",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action: steer while workflow is blocked
    await create_test_response(
        client,
        input_text="Steering message",
        previous_response_id=first_id,
    )

    # Release: workflow persists assistant at next position
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify position ordering: user, steered, assistant
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Question 1"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Steering message"
    assert items[2]["role"] == "assistant"
    assert items[2]["content"][0]["text"] == "The answer"


async def test_multiple_steering_messages_while_blocked(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two steering messages delivered while the workflow is
    blocked in the LLM call. Both must be persisted in
    correct position order.

    Race window: both try_deliver calls acquire the
    conversation lock sequentially (serialized by FOR
    UPDATE / SQLite locking), each computing MAX(position)+1.

    Breakage this catches:
    - Position collision between two steered messages
    - Second try_deliver fails (inbox closed prematurely)
    - Messages persisted out of order
    - Workflow makes extra LLM calls for steered messages
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(
        text="Final answer",
        block=True,
    )

    first = await create_test_response(
        client,
        input_text="Original question",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action 1: first steering message
    steer_1 = await create_test_response(
        client,
        input_text="Clarification A",
        previous_response_id=first_id,
    )
    assert steer_1.body["id"] == first_id

    # Concurrent action 2: second steering message
    steer_2 = await create_test_response(
        client,
        input_text="Clarification B",
        previous_response_id=first_id,
    )
    assert steer_2.body["id"] == first_id

    # Release: workflow completes
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify all 4 items in position order
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    assert len(items) == 4, f"Expected 4 items (user + 2 steered + assistant), got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Original question"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Clarification A"
    assert items[2]["role"] == "user"
    assert items[2]["content"][0]["text"] == "Clarification B"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "Final answer"

    # Only 1 LLM call — steered messages don't trigger loops
    assert mock_llm.call_count == 1


# ── Cancel Races ─────────────────────────────────────────


async def test_cancel_during_llm_call(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Cancel a response while the LLM call is in progress.

    Race window: workflow thread is blocked inside the mock
    LLM's create(). The cancel API sets the DBOS workflow's
    cancelled flag. When the mock is released, the workflow
    detects cancellation at the next checkpoint.

    Breakage this catches:
    - Cancel returns wrong status (not cancelled)
    - Cancel returns non-empty output (workflow produced
      results despite cancellation)
    - Workflow hangs after cancellation
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action: cancel while blocked in LLM
    cancel_resp = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"
    assert cancel_resp.json()["output"] == []

    # Release: workflow must terminate (not hang)
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{response_id}")
        if resp.json()["status"] in ("completed", "cancelled"):
            break
    assert resp.json()["status"] == "cancelled"


async def test_cancel_idempotent_while_blocked(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two cancel requests issued while the workflow is blocked.
    Both must return cancelled status — the second cancel
    must not fail or return a different status.

    Race window: the workflow is blocked in the LLM call.
    The first cancel sets DBOS cancelled flag. The second
    cancel hits the already-cancelled task.

    Breakage this catches:
    - Second cancel raises an error (not idempotent)
    - Second cancel returns wrong status
    - Task status flips between cancelled and something else
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action: cancel twice while still blocked
    resp1 = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )
    resp2 = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["status"] == "cancelled"
    assert resp2.json()["status"] == "cancelled"


async def test_steering_then_cancel_preserves_message(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message then cancel the workflow. The steered
    message must persist in the conversation because
    try_deliver commits in its own transaction, independent
    of the workflow's DBOS lifecycle.

    Race window: workflow is blocked in LLM. try_deliver
    writes the steered message to conversation_items (its
    own DB transaction). Then cancel stops the workflow.
    The message must survive the cancellation.

    Breakage this catches:
    - Steered message rolled back by cancellation
    - try_deliver transaction tied to workflow transaction
    - Conversation items lost on cancel
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action 1: steer while blocked
    steer = await create_test_response(
        client,
        input_text="Follow-up",
        previous_response_id=first_id,
    )
    assert steer.body["id"] == first_id

    # Concurrent action 2: cancel the workflow
    cancel_resp = await client.post(
        f"/v1/responses/{first_id}/cancel",
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Verify steered message persists despite cancellation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts, "Original user message missing after cancel"
    assert "Follow-up" in user_texts, (
        "Steered message lost after cancel — try_deliver's "
        "transaction must be independent of the workflow"
    )


# ── Cross-Server Steering ────────────────────────────────
#
# These tests launch real ``ap server`` subprocesses sharing
# a database, with a mock LLM HTTP server providing the
# synchronization gate.


def _find_free_port() -> int:
    """
    Bind to port 0 and return the OS-assigned port number.

    The socket is closed before returning, so the port may
    theoretically be reused — but in practice the window is
    negligible for sequential test setup.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_mock_llm_server(port: int) -> subprocess.Popen[bytes]:
    """
    Start the mock LLM server on the given port.

    :param port: TCP port for the mock server.
    :returns: The subprocess handle.
    """
    script = Path(__file__).parent / "mock_llm_server.py"
    return subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _start_ap_server(
    port: int,
    db_uri: str,
    artifact_dir: Path,
) -> subprocess.Popen[bytes]:
    """
    Start an ``ap server`` subprocess.

    :param port: TCP port for the server.
    :param db_uri: SQLAlchemy database URI (shared across servers).
    :param artifact_dir: Path for artifact storage.
    :returns: The subprocess handle.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-uri",
            db_uri,
            "--artifact-location",
            str(artifact_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


async def _poll_until_ready(
    url: str,
    timeout: float = 15.0,
) -> None:
    """
    Poll a URL until it returns HTTP 200, or raise on timeout.

    :param url: The URL to poll.
    :param timeout: Maximum seconds to wait.
    :raises TimeoutError: If the server doesn't respond in time.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError):
                pass
            await asyncio.sleep(0.3)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s")


def _build_mock_llm_bundle(
    name: str,
    mock_llm_port: int,
) -> bytes:
    """
    Build an agent bundle configured to use the mock LLM.

    The agent's ``llm.connection.base_url`` points at the mock
    LLM server so all LLM calls route there instead of OpenAI.

    :param name: Agent name (also used as model prefix).
    :param mock_llm_port: Port of the mock LLM server.
    :returns: A tar.gz bundle as bytes.
    """
    # Any: YAML config values are heterogeneous (str, int, dict, etc.)
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": name,
        "llm": {
            "model": f"openai/{name}",
            "connection": {
                "api_key": "fake-key",
                "base_url": f"http://127.0.0.1:{mock_llm_port}/v1",
            },
        },
    }
    config_bytes = yaml.dump(config).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


async def _assert_cross_server_steering(
    client_a: httpx.AsyncClient,
    client_b: httpx.AsyncClient,
    mock_client: httpx.AsyncClient,
    mock_llm_port: int,
) -> None:
    """
    Run the cross-server steering sequence and assert results.

    :param client_a: HTTP client for server A.
    :param client_b: HTTP client for server B.
    :param mock_client: HTTP client for the mock LLM server.
    :param mock_llm_port: Port of the mock LLM server.
    """
    # Deploy agent to server A (writes to shared DB)
    bundle = _build_mock_llm_bundle("test-agent", mock_llm_port)
    resp = await client_a.post(
        "/api/agents",
        files={
            "bundle": ("agent.tar.gz", bundle, "application/gzip"),
        },
    )
    assert resp.status_code == 201

    # Create response on server A — workflow calls mock LLM
    resp = await client_a.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Hello from server A",
            "background": True,
        },
    )
    assert resp.status_code == 200
    first_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    # Gate: wait for mock LLM to receive the request
    for _ in range(150):
        status = await mock_client.get("/gate/pending")
        if status.json()["pending"]:
            break
        await asyncio.sleep(0.1)
    assert status.json()["pending"], "Mock LLM never received request"

    # Concurrent action: steer from server B
    resp = await client_b.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Steered from server B",
            "previous_response_id": first_id,
            "background": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == first_id

    # Release: mock LLM responds, workflow completes
    await mock_client.post("/gate/release")

    # Wait for completion on server A
    for _ in range(150):
        resp = await client_a.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.1)
    assert resp.json()["status"] == "completed"

    # Verify steered message from server B is in the conversation
    items_resp = await client_a.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello from server A" in user_texts
    assert "Steered from server B" in user_texts

    # Only 1 LLM call — steering didn't trigger a second
    stats_resp = await mock_client.get("/stats")
    assert stats_resp.json()["request_count"] == 1


async def test_cross_server_steering_via_shared_db(
    tmp_path: Path,
) -> None:
    """
    Two real server processes sharing a database. A steering
    request from server B delivers to a workflow running on
    server A via the shared DB.

    Race window: server A's workflow is blocked in the mock
    LLM (HTTP gate). Server B receives the steering request,
    looks up the task in the shared DB (sees IN_PROGRESS),
    and calls try_deliver(). After gate release, server A's
    workflow completes with both messages in the conversation.

    Breakage this catches:
    - try_deliver fails across server processes
    - Steered message lost due to cross-process isolation
    - Duplicate LLM calls from cross-server steering
    - Task lookup fails on server B (stale or missing data)
    """
    mock_port = _find_free_port()
    port_a = _find_free_port()
    port_b = _find_free_port()
    db_uri = f"sqlite:///{tmp_path / 'shared.db'}"

    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(_start_mock_llm_server(mock_port))
        await _poll_until_ready(
            f"http://127.0.0.1:{mock_port}/stats",
        )

        procs.append(
            _start_ap_server(port_a, db_uri, tmp_path / "art_a"),
        )
        await _poll_until_ready(
            f"http://127.0.0.1:{port_a}/api/agents",
        )

        procs.append(
            _start_ap_server(port_b, db_uri, tmp_path / "art_b"),
        )
        await _poll_until_ready(
            f"http://127.0.0.1:{port_b}/api/agents",
        )

        async with (
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port_a}",
            ) as client_a,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port_b}",
            ) as client_b,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{mock_port}",
            ) as mock_client,
        ):
            await _assert_cross_server_steering(
                client_a,
                client_b,
                mock_client,
                mock_port,
            )
    finally:
        for proc in reversed(procs):
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=10)
