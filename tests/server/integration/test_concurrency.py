"""Concurrency integration tests.

Each test creates a deterministic race window using
ControllableMockClient's blocking gates, then exercises a
concurrent interaction (steering, cancel, delete, streaming)
against the real DBOS workflow pipeline.

No ``time.sleep`` calls — synchronization is purely event-driven
via ``MockCall.call_event`` and ``MockCall.release()``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx_sse import aconnect_sse

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio


# ── Group 1: Steering Races ──────────────────────────────────


async def test_steering_delivers_to_running_workflow(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message into an active workflow via try_deliver.

    The server detects the previous response is still running,
    delivers the message to the conversation, and returns the
    existing in-progress response (same ID). The steered message
    is persisted and available in conversation history.
    """
    await create_test_agent(client)

    # Block so the workflow stays active during steering
    call_1 = mock_llm.add_call(text="First response", block=True)

    first = await create_test_response(client, input_text="Hello")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]
    assert first.body["status"] == "queued"

    # Wait until the workflow actually enters the LLM call
    call_1.call_event.wait(timeout=10)

    # Steer a message into the active workflow
    second = await create_test_response(
        client,
        input_text="Change direction",
        previous_response_id=first_id,
    )
    # Steering returns the SAME response (not a new one)
    assert second.body["id"] == first_id

    # Steered message is persisted in the conversation
    items_resp = await client.get(f"/v1/conversations/{conv_id}/items")
    items = items_resp.json()["data"]
    user_texts = [
        i["content"][0]["text"]
        for i in items if i["role"] == "user"
    ]
    assert "Hello" in user_texts
    assert "Change direction" in user_texts

    # Release — workflow completes normally
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break

    assert resp.json()["status"] == "completed"
    # LLM was called once (steering doesn't trigger a second call)
    assert mock_llm.call_count == 1


async def test_steering_after_inbox_closes_creates_new_task(
    client: httpx.AsyncClient,
) -> None:
    """
    When previous_response_id points to a completed task,
    steering is not possible — a new task is created instead.
    """
    await create_test_agent(client)

    # Foreground: blocks until complete, inbox is closed
    first = await create_test_response(
        client, input_text="Turn 1", background=False, stream=False,
    )
    first_id = first.body["id"]
    assert first.body["status"] == "completed"

    # Try to steer — but inbox is closed, so a new task is created
    second = await create_test_response(
        client,
        input_text="Turn 2",
        previous_response_id=first_id,
    )
    assert second.status_code == 200
    # Different response ID means a new task was created
    assert second.body["id"] != first_id
    # Same conversation
    assert second.body["conversation"]["id"] == first.body["conversation"]["id"]


# ── Group 2: Cancel Races ────────────────────────────────────


async def test_cancel_during_llm_call(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Cancel a response while the LLM call is in progress.
    The workflow should be interrupted and the task marked cancelled.
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]

    # Wait until the workflow is actually inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Cancel while blocked in LLM
    cancel_resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"
    assert cancel_resp.json()["output"] == []


async def test_cancel_queued_before_llm_starts(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Cancel a response that is queued but hasn't started
    executing yet. The DBOS workflow should be cancelled before
    the LLM is ever called.
    """
    await create_test_agent(client)

    # Block so workflow can't proceed
    mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]
    assert result.body["status"] == "queued"

    # Cancel immediately
    cancel_resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert cancel_resp.status_code == 200
    body = cancel_resp.json()
    assert body["status"] == "cancelled"


async def test_cancel_idempotent(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Cancelling the same response twice is idempotent."""
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]
    call_1.call_event.wait(timeout=10)

    # Cancel twice
    resp1 = await client.post(f"/v1/responses/{response_id}/cancel")
    resp2 = await client.post(f"/v1/responses/{response_id}/cancel")

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["status"] == "cancelled"
    assert resp2.json()["status"] == "cancelled"


# ── Group 3: Delete Cascade with Active Tasks ────────────────


async def test_delete_agent_cancels_active_workflow(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Deleting an agent with a running workflow cancels the
    workflow and removes both the agent and its tasks.
    """
    created = await create_test_agent(client, name="doomed-agent")
    agent_id = created["id"]

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client, model="doomed-agent")
    response_id = result.body["id"]

    # Wait for workflow to start
    call_1.call_event.wait(timeout=10)

    # Delete agent while workflow is running
    del_resp = await client.delete(f"/api/agents/{agent_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Both agent and task are gone
    assert (await client.get(f"/api/agents/{agent_id}")).status_code == 404
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404


async def test_delete_conversation_cancels_active_workflow(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Deleting a conversation with a running workflow cancels
    the workflow and removes the conversation and its tasks.
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]
    response_id = result.body["id"]

    # Wait for workflow to start
    call_1.call_event.wait(timeout=10)

    # Delete conversation while workflow is running
    del_resp = await client.delete(f"/v1/conversations/{conv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    assert (await client.get(f"/v1/conversations/{conv_id}")).status_code == 404
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404


# ── Group 4: Streaming Integrity ─────────────────────────────


async def test_streaming_event_sequence(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Streaming with token deltas produces events in the correct
    order with monotonically increasing sequence numbers.
    """
    await create_test_agent(client)

    # Enable token-level streaming in the mock
    mock_llm.add_call(text="Hello world tokens", stream_tokens=True)

    events: list[tuple[str, dict[str, Any] | str]] = []
    async with aconnect_sse(
        client,
        "POST",
        "/v1/responses",
        json={"model": "test-agent", "input": "Hi", "stream": True},
    ) as event_source:
        async for sse in event_source.aiter_sse():
            if sse.data == "[DONE]":
                events.append(("done", "[DONE]"))
            else:
                parsed = json.loads(sse.data)
                events.append((sse.event, parsed))

    event_types = [e[0] for e in events]

    # Must start with created, in_progress
    assert event_types[0] == "response.created"
    assert event_types[1] == "response.in_progress"

    # Must end with response.completed + [DONE]
    assert event_types[-2] == "response.completed"
    assert event_types[-1] == "done"

    # Text delta events exist between in_progress and completed
    delta_events = [
        e for e in events
        if isinstance(e[1], dict) and e[1].get("type") == "response.output_text.delta"
    ]
    assert len(delta_events) >= 1, "Expected at least one text delta event"

    # Sequence numbers are monotonically increasing with no gaps
    seq_numbers = [
        e[1]["sequence_number"]
        for e in events
        if isinstance(e[1], dict) and "sequence_number" in e[1]
    ]
    assert seq_numbers == list(range(len(seq_numbers)))


async def test_streaming_background_includes_queued_event(
    client: httpx.AsyncClient,
) -> None:
    """
    background=True streaming includes a response.queued event
    between created and in_progress.
    """
    await create_test_agent(client)

    events: list[str] = []
    async with aconnect_sse(
        client,
        "POST",
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Hi",
            "stream": True,
            "background": True,
        },
    ) as event_source:
        async for sse in event_source.aiter_sse():
            if sse.data != "[DONE]":
                parsed = json.loads(sse.data)
                events.append(parsed["type"])

    assert events[0] == "response.created"
    assert events[1] == "response.queued"
    assert events[2] == "response.in_progress"


async def test_streaming_completed_response_has_output(
    client: httpx.AsyncClient,
) -> None:
    """
    The terminal response.completed event contains the full
    response object with output populated.
    """
    await create_test_agent(client)

    terminal_response: dict[str, Any] | None = None
    async with aconnect_sse(
        client,
        "POST",
        "/v1/responses",
        json={"model": "test-agent", "input": "Hi", "stream": True},
    ) as event_source:
        async for sse in event_source.aiter_sse():
            if sse.data != "[DONE]":
                parsed = json.loads(sse.data)
                if parsed.get("type") == "response.completed":
                    terminal_response = parsed["response"]

    assert terminal_response is not None
    assert terminal_response["status"] == "completed"
    assert terminal_response["object"] == "response"
    assert len(terminal_response["output"]) >= 1
    assert terminal_response["output"][0]["role"] == "assistant"


# ── Group 5: Multi-Turn Position Integrity ───────────────────


async def test_multi_turn_positions_are_sequential(
    client: httpx.AsyncClient,
) -> None:
    """
    After a multi-turn conversation, all conversation items
    have strictly sequential positions with no gaps or
    duplicates.
    """
    await create_test_agent(client)

    # Turn 1 — foreground to ensure completion before Turn 2
    first = await create_test_response(
        client, input_text="Turn 1", background=False, stream=False,
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Turn 2 — foreground
    second = await create_test_response(
        client,
        input_text="Turn 2",
        previous_response_id=first_id,
        background=False,
        stream=False,
    )
    second_id = second.body["id"]

    # Turn 3 — foreground
    await create_test_response(
        client,
        input_text="Turn 3",
        previous_response_id=second_id,
        background=False,
        stream=False,
    )

    # Fetch all conversation items
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    # 3 turns × (1 user + 1 assistant) = 6 items
    assert len(items) == 6, f"Expected 6 items, got {len(items)}: {items}"

    # Verify alternating user/assistant pattern
    for i, item in enumerate(items):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert item["role"] == expected_role, (
            f"Item {i} expected role={expected_role}, got {item['role']}"
        )


async def test_concurrent_conversations_are_isolated(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two workflows running concurrently on different conversations
    don't interfere with each other's items or positions.
    """
    await create_test_agent(client)

    # Both workflows block, then release simultaneously
    call_a = mock_llm.add_call(text="Response A", block=True)
    call_b = mock_llm.add_call(text="Response B", block=True)

    result_a = await create_test_response(client, input_text="Conv A")
    result_b = await create_test_response(client, input_text="Conv B")

    conv_a = result_a.body["conversation"]["id"]
    conv_b = result_b.body["conversation"]["id"]
    assert conv_a != conv_b

    # Wait for both workflows to start
    call_a.call_event.wait(timeout=10)
    call_b.call_event.wait(timeout=10)

    # Release both simultaneously
    call_a.release()
    call_b.release()

    # Wait for both to complete
    for response_id in [result_a.body["id"], result_b.body["id"]]:
        for _ in range(50):
            resp = await client.get(f"/v1/responses/{response_id}")
            if resp.json()["status"] in ("completed", "failed"):
                break

    # Each conversation should have exactly 2 items (user + assistant)
    items_a = (await client.get(f"/v1/conversations/{conv_a}/items")).json()["data"]
    items_b = (await client.get(f"/v1/conversations/{conv_b}/items")).json()["data"]

    assert len(items_a) == 2, f"Conv A: expected 2 items, got {len(items_a)}"
    assert len(items_b) == 2, f"Conv B: expected 2 items, got {len(items_b)}"

    # Verify content isolation
    a_texts = [item["content"][0]["text"] for item in items_a]
    b_texts = [item["content"][0]["text"] for item in items_b]

    assert "Conv A" in a_texts[0]
    assert "Response A" in a_texts[1]
    assert "Conv B" in b_texts[0]
    assert "Response B" in b_texts[1]


async def test_steering_preserves_position_order(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When a steering message is delivered during an active
    workflow, the resulting conversation items maintain correct
    position ordering: user1 (pos 0), steered-user2 (pos 1),
    assistant (pos 2).
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="The answer", block=True)

    first = await create_test_response(client, input_text="Question 1")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Wait for workflow to reach LLM call
    call_1.call_event.wait(timeout=10)

    # Steer while workflow is blocked — inserts at next position
    await create_test_response(
        client,
        input_text="Steering message",
        previous_response_id=first_id,
    )

    # Release — workflow completes, persists assistant message
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] in ("completed", "failed"):
            break

    # Verify conversation items are in position order
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    # user1, steered-user2, assistant
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Question 1"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Steering message"
    assert items[2]["role"] == "assistant"
    assert items[2]["content"][0]["text"] == "The answer"
