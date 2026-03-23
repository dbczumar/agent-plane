"""Integration tests for /v1/responses endpoints."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx_sse import aconnect_sse

from tests.server.conftest import IntegrationTaskStore
from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio


async def test_create_response_background(client: httpx.AsyncClient) -> None:
    """background=True, stream=False returns immediately with queued status."""
    await create_test_agent(client)
    result = await create_test_response(client, background=True, stream=False)
    assert result.status_code == 200
    assert result.body["object"] == "response"
    assert result.body["status"] == "queued"
    assert result.body["model"] == "test-agent"
    assert isinstance(result.body["id"], str)
    assert isinstance(result.body["created_at"], int)
    assert result.body["conversation"] is not None
    assert result.body["output"] == []


async def test_create_response_foreground(client: httpx.AsyncClient) -> None:
    """background=False, stream=False blocks until completion and returns output."""
    await create_test_agent(client)
    result = await create_test_response(client, background=False, stream=False)
    assert result.status_code == 200
    assert result.body["status"] == "completed"
    assert isinstance(result.body["completed_at"], int)
    assert len(result.body["output"]) > 0
    assert result.body["conversation"] is not None


async def test_create_response_streaming(client: httpx.AsyncClient) -> None:
    """stream=True returns SSE events in the correct sequence."""
    await create_test_agent(client)

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

    # Verify event sequence
    assert events[0][0] == "response.created"
    assert events[0][1]["type"] == "response.created"

    assert events[1][0] == "response.in_progress"
    assert events[1][1]["type"] == "response.in_progress"

    # At least one stream event between in_progress and terminal
    assert len(events) >= 4

    # Terminal event is response.completed with full response object
    terminal = events[-2]
    assert terminal[0] == "response.completed"
    terminal_resp = terminal[1]["response"]
    assert terminal_resp["status"] == "completed"
    assert terminal_resp["object"] == "response"
    assert isinstance(terminal_resp["id"], str)
    assert len(terminal_resp["output"]) >= 1

    # Last event is [DONE]
    assert events[-1] == ("done", "[DONE]")

    # Sequence numbers are monotonically increasing
    seq_numbers = [
        e[1]["sequence_number"]
        for e in events
        if isinstance(e[1], dict) and "sequence_number" in e[1]
    ]
    assert seq_numbers == sorted(seq_numbers)
    assert len(set(seq_numbers)) == len(seq_numbers)


async def test_get_response(client: httpx.AsyncClient) -> None:
    """GET /responses/{id} returns the full response object with all fields."""
    await create_test_agent(client)
    created = await create_test_response(client)
    response_id = created.body["id"]

    resp = await client.get(f"/v1/responses/{response_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == response_id
    assert body["object"] == "response"
    # After start(), the stub marks as completed
    assert body["status"] == "completed"
    assert body["model"] == "test-agent"
    assert isinstance(body["created_at"], int)
    assert body["conversation"] is not None
    assert isinstance(body["conversation"]["id"], str)
    assert len(body["output"]) >= 1


async def test_get_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/responses/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


async def test_delete_response(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    created = await create_test_response(client)
    response_id = created.body["id"]

    del_resp = await client.delete(f"/v1/responses/{response_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == response_id
    assert body["object"] == "response.deleted"
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/v1/responses/{response_id}")
    assert get_resp.status_code == 404


async def test_delete_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/v1/responses/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


async def test_cancel_completed_response(client: httpx.AsyncClient) -> None:
    """Cancelling an already-completed response is a no-op — status stays completed."""
    await create_test_agent(client)
    created = await create_test_response(client)
    response_id = created.body["id"]

    resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    # Output preserved since task was already completed
    assert len(body["output"]) >= 1


async def test_cancel_active_response(
    client: httpx.AsyncClient,
    task_store: IntegrationTaskStore,
) -> None:
    """Cancelling an active response returns cancelled status with empty output."""
    await create_test_agent(client)

    task_store.defer_all_completions = True
    created = await create_test_response(client)
    response_id = created.body["id"]
    assert created.body["status"] == "queued"

    resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cancelled"
    # Cancelled responses have empty output
    assert body["output"] == []


async def test_cancel_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/responses/nonexistent/cancel")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


async def test_create_response_unknown_model(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/responses",
        json={"model": "nonexistent-model", "input": "Hi"},
    )
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


async def test_create_response_store_false(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={"model": "test-agent", "input": "Hi", "store": False},
    )
    assert resp.status_code == 400


async def test_create_response_with_instructions(
    client: httpx.AsyncClient,
) -> None:
    """Instructions are returned on creation and survive a GET round-trip."""
    await create_test_agent(client)
    result = await create_test_response(client, instructions="Be concise")
    assert result.body["instructions"] == "Be concise"

    # Verify instructions survive a GET round-trip
    resp = await client.get(f"/v1/responses/{result.body['id']}")
    assert resp.json()["instructions"] == "Be concise"


async def test_create_response_with_reasoning(
    client: httpx.AsyncClient,
) -> None:
    """Reasoning config is returned on creation and survives a GET round-trip."""
    await create_test_agent(client)
    reasoning = {"effort": "high"}
    result = await create_test_response(client, reasoning=reasoning)
    assert result.body["reasoning"] == reasoning

    # Verify reasoning survives a GET round-trip
    resp = await client.get(f"/v1/responses/{result.body['id']}")
    assert resp.json()["reasoning"] == reasoning


async def test_create_response_with_previous_response_id(
    client: httpx.AsyncClient,
) -> None:
    """Multi-turn: second response references the first via previous_response_id."""
    await create_test_agent(client)

    # First turn
    first = await create_test_response(client, input_text="Turn 1")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Second turn referencing the first
    second = await create_test_response(
        client,
        input_text="Turn 2",
        previous_response_id=first_id,
    )
    assert second.status_code == 200
    assert second.body["previous_response_id"] == first_id
    # Should be in the same conversation
    assert second.body["conversation"]["id"] == conv_id


async def test_create_response_invalid_previous_response_id(
    client: httpx.AsyncClient,
) -> None:
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Hi",
            "previous_response_id": "nonexistent",
        },
    )
    assert resp.status_code == 400


async def test_create_response_conversation_without_previous(
    client: httpx.AsyncClient,
) -> None:
    """conversation provided without previous_response_id is invalid."""
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Hi",
            "conversation": {"id": "conv_123"},
        },
    )
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)


async def test_create_response_conversation_mismatch(
    client: httpx.AsyncClient,
) -> None:
    """previous_response_id from a different conversation than the one specified returns 400."""
    await create_test_agent(client)

    # Create two separate conversations
    first = await create_test_response(client, input_text="Conv A")
    first_id = first.body["id"]

    second = await create_test_response(client, input_text="Conv B")
    second_conv_id = second.body["conversation"]["id"]

    # Try to use first's response_id with second's conversation
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Mismatch",
            "previous_response_id": first_id,
            "conversation": {"id": second_conv_id},
        },
    )
    assert resp.status_code == 400
    assert "does not belong" in resp.json()["detail"]


async def test_steering_try_deliver(
    client: httpx.AsyncClient,
    task_store: IntegrationTaskStore,
) -> None:
    """
    When previous_response_id points to an active task, the server
    delivers the message to the running agent's inbox and returns
    the existing in-progress response.
    """
    await create_test_agent(client)

    # Keep the first task active (not auto-completed)
    task_store.defer_all_completions = True
    first = await create_test_response(client, input_text="Turn 1")
    first_id = first.body["id"]
    assert first.body["status"] == "queued"

    # Second request targets the active task — should deliver and
    # return the existing response rather than creating a new one.
    second = await create_test_response(
        client,
        input_text="Steer this",
        previous_response_id=first_id,
    )
    assert second.status_code == 200
    # Returns the SAME response (steering, not a new task)
    assert second.body["id"] == first_id
    assert second.body["status"] == "queued"


async def test_background_streaming_queued_event(
    client: httpx.AsyncClient,
) -> None:
    """background=True streaming emits response.queued between created and in_progress."""
    await create_test_agent(client)

    events: list[tuple[str, dict[str, Any] | str]] = []
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
            if sse.data == "[DONE]":
                events.append(("done", "[DONE]"))
            else:
                parsed = json.loads(sse.data)
                events.append((sse.event, parsed))

    event_types = [e[0] for e in events]
    assert event_types[0] == "response.created"
    # background=True adds response.queued before in_progress
    assert event_types[1] == "response.queued"
    assert event_types[2] == "response.in_progress"


async def test_fork_detection(client: httpx.AsyncClient) -> None:
    """
    previous_response_id that isn't the latest response in the
    conversation (with conversation explicitly provided) returns 400.
    """
    await create_test_agent(client)

    # Turn 1
    first = await create_test_response(client, input_text="Turn 1")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Turn 2
    second = await create_test_response(
        client,
        input_text="Turn 2",
        previous_response_id=first_id,
    )
    assert second.body["conversation"]["id"] == conv_id

    # Turn 3 tries to fork: points to first (not second/latest)
    # with the conversation explicitly specified
    fork_resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Fork attempt",
            "previous_response_id": first_id,
            "conversation": {"id": conv_id},
        },
    )
    assert fork_resp.status_code == 400
    assert "fork" in fork_resp.json()["detail"].lower()


async def test_create_response_list_input(client: httpx.AsyncClient) -> None:
    """input accepts a list of content blocks, not just a string."""
    await create_test_agent(client)
    result = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [{"type": "input_text", "text": "Hello from list"}],
            "background": True,
        },
    )
    assert result.status_code == 200
    body = result.json()
    assert body["object"] == "response"

    # Verify the input was stored — check conversation items
    conv_id = body["conversation"]["id"]
    items_resp = await client.get(f"/v1/conversations/{conv_id}/items")
    items = items_resp.json()["data"]
    user_msg = items[0]
    assert user_msg["content"][0]["type"] == "input_text"
    assert user_msg["content"][0]["text"] == "Hello from list"


async def test_response_output_shape(client: httpx.AsyncClient) -> None:
    """Completed response has correct top-level fields and structured output."""
    await create_test_agent(client)
    result = await create_test_response(client, background=False, stream=False)
    body = result.body
    assert body["status"] == "completed"
    assert body["object"] == "response"
    assert isinstance(body["completed_at"], int)
    assert isinstance(body["conversation"]["id"], str)

    # Output structure: single assistant message with text content
    output = body["output"]
    assert len(output) == 1
    msg = output[0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "output_text"
    assert isinstance(msg["content"][0]["text"], str)
    assert len(msg["content"][0]["text"]) > 0
