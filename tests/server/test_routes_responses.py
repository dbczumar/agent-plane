"""Integration tests for /v1/responses endpoints."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx_sse import aconnect_sse

from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio


async def test_create_response_background(client: httpx.AsyncClient) -> None:
    """background=True, stream=False returns immediately with queued status."""
    await create_test_agent(client)
    status, body = await create_test_response(
        client, background=True, stream=False
    )
    assert status == 200
    assert body["object"] == "response"
    assert body["status"] == "queued"
    assert body["model"] == "test-agent"
    assert isinstance(body["id"], str)
    assert isinstance(body["created_at"], int)
    assert body["conversation"] is not None
    assert body["output"] == []


async def test_create_response_foreground(client: httpx.AsyncClient) -> None:
    """background=False, stream=False blocks until completion."""
    await create_test_agent(client)
    status, body = await create_test_response(
        client, background=False, stream=False
    )
    assert status == 200
    assert body["status"] == "completed"
    assert len(body["output"]) > 0


async def test_create_response_streaming(client: httpx.AsyncClient) -> None:
    """stream=True returns SSE events in the correct sequence."""
    await create_test_agent(client)

    events: list[tuple[str, Any]] = []
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

    # Terminal event is response.completed
    terminal = events[-2]
    assert terminal[0] == "response.completed"
    assert terminal[1]["response"]["status"] == "completed"

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
    await create_test_agent(client)
    _, created = await create_test_response(client)
    response_id = created["id"]

    resp = await client.get(f"/v1/responses/{response_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == response_id
    # After start(), the stub marks as completed
    assert body["status"] == "completed"


async def test_get_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/responses/nonexistent")
    assert resp.status_code == 404


async def test_delete_response(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    _, created = await create_test_response(client)
    response_id = created["id"]

    del_resp = await client.delete(f"/v1/responses/{response_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == response_id
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/v1/responses/{response_id}")
    assert get_resp.status_code == 404


async def test_delete_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/v1/responses/nonexistent")
    assert resp.status_code == 404


async def test_cancel_completed_response(client: httpx.AsyncClient) -> None:
    """Cancelling an already-completed response is a no-op."""
    await create_test_agent(client)
    _, created = await create_test_response(client)
    response_id = created["id"]

    resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    # Already completed — cancel is a no-op
    assert body["status"] == "completed"


async def test_cancel_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/responses/nonexistent/cancel")
    assert resp.status_code == 404


async def test_create_response_unknown_model(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/responses",
        json={"model": "nonexistent-model", "input": "Hi"},
    )
    assert resp.status_code == 404


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
    await create_test_agent(client)
    _, body = await create_test_response(
        client, instructions="Be concise"
    )
    assert body["instructions"] == "Be concise"


async def test_create_response_with_previous_response_id(
    client: httpx.AsyncClient,
) -> None:
    """Multi-turn: second response references the first via previous_response_id."""
    await create_test_agent(client)

    # First turn
    _, first = await create_test_response(client, input_text="Turn 1")
    first_id = first["id"]
    conv_id = first["conversation"]["id"]

    # Second turn referencing the first
    status, second = await create_test_response(
        client,
        input_text="Turn 2",
        previous_response_id=first_id,
    )
    assert status == 200
    assert second["previous_response_id"] == first_id
    # Should be in the same conversation
    assert second["conversation"]["id"] == conv_id


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
