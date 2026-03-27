"""Integration tests for /v1/responses endpoints."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx_sse import aconnect_sse

from agent_plane.entities import FunctionCallOutputData, MessageData
from agent_plane.server.routes.responses import _split_input_to_items
from tests.server.conftest import ControllableMockClient
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
    # background=False so the task completes before we GET it
    created = await create_test_response(client, background=False, stream=False)
    response_id = created.body["id"]

    resp = await client.get(f"/v1/responses/{response_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == response_id
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "test-agent"
    assert isinstance(body["created_at"], int)
    assert body["conversation"] is not None
    assert isinstance(body["conversation"]["id"], str)
    assert len(body["output"]) >= 1


async def test_get_response_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/responses/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["error"]["message"], str)


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
    assert isinstance(resp.json()["error"]["message"], str)


async def test_cancel_completed_response(client: httpx.AsyncClient) -> None:
    """Cancelling an already-completed response is a no-op — status stays completed."""
    await create_test_agent(client)
    # background=False so the task completes before we cancel it
    created = await create_test_response(client, background=False, stream=False)
    response_id = created.body["id"]

    resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    # Output preserved since task was already completed
    assert len(body["output"]) >= 1


async def test_cancel_active_response(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Cancelling an active response returns cancelled status with empty output."""
    await create_test_agent(client)

    # Block the LLM call so the task stays active
    mock_llm.add_call(block=True)
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
    assert isinstance(resp.json()["error"]["message"], str)


async def test_create_response_unknown_model(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/responses",
        json={"model": "nonexistent-model", "input": "Hi"},
    )
    assert resp.status_code == 404
    assert isinstance(resp.json()["error"]["message"], str)


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

    # First turn — background=False so it completes before Turn 2 starts,
    # avoiding position races with the background workflow thread.
    first = await create_test_response(
        client,
        input_text="Turn 1",
        background=False,
        stream=False,
    )
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
    assert isinstance(resp.json()["error"]["message"], str)


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
    assert "does not belong" in resp.json()["error"]["message"]


async def test_steering_try_deliver(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When previous_response_id points to an active task, the server
    delivers the message to the running agent's inbox and returns
    the existing in-progress response.
    """
    await create_test_agent(client)

    # Block the LLM call so the first task stays active
    mock_llm.add_call(block=True)
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

    # Turn 1 — background=False so it completes before Turn 2 starts
    first = await create_test_response(
        client,
        input_text="Turn 1",
        background=False,
        stream=False,
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Turn 2 — also foreground to complete before fork attempt
    second = await create_test_response(
        client,
        input_text="Turn 2",
        previous_response_id=first_id,
        background=False,
        stream=False,
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
    assert "fork" in fork_resp.json()["error"]["message"].lower()


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


# ── _split_input_to_items unit tests ─────────────────────


def test_split_input_plain_text_produces_user_message() -> None:
    """
    Plain text input (normalized to ``input_text`` blocks) produces
    a single user message item with the text content.
    """
    content = [{"type": "input_text", "text": "Hello"}]
    items = _split_input_to_items(content, response_id="resp_1")

    # Exactly 1 item: user message.
    assert len(items) == 1, f"Plain text input should produce 1 item; got {len(items)}"
    assert items[0].type == "message"
    assert isinstance(items[0].data, MessageData)
    assert items[0].data.role == "user"
    assert items[0].data.content == content
    assert items[0].response_id == "resp_1"


def test_split_input_function_call_output_produces_fco_item() -> None:
    """
    When input contains only ``function_call_output`` blocks, they
    are persisted as separate ``FunctionCallOutputData`` items — NOT
    wrapped in a user message.

    This is the bug that caused the 400 error: without splitting,
    the LLM received a user message containing function_call_output
    dicts instead of proper tool result items.
    """
    content = [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "72 degrees, sunny",
        },
    ]
    items = _split_input_to_items(content, response_id="resp_2")

    # Exactly 1 item: function_call_output (no user message).
    assert len(items) == 1, (
        f"Expected 1 function_call_output item; got {len(items)}. "
        f"If 0, the output was dropped. If more, a spurious user "
        f"message was created for the empty message_blocks list."
    )
    assert items[0].type == "function_call_output"
    assert isinstance(items[0].data, FunctionCallOutputData)
    assert items[0].data.call_id == "call_1"
    assert items[0].data.output == "72 degrees, sunny"
    assert items[0].response_id == "resp_2"


def test_split_input_mixed_separates_text_and_fco() -> None:
    """
    When input contains both ``input_text`` and
    ``function_call_output`` blocks, text goes into a user
    message and tool outputs become separate items. The user
    message appears first (before function_call_output items).
    """
    content = [
        {"type": "input_text", "text": "Here are the results:"},
        {
            "type": "function_call_output",
            "call_id": "call_a",
            "output": "result A",
        },
        {
            "type": "function_call_output",
            "call_id": "call_b",
            "output": "result B",
        },
    ]
    items = _split_input_to_items(content, response_id="resp_3")

    # 3 items: 1 user message + 2 function_call_outputs.
    assert len(items) == 3, f"Expected 3 items (1 message + 2 fco); got {len(items)}"

    # First item: user message with only the text block.
    assert items[0].type == "message"
    assert isinstance(items[0].data, MessageData)
    assert items[0].data.content == [
        {"type": "input_text", "text": "Here are the results:"},
    ], "User message should contain only the input_text block"

    # Items 1 and 2: function_call_output items in order.
    assert items[1].type == "function_call_output"
    assert isinstance(items[1].data, FunctionCallOutputData)
    assert items[1].data.call_id == "call_a"
    assert items[1].data.output == "result A"

    assert items[2].type == "function_call_output"
    assert isinstance(items[2].data, FunctionCallOutputData)
    assert items[2].data.call_id == "call_b"
    assert items[2].data.output == "result B"


def test_split_input_multiple_fco_no_text() -> None:
    """
    Multiple ``function_call_output`` blocks with no text produces
    only function_call_output items — no empty user message.
    """
    content = [
        {
            "type": "function_call_output",
            "call_id": "call_x",
            "output": "result X",
        },
        {
            "type": "function_call_output",
            "call_id": "call_y",
            "output": "result Y",
        },
    ]
    items = _split_input_to_items(content, response_id="resp_4")

    # 2 items, no user message (message_blocks is empty).
    assert len(items) == 2, (
        f"Expected 2 function_call_output items; got {len(items)}. "
        f"If 3, an empty user message was created."
    )
    assert all(i.type == "function_call_output" for i in items)


# ── file_id validation tests ────────────────────────────────────────


async def test_create_response_rejects_nonexistent_file_id(
    client: httpx.AsyncClient,
) -> None:
    """
    Posting a request with a file_id that does not exist in the file
    store must return 400 immediately — not a deferred workflow error.
    """
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_image", "file_id": "file_nonexistent"},
            ],
        },
    )
    # 400 with INVALID_INPUT — file_id validated at request time.
    assert resp.status_code == 400
    body = resp.json()
    assert "file_nonexistent" in body["error"]["message"]


async def test_create_response_accepts_valid_file_id(
    client: httpx.AsyncClient,
) -> None:
    """
    Posting a request with a file_id that exists in the file store
    must succeed (not rejected by validation).
    """
    await create_test_agent(client)

    # Upload a file first via the files API.
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    # File upload returns 201 Created per the OpenResponses spec.
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    # Now reference that file_id in a response request.
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_file", "file_id": file_id},
            ],
            "background": True,
            "stream": False,
        },
    )
    # Should succeed — file_id is valid.
    assert resp.status_code == 200


async def test_create_response_no_file_id_skips_validation(
    client: httpx.AsyncClient,
) -> None:
    """
    Requests without file_id references must skip validation
    entirely — no error even if file store has no files.
    """
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Hello, no file references here",
            "background": True,
            "stream": False,
        },
    )
    assert resp.status_code == 200
