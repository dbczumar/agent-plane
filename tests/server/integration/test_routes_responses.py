"""Integration tests for /v1/responses endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest
from httpx_sse import aconnect_sse

from agent_plane.entities import FunctionCallOutputData, MessageData
from agent_plane.server.routes.responses import _split_input_to_items
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import (
    build_agent_bundle,
    create_test_agent,
    create_test_response,
)

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
    assert result.body["conversation"] is not None
    # Verify output has a well-formed assistant message
    output = result.body["output"]
    assert len(output) >= 1
    msg = output[0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert len(msg["content"]) >= 1
    assert msg["content"][0]["type"] == "output_text"
    assert isinstance(msg["content"][0]["text"], str)
    assert len(msg["content"][0]["text"]) > 0


async def test_create_response_streaming(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """stream=True returns SSE events in the correct sequence."""
    await create_test_agent(client)
    # stream_tokens=True so the workflow emits text delta events
    mock_llm.add_call(text="Hello world", stream_tokens=True)

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
    # Output has a well-formed assistant message
    assert len(terminal_resp["output"]) >= 1
    msg = terminal_resp["output"][0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"

    # Text delta events were emitted during streaming
    text_deltas = [e for e in events if e[0] == "response.output_text.delta"]
    assert len(text_deltas) >= 1
    # Each delta has a non-empty string
    for _, delta_data in text_deltas:
        assert isinstance(delta_data["delta"], str)
        assert len(delta_data["delta"]) > 0
    # Concatenated deltas contain the full mock text
    # (tokenizer may add trailing whitespace)
    full_text = "".join(d[1]["delta"] for d in text_deltas)
    assert "Hello world" in full_text

    # output_item.done event has the assistant message
    item_done_events = [e for e in events if e[0] == "response.output_item.done"]
    assert len(item_done_events) >= 1
    done_item = item_done_events[0][1]["item"]
    assert done_item["type"] == "message"
    assert done_item["role"] == "assistant"

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
    # Output has a well-formed assistant message
    assert len(body["output"]) >= 1
    msg = body["output"][0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["content"][0]["type"] == "output_text"
    assert isinstance(msg["content"][0]["text"], str)
    assert len(msg["content"][0]["text"]) > 0


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
    original_output = created.body["output"]

    resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    # Output must be identical to the pre-cancel output
    assert body["output"] == original_output


async def test_cancel_active_response(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Cancelling an active response returns cancelled status with empty output."""
    await create_test_agent(client)

    # Block the LLM call so the task stays active
    call = mock_llm.add_call(block=True)
    created = await create_test_response(client)
    response_id = created.body["id"]
    assert created.body["status"] == "queued"

    # Wait for the LLM call to start so we know the workflow is active
    call.call_event.wait(timeout=5)

    resp = await client.post(f"/v1/responses/{response_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cancelled"
    # Cancelled responses have empty output
    assert body["output"] == []
    # The blocked LLM call was the only one — no second call happened
    assert mock_llm.call_count == 1


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
    """store=False is rejected with a clear error message."""
    await create_test_agent(client)
    resp = await client.post(
        "/v1/responses",
        json={"model": "test-agent", "input": "Hi", "store": False},
    )
    assert resp.status_code == 400
    error_msg = resp.json()["error"]["message"].lower()
    assert "store" in error_msg


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
    mock_llm: ControllableMockClient,
) -> None:
    """Per-request reasoning persists and propagates to the LLM call."""
    await create_test_agent(client)
    mock_llm.add_call(text="Reasoned response.")
    reasoning = {"effort": "high"}
    result = await create_test_response(client, reasoning=reasoning, background=False)
    assert result.body["reasoning"] == reasoning

    # Verify reasoning survives a GET round-trip
    resp = await client.get(f"/v1/responses/{result.body['id']}")
    assert resp.json()["reasoning"] == reasoning

    # Verify the LLM actually received the reasoning parameter
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_reasoning = received.get("reasoning")
    assert llm_reasoning is not None, "per-request reasoning must propagate to the LLM call"
    assert llm_reasoning["effort"] == "high"
    assert "summary" in llm_reasoning


async def test_agent_reasoning_effort_reaches_llm(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Agent spec reasoning_effort propagates to the LLM call,
    and per-request reasoning overrides it.

    1. Deploys agent with ``reasoning_effort: medium``
    2. Verifies the LLM receives ``reasoning.effort == "medium"``
    3. Sends a second request with ``reasoning: {effort: "low"}``
    4. Verifies the LLM receives ``reasoning.effort == "low"``
       (per-request overrides agent spec)
    """
    import io
    import tarfile

    import yaml

    # Deploy agent with reasoning_effort: medium in its spec
    config = {
        "spec_version": 1,
        "name": "reasoning-agent",
        "llm": {
            "model": "reasoning-agent",
            "reasoning_effort": "medium",
        },
    }
    config_bytes = yaml.dump(config).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    resp = await client.post(
        "/api/agents",
        files={
            "bundle": (
                "agent.tar.gz",
                buf.getvalue(),
                "application/gzip",
            )
        },
    )
    assert resp.status_code == 201

    # --- Call 1: spec-level reasoning_effort = medium ---
    mock_llm.add_call(text="Medium thought.")
    result = await create_test_response(
        client,
        model="reasoning-agent",
        background=False,
    )
    assert result.body["status"] == "completed"

    # LLM should receive effort="medium" from the agent spec
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_reasoning = received.get("reasoning")
    assert llm_reasoning is not None, (
        "reasoning_effort from agent spec must propagate to "
        "the LLM call as the 'reasoning' parameter"
    )
    assert llm_reasoning["effort"] == "medium"
    assert llm_reasoning["summary"] == "detailed"

    # --- Call 2: per-request reasoning overrides to low ---
    mock_llm.add_call(text="Quick thought.")
    result2 = await create_test_response(
        client,
        model="reasoning-agent",
        reasoning={"effort": "low"},
        background=False,
    )
    assert result2.body["status"] == "completed"

    # LLM should receive effort="low" — per-request overrides spec
    assert mock_llm.call_count == 2
    received2 = mock_llm.get_call(1).received_kwargs
    assert received2 is not None
    llm_reasoning2 = received2.get("reasoning")
    assert llm_reasoning2 is not None, (
        "per-request reasoning must override agent spec reasoning_effort"
    )
    assert llm_reasoning2["effort"] == "low"
    assert llm_reasoning2["summary"] == "detailed"


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
        background=False,
        stream=False,
    )
    assert second.status_code == 200
    assert second.body["previous_response_id"] == first_id
    assert second.body["conversation"]["id"] == conv_id

    # Verify the conversation has items from both turns
    items_resp = await client.get(f"/v1/conversations/{conv_id}/items")
    items = items_resp.json()["data"]
    # At minimum: user msg 1, assistant msg 1, user msg 2, assistant msg 2
    assert len(items) >= 4
    message_roles = [i.get("role") for i in items if i.get("type") == "message"]
    # Alternating user/assistant across both turns
    assert message_roles[0] == "user"
    assert message_roles[1] == "assistant"
    assert message_roles[2] == "user"
    assert message_roles[3] == "assistant"


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

    # Verify each lifecycle event has a well-formed response object
    for event_type in ("response.created", "response.queued", "response.in_progress"):
        evt = next(e[1] for e in events if e[0] == event_type)
        assert evt["type"] == event_type
        resp_obj = evt["response"]
        assert isinstance(resp_obj["id"], str)
        assert resp_obj["object"] == "response"


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


# ── Multimodal integration tests ────────────────────────────────────


async def test_multimodal_image_file_id_resolves_to_llm(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Upload an image via /v1/files, reference it via file_id in a
    response request, and verify the task completes with output.

    This is the full end-to-end path:
    upload → file_id validation → task creation → workflow →
    content resolver → prompt builder → LLM call.

    The content resolver resolves file_id to a data: URI before
    the LLM sees it. If resolution fails, the workflow errors out
    and the task status would be 'failed', not 'completed'.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="It is red.")

    # Upload a small PNG (1x1 red pixel).
    png_bytes = _make_tiny_png()
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("red.png", png_bytes, "image/png")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    # Send a response request with text + image file_id.
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "What color is this?"},
                {"type": "input_image", "file_id": file_id},
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Task completed — file_id was resolved and LLM responded.
    # If the content resolver failed, status would be 'failed'.
    assert body["status"] == "completed"
    assert len(body["output"]) > 0

    # Verify the LLM received the resolved image in its input.
    # The mock captures received_kwargs on each call.
    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])

    # Find the resolved image block in the LLM input.
    image_block = _find_content_block(llm_input, "input_image")
    assert image_block is not None, "LLM input must contain an input_image block"
    # file_id must be gone — replaced by image_url with a data: URI.
    assert "file_id" not in image_block, "file_id must be resolved before reaching the LLM"
    assert "image_url" in image_block, "Resolved image must have image_url field"
    # image_url must be a well-formed data: URI with the correct
    # media type and base64 payload. A wrong format (e.g. raw base64
    # without the data: prefix) causes HTTP 400 from providers.
    expected_b64 = base64.b64encode(png_bytes).decode("ascii")
    assert image_block["image_url"] == f"data:image/png;base64,{expected_b64}"


async def test_multimodal_file_id_resolves_to_llm(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Upload a PDF via /v1/files, reference it via file_id in a
    response request, and verify the task completes with the
    file_id resolved to inline base64 content.

    Tests the input_file content block path end-to-end.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="The document discusses testing.")

    # Upload a fake PDF.
    pdf_bytes = b"%PDF-1.4 fake pdf for testing"
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    # Send a response with text + file.
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "Summarize this document"},
                {
                    "type": "input_file",
                    "file_id": file_id,
                    "filename": "report.pdf",
                },
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert len(body["output"]) > 0

    # Verify the LLM received resolved file content.
    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])

    # Find the resolved file block in the LLM input.
    file_block = _find_content_block(llm_input, "input_file")
    assert file_block is not None, "LLM input must contain an input_file block"
    # file_id must be gone — replaced by file_data with a data: URI.
    assert "file_id" not in file_block, "file_id must be resolved before reaching the LLM"
    assert "file_data" in file_block, "Resolved file must have file_data field"
    # file_data must be a well-formed data: URI with the correct
    # media type and base64 payload. Raw base64 without the data:
    # prefix causes HTTP 400 from providers like OpenAI.
    expected_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    assert file_block["file_data"] == f"data:application/pdf;base64,{expected_b64}"


async def test_multimodal_mixed_image_and_file(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Upload both an image and a PDF, then send a request
    referencing both. Verifies the full pipeline handles
    multiple file types in a single request.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="The image shows a red pixel and the doc is a test.")

    # Upload image.
    png_bytes = _make_tiny_png()
    img_resp = await client.post(
        "/v1/files",
        files={"file": ("photo.png", png_bytes, "image/png")},
    )
    assert img_resp.status_code == 201
    img_id = img_resp.json()["id"]

    # Upload PDF.
    pdf_bytes = b"%PDF-1.4 test document content"
    pdf_resp = await client.post(
        "/v1/files",
        files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
    )
    assert pdf_resp.status_code == 201
    pdf_id = pdf_resp.json()["id"]

    # Send request with text + image + file.
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "Compare the image to the doc"},
                {"type": "input_image", "file_id": img_id},
                {
                    "type": "input_file",
                    "file_id": pdf_id,
                    "filename": "doc.pdf",
                },
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert len(body["output"]) > 0
    # Verify conversation was created.
    assert body["conversation"] is not None
    assert isinstance(body["conversation"]["id"], str)

    # Verify both file references were resolved to proper data: URIs.
    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])

    image_block = _find_content_block(llm_input, "input_image")
    assert image_block is not None, "LLM input must contain an input_image block"
    assert "file_id" not in image_block
    img_b64 = base64.b64encode(png_bytes).decode("ascii")
    assert image_block["image_url"] == f"data:image/png;base64,{img_b64}"

    file_block = _find_content_block(llm_input, "input_file")
    assert file_block is not None, "LLM input must contain an input_file block"
    assert "file_id" not in file_block
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    assert file_block["file_data"] == f"data:application/pdf;base64,{pdf_b64}"


async def test_multimodal_jpeg_image_uses_correct_media_type(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Upload a JPEG via /v1/files and verify the resolved data: URI
    uses image/jpeg, not a hardcoded image/png.

    Catches regressions where the media type is derived from the
    file metadata rather than hardcoded to a single image format.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="I see a photo.")

    # Minimal JPEG: SOI + APP0 (JFIF header) + EOI markers.
    jpeg_bytes = (
        b"\xff\xd8"  # SOI
        b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"  # EOI
    )
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_image", "file_id": file_id},
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    image_block = _find_content_block(llm_input, "input_image")
    assert image_block is not None
    expected_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    # Must use image/jpeg, not image/png or application/octet-stream.
    assert image_block["image_url"] == (f"data:image/jpeg;base64,{expected_b64}")


async def test_multimodal_plain_text_file_resolves(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Upload a plain text file and verify it resolves to a data: URI
    with text/plain media type.

    Ensures the pipeline works for non-PDF document types — the
    content resolver must use the actual content_type from the file
    store, not assume application/pdf.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="The file contains a greeting.")

    txt_bytes = b"Hello, world!"
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("notes.txt", txt_bytes, "text/plain")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "What does this file say?"},
                {
                    "type": "input_file",
                    "file_id": file_id,
                    "filename": "notes.txt",
                },
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    file_block = _find_content_block(llm_input, "input_file")
    assert file_block is not None
    expected_b64 = base64.b64encode(txt_bytes).decode("ascii")
    # Must use text/plain, not application/pdf.
    assert file_block["file_data"] == (f"data:text/plain;base64,{expected_b64}")


async def test_multimodal_file_without_content_type_uses_fallback(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Upload a file with no Content-Type header and verify the resolved
    data: URI falls back to application/octet-stream.

    Catches regressions where a missing content_type causes a crash
    or produces a malformed data: URI (e.g. "data:None;base64,...").
    """
    await create_test_agent(client)
    mock_llm.add_call(text="Binary data received.")

    raw_bytes = b"\x00\x01\x02\x03binary"
    upload_resp = await client.post(
        "/v1/files",
        # No content_type — server must handle gracefully.
        files={"file": ("blob.bin", raw_bytes)},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {
                    "type": "input_file",
                    "file_id": file_id,
                    "filename": "blob.bin",
                },
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    file_block = _find_content_block(llm_input, "input_file")
    assert file_block is not None
    expected_b64 = base64.b64encode(raw_bytes).decode("ascii")
    # Must fall back to application/octet-stream, not crash or
    # produce "data:None;base64,...".
    assert file_block["file_data"] == (f"data:application/octet-stream;base64,{expected_b64}")


async def test_multimodal_multi_turn_resolves_file_from_history(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Turn 1 uploads a file. Turn 2 is plain text referencing the same
    conversation. Verify the LLM receives the resolved file from
    Turn 1's history — not the raw file_id.

    Catches bugs where file_id resolution is skipped for historical
    messages, causing the LLM to see unresolved references.
    """
    await create_test_agent(client)
    # Turn 1 mock (file upload turn) and Turn 2 mock (follow-up).
    mock_llm.add_call(text="I see a red pixel.")
    turn2_call = mock_llm.add_call(text="It was a 1x1 PNG.")

    png_bytes = _make_tiny_png()
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("red.png", png_bytes, "image/png")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    # Turn 1: send image.
    turn1_resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "file_id": file_id},
            ],
            "background": False,
            "stream": False,
        },
    )
    assert turn1_resp.status_code == 200
    turn1_id = turn1_resp.json()["id"]

    # Turn 2: plain text follow-up in the same conversation.
    turn2_resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Can you describe the image dimensions?",
            "previous_response_id": turn1_id,
            "background": False,
            "stream": False,
        },
    )
    assert turn2_resp.status_code == 200
    assert turn2_resp.json()["status"] == "completed"

    # Verify Turn 2's LLM call contains the resolved image from
    # Turn 1's history — not the raw file_id.
    received = turn2_call.received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    input_str = json.dumps(llm_input)
    # file_id must NOT appear anywhere in the serialized input.
    assert file_id not in input_str, "file_id from Turn 1 must be resolved in Turn 2's history"
    # The resolved base64 content must appear (from Turn 1's image).
    expected_b64 = base64.b64encode(png_bytes).decode("ascii")
    assert expected_b64 in input_str, (
        "Resolved image content from Turn 1 must appear in Turn 2's input"
    )


async def test_multimodal_streaming_resolves_file(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    stream=True with a file_id: verify the streaming path resolves
    file references and completes successfully.

    Catches bugs where streaming handlers bypass content resolution
    or where resolution races with the SSE event emitter.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="The document is about testing.")

    pdf_bytes = b"%PDF-1.4 streaming test"
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("stream.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    events: list[tuple[str, dict[str, Any] | str]] = []
    async with aconnect_sse(
        client,
        "POST",
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "Summarize"},
                {
                    "type": "input_file",
                    "file_id": file_id,
                    "filename": "stream.pdf",
                },
            ],
            "stream": True,
        },
    ) as event_source:
        async for sse in event_source.aiter_sse():
            if sse.data == "[DONE]":
                events.append(("done", "[DONE]"))
            else:
                parsed = json.loads(sse.data)
                events.append((sse.event, parsed))

    # Must complete — not fail due to unresolved file_id.
    terminal = events[-2]
    assert terminal[0] == "response.completed"
    assert terminal[1]["response"]["status"] == "completed"
    assert events[-1] == ("done", "[DONE]")

    # Verify the LLM received the resolved file, not the raw file_id.
    # call_count == 1: one LLM turn for the non-tool mock response.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    file_block = _find_content_block(llm_input, "input_file")
    assert file_block is not None
    assert "file_id" not in file_block
    expected_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    assert file_block["file_data"] == (f"data:application/pdf;base64,{expected_b64}")


async def test_multimodal_background_resolves_file(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    background=True with a file_id: verify the background task path
    resolves file references and the task completes.

    Catches bugs where file resolution state is lost between
    queueing and execution, or where the background workflow
    thread doesn't have access to the file/artifact stores.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="Background file processed.")

    pdf_bytes = b"%PDF-1.4 background test"
    upload_resp = await client.post(
        "/v1/files",
        files={"file": ("bg.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["id"]

    events: list[tuple[str, dict[str, Any] | str]] = []
    async with aconnect_sse(
        client,
        "POST",
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {
                    "type": "input_file",
                    "file_id": file_id,
                    "filename": "bg.pdf",
                },
            ],
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
    # background=True adds response.queued between created and
    # in_progress.
    assert event_types[0] == "response.created"
    assert event_types[1] == "response.queued"
    assert event_types[2] == "response.in_progress"

    # Must complete — not fail due to unresolved file_id.
    terminal = events[-2]
    assert terminal[0] == "response.completed"
    assert terminal[1]["response"]["status"] == "completed"

    # Verify the LLM received the resolved file.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    file_block = _find_content_block(llm_input, "input_file")
    assert file_block is not None
    assert "file_id" not in file_block
    expected_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    assert file_block["file_data"] == (f"data:application/pdf;base64,{expected_b64}")


async def test_multimodal_two_images_in_one_request(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two images in a single request: verify both file_ids resolve
    to distinct data: URIs with correct content.

    Catches cache-collision bugs where two different files map to
    the same cached content, causing the LLM to see the same
    image twice.
    """
    await create_test_agent(client)
    mock_llm.add_call(text="Two different images.")

    png_bytes = _make_tiny_png()
    # Second image: a minimal JPEG (different content from PNG).
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

    img1_resp = await client.post(
        "/v1/files",
        files={"file": ("red.png", png_bytes, "image/png")},
    )
    assert img1_resp.status_code == 201
    img1_id = img1_resp.json()["id"]

    img2_resp = await client.post(
        "/v1/files",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert img2_resp.status_code == 201
    img2_id = img2_resp.json()["id"]

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": [
                {"type": "input_text", "text": "Compare these two images"},
                {"type": "input_image", "file_id": img1_id},
                {"type": "input_image", "file_id": img2_id},
            ],
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    # Verify the LLM received BOTH resolved images with distinct
    # content — not the same image duplicated.
    assert mock_llm.call_count == 1
    received = mock_llm.get_call(0).received_kwargs
    assert received is not None
    llm_input = received.get("input", [])
    input_str = json.dumps(llm_input)

    # Neither file_id should appear.
    assert img1_id not in input_str
    assert img2_id not in input_str

    # Both base64 payloads must appear — distinct content proves
    # no cache collision.
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    assert f"data:image/png;base64,{png_b64}" in input_str
    assert f"data:image/jpeg;base64,{jpeg_b64}" in input_str


def _find_content_block(
    llm_input: list[dict[str, Any]],
    block_type: str,
) -> dict[str, Any] | None:
    """
    Find a content block by type in the LLM input list.

    Searches through Responses API input items for a content block
    with the given type. Handles both top-level content blocks and
    blocks nested inside message items.

    :param llm_input: The ``input`` list passed to the LLM, e.g.
        ``[{"role": "user", "content": [...]}]``.
    :param block_type: The block type to find, e.g.
        ``"input_image"`` or ``"input_file"``.
    :returns: The first matching content block dict, or ``None``.
    """
    for item in llm_input:
        # Top-level content block (flat Responses API input).
        if item.get("type") == block_type:
            return item
        # Nested inside a message's content list.
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == block_type:
                    return block
    return None


def _make_tiny_png() -> bytes:
    """
    Generate a minimal valid 1x1 red PNG.

    :returns: PNG file bytes.
    """
    import struct
    import zlib

    # IHDR: 1x1, 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    # IDAT: single row, filter byte 0, then RGB (255, 0, 0)
    raw_row = b"\x00\xff\x00\x00"
    compressed = zlib.compress(raw_row)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)

    # IEND
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    # PNG signature + chunks
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


# ── PATCH /responses/{id} tests ────────────────────────────────────


async def test_patch_response_not_found(
    client: httpx.AsyncClient,
) -> None:
    """
    PATCH for a non-existent response returns 404.
    """
    resp = await client.patch(
        "/v1/responses/nonexistent",
        json={"tool_results": [{"call_id": "c1", "output": "x"}]},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "not_found"
    assert "not found" in body["error"]["message"].lower()


async def test_patch_response_call_id_not_found(
    client: httpx.AsyncClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    PATCH with a call_id that doesn't exist in pending_tool_calls
    returns 404 with a message naming the missing call_id.
    """
    await create_test_agent(client)
    result = await create_test_response(client, background=True)
    response_id = result.body["id"]

    resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {"call_id": "call_ghost", "output": "irrelevant"},
            ],
        },
    )
    assert resp.status_code == 404
    assert "call_ghost" in resp.json()["error"]["message"]


async def test_patch_response_completes_pending_tool_call(
    client: httpx.AsyncClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    PATCH with a valid call_id transitions the pending tool call
    to completed and returns the response.
    """
    await create_test_agent(client)
    result = await create_test_response(client, background=True)
    response_id = result.body["id"]

    # Directly insert a pending tool call row for this response
    # (simulates what the sub-agent park branch would do).
    task_store.create_pending_tool_call(
        call_id="call_test_1",
        root_task_id=response_id,
        task_id=response_id,
        tool_name="test_tool",
        arguments='{"key": "value"}',
    )

    resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {"call_id": "call_test_1", "output": "tool output"},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == response_id

    # Verify the pending tool call is now completed in the store
    rows = task_store.list_pending_tool_calls(task_id=response_id, status="completed")
    assert len(rows) == 1, "Expected exactly 1 completed row"
    assert rows[0].call_id == "call_test_1"
    assert rows[0].result == "tool output"


async def test_patch_response_idempotent(
    client: httpx.AsyncClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    Re-PATCHing the same call_id returns 200 (idempotent no-op).
    The stored result is NOT overwritten by the second PATCH.
    """
    await create_test_agent(client)
    result = await create_test_response(client, background=True)
    response_id = result.body["id"]

    task_store.create_pending_tool_call(
        call_id="call_idem",
        root_task_id=response_id,
        task_id=response_id,
        tool_name="test_tool",
        arguments="{}",
    )

    # First PATCH
    resp1 = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {"call_id": "call_idem", "output": "first"},
            ],
        },
    )
    assert resp1.status_code == 200

    # Second PATCH with different output — should still succeed
    # but the stored result remains "first" (first writer wins).
    resp2 = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {"call_id": "call_idem", "output": "second"},
            ],
        },
    )
    assert resp2.status_code == 200

    rows = task_store.list_pending_tool_calls(task_id=response_id, status="completed")
    assert len(rows) == 1, "Expected exactly 1 completed row"
    assert rows[0].result == "first", "First writer wins — stored result must not be overwritten"


# ── Multi-agent (spawn/collect) tests ──────────────────────────────


async def test_spawn_sub_agent_creates_child_task(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    Parent agent calls spawn_sub_agents, which creates a child
    task that runs independently and completes. The parent then
    receives the spawn result and produces a final response.

    Verifies the full multi-agent workflow:
    1. Parent LLM returns spawn_sub_agents tool call.
    2. SpawnTool creates a sub-agent task+workflow.
    3. Sub-agent's LLM call completes with text.
    4. Parent's next LLM call includes spawn output in context.
    5. Parent produces a final text response.
    """
    # Upload agent with a "researcher" sub-agent
    bundle = build_agent_bundle(
        name="orchestrator",
        sub_agents=[
            {"name": "researcher", "description": "Research helper"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # Mock call 1 (parent): spawn the researcher sub-agent
    spawn_args = json.dumps(
        {
            "agents": [
                {"name": "researcher", "input": "What is Python 3.14?"},
            ],
        }
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn_1",
                "name": "spawn_sub_agents",
                "arguments": spawn_args,
            },
        ],
    )

    # Mock call 2 (sub-agent researcher): complete with text
    mock_llm.add_call(text="Python 3.14 introduces JIT compilation.")

    # Mock call 3 (parent, after spawn result): final text —
    # triggers auto-collect since sub-agent wasn't collected.
    mock_llm.add_call(text="Based on research: Python 3.14 has JIT.")

    # Mock call 4 (parent, after auto-collect injects sub-agent
    # results): final text incorporating collected output.
    mock_llm.add_call(text="Based on research: Python 3.14 has JIT.")

    # Create the response (foreground blocking wait)
    result = await create_test_response(
        client,
        model="orchestrator",
        input_text="Research Python 3.14 features",
        background=False,
        stream=False,
    )

    assert result.status_code == 200, f"Expected 200, got {result.status_code}: {result.body}"
    assert result.body["status"] == "completed"
    assert result.body["model"] == "orchestrator"

    # The parent's output should contain the final text message with one of the
    # mock LLM's responses, proving data traversed the full pipeline. The parent
    # and sub-agent workflows run concurrently and share a FIFO mock queue, so
    # either mock text may end up as the parent's final output depending on
    # thread scheduling.
    expected_texts = {
        "Python 3.14 introduces JIT compilation.",
        "Based on research: Python 3.14 has JIT.",
    }
    output = result.body["output"]
    text_items = [item for item in output if item.get("type") == "message"]
    assert len(text_items) >= 1, f"Expected at least one message in output, got: {output}"
    final_msg = text_items[-1]
    assert final_msg["role"] == "assistant"
    actual_texts = {c.get("text") for c in final_msg["content"]}
    assert actual_texts & expected_texts, (
        f"Expected one of {expected_texts} in final message, got: {final_msg['content']}"
    )

    # Parent needs at least 2 LLM calls (spawn + final). The sub-agent
    # runs concurrently and may or may not have completed by this point,
    # so we assert >= 2 rather than >= 3 to avoid scheduling flakiness.
    assert mock_llm.call_count >= 2, (
        f"Expected >= 2 LLM calls (parent spawn + parent final), got {mock_llm.call_count}"
    )

    # Verify the sub-agent task was created with root_task_id
    parent_task_id = result.body["id"]
    all_tasks = await task_store.list_tasks()
    child_tasks = [t for t in all_tasks if t.root_task_id == parent_task_id]
    assert len(child_tasks) == 1, (
        f"Expected 1 child task with root_task_id={parent_task_id}, got {len(child_tasks)}"
    )


async def test_spawn_and_auto_collect_sub_agent(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    Parent spawns a sub-agent; auto-collect injects results before
    the parent completes.

    Verifies that ``wait_sync`` polling works end-to-end (this
    caught a ``dbos_sleep`` bug where ``wait_sync`` was called
    from a ``@step`` context).

    Flow:
    1. Parent LLM calls ``spawn_sub_agents``.
    2. Sub-agent LLM produces text.
    3. Parent LLM produces text (triggers auto-collect since
       there are uncollected sub-agents).
    4. Auto-collect waits for sub-agent, injects results.
    5. Parent LLM produces final text incorporating results.

    Calls 2–3 are consumed by the sub-agent and parent in
    non-deterministic order (the mock client serves them FIFO
    from a thread-safe queue). The test passes regardless of
    which agent consumes which call.
    """
    bundle = build_agent_bundle(
        name="collector",
        sub_agents=[
            {"name": "researcher", "description": "Research helper"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # Mock call 1 (parent): spawn researcher
    spawn_args = json.dumps(
        {"agents": [{"name": "researcher", "input": "What is Rust?"}]},
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn",
                "name": "spawn_sub_agents",
                "arguments": spawn_args,
            },
        ],
    )

    # Mock calls 2–3: consumed by sub-agent and parent in
    # non-deterministic order. Parent's text triggers auto-collect
    # (uncollected sub-agents). Either scheduling order is valid.
    mock_llm.add_call(text="Rust is a systems programming language.")
    mock_llm.add_call(text="Rust is a systems programming language.")

    # Mock call 4 (parent after auto-collect injects results):
    # final answer incorporating the sub-agent's findings.
    mock_llm.add_call(
        text="Rust is a systems language focused on safety.",
    )

    result = await create_test_response(
        client,
        model="collector",
        input_text="Tell me about Rust using your researcher",
        background=False,
        stream=False,
    )

    assert result.status_code == 200, f"Expected 200, got {result.status_code}: {result.body}"
    assert result.body["status"] == "completed"

    # Verify output has the final text from mock call 4,
    # proving the full spawn → auto-collect → final pipeline.
    output = result.body["output"]
    text_items = [item for item in output if item.get("type") == "message"]
    # At least one message output proves the parent completed.
    assert len(text_items) >= 1, f"Expected at least one message in output, got: {output}"
    actual_texts = set()
    for msg in text_items:
        for c in msg.get("content", []):
            if c.get("text"):
                actual_texts.add(c["text"])

    # The final text from mock call 4 must appear, proving the
    # parent saw the auto-collected sub-agent results and ran
    # one more LLM turn.
    assert "Rust is a systems language focused on safety." in actual_texts, (
        f"Expected final text in output. If missing, auto-collect "
        f"did not inject sub-agent results or the parent did not "
        f"get an additional LLM turn. Got: {actual_texts}"
    )

    # Verify spawn_sub_agents was called (appears in output as
    # a function_call item).
    spawn_calls = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == "spawn_sub_agents"
    ]
    # Exactly 1 spawn call proves the parent initiated the
    # sub-agent. 0 means spawn was never called; >1 means the
    # parent re-spawned unexpectedly.
    assert len(spawn_calls) == 1, (
        f"Expected exactly 1 spawn_sub_agents call, got "
        f"{len(spawn_calls)}: "
        f"{[i.get('name') for i in output if i.get('type') == 'function_call']}"
    )


async def test_spawn_recovery_across_client_tool_boundary(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Sub-agent spawned in one response is auto-collected in the
    next response after a client-tool round-trip.

    When the parent LLM calls both ``spawn_sub_agents``
    (server-side) and a client-side tool in one turn,
    ``_complete_for_client_tools`` completes the response.
    A new response starts when the client sends tool results.
    The new workflow must recover ``spawned_ids`` from
    conversation history so auto-collect runs for the
    sub-agent spawned in the previous response.

    Without ``_recover_spawn_state``, ``spawned_ids`` is empty
    in the new workflow. Auto-collect never runs. The sub-agent
    is orphaned.

    Breakage this catches:
    - ``spawned_ids`` not recovered from history → no
      auto-collect, output lacks collected results.
    - ``_recover_spawn_state`` misparses history items →
      same symptom.
    """
    bundle = build_agent_bundle(
        name="cross-boundary",
        sub_agents=[
            {"name": "helper", "description": "Background helper"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={
            "bundle": ("agent.tar.gz", bundle, "application/gzip"),
        },
    )
    assert resp.status_code == 201

    # Call 1 (parent): spawn helper AND call Read (client-side).
    # Server executes spawn, sees Read is client-side, calls
    # _complete_for_client_tools → response completes with the
    # unexecuted Read function_call in the output.
    spawn_args = json.dumps(
        {"agents": [{"name": "helper", "input": "Do background work"}]},
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn",
                "name": "spawn_sub_agents",
                "arguments": spawn_args,
            },
            {
                "call_id": "call_read",
                "name": "Read",
                "arguments": '{"file_path": "/tmp/test.txt"}',
            },
        ],
    )

    # Call 2 (sub-agent): block=True so we can guarantee the
    # sub-agent has consumed this call before response 2 starts.
    # Without this gate, response 2 might race and steal call 2.
    call_2 = mock_llm.add_call(
        text="Background work done: result XYZ",
        block=True,
    )

    # Call 3 (parent, response 2 first turn): text response.
    # Auto-collect detects uncollected sub-agent → polls until
    # complete → injects results → triggers call 4.
    mock_llm.add_call(text="Checking results...")

    # Call 4 (parent after auto-collect injects results): final
    # answer incorporating collected sub-agent output.
    mock_llm.add_call(
        text="Final: helper returned result XYZ",
    )

    read_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    }

    # Response 1: foreground (background=False). Blocks until
    # _complete_for_client_tools completes the response.
    # Top-level tasks with client-side tools do NOT create
    # pending_tool_calls rows — they complete immediately with
    # the function_call items in the output.
    r1 = await create_test_response(
        client,
        model="cross-boundary",
        input_text="Spawn helper and read /tmp/test.txt",
        background=False,
        tools=[read_tool],
    )
    # _complete_for_client_tools sets status to "completed".
    assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.body}"
    assert r1.body["status"] == "completed", (
        f"Expected completed, got {r1.body['status']}. Output: {r1.body.get('output')}"
    )

    # Verify the output contains the Read function_call.
    r1_output = r1.body["output"]
    read_calls = [
        item
        for item in r1_output
        if (item.get("type") == "function_call" and item.get("name") == "Read")
    ]
    # 1 Read call = the client-side tool returned by the LLM.
    # If 0, spawn consumed both tool calls or the function_call
    # items were not persisted in the output.
    assert len(read_calls) == 1, (
        f"Expected 1 Read function_call in output, got "
        f"{len(read_calls)}: "
        f"{[i.get('name') for i in r1_output if i.get('type') == 'function_call']}"
    )

    # Synchronization gate: wait for sub-agent to consume call 2,
    # then release it so it completes before response 2 starts.
    # This guarantees mock call ordering (call 3 goes to the
    # parent's second workflow, not the sub-agent).
    call_2.call_event.wait(timeout=10)
    call_2.release()

    # Response 2: send tool results as a new request.
    # This creates a NEW workflow that must recover spawned_ids
    # from conversation history so auto-collect runs.
    r1_id = r1.body["id"]
    r2_resp = await client.post(
        "/v1/responses",
        json={
            "model": "cross-boundary",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_read",
                    "output": "file contents: hello world",
                },
            ],
            "previous_response_id": r1_id,
            "background": False,
            "stream": False,
        },
    )
    r2 = r2_resp.json()

    # Response 2 completes — its workflow recovered spawned_ids,
    # ran auto-collect, and produced a final answer.
    assert r2_resp.status_code == 200, f"Expected 200, got {r2_resp.status_code}: {r2}"
    assert r2["status"] == "completed", (
        f"Expected completed, got {r2['status']}. Output: {r2.get('output')}"
    )

    # The final output must contain "result XYZ" — this proves
    # auto-collect ran in response 2 and injected the sub-agent's
    # output into the parent's history, which the LLM then
    # incorporated into its final answer. If missing,
    # _recover_spawn_state failed to reconstruct spawned_ids.
    text_items = [item for item in r2["output"] if item.get("type") == "message"]
    all_text = " ".join(c.get("text", "") for item in text_items for c in item.get("content", []))
    assert "result XYZ" in all_text, (
        "Auto-collect did not run in the second response. "
        "The sub-agent's output ('result XYZ') was not "
        "included in the parent's final answer. This means "
        "_recover_spawn_state failed to reconstruct "
        f"spawned_ids from history. Output: {all_text[:300]}"
    )


# ── Sub-agent parking and tunneling tests ──────────────


async def _poll_until_terminal(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until the response reaches a terminal state.

    :param client: The HTTP client.
    :param response_id: The response ID to poll.
    :returns: The terminal response body.
    """
    for _ in range(200):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        await asyncio.sleep(0.1)
    raise AssertionError(f"Response {response_id} never reached terminal state")


async def test_get_response_surfaces_pending_tool_calls(
    client: httpx.AsyncClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    GET /v1/responses/{id} includes action_required function_call
    items in the output when pending tool calls exist for the
    root task — with tool_name and arguments populated.

    This would have caught the bug where the polling client never
    saw tunneled tool calls because the GET response returned
    empty output for in-progress tasks.
    """
    await create_test_agent(client)
    result = await create_test_response(client, background=True)
    response_id = result.body["id"]

    # Directly insert a pending tool call (simulates what
    # _park_for_client_tools does).
    task_store.create_pending_tool_call(
        call_id="call_tunnel_1",
        root_task_id=response_id,
        task_id=response_id,
        tool_name="Read",
        arguments='{"file_path": "/tmp/test.txt"}',
    )

    # GET should surface the pending call with action_required.
    resp = await client.get(f"/v1/responses/{response_id}")
    body = resp.json()
    output = body.get("output", [])

    # Find action_required function_call items.
    action_items = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("status") == "action_required"
    ]
    # At least one action_required item must be present.
    # If empty, the GET endpoint didn't query pending_tool_calls
    # or didn't include them in the output.
    assert len(action_items) >= 1, (
        f"Expected action_required function_call in GET output. Got output: {output}"
    )
    fc = action_items[0]
    # Must have the tool name so the client knows what to execute.
    assert fc["name"] == "Read", (
        f"Expected tool name 'Read', got {fc.get('name')!r}. "
        "If wrong, tool_name wasn't populated in pending_tool_calls."
    )
    # Must have arguments so the client has the full call details.
    assert "file_path" in fc.get("arguments", ""), (
        "Expected arguments to contain 'file_path'. If missing, "
        "the arguments column wasn't stored at park time."
    )
    assert fc["call_id"] == "call_tunnel_1", (
        f"Expected call_id 'call_tunnel_1', got {fc.get('call_id')!r}"
    )


async def _poll_for_pending(
    client: httpx.AsyncClient,
    response_id: str,
    max_attempts: int = 100,
) -> list[dict[str, Any]]:
    """
    Poll until action_required function_calls appear.

    :param client: HTTP client.
    :param response_id: Root response ID.
    :param max_attempts: Max poll iterations.
    :returns: List of action_required items.
    """
    for _ in range(max_attempts):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        pending = [
            item
            for item in body.get("output", [])
            if item.get("type") == "function_call" and item.get("status") == "action_required"
        ]
        if pending:
            return pending
        if body["status"] in ("completed", "failed"):
            return []
        await asyncio.sleep(0.1)
    return []


async def test_park_patch_resume_end_to_end(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    Full park→PATCH→resume→collect flow: sub-agent parks for a
    client tool, client discovers it via GET, PATCHes the result,
    sub-agent resumes with the tool result and completes, parent
    auto-collects and produces a final response.

    This test previously deadlocked due to DBOS thread pool
    exhaustion from ``time.sleep`` polling loops. The fix
    (DBOS recv/send signaling) eliminated the deadlock.
    """
    bundle = build_agent_bundle(
        name="e2e-parent",
        sub_agents=[
            {"name": "reader", "description": "File reader"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={
            "bundle": ("agent.tar.gz", bundle, "application/gzip"),
        },
    )
    assert resp.status_code == 201

    # Mock call 1 (parent): spawn reader
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_sp",
                "name": "spawn_sub_agents",
                "arguments": json.dumps(
                    {
                        "agents": [
                            {"name": "reader", "input": "read file"},
                        ],
                    }
                ),
            },
        ],
    )

    # tool_calls_fn routes Read to the sub-agent (input contains
    # "read file") and text to the parent.
    def _maybe_read(
        kwargs: dict[str, Any],
    ) -> list[dict[str, str]] | None:
        input_str = json.dumps(kwargs.get("input", []))
        if "read file" in input_str and "function_call_output" not in input_str:
            return [
                {
                    "call_id": "call_rd",
                    "name": "Read",
                    "arguments": '{"file_path": "/tmp/f.txt"}',
                },
            ]
        return None

    # Calls 2-3: race between parent and sub-agent.
    mock_llm.add_call(text="Let me check...", tool_calls_fn=_maybe_read)
    mock_llm.add_call(text="Let me check...", tool_calls_fn=_maybe_read)
    # Call 4: sub-agent after tool result → final text
    mock_llm.add_call(text="File says: test content")
    # Call 5: parent after auto-collect → final answer
    mock_llm.add_call(text="The file contains: test content")

    # Client-side tool schema for Read.
    read_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    }

    result = await create_test_response(
        client,
        model="e2e-parent",
        input_text="Read /tmp/f.txt",
        background=True,
        tools=[read_tool],
    )
    response_id = result.body["id"]

    # Poll until the sub-agent parks and pending calls appear.
    pending = await _poll_for_pending(client, response_id)
    assert len(pending) >= 1, (
        "Sub-agent didn't park or pending calls not surfaced. "
        "If this hangs, DBOS thread pool exhaustion is back."
    )

    # PATCH the tool result — wakes the sub-agent via DBOS send.
    patch_resp = await client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {"call_id": "call_rd", "output": "test content"},
            ],
        },
    )
    assert patch_resp.status_code == 200

    # Wait for the full workflow to complete.
    final = await _poll_until_terminal(client, response_id)
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )
    # The output must contain "test content" — proves the tool
    # result traversed: PATCH → pending_tool_calls table →
    # DBOS send → sub-agent recv → LLM → collect → parent LLM.
    output = final["output"]
    text_items = [item for item in output if item.get("type") == "message"]
    all_text = " ".join(c.get("text", "") for item in text_items for c in item.get("content", []))
    assert "test content" in all_text, (
        f"Expected 'test content' in final output. Got: {all_text!r}"
    )


async def test_auto_collect_at_turn_end(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    When the parent spawns a sub-agent but the LLM produces a
    final text response without calling collect_sub_agents, the
    workflow auto-collects before completing the turn.

    This would have caught the bug where the parent completed
    early, leaving the sub-agent orphaned.
    """
    bundle = build_agent_bundle(
        name="auto-parent",
        sub_agents=[{"name": "worker", "description": "Worker"}],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # Mock call 1 (parent): spawn worker
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_sp",
                "name": "spawn_sub_agents",
                "arguments": json.dumps(
                    {
                        "agents": [
                            {"name": "worker", "input": "do work"},
                        ],
                    }
                ),
            },
        ],
    )
    # Mock call 2 (sub-agent): completes with text (no tools)
    mock_llm.add_call(text="Work done: result is 42")
    # Mock call 3 (parent): text WITHOUT collect — auto-collect
    # should trigger and re-run the LLM.
    mock_llm.add_call(text="Spawned the worker.")
    # Mock call 4 (parent, after auto-collect): final response
    # incorporating collected results.
    mock_llm.add_call(text="Worker result: 42")

    result = await create_test_response(
        client,
        model="auto-parent",
        input_text="Do work via worker",
        background=True,
    )
    response_id = result.body["id"]

    final = await _poll_until_terminal(client, response_id)
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )

    # The parent must have called the LLM at least 4 times:
    # spawn, sub-agent text, parent text (triggers auto-collect),
    # and parent final (after auto-collect injects results).
    # If only 3, auto-collect didn't trigger the extra iteration.
    assert mock_llm.call_count >= 4, (
        f"Expected >= 4 LLM calls (spawn + sub-agent + parent "
        f"text + post-auto-collect), got {mock_llm.call_count}. "
        f"If 3, auto-collect didn't re-run the LLM."
    )

    # The final output must contain the post-auto-collect text.
    output = final["output"]
    text_items = [item for item in output if item.get("type") == "message"]
    all_text = " ".join(c.get("text", "") for item in text_items for c in item.get("content", []))
    # "42" in the output proves the auto-collected sub-agent
    # result was visible to the final LLM call.
    assert "42" in all_text, (
        f"Expected '42' in final output (proves auto-collect "
        f"injected sub-agent results before final LLM call). "
        f"Got: {all_text!r}"
    )

    # Verify: no uncollected child tasks remain.
    tasks = await task_store.list_tasks()
    child_tasks = [t for t in tasks if t.root_task_id == response_id]
    # Every child must be in a terminal state.
    for child in child_tasks:
        assert child.status in ("completed", "failed"), (
            f"Child task {child.id} has status {child.status!r} — "
            f"auto-collect should have waited for it to finish."
        )


# ── Parallel same-name sub-agent isolation tests ─────


def _make_collect_router(
    spawn_call_id: str,
) -> Any:
    """
    Build a tool_calls_fn that emits collect_sub_agents when the
    LLM input contains the spawn function_call_output, and falls
    back to text otherwise (for sub-agent calls).

    :param spawn_call_id: The call_id of the spawn tool call to
        match, e.g. ``"call_spawn_multi"``.
    :returns: A callable suitable for ``MockCall.tool_calls_fn``.
    """

    # Any: kwargs from responses.create() are heterogeneous dicts.
    def _route(kwargs: dict[str, Any]) -> list[dict[str, str]] | None:
        for item in kwargs.get("input", []):
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and item.get("call_id") == spawn_call_id
            ):
                output = json.loads(item["output"])
                return [
                    {
                        "call_id": f"call_collect_{spawn_call_id}",
                        "name": "collect_sub_agents",
                        "arguments": json.dumps(
                            {
                                "response_ids": output["response_ids"],
                                "timeout": 30,
                            }
                        ),
                    }
                ]
        return None

    return _route


def _make_reader_router() -> Any:
    """
    Build a tool_calls_fn that returns a Read tool call with a
    distinct call_id for alpha/bravo sub-agents, based on which
    input string is present. Returns None for resumed calls
    (containing function_call_output) so the mock falls back
    to text.

    :returns: A callable suitable for ``MockCall.tool_calls_fn``.
    """

    # Any: kwargs from responses.create() are heterogeneous dicts.
    def _route(kwargs: dict[str, Any]) -> list[dict[str, str]] | None:
        input_str = json.dumps(kwargs.get("input", []))
        # Skip resumed calls — sub-agent already has tool result.
        if "function_call_output" in input_str:
            return None
        if "read file alpha" in input_str:
            return [
                {
                    "call_id": "call_read_alpha",
                    "name": "Read",
                    "arguments": '{"file_path": "/tmp/alpha.txt"}',
                }
            ]
        if "read file bravo" in input_str:
            return [
                {
                    "call_id": "call_read_bravo",
                    "name": "Read",
                    "arguments": '{"file_path": "/tmp/bravo.txt"}',
                }
            ]
        return None

    return _route


async def _patch_pending_tool_calls(
    client: httpx.AsyncClient,
    response_id: str,
    expected_outputs: dict[str, str],
) -> None:
    """
    Poll for pending tool calls and PATCH their results. Handles
    the case where sub-agents park at different times by polling
    in rounds until all expected call_ids have been patched.

    :param client: The HTTP test client.
    :param response_id: The root response ID to poll.
    :param expected_outputs: Mapping of call_id to tool output
        string, e.g. ``{"call_read_alpha": "alpha content"}``.
    """
    patched: set[str] = set()
    # 2 rounds: sub-agents may park at different times, so the
    # first poll may surface only one pending call.
    for _ in range(2):
        pending = await _poll_for_pending(client, response_id)
        for item in pending:
            call_id = item["call_id"]
            if call_id in patched:
                continue
            assert call_id in expected_outputs, (
                f"Unexpected call_id {call_id!r}. Expected one of {set(expected_outputs)}."
            )
            patch_resp = await client.patch(
                f"/v1/responses/{response_id}",
                json={
                    "tool_results": [
                        {
                            "call_id": call_id,
                            "output": expected_outputs[call_id],
                        },
                    ],
                },
            )
            assert patch_resp.status_code == 200, (
                f"PATCH for {call_id} failed with "
                f"{patch_resp.status_code}. Tool result delivery "
                "to the sub-agent may be broken."
            )
            patched.add(call_id)
        if patched == set(expected_outputs):
            return


async def _assert_child_task_isolation(
    task_store: SqlAlchemyTaskStore,
    parent_task_id: str,
    expected_count: int,
    expected_agent_name: str,
) -> None:
    """
    Verify that child tasks are isolated: distinct IDs, distinct
    conversations, separate from the parent, and conversation
    contents are disjoint.

    :param task_store: The task store to query.
    :param parent_task_id: The parent response/task ID.
    :param expected_count: How many child tasks to expect.
    :param expected_agent_name: The agent_name all children share.
    """
    all_tasks = await task_store.list_tasks()
    child_tasks = [t for t in all_tasks if t.root_task_id == parent_task_id]
    _assert_child_task_identity(
        all_tasks,
        child_tasks,
        parent_task_id,
        expected_count,
        expected_agent_name,
    )
    _assert_child_conversation_contents_disjoint(child_tasks)


def _assert_child_task_identity(
    all_tasks: list[Any],
    child_tasks: list[Any],
    parent_task_id: str,
    expected_count: int,
    expected_agent_name: str,
) -> None:
    """
    Verify child task counts, names, IDs, and conversation IDs.

    :param all_tasks: All tasks from the store.
    :param child_tasks: Tasks whose root_task_id matches parent.
    :param parent_task_id: The parent response/task ID.
    :param expected_count: How many child tasks to expect.
    :param expected_agent_name: The agent_name all children share.
    """
    assert len(child_tasks) == expected_count, (
        f"Expected {expected_count} child tasks with "
        f"root_task_id={parent_task_id}, got {len(child_tasks)}. "
        f"If fewer, a spawn was deduplicated by agent_name "
        f"(wrong — task_id is identity)."
    )

    child_names = {t.agent_name for t in child_tasks}
    assert child_names == {expected_agent_name}, (
        f"Expected all children named {expected_agent_name!r}, got {child_names}"
    )
    child_ids = {t.id for t in child_tasks}
    assert len(child_ids) == expected_count, (
        "Child task IDs must be distinct — each spawn creates a new task regardless of agent_name."
    )

    child_conv_ids = {t.conversation_id for t in child_tasks}
    assert len(child_conv_ids) == expected_count, (
        f"Expected {expected_count} distinct conversation IDs, "
        f"got {len(child_conv_ids)}. If fewer, sub-agents share "
        "a conversation, which would cause prompt pollution."
    )

    parent_task = next(t for t in all_tasks if t.id == parent_task_id)
    assert parent_task.conversation_id not in child_conv_ids, (
        "Parent conversation_id must differ from child "
        "conversation_ids — they are independent threads."
    )


def _assert_child_conversation_contents_disjoint(
    child_tasks: list[Any],
) -> None:
    """
    Verify that conversation items across child tasks are fully
    disjoint: no item ID appears in more than one conversation,
    and each conversation's items reference only its own task's
    response_id.

    This is the content-level isolation check — structural ID
    uniqueness alone doesn't prove that ``list_items`` actually
    returns the right items for each conversation.

    :param child_tasks: Child task objects with ``id`` and
        ``conversation_id`` attributes.
    """
    from agent_plane.runtime import get_conversation_store

    conv_store = get_conversation_store()

    # Collect item IDs and response_ids per child conversation.
    all_item_ids: list[set[str]] = []
    for child in child_tasks:
        page = conv_store.list_items(child.conversation_id)
        # Each sub-agent conversation must have at least the
        # initial user message appended by _spawn_one.
        assert len(page.data) >= 1, (
            f"Child {child.id} conversation {child.conversation_id} "
            f"is empty — _spawn_one should have appended the user "
            f"input as the first message."
        )
        item_ids = {item.id for item in page.data}
        all_item_ids.append(item_ids)

        # Every item in this conversation must carry this child's
        # task ID as its response_id. If an item from a sibling
        # leaked in, its response_id would be wrong.
        for item in page.data:
            assert item.response_id == child.id, (
                f"Item {item.id} in child {child.id}'s conversation "
                f"has response_id={item.response_id!r}, expected "
                f"{child.id!r}. An item from a sibling sub-agent "
                f"leaked into this conversation."
            )

    # Item IDs across child conversations must be disjoint.
    # Hardcoded for 2 children — callers always pass exactly 2.
    assert all_item_ids[0].isdisjoint(all_item_ids[1]), (
        f"Item IDs overlap between child conversations: "
        f"{all_item_ids[0] & all_item_ids[1]}. "
        f"An item leaked from one sub-agent's conversation into "
        f"another's."
    )


async def test_parallel_same_name_subagents_have_isolated_conversations(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    Spawn two instances of the same sub-agent spec ("researcher")
    in a single spawn_sub_agents call. Verify each gets a distinct
    task ID, distinct conversation ID, and the parent conversation
    is separate from both.

    Catches regressions where agent_name is used as a unique key
    (it shouldn't be — task_id is the identity), or where
    conversation creation reuses an existing sub-agent conversation.
    """
    bundle = build_agent_bundle(
        name="multi-spawn",
        sub_agents=[
            {"name": "researcher", "description": "Research helper"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # Mock call 1 (parent): spawn two researchers in one call.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn_multi",
                "name": "spawn_sub_agents",
                "arguments": json.dumps(
                    {
                        "agents": [
                            {"name": "researcher", "input": "Tell me about Python"},
                            {"name": "researcher", "input": "Tell me about Rust"},
                        ],
                    }
                ),
            }
        ],
    )

    route = _make_collect_router("call_spawn_multi")
    # Calls 2-4: race between 2 sub-agents and parent collect.
    mock_llm.add_call(text="Python is a dynamic language.", tool_calls_fn=route)
    mock_llm.add_call(text="Rust is a systems language.", tool_calls_fn=route)
    mock_llm.add_call(text="Sub-agent fallback text.", tool_calls_fn=route)
    # Call 5 (parent, after collect): final text.
    mock_llm.add_call(text="Research complete: Python and Rust.")

    result = await create_test_response(
        client,
        model="multi-spawn",
        input_text="Research Python and Rust",
        background=False,
        stream=False,
    )

    assert result.status_code == 200, f"Expected 200, got {result.status_code}: {result.body}"
    assert result.body["status"] == "completed", (
        f"Workflow did not complete: {result.body.get('error')}"
    )

    await _assert_child_task_isolation(
        task_store,
        result.body["id"],
        expected_count=2,
        expected_agent_name="researcher",
    )


async def test_parallel_subagents_park_and_patch_independently(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
) -> None:
    """
    Spawn two sub-agents that both park on client-side tool calls.
    Verify that PATCHing a tool result for one sub-agent wakes
    only that sub-agent, and both complete independently.

    Tests the tool-call routing invariant: pending_tool_calls are
    keyed by call_id and routed by task_id, so two sub-agents
    parking simultaneously never cross-contaminate.

    A failure means PATCH delivered a tool result to the wrong
    sub-agent (e.g. routing by agent_name instead of task_id).
    """
    bundle = build_agent_bundle(
        name="dual-park-parent",
        sub_agents=[{"name": "reader", "description": "File reader"}],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # Mock call 1 (parent): spawn two reader instances.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_sp_dual",
                "name": "spawn_sub_agents",
                "arguments": json.dumps(
                    {
                        "agents": [
                            {"name": "reader", "input": "read file alpha"},
                            {"name": "reader", "input": "read file bravo"},
                        ],
                    }
                ),
            }
        ],
    )

    # Calls 2-4: sub-agents park on Read tool calls.
    reader_route = _make_reader_router()
    mock_llm.add_call(text="parent placeholder", tool_calls_fn=reader_route)
    mock_llm.add_call(text="parent placeholder", tool_calls_fn=reader_route)
    mock_llm.add_call(text="parent placeholder", tool_calls_fn=reader_route)
    # After PATCH: sub-agents resume and produce final text.
    mock_llm.add_call(text="Alpha file says: alpha content")
    mock_llm.add_call(text="Bravo file says: bravo content")
    # Parent after auto-collect: final answer.
    mock_llm.add_call(text="Files read: alpha and bravo content.")

    # Client-side tool schema for Read.
    read_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }

    result = await create_test_response(
        client,
        model="dual-park-parent",
        input_text="Read alpha and bravo files",
        background=True,
        tools=[read_tool],
    )
    response_id = result.body["id"]

    # Poll and PATCH both pending tool calls.
    await _patch_pending_tool_calls(
        client,
        response_id,
        {
            "call_read_alpha": "alpha content",
            "call_read_bravo": "bravo content",
        },
    )

    # Wait for the full workflow to complete.
    final = await _poll_until_terminal(client, response_id)
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )

    # Verify both children completed and have isolated conversations.
    await _assert_child_task_isolation(
        task_store,
        response_id,
        expected_count=2,
        expected_agent_name="reader",
    )
    # Verify both children are in completed state (not just
    # created) — proves each received its own tool result.
    all_tasks = await task_store.list_tasks()
    child_tasks = [t for t in all_tasks if t.root_task_id == response_id]
    for child in child_tasks:
        assert child.status == "completed", (
            f"Child {child.id} has status {child.status!r}, "
            "expected 'completed'. Tool result may have been "
            "delivered to the wrong sub-agent."
        )
