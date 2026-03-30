"""Integration tests for /v1/responses endpoints."""

from __future__ import annotations

import base64
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
