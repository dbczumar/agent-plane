"""Integration tests for /v1/conversations endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status.

    :param client: HTTP client for the server.
    :param response_id: The response/task ID to poll.
    :returns: The response JSON dict once completed or failed.
    :raises AssertionError: If the response doesn't complete
        within 50 iterations.
    """
    for _ in range(50):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
    raise AssertionError(f"Response {response_id} did not reach terminal status")


async def _get_items(
    client: httpx.AsyncClient,
    conv_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all conversation items.

    :param client: HTTP client for the server.
    :param conv_id: The conversation ID.
    :returns: List of item dicts sorted by position.
    """
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    # Any: API returns heterogeneous item dicts
    result: list[dict[str, Any]] = resp.json()["data"]
    return result


async def test_list_conversations_empty(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/conversations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []
    assert body["has_more"] is False


async def test_get_conversation(client: httpx.AsyncClient) -> None:
    """Conversations are created implicitly via POST /responses."""
    await create_test_agent(client)
    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]

    resp = await client.get(f"/v1/conversations/{conv_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == conv_id
    assert body["object"] == "conversation"
    assert isinstance(body["created_at"], int)
    # New conversation has no title
    assert body["title"] is None


async def test_get_conversation_includes_updated_at(
    client: httpx.AsyncClient,
) -> None:
    """
    GET /conversations/{id} returns an updated_at timestamp.

    On a freshly created conversation, updated_at equals
    created_at. After a second turn, updated_at is >=
    created_at (may still be equal if the turn completes
    within the same epoch second).
    """
    await create_test_agent(client)
    first = await create_test_response(
        client,
        input_text="Hello",
        background=False,
        stream=False,
    )
    conv_id = first.body["conversation"]["id"]

    resp = await client.get(f"/v1/conversations/{conv_id}")
    body = resp.json()
    assert "updated_at" in body, "Conversation object must include updated_at field."
    assert isinstance(body["updated_at"], int)
    assert body["updated_at"] >= body["created_at"], (
        f"updated_at ({body['updated_at']}) must be >= created_at ({body['created_at']})."
    )


async def test_list_conversations_includes_updated_at(
    client: httpx.AsyncClient,
) -> None:
    """
    GET /conversations list items include updated_at with a
    value >= created_at.
    """
    await create_test_agent(client)
    await create_test_response(client, input_text="Hi")

    resp = await client.get("/v1/conversations")
    body = resp.json()
    assert len(body["data"]) == 1
    conv = body["data"][0]
    assert "updated_at" in conv, "Conversation list items must include updated_at."
    assert isinstance(conv["updated_at"], int)
    assert conv["updated_at"] >= conv["created_at"], (
        f"updated_at ({conv['updated_at']}) must be >= created_at ({conv['created_at']})."
    )


async def test_get_conversation_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/conversations/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["error"]["message"], str)


async def test_list_conversations_after_response(
    client: httpx.AsyncClient,
) -> None:
    await create_test_agent(client)
    result = await create_test_response(client, input_text="First")
    conv_id = result.body["conversation"]["id"]

    resp = await client.get("/v1/conversations")
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == conv_id
    # Verify PaginatedList structure
    assert isinstance(body["first_id"], str)
    assert isinstance(body["last_id"], str)
    assert body["has_more"] is False


async def test_update_conversation_title(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]

    patch_resp = await client.patch(
        f"/v1/conversations/{conv_id}",
        json={"title": "My Chat"},
    )
    assert patch_resp.status_code == 200
    body = patch_resp.json()
    assert body["title"] == "My Chat"

    # Verify via GET
    get_resp = await client.get(f"/v1/conversations/{conv_id}")
    assert get_resp.json()["title"] == "My Chat"


async def test_update_conversation_not_found(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.patch(
        "/v1/conversations/nonexistent",
        json={"title": "Nope"},
    )
    assert resp.status_code == 404
    assert isinstance(resp.json()["error"]["message"], str)


async def test_delete_conversation(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]

    del_resp = await client.delete(f"/v1/conversations/{conv_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == conv_id
    assert body["object"] == "conversation.deleted"
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/v1/conversations/{conv_id}")
    assert get_resp.status_code == 404


async def test_delete_conversation_not_found(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.delete("/v1/conversations/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["error"]["message"], str)


async def test_list_conversation_items(client: httpx.AsyncClient) -> None:
    """After creating a response, exactly one user message item exists."""
    await create_test_agent(client)
    result = await create_test_response(client, input_text="Hello agent")
    conv_id = result.body["conversation"]["id"]
    response_id = result.body["id"]

    resp = await client.get(f"/v1/conversations/{conv_id}/items")
    assert resp.status_code == 200
    body = resp.json()
    # Exactly one item: the user message appended during create_response
    assert len(body["data"]) == 1

    user_msg = body["data"][0]
    assert isinstance(user_msg["id"], str)
    assert user_msg["response_id"] == response_id
    assert user_msg["type"] == "message"
    assert user_msg["status"] == "completed"
    assert user_msg["role"] == "user"
    # Content block structure
    assert len(user_msg["content"]) == 1
    text_block = user_msg["content"][0]
    assert text_block["type"] == "input_text"
    assert text_block["text"] == "Hello agent"


async def test_list_conversation_items_not_found(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get("/v1/conversations/nonexistent/items")
    assert resp.status_code == 404
    assert isinstance(resp.json()["error"]["message"], str)


async def test_list_conversations_pagination(
    client: httpx.AsyncClient,
) -> None:
    """Conversation listing supports cursor-based pagination."""
    await create_test_agent(client)
    # Create 3 conversations (each response creates a new one)
    for i in range(3):
        await create_test_response(client, input_text=f"Conv {i}")

    # Fetch first page of 2
    resp = await client.get("/v1/conversations", params={"limit": 2})
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["has_more"] is True

    # Fetch second page
    resp2 = await client.get("/v1/conversations", params={"limit": 2, "after": body["last_id"]})
    body2 = resp2.json()
    assert len(body2["data"]) == 1
    assert body2["has_more"] is False


async def test_sort_by_updated_at(
    client: httpx.AsyncClient,
) -> None:
    """
    sort_by=updated_at reorders conversations by last activity.

    Create two conversations. Then add a second turn to the
    FIRST conversation so its updated_at advances. Listing with
    sort_by=updated_at desc should return the first conversation
    before the second, even though the second was created later.
    """
    await create_test_agent(client)

    # Conv A: created first
    first = await create_test_response(
        client,
        input_text="Conv A",
        background=False,
        stream=False,
    )
    conv_a_id = first.body["conversation"]["id"]
    resp_a_id = first.body["id"]

    # Conv B: created second (newer created_at)
    second = await create_test_response(
        client,
        input_text="Conv B",
        background=False,
        stream=False,
    )
    conv_b_id = second.body["conversation"]["id"]

    # Add a second turn to Conv A so its updated_at advances
    await create_test_response(
        client,
        input_text="Follow-up in A",
        previous_response_id=resp_a_id,
        background=False,
        stream=False,
    )

    # Default (sort_by=created_at desc): Conv B first
    resp_created = await client.get("/v1/conversations", params={"order": "desc"})
    created_ids = [c["id"] for c in resp_created.json()["data"]]
    assert created_ids[0] == conv_b_id, (
        f"Expected Conv B first when sorting by created_at desc, got {created_ids}."
    )

    # sort_by=updated_at desc: Conv A first (most recently updated)
    resp_updated = await client.get(
        "/v1/conversations",
        params={"sort_by": "updated_at", "order": "desc"},
    )
    updated_ids = [c["id"] for c in resp_updated.json()["data"]]
    assert updated_ids[0] == conv_a_id, (
        f"Expected Conv A first when sorting by updated_at desc "
        f"(it received a second turn), got {updated_ids}."
    )

    # sort_by=updated_at asc: Conv B first (oldest update)
    resp_asc = await client.get(
        "/v1/conversations",
        params={"sort_by": "updated_at", "order": "asc"},
    )
    asc_ids = [c["id"] for c in resp_asc.json()["data"]]
    assert asc_ids[0] == conv_b_id, (
        f"Expected Conv B first when sorting by updated_at asc, got {asc_ids}."
    )


async def test_list_conversation_items_pagination(
    client: httpx.AsyncClient,
) -> None:
    """Conversation items support cursor-based pagination."""
    await create_test_agent(client)

    # Create a multi-turn conversation (each turn adds a user message item).
    # background=False so Turn 1 completes before Turn 2 starts,
    # avoiding position races with the background workflow thread.
    first = await create_test_response(
        client,
        input_text="Turn 1",
        background=False,
        stream=False,
    )
    conv_id = first.body["conversation"]["id"]
    first_id = first.body["id"]

    await create_test_response(client, input_text="Turn 2", previous_response_id=first_id)

    # Fetch first page of 1 item
    resp = await client.get(f"/v1/conversations/{conv_id}/items", params={"limit": 1})
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["has_more"] is True

    # Fetch remaining items
    resp2 = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 10, "after": body["last_id"]},
    )
    body2 = resp2.json()
    assert len(body2["data"]) >= 1
    assert body2["has_more"] is False


async def test_delete_conversation_deletes_tasks(
    client: httpx.AsyncClient,
) -> None:
    """Deleting a conversation deletes its tasks and removes the conversation."""
    await create_test_agent(client)
    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]
    response_id = result.body["id"]

    del_resp = await client.delete(f"/v1/conversations/{conv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Conversation is gone
    assert (await client.get(f"/v1/conversations/{conv_id}")).status_code == 404
    # Task was deleted along with the conversation
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404


async def test_delete_conversation_with_active_tasks(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Deleting a conversation with in-flight tasks deletes the tasks and the conversation."""
    await create_test_agent(client)

    # Block the LLM call so the task stays active
    mock_llm.add_call(block=True)
    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]
    response_id = result.body["id"]
    assert result.body["status"] == "queued"

    del_resp = await client.delete(f"/v1/conversations/{conv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    assert (await client.get(f"/v1/conversations/{conv_id}")).status_code == 404
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404


# ── History Correctness ──────────────────────────────────


async def test_tool_call_items_position_order(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Single tool call round produces items in order:
    [user, function_call, function_call_output, assistant].
    """
    await create_test_agent(client)

    # Call 1: LLM returns a tool call
    tool_call_spec = [
        {
            "call_id": "call_order_1",
            "name": "load_skill",
            "arguments": '{"name": "nonexistent"}',
        },
    ]
    mock_llm.add_call(tool_calls=tool_call_spec)
    # Call 2: after tool execution, LLM returns final text
    mock_llm.add_call(text="Done with tools")

    result = await create_test_response(
        client,
        input_text="Use a tool please",
        background=True,
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed"

    items = await _get_items(client, conv_id)

    assert len(items) == 4, f"Expected [user, fc, fco, assistant], got {len(items)}"
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Use a tool please"

    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_order_1"
    assert items[1]["name"] == "load_skill"

    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_order_1"
    # Tool produced some output string
    assert isinstance(items[2]["output"], str)

    assert items[3]["type"] == "message"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "Done with tools"


async def test_multiple_tool_call_rounds_ordering(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two consecutive tool call rounds produce items in order:
    [user, fc_1, fco_1, fc_2, fco_2, assistant].
    """
    await create_test_agent(client)

    # Round 1: tool call
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_round_1",
                "name": "load_skill",
                "arguments": '{"name": "first"}',
            },
        ],
    )
    # Round 2: another tool call
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_round_2",
                "name": "load_skill",
                "arguments": '{"name": "second"}',
            },
        ],
    )
    # Round 3: final text response
    mock_llm.add_call(text="Both tools done")

    result = await create_test_response(
        client,
        input_text="Two tools please",
        background=True,
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed"

    items = await _get_items(client, conv_id)

    assert len(items) == 6, (
        f"Expected [user, fc_1, fco_1, fc_2, fco_2, assistant], got {len(items)}"
    )
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Two tools please"

    # Round 1
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_round_1"
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_round_1"

    # Round 2
    assert items[3]["type"] == "function_call"
    assert items[3]["call_id"] == "call_round_2"
    assert items[4]["type"] == "function_call_output"
    assert items[4]["call_id"] == "call_round_2"

    assert items[5]["type"] == "message"
    assert items[5]["role"] == "assistant"
    assert items[5]["content"][0]["text"] == "Both tools done"


async def test_multi_turn_after_steering_sees_full_history(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    After steering produces two assistant messages, a new turn
    (via previous_response_id on the completed task) sees the
    full conversation history including steered messages.
    """
    await create_test_agent(client)

    # Turn 1: block so we can steer
    call_1 = mock_llm.add_call(text="First answer", block=True)
    # Continuation after steering
    mock_llm.add_call(text="Steered answer")

    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Steer while blocked
    await create_test_response(
        client,
        input_text="Actually, change this",
        previous_response_id=first_id,
    )
    call_1.release()

    # Wait for turn 1 to complete (with steering continuation)
    body = await _wait_for_completion(client, first_id)
    assert body["status"] == "completed"

    # Turn 2: new task in the same conversation
    mock_llm.add_call(text="Turn two answer")
    second = await create_test_response(
        client,
        input_text="Follow-up question",
        previous_response_id=first_id,
    )
    second_id = second.body["id"]
    # New task, same conversation
    assert second_id != first_id

    body = await _wait_for_completion(client, second_id)
    assert body["status"] == "completed"

    # Full history: user, steered-user, assistant, steered-assistant,
    # turn-2-user, turn-2-assistant
    items = await _get_items(client, conv_id)
    user_texts = [i["content"][0]["text"] for i in items if i.get("role") == "user"]
    assistant_texts = [i["content"][0]["text"] for i in items if i.get("role") == "assistant"]
    assert "Hello" in user_texts
    assert "Actually, change this" in user_texts
    assert "Follow-up question" in user_texts
    assert "First answer" in assistant_texts
    assert "Steered answer" in assistant_texts
    assert "Turn two answer" in assistant_texts
    # Total: 3 user + 3 assistant = 6 message items
    msg_items = [i for i in items if i["type"] == "message"]
    assert len(msg_items) == 6


async def test_steering_position_among_tool_items(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer during tool execution. The steered user message must
    appear in the conversation items alongside tool items, and
    the final assistant response must reflect the steered context.
    """
    await create_test_agent(client)

    # Call 1: returns tool call, blocks so tool hasn't run yet
    tool_call_spec = [
        {
            "call_id": "call_steer_pos_1",
            "name": "load_skill",
            "arguments": '{"name": "nonexistent"}',
        },
    ]
    call_1 = mock_llm.add_call(
        tool_calls=tool_call_spec,
        block=True,
    )
    # Call 2: after tool execution, blocks so we can steer
    call_2 = mock_llm.add_call(
        text="Post-tool answer",
        block=True,
    )
    # Call 3: continuation after steering detected
    mock_llm.add_call(text="Steered tool answer")

    first = await create_test_response(
        client,
        input_text="Run a tool",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Release tool call so tool executes
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)
    call_1.release()

    # Wait for second LLM call (post-tool)
    await asyncio.wait_for(call_2.call_event.wait(), timeout=10)

    # Steer while blocked in second LLM call
    await create_test_response(
        client,
        input_text="Steering during tools",
        previous_response_id=first_id,
    )

    call_2.release()

    body = await _wait_for_completion(client, first_id)
    assert body["status"] == "completed"

    items = await _get_items(client, conv_id)
    types = [i["type"] for i in items]

    # Positional assertions: user, fc, fco, steered-user,
    # post-tool assistant, steered-continuation assistant
    assert types[0] == "message", "First item must be user message"
    assert types[1] == "function_call"
    assert types[2] == "function_call_output"

    # Steered message appears after tool items
    steered_idx = next(
        idx
        for idx, i in enumerate(items)
        if i.get("role") == "user" and i["content"][0]["text"] == "Steering during tools"
    )
    assert steered_idx > 2, "Steered message must appear after function_call/output items"

    # Verify both user messages
    user_texts = [i["content"][0]["text"] for i in items if i.get("role") == "user"]
    assert "Run a tool" in user_texts
    assert "Steering during tools" in user_texts

    # Verify assistant produced the steered answer
    assistant_texts = [i["content"][0]["text"] for i in items if i.get("role") == "assistant"]
    assert "Steered tool answer" in assistant_texts

    # 3 LLM calls: tool call + post-tool + steering continuation
    assert mock_llm.call_count == 3


# ── Client-side tools + steering ─────────────────────────


# OpenAI-format client-side tool schema used across
# client-tool tests — avoids duplicating the dict literal.
_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def test_client_side_tool_completes_with_function_call(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When the LLM invokes a client-side tool, the response
    completes with the ``function_call`` item in the output
    and no ``function_call_output`` (not executed server-side).
    """
    await create_test_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_weather_1",
                "name": "get_weather",
                "arguments": '{"city": "San Francisco"}',
            },
        ],
    )

    result = await create_test_response(
        client,
        input_text="What is the weather?",
        tools=[_WEATHER_TOOL],
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)

    # Response completed — not failed or incomplete
    assert body["status"] == "completed", (
        f"Expected 'completed' for client-side tool path; "
        f"got '{body['status']}'. If 'failed', the workflow "
        f"tried to execute the client-side tool server-side."
    )

    items = await _get_items(client, conv_id)
    types = [i["type"] for i in items]

    # [user message, function_call] — no function_call_output
    assert types == ["message", "function_call"], (
        f"Expected [message, function_call] for client-side "
        f"tool; got {types}. If function_call_output is "
        f"present, the tool was executed server-side."
    )
    assert items[1]["name"] == "get_weather"
    assert items[1]["call_id"] == "call_weather_1"
    assert items[1]["arguments"] == '{"city": "San Francisco"}'

    # 1 LLM call only — no tool execution loop
    assert mock_llm.call_count == 1, (
        f"Expected 1 LLM call; got {mock_llm.call_count}. "
        f"If 2, the loop continued after client-side tool call."
    )


async def test_mixed_batch_executes_server_tools_returns_client_tools(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When the LLM returns both a server-side tool and a client-side
    tool in one batch, the server-side tool is executed (producing
    a ``function_call_output``) and the client-side tool is returned
    as a ``function_call`` without execution.

    This is the critical mixed-batch scenario. The old code used an
    ``any()`` check that skipped ALL tools when any client tool was
    present — server-side tools would never execute.
    """
    await create_test_agent(client)

    # LLM returns two tool calls in one batch:
    # - load_skill: not registered on the minimal test agent,
    #   so call_tool returns an error string — but it IS
    #   executed server-side (produces function_call_output).
    # - get_weather: client-side tool, must NOT be executed.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_server_1",
                "name": "load_skill",
                "arguments": '{"name": "nonexistent"}',
            },
            {
                "call_id": "call_client_1",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
        ],
    )

    result = await create_test_response(
        client,
        input_text="Use both tools",
        tools=[_WEATHER_TOOL],
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed", (
        f"Expected 'completed'; got '{body['status']}'. "
        f"If 'failed', a client-side tool was invoked server-side."
    )

    items = await _get_items(client, conv_id)
    types = [i["type"] for i in items]

    # Expected: [user, fc(load_skill), fc(get_weather),
    #            fco(load_skill)]
    # function_call items are persisted first (both tools),
    # then server-side tools are executed (only load_skill).
    # No function_call_output for get_weather.
    assert types.count("function_call") == 2, (
        f"Expected 2 function_call items (one per tool in the "
        f"batch); got {types.count('function_call')}. Types: {types}"
    )
    assert types.count("function_call_output") == 1, (
        f"Expected 1 function_call_output (only load_skill "
        f"executed server-side); got "
        f"{types.count('function_call_output')}. If 0, "
        f"server-side tool was skipped (the old any() bug). "
        f"If 2, client-side tool was executed server-side. "
        f"Types: {types}"
    )

    # Verify the function_call_output belongs to load_skill
    fco_items = [i for i in items if i["type"] == "function_call_output"]
    assert fco_items[0]["call_id"] == "call_server_1", (
        f"function_call_output should be for load_skill "
        f"(call_server_1); got call_id={fco_items[0]['call_id']}"
    )
    # load_skill returns an error string for nonexistent skills
    assert "not found" in fco_items[0]["output"].lower(), (
        "load_skill output should contain an error for the "
        "nonexistent skill, proving it was actually executed"
    )

    # Verify each function_call has the correct name↔call_id
    fc_items = [i for i in items if i["type"] == "function_call"]
    fc_by_name = {i["name"]: i for i in fc_items}
    assert fc_by_name["load_skill"]["call_id"] == "call_server_1", (
        "load_skill function_call must have the correct call_id"
    )
    assert fc_by_name["get_weather"]["call_id"] == "call_client_1", (
        "get_weather function_call must have the correct call_id"
    )

    # Verify NO function_call_output for get_weather
    fco_call_ids = {i["call_id"] for i in fco_items}
    assert "call_client_1" not in fco_call_ids, (
        "get_weather must not have a function_call_output — it is a client-side tool"
    )

    # 1 LLM call only — response completed after mixed batch
    assert mock_llm.call_count == 1, (
        f"Expected 1 LLM call; got {mock_llm.call_count}. "
        f"The loop should not continue after a mixed batch."
    )


async def test_steering_during_llm_with_client_tool_persists_steered_message(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steering a message while the LLM is streaming, followed by
    the LLM returning a client-side tool call.

    The steered message is persisted in the conversation but not
    processed in this turn (known gap — the client-tool path
    completes immediately). On the next request the steered
    message appears in the prompt.
    """
    await create_test_agent(client)

    # Block so we can steer while LLM is "streaming"
    call_1 = mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_weather_steer",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
        ],
        block=True,
    )
    # call_2 is enqueued later (after first turn completes) so
    # we can capture received_kwargs on the named reference.

    first = await create_test_response(
        client,
        input_text="What is the weather in SF?",
        tools=[_WEATHER_TOOL],
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Wait for LLM to be called, then steer
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)
    steer_resp = await create_test_response(
        client,
        input_text="Also check NYC",
        previous_response_id=first_id,
    )
    # Steering delivered to the running task
    assert steer_resp.body["id"] == first_id, (
        "Steering should return the same response ID, indicating try_deliver succeeded."
    )

    # Release the LLM — it returns client-side tool call,
    # response completes immediately
    call_1.release()

    body = await _wait_for_completion(client, first_id)
    assert body["status"] == "completed"

    # Conversation items after first turn
    items = await _get_items(client, conv_id)
    types = [i["type"] for i in items]

    # [user, steered-user, function_call]
    # The steered message is persisted even though the agent
    # didn't process it this turn.
    assert "function_call" in types, f"Expected function_call in items; got {types}"
    # Verify the function_call item has the correct tool content
    fc_item = next(i for i in items if i["type"] == "function_call")
    assert fc_item["name"] == "get_weather", (
        f"function_call should invoke get_weather; got {fc_item['name']}"
    )
    assert fc_item["call_id"] == "call_weather_steer"
    assert fc_item["arguments"] == '{"city": "SF"}'
    user_texts = [i["content"][0]["text"] for i in items if i.get("role") == "user"]
    assert "What is the weather in SF?" in user_texts
    # Steered message is persisted in the conversation
    assert "Also check NYC" in user_texts, (
        "Steered message must be persisted in the conversation "
        "even though the client-tool path completes without "
        "processing it. If missing, try_deliver failed."
    )
    # No function_call_output — client-side tool not executed
    assert "function_call_output" not in types, (
        "Client-side tool must not produce function_call_output"
    )

    # ── Next turn picks up steered message ──────────────
    # Client continues with previous_response_id using a
    # plain text message. The LLM sees the full conversation
    # history — including the steered message from turn 1 —
    # in its prompt.
    call_2 = mock_llm.add_call(text="Weather is sunny and also NYC is rainy")
    second = await create_test_response(
        client,
        input_text="The weather result was 72F sunny",
        previous_response_id=first_id,
        tools=[_WEATHER_TOOL],
    )
    second_id = second.body["id"]

    body = await _wait_for_completion(client, second_id)
    assert body["status"] == "completed"

    # Verify the LLM actually received the steered message
    # in its input. This is the critical assertion — without
    # it, the test would pass even if the steered message was
    # silently dropped, because the mock returns a fixed string.
    assert call_2.received_kwargs is not None, (
        "Second LLM call was never made — the workflow did not continue to a second turn."
    )
    llm_input = call_2.received_kwargs["input"]
    # Flatten all text content from the LLM input items into
    # a single string for substring matching
    llm_input_str = str(llm_input)
    assert "Also check NYC" in llm_input_str, (
        "Steered message must appear in the LLM prompt on "
        "the second turn. If missing, the steered message "
        "was persisted in the conversation store but not "
        "included in the history sent to the LLM."
    )

    # Full conversation includes the steered message
    items = await _get_items(client, conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    assert "What is the weather in SF?" in user_texts
    assert "Also check NYC" in user_texts
    assert "The weather result was 72F sunny" in user_texts


async def test_client_tool_continuation_with_function_call_output(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When the client sends ``function_call_output`` items as input
    (continuing after a client-side tool call), the LLM receives
    the tool outputs in the correct Responses API format and
    produces a final text response.

    This is the full round-trip: LLM returns function_call →
    client executes tool → client sends function_call_output →
    LLM sees tool result in history → LLM responds with text.

    Without _split_input_to_items, the function_call_output items
    are incorrectly wrapped in a user message, causing a 400 error
    from the LLM because it sees a broken user message where tool
    results should be.
    """
    await create_test_agent(client)

    # Turn 1: LLM invokes a client-side tool.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_weather_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
        ],
    )
    turn_1 = await create_test_response(
        client,
        input_text="What is the weather?",
        tools=[_WEATHER_TOOL],
    )
    turn_1_id = turn_1.body["id"]
    body_1 = await _wait_for_completion(client, turn_1_id)
    # Turn 1 completes with the function_call (not executed).
    assert body_1["status"] == "completed", (
        f"Turn 1 should complete with function_call; got {body_1['status']}"
    )

    # Turn 2: Client sends the tool result back as input.
    # The LLM should see the function_call_output in its
    # history and respond with text.
    call_2 = mock_llm.add_call(text="The weather in SF is 72F sunny.")
    turn_2 = await create_test_response(
        client,
        # function_call_output items as input — this is what
        # the TUI / client.py sends after executing tools locally.
        input_text=[
            {
                "type": "function_call_output",
                "call_id": "call_weather_1",
                "output": "72 degrees, sunny",
            },
        ],
        previous_response_id=turn_1_id,
        tools=[_WEATHER_TOOL],
    )
    turn_2_id = turn_2.body["id"]

    body_2 = await _wait_for_completion(client, turn_2_id)
    # If _split_input_to_items is broken, the LLM gets a 400
    # and this status is "failed".
    assert body_2["status"] == "completed", (
        f"Turn 2 should complete after receiving tool results; "
        f"got {body_2['status']}. If 'failed', the "
        f"function_call_output was likely wrapped in a user "
        f"message instead of being persisted as a separate "
        f"function_call_output item."
    )

    # Verify the LLM received the tool output in its history.
    assert call_2.received_kwargs is not None, (
        "Second LLM call was never made — the workflow did "
        "not start or crashed before reaching the LLM."
    )
    llm_input_str = str(call_2.received_kwargs["input"])
    # The tool result text must appear in the LLM prompt,
    # proving it was persisted as a function_call_output item
    # (not buried inside a user message where _extract_text
    # would discard it).
    assert "72 degrees, sunny" in llm_input_str, (
        "Tool result must appear in the LLM prompt. If missing, "
        "the function_call_output was persisted as a user message "
        "and _extract_text stripped the non-text content."
    )

    # Verify conversation items have the correct types.
    conv_id = turn_1.body["conversation"]["id"]
    items = await _get_items(client, conv_id)
    types = [i["type"] for i in items]

    # Expected sequence:
    # [user("What is the weather?"),
    #  function_call(get_weather),
    #  function_call_output(call_weather_1),  ← from client input
    #  message(assistant: "72F sunny")]
    assert types.count("function_call_output") == 1, (
        f"Expected 1 function_call_output from the client's "
        f"tool result; got {types.count('function_call_output')}. "
        f"If 0, the output was persisted as a user message "
        f"instead. Types: {types}"
    )

    # The function_call_output must have the correct call_id.
    fco = [i for i in items if i["type"] == "function_call_output"]
    assert fco[0]["call_id"] == "call_weather_1", (
        f"function_call_output call_id should be 'call_weather_1'; got '{fco[0]['call_id']}'"
    )
    assert fco[0]["output"] == "72 degrees, sunny", (
        "function_call_output should contain the tool result"
    )

    # 2 LLM calls: turn 1 (returns tool call) + turn 2 (after
    # receiving tool result, returns text).
    assert mock_llm.call_count == 2, (
        f"Expected 2 LLM calls (one per turn); got {mock_llm.call_count}."
    )


# ── Error Handling ───────────────────────────────────────


async def test_unhandled_exception_returns_json_500(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unhandled exceptions (e.g. OperationalError from SQLITE_BUSY)
    return the standard JSON error schema, not a bare Starlette 500.
    """
    from sqlalchemy.exc import OperationalError

    from agent_plane.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )

    def _blow_up(*args: object, **kwargs: object) -> None:
        """
        Simulate SQLITE_BUSY by raising OperationalError.
        """
        raise OperationalError(
            "database is locked",
            params=None,
            orig=Exception("database is locked"),
        )

    monkeypatch.setattr(
        SqlAlchemyConversationStore,
        "list_conversations",
        _blow_up,
    )

    # raise_app_exceptions=False lets the ASGI error handler run
    # instead of re-raising through the transport — matches how
    # a real HTTP client sees the response.
    transport = httpx.ASGITransport(
        app=app,
        raise_app_exceptions=False,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as error_client:
        resp = await error_client.get("/v1/conversations")

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert isinstance(body["error"]["message"], str)
