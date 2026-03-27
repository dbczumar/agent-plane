"""Integration tests for /v1/conversations endpoints."""

from __future__ import annotations

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

    call_1.call_event.wait(timeout=10)

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
    call_1.call_event.wait(timeout=10)
    call_1.release()

    # Wait for second LLM call (post-tool)
    call_2.call_event.wait(timeout=10)

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
