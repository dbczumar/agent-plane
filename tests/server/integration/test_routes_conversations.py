"""Integration tests for /v1/conversations endpoints."""

from __future__ import annotations

import httpx
import pytest

from tests.server.conftest import IntegrationTaskStore
from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio


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

    # Create a multi-turn conversation (each turn adds a user message item)
    first = await create_test_response(client, input_text="Turn 1")
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
    task_store: IntegrationTaskStore,
) -> None:
    """Deleting a conversation with in-flight tasks deletes the tasks and the conversation."""
    await create_test_agent(client)

    # Keep the task active (not auto-completed)
    task_store.defer_all_completions = True
    result = await create_test_response(client)
    conv_id = result.body["conversation"]["id"]
    response_id = result.body["id"]
    assert result.body["status"] == "queued"

    del_resp = await client.delete(f"/v1/conversations/{conv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    assert (await client.get(f"/v1/conversations/{conv_id}")).status_code == 404
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404
