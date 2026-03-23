"""Integration tests for /v1/conversations endpoints."""

from __future__ import annotations

import httpx
import pytest

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
    _, response_body = await create_test_response(client)
    conv_id = response_body["conversation"]["id"]

    resp = await client.get(f"/v1/conversations/{conv_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == conv_id
    assert body["object"] == "conversation"
    assert isinstance(body["created_at"], int)


async def test_get_conversation_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/conversations/nonexistent")
    assert resp.status_code == 404


async def test_list_conversations_after_response(
    client: httpx.AsyncClient,
) -> None:
    await create_test_agent(client)
    _, resp1 = await create_test_response(client, input_text="First")
    conv_id = resp1["conversation"]["id"]

    resp = await client.get("/v1/conversations")
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == conv_id


async def test_update_conversation_title(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    _, resp_body = await create_test_response(client)
    conv_id = resp_body["conversation"]["id"]

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


async def test_delete_conversation(client: httpx.AsyncClient) -> None:
    await create_test_agent(client)
    _, resp_body = await create_test_response(client)
    conv_id = resp_body["conversation"]["id"]

    del_resp = await client.delete(f"/v1/conversations/{conv_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == conv_id
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/v1/conversations/{conv_id}")
    assert get_resp.status_code == 404


async def test_delete_conversation_not_found(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.delete("/v1/conversations/nonexistent")
    assert resp.status_code == 404


async def test_list_conversation_items(client: httpx.AsyncClient) -> None:
    """After creating a response, the user message should appear as an item."""
    await create_test_agent(client)
    _, resp_body = await create_test_response(client, input_text="Hello agent")
    conv_id = resp_body["conversation"]["id"]

    resp = await client.get(f"/v1/conversations/{conv_id}/items")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) >= 1

    # The first item should be the user message
    user_msg = body["data"][0]
    assert user_msg["type"] == "message"
    assert user_msg["role"] == "user"
    # Verify the content contains our input text
    text_block = user_msg["content"][0]
    assert text_block["type"] == "input_text"
    assert text_block["text"] == "Hello agent"


async def test_list_conversation_items_not_found(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get("/v1/conversations/nonexistent/items")
    assert resp.status_code == 404
