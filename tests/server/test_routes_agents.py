"""Integration tests for /api/agents endpoints."""

from __future__ import annotations

import httpx
import pytest

from tests.server.helpers import create_test_agent

pytestmark = pytest.mark.asyncio


async def test_create_agent(client: httpx.AsyncClient) -> None:
    body = await create_test_agent(client, name="my-agent")
    assert body["object"] == "agent"
    assert body["name"] == "my-agent"
    assert body["description"] is None
    assert isinstance(body["id"], str)
    assert isinstance(body["created_at"], int)


async def test_create_agent_with_description(client: httpx.AsyncClient) -> None:
    body = await create_test_agent(
        client, name="described-agent", description="A helpful agent"
    )
    assert body["description"] == "A helpful agent"


async def test_create_agent_duplicate_name(client: httpx.AsyncClient) -> None:
    await create_test_agent(client, name="unique-agent")
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("b.tar.gz", b"data", "application/gzip")},
        data={"name": "unique-agent"},
    )
    assert resp.status_code == 409


async def test_list_agents_empty(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/agents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []
    assert body["has_more"] is False


async def test_list_agents(client: httpx.AsyncClient) -> None:
    await create_test_agent(client, name="agent-a")
    await create_test_agent(client, name="agent-b")

    resp = await client.get("/api/agents")
    body = resp.json()
    assert len(body["data"]) == 2
    names = {a["name"] for a in body["data"]}
    assert names == {"agent-a", "agent-b"}


async def test_get_agent(client: httpx.AsyncClient) -> None:
    created = await create_test_agent(client, name="get-me")
    agent_id = created["id"]

    resp = await client.get(f"/api/agents/{agent_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == agent_id
    assert body["name"] == "get-me"


async def test_get_agent_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/agents/nonexistent")
    assert resp.status_code == 404


async def test_delete_agent(client: httpx.AsyncClient) -> None:
    created = await create_test_agent(client, name="delete-me")
    agent_id = created["id"]

    del_resp = await client.delete(f"/api/agents/{agent_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == agent_id
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/api/agents/{agent_id}")
    assert get_resp.status_code == 404


async def test_delete_agent_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/api/agents/nonexistent")
    assert resp.status_code == 404


async def test_list_agents_pagination(client: httpx.AsyncClient) -> None:
    for i in range(3):
        await create_test_agent(client, name=f"agent-{i}")

    resp = await client.get("/api/agents", params={"limit": 2})
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["has_more"] is True

    resp2 = await client.get(
        "/api/agents", params={"limit": 2, "after": body["last_id"]}
    )
    body2 = resp2.json()
    assert len(body2["data"]) == 1
    assert body2["has_more"] is False
