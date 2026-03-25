"""Integration tests for /api/agents endpoints."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import build_agent_bundle, create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio


async def test_create_agent(client: httpx.AsyncClient) -> None:
    body = await create_test_agent(client, name="my-agent")
    assert body["object"] == "agent"
    assert body["name"] == "my-agent"
    assert body["description"] is None
    assert isinstance(body["id"], str)
    assert isinstance(body["created_at"], int)


async def test_create_agent_with_description(client: httpx.AsyncClient) -> None:
    body = await create_test_agent(client, name="described-agent", description="A helpful agent")
    assert body["description"] == "A helpful agent"


async def test_create_agent_duplicate_name(client: httpx.AsyncClient) -> None:
    await create_test_agent(client, name="unique-agent")
    bundle = build_agent_bundle(name="unique-agent")
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["error"]["message"]


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
    # Verify PaginatedList structure
    assert isinstance(body["first_id"], str)
    assert isinstance(body["last_id"], str)
    assert body["has_more"] is False


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
    assert "not found" in resp.json()["error"]["message"].lower()


async def test_delete_agent(client: httpx.AsyncClient) -> None:
    created = await create_test_agent(client, name="delete-me")
    agent_id = created["id"]

    del_resp = await client.delete(f"/api/agents/{agent_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == agent_id
    assert body["object"] == "agent.deleted"
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/api/agents/{agent_id}")
    assert get_resp.status_code == 404


async def test_delete_agent_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/api/agents/nonexistent")
    assert resp.status_code == 404
    assert "not found" in resp.json()["error"]["message"].lower()


async def test_list_agents_pagination(client: httpx.AsyncClient) -> None:
    for i in range(3):
        await create_test_agent(client, name=f"agent-{i}")

    resp = await client.get("/api/agents", params={"limit": 2})
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["has_more"] is True

    resp2 = await client.get("/api/agents", params={"limit": 2, "after": body["last_id"]})
    body2 = resp2.json()
    assert len(body2["data"]) == 1
    assert body2["has_more"] is False


async def test_delete_agent_with_tasks(client: httpx.AsyncClient) -> None:
    """Deleting an agent deletes its tasks and removes the agent."""
    created = await create_test_agent(client, name="agent-with-tasks")
    agent_id = created["id"]

    result = await create_test_response(client, model="agent-with-tasks")
    response_id = result.body["id"]

    del_resp = await client.delete(f"/api/agents/{agent_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Agent is gone
    assert (await client.get(f"/api/agents/{agent_id}")).status_code == 404
    # Task was deleted along with the agent
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404


async def test_delete_agent_with_active_tasks(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Deleting an agent with in-flight tasks deletes the tasks and the agent."""
    created = await create_test_agent(client, name="busy-agent")
    agent_id = created["id"]

    # Block the LLM call so the task stays active
    mock_llm.add_call(block=True)
    result = await create_test_response(client, model="busy-agent")
    response_id = result.body["id"]
    assert result.body["status"] == "queued"

    del_resp = await client.delete(f"/api/agents/{agent_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    assert (await client.get(f"/api/agents/{agent_id}")).status_code == 404
    assert (await client.get(f"/v1/responses/{response_id}")).status_code == 404


async def test_create_agent_invalid_bundle(client: httpx.AsyncClient) -> None:
    """Uploading a corrupt/non-tarball bundle returns 400."""
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", b"not-a-tarball", "application/gzip")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


async def test_create_agent_bundle_missing_name(client: httpx.AsyncClient) -> None:
    """A valid tarball whose config.yaml has no name returns 400."""
    import io
    import tarfile

    import yaml

    config_bytes = yaml.dump({"spec_version": 1}).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))

    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", buf.getvalue(), "application/gzip")},
    )
    assert resp.status_code == 400
    assert "name" in resp.json()["error"]["message"].lower()


async def test_create_agent_invalid_spec_version(client: httpx.AsyncClient) -> None:
    """A bundle with an unsupported spec version returns 400."""
    import io
    import tarfile

    import yaml

    config_bytes = yaml.dump({"spec_version": 99, "name": "bad"}).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))

    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", buf.getvalue(), "application/gzip")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"
    assert "invalid agent spec" in resp.json()["error"]["message"]


async def test_create_agent_stores_bundle(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    """Successful creation stores the original bundle bytes in the artifact store."""
    bundle = build_agent_bundle(name="stored-agent")
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201
    agent_id = resp.json()["id"]

    # LocalArtifactStore writes to tmp_path/artifacts/<key>
    stored = (tmp_path / "artifacts" / agent_id).read_bytes()
    assert stored == bundle
