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
    import hashlib

    bundle = build_agent_bundle(name="stored-agent")
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201
    agent_id = resp.json()["id"]

    # LocalArtifactStore writes to tmp_path/artifacts/<agent_id>/<sha256>
    digest = hashlib.sha256(bundle).hexdigest()
    stored = (tmp_path / "artifacts" / agent_id / digest).read_bytes()
    assert stored == bundle


# ── Update (PUT) tests ─────────────────────────────────────────


def _build_minimal_bundle(
    name: str,
    description: str | None = None,
) -> bytes:
    """
    Build a minimal valid agent bundle without an LLM config.

    Uses only ``spec_version`` and ``name`` so the bundle passes
    spec validation without requiring an ``llm.connection`` block.

    :param name: Agent name, e.g. ``"test-agent"``.
    :param description: Optional description.
    :returns: Raw bytes of the ``.tar.gz`` bundle.
    """
    import io
    import tarfile

    import yaml

    config: dict[str, object] = {"spec_version": 1, "name": name}
    if description is not None:
        config["description"] = description
    config_bytes = yaml.dump(config).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


async def _create_minimal_agent(
    client: httpx.AsyncClient,
    name: str,
    description: str | None = None,
) -> dict[str, object]:
    """
    Create an agent using a minimal bundle (no LLM config).

    :param client: The test HTTP client.
    :param name: Agent name.
    :param description: Optional description.
    :returns: The created agent JSON body.
    """
    bundle = _build_minimal_bundle(name, description)
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_update_agent(client: httpx.AsyncClient) -> None:
    """PUT with a new bundle returns 200 with bumped version."""
    created = await _create_minimal_agent(client, "upd-agent")
    agent_id = created["id"]
    assert created["version"] == 1
    assert created["updated_at"] is None

    new_bundle = _build_minimal_bundle("upd-agent", description="v2")
    resp = await client.put(
        f"/api/agents/{agent_id}",
        files={"bundle": ("agent.tar.gz", new_bundle, "application/gzip")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 2
    assert body["updated_at"] is not None
    assert body["name"] == "upd-agent"


async def test_update_agent_not_found(client: httpx.AsyncClient) -> None:
    """PUT to a nonexistent agent returns 404."""
    bundle = _build_minimal_bundle("ghost")
    resp = await client.put(
        "/api/agents/ag_nonexistent",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 404


async def test_update_agent_name_mismatch(client: httpx.AsyncClient) -> None:
    """PUT with a spec whose name differs from the agent returns 400."""
    created = await _create_minimal_agent(client, "original-name")
    agent_id = created["id"]

    wrong_bundle = _build_minimal_bundle("different-name")
    resp = await client.put(
        f"/api/agents/{agent_id}",
        files={"bundle": ("agent.tar.gz", wrong_bundle, "application/gzip")},
    )
    assert resp.status_code == 400
    assert "immutable" in resp.json()["error"]["message"]


async def test_update_agent_invalid_bundle(client: httpx.AsyncClient) -> None:
    """PUT with corrupt bytes returns 400."""
    created = await _create_minimal_agent(client, "valid-agent")
    agent_id = created["id"]

    resp = await client.put(
        f"/api/agents/{agent_id}",
        files={"bundle": ("agent.tar.gz", b"garbage", "application/gzip")},
    )
    assert resp.status_code == 400


async def test_update_same_bundle_is_idempotent(
    client: httpx.AsyncClient,
) -> None:
    """PUT with the identical bundle is a no-op (no version bump).

    The route handler computes the content-addressed bundle_location
    from SHA-256(bytes). When the new location matches the existing
    one, it short-circuits before writing to the artifact store or
    updating the DB, returning the current agent unchanged.

    This test verifies the short-circuit path by asserting the
    returned version and updated_at are unchanged. If the handler
    failed to compare bundle_locations and always called
    agent_store.update(), version would be 2 and updated_at would
    be set — both assertions would fail.
    """
    bundle = _build_minimal_bundle("idem-agent")
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201
    created = resp.json()
    agent_id = created["id"]

    # PUT the same bytes again
    resp2 = await client.put(
        f"/api/agents/{agent_id}",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp2.status_code == 200
    body = resp2.json()
    # Version stays at 1 — the content hash matched, so the handler
    # returned early. If version is 2, the short-circuit failed and
    # agent_store.update() was called unnecessarily.
    assert body["version"] == 1
    # updated_at stays None — no DB mutation occurred. If set, the
    # handler wrote to the DB despite identical content.
    assert body["updated_at"] is None


async def test_update_preserves_old_bundle(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    """After update, the old bundle still exists in the artifact store."""
    bundle_v1 = _build_minimal_bundle("preserve-agent")
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle_v1, "application/gzip")},
    )
    assert resp.status_code == 201
    agent_id = resp.json()["id"]

    # Find the v1 bundle location on disk
    import hashlib

    v1_hash = hashlib.sha256(bundle_v1).hexdigest()
    v1_path = tmp_path / "artifacts" / agent_id / v1_hash

    bundle_v2 = _build_minimal_bundle("preserve-agent", description="v2")
    resp2 = await client.put(
        f"/api/agents/{agent_id}",
        files={"bundle": ("agent.tar.gz", bundle_v2, "application/gzip")},
    )
    assert resp2.status_code == 200
    assert resp2.json()["version"] == 2

    # Old bundle is still on disk (not deleted)
    assert v1_path.exists(), "Old bundle should be preserved"


async def test_create_agent_has_null_updated_at(
    client: httpx.AsyncClient,
) -> None:
    """Newly created agents have version=1 and updated_at=null.

    This is a focused regression test for the create endpoint's
    initial field values — separate from test_update_agent which
    also checks v1/None before updating. Catches regressions where
    create() accidentally sets updated_at or starts at version != 1.
    """
    created = await _create_minimal_agent(client, "fresh-agent")
    assert created["version"] == 1
    assert created["updated_at"] is None
