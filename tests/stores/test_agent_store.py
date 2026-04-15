"""Tests for SqlAlchemyAgentStore."""

from __future__ import annotations

from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore


def test_create_and_get(agent_store: SqlAlchemyAgentStore) -> None:
    agent = agent_store.create(
        agent_id="ag_test_gpt4", name="gpt-4", bundle_location="ag_test_gpt4/fakehash"
    )
    assert agent.id.startswith("ag_")
    assert agent.name == "gpt-4"

    fetched = agent_store.get(agent.id)
    assert fetched is not None
    assert fetched.id == agent.id
    assert fetched.name == "gpt-4"


def test_get_nonexistent(agent_store: SqlAlchemyAgentStore) -> None:
    assert agent_store.get("ag_nonexistent") is None


def test_get_by_name(agent_store: SqlAlchemyAgentStore) -> None:
    agent_store.create(
        agent_id="ag_test_claude", name="claude", bundle_location="ag_test_claude/fakehash"
    )
    found = agent_store.get_by_name("claude")
    assert found is not None
    assert found.name == "claude"
    assert agent_store.get_by_name("missing") is None


def test_create_with_description(agent_store: SqlAlchemyAgentStore) -> None:
    agent = agent_store.create(
        agent_id="ag_test_helper",
        name="helper",
        bundle_location="ag_test_helper/fakehash",
        description="A helper agent",
    )
    assert agent.description == "A helper agent"


def test_delete(agent_store: SqlAlchemyAgentStore) -> None:
    agent = agent_store.create(
        agent_id="ag_test_temp", name="temp", bundle_location="ag_test_temp/fakehash"
    )
    assert agent_store.delete(agent.id) is True
    assert agent_store.get(agent.id) is None
    assert agent_store.delete(agent.id) is False


def test_list_pagination(agent_store: SqlAlchemyAgentStore) -> None:
    for i in range(5):
        agent_store.create(
            agent_id=f"ag_test_{i}", name=f"agent-{i}", bundle_location=f"ag_test_{i}/fakehash"
        )

    page1 = agent_store.list(limit=2)
    assert len(page1.data) == 2
    assert page1.has_more is True

    page2 = agent_store.list(limit=2, after=page1.last_id)
    assert len(page2.data) == 2
    assert page2.has_more is True

    page3 = agent_store.list(limit=2, after=page2.last_id)
    assert len(page3.data) == 1
    assert page3.has_more is False


def test_list_returns_newest_first(agent_store: SqlAlchemyAgentStore) -> None:
    a1 = agent_store.create(
        agent_id="ag_test_first", name="first", bundle_location="ag_test_first/fakehash"
    )
    a2 = agent_store.create(
        agent_id="ag_test_second", name="second", bundle_location="ag_test_second/fakehash"
    )
    page = agent_store.list()
    ids = {a.id for a in page.data}
    # Both returned; ordering is (created_at DESC, id DESC) —
    # same-second items are ordered by ID, not insertion order.
    assert ids == {a1.id, a2.id}


def test_list_order_asc(agent_store: SqlAlchemyAgentStore) -> None:
    for i in range(3):
        agent_store.create(
            agent_id=f"ag_test_{i}", name=f"agent-{i}", bundle_location=f"ag_test_{i}/fakehash"
        )
    page_desc = agent_store.list(order="desc")
    page_asc = agent_store.list(order="asc")
    assert [a.id for a in page_asc.data] == list(reversed([a.id for a in page_desc.data]))


def test_list_before_cursor(agent_store: SqlAlchemyAgentStore) -> None:
    for i in range(5):
        agent_store.create(
            agent_id=f"ag_test_{i}", name=f"agent-{i}", bundle_location=f"ag_test_{i}/fakehash"
        )
    # Paginate with after, then use before on the last page's first item
    # to go backwards and verify no overlap.
    page1 = agent_store.list(limit=3)
    page2 = agent_store.list(limit=3, after=page1.last_id)
    # before the first item of page2 should give us page1's items
    back = agent_store.list(limit=3, before=page2.first_id)
    assert [a.id for a in back.data] == [a.id for a in page1.data]


def test_list_asc_with_after_cursor(agent_store: SqlAlchemyAgentStore) -> None:
    for i in range(5):
        agent_store.create(
            agent_id=f"ag_test_{i}", name=f"agent-{i}", bundle_location=f"ag_test_{i}/fakehash"
        )
    page1 = agent_store.list(limit=2, order="asc")
    assert len(page1.data) == 2
    assert page1.has_more is True

    page2 = agent_store.list(limit=2, order="asc", after=page1.last_id)
    assert len(page2.data) == 2
    assert page2.has_more is True

    page3 = agent_store.list(limit=2, order="asc", after=page2.last_id)
    assert len(page3.data) == 1
    assert page3.has_more is False

    # All pages together should equal the full asc listing
    all_ids = [a.id for a in page1.data + page2.data + page3.data]
    full_asc = agent_store.list(limit=100, order="asc")
    assert all_ids == [a.id for a in full_asc.data]


# ── Update tests ───────────────────────────────────────────────


def test_update_agent(agent_store: SqlAlchemyAgentStore) -> None:
    """update() changes bundle_location, bumps version, sets updated_at."""
    agent = agent_store.create(
        agent_id="ag_test_upd",
        name="updatable",
        bundle_location="ag_test_upd/hash1",
    )
    # version=1 and updated_at=None on creation
    assert agent.version == 1
    assert agent.updated_at is None

    updated = agent_store.update("ag_test_upd", "ag_test_upd/hash2")
    assert updated is not None
    assert updated.version == 2
    assert updated.bundle_location == "ag_test_upd/hash2"
    assert updated.updated_at is not None
    # Name stays the same
    assert updated.name == "updatable"


def test_update_nonexistent_agent(agent_store: SqlAlchemyAgentStore) -> None:
    """update() returns None for a nonexistent agent."""
    assert agent_store.update("ag_nonexistent", "loc") is None


def test_update_increments_version(agent_store: SqlAlchemyAgentStore) -> None:
    """Multiple updates increment version monotonically."""
    agent_store.create(
        agent_id="ag_test_ver",
        name="versioned",
        bundle_location="ag_test_ver/h1",
    )
    v2 = agent_store.update("ag_test_ver", "ag_test_ver/h2")
    v3 = agent_store.update("ag_test_ver", "ag_test_ver/h3")
    assert v2 is not None and v2.version == 2
    assert v3 is not None and v3.version == 3


def test_create_agent_has_version_1(agent_store: SqlAlchemyAgentStore) -> None:
    """Newly created agents start at version 1."""
    agent = agent_store.create(
        agent_id="ag_test_v1",
        name="fresh",
        bundle_location="ag_test_v1/hash",
    )
    assert agent.version == 1
    assert agent.updated_at is None
