"""Tests for agent_plane.runtime.workflow pagination helpers."""

from __future__ import annotations

import pytest

from agent_plane.entities import MessageData, NewConversationItem
from agent_plane.runtime.workflow import fetch_all_items
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


@pytest.fixture()
def conversation_store(db_uri: str) -> SqlAlchemyConversationStore:
    return SqlAlchemyConversationStore(db_uri)


def _make_user_message(index: int) -> NewConversationItem:
    """Build a simple user message item for testing."""
    return NewConversationItem(
        type="message",
        response_id="resp_001",
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": f"msg {index}"}],
        ),
    )


# ── fetch_all_items ──────────────────────────────────


def test_fetch_all_items_empty_conversation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    items = fetch_all_items(conversation_store, conv.id)
    assert items == []


def test_fetch_all_items_single_page(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Under the default limit of 100, all items come back in one page."""
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [_make_user_message(i) for i in range(5)],
    )
    items = fetch_all_items(conversation_store, conv.id)
    assert len(items) == 5


def test_fetch_all_items_paginates_beyond_limit(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    When a conversation has more items than the default page size
    (100), fetch_all_items must paginate through all pages.
    """
    conv = conversation_store.create_conversation()
    total = 150
    # Append in batches to keep individual appends manageable
    batch_size = 50
    for start in range(0, total, batch_size):
        conversation_store.append(
            conv.id,
            [_make_user_message(i) for i in range(start, start + batch_size)],
        )

    items = fetch_all_items(conversation_store, conv.id)
    assert len(items) == total

    # Verify ordering is preserved (ascending by position)
    texts = [item.data.content[0]["text"] for item in items]
    assert texts == [f"msg {i}" for i in range(total)]


def test_fetch_all_items_with_after_cursor(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    When given an after cursor, fetch_all_items only returns items
    after that cursor — and still paginates through all remaining pages.
    """
    conv = conversation_store.create_conversation()
    total = 150
    batch_size = 50
    for start in range(0, total, batch_size):
        conversation_store.append(
            conv.id,
            [_make_user_message(i) for i in range(start, start + batch_size)],
        )

    # Get the first page to grab a cursor from the middle
    first_page = conversation_store.list_items(conv.id, limit=50)
    cursor = first_page.last_id

    items = fetch_all_items(
        conversation_store,
        conv.id,
        after=cursor,
    )
    # Should get items 50..149 (the remaining 100)
    assert len(items) == 100

    texts = [item.data.content[0]["text"] for item in items]
    assert texts == [f"msg {i}" for i in range(50, total)]


def test_fetch_all_items_exactly_at_page_boundary(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    When item count equals the page size exactly, has_more is False
    and no extra page is fetched.
    """
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [_make_user_message(i) for i in range(100)],
    )
    items = fetch_all_items(conversation_store, conv.id)
    assert len(items) == 100
