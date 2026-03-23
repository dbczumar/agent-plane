"""Tests for SqlAlchemyConversationStore."""

from __future__ import annotations

import pytest

from agent_plane.entities import (
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
    NewConversationItem,
    ReasoningData,
)
from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore

# ── CRUD ──────────────────────────────────────────────


def test_create_and_get(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    assert conv.id.startswith("conv_")

    fetched = conversation_store.get_conversation(conv.id)
    assert fetched is not None
    assert fetched.id == conv.id


def test_get_nonexistent(conversation_store: SqlAlchemyConversationStore) -> None:
    assert conversation_store.get_conversation("conv_none") is None


def test_update_title(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    updated = conversation_store.update_conversation(conv.id, title="Chat 1")
    assert updated is not None
    assert updated.title == "Chat 1"

    assert conversation_store.update_conversation("conv_none", title="x") is None


# ── Append & list items ──────────────────────────────


def test_append_and_list_items(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    items = conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_001",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "Hello"}]),
            ),
            NewConversationItem(
                type="message",
                response_id="resp_001",
                data=MessageData(
                    role="assistant",
                    content=[{"type": "output_text", "text": "Hi there!"}],
                    agent="test-agent",
                ),
            ),
        ],
    )
    assert len(items) == 2
    assert items[0].id.startswith("msg_")
    assert items[1].id.startswith("msg_")

    page = conversation_store.list_items(conv.id)
    assert len(page.data) == 2
    assert page.data[0].data.role == "user"
    assert page.data[1].data.role == "assistant"


def test_append_function_call_items(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    items = conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="function_call",
                response_id="resp_002",
                data=FunctionCallData(
                    agent="test-agent",
                    name="get_weather",
                    arguments='{"city": "SF"}',
                    call_id="call_001",
                ),
            ),
            NewConversationItem(
                type="function_call_output",
                response_id="resp_002",
                data=FunctionCallOutputData(
                    call_id="call_001",
                    output='{"temp": 65}',
                ),
            ),
        ],
    )
    assert items[0].id.startswith("fc_")
    assert items[1].id.startswith("fco_")


def test_append_reasoning_item(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    items = conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="reasoning",
                response_id="resp_003",
                data=ReasoningData(
                    agent="test-agent",
                    summary=[{"type": "summary_text", "text": "Thinking..."}],
                ),
            ),
        ],
    )
    assert items[0].id.startswith("rs_")


# ── Ordering & cursors ───────────────────────────────


def test_position_ordering(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_a",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "First"}]),
            ),
        ],
    )
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_b",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "Second"}]),
            ),
        ],
    )
    page = conversation_store.list_items(conv.id)
    assert len(page.data) == 2
    texts = [page.data[i].data.content[0]["text"] for i in range(2)]
    assert texts == ["First", "Second"]


def _make_5_items(conversation_store: SqlAlchemyConversationStore, conv_id: str):
    """Helper: append 5 messages and return the persisted items."""
    return conversation_store.append(
        conv_id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_x",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": f"msg-{i}"}],
                ),
            )
            for i in range(5)
        ],
    )


def test_list_items_after_cursor(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    items = _make_5_items(conversation_store, conv.id)

    page = conversation_store.list_items(conv.id, after=items[1].id, limit=2)
    assert len(page.data) == 2
    assert page.data[0].id == items[2].id


def test_list_items_desc_order(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    _make_5_items(conversation_store, conv.id)
    page_asc = conversation_store.list_items(conv.id, order="asc")
    page_desc = conversation_store.list_items(conv.id, order="desc")
    assert [it.id for it in page_asc.data] == list(reversed([it.id for it in page_desc.data]))


def test_list_items_desc_with_after_cursor(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """In desc order, 'after' means items with lower position."""
    conv = conversation_store.create_conversation()
    items = _make_5_items(conversation_store, conv.id)
    # desc full page: [4, 3, 2, 1, 0]
    page1 = conversation_store.list_items(conv.id, limit=2, order="desc")
    assert page1.data[0].id == items[4].id
    assert page1.data[1].id == items[3].id
    assert page1.has_more is True

    page2 = conversation_store.list_items(conv.id, limit=2, order="desc", after=page1.last_id)
    assert page2.data[0].id == items[2].id
    assert page2.data[1].id == items[1].id


def test_list_items_before_cursor(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    items = _make_5_items(conversation_store, conv.id)
    # asc order: [0, 1, 2, 3, 4]; before item[3] should give [0, 1, 2]
    page = conversation_store.list_items(conv.id, before=items[3].id, order="asc")
    assert [it.id for it in page.data] == [items[i].id for i in range(3)]


# ── Conversation ID / response ID lookups ────────────


def test_get_conversation_id(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_lookup",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "test"}]),
            ),
        ],
    )
    assert conversation_store.get_conversation_id("resp_lookup") == conv.id


def test_get_conversation_id_raises_for_missing(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    with pytest.raises(LookupError):
        conversation_store.get_conversation_id("resp_nonexistent")


def test_get_latest_response_id(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    assert conversation_store.get_latest_response_id(conv.id) is None

    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_first",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "a"}]),
            ),
        ],
    )
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_second",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "b"}]),
            ),
        ],
    )
    assert conversation_store.get_latest_response_id(conv.id) == "resp_second"


# ── Search (FTS) ─────────────────────────────────────


def test_search(conversation_store: SqlAlchemyConversationStore) -> None:
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_s1",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "weather in Paris"}],
                ),
            ),
            NewConversationItem(
                type="message",
                response_id="resp_s1",
                data=MessageData(
                    role="assistant",
                    content=[{"type": "output_text", "text": "sunny and warm"}],
                    agent="test-agent",
                ),
            ),
        ],
    )
    results = conversation_store.search("Paris")
    assert len(results) == 1
    assert results[0].type == "message"

    results = conversation_store.search("sunny")
    assert len(results) == 1

    assert conversation_store.search("nonexistent") == []


def test_search_scoped_to_conversation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv1 = conversation_store.create_conversation()
    conv2 = conversation_store.create_conversation()
    conversation_store.append(
        conv1.id,
        [
            NewConversationItem(
                type="message",
                response_id="r1",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hello world"}],
                ),
            ),
        ],
    )
    conversation_store.append(
        conv2.id,
        [
            NewConversationItem(
                type="message",
                response_id="r2",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hello universe"}],
                ),
            ),
        ],
    )
    # Unscoped: both match "hello"
    assert len(conversation_store.search("hello")) == 2

    # Scoped: only one per conversation
    assert len(conversation_store.search("hello", conversation_id=conv1.id)) == 1
    assert len(conversation_store.search("hello", conversation_id=conv2.id)) == 1


def test_search_function_call_item(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """FTS indexes function_call items by name and arguments."""
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="function_call",
                response_id="resp_fc",
                data=FunctionCallData(
                    agent="test-agent",
                    name="get_weather",
                    arguments='{"city": "Tokyo"}',
                    call_id="call_1",
                ),
            ),
        ],
    )
    assert len(conversation_store.search("get_weather")) == 1
    assert len(conversation_store.search("Tokyo")) == 1
    assert conversation_store.search("nonexistent") == []


def test_search_finds_try_deliver_messages(
    conversation_store: SqlAlchemyConversationStore,
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
) -> None:
    """Messages inserted via try_deliver are indexed for FTS."""
    agent = agent_store.create(name="search-agent")
    conv = conversation_store.create_conversation()
    task = task_store.create(conversation_id=conv.id, agent_id=agent.id, agent_name=agent.name)

    msg = NewConversationItem(
        type="message",
        response_id=task.id,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": "steered banana"}],
        ),
    )
    assert task_store.try_deliver(task.id, conv.id, msg) is True

    results = conversation_store.search("banana")
    assert len(results) == 1
    assert results[0].type == "message"


# ── Delete ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_conversation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_del",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "bye"}]),
            ),
        ],
    )
    assert await conversation_store.delete_conversation(conv.id) is True
    assert conversation_store.get_conversation(conv.id) is None
    assert conversation_store.list_items(conv.id).data == []
    assert await conversation_store.delete_conversation(conv.id) is False


@pytest.mark.asyncio
async def test_delete_conversation_with_tasks(
    conversation_store: SqlAlchemyConversationStore,
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
) -> None:
    agent = agent_store.create(name="a")
    conv = conversation_store.create_conversation()
    task_store.create(conversation_id=conv.id, agent_id=agent.id, agent_name=agent.name)
    task_store.create(conversation_id=conv.id, agent_id=agent.id, agent_name=agent.name)
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_x",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hi"}],
                ),
            ),
        ],
    )
    assert await conversation_store.delete_conversation(conv.id) is True
    assert conversation_store.get_conversation(conv.id) is None
    assert task_store.list_tasks(conversation_id=conv.id) == []


# ── List conversations pagination ────────────────────


def test_list_conversations_pagination(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    for _ in range(4):
        conversation_store.create_conversation()

    page1 = conversation_store.list_conversations(limit=2)
    assert len(page1.data) == 2
    assert page1.has_more is True

    page2 = conversation_store.list_conversations(limit=2, after=page1.last_id)
    assert len(page2.data) == 2
    assert page2.has_more is False


def test_list_conversations_order_asc(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    for _ in range(3):
        conversation_store.create_conversation()
    page_desc = conversation_store.list_conversations(order="desc")
    page_asc = conversation_store.list_conversations(order="asc")
    assert [c.id for c in page_asc.data] == list(reversed([c.id for c in page_desc.data]))


def test_list_conversations_asc_with_after_cursor(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    for _ in range(5):
        conversation_store.create_conversation()

    page1 = conversation_store.list_conversations(limit=2, order="asc")
    page2 = conversation_store.list_conversations(limit=2, order="asc", after=page1.last_id)
    page3 = conversation_store.list_conversations(limit=2, order="asc", after=page2.last_id)

    all_ids = [c.id for c in page1.data + page2.data + page3.data]
    full_asc = conversation_store.list_conversations(limit=100, order="asc")
    assert all_ids == [c.id for c in full_asc.data]
