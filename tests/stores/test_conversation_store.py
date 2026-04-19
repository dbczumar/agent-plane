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


def test_unique_position_constraint(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    The (conversation_id, position) pair has a unique index.

    Verify that manually inserting a duplicate position raises
    IntegrityError, confirming the safety net is in place.
    """
    from sqlalchemy.exc import IntegrityError

    from agent_plane.db.db_models import SqlConversationItem
    from agent_plane.db.utils import generate_item_id

    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_dup",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "first"}],
                ),
            ),
        ],
    )
    # Directly insert a row at position 0 (already taken) to
    # confirm the unique constraint rejects it.
    with pytest.raises(IntegrityError):
        with conversation_store._session() as session:
            session.add(
                SqlConversationItem(
                    id=generate_item_id("message"),
                    conversation_id=conv.id,
                    response_id="resp_dup",
                    created_at=0,
                    status="completed",
                    position=0,  # duplicate
                    type="message",
                    data='{"role":"user","content":[]}',
                    search_text="",
                )
            )


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
    agent = agent_store.create(
        agent_id="ag_test_search", name="search-agent", bundle_location="ag_test_search/fakehash"
    )
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
    agent = agent_store.create(
        agent_id="ag_test_a", name="a", bundle_location="ag_test_a/fakehash"
    )
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
    assert await task_store.list_tasks(conversation_id=conv.id) == []


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


# ── list_items type filter ────────────────────────────


def test_list_items_type_filter_returns_only_matching_type(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    list_items(type=...) returns only items of the specified type,
    while list_items() without a filter returns all types.
    """
    from agent_plane.entities import CompactionData

    conv = conversation_store.create_conversation()

    # Append a mix of message and compaction items
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_001",
                data=MessageData(role="user", content=[{"type": "input_text", "text": "hi"}]),
            ),
        ],
    )
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="compaction",
                response_id="resp_001",
                data=CompactionData(
                    summary="Summary text",
                    last_item_id="msg_001",
                    model="openai/gpt-4o",
                    token_count=50,
                ),
            ),
        ],
    )
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_002",
                data=MessageData(
                    role="assistant",
                    content=[{"type": "output_text", "text": "hello"}],
                    agent="test-agent",
                ),
            ),
        ],
    )

    compaction_items = conversation_store.list_items(conv.id, type="compaction")
    message_items = conversation_store.list_items(conv.id, type="message")
    all_items = conversation_store.list_items(conv.id)

    # Only the one compaction item must be returned.
    assert len(compaction_items.data) == 1, (
        f"Expected 1 compaction item, got {len(compaction_items.data)}. "
        "Failure means type filter did not exclude message items."
    )
    assert compaction_items.data[0].type == "compaction"

    # Only message items (2) must be returned.
    assert len(message_items.data) == 2, (
        f"Expected 2 message items, got {len(message_items.data)}. "
        "Failure means type filter did not exclude the compaction item."
    )
    assert all(i.type == "message" for i in message_items.data)

    # No filter returns all 3 items.
    assert len(all_items.data) == 3, (
        f"Expected 3 total items (2 message + 1 compaction), got {len(all_items.data)}."
    )


def test_list_items_type_filter_with_order_and_limit(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    list_items(type="compaction", order="desc", limit=1) returns only
    the most recently appended compaction item.
    """
    from agent_plane.entities import CompactionData

    conv = conversation_store.create_conversation()

    # Append two compaction items
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="compaction",
                response_id="resp_001",
                data=CompactionData(
                    summary="First summary",
                    last_item_id="msg_010",
                    model="openai/gpt-4o",
                    token_count=100,
                ),
            ),
        ],
    )
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="compaction",
                response_id="resp_002",
                data=CompactionData(
                    summary="Second summary",
                    last_item_id="msg_020",
                    model="openai/gpt-4o",
                    token_count=120,
                ),
            ),
        ],
    )

    result = conversation_store.list_items(conv.id, type="compaction", order="desc", limit=1)

    # Only one item returned (limit=1).
    assert len(result.data) == 1, f"Expected 1 item with limit=1, got {len(result.data)}."
    # The most recent compaction item (second) should be returned (order=desc).
    assert result.data[0].data.summary == "Second summary", (
        f"Expected the latest compaction item with 'Second summary', "
        f"got: {result.data[0].data.summary!r}. "
        "Failure means order=desc with limit=1 did not return the newest item."
    )


# ── Sub-agent conversation isolation ────────────────


def test_subagent_conversations_are_isolated(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    Two sub-agent conversations created independently must have
    fully isolated item sets. list_items on one must never return
    items belonging to the other.

    This is the foundational invariant that prevents sub-agent
    "pollution": each sub-agent writes to its own conversation,
    and the agent loop loads history via
    ``list_items(conversation_id)`` — so items from sibling
    sub-agents are structurally invisible.

    A failure here means the WHERE clause on ``conversation_id``
    in ``list_items`` is broken, which would cause sub-agents to
    see each other's messages and produce incoherent LLM prompts.
    """
    conv_a = conversation_store.create_conversation(kind="sub_agent")
    conv_b = conversation_store.create_conversation(kind="sub_agent")

    # Append distinct items to each conversation.
    conversation_store.append(
        conv_a.id,
        [
            NewConversationItem(
                type="message",
                response_id="task_a",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "alpha input"}],
                ),
            ),
            NewConversationItem(
                type="message",
                response_id="task_a",
                data=MessageData(
                    role="assistant",
                    content=[{"type": "output_text", "text": "alpha output"}],
                    agent="researcher",
                ),
            ),
        ],
    )
    conversation_store.append(
        conv_b.id,
        [
            NewConversationItem(
                type="message",
                response_id="task_b",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "bravo input"}],
                ),
            ),
            NewConversationItem(
                type="message",
                response_id="task_b",
                data=MessageData(
                    role="assistant",
                    content=[{"type": "output_text", "text": "bravo output"}],
                    agent="researcher",
                ),
            ),
        ],
    )

    # List items for conv_a — must contain only alpha items.
    page_a = conversation_store.list_items(conv_a.id)
    # 2 items: user + assistant for the alpha sub-agent.
    assert len(page_a.data) == 2, (
        f"Expected 2 items in conv_a, got {len(page_a.data)}. "
        "If > 2, items from conv_b leaked into conv_a's listing."
    )
    texts_a = [item.data.content[0]["text"] for item in page_a.data]
    assert texts_a == ["alpha input", "alpha output"], (
        f"Expected alpha items only in conv_a, got {texts_a}. "
        "If bravo items appear, the conversation_id filter is broken."
    )
    # Every item must carry the correct response_id.
    for item in page_a.data:
        assert item.response_id == "task_a", (
            f"Item {item.id} in conv_a has response_id {item.response_id!r}, expected 'task_a'."
        )

    # List items for conv_b — must contain only bravo items.
    page_b = conversation_store.list_items(conv_b.id)
    # 2 items: user + assistant for the bravo sub-agent.
    assert len(page_b.data) == 2, (
        f"Expected 2 items in conv_b, got {len(page_b.data)}. "
        "If > 2, items from conv_a leaked into conv_b's listing."
    )
    texts_b = [item.data.content[0]["text"] for item in page_b.data]
    assert texts_b == ["bravo input", "bravo output"], (
        f"Expected bravo items only in conv_b, got {texts_b}. "
        "If alpha items appear, the conversation_id filter is broken."
    )
    for item in page_b.data:
        assert item.response_id == "task_b", (
            f"Item {item.id} in conv_b has response_id {item.response_id!r}, expected 'task_b'."
        )

    # Cross-check: item IDs must be disjoint.
    ids_a = {item.id for item in page_a.data}
    ids_b = {item.id for item in page_b.data}
    assert ids_a.isdisjoint(ids_b), (
        f"Item IDs overlap between conversations: "
        f"{ids_a & ids_b}. Each conversation must have "
        "unique item IDs."
    )


# ── updated_at ─────────────────────────────────────────


def test_create_sets_updated_at_equal_to_created_at(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    A newly created conversation has updated_at == created_at.
    """
    conv = conversation_store.create_conversation()
    assert conv.updated_at == conv.created_at, (
        f"Expected updated_at ({conv.updated_at}) to equal "
        f"created_at ({conv.created_at}) on a brand-new conversation."
    )


def test_append_bumps_updated_at(
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Appending items to a conversation advances updated_at
    to the current time.
    """
    import agent_plane.stores.conversation_store.sqlalchemy_store as store_mod

    # Freeze time at creation
    monkeypatch.setattr(store_mod, "now_epoch", lambda: 1000)
    conv = conversation_store.create_conversation()
    assert conv.updated_at == 1000

    # Advance time, then append
    monkeypatch.setattr(store_mod, "now_epoch", lambda: 2000)
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_bump",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hi"}],
                ),
            ),
        ],
    )
    fetched = conversation_store.get_conversation(conv.id)
    assert fetched is not None
    assert fetched.updated_at == 2000, (
        f"Expected updated_at to advance to 2000 after append, got {fetched.updated_at}."
    )


def test_update_title_bumps_updated_at(
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Updating the title of a conversation advances updated_at.
    """
    import agent_plane.stores.conversation_store.sqlalchemy_store as store_mod

    monkeypatch.setattr(store_mod, "now_epoch", lambda: 1000)
    conv = conversation_store.create_conversation()
    assert conv.updated_at == 1000

    monkeypatch.setattr(store_mod, "now_epoch", lambda: 3000)
    updated = conversation_store.update_conversation(conv.id, title="New title")
    assert updated is not None
    assert updated.updated_at == 3000, (
        f"Expected updated_at to advance to 3000 after title update, got {updated.updated_at}."
    )


# ── sort_by=updated_at ────────────────────────────────


def test_list_conversations_sort_by_updated_at(
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Sorting by updated_at returns conversations in order of
    last activity, not creation order.
    """
    import agent_plane.stores.conversation_store.sqlalchemy_store as store_mod

    # Create conv_a at t=100, conv_b at t=200
    monkeypatch.setattr(store_mod, "now_epoch", lambda: 100)
    conv_a = conversation_store.create_conversation()
    monkeypatch.setattr(store_mod, "now_epoch", lambda: 200)
    conv_b = conversation_store.create_conversation()

    # Append to conv_a at t=300, making it the most recently updated
    monkeypatch.setattr(store_mod, "now_epoch", lambda: 300)
    conversation_store.append(
        conv_a.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_sort",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hello"}],
                ),
            ),
        ],
    )

    # sort_by=created_at desc → conv_b first (created later)
    by_created = conversation_store.list_conversations(
        sort_by="created_at",
        order="desc",
        kind=None,
    )
    assert by_created.data[0].id == conv_b.id, (
        "Expected conv_b first when sorting by created_at desc."
    )

    # sort_by=updated_at desc → conv_a first (updated more recently)
    by_updated = conversation_store.list_conversations(
        sort_by="updated_at",
        order="desc",
        kind=None,
    )
    assert by_updated.data[0].id == conv_a.id, (
        "Expected conv_a first when sorting by updated_at desc, "
        "because it was updated at t=300 vs conv_b at t=200."
    )


def test_list_conversations_sort_by_updated_at_with_pagination(
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Cursor-based pagination works correctly when sorting
    by updated_at.
    """
    import agent_plane.stores.conversation_store.sqlalchemy_store as store_mod

    # Create 3 conversations with distinct updated_at values
    ids = []
    for t in (100, 200, 300):
        monkeypatch.setattr(store_mod, "now_epoch", lambda _t=t: _t)
        conv = conversation_store.create_conversation()
        ids.append(conv.id)

    # Reverse the update order: bump the oldest conversation last
    monkeypatch.setattr(store_mod, "now_epoch", lambda: 400)
    conversation_store.append(
        ids[0],
        [
            NewConversationItem(
                type="message",
                response_id="resp_pg",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "pg"}],
                ),
            ),
        ],
    )

    # sort_by=updated_at desc: ids[0] (400), ids[2] (300), ids[1] (200)
    page1 = conversation_store.list_conversations(
        limit=2,
        sort_by="updated_at",
        order="desc",
        kind=None,
    )
    # 2 results with has_more=True
    assert len(page1.data) == 2
    assert page1.has_more is True
    assert page1.data[0].id == ids[0]
    assert page1.data[1].id == ids[2]

    page2 = conversation_store.list_conversations(
        limit=2,
        sort_by="updated_at",
        order="desc",
        after=page1.last_id,
        kind=None,
    )
    # 1 result remaining
    assert len(page2.data) == 1
    assert page2.has_more is False
    assert page2.data[0].id == ids[1]


# ─── Phase 4: parent_conversation_id + name uniqueness ──────


def test_create_conversation_with_parent_pointer_and_title(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Setting ``parent_conversation_id`` + ``title`` round-trips through the row."""
    parent = conversation_store.create_conversation()
    child = conversation_store.create_conversation(
        kind="sub_agent",
        title="coder:auth",
        parent_conversation_id=parent.id,
    )
    # Both fields surface on the entity — proves the row was
    # populated AND the converter pulls the column. Without the
    # converter update, parent_conversation_id would always be
    # None on the returned entity even after the row stores it.
    fetched = conversation_store.get_conversation(child.id)
    assert fetched is not None
    assert fetched.title == "coder:auth"
    assert fetched.parent_conversation_id == parent.id
    assert fetched.kind == "sub_agent"


def test_create_duplicate_title_under_same_parent_raises(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """G36: partial unique index rejects ``(parent_id, title)`` duplicates."""
    from agent_plane.stores.conversation_store import NameAlreadyExistsError

    parent = conversation_store.create_conversation()
    conversation_store.create_conversation(
        kind="sub_agent",
        title="coder:auth",
        parent_conversation_id=parent.id,
    )
    # Without the partial unique index + IntegrityError-to-
    # NameAlreadyExistsError translation, the second create
    # would either succeed silently (creating a duplicate row)
    # or raise a raw sqlalchemy IntegrityError that would leak
    # through to the LLM as an opaque error.
    with pytest.raises(NameAlreadyExistsError):
        conversation_store.create_conversation(
            kind="sub_agent",
            title="coder:auth",
            parent_conversation_id=parent.id,
        )


def test_create_same_title_under_different_parents_succeeds(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """The unique constraint is per-parent — ``(p1, "auth")`` and ``(p2, "auth")`` coexist."""
    p1 = conversation_store.create_conversation()
    p2 = conversation_store.create_conversation()
    conversation_store.create_conversation(
        kind="sub_agent", title="coder:auth", parent_conversation_id=p1.id
    )
    # Same title, different parent — no conflict.
    conversation_store.create_conversation(
        kind="sub_agent", title="coder:auth", parent_conversation_id=p2.id
    )
    # Both children must exist; if the unique constraint were
    # global (not partial-by-parent), the second create would
    # raise.
    p1_children = conversation_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=p1.id,
    )
    p2_children = conversation_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=p2.id,
    )
    assert len(p1_children.data) == 1
    assert len(p2_children.data) == 1
    assert p1_children.data[0].title == "coder:auth"
    assert p2_children.data[0].title == "coder:auth"


def test_create_null_parent_allows_duplicate_titles(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Top-level conversations (NULL parent) are NOT subject to the unique constraint."""
    # Both conversations share title=None and parent=None.
    # The partial index excludes NULL parents, so two NULL-NULL
    # rows are valid. Without the WHERE clause on the index,
    # this would raise.
    a = conversation_store.create_conversation()
    b = conversation_store.create_conversation()
    assert a.id != b.id
    assert a.parent_conversation_id is None
    assert b.parent_conversation_id is None


def test_list_conversations_filtered_by_parent_returns_children_only(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """``parent_conversation_id`` filter scopes results to one parent's sub-tree."""
    parent_a = conversation_store.create_conversation()
    parent_b = conversation_store.create_conversation()
    conversation_store.create_conversation(
        kind="sub_agent", title="coder:auth", parent_conversation_id=parent_a.id
    )
    conversation_store.create_conversation(
        kind="sub_agent", title="coder:payments", parent_conversation_id=parent_a.id
    )
    conversation_store.create_conversation(
        kind="sub_agent", title="coder:other", parent_conversation_id=parent_b.id
    )

    page = conversation_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=parent_a.id,
    )
    # Exactly 2 children for parent_a — proves the WHERE clause
    # excludes parent_b's child. If the filter were a no-op,
    # all 3 sub-agent rows would appear.
    titles = sorted(c.title for c in page.data if c.title)
    assert titles == ["coder:auth", "coder:payments"]


def test_cascade_delete_removes_descendants(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Deleting a parent recursively removes children + grandchildren (FK CASCADE)."""
    import asyncio

    parent = conversation_store.create_conversation()
    child = conversation_store.create_conversation(
        kind="sub_agent", title="coder:auth", parent_conversation_id=parent.id
    )
    grandchild = conversation_store.create_conversation(
        kind="sub_agent", title="reviewer:nested", parent_conversation_id=child.id
    )
    # Delete the root — both descendants must vanish via the
    # ON DELETE CASCADE on parent_conversation_id.
    asyncio.run(conversation_store.delete_conversation(parent.id))
    assert conversation_store.get_conversation(parent.id) is None
    assert conversation_store.get_conversation(child.id) is None, (
        "Child not cascaded — FK ondelete=CASCADE missing or migration didn't apply it"
    )
    assert conversation_store.get_conversation(grandchild.id) is None, (
        "Grandchild not cascaded — recursive FK cascade missing"
    )
