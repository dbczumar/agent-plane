"""Integration tests for all SQLAlchemy store implementations."""

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
from agent_plane.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore

# ── Fixtures ───────────────────────────────────────────


@pytest.fixture()
def agent_store(db_uri: str) -> SqlAlchemyAgentStore:
    return SqlAlchemyAgentStore(db_uri)


@pytest.fixture()
def file_store(db_uri: str) -> SqlAlchemyFileStore:
    return SqlAlchemyFileStore(db_uri)


@pytest.fixture()
def conversation_store(db_uri: str) -> SqlAlchemyConversationStore:
    return SqlAlchemyConversationStore(db_uri)


@pytest.fixture()
def task_store(db_uri: str) -> SqlAlchemyTaskStore:
    return SqlAlchemyTaskStore(db_uri)


# ═══════════════════════════════════════════════════════
# AgentStore
# ═══════════════════════════════════════════════════════


class TestAgentStore:
    def test_create_and_get(self, agent_store: SqlAlchemyAgentStore) -> None:
        agent = agent_store.create(name="gpt-4")
        assert agent.id.startswith("ag_")
        assert agent.name == "gpt-4"

        fetched = agent_store.get(agent.id)
        assert fetched is not None
        assert fetched.id == agent.id
        assert fetched.name == "gpt-4"

    def test_get_nonexistent(self, agent_store: SqlAlchemyAgentStore) -> None:
        assert agent_store.get("ag_nonexistent") is None

    def test_get_by_name(self, agent_store: SqlAlchemyAgentStore) -> None:
        agent_store.create(name="claude")
        found = agent_store.get_by_name("claude")
        assert found is not None
        assert found.name == "claude"
        assert agent_store.get_by_name("missing") is None

    def test_create_with_description(self, agent_store: SqlAlchemyAgentStore) -> None:
        agent = agent_store.create(name="helper", description="A helper agent")
        assert agent.description == "A helper agent"

    def test_delete(self, agent_store: SqlAlchemyAgentStore) -> None:
        agent = agent_store.create(name="temp")
        assert agent_store.delete(agent.id) is True
        assert agent_store.get(agent.id) is None
        assert agent_store.delete(agent.id) is False

    def test_list_pagination(self, agent_store: SqlAlchemyAgentStore) -> None:
        for i in range(5):
            agent_store.create(name=f"agent-{i}")

        page1 = agent_store.list(limit=2)
        assert len(page1.data) == 2
        assert page1.has_more is True

        page2 = agent_store.list(limit=2, after=page1.last_id)
        assert len(page2.data) == 2
        assert page2.has_more is True

        page3 = agent_store.list(limit=2, after=page2.last_id)
        assert len(page3.data) == 1
        assert page3.has_more is False

    def test_list_returns_newest_first(self, agent_store: SqlAlchemyAgentStore) -> None:
        a1 = agent_store.create(name="first")
        a2 = agent_store.create(name="second")
        page = agent_store.list()
        ids = {a.id for a in page.data}
        # Both returned; ordering is (created_at DESC, id DESC) —
        # same-second items are ordered by ID, not insertion order.
        assert ids == {a1.id, a2.id}


# ═══════════════════════════════════════════════════════
# FileStore
# ═══════════════════════════════════════════════════════


class TestFileStore:
    def test_create_and_get(self, file_store: SqlAlchemyFileStore) -> None:
        f = file_store.create(filename="data.csv", bytes=1024)
        assert f.id.startswith("file_")
        assert f.filename == "data.csv"
        assert f.bytes == 1024

        fetched = file_store.get(f.id)
        assert fetched is not None
        assert fetched.filename == "data.csv"

    def test_get_nonexistent(self, file_store: SqlAlchemyFileStore) -> None:
        assert file_store.get("file_nonexistent") is None

    def test_create_with_content_type(self, file_store: SqlAlchemyFileStore) -> None:
        f = file_store.create(
            filename="img.png",
            bytes=2048,
            content_type="image/png",
        )
        assert f.content_type == "image/png"

    def test_delete(self, file_store: SqlAlchemyFileStore) -> None:
        f = file_store.create(filename="temp.txt", bytes=10)
        assert file_store.delete(f.id) is True
        assert file_store.get(f.id) is None
        assert file_store.delete(f.id) is False

    def test_list_pagination(self, file_store: SqlAlchemyFileStore) -> None:
        for i in range(4):
            file_store.create(filename=f"f{i}.txt", bytes=i)

        page1 = file_store.list(limit=2)
        assert len(page1.data) == 2
        assert page1.has_more is True

        page2 = file_store.list(limit=2, after=page1.last_id)
        assert len(page2.data) == 2
        assert page2.has_more is False


# ═══════════════════════════════════════════════════════
# ConversationStore
# ═══════════════════════════════════════════════════════


class TestConversationStore:
    def test_create_and_get(self, conversation_store: SqlAlchemyConversationStore) -> None:
        conv = conversation_store.create_conversation()
        assert conv.id.startswith("conv_")

        fetched = conversation_store.get_conversation(conv.id)
        assert fetched is not None
        assert fetched.id == conv.id

    def test_get_nonexistent(self, conversation_store: SqlAlchemyConversationStore) -> None:
        assert conversation_store.get_conversation("conv_none") is None

    def test_update_title(self, conversation_store: SqlAlchemyConversationStore) -> None:
        conv = conversation_store.create_conversation()
        updated = conversation_store.update_conversation(conv.id, title="Chat 1")
        assert updated is not None
        assert updated.title == "Chat 1"

        assert conversation_store.update_conversation("conv_none", title="x") is None

    def test_append_and_search_items(
        self, conversation_store: SqlAlchemyConversationStore
    ) -> None:
        conv = conversation_store.create_conversation()
        items = conversation_store.append(
            conv.id,
            [
                NewConversationItem(
                    type="message",
                    response_id="resp_001",
                    data=MessageData(
                        role="user", content=[{"type": "input_text", "text": "Hello"}]
                    ),
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

        page = conversation_store.search_items(conv.id)
        assert len(page.data) == 2
        assert page.data[0].data.role == "user"
        assert page.data[1].data.role == "assistant"

    def test_append_function_call_items(
        self, conversation_store: SqlAlchemyConversationStore
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

    def test_append_reasoning_item(self, conversation_store: SqlAlchemyConversationStore) -> None:
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

    def test_position_ordering(self, conversation_store: SqlAlchemyConversationStore) -> None:
        conv = conversation_store.create_conversation()
        conversation_store.append(
            conv.id,
            [
                NewConversationItem(
                    type="message",
                    response_id="resp_a",
                    data=MessageData(
                        role="user", content=[{"type": "input_text", "text": "First"}]
                    ),
                ),
            ],
        )
        conversation_store.append(
            conv.id,
            [
                NewConversationItem(
                    type="message",
                    response_id="resp_b",
                    data=MessageData(
                        role="user", content=[{"type": "input_text", "text": "Second"}]
                    ),
                ),
            ],
        )
        page = conversation_store.search_items(conv.id)
        assert len(page.data) == 2
        texts = [page.data[i].data.content[0]["text"] for i in range(2)]
        assert texts == ["First", "Second"]

    def test_search_items_after_cursor(
        self, conversation_store: SqlAlchemyConversationStore
    ) -> None:
        conv = conversation_store.create_conversation()
        items = conversation_store.append(
            conv.id,
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

        page = conversation_store.search_items(conv.id, after=items[1].id, limit=2)
        assert len(page.data) == 2
        assert page.data[0].id == items[2].id

    def test_get_conversation_id(self, conversation_store: SqlAlchemyConversationStore) -> None:
        conv = conversation_store.create_conversation()
        conversation_store.append(
            conv.id,
            [
                NewConversationItem(
                    type="message",
                    response_id="resp_lookup",
                    data=MessageData(
                        role="user", content=[{"type": "input_text", "text": "test"}]
                    ),
                ),
            ],
        )
        assert conversation_store.get_conversation_id("resp_lookup") == conv.id

    def test_get_conversation_id_raises_for_missing(
        self, conversation_store: SqlAlchemyConversationStore
    ) -> None:
        with pytest.raises(LookupError):
            conversation_store.get_conversation_id("resp_nonexistent")

    def test_get_latest_response_id(self, conversation_store: SqlAlchemyConversationStore) -> None:
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

    def test_search(self, conversation_store: SqlAlchemyConversationStore) -> None:
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
        self, conversation_store: SqlAlchemyConversationStore
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

    @pytest.mark.asyncio
    async def test_delete_conversation(
        self, conversation_store: SqlAlchemyConversationStore
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
        assert conversation_store.search_items(conv.id).data == []
        assert await conversation_store.delete_conversation(conv.id) is False

    @pytest.mark.asyncio
    async def test_delete_conversation_with_tasks(
        self,
        conversation_store: SqlAlchemyConversationStore,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
    ) -> None:
        agent = agent_store.create(name="a")
        conv = conversation_store.create_conversation()
        task_store.create(conversation_id=conv.id, agent_id=agent.id)
        task_store.create(conversation_id=conv.id, agent_id=agent.id)
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

    def test_list_conversations_pagination(
        self, conversation_store: SqlAlchemyConversationStore
    ) -> None:
        for _ in range(4):
            conversation_store.create_conversation()

        page1 = conversation_store.list_conversations(limit=2)
        assert len(page1.data) == 2
        assert page1.has_more is True

        page2 = conversation_store.list_conversations(limit=2, after=page1.last_id)
        assert len(page2.data) == 2
        assert page2.has_more is False


# ═══════════════════════════════════════════════════════
# TaskStore
# ═══════════════════════════════════════════════════════


class TestTaskStore:
    def _make_agent(self, agent_store: SqlAlchemyAgentStore) -> str:
        return agent_store.create(name="test-agent").id

    def _make_conversation(self, conversation_store: SqlAlchemyConversationStore) -> str:
        return conversation_store.create_conversation().id

    def test_create_and_get(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)

        task = task_store.create(
            conversation_id=conv_id,
            agent_id=agent_id,
            instructions="Be helpful",
            background=True,
        )
        assert task.task_id.startswith("resp_")
        assert task.conversation_id == conv_id
        assert task.agent_id == agent_id
        assert task.status == "queued"
        assert task.instructions == "Be helpful"
        assert task.background is True

        fetched = task_store.get(task.task_id)
        assert fetched is not None
        assert fetched.task_id == task.task_id

    def test_get_nonexistent(self, task_store: SqlAlchemyTaskStore) -> None:
        assert task_store.get("resp_nonexistent") is None

    def test_list_tasks_by_conversation(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv1 = self._make_conversation(conversation_store)
        conv2 = self._make_conversation(conversation_store)

        task_store.create(conversation_id=conv1, agent_id=agent_id)
        task_store.create(conversation_id=conv1, agent_id=agent_id)
        task_store.create(conversation_id=conv2, agent_id=agent_id)

        assert len(task_store.list_tasks(conversation_id=conv1)) == 2
        assert len(task_store.list_tasks(conversation_id=conv2)) == 1

    def test_list_tasks_by_agent(
        self,
        db_uri: str,
        task_store: SqlAlchemyTaskStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_store = SqlAlchemyAgentStore(db_uri)
        a1 = agent_store.create(name="agent-a").id
        a2 = agent_store.create(name="agent-b").id
        conv = self._make_conversation(conversation_store)

        task_store.create(conversation_id=conv, agent_id=a1)
        task_store.create(conversation_id=conv, agent_id=a2)

        assert len(task_store.list_tasks(agent_id=a1)) == 1
        assert len(task_store.list_tasks(agent_id=a2)) == 1

    @pytest.mark.asyncio
    async def test_delete(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)
        task = task_store.create(conversation_id=conv_id, agent_id=agent_id)

        await task_store.delete(task.task_id)
        assert task_store.get(task.task_id) is None

    def test_try_deliver_open_inbox(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)
        task = task_store.create(conversation_id=conv_id, agent_id=agent_id)

        msg = NewConversationItem(
            type="message",
            response_id=task.task_id,
            data=MessageData(
                role="user",
                content=[{"type": "input_text", "text": "steering msg"}],
            ),
        )
        assert task_store.try_deliver(task.task_id, conv_id, msg) is True

        items = conversation_store.search_items(conv_id)
        assert len(items.data) == 1
        assert items.data[0].data.content[0]["text"] == "steering msg"

    def test_try_deliver_closed_inbox(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)
        task = task_store.create(conversation_id=conv_id, agent_id=agent_id)

        task_store.close_inbox(task.task_id, conv_id, None)

        msg = NewConversationItem(
            type="message",
            response_id=task.task_id,
            data=MessageData(
                role="user",
                content=[{"type": "input_text", "text": "too late"}],
            ),
        )
        assert task_store.try_deliver(task.task_id, conv_id, msg) is False

    def test_close_inbox_no_new_messages(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)
        task = task_store.create(conversation_id=conv_id, agent_id=agent_id)

        late = task_store.close_inbox(task.task_id, conv_id, None)
        assert late == []

        fetched = task_store.get(task.task_id)
        assert fetched is not None
        assert fetched.inbox_closed is True

    def test_close_inbox_with_new_messages(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)
        task = task_store.create(conversation_id=conv_id, agent_id=agent_id)

        items = conversation_store.append(
            conv_id,
            [
                NewConversationItem(
                    type="message",
                    response_id=task.task_id,
                    data=MessageData(
                        role="user",
                        content=[{"type": "input_text", "text": "late msg"}],
                    ),
                ),
            ],
        )

        late = task_store.close_inbox(task.task_id, conv_id, None)
        assert len(late) == 1
        assert late[0].id == items[0].id

        fetched = task_store.get(task.task_id)
        assert fetched is not None
        assert fetched.inbox_closed is False

    def test_steering_handshake_sequence(
        self,
        task_store: SqlAlchemyTaskStore,
        agent_store: SqlAlchemyAgentStore,
        conversation_store: SqlAlchemyConversationStore,
    ) -> None:
        """Full steering handshake: deliver → agent sees it → close inbox."""
        agent_id = self._make_agent(agent_store)
        conv_id = self._make_conversation(conversation_store)
        task = task_store.create(conversation_id=conv_id, agent_id=agent_id)

        # Server delivers a steering message
        msg = NewConversationItem(
            type="message",
            response_id=task.task_id,
            data=MessageData(
                role="user",
                content=[{"type": "input_text", "text": "redirect"}],
            ),
        )
        assert task_store.try_deliver(task.task_id, conv_id, msg) is True

        # Agent tries to close inbox — sees the new message
        late = task_store.close_inbox(task.task_id, conv_id, None)
        assert len(late) == 1

        # Agent processes the message, then closes inbox with updated cursor
        late2 = task_store.close_inbox(task.task_id, conv_id, late[-1].id)
        assert late2 == []

        # Inbox is now closed — further deliveries fail
        msg2 = NewConversationItem(
            type="message",
            response_id=task.task_id,
            data=MessageData(
                role="user",
                content=[{"type": "input_text", "text": "too late"}],
            ),
        )
        assert task_store.try_deliver(task.task_id, conv_id, msg2) is False
