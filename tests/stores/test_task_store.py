"""Tests for SqlAlchemyTaskStore."""

from __future__ import annotations

import io
import tarfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.entities import MessageData, NewConversationItem
from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore

# ── Helpers ──────────────────────────────────────────

_AGENT_NAME = "test-agent"

# Canned litellm response matching the shape of litellm.ModelResponse.model_dump()
_CANNED_LLM_RESPONSE: dict[str, Any] = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello from the test LLM!",
                "tool_calls": None,
            },
            "finish_reason": "stop",
            "index": 0,
        }
    ],
    "model": "test-model",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_agent_bundle() -> bytes:
    """Create a minimal valid agent tarball."""
    config_yaml = b"spec_version: 1\nllm:\n  model: test-model\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_yaml)
        tf.addfile(info, io.BytesIO(config_yaml))
    return buf.getvalue()


def _make_agent(
    agent_store: SqlAlchemyAgentStore,
    artifact_store: LocalArtifactStore | None = None,
) -> str:
    agent = agent_store.create(name=_AGENT_NAME)
    if artifact_store is not None:
        artifact_store.put(agent.id, _make_agent_bundle())
    return agent.id


def _make_conversation(conversation_store: SqlAlchemyConversationStore) -> str:
    return conversation_store.create_conversation().id


# ── CRUD ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_and_get(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)

    task = task_store.create(
        conversation_id=conv_id,
        agent_id=agent_id,
        agent_name=_AGENT_NAME,
        background=True,
    )
    assert task.id.startswith("resp_")
    assert task.conversation_id == conv_id
    assert task.agent_id == agent_id
    assert task.status == "queued"
    assert task.background is True
    # instructions/reasoning are not set by create() — they are
    # workflow inputs passed to start() and stored by DBOS.
    assert task.instructions is None
    assert task.reasoning is None

    fetched = await task_store.get(task.id)
    assert fetched is not None
    assert fetched.id == task.id


@pytest.mark.asyncio
async def test_get_nonexistent(task_store: SqlAlchemyTaskStore) -> None:
    assert await task_store.get("resp_nonexistent") is None


@pytest.mark.asyncio
async def test_list_tasks_by_conversation(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv1 = _make_conversation(conversation_store)
    conv2 = _make_conversation(conversation_store)

    task_store.create(conversation_id=conv1, agent_id=agent_id, agent_name=_AGENT_NAME)
    task_store.create(conversation_id=conv1, agent_id=agent_id, agent_name=_AGENT_NAME)
    task_store.create(conversation_id=conv2, agent_id=agent_id, agent_name=_AGENT_NAME)

    assert len(await task_store.list_tasks(conversation_id=conv1)) == 2
    assert len(await task_store.list_tasks(conversation_id=conv2)) == 1


@pytest.mark.asyncio
async def test_list_tasks_by_agent(
    db_uri: str,
    task_store: SqlAlchemyTaskStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_store = SqlAlchemyAgentStore(db_uri)
    a1 = agent_store.create(name="agent-a").id
    a2 = agent_store.create(name="agent-b").id
    conv = _make_conversation(conversation_store)

    task_store.create(conversation_id=conv, agent_id=a1, agent_name="agent-a")
    task_store.create(conversation_id=conv, agent_id=a2, agent_name="agent-b")

    assert len(await task_store.list_tasks(agent_id=a1)) == 1
    assert len(await task_store.list_tasks(agent_id=a2)) == 1


@pytest.mark.asyncio
async def test_delete(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)
    task = task_store.create(conversation_id=conv_id, agent_id=agent_id, agent_name=_AGENT_NAME)

    await task_store.delete(task.id)
    assert await task_store.get(task.id) is None


@pytest.mark.asyncio
async def test_delete_all_by_agent(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    a1 = agent_store.create(name="agent-a").id
    a2 = agent_store.create(name="agent-b").id
    conv = _make_conversation(conversation_store)

    task_store.create(conversation_id=conv, agent_id=a1, agent_name="agent-a")
    task_store.create(conversation_id=conv, agent_id=a1, agent_name="agent-a")
    task_store.create(conversation_id=conv, agent_id=a2, agent_name="agent-b")

    await task_store.delete_all(agent_id=a1)

    assert len(await task_store.list_tasks(agent_id=a1)) == 0
    # agent-b's task is untouched
    assert len(await task_store.list_tasks(agent_id=a2)) == 1


@pytest.mark.asyncio
async def test_delete_all_by_conversation(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv1 = _make_conversation(conversation_store)
    conv2 = _make_conversation(conversation_store)

    task_store.create(conversation_id=conv1, agent_id=agent_id, agent_name=_AGENT_NAME)
    task_store.create(conversation_id=conv1, agent_id=agent_id, agent_name=_AGENT_NAME)
    task_store.create(conversation_id=conv2, agent_id=agent_id, agent_name=_AGENT_NAME)

    await task_store.delete_all(conversation_id=conv1)

    assert len(await task_store.list_tasks(conversation_id=conv1)) == 0
    # conv2's task is untouched
    assert len(await task_store.list_tasks(conversation_id=conv2)) == 1


# ── Steering handshake ───────────────────────────────


def test_try_deliver_open_inbox(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)
    task = task_store.create(conversation_id=conv_id, agent_id=agent_id, agent_name=_AGENT_NAME)

    msg = NewConversationItem(
        type="message",
        response_id=task.id,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": "steering msg"}],
        ),
    )
    assert task_store.try_deliver(task.id, conv_id, msg) is True

    items = conversation_store.list_items(conv_id)
    assert len(items.data) == 1
    assert items.data[0].data.content[0]["text"] == "steering msg"


def test_try_deliver_closed_inbox(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)
    task = task_store.create(conversation_id=conv_id, agent_id=agent_id, agent_name=_AGENT_NAME)

    task_store.close_inbox(task.id, conv_id, None)

    msg = NewConversationItem(
        type="message",
        response_id=task.id,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": "too late"}],
        ),
    )
    assert task_store.try_deliver(task.id, conv_id, msg) is False


@pytest.mark.asyncio
async def test_close_inbox_no_new_messages(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)
    task = task_store.create(conversation_id=conv_id, agent_id=agent_id, agent_name=_AGENT_NAME)

    late = task_store.close_inbox(task.id, conv_id, None)
    assert late == []

    fetched = await task_store.get(task.id)
    assert fetched is not None
    assert fetched.inbox_closed is True


@pytest.mark.asyncio
async def test_close_inbox_with_new_messages(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)
    task = task_store.create(conversation_id=conv_id, agent_id=agent_id, agent_name=_AGENT_NAME)

    items = conversation_store.append(
        conv_id,
        [
            NewConversationItem(
                type="message",
                response_id=task.id,
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "late msg"}],
                ),
            ),
        ],
    )

    late = task_store.close_inbox(task.id, conv_id, None)
    assert len(late) == 1
    assert late[0].id == items[0].id

    fetched = await task_store.get(task.id)
    assert fetched is not None
    assert fetched.inbox_closed is False


def test_steering_handshake_sequence(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Full steering handshake: deliver -> agent sees it -> close inbox."""
    agent_id = _make_agent(agent_store)
    conv_id = _make_conversation(conversation_store)
    task = task_store.create(conversation_id=conv_id, agent_id=agent_id, agent_name=_AGENT_NAME)

    # Server delivers a steering message
    msg = NewConversationItem(
        type="message",
        response_id=task.id,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": "redirect"}],
        ),
    )
    assert task_store.try_deliver(task.id, conv_id, msg) is True

    # Agent tries to close inbox — sees the new message
    late = task_store.close_inbox(task.id, conv_id, None)
    assert len(late) == 1

    # Agent processes the message, then closes inbox with updated cursor
    late2 = task_store.close_inbox(task.id, conv_id, late[-1].id)
    assert late2 == []

    # Inbox is now closed — further deliveries fail
    msg2 = NewConversationItem(
        type="message",
        response_id=task.id,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": "too late"}],
        ),
    )
    assert task_store.try_deliver(task.id, conv_id, msg2) is False


# ── DBOS workflow integration ──────────────────────────


@pytest.mark.asyncio
async def test_start_and_get_completed(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    artifact_store: LocalArtifactStore,
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start() launches a DBOS workflow; get() reflects completion."""
    # Monkeypatch litellm.completion so no real API call is made.
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = _CANNED_LLM_RESPONSE
    monkeypatch.setattr("litellm.completion", lambda **kwargs: mock_resp)

    agent_id = _make_agent(agent_store, artifact_store)
    conv_id = _make_conversation(conversation_store)
    # The workflow loads history — seed with a user message.
    conversation_store.append(
        conv_id,
        [
            NewConversationItem(
                type="message",
                response_id="seed",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hi"}],
                ),
            )
        ],
    )
    task = task_store.create(
        conversation_id=conv_id,
        agent_id=agent_id,
        agent_name=_AGENT_NAME,
    )

    assert task.status == "queued"
    task_store.start(task.id, instructions="Be helpful")

    result = await task_store.wait(task.id)
    assert result.status == "completed"
    assert len(result.output) >= 1
    assert result.output[0]["role"] == "assistant"
    # instructions are stored in DBOS and restored by get()/wait()
    assert result.instructions == "Be helpful"


@pytest.mark.asyncio
async def test_wait_returns_completed_task(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    artifact_store: LocalArtifactStore,
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait() blocks until the workflow completes and returns the task."""
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = _CANNED_LLM_RESPONSE
    monkeypatch.setattr("litellm.completion", lambda **kwargs: mock_resp)

    agent_id = _make_agent(agent_store, artifact_store)
    conv_id = _make_conversation(conversation_store)
    conversation_store.append(
        conv_id,
        [
            NewConversationItem(
                type="message",
                response_id="seed",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "hello"}],
                ),
            )
        ],
    )
    task = task_store.create(
        conversation_id=conv_id,
        agent_id=agent_id,
        agent_name=_AGENT_NAME,
    )
    task_store.start(task.id, reasoning={"effort": "high"})

    result = await task_store.wait(task.id)
    assert result.status == "completed"
    assert len(result.output) >= 1
    assert result.reasoning == {"effort": "high"}
