"""Tests for SqlAlchemyTaskStore."""

from __future__ import annotations

import asyncio
import io
import tarfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.entities import MessageData, NewConversationItem
from agent_plane.runtime import live_stream
from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore

# ── Helpers ──────────────────────────────────────────

_AGENT_NAME = "test-agent"


def _make_mock_streaming_client() -> MagicMock:
    """
    Build a mock OpenAI client whose ``responses.create()`` returns a
    one-event stream yielding a ``response.completed`` event with a
    single text output item.

    :returns: A MagicMock suitable for patching ``_get_openai_client``.
    """
    mock_content = MagicMock()
    mock_content.type = "output_text"
    mock_content.text = "Hello from the test LLM!"

    mock_output_item = MagicMock()
    mock_output_item.type = "message"
    mock_output_item.content = [mock_content]

    mock_response = MagicMock()
    mock_response.output = [mock_output_item]
    mock_response.model = "test-model"

    mock_completed_event = MagicMock()
    mock_completed_event.type = "response.completed"
    mock_completed_event.response = mock_response

    mock_client = MagicMock()
    # responses.create is called with stream=True; return an iterable
    # that yields one completed event so _accumulate_stream terminates.
    mock_client.responses.create.return_value = iter([mock_completed_event])
    return mock_client


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
    # Monkeypatch the OpenAI client so no real API call is made.
    mock_client = _make_mock_streaming_client()
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_openai_client",
        lambda: mock_client,
    )

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
    # Monkeypatch the OpenAI client so no real API call is made.
    mock_client = _make_mock_streaming_client()
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_openai_client",
        lambda: mock_client,
    )

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


@pytest.mark.asyncio
async def test_stream_closed_on_workflow_exception(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    artifact_store: LocalArtifactStore,
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the workflow raises mid-loop, the finally block still
    calls _close_output so the live stream's _DONE sentinel is
    pushed and SSE consumers don't hang forever.

    Without the fix (finally-block _close_output), subscribe()
    would block indefinitely on ``await queue.get()`` and this
    test would time out.
    """
    # Make the LLM call raise to simulate a mid-loop crash.
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = RuntimeError("simulated LLM timeout")
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_openai_client",
        lambda: mock_client,
    )

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

    # Register the live stream BEFORE starting the workflow so the
    # finally-block's _close_output can push _DONE to the queue.
    loop = asyncio.get_running_loop()
    live_stream.register(task.id, loop)

    try:
        task_store.start(task.id)

        # Drain the live stream. If the fix were missing, subscribe()
        # would hang forever — the 10 s timeout catches that.
        async def drain() -> list[dict[str, Any]]:
            """Collect all events until _DONE sentinel."""
            items: list[dict[str, Any]] = []
            async for event in live_stream.subscribe(task.id):
                items.append(event)
            return items

        events = await asyncio.wait_for(drain(), timeout=10.0)
        # subscribe() terminated — _DONE sentinel was received.
        # LLM raised before producing output, so no events.
        assert events == []
    finally:
        live_stream.unregister(task.id)
