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
from llms.types import (
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
)

# ── Helpers ──────────────────────────────────────────

_AGENT_NAME = "test-agent"


def _make_completed_event(
    text: str = "Hello from the test LLM!",
) -> ResponseCompletedEvent:
    """
    Build a ``ResponseCompletedEvent`` with a single text output
    item. Uses real ``llms.types`` dataclasses so ``isinstance``
    checks in ``_accumulate_stream`` pass.

    :param text: The assistant's reply text, e.g.
        ``"Hello from the test LLM!"``.
    :returns: A ``ResponseCompletedEvent`` ready to be yielded
        from a mock stream.
    """
    return ResponseCompletedEvent(
        response=Response(
            output=[
                MessageOutput(
                    content=[OutputText(text=text)],
                ),
            ],
            model="test-model",
        ),
    )


def _make_mock_streaming_client() -> MagicMock:
    """
    Build a mock OpenAI client whose ``responses.create()`` returns a
    one-event stream yielding a ``response.completed`` event with a
    single text output item.

    :returns: A MagicMock suitable for patching ``_get_llm_client``.
    """
    mock_client = MagicMock()
    # responses.create is called with stream=True; return an iterable
    # that yields one completed event so _accumulate_stream terminates.
    mock_client.responses.create.return_value = iter([_make_completed_event()])
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
        "agent_plane.runtime.workflow._get_llm_client",
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
    # Verify the mock's text actually made it through the full
    # pipeline (_accumulate_stream → _response_to_dict → persist).
    assert result.output[0]["content"][0]["text"] == "Hello from the test LLM!"
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
        "agent_plane.runtime.workflow._get_llm_client",
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
    assert result.output[0]["content"][0]["text"] == "Hello from the test LLM!"
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
        "agent_plane.runtime.workflow._get_llm_client",
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
        # The workflow emits a response.error SSE event for the
        # terminal LLM failure before closing the stream.
        assert len(events) == 1
        assert events[0]["type"] == "response.error"
        assert events[0]["source"] == "llm"
    finally:
        live_stream.unregister(task.id)


def _make_streaming_client_with_steering(
    conv_store: SqlAlchemyConversationStore,
    conv_id: str,
) -> MagicMock:
    """
    Build a mock OpenAI client that injects a steering message
    into the conversation on the FIRST ``responses.create()``
    call (simulating a user message arriving during LLM
    streaming). Returns different text on each call so the
    test can distinguish the first response from the follow-up.

    :param conv_store: ConversationStore to inject the
        steering message into.
    :param conv_id: Conversation ID to append the steering
        message to, e.g. ``"conv_abc123"``.
    :returns: A MagicMock suitable for patching
        ``_get_llm_client``.
    """
    call_count = 0

    def _fake_responses_create(**kwargs: Any) -> list[MagicMock]:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Simulate a steering message arriving while the
            # LLM is "streaming" — inject it into the
            # conversation so close_inbox will find it.
            conv_store.append(
                conv_id,
                [
                    NewConversationItem(
                        type="message",
                        # Use a distinct response_id to mark
                        # this as an external steering message
                        # (not the agent's own output).
                        response_id="user-steered",
                        data=MessageData(
                            role="user",
                            content=[
                                {
                                    "type": "input_text",
                                    "text": "new priority!",
                                }
                            ],
                        ),
                    )
                ],
            )
            text = "First response (before steering)"
        else:
            text = "Follow-up addressing steering"

        return iter([_make_completed_event(text)])

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = _fake_responses_create
    return mock_client


@pytest.mark.asyncio
async def test_persist_first_prevents_ghost_tokens(
    task_store: SqlAlchemyTaskStore,
    agent_store: SqlAlchemyAgentStore,
    artifact_store: LocalArtifactStore,
    conversation_store: SqlAlchemyConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When a steering message arrives during LLM streaming,
    the persist-first-then-check pattern ensures:
    1. The first assistant response is persisted (not ghost
       tokens that vanish).
    2. The LLM is called again to address the steering
       message.
    3. Both responses exist in the conversation.

    Before the fix, the first response would be discarded,
    leaving SSE consumers with ghost tokens that don't
    correspond to any persisted message.
    """
    agent_id = _make_agent(agent_store, artifact_store)
    conv_id = _make_conversation(conversation_store)
    # Seed with a user message so the workflow has input.
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

    mock_client = _make_streaming_client_with_steering(conversation_store, conv_id)
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_llm_client",
        lambda: mock_client,
    )

    task = task_store.create(
        conversation_id=conv_id,
        agent_id=agent_id,
        agent_name=_AGENT_NAME,
    )
    task_store.start(task.id)
    result = await task_store.wait(task.id)

    assert result.status == "completed"

    # The LLM should have been called twice: once for the
    # original request, once for the follow-up after steering.
    assert mock_client.responses.create.call_count == 2

    # Both assistant responses must be persisted in the
    # conversation — the first is NOT discarded.
    all_items = conversation_store.list_items(conv_id)
    assistant_texts = [
        item.data.content[0]["text"]
        for item in all_items.data
        if item.type == "message"
        and isinstance(item.data, MessageData)
        and item.data.role == "assistant"
    ]
    assert "First response (before steering)" in assistant_texts
    assert "Follow-up addressing steering" in assistant_texts

    # Verify no duplicate items in the conversation — each
    # message should appear exactly once. A broken last_seen
    # cursor would cause _sync_history to re-fetch the
    # assistant message, producing duplicates in the LLM
    # prompt (even if the store itself has no duplicates).
    all_ids = [item.id for item in all_items.data]
    assert len(all_ids) == len(set(all_ids)), f"Duplicate items in conversation: {all_ids}"

    # Verify the second LLM call's input contains the first
    # assistant response and the steering message — and that
    # each appears exactly once (no re-fetch duplicates).
    second_call_kwargs = mock_client.responses.create.call_args_list[1].kwargs
    second_call_input = second_call_kwargs["input"]
    assistant_in_prompt = [item for item in second_call_input if item.get("role") == "assistant"]
    user_in_prompt = [item for item in second_call_input if item.get("role") == "user"]
    # Exactly one assistant message (the first response)
    assert len(assistant_in_prompt) == 1
    # Content is now a list of content-block dicts (not a plain string)
    # after the multimodal pass-through change in history_to_input_items.
    # .get("text", "") handles non-text blocks (e.g. input_image) that
    # lack a "text" key — they contribute nothing to the joined string.
    assistant_content = assistant_in_prompt[0]["content"]
    assistant_text = " ".join(block.get("text", "") for block in assistant_content)
    assert "First response" in assistant_text
    # Two user messages: the seed + the steering message
    assert len(user_in_prompt) == 2

    # Verify the inbox is closed after completion — further
    # try_deliver calls must be rejected. Without the second
    # close_inbox call (to advance past own output), the inbox
    # would remain open and accept orphaned messages.
    fetched = await task_store.get(task.id)
    assert fetched is not None
    assert fetched.inbox_closed is True
