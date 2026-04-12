"""Concurrency integration tests.

In-process tests use ControllableMockClient's blocking gates to
create deterministic race windows within one server. The
cross-server test launches real ``ap server`` subprocesses
sharing a database and uses a mock LLM HTTP server as the gate.

No ``time.sleep`` — synchronization is purely event-driven
via ``MockCall.call_event`` / ``MockCall.release()`` (in-process)
or ``/gate/pending`` / ``/gate/release`` (cross-server).
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from agent_plane.entities.conversation import (
    ConversationItem,
    MessageData,
    NewConversationItem,
)
from agent_plane.stores.task_store.sqlalchemy_store import (
    SqlAlchemyTaskStore,
)
from tests.server.conftest import ControllableMockClient, MockCall
from tests.server.helpers import (
    build_agent_bundle,
    create_test_agent,
    create_test_response,
)

pytestmark = pytest.mark.asyncio


# ── Steering Races ───────────────────────────────────────


async def test_steering_delivers_to_running_workflow(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message into an active workflow via try_deliver.

    Race window: workflow is blocked in the LLM call. The
    HTTP request checks prev_task.status (sees IN_PROGRESS),
    calls try_deliver() which atomically checks inbox_closed
    (False) and appends the steered message.

    Breakage this catches:
    - try_deliver fails to insert the message
    - Steering returns a different response ID (new task)
    - Steered message not persisted in conversation items
    - Workflow does not continue with steered message (missing
      second LLM call)
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="First response", block=True)
    # Second call: workflow continues with steered context
    mock_llm.add_call(text="Steered response")

    first = await create_test_response(client, input_text="Hello")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]
    assert first.body["status"] == "queued"

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action: steer while workflow is blocked
    second = await create_test_response(
        client,
        input_text="Change direction",
        previous_response_id=first_id,
    )
    # Steering returns the SAME response (not a new task)
    assert second.body["id"] == first_id

    # Steered message is persisted in the conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts
    assert "Change direction" in user_texts

    # Release: workflow continues with steered message, then completes
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break

    assert resp.json()["status"] == "completed"
    # Steering triggers a second LLM call: the workflow detects
    # the steered message in close_inbox and continues the agent
    # loop with the updated conversation history.
    assert mock_llm.call_count == 2


async def test_steering_preserves_position_order(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steered message gets the correct position between the
    original user message and the assistant response.

    Race window: try_deliver inserts at MAX(position)+1
    while the workflow is blocked before persisting its
    own assistant message. After release, the assistant
    message gets MAX(position)+1 which is after the
    steered message.

    Breakage this catches:
    - Position collision (steered and assistant at same pos)
    - Wrong ordering (assistant before steered message)
    - Steered message lost (not in items at all)
    - Follow-up assistant response missing after steering
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="The answer", block=True)
    # Follow-up call after steering is detected
    mock_llm.add_call(text="Follow-up answer")

    first = await create_test_response(
        client,
        input_text="Question 1",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action: steer while workflow is blocked
    await create_test_response(
        client,
        input_text="Steering message",
        previous_response_id=first_id,
    )

    # Release: workflow persists first assistant, detects
    # steered message, continues with second LLM call
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify position ordering: user, steered, assistant,
    # follow-up assistant (from steering continuation)
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    assert len(items) == 4, f"Expected 4 items, got {len(items)}: {items}"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Question 1"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Steering message"
    assert items[2]["role"] == "assistant"
    assert items[2]["content"][0]["text"] == "The answer"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "Follow-up answer"


async def test_multiple_steering_messages_while_blocked(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two steering messages delivered while the workflow is
    blocked in the LLM call. Both must be persisted in
    correct position order.

    Race window: both try_deliver calls acquire the
    conversation lock sequentially (serialized by FOR
    UPDATE / SQLite locking), each computing MAX(position)+1.

    Breakage this catches:
    - Position collision between two steered messages
    - Second try_deliver fails (inbox closed prematurely)
    - Messages persisted out of order
    - Workflow does not continue with steered messages
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(
        text="First answer",
        block=True,
    )
    # Follow-up call after both steered messages are detected
    mock_llm.add_call(text="Final answer")

    first = await create_test_response(
        client,
        input_text="Original question",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action 1: first steering message
    steer_1 = await create_test_response(
        client,
        input_text="Clarification A",
        previous_response_id=first_id,
    )
    assert steer_1.body["id"] == first_id

    # Concurrent action 2: second steering message
    steer_2 = await create_test_response(
        client,
        input_text="Clarification B",
        previous_response_id=first_id,
    )
    assert steer_2.body["id"] == first_id

    # Release: workflow persists first answer, detects
    # both steered messages, continues with second LLM call
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify all 5 items: user, 2 steered, first assistant,
    # follow-up assistant
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    assert len(items) == 5, (
        f"Expected 5 items (user + 2 steered + 2 assistant), got {len(items)}: {items}"
    )
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Original question"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Clarification A"
    assert items[2]["role"] == "user"
    assert items[2]["content"][0]["text"] == "Clarification B"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "First answer"
    assert items[4]["role"] == "assistant"
    assert items[4]["content"][0]["text"] == "Final answer"

    # 2 LLM calls: original + continuation after detecting steered messages
    assert mock_llm.call_count == 2


async def test_steering_during_tool_execution(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message while the workflow is executing tool calls.

    Race window: the LLM returns a tool call. While the tool is
    executing, a steered message arrives via try_deliver with a
    position interleaved among the tool call items. After tool
    execution, the workflow must detect the steered message and
    include it in the next LLM call.

    Breakage this catches:
    - Steered message lost because post-tool last_seen cursor
      skips over interleaved positions
    - Workflow ignores steered input during tool execution
    - Second LLM call missing (no continuation after tools)
    """
    await create_test_agent(client)

    # Call 1: returns a tool call (blocks so we can steer
    # during tool execution)
    tool_call_spec = [
        {
            "call_id": "call_steer_tool_1",
            "name": "load_skill",
            "arguments": '{"name": "nonexistent"}',
        },
    ]
    call_1 = mock_llm.add_call(tool_calls=tool_call_spec, block=True)
    # Call 2: after tool execution, the LLM is called again
    # with tool results. This also blocks so we can steer.
    call_2 = mock_llm.add_call(text="Post-tool answer", block=True)
    # Call 3: continuation after steering is detected
    mock_llm.add_call(text="Steered answer")

    first = await create_test_response(
        client,
        input_text="Use a tool",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate 1: workflow is blocked before returning tool calls
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)
    call_1.release()

    # Gate 2: tool executed, LLM called again with results,
    # now blocked in second LLM call
    await asyncio.wait_for(call_2.call_event.wait(), timeout=10)

    # Concurrent action: steer while blocked in second LLM call
    # (tool results are already persisted at this point)
    steer = await create_test_response(
        client,
        input_text="Change direction mid-tools",
        previous_response_id=first_id,
    )
    assert steer.body["id"] == first_id

    # Release: second LLM call completes, workflow detects
    # steered message, continues with third LLM call
    call_2.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify steered message is in the conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i.get("role") == "user"]
    assert "Use a tool" in user_texts
    assert "Change direction mid-tools" in user_texts

    # 3 LLM calls: tool call + post-tool + steering continuation
    assert mock_llm.call_count == 3


async def test_steer_during_handle_final_response_creates_new_task(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Steer arrives while _handle_final_response is closing the inbox.

    Simulates the archer deep-research scenario: the LLM call for
    the first steer's continuation completes, _handle_final_response
    closes the inbox, and the second steer arrives moments later.
    Because inbox_closed is already True, try_deliver fails and the
    route creates a new task (status "queued").

    This is NOT a bug — it's the expected behavior when steering
    arrives after the workflow completes. The test documents this
    edge case and verifies the new task correctly continues the
    conversation.

    Breakage this catches:
    - New task for late steer doesn't share the conversation
    - New task doesn't include the steered message in history
    - Server errors when steering into a just-completed task
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="Monkey answer", block=True)
    # Second call: leopard continuation
    mock_llm.add_call(text="Leopard answer")
    # Third call: moose gets its own task (new workflow)
    mock_llm.add_call(text="Moose answer")

    first = await create_test_response(
        client,
        input_text="Research monkeys",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow blocked in LLM call 1
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Steer A: leopard delivered during LLM call 1
    steer_a = await create_test_response(
        client,
        input_text="Research leopards instead",
        previous_response_id=first_id,
    )
    assert steer_a.body["id"] == first_id

    # Release LLM call 1 → _SteeringRetry → LLM call 2 →
    # completes (no blocking) → inbox closes
    call_1.release()
    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Steer B: moose arrives AFTER the task completed.
    # try_deliver will fail (inbox closed), creating a new task.
    steer_b = await create_test_response(
        client,
        input_text="Research moose instead",
        previous_response_id=first_id,
    )
    moose_id = steer_b.body["id"]
    # New task created — different ID from the original
    assert moose_id != first_id, (
        "Moose steer should have created a new task (original "
        "task already completed with inbox closed)"
    )

    # The new task shares the SAME conversation
    assert steer_b.body["conversation"]["id"] == conv_id, (
        "New task must continue in the same conversation"
    )

    # Wait for moose task to complete
    for _ in range(50):
        resp = await client.get(f"/v1/responses/{moose_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # All messages are in the shared conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Research monkeys" in user_texts
    assert "Research leopards instead" in user_texts
    assert "Research moose instead" in user_texts


async def test_chained_steering_across_iterations(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Three-turn chained steering: original → steer A → steer B.

    Reproduces: user sends request, steers once (during LLM call 1),
    then steers again (during LLM call 2). Both steers must deliver
    to the SAME task (same response ID), and all three topics must
    appear in the conversation with the agent addressing each via
    a separate LLM call.

    Race window: each steer arrives while the workflow is blocked
    in a different LLM call. The workflow detects each steered
    message in _handle_final_response and continues the loop.

    Breakage this catches:
    - Second steer creates a new task instead of delivering to
      the existing one (inbox prematurely closed)
    - Second steered message not detected (skipped by cursor)
    - Workflow completes after 2 LLM calls instead of 3
    """
    await create_test_agent(client)

    # 3 blocking LLM calls: original, steer A continuation,
    # steer B continuation
    call_1 = mock_llm.add_call(text="Monkey answer", block=True)
    call_2 = mock_llm.add_call(text="Leopard answer", block=True)
    call_3 = mock_llm.add_call(text="Moose answer", block=True)

    first = await create_test_response(
        client,
        input_text="Research monkeys",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate 1: workflow blocked in LLM call 1
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Steer A: deliver "leopard" while LLM call 1 is in progress
    steer_a = await create_test_response(
        client,
        input_text="Research leopards instead",
        previous_response_id=first_id,
    )
    # Must return the SAME response (steered, not new task)
    assert steer_a.body["id"] == first_id, (
        "First steer created a new task instead of delivering to the running workflow"
    )

    # Release LLM call 1 → _handle_final_response detects
    # leopard → _SteeringRetry → loop continues → LLM call 2
    call_1.release()
    await asyncio.wait_for(call_2.call_event.wait(), timeout=10)

    # Steer B: deliver "moose" while LLM call 2 is in progress.
    # This is the critical test: the inbox must still be open
    # after the first _SteeringRetry.
    steer_b = await create_test_response(
        client,
        input_text="Research moose instead",
        previous_response_id=first_id,
    )
    # KEY ASSERTION: second steer must also return the same
    # response ID. If it returns a different ID, the inbox was
    # prematurely closed after the first steering cycle.
    assert steer_b.body["id"] == first_id, (
        f"Second steer created a new task ({steer_b.body['id']}) "
        f"instead of delivering to {first_id} — inbox was "
        f"prematurely closed after the first steering cycle"
    )

    # Release LLM call 2 → detects moose → _SteeringRetry →
    # LLM call 3
    call_2.release()
    await asyncio.wait_for(call_3.call_event.wait(), timeout=10)

    # Release LLM call 3 → completes
    call_3.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify all 6 items: 3 user messages + 3 assistant responses
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assistant_texts = [i["content"][0]["text"] for i in items if i["role"] == "assistant"]
    assert "Research monkeys" in user_texts
    assert "Research leopards instead" in user_texts
    assert "Research moose instead" in user_texts
    assert "Monkey answer" in assistant_texts
    assert "Leopard answer" in assistant_texts
    assert "Moose answer" in assistant_texts

    # 3 LLM calls: original + 2 steering continuations
    assert mock_llm.call_count == 3, (
        f"Expected 3 LLM calls (original + 2 steering continuations), got {mock_llm.call_count}"
    )


async def test_steering_between_persist_and_close_inbox(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    task_store: SqlAlchemyTaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Steering message arrives between Step 2 (_check_steering_inbox)
    and Step 3 (close_inbox with persisted cursor) in
    _handle_final_response.

    Race window: _check_steering_inbox returns empty (no steered
    messages detected). Between that return and the second
    close_inbox call, a steering message is delivered via
    try_deliver. The second close_inbox must detect it and trigger
    _SteeringRetry.

    Before the fix: the second close_inbox return value was
    ignored, so the workflow completed after 1 LLM call without
    ever addressing the late steering message.

    After the fix: the second close_inbox return value is checked,
    _SteeringRetry fires, and a second LLM call processes the
    steered message.

    Breakage this catches:
    - Second close_inbox return value ignored (the exact bug)
    - Late steering message silently dropped
    - Workflow completes without addressing steered input
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="First response", block=True)
    # Second LLM call: the steering continuation. Named so we can
    # inspect received_kwargs to verify the steered message was
    # included in the LLM input.
    call_2 = mock_llm.add_call(text="Steered response")

    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # NOTE: Monkeypatching _check_steering_inbox (a private function)
    # is necessary because the race window between Step 2 and Step 3
    # in _handle_final_response is sub-millisecond and cannot be
    # triggered through public APIs alone. This is the same pattern
    # used by the test fixtures to monkeypatch _get_llm_client.
    import agent_plane.runtime.workflow as workflow_mod

    original_check = workflow_mod._check_steering_inbox
    injected = threading.Event()

    def check_then_inject(
        t_id: str,
        c_id: str,
        last_seen: str | None,
        persisted: list[ConversationItem],
        ts: SqlAlchemyTaskStore,
        **kwargs: object,
    ) -> list[ConversationItem]:
        """
        Run the real _check_steering_inbox, then inject a
        steering message before returning.

        :param t_id: Task ID.
        :param c_id: Conversation ID.
        :param last_seen: Pre-LLM cursor.
        :param persisted: Items just persisted.
        :param ts: The TaskStore.
        :returns: Result from original check (empty list).
        """
        result = original_check(
            t_id,
            c_id,
            last_seen,
            persisted,
            ts,
        )
        # Inject exactly once: on the first call within
        # _handle_final_response (the no-tool-calls path).
        if not injected.is_set():
            injected.set()
            steering_item = NewConversationItem(
                type="message",
                response_id=t_id,
                data=MessageData(
                    role="user",
                    content=[
                        {
                            "type": "input_text",
                            "text": "Late steering",
                        }
                    ],
                ),
            )
            delivered = task_store.try_deliver(
                t_id,
                c_id,
                steering_item,
            )
            assert delivered, (
                "try_deliver returned False — inbox was already closed before the injection point"
            )
        return result

    monkeypatch.setattr(
        workflow_mod,
        "_check_steering_inbox",
        check_then_inject,
    )

    # Release: workflow enters _handle_final_response, Step 2
    # finds nothing, injection delivers steering message, Step 3
    # must detect it.
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify the late steering message is in the conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts
    assert "Late steering" in user_texts, (
        "Late steering message was delivered but not found in "
        "conversation items — it was silently dropped"
    )

    # KEY ASSERTION 1: 2 LLM calls prove the workflow retried.
    # Before the fix: 1 call (second close_inbox return ignored,
    # workflow completed without addressing the steering message).
    # After the fix: 2 calls (second close_inbox detected the
    # message, returned _SteeringRetry, agent loop continued).
    assert mock_llm.call_count == 2, (
        f"Expected 2 LLM calls (original + steering continuation)"
        f", got {mock_llm.call_count}. If 1, the second "
        f"close_inbox return value is being ignored."
    )

    # KEY ASSERTION 2: The second LLM call's input includes the
    # late steering message. This proves _SteeringRetry extended
    # history with the steered content, not just retried blindly.
    assert call_2.received_kwargs is not None, "Second LLM call was never made"
    llm_input = str(call_2.received_kwargs["input"])
    assert "Late steering" in llm_input, (
        f"Second LLM call input does not contain the steered "
        f"message — _SteeringRetry fired but history was not "
        f"extended with the late message. Input: {llm_input}"
    )


async def test_steering_during_streaming_processed_after_complete(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steering delivered during a streaming LLM call (simulating
    native tools like web_search) is processed after the
    stream completes.

    Race window: the LLM is streaming tokens (stream_tokens=True).
    A steered message arrives via try_deliver while the workflow is
    inside _accumulate_stream. Because _accumulate_stream has no
    mid-stream inbox check, the steering message sits in the inbox
    until the stream finishes and _handle_final_response runs.

    This documents the architectural limitation that caused the
    archer deep-research bug: native tool streams cannot be
    interrupted by steering. The message is eventually processed
    (not lost) but only after the entire stream completes.

    Breakage this catches:
    - Steering message lost during streaming (never detected)
    - _handle_final_response fails to detect messages delivered
      during a streaming (vs non-streaming) LLM call
    - Second LLM call missing after streaming + steering
    """
    await create_test_agent(client)

    # stream_tokens=True: the mock yields individual word deltas
    # before the completed event, simulating a streaming LLM call
    # with intermediate events (like native tool output).
    call_1 = mock_llm.add_call(
        text="Deep research results about monkeys",
        block=True,
        stream_tokens=True,
    )
    # Steering continuation: the second LLM call after steering
    # is detected in _handle_final_response.
    call_2 = mock_llm.add_call(text="Leopard research results")

    first = await create_test_response(
        client,
        input_text="Research monkeys",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is blocked before the stream starts
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action: steer while workflow is blocked.
    # In production, this would arrive mid-stream (during
    # _accumulate_stream). Here, the steer is delivered before
    # the stream starts — functionally equivalent because
    # _accumulate_stream never checks the inbox regardless.
    steer = await create_test_response(
        client,
        input_text="Switch to leopards",
        previous_response_id=first_id,
    )
    assert steer.body["id"] == first_id

    # Release: stream runs to completion (all tokens emitted),
    # then _handle_final_response detects the steered message
    # and triggers _SteeringRetry.
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify steering message is in conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Research monkeys" in user_texts
    assert "Switch to leopards" in user_texts

    # 2 LLM calls: streaming call + steering continuation.
    # The streaming path (stream_tokens=True) is handled
    # identically to non-streaming for steering detection.
    assert mock_llm.call_count == 2, (
        f"Expected 2 LLM calls (streaming + steering continuation), got {mock_llm.call_count}"
    )

    # Second LLM call received the steered message in its input
    assert call_2.received_kwargs is not None, "Second LLM call was never made"
    llm_input = str(call_2.received_kwargs["input"])
    assert "Switch to leopards" in llm_input, (
        f"Second LLM call input does not contain the steered "
        f"message — steering was detected but history was not "
        f"extended. Input: {llm_input}"
    )


async def test_steering_during_llm_retry(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Steering delivered while the LLM call is being retried after
    a transient error (HTTP 429).

    Race window: the first LLM attempt fails with a retryable
    error. execute_with_retry sleeps (mocked to zero) and retries.
    During the retry's blocking gate, a steering message arrives
    via try_deliver. The retry succeeds, and the steering message
    is detected in _handle_final_response.

    Breakage this catches:
    - Steering message lost during retry window
    - Retry mechanism interferes with inbox state
    - _handle_final_response doesn't run after a retried call
    - Steered message not included in the steering continuation
    """
    await create_test_agent(client)

    # Skip backoff sleep so tests don't wait 2+ seconds.
    # The steering mechanism is orthogonal to sleep duration.
    monkeypatch.setattr(
        "agent_plane.runtime.llm_retry.time.sleep",
        lambda _: None,
    )

    # Call 1: raises HTTP 429 — classified as retryable.
    # execute_with_retry catches it, emits retry event, retries.
    fake_request = httpx.Request("POST", "http://test/v1/responses")
    fake_response = httpx.Response(429, request=fake_request)
    retryable_error = httpx.HTTPStatusError(
        "Rate limited",
        request=fake_request,
        response=fake_response,
    )
    # Error call — not referenced directly; consumed by the retry loop
    mock_llm.add_call(exception=retryable_error)
    # Call 2: retry succeeds, blocks so we can inject a steer.
    call_2 = mock_llm.add_call(
        text="Retry succeeded",
        block=True,
        stream_tokens=True,
    )
    # Call 3: steering continuation after _SteeringRetry.
    call_3 = mock_llm.add_call(text="Steered response")

    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: call_1 fires and raises immediately. execute_with_retry
    # retries → call_2 fires and blocks.
    await asyncio.wait_for(call_2.call_event.wait(), timeout=10)

    # Concurrent action: steer while the retry is blocked in
    # the LLM call.
    steer = await create_test_response(
        client,
        input_text="Change topic",
        previous_response_id=first_id,
    )
    assert steer.body["id"] == first_id

    # Release: retry completes, _handle_final_response detects
    # the steered message, triggers _SteeringRetry → call_3.
    call_2.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify steering message is in conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts
    assert "Change topic" in user_texts

    # 3 mock invocations: error (call_1) + retry success (call_2)
    # + steering continuation (call_3). The agent loop made 2
    # logical LLM calls (the retry is internal to the @step).
    assert mock_llm.call_count == 3, (
        f"Expected 3 mock invocations (error + retry + steering "
        f"continuation), got {mock_llm.call_count}"
    )

    # Third call received the steered message
    assert call_3.received_kwargs is not None, "Steering continuation call was never made"
    llm_input = str(call_3.received_kwargs["input"])
    assert "Change topic" in llm_input, (
        f"Steering continuation did not receive the steered "
        f"message in its input. Input: {llm_input}"
    )


async def test_foreground_steering_returns_immediately(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A foreground (background=false) steering request returns the
    existing in-progress response immediately without blocking.

    All other steering tests use background=true. This verifies
    that the route layer short-circuits on successful steering
    regardless of the background flag — _attempt_steering returns
    a ResponseObject before the background/stream branching logic.

    Race window: workflow is blocked in the LLM call. The
    foreground steering request hits _attempt_steering, which
    calls try_deliver (succeeds because inbox is open) and
    returns the existing ResponseObject. The route returns it
    immediately — no blocking wait.

    Breakage this catches:
    - Foreground steering enters _handle_blocking_wait (hangs)
    - Foreground steering creates a new task instead of delivering
    - Route layer checks background flag before _attempt_steering
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="First response", block=True)
    # Steering continuation after _SteeringRetry
    mock_llm.add_call(text="Steered response")

    # First request: background=true to get ID immediately
    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow blocked in LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action: steer with background=false.
    # This MUST return immediately (not block until completion).
    steer = await create_test_response(
        client,
        input_text="Foreground steer",
        previous_response_id=first_id,
        background=False,
    )
    # Returns the SAME in-progress response — steering delivered,
    # not a new blocking task.
    assert steer.body["id"] == first_id
    assert steer.body["status"] in ("queued", "in_progress"), (
        f"Foreground steer should return the existing active "
        f"response, got status={steer.body['status']}"
    )

    # Release and verify completion
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify steered message persisted and processed
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts
    assert "Foreground steer" in user_texts

    # 2 LLM calls: original + steering continuation
    assert mock_llm.call_count == 2, (
        f"Expected 2 LLM calls (original + steering continuation), got {mock_llm.call_count}"
    )


# ── Steering during auto-collect ─────────────────────────


def _identify_parent_and_sub(
    call_a: MockCall,
    call_b: MockCall,
) -> tuple[MockCall, MockCall]:
    """
    Identify which blocked MockCall belongs to the parent and
    which to the sub-agent.

    The sub-agent's first LLM input contains the user message
    from ``spawn_one`` (``"Do work"``). The parent's input
    contains the ``function_call_output`` from the spawn tool.

    :param call_a: First blocked call.
    :param call_b: Second blocked call.
    :returns: ``(parent_call, sub_call)``.
    """

    def _is_sub_agent(call: MockCall) -> bool:
        for item in call.received_kwargs.get("input", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_text":
                if "Do work" in item.get("text", ""):
                    return True
        return False

    if _is_sub_agent(call_a):
        return call_b, call_a
    return call_a, call_b


async def test_steering_during_auto_collect(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message while the parent is blocked in auto-collect.

    Race window: the parent LLM produces a final response (no
    tool calls) while a sub-agent is still running. Auto-collect
    polls ``get_sync`` + ``time.sleep`` until the sub-agent
    finishes. A steering message arrives during this window.

    The parent must detect the steering during the auto-collect
    poll, break early, and include the steering text in a
    subsequent LLM call's input.

    Synchronization (deterministic — no FIFO ordering guesses):
    1. Call 1 (parent): spawn tool call — consumed first since
       the parent workflow starts before the sub-agent.
    2. Calls 2–3: BOTH block. Parent and sub-agent each consume
       one (order is non-deterministic).
    3. Test waits for both ``call_event``s, then inspects
       ``received_kwargs`` to identify which is which.
    4. Release the parent — it gets a text response (no tool
       calls), enters auto-collect, starts polling.
    5. Send the steering message. Auto-collect's next poll
       cycle detects it via ``fetch_all_items`` and breaks.
    6. Release the sub-agent — it completes.
    7. Assert: the steering text appears in a subsequent
       parent LLM call's ``received_kwargs.input``.

    Breakage this catches:
    - Auto-collect blocks indefinitely in ``wait_sync``, so
      the steering message is never seen by the LLM.
    - Auto-collect detects steering but doesn't break early.
    - Parent completes without an LLM turn that includes the
      steered message in its input.
    """
    bundle = build_agent_bundle(
        name="steer-collect",
        sub_agents=[
            {"name": "worker", "description": "Background worker"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={
            "bundle": ("agent.tar.gz", bundle, "application/gzip"),
        },
    )
    assert resp.status_code == 201

    # Call 1 (parent): spawn the worker sub-agent.
    spawn_args = json.dumps(
        {"agents": [{"name": "worker", "input": "Do work"}]},
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn",
                "name": "spawn_sub_agents",
                "arguments": spawn_args,
            },
        ],
    )

    # Calls 2–3: BOTH block. Parent and sub-agent each get one
    # (order is non-deterministic). We identify them afterward
    # via received_kwargs.
    call_a = mock_llm.add_call(
        text="Generic response",
        block=True,
    )
    call_b = mock_llm.add_call(
        text="Generic response",
        block=True,
    )

    # Post-steering LLM calls (parent sees collected results +
    # steering message and produces a final answer).
    mock_llm.add_call(text="Post-collect answer.")
    mock_llm.add_call(text="Steered final answer.")

    first = await create_test_response(
        client,
        model="steer-collect",
        input_text="Start the worker",
    )
    first_id = first.body["id"]

    # Gate: both blocked calls are entered. Now we know the
    # parent and sub-agent are each sitting on a blocked call.
    call_a.call_event.wait(timeout=10)
    call_b.call_event.wait(timeout=10)

    parent_call, sub_call = _identify_parent_and_sub(
        call_a,
        call_b,
    )

    # Release the PARENT only. It receives a text response
    # (no tool calls) → triggers auto-collect → starts polling
    # for the sub-agent (which is still blocked).
    parent_call.release()

    # Give the parent time to enter the auto-collect poll loop.
    # One poll cycle is _COLLECT_POLL_S = 0.5s.
    await asyncio.sleep(0.3)

    # Concurrent action: steer while parent is in the
    # auto-collect poll loop.
    steer = await create_test_response(
        client,
        model="steer-collect",
        input_text="ABORT: cancel the worker immediately",
        previous_response_id=first_id,
    )
    # Steering returns the SAME response (not a new task).
    assert steer.body["id"] == first_id

    # Release the sub-agent so it completes. Auto-collect
    # should have already detected the steering and broken
    # early, but releasing ensures no deadlock either way.
    sub_call.release()

    # Poll until parent completes.
    for _ in range(100):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.1)

    body = resp.json()
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Output: {body.get('output')}"
    )

    # Content-based assertion: the steering text must appear
    # in the input of at least one LLM call AFTER the parent
    # was released. This proves the parent's workflow thread
    # saw the steering message and fed it to the LLM — not
    # just that the message was persisted in the DB.
    steering_seen_by_llm = False
    for call in mock_llm._calls:
        if call.received_kwargs is None:
            continue
        input_str = json.dumps(
            call.received_kwargs.get("input", []),
        )
        if "ABORT: cancel the worker immediately" in input_str:
            steering_seen_by_llm = True
            break

    assert steering_seen_by_llm, (
        "Steering message was never included in any LLM call's "
        "input. The auto-collect poll loop blocked the workflow "
        "thread, preventing _sync_history from picking up the "
        "steered message. LLM inputs received: "
        + str(
            [
                json.dumps(c.received_kwargs.get("input", []))[:200]
                for c in mock_llm._calls
                if c.received_kwargs is not None
            ]
        )
    )


# ── Ghost text persistence ────────────────────────────────


async def test_ghost_text_persisted_before_auto_collect(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    LLM text is persisted BEFORE auto-collect starts polling.

    Ghost text bug: when the parent LLM returns text (no tool
    calls) while sub-agents are still running, the old code
    entered auto-collect before ``_handle_final_response`` could
    persist the text. The text was streamed to SSE but never
    committed to the conversation store — a "ghost" that
    subsequent LLM calls could not see.

    Synchronization:
    1. Call 1 (parent): spawn tool call.
    2. Calls 2–3: BOTH block. Parent and sub-agent each get
       one (order non-deterministic).
    3. Release the parent → text ``"GHOST_MARKER"`` → enters
       ``_persist_text_before_auto_collect`` → persists text
       → enters auto-collect → polls (sub-agent still blocked).
    4. While auto-collect polls, query conversation items API.
    5. Assert ``"GHOST_MARKER"`` IS in persisted assistant items.
    6. Release sub-agent → auto-collect finishes.

    Breakage this catches:
    - Text streamed to SSE but not persisted in conversation
      store (the original ghost text bug).
    - Subsequent LLM calls cannot see the parent's text
      response, breaking steering (steering response is lost).
    """
    bundle = build_agent_bundle(
        name="ghost-text",
        sub_agents=[
            {"name": "bg-worker", "description": "Background worker"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={
            "bundle": ("agent.tar.gz", bundle, "application/gzip"),
        },
    )
    assert resp.status_code == 201

    # Call 1 (parent): spawn the sub-agent.
    spawn_args = json.dumps(
        {"agents": [{"name": "bg-worker", "input": "Do work"}]},
    )
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn_ghost",
                "name": "spawn_sub_agents",
                "arguments": spawn_args,
            },
        ],
    )

    # Calls 2–3: BOTH block. Parent and sub-agent each consume
    # one (order non-deterministic).
    call_a = mock_llm.add_call(
        text="GHOST_MARKER",
        block=True,
    )
    call_b = mock_llm.add_call(
        text="GHOST_MARKER",
        block=True,
    )

    # Post-auto-collect LLM call (parent sees collected results
    # and produces a final answer).
    mock_llm.add_call(text="Final answer after collect.")

    first = await create_test_response(
        client,
        model="ghost-text",
        input_text="Start the worker",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: both blocked calls are entered.
    call_a.call_event.wait(timeout=10)
    call_b.call_event.wait(timeout=10)

    parent_call, sub_call = _identify_parent_and_sub(
        call_a,
        call_b,
    )

    # Release the PARENT only. It receives text "GHOST_MARKER"
    # (no tool calls) → _persist_text_before_auto_collect
    # persists the text → enters auto-collect → polls for the
    # sub-agent (still blocked).
    parent_call.release()

    # KEY ASSERTION: poll conversation items until GHOST_MARKER
    # appears. The text MUST be persisted before auto-collect
    # starts. Before the fix, this text was only in SSE
    # (streamed but not persisted) and would never appear.
    # Poll instead of sleep for deterministic synchronization.
    ghost_found = False
    for _ in range(50):
        items_resp = await client.get(
            f"/v1/conversations/{conv_id}/items",
            params={"limit": 100},
        )
        items = items_resp.json()["data"]
        assistant_texts = [
            block.get("text", "")
            for item in items
            if item.get("role") == "assistant"
            for block in item.get("content", [])
        ]
        if any("GHOST_MARKER" in t for t in assistant_texts):
            ghost_found = True
            break
        await asyncio.sleep(0.1)

    # "GHOST_MARKER" must be persisted in the conversation store
    # while auto-collect is still running (sub-agent is still
    # blocked). If absent, the text was ghost — streamed to SSE
    # but never committed.
    assert ghost_found, (
        f"Parent text 'GHOST_MARKER' not found in persisted "
        f"assistant items during auto-collect. This means text "
        f"was streamed to SSE but not committed to the "
        f"conversation store (ghost text bug). "
        f"Assistant texts: {assistant_texts}"
    )

    # Release the sub-agent → completes → auto-collect finishes.
    sub_call.release()

    # Poll until parent completes.
    for _ in range(100):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.1)

    body = resp.json()
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Output: {body.get('output')}"
    )


# ── Cancel Races ─────────────────────────────────────────


async def test_cancel_during_llm_call(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Cancel a response while the LLM call is in progress.

    Race window: workflow thread is blocked inside the mock
    LLM's create(). The cancel API sets the DBOS workflow's
    cancelled flag. When the mock is released, the workflow
    detects cancellation at the next checkpoint.

    Breakage this catches:
    - Cancel returns wrong status (not cancelled)
    - Cancel returns non-empty output (workflow produced
      results despite cancellation)
    - Workflow hangs after cancellation
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action: cancel while blocked in LLM
    cancel_resp = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"
    assert cancel_resp.json()["output"] == []

    # Release: workflow must terminate (not hang)
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{response_id}")
        if resp.json()["status"] in ("completed", "cancelled"):
            break
    assert resp.json()["status"] == "cancelled"


async def test_cancel_idempotent_while_blocked(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two cancel requests issued while the workflow is blocked.
    Both must return cancelled status — the second cancel
    must not fail or return a different status.

    Race window: the workflow is blocked in the LLM call.
    The first cancel sets DBOS cancelled flag. The second
    cancel hits the already-cancelled task.

    Breakage this catches:
    - Second cancel raises an error (not idempotent)
    - Second cancel returns wrong status
    - Task status flips between cancelled and something else
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    result = await create_test_response(client)
    response_id = result.body["id"]

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action: cancel twice while still blocked
    resp1 = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )
    resp2 = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["status"] == "cancelled"
    assert resp2.json()["status"] == "cancelled"


async def test_steering_then_cancel_preserves_message(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer a message then cancel the workflow. The steered
    message must persist in the conversation because
    try_deliver commits in its own transaction, independent
    of the workflow's DBOS lifecycle.

    Race window: workflow is blocked in LLM. try_deliver
    writes the steered message to conversation_items (its
    own DB transaction). Then cancel stops the workflow.
    The message must survive the cancellation.

    Breakage this catches:
    - Steered message rolled back by cancellation
    - try_deliver transaction tied to workflow transaction
    - Conversation items lost on cancel
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(block=True)

    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    await asyncio.wait_for(call_1.call_event.wait(), timeout=10)

    # Concurrent action 1: steer while blocked
    steer = await create_test_response(
        client,
        input_text="Follow-up",
        previous_response_id=first_id,
    )
    assert steer.body["id"] == first_id

    # Concurrent action 2: cancel the workflow
    cancel_resp = await client.post(
        f"/v1/responses/{first_id}/cancel",
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Verify steered message persists despite cancellation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts, "Original user message missing after cancel"
    assert "Follow-up" in user_texts, (
        "Steered message lost after cancel — try_deliver's "
        "transaction must be independent of the workflow"
    )


async def test_steering_after_inbox_closed_creates_new_task(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Steer into a completed response (inbox closed). The server
    must create a new task instead of delivering to the old one.

    Race window: the workflow has finished and closed its inbox.
    The steering request finds the task completed, so
    ``_attempt_steering`` skips ``try_deliver`` entirely and
    returns the ``conversation_id`` for normal task creation.

    Breakage this catches:
    - Steering into completed task hangs or errors
    - No new task created (steered message lost)
    - New task uses wrong conversation (message orphaned)
    """
    await create_test_agent(client)

    # First response: completes normally (no blocking)
    mock_llm.add_call(text="First answer")
    first = await create_test_response(
        client,
        input_text="Hello",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Wait for completion (inbox closes)
    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Steer into the completed response — should create new task
    mock_llm.add_call(text="Second answer")
    second = await create_test_response(
        client,
        input_text="Follow-up after completion",
        previous_response_id=first_id,
    )
    second_id = second.body["id"]

    # Must be a NEW task (different ID)
    assert second_id != first_id, "Steering into completed task must create a new task"

    # Wait for the new task to complete
    for _ in range(50):
        resp = await client.get(f"/v1/responses/{second_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Both responses share the same conversation
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    assert "Hello" in user_texts
    assert "Follow-up after completion" in user_texts


# ── Cross-Server Steering ────────────────────────────────
#
# These tests launch real ``ap server`` subprocesses sharing
# a database, with a mock LLM HTTP server providing the
# synchronization gate.


def _find_free_port() -> int:
    """
    Bind to port 0 and return the OS-assigned port number.

    The socket is closed before returning, so the port may
    theoretically be reused — but in practice the window is
    negligible for sequential test setup.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_mock_llm_server(port: int) -> subprocess.Popen[bytes]:
    """
    Start the mock LLM server on the given port.

    :param port: TCP port for the mock server.
    :returns: The subprocess handle.
    """
    script = Path(__file__).parent / "mock_llm_server.py"
    return subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _start_ap_server(
    port: int,
    db_uri: str,
    artifact_dir: Path,
) -> subprocess.Popen[bytes]:
    """
    Start an ``ap server`` subprocess.

    :param port: TCP port for the server.
    :param db_uri: SQLAlchemy database URI (shared across servers).
    :param artifact_dir: Path for artifact storage.
    :returns: The subprocess handle.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-uri",
            db_uri,
            "--artifact-location",
            str(artifact_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


async def _poll_until_ready(
    url: str,
    timeout: float = 15.0,
) -> None:
    """
    Poll a URL until it returns HTTP 200, or raise on timeout.

    :param url: The URL to poll.
    :param timeout: Maximum seconds to wait.
    :raises TimeoutError: If the server doesn't respond in time.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError):
                pass
            await asyncio.sleep(0.3)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s")


def _build_mock_llm_bundle(
    name: str,
    mock_llm_port: int,
) -> bytes:
    """
    Build an agent bundle configured to use the mock LLM.

    The agent's ``llm.connection.base_url`` points at the mock
    LLM server so all LLM calls route there instead of OpenAI.

    :param name: Agent name (also used as model prefix).
    :param mock_llm_port: Port of the mock LLM server.
    :returns: A tar.gz bundle as bytes.
    """
    # Any: YAML config values are heterogeneous (str, int, dict, etc.)
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": name,
        "llm": {
            "model": f"openai/{name}",
            "connection": {
                "api_key": "fake-key",
                "base_url": f"http://127.0.0.1:{mock_llm_port}/v1",
            },
        },
    }
    config_bytes = yaml.dump(config).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


async def _assert_cross_server_steering(
    client_a: httpx.AsyncClient,
    client_b: httpx.AsyncClient,
    mock_client: httpx.AsyncClient,
    mock_llm_port: int,
) -> None:
    """
    Run the cross-server steering sequence and assert results.

    :param client_a: HTTP client for server A.
    :param client_b: HTTP client for server B.
    :param mock_client: HTTP client for the mock LLM server.
    :param mock_llm_port: Port of the mock LLM server.
    """
    # Deploy agent to server A (writes to shared DB)
    bundle = _build_mock_llm_bundle("test-agent", mock_llm_port)
    resp = await client_a.post(
        "/api/agents",
        files={
            "bundle": ("agent.tar.gz", bundle, "application/gzip"),
        },
    )
    # Failure: server A can't register agents — infra problem,
    # not concurrency.
    assert resp.status_code == 201

    # Create response on server A — workflow calls mock LLM
    resp = await client_a.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Hello from server A",
            "background": True,
        },
    )
    # Failure: server A can't create tasks — infra problem,
    # not concurrency.
    assert resp.status_code == 200
    first_id = resp.json()["id"]
    conv_id = resp.json()["conversation"]["id"]

    # Gate: wait for mock LLM to receive the request
    for _ in range(150):
        status = await mock_client.get("/gate/pending")
        if status.json()["pending"]:
            break
        await asyncio.sleep(0.1)
    # Failure: workflow never reached the LLM call — the agent
    # spec's connection.base_url didn't route to the mock, or
    # the workflow crashed before calling the LLM.
    assert status.json()["pending"], "Mock LLM never received request"

    # Concurrent action: steer from server B
    resp = await client_b.post(
        "/v1/responses",
        json={
            "model": "test-agent",
            "input": "Steered from server B",
            "previous_response_id": first_id,
            "background": True,
        },
    )
    assert resp.status_code == 200
    # Failure (new ID returned): server B couldn't steer into
    # server A's running workflow. Causes: try_deliver failed
    # across processes (DB locking), server B saw the task as
    # inactive (stale read from shared SQLite), or inbox was
    # already closed.
    assert resp.json()["id"] == first_id

    # Release: mock LLM responds, workflow completes
    await mock_client.post("/gate/release")

    # Wait for completion on server A
    for _ in range(150):
        resp = await client_a.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.1)
    # Failure: workflow didn't finish after gate release — hung,
    # crashed on the steered message, or the steering
    # continuation's second LLM call failed.
    assert resp.json()["status"] == "completed"

    # Verify steered message from server B is in the conversation
    items_resp = await client_a.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items = items_resp.json()["data"]
    user_texts = [i["content"][0]["text"] for i in items if i["role"] == "user"]
    # Failure: original user message lost — fundamental store
    # corruption.
    assert "Hello from server A" in user_texts
    # Failure: steered message not persisted — try_deliver's
    # cross-process transaction was lost (rolled back, or
    # committed to a different DB connection). This is the
    # assertion most specific to cross-server: it proves
    # try_deliver's atomic check-and-insert works across
    # process boundaries with SQLite's file-level locking.
    assert "Steered from server B" in user_texts

    stats_resp = await mock_client.get("/stats")
    # Failure (== 1): steering was delivered but the workflow
    # didn't continue — steered messages filtered out instead
    # of triggering a follow-up LLM call.
    # Failure (== 0): workflow never called the LLM at all.
    # Failure (> 2): duplicate work — workflow looped more
    # than expected.
    assert stats_resp.json()["request_count"] == 2


async def test_cross_server_steering_via_shared_db(
    tmp_path: Path,
) -> None:
    """
    Two real server processes sharing a database. A steering
    request from server B delivers to a workflow running on
    server A via the shared DB.

    Race window: server A's workflow is blocked in the mock
    LLM (HTTP gate). Server B receives the steering request,
    looks up the task in the shared DB (sees IN_PROGRESS),
    and calls try_deliver(). After gate release, server A's
    workflow completes with both messages in the conversation.

    Breakage this catches:
    - try_deliver fails across server processes
    - Steered message lost due to cross-process isolation
    - Duplicate LLM calls from cross-server steering
    - Task lookup fails on server B (stale or missing data)
    """
    mock_port = _find_free_port()
    port_a = _find_free_port()
    port_b = _find_free_port()
    db_uri = f"sqlite:///{tmp_path / 'shared.db'}"

    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(_start_mock_llm_server(mock_port))
        await _poll_until_ready(
            f"http://127.0.0.1:{mock_port}/stats",
        )

        procs.append(
            _start_ap_server(port_a, db_uri, tmp_path / "art_a"),
        )
        await _poll_until_ready(
            f"http://127.0.0.1:{port_a}/api/agents",
        )

        procs.append(
            _start_ap_server(port_b, db_uri, tmp_path / "art_b"),
        )
        await _poll_until_ready(
            f"http://127.0.0.1:{port_b}/api/agents",
        )

        async with (
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port_a}",
            ) as client_a,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port_b}",
            ) as client_b,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{mock_port}",
            ) as mock_client,
        ):
            await _assert_cross_server_steering(
                client_a,
                client_b,
                mock_client,
                mock_port,
            )
    finally:
        for proc in reversed(procs):
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=10)
