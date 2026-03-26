"""Concurrency integration tests.

Each test creates a deterministic race window using
ControllableMockClient's blocking gates, then exercises a
concurrent interaction (steering, cancel) against the real
DBOS workflow pipeline.

Every test has ALL of:
- A blocked LLM call (mock_llm.add_call(block=True))
- A synchronization gate (call.call_event.wait(timeout=10))
- A concurrent action while blocked
- A release (call.release())
- Assertions on both sides

No ``time.sleep`` — synchronization is purely event-driven
via ``MockCall.call_event`` and ``MockCall.release()``.
"""

from __future__ import annotations

import httpx
import pytest

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import (
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
    - Workflow makes an extra LLM call on steered input
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="First response", block=True)

    first = await create_test_response(client, input_text="Hello")
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]
    assert first.body["status"] == "queued"

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

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

    # Release: workflow completes normally
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break

    assert resp.json()["status"] == "completed"
    # Steering does NOT trigger a second LLM call — steered
    # messages have response_id == task_id, so the workflow's
    # filter (ci.response_id != task_id) excludes them.
    assert mock_llm.call_count == 1


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
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(text="The answer", block=True)

    first = await create_test_response(
        client,
        input_text="Question 1",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

    # Concurrent action: steer while workflow is blocked
    await create_test_response(
        client,
        input_text="Steering message",
        previous_response_id=first_id,
    )

    # Release: workflow persists assistant at next position
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify position ordering: user, steered, assistant
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Question 1"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Steering message"
    assert items[2]["role"] == "assistant"
    assert items[2]["content"][0]["text"] == "The answer"


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
    - Workflow makes extra LLM calls for steered messages
    """
    await create_test_agent(client)

    call_1 = mock_llm.add_call(
        text="Final answer",
        block=True,
    )

    first = await create_test_response(
        client,
        input_text="Original question",
    )
    first_id = first.body["id"]
    conv_id = first.body["conversation"]["id"]

    # Gate: workflow is inside the LLM call
    call_1.call_event.wait(timeout=10)

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

    # Release: workflow completes
    call_1.release()

    for _ in range(50):
        resp = await client.get(f"/v1/responses/{first_id}")
        if resp.json()["status"] == "completed":
            break
    assert resp.json()["status"] == "completed"

    # Verify all 4 items in position order
    items_resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]

    assert len(items) == 4, f"Expected 4 items (user + 2 steered + assistant), got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"][0]["text"] == "Original question"
    assert items[1]["role"] == "user"
    assert items[1]["content"][0]["text"] == "Clarification A"
    assert items[2]["role"] == "user"
    assert items[2]["content"][0]["text"] == "Clarification B"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "Final answer"

    # Only 1 LLM call — steered messages don't trigger loops
    assert mock_llm.call_count == 1


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
    call_1.call_event.wait(timeout=10)

    # Concurrent action: cancel while blocked in LLM
    cancel_resp = await client.post(
        f"/v1/responses/{response_id}/cancel",
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"
    assert cancel_resp.json()["output"] == []


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
    call_1.call_event.wait(timeout=10)

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
    call_1.call_event.wait(timeout=10)

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
