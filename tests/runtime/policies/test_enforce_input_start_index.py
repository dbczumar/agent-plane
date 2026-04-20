"""
Regression tests for :func:`_last_assistant_index` and the
start-index logic in :func:`_enforce_input_policies`.

The bug this pins: before the fix, a workflow with any prior
conversation history would re-enforce every user message on
its first iteration — the cursor starts ``None`` and the
loop walked from index 0. Users observed three approval
prompts for one new message in ``ap chat`` when their
conversation had prior turns. See the ``⚠ approval required``
triplicate reported by the user.

The fix: on a fresh workflow invocation (cursor ``None``),
start from the index after the last assistant message.
Anything before it was already responded to in a prior turn
and must not be re-enforced. A brand-new conversation has no
assistant yet, so we start at index 0 and enforce everything
— the happy path is preserved.
"""

from __future__ import annotations

from agent_plane.entities import (
    ConversationItem,
    FunctionCallData,
    MessageData,
)
from agent_plane.runtime.workflow import _last_assistant_index


def _user(item_id: str, text: str) -> ConversationItem:
    """Build a user-role message ConversationItem for tests."""
    return ConversationItem(
        id=item_id,
        type="message",
        status="completed",
        response_id="resp",
        created_at=1,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": text}],
        ),
    )


def _assistant(item_id: str, text: str) -> ConversationItem:
    """Build an assistant-role message ConversationItem for tests."""
    return ConversationItem(
        id=item_id,
        type="message",
        status="completed",
        response_id="resp",
        created_at=1,
        data=MessageData(
            role="assistant",
            agent="demo",
            content=[{"type": "output_text", "text": text}],
        ),
    )


def _function_call(item_id: str) -> ConversationItem:
    """Build a function_call ConversationItem — neither user nor assistant."""
    return ConversationItem(
        id=item_id,
        type="function_call",
        status="completed",
        response_id="resp",
        created_at=1,
        data=FunctionCallData(
            call_id="call_1",
            name="noop",
            arguments="{}",
            agent="demo",
        ),
    )


def test_empty_history_returns_negative_one() -> None:
    """
    No history → no assistant → ``-1``. Caller adds 1, so the
    start index is 0 — the first incoming user message gets
    enforced, as intended on a brand-new conversation.
    """
    assert _last_assistant_index([]) == -1


def test_all_user_messages_no_assistant_returns_negative_one() -> None:
    """
    A conversation where only the current turn's user input
    has arrived (assistant hasn't replied yet) returns ``-1``.
    Every user message gets enforced. This is the common case
    on the very first turn of a new conversation — nothing has
    been responded to yet.
    """
    history = [_user("u1", "hi")]
    assert _last_assistant_index(history) == -1


def test_single_assistant_at_end_returns_its_index() -> None:
    """
    After a complete turn (user → assistant), the last
    assistant is at index 1. On the next workflow invocation,
    start_index = 2 — skipping the already-responded user
    message at index 0.
    """
    history = [_user("u1", "hi"), _assistant("a1", "hello")]
    assert _last_assistant_index(history) == 1


def test_multiple_assistants_returns_most_recent() -> None:
    """
    A two-turn conversation (user → assistant → user →
    assistant) has two assistant messages. ``_last_assistant_index``
    must return the MOST RECENT one (index 3) so the next
    turn's enforcement starts after it, not after the first
    one. This is the exact scenario that produced the
    duplicate-approval bug.
    """
    history = [
        _user("u1", "first"),
        _assistant("a1", "response one"),
        _user("u2", "second"),
        _assistant("a2", "response two"),
    ]
    assert _last_assistant_index(history) == 3


def test_non_message_items_ignored() -> None:
    """
    function_call / function_call_output items are not
    messages — they must be skipped when searching for the
    last assistant. Otherwise a turn with tool calls could
    confuse the cursor.
    """
    history = [
        _user("u1", "hi"),
        _assistant("a1", "reasoning"),
        _function_call("fc1"),
        _user("u2", "next"),
    ]
    # Only one assistant, at index 1. Later function_call
    # items are ignored.
    assert _last_assistant_index(history) == 1


def test_user_message_after_assistant_does_not_shift_index() -> None:
    """
    Sanity check: the returned index points at the assistant
    message itself, not at later user messages. The caller
    adds 1 to get the start index for user-message
    enforcement.

    Scenario: prior turn (user → assistant) → current turn
    user input arrived but agent hasn't responded yet.
    Start enforcement from the current turn's user, not
    earlier.
    """
    history = [
        _user("u_old", "prior"),
        _assistant("a_prior", "handled prior"),
        _user("u_new", "this turn"),
    ]
    assert _last_assistant_index(history) == 1
    # start_index = 1 + 1 = 2, which points at "u_new" —
    # exactly the message we want to enforce.
