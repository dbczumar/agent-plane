"""Unit tests for Claude executor helpers.

Covers ``_build_history_prompt`` (crash-recovery prompt serialization)
and ``_build_prompt`` (prompt routing logic that decides whether to
send the latest message or full history).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_plane.runtime.executors.claude import (
    _build_history_prompt,
    _build_prompt,
)

# ── _build_history_prompt ──────────────────────────────────


def test_build_history_prompt_includes_tool_calls() -> None:
    """
    ``_build_history_prompt`` must serialize ``function_call`` and
    ``function_call_output`` items so Claude knows what tools it
    ran in the previous session.

    **What breaks if wrong**: after a server restart, Claude has
    no memory of tool calls from the prior session. It may
    redundantly re-run tools or produce inconsistent responses.
    """
    messages = [
        {"role": "user", "content": "Read the file"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "bash",
            "arguments": '{"cmd": "cat foo.py"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "print('hello')",
        },
        {"role": "assistant", "content": "Here is the file content."},
        {"role": "user", "content": "Thanks, now explain it"},
    ]

    result = _build_history_prompt(messages)

    # Must contain the real user/assistant messages.
    assert "user: Read the file" in result
    assert "assistant: Here is the file content." in result
    assert "user: Thanks, now explain it" in result

    # Must contain the tool call and its result.
    assert "bash" in result
    assert "cat foo.py" in result
    assert "print('hello')" in result

    # Must NOT contain empty "user: " lines — tool calls must
    # be serialized with meaningful content, not skipped or empty.
    assert "user: \n" not in result


def test_build_history_prompt_normal_conversation() -> None:
    """
    ``_build_history_prompt`` serializes a normal conversation
    (no tool calls) into ``role: content`` lines with the
    continuation instruction.

    **What breaks if wrong**: The SDK sees malformed context and
    either ignores the history or hallucinates.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = _build_history_prompt(messages)

    assert result.startswith("Conversation so far:")
    assert "user: Hello" in result
    assert "assistant: Hi there!" in result
    assert "Respond to the latest user message" in result


def test_build_history_prompt_multiple_tool_calls() -> None:
    """
    Multiple sequential tool calls are all serialized, preserving
    the call/result pairing order.

    **What breaks if wrong**: Claude sees partial tool history
    after a restart — it knows about the first tool but not the
    second, leading to inconsistent reasoning.
    """
    messages = [
        {"role": "user", "content": "Search for X and Y"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "web_search",
            "arguments": '{"query": "X"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Result for X",
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "web_search",
            "arguments": '{"query": "Y"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "Result for Y",
        },
        {"role": "assistant", "content": "Here are both results."},
    ]

    result = _build_history_prompt(messages)

    # Both tool calls and results must be present.
    assert "web_search" in result
    assert "Result for X" in result
    assert "Result for Y" in result
    # The calls must appear in order (first X, then Y).
    idx_x = result.index('query": "X')
    idx_y = result.index('query": "Y')
    assert idx_x < idx_y, "Tool calls must appear in chronological order"


def test_build_history_prompt_empty_arguments_and_output() -> None:
    """
    Tool calls with empty arguments or empty output are serialized
    without crashing — the fields default to empty strings.

    **What breaks if wrong**: KeyError or empty lines that confuse
    the prompt structure.
    """
    messages = [
        {"role": "user", "content": "Do something"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "noop_tool",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
        },
        {"role": "assistant", "content": "Done."},
    ]

    result = _build_history_prompt(messages)

    # Should not crash and should include the tool name.
    assert "noop_tool" in result
    assert "Done." in result


def test_build_history_prompt_list_content() -> None:
    """
    Messages with list-format content (e.g. multimodal) are
    JSON-serialized rather than crashing on non-string content.

    **What breaks if wrong**: TypeError when trying to format
    a list as a string.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image"},
            ],
        },
        {"role": "assistant", "content": "It shows a cat."},
    ]

    result = _build_history_prompt(messages)

    assert "Describe this image" in result
    assert "It shows a cat." in result


# ── _build_prompt routing ──────────────────────────────────


def test_build_prompt_continuing_session_returns_latest_message() -> None:
    """
    When ``is_continuing=True`` (SDK client in memory), only the
    latest user message is sent — the SDK already has full context.

    **What breaks if wrong**: the full conversation is sent to an
    SDK that already has it, causing duplicate context.
    """
    messages = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First reply"},
        {"role": "user", "content": "Second message"},
    ]

    result = _build_prompt(messages, Path("/nonexistent"), is_continuing=True)

    assert result == "Second message"


def test_build_prompt_with_transcript_returns_latest_message(
    tmp_path: Path,
) -> None:
    """
    When a session transcript exists on disk (restored from artifact
    store), only the latest user message is sent — the SDK replays
    its own transcript.

    **What breaks if wrong**: full history is sent to an SDK that
    will also replay its transcript, causing duplicate context.
    """
    # Create a .claude dir with a file to simulate a transcript.
    # _has_session_transcript checks storage_dir / "workspace" / ".claude".
    claude_dir = tmp_path / "workspace" / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "transcript.json").write_text("{}")

    messages = [
        {"role": "user", "content": "Turn one"},
        {"role": "assistant", "content": "Reply one"},
        {"role": "user", "content": "Turn two"},
    ]

    result = _build_prompt(messages, tmp_path, is_continuing=False)

    assert result == "Turn two"


def test_build_prompt_single_user_message_returns_latest(
    tmp_path: Path,
) -> None:
    """
    When there is only one user message (first turn, no recovery
    needed), the latest message is sent directly.

    **What breaks if wrong**: a single-message conversation gets
    wrapped in the verbose history format unnecessarily.
    """
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    result = _build_prompt(messages, tmp_path, is_continuing=False)

    assert result == "Hello"
    # Must NOT have the history wrapper.
    assert "Conversation so far:" not in (result or "")


def test_build_prompt_crash_recovery_uses_full_history(
    tmp_path: Path,
) -> None:
    """
    When ``is_continuing=False``, no transcript exists, and there
    are multiple user messages, the full history prompt is built
    for crash recovery.

    **What breaks if wrong**: Claude sees only the latest message
    after a restart and loses all prior conversation context.
    """
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up question"},
    ]

    result = _build_prompt(messages, tmp_path, is_continuing=False)

    assert result is not None
    assert "Conversation so far:" in result
    assert "First question" in result
    assert "First answer" in result
    assert "Follow-up question" in result


def test_build_prompt_crash_recovery_includes_tool_calls(
    tmp_path: Path,
) -> None:
    """
    The crash-recovery path includes tool call history so Claude
    knows what tools it ran before the restart.

    **What breaks if wrong**: Claude doesn't know it already ran
    a tool and may redundantly re-execute it, wasting time and
    potentially causing side effects.
    """
    messages = [
        {"role": "user", "content": "Search for Python docs"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "web_search",
            "arguments": '{"query": "python docs"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "docs.python.org - official documentation",
        },
        {"role": "assistant", "content": "I found the Python docs."},
        {"role": "user", "content": "Now search for Rust docs"},
    ]

    result = _build_prompt(messages, tmp_path, is_continuing=False)

    assert result is not None
    # Must include tool history from the previous session.
    assert "web_search" in result
    assert "python docs" in result
    assert "docs.python.org" in result
    # Must also include the new user message.
    assert "Rust docs" in result


def test_build_prompt_no_user_messages_returns_none() -> None:
    """
    When the message list has no user messages, ``_build_prompt``
    returns ``None`` — there is nothing to send.

    **What breaks if wrong**: an empty or malformed prompt is sent
    to the SDK, causing an error or hallucination.
    """
    messages = [
        {"role": "assistant", "content": "Unprompted reply"},
    ]

    result = _build_prompt(messages, Path("/nonexistent"), is_continuing=True)

    assert result is None


@pytest.mark.parametrize(
    "is_continuing",
    [True, False],
    ids=["continuing", "fresh"],
)
def test_build_prompt_empty_messages_returns_none(
    is_continuing: bool,
    tmp_path: Path,
) -> None:
    """
    An empty message list returns ``None`` regardless of session state.

    **What breaks if wrong**: IndexError or empty prompt sent to SDK.
    """
    result = _build_prompt([], tmp_path, is_continuing=is_continuing)

    assert result is None
