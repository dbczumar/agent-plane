"""Unit tests for Claude executor helpers.

Covers ``_build_history_prompt`` — the crash-recovery prompt builder
that serializes conversation history into a single text block for the
Claude SDK.
"""

from __future__ import annotations

from agent_plane.runtime.executors.claude import _build_history_prompt


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
