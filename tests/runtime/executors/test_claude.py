"""Unit tests for Claude executor helpers.

Covers ``_build_history_prompt`` — the crash-recovery prompt builder
that serializes conversation history into a single text block for the
Claude SDK.
"""

from __future__ import annotations

from agent_plane.runtime.executors.claude import _build_history_prompt


def test_build_history_prompt_skips_function_call_items() -> None:
    """
    ``_build_history_prompt`` must skip ``function_call`` and
    ``function_call_output`` items that were persisted by
    ``_persist_observed_tool_calls``.

    **What breaks if wrong**: function_call items lack ``role`` and
    ``content`` keys, so they render as empty ``"user: "`` lines.
    The Claude SDK sees gibberish context and produces no useful
    output on the second turn — the multi-turn bug reported in
    production.
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

    # Must NOT contain empty "user: " lines from function_call items.
    # If function_call items leak through, they produce "user: " with
    # no content (the item has no "role" key, defaulting to "user",
    # and no "content" key, defaulting to "").
    assert "user: \n" not in result
    # Must not contain raw function_call data.
    assert "function_call" not in result
    assert "call_1" not in result
    assert "bash" not in result


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
