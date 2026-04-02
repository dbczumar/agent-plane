"""Tests for llms.adapters.anthropic — translation logic."""

import json

import pytest

from agent_plane.llms.adapters.anthropic import (
    _anthropic_to_chat,
    _chat_to_anthropic,
    _convert_tool_choice,
    _convert_tools,
    _translate_part_to_anthropic,
)


def test_system_messages_extracted() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    assert payload["system"] == "Be helpful."
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


def test_multiple_system_messages_joined() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    assert payload["system"] == "Be helpful.\nBe concise."


def test_assistant_tool_calls_converted() -> None:
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "London"}',
                    },
                }
            ],
        },
    ]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    assistant_msg = payload["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0]["type"] == "tool_use"
    assert assistant_msg["content"][0]["id"] == "call_1"
    assert assistant_msg["content"][0]["input"] == {"city": "London"}


def test_tool_messages_converted_to_tool_result() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Sunny, 22C",
        }
    ]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    msg = payload["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "tool_result"
    assert msg["content"][0]["tool_use_id"] == "call_1"


def test_temperature_halved() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    payload = _chat_to_anthropic(messages, "claude-test", None, {"temperature": 1.0})
    assert payload["temperature"] == 0.5


def test_default_max_tokens() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    assert payload["max_tokens"] == 16384


def test_tools_converted() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = _convert_tools(tools)
    assert len(result) == 1
    assert result[0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {}},
    }


@pytest.mark.parametrize(
    ("openai_choice", "expected"),
    [
        ("none", {"type": "none"}),
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        (
            {"type": "function", "function": {"name": "foo"}},
            {"type": "tool", "name": "foo"},
        ),
    ],
)
def test_tool_choice_mapping(
    openai_choice: str | dict,
    expected: dict,
) -> None:
    assert _convert_tool_choice(openai_choice) == expected


def test_anthropic_text_response_to_chat() -> None:
    resp = {
        "id": "msg_123",
        "model": "claude-test",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    chat = _anthropic_to_chat(resp)
    assert chat["model"] == "claude-test"
    assert chat["choices"][0]["message"]["content"] == "Hello!"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"]["prompt_tokens"] == 10
    assert chat["usage"]["completion_tokens"] == 5


def test_anthropic_tool_use_response_to_chat() -> None:
    resp = {
        "id": "msg_456",
        "model": "claude-test",
        "content": [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "London"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    chat = _anthropic_to_chat(resp)
    tool_calls = chat["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tu_1"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "London"}
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_max_tokens_stop_reason() -> None:
    resp = {
        "id": "msg_789",
        "model": "claude-test",
        "content": [{"type": "text", "text": "Truncat"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 5, "output_tokens": 100},
    }
    chat = _anthropic_to_chat(resp)
    assert chat["choices"][0]["finish_reason"] == "length"


# ── Multimodal content translation ──────────────────────


def test_user_message_with_image_data_uri() -> None:
    """
    User message with image_url data URI translates to Anthropic
    base64 image source.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ],
        },
    ]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    content = payload["messages"][0]["content"]
    # Two blocks: text + image.
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Describe this"}
    assert content[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "abc123",
        },
    }


def test_user_message_with_external_url() -> None:
    """
    External image URL translates to Anthropic URL source type.
    """
    part = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/photo.png"},
    }
    result = _translate_part_to_anthropic(part)
    assert result == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/photo.png"},
    }


def test_user_message_with_file_data() -> None:
    """
    input_file with file_data translates to Anthropic document type.
    """
    part = {
        "type": "input_file",
        "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
    }
    result = _translate_part_to_anthropic(part)
    assert result == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjQK",
        },
    }


def test_string_user_content_passes_through() -> None:
    """
    String user content passes through unchanged — no translation
    needed for text-only messages.
    """
    messages = [{"role": "user", "content": "Hello"}]
    payload = _chat_to_anthropic(messages, "claude-test", None, {})
    # String content passed through as-is.
    assert payload["messages"][0]["content"] == "Hello"
