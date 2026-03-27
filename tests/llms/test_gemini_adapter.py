"""Tests for llms.adapters.gemini — translation logic."""

import json

import pytest

from llms.adapters.gemini import (
    _chat_to_gemini,
    _convert_tools,
    _extract_usage,
    _gemini_to_chat,
    _normalize_finish_reason,
    _translate_part_to_gemini,
)

# ── Request translation ──────────────────────────────────


def test_system_messages_become_system_instruction() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    payload = _chat_to_gemini(messages, None, {})
    assert payload["system_instruction"] == {"parts": [{"text": "Be helpful."}]}
    # System message should not appear in contents
    assert len(payload["contents"]) == 1
    assert payload["contents"][0]["role"] == "user"


def test_multiple_system_messages_joined_as_parts() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    payload = _chat_to_gemini(messages, None, {})
    assert payload["system_instruction"]["parts"] == [
        {"text": "Be helpful."},
        {"text": "Be concise."},
    ]


def test_assistant_role_remapped_to_model() -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    payload = _chat_to_gemini(messages, None, {})
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"


def test_assistant_tool_calls_become_function_call_parts() -> None:
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
    payload = _chat_to_gemini(messages, None, {})
    model_msg = payload["contents"][1]
    assert model_msg["role"] == "model"
    fc_part = model_msg["parts"][0]
    assert fc_part["functionCall"]["name"] == "get_weather"
    assert fc_part["functionCall"]["args"] == {"city": "London"}


def test_tool_messages_become_function_response() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "_tool_name": "get_weather",
            "content": "Sunny, 22C",
        }
    ]
    payload = _chat_to_gemini(messages, None, {})
    msg = payload["contents"][0]
    assert msg["role"] == "user"
    fr = msg["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather"
    assert fr["response"] == {"result": "Sunny, 22C"}


def test_generation_config_keys_mapped() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    extra = {
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "stop": ["END"],
    }
    payload = _chat_to_gemini(messages, None, extra)
    gen = payload["generationConfig"]
    assert gen["temperature"] == 0.7
    assert gen["maxOutputTokens"] == 100
    assert gen["topP"] == 0.9
    assert gen["stopSequences"] == ["END"]


def test_tools_converted_to_function_declarations() -> None:
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
    assert result[0]["name"] == "get_weather"
    assert result[0]["description"] == "Get weather"
    assert result[0]["parameters"] == {"type": "object", "properties": {}}


def test_tools_payload_wraps_declarations() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fn",
                "description": "d",
                "parameters": {},
            },
        }
    ]
    payload = _chat_to_gemini(messages, tools, {})
    assert "functionDeclarations" in payload["tools"][0]


# ── Response translation ─────────────────────────────────


def test_gemini_text_response_to_chat() -> None:
    resp = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello!"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }
    chat = _gemini_to_chat(resp, "gemini-test")
    assert chat["model"] == "gemini-test"
    assert chat["choices"][0]["message"]["content"] == "Hello!"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"]["prompt_tokens"] == 10
    assert chat["usage"]["completion_tokens"] == 5


def test_gemini_function_call_response_to_chat() -> None:
    resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "London"},
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {},
    }
    chat = _gemini_to_chat(resp, "gemini-test")
    tool_calls = chat["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"].startswith("call_")
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "London"}


def test_gemini_empty_candidates_returns_empty_response() -> None:
    resp = {"candidates": []}
    chat = _gemini_to_chat(resp, "gemini-test")
    assert chat["choices"][0]["message"]["content"] is None
    assert chat["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("gemini_reason", "expected"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "safety"),
        (None, None),
    ],
)
def test_finish_reason_normalization(
    gemini_reason: str | None,
    expected: str | None,
) -> None:
    assert _normalize_finish_reason(gemini_reason) == expected


def test_usage_extraction() -> None:
    meta = {
        "promptTokenCount": 10,
        "candidatesTokenCount": 20,
        "totalTokenCount": 30,
    }
    usage = _extract_usage(meta)
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


# ── Multimodal content translation ──────────────────────


def test_user_message_with_image_data_uri() -> None:
    """
    User message with image_url data URI translates to Gemini
    inlineData part.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/abc"},
                },
            ],
        },
    ]
    payload = _chat_to_gemini(messages, None, {})
    parts = payload["contents"][0]["parts"]
    # Two parts: text + inlineData.
    assert len(parts) == 2
    assert parts[0] == {"text": "Describe this"}
    assert parts[1] == {
        "inlineData": {"mimeType": "image/jpeg", "data": "/9j/abc"},
    }


def test_user_message_with_external_url_becomes_text() -> None:
    """
    External URL falls back to text placeholder since Gemini
    does not support URL references in content parts.
    """
    part = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/photo.png"},
    }
    result = _translate_part_to_gemini(part)
    assert result == {"text": "[image: https://example.com/photo.png]"}


def test_user_message_with_file_data() -> None:
    """
    input_file with file_data translates to Gemini inlineData.
    """
    part = {
        "type": "input_file",
        "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
    }
    result = _translate_part_to_gemini(part)
    assert result == {
        "inlineData": {
            "mimeType": "application/pdf",
            "data": "JVBERi0xLjQK",
        },
    }


def test_string_user_content_becomes_text_part() -> None:
    """
    String user content becomes a single text part —
    backward compatibility with text-only messages.
    """
    messages = [{"role": "user", "content": "Hello"}]
    payload = _chat_to_gemini(messages, None, {})
    assert payload["contents"][0]["parts"] == [{"text": "Hello"}]
