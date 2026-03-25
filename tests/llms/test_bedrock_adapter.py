"""Tests for llms.adapters.bedrock — translation logic."""

import json

from llms.adapters.bedrock import (
    _build_converse_kwargs,
    _converse_to_chat,
    _convert_tools,
    _messages_to_converse,
)


# ── Request translation ──────────────────────────────────


def test_system_messages_extracted_as_system_prompts() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    converse_msgs, system_prompts = _messages_to_converse(messages)
    assert system_prompts == [{"text": "Be helpful."}]
    assert len(converse_msgs) == 1
    assert converse_msgs[0]["role"] == "user"


def test_user_message_converted_to_text_block() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    converse_msgs, _ = _messages_to_converse(messages)
    assert converse_msgs[0] == {
        "role": "user",
        "content": [{"text": "Hello"}],
    }


def test_assistant_message_with_text() -> None:
    messages = [{"role": "assistant", "content": "Hi there"}]
    converse_msgs, _ = _messages_to_converse(messages)
    assert converse_msgs[0] == {
        "role": "assistant",
        "content": [{"text": "Hi there"}],
    }


def test_assistant_tool_calls_converted_to_tool_use() -> None:
    messages = [
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
        }
    ]
    converse_msgs, _ = _messages_to_converse(messages)
    msg = converse_msgs[0]
    assert msg["role"] == "assistant"
    tu = msg["content"][0]["toolUse"]
    assert tu["toolUseId"] == "call_1"
    assert tu["name"] == "get_weather"
    assert tu["input"] == {"city": "London"}


def test_tool_messages_converted_to_tool_result() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Sunny, 22C",
        }
    ]
    converse_msgs, _ = _messages_to_converse(messages)
    msg = converse_msgs[0]
    assert msg["role"] == "user"
    tr = msg["content"][0]["toolResult"]
    assert tr["toolUseId"] == "call_1"
    assert tr["content"] == [{"text": "Sunny, 22C"}]


def test_inference_config_mapped() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    extra = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100}
    kwargs = _build_converse_kwargs(messages, "model-id", None, extra)
    config = kwargs["inferenceConfig"]
    assert config["temperature"] == 0.7
    assert config["topP"] == 0.9
    assert config["maxTokens"] == 100


def test_stop_sequences_mapped() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    extra = {"stop": ["END", "STOP"]}
    kwargs = _build_converse_kwargs(messages, "model-id", None, extra)
    assert kwargs["inferenceConfig"]["stopSequences"] == ["END", "STOP"]


def test_single_stop_string_wrapped_in_list() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    extra = {"stop": "END"}
    kwargs = _build_converse_kwargs(messages, "model-id", None, extra)
    assert kwargs["inferenceConfig"]["stopSequences"] == ["END"]


def test_tools_converted_to_tool_spec() -> None:
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
    spec = result[0]["toolSpec"]
    assert spec["name"] == "get_weather"
    assert spec["description"] == "Get weather"
    assert spec["inputSchema"] == {
        "json": {"type": "object", "properties": {}}
    }


def test_tool_config_added_to_kwargs() -> None:
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
    kwargs = _build_converse_kwargs(messages, "model-id", tools, {})
    assert "toolConfig" in kwargs
    assert "tools" in kwargs["toolConfig"]


# ── Response translation ─────────────────────────────────


def test_converse_text_response_to_chat() -> None:
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello!"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
        },
    }
    chat = _converse_to_chat(response, "bedrock-model")
    assert chat["model"] == "bedrock-model"
    assert chat["choices"][0]["message"]["content"] == "Hello!"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"]["prompt_tokens"] == 10
    assert chat["usage"]["completion_tokens"] == 5


def test_converse_tool_use_response_to_chat() -> None:
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "tu_1",
                        "name": "get_weather",
                        "input": {"city": "London"},
                    }
                }],
            }
        },
        "stopReason": "tool_use",
        "usage": {},
    }
    chat = _converse_to_chat(response, "bedrock-model")
    tool_calls = chat["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tu_1"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "city": "London"
    }
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_converse_mixed_text_and_tool_use() -> None:
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Let me check."},
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "search",
                            "input": {"q": "test"},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {},
    }
    chat = _converse_to_chat(response, "bedrock-model")
    assert chat["choices"][0]["message"]["content"] == "Let me check."
    assert len(chat["choices"][0]["message"]["tool_calls"]) == 1
