"""Tests for llms._responses_to_chat — bidirectional translation."""

import pytest

from llms._responses_to_chat import (
    chat_response_to_response,
    chat_stream_to_response_events,
    responses_input_to_chat_messages,
)
from llms.types import (
    FunctionCallOutput,
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
    Usage,
)


# ── Input direction: Responses API -> Chat Completions ────


def test_instructions_become_system_message() -> None:
    messages = responses_input_to_chat_messages([], "Be helpful.")
    assert messages == [{"role": "system", "content": "Be helpful."}]


def test_no_instructions_no_system_message() -> None:
    items = [{"role": "user", "content": "Hi"}]
    messages = responses_input_to_chat_messages(items, None)
    assert messages == [{"role": "user", "content": "Hi"}]


def test_user_and_assistant_messages_passthrough() -> None:
    items = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    messages = responses_input_to_chat_messages(items, None)
    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_function_call_items_grouped_into_assistant_message() -> None:
    items = [
        {"role": "user", "content": "What's the weather?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "London"}',
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "get_time",
            "arguments": '{"tz": "UTC"}',
        },
    ]
    messages = responses_input_to_chat_messages(items, None)
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "What's the weather?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] is None
    assert len(messages[1]["tool_calls"]) == 2
    assert messages[1]["tool_calls"][0] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
    }


def test_function_call_output_becomes_tool_message() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Sunny, 22C",
        },
    ]
    messages = responses_input_to_chat_messages(items, None)
    # First: assistant with tool_calls, second: tool message
    assert len(messages) == 2
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Sunny, 22C",
    }


def test_full_conversation_round_trip() -> None:
    """
    Test a realistic conversation: user -> assistant tool call ->
    tool output -> assistant follow-up.
    """
    items = [
        {"role": "user", "content": "Weather in London?"},
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": '{"city": "London"}',
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "Rainy, 15C",
        },
        {"role": "assistant", "content": "It's rainy and 15C in London."},
    ]
    messages = responses_input_to_chat_messages(items, "Be concise.")
    assert len(messages) == 5
    assert messages[0] == {"role": "system", "content": "Be concise."}
    assert messages[1] == {"role": "user", "content": "Weather in London?"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == "c1"
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Rainy, 15C",
    }
    assert messages[4] == {
        "role": "assistant",
        "content": "It's rainy and 15C in London.",
    }


# ── Output direction: Chat Completions -> Responses API ───


def test_chat_text_response_to_response() -> None:
    chat_dict = {
        "id": "chatcmpl-123",
        "model": "gpt-5.4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    resp = chat_response_to_response(chat_dict)
    assert isinstance(resp, Response)
    assert resp.model == "gpt-5.4"
    assert len(resp.output) == 1
    assert isinstance(resp.output[0], MessageOutput)
    assert resp.output[0].content[0].text == "Hello!"
    assert resp.usage == Usage(
        input_tokens=10, output_tokens=5, total_tokens=15
    )


def test_chat_tool_calls_to_response() -> None:
    chat_dict = {
        "id": "chatcmpl-456",
        "model": "gpt-5.4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": None,
    }
    resp = chat_response_to_response(chat_dict)
    assert len(resp.output) == 1
    assert isinstance(resp.output[0], FunctionCallOutput)
    assert resp.output[0].call_id == "call_abc"
    assert resp.output[0].name == "get_weather"
    assert resp.output[0].arguments == '{"city": "London"}'


def test_chat_mixed_text_and_tool_calls() -> None:
    chat_dict = {
        "model": "gpt-5.4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "test"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    resp = chat_response_to_response(chat_dict)
    assert len(resp.output) == 2
    assert isinstance(resp.output[0], MessageOutput)
    assert isinstance(resp.output[1], FunctionCallOutput)


# ── Streaming: Chat Completions chunks -> events ──────────


def test_streaming_text_deltas() -> None:
    chunks = [
        {
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
        },
        {
            "choices": [{"delta": {"content": " world"}, "finish_reason": None}]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}]
        },
    ]
    events = list(chat_stream_to_response_events(iter(chunks), model="test"))

    # Two text deltas + one completed event
    assert len(events) == 3
    assert isinstance(events[0], ResponseTextDeltaEvent)
    assert events[0].delta == "Hello"
    assert isinstance(events[1], ResponseTextDeltaEvent)
    assert events[1].delta == " world"
    assert isinstance(events[2], ResponseCompletedEvent)
    assert events[2].response.output[0].content[0].text == "Hello world"


def test_streaming_tool_calls() -> None:
    chunks = [
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather"},
                    }]
                },
                "finish_reason": None,
            }]
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"city":'},
                    }]
                },
                "finish_reason": None,
            }]
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '"London"}'},
                    }]
                },
                "finish_reason": None,
            }]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = list(chat_stream_to_response_events(iter(chunks), model="test"))

    # No text deltas, just the completed event
    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    fc = completed.response.output[0]
    assert isinstance(fc, FunctionCallOutput)
    assert fc.call_id == "call_1"
    assert fc.name == "get_weather"
    assert fc.arguments == '{"city":"London"}'


def test_streaming_with_usage() -> None:
    chunks = [
        {
            "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        },
    ]
    events = list(chat_stream_to_response_events(iter(chunks), model="test"))
    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.usage == Usage(
        input_tokens=5, output_tokens=1, total_tokens=6
    )
