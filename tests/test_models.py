"""Tests for runtime data models — ConversationItem validation."""

import pytest
from pydantic import ValidationError

from agent_plane.runtime.models import (
    ConversationItem,
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
    NewConversationItem,
    ReasoningData,
    parse_item_data,
)


# ── MessageData ────────────────────────────────────────


class TestMessageData:
    def test_user_message(self):
        msg = MessageData(role="user", content=[{"type": "input_text", "text": "hi"}])
        assert msg.role == "user"
        assert msg.agent is None

    def test_assistant_message(self):
        msg = MessageData(
            role="assistant",
            agent="my-agent",
            content=[{"type": "output_text", "text": "hello"}],
        )
        assert msg.role == "assistant"
        assert msg.agent == "my-agent"

    def test_assistant_requires_agent(self):
        with pytest.raises(ValidationError, match="assistant messages require 'agent'"):
            MessageData(role="assistant", content=[])

    def test_user_message_excludes_none_agent(self):
        msg = MessageData(role="user", content=[])
        dumped = msg.model_dump(exclude_none=True)
        assert "agent" not in dumped
        assert dumped == {"role": "user", "content": []}

    def test_assistant_message_includes_agent(self):
        msg = MessageData(role="assistant", agent="my-agent", content=[])
        dumped = msg.model_dump(exclude_none=True)
        assert dumped == {"role": "assistant", "agent": "my-agent", "content": []}

    def test_serialization_alias(self):
        msg = MessageData(role="assistant", agent="my-agent", content=[])
        dumped = msg.model_dump(exclude_none=True, by_alias=True)
        assert dumped == {"role": "assistant", "model": "my-agent", "content": []}

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            MessageData(role="system", content=[])


# ── FunctionCallData ───────────────────────────────────


class TestFunctionCallData:
    def test_valid(self):
        fc = FunctionCallData(
            agent="my-agent",
            name="get_weather",
            arguments='{"city": "SF"}',
            call_id="call_1",
        )
        assert fc.name == "get_weather"
        assert fc.call_id == "call_1"

    def test_missing_call_id(self):
        with pytest.raises(ValidationError, match="call_id"):
            FunctionCallData(agent="a", name="b", arguments="c")

    def test_missing_agent(self):
        with pytest.raises(ValidationError, match="agent"):
            FunctionCallData(name="b", arguments="c", call_id="d")


# ── FunctionCallOutputData ─────────────────────────────


class TestFunctionCallOutputData:
    def test_valid(self):
        fco = FunctionCallOutputData(call_id="call_1", output='{"temp": 72}')
        assert fco.call_id == "call_1"

    def test_missing_output(self):
        with pytest.raises(ValidationError, match="output"):
            FunctionCallOutputData(call_id="call_1")


# ── ReasoningData ──────────────────────────────────────


class TestReasoningData:
    def test_valid_minimal(self):
        r = ReasoningData(agent="my-agent", summary=[{"type": "summary_text", "text": "..."}])
        assert r.content is None
        assert r.encrypted_content is None

    def test_valid_full(self):
        r = ReasoningData(
            agent="my-agent",
            summary=[],
            content=[{"type": "text", "text": "thinking..."}],
            encrypted_content="enc_abc",
        )
        assert r.content is not None
        assert r.encrypted_content == "enc_abc"

    def test_missing_agent(self):
        with pytest.raises(ValidationError, match="agent"):
            ReasoningData(summary=[])


# ── NewConversationItem ────────────────────────────────


class TestNewConversationItem:
    def test_user_message(self):
        item = NewConversationItem(
            type="message",
            response_id="resp_1",
            data=MessageData(role="user", content=[]),
        )
        assert item.type == "message"
        assert item.data.role == "user"

    def test_assistant_message(self):
        item = NewConversationItem(
            type="message",
            response_id="resp_1",
            data=MessageData(role="assistant", agent="my-agent", content=[]),
        )
        assert item.data.agent == "my-agent"

    def test_function_call(self):
        item = NewConversationItem(
            type="function_call",
            response_id="resp_1",
            data=FunctionCallData(
                agent="my-agent", name="fn", arguments="{}", call_id="c1"
            ),
        )
        assert item.data.name == "fn"

    def test_function_call_output(self):
        item = NewConversationItem(
            type="function_call_output",
            response_id="resp_1",
            data=FunctionCallOutputData(call_id="c1", output="{}"),
        )
        assert item.data.call_id == "c1"

    def test_reasoning(self):
        item = NewConversationItem(
            type="reasoning",
            response_id="resp_1",
            data=ReasoningData(agent="my-agent", summary=[]),
        )
        assert item.type == "reasoning"

    def test_type_data_mismatch_rejected(self):
        with pytest.raises(
            ValidationError, match="requires FunctionCallData, got MessageData"
        ):
            NewConversationItem(
                type="function_call",
                response_id="resp_1",
                data=MessageData(role="user", content=[]),
            )

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError, match="unknown item type"):
            NewConversationItem(
                type="unknown",
                response_id="resp_1",
                data=MessageData(role="user", content=[]),
            )


# ── ConversationItem ───────────────────────────────────


class TestConversationItem:
    def test_valid(self):
        item = ConversationItem(
            id="item_1",
            type="message",
            response_id="resp_1",
            created_at=1700000000,
            data=MessageData(role="user", content=[{"type": "input_text", "text": "hi"}]),
        )
        assert item.id == "item_1"
        assert item.created_at == 1700000000

    def test_type_data_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            ConversationItem(
                id="item_1",
                type="reasoning",
                response_id="resp_1",
                created_at=1700000000,
                data=MessageData(role="user", content=[]),
            )


# ── parse_item_data ────────────────────────────────────


class TestParseItemData:
    def test_parse_message(self):
        data = parse_item_data("message", {"role": "user", "content": []})
        assert isinstance(data, MessageData)
        assert data.role == "user"

    def test_parse_function_call(self):
        data = parse_item_data(
            "function_call",
            {"agent": "a", "name": "fn", "arguments": "{}", "call_id": "c1"},
        )
        assert isinstance(data, FunctionCallData)

    def test_parse_function_call_output(self):
        data = parse_item_data(
            "function_call_output", {"call_id": "c1", "output": "{}"}
        )
        assert isinstance(data, FunctionCallOutputData)

    def test_parse_reasoning(self):
        data = parse_item_data("reasoning", {"agent": "a", "summary": []})
        assert isinstance(data, ReasoningData)

    def test_parse_unknown_type(self):
        with pytest.raises(ValueError, match="unknown item type"):
            parse_item_data("bogus", {})

    def test_parse_invalid_data(self):
        with pytest.raises(ValidationError):
            parse_item_data("function_call", {"agent": "a"})
