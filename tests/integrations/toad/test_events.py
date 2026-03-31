"""Tests for ACP event translation from agent-plane SSE events."""

from __future__ import annotations

import pytest

from integrations.toad.events import EventTranslator


@pytest.fixture
def translator() -> EventTranslator:
    """Fresh event translator instance."""
    return EventTranslator()


def test_text_delta_produces_agent_message_chunk(
    translator: EventTranslator,
) -> None:
    """Text deltas map to ``agent_message_chunk`` updates."""
    updates = translator.translate(
        "response.output_text.delta",
        {"delta": "Hello world"},
    )
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_message_chunk"
    content = updates[0]["content"]
    assert isinstance(content, dict)
    assert content["text"] == "Hello world"


def test_reasoning_delta_produces_thought_chunk(
    translator: EventTranslator,
) -> None:
    """Reasoning summary deltas map to ``agent_thought_chunk``."""
    updates = translator.translate(
        "response.reasoning_summary.delta",
        {"delta": "Let me think..."},
    )
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_thought_chunk"
    content = updates[0]["content"]
    assert isinstance(content, dict)
    assert content["text"] == "Let me think..."


@pytest.mark.parametrize(
    "event_type",
    [
        "response.reasoning.delta",
        "response.reasoning_summary.delta",
        "response.reasoning_summary_text.delta",
    ],
)
def test_all_reasoning_event_types_produce_thought_chunk(
    translator: EventTranslator,
    event_type: str,
) -> None:
    """All three reasoning event variants produce thought chunks."""
    updates = translator.translate(event_type, {"delta": "thinking"})
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_thought_chunk"


def test_function_call_item_done_produces_tool_call(
    translator: EventTranslator,
) -> None:
    """Completed function_call items map to ``tool_call`` updates."""
    updates = translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "search.web",
                "arguments": '{"query": "test"}',
            }
        },
    )
    assert len(updates) == 1
    update = updates[0]
    assert update["sessionUpdate"] == "tool_call"
    assert update["toolCallId"] == "call_abc"
    assert update["title"] == "search.web"
    assert update["status"] == "running"
    tool = update["tool"]
    assert isinstance(tool, dict)
    assert tool["name"] == "search.web"
    assert tool["parameters"] == '{"query": "test"}'


def test_function_call_output_produces_tool_call_update(
    translator: EventTranslator,
) -> None:
    """Completed function_call_output items map to ``tool_call_update``."""
    updates = translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": '{"results": [1, 2]}',
            }
        },
    )
    assert len(updates) == 1
    update = updates[0]
    assert update["sessionUpdate"] == "tool_call_update"
    assert update["toolCallId"] == "call_abc"
    assert update["status"] == "completed"
    content = update["content"]
    assert isinstance(content, dict)
    assert content["text"] == '{"results": [1, 2]}'


def test_response_completed_captures_response_id(
    translator: EventTranslator,
) -> None:
    """``response.completed`` stores the response ID for multi-turn."""
    assert translator.last_response_id is None
    updates = translator.translate(
        "response.completed",
        {"response": {"id": "resp_xyz", "status": "completed"}},
    )
    # No ACP update emitted, but ID is captured
    assert updates == []
    assert translator.last_response_id == "resp_xyz"


def test_unknown_event_type_ignored(
    translator: EventTranslator,
) -> None:
    """Unrecognized event types produce no updates."""
    updates = translator.translate(
        "response.some_unknown_event",
        {"foo": "bar"},
    )
    assert updates == []


def test_item_done_with_message_type_ignored(
    translator: EventTranslator,
) -> None:
    """Message-type items in output_item.done produce no updates.

    Messages are handled via text deltas, not item completion.
    """
    updates = translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        },
    )
    assert updates == []


def test_item_done_with_non_dict_item_ignored(
    translator: EventTranslator,
) -> None:
    """Malformed item data (non-dict) produces no updates."""
    updates = translator.translate(
        "response.output_item.done",
        {"item": "not a dict"},
    )
    assert updates == []


def test_multi_turn_response_id_tracking(
    translator: EventTranslator,
) -> None:
    """Response ID updates on each completed response."""
    translator.translate(
        "response.completed",
        {"response": {"id": "resp_1"}},
    )
    assert translator.last_response_id == "resp_1"

    translator.translate(
        "response.completed",
        {"response": {"id": "resp_2"}},
    )
    assert translator.last_response_id == "resp_2"


def test_response_completed_captures_conversation_id(
    translator: EventTranslator,
) -> None:
    """``response.completed`` extracts conversation ID from response."""
    translator.translate(
        "response.completed",
        {
            "response": {
                "id": "resp_1",
                "conversation": {"id": "conv_abc"},
            }
        },
    )
    assert translator.last_conversation_id == "conv_abc"


def test_response_completed_sets_stop_reason_end_turn(
    translator: EventTranslator,
) -> None:
    """``response.completed`` sets stop_reason to end_turn."""
    translator.translate(
        "response.completed",
        {"response": {"id": "resp_1"}},
    )
    assert translator.stop_reason == "end_turn"


def test_reset_for_prompt_clears_stop_reason(
    translator: EventTranslator,
) -> None:
    """reset_for_prompt clears stop_reason but keeps IDs."""
    translator.translate(
        "response.completed",
        {
            "response": {
                "id": "resp_1",
                "conversation": {"id": "conv_1"},
            }
        },
    )
    assert translator.stop_reason == "end_turn"
    translator.reset_for_prompt()
    assert translator.stop_reason is None
    # IDs persist across resets
    assert translator.last_response_id == "resp_1"
    assert translator.last_conversation_id == "conv_1"


def test_response_failed_sets_stop_reason(
    translator: EventTranslator,
) -> None:
    """``response.failed`` sets stop_reason to end_turn."""
    updates = translator.translate(
        "response.failed",
        {"response": {"id": "resp_f1"}},
    )
    assert updates == []
    assert translator.stop_reason == "end_turn"
    assert translator.last_response_id == "resp_f1"


@pytest.mark.parametrize(
    ("reason", "expected_stop"),
    [
        ("max_iterations", "max_turn_requests"),
        ("execution_timeout", "max_turn_requests"),
        ("max_output_tokens", "max_tokens"),
        ("unknown_reason", "end_turn"),
    ],
)
def test_response_incomplete_maps_reason(
    translator: EventTranslator,
    reason: str,
    expected_stop: str,
) -> None:
    """``response.incomplete`` maps reason to ACP stop reason."""
    translator.translate(
        "response.incomplete",
        {
            "response": {
                "id": "resp_inc",
                "incomplete_details": {"reason": reason},
            }
        },
    )
    assert translator.stop_reason == expected_stop
    assert translator.last_response_id == "resp_inc"


def test_response_cancelled_sets_stop_reason(
    translator: EventTranslator,
) -> None:
    """``response.cancelled`` sets stop_reason to cancelled."""
    updates = translator.translate(
        "response.cancelled",
        {"response": {"id": "resp_c1"}},
    )
    assert updates == []
    assert translator.stop_reason == "cancelled"
    assert translator.last_response_id == "resp_c1"


def test_response_error_emits_message_chunk(
    translator: EventTranslator,
) -> None:
    """``response.error`` emits an agent_message_chunk with error."""
    updates = translator.translate(
        "response.error",
        {"message": "Rate limit exceeded"},
    )
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_message_chunk"
    content = updates[0]["content"]
    assert isinstance(content, dict)
    assert "[Error]" in content["text"]
    assert "Rate limit exceeded" in content["text"]


def test_response_retry_emits_thought_chunk(
    translator: EventTranslator,
) -> None:
    """``response.retry`` emits an agent_thought_chunk with info."""
    updates = translator.translate(
        "response.retry",
        {"message": "Attempt 2 of 3"},
    )
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_thought_chunk"
    content = updates[0]["content"]
    assert isinstance(content, dict)
    assert "[Retry]" in content["text"]
    assert "Attempt 2 of 3" in content["text"]


def test_native_tool_item_produces_completed_tool_call(
    translator: EventTranslator,
) -> None:
    """Native tool items (web_search_call) produce completed tool_call."""
    updates = translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "web_search_call",
                "id": "ws_123",
                "query": "test search",
            }
        },
    )
    assert len(updates) == 1
    update = updates[0]
    assert update["sessionUpdate"] == "tool_call"
    assert update["toolCallId"] == "ws_123"
    assert update["title"] == "web_search_call"
    assert update["status"] == "completed"


def test_response_failed_captures_conversation_id(
    translator: EventTranslator,
) -> None:
    """Terminal events capture conversation_id like completed does."""
    translator.translate(
        "response.failed",
        {
            "response": {
                "id": "resp_f2",
                "conversation": {"id": "conv_fail"},
            }
        },
    )
    assert translator.last_conversation_id == "conv_fail"


def test_pending_client_tool_calls_detected(
    translator: EventTranslator,
) -> None:
    """Function calls without matching outputs are pending client tools."""
    # Simulate function_call streamed
    translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path": "/tmp/x"}',
            }
        },
    )
    # No function_call_output for call_1
    translator.translate(
        "response.completed",
        {"response": {"id": "resp_1"}},
    )
    pending = translator.pending_client_tool_calls
    assert len(pending) == 1
    assert pending[0]["call_id"] == "call_1"
    assert pending[0]["name"] == "read_file"
    assert pending[0]["arguments"] == '{"path": "/tmp/x"}'


def test_matched_tool_calls_not_pending(
    translator: EventTranslator,
) -> None:
    """Function calls with matching outputs are not pending."""
    translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call",
                "call_id": "call_2",
                "name": "web_search",
                "arguments": '{"q": "test"}',
            }
        },
    )
    # Server-side tool — output arrives
    translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "results here",
            }
        },
    )
    translator.translate(
        "response.completed",
        {"response": {"id": "resp_2"}},
    )
    assert translator.pending_client_tool_calls == []


def test_reset_for_prompt_clears_tool_tracking(
    translator: EventTranslator,
) -> None:
    """reset_for_prompt clears seen function calls and outputs."""
    translator.translate(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call",
                "call_id": "call_3",
                "name": "tool_x",
                "arguments": "{}",
            }
        },
    )
    assert len(translator.pending_client_tool_calls) == 1
    translator.reset_for_prompt()
    assert translator.pending_client_tool_calls == []
