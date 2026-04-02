"""Tests for the Executor ABC, DefaultExecutor, and event serialization.

Covers: event roundtrip serialization, response extraction helpers,
stream consumption, retry with error classification, and DefaultExecutor
construction from AgentSpec.

Tests monkeypatch ``_create_stream`` and ``_get_model_context_window``
to avoid real LLM calls. All data objects use real types from
``agent_plane.llms.types``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_plane.llms.errors import (
    ContextWindowExceededError,
    PermanentLLMError,
    RetryableLLMError,
)
from agent_plane.llms.types import (
    FunctionCallOutput,
    MessageOutput,
    NativeToolOutputAddedEvent,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseReasoningStartedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from agent_plane.llms.types import NativeToolOutput as LLMNativeToolOutput
from agent_plane.runtime.executor import (
    ContextWindowExceeded,
    DefaultExecutor,
    ExecutorContext,
    ExecutorError,
    NativeToolOutput,
    ReasoningChunk,
    TextChunk,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
    _build_responses_args,
    _consume_stream,
    _extract_native_tool_items,
    _extract_text,
    _extract_tool_calls,
    _open_stream_with_retry,
    _run_streaming_turn,
    _yield_final_events,
    dict_to_event,
    event_to_dict,
)
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig, RetryConfig

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def llm_config() -> LLMConfig:
    """
    Minimal LLM config for tests.

    :returns: An :class:`LLMConfig` with fast retry settings.
    """
    return LLMConfig(
        model="openai/gpt-4o",
        request_timeout=30,
        retry=RetryConfig(
            max_attempts=3,
            backoff_base=0.001,
            backoff_max=0.01,
        ),
    )


@pytest.fixture()
def executor_context() -> ExecutorContext:
    """
    Minimal executor context for tests.

    :returns: An :class:`ExecutorContext` with stub values.
    """
    return ExecutorContext(
        task_id="task_test_123",
        conversation_id="conv_test_456",
        storage_dir=Path("/tmp/test-storage"),
        await_tool_output=lambda _req: ToolResult(content="stub output", status="success"),
    )


def _make_text_response(text: str) -> Response:
    """
    Build a completed LLM Response with a single text output.

    :param text: The assistant text, e.g. ``"Hello!"``.
    :returns: A real :class:`Response` with one :class:`MessageOutput`.
    """
    return Response(
        output=[MessageOutput(content=[OutputText(text=text)])],
        model="test-model",
    )


def _make_tool_call_response(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> Response:
    """
    Build a completed LLM Response with a single tool call.

    :param call_id: Tool call ID, e.g. ``"call_abc"``.
    :param name: Tool name, e.g. ``"get_weather"``.
    :param arguments: Tool arguments dict.
    :returns: A real :class:`Response` with one :class:`FunctionCallOutput`.
    """
    return Response(
        output=[
            FunctionCallOutput(
                call_id=call_id,
                name=name,
                arguments=json.dumps(arguments),
            )
        ],
        model="test-model",
    )


# ── Event serialization roundtrip ──────────────────────────────


@pytest.mark.parametrize(
    "event",
    [
        TextChunk(text="Hello"),
        ReasoningChunk(delta="thinking...", event_type="reasoning_text"),
        NativeToolOutput(item={"type": "web_search_call", "id": "ws_1"}),
        ToolCallRequested(
            call_id="call_abc",
            name="get_weather",
            arguments={"city": "London"},
        ),
        TurnComplete(text="Done."),
        TurnComplete(text=None),
        ContextWindowExceeded(max_tokens=128000, actual_tokens=142000),
        ExecutorError(message="auth failed", code="401"),
        ExecutorError(message="unknown error", code=None),
    ],
    ids=[
        "text_chunk",
        "reasoning_chunk",
        "native_tool_output",
        "tool_call_requested",
        "turn_complete_with_text",
        "turn_complete_no_text",
        "context_window_exceeded",
        "executor_error_with_code",
        "executor_error_no_code",
    ],
)
def test_event_serialization_roundtrip(
    event: TextChunk
    | ReasoningChunk
    | NativeToolOutput
    | ToolCallRequested
    | TurnComplete
    | ContextWindowExceeded
    | ExecutorError,
) -> None:
    """
    Every executor event type must survive a serialize→deserialize roundtrip.

    This is critical for DBOS @step caching: events are serialized to JSON
    on first execution and deserialized on replay. A broken roundtrip means
    cached events lose data on crash recovery.

    :param event: The executor event to roundtrip.
    """
    serialized = event_to_dict(event)

    # Serialized form must be a dict with a "type" key matching the class name.
    assert isinstance(serialized, dict)
    assert serialized["type"] == type(event).__name__

    deserialized = dict_to_event(serialized)

    # Roundtrip must produce an identical event. If not, the @step cache
    # would return corrupted data on replay.
    assert deserialized == event


def test_event_to_dict_unknown_type_raises() -> None:
    """
    ``event_to_dict`` must reject unknown event types with a clear error.
    """
    with pytest.raises(ValueError, match="Unknown event type"):
        event_to_dict("not an event")  # type: ignore[arg-type]


def test_dict_to_event_unknown_type_raises() -> None:
    """
    ``dict_to_event`` must reject dicts with unknown type keys.
    """
    with pytest.raises(ValueError, match="Unknown event type"):
        dict_to_event({"type": "FakeEventType", "data": 42})


# ── ToolCallObserved roundtrip (separate — more fields) ─────────


def test_tool_call_observed_serialization_roundtrip() -> None:
    """
    ToolCallObserved has 6 fields — verify all survive the roundtrip.
    """
    from agent_plane.runtime.executor import ToolCallObserved

    event = ToolCallObserved(
        call_id="call_xyz",
        name="Bash",
        arguments={"command": "ls -la"},
        result="/home/user\n",
        status="success",
        duration_ms=342.1,
    )
    serialized = event_to_dict(event)
    deserialized = dict_to_event(serialized)

    # All 6 fields must match after roundtrip. duration_ms is a float —
    # JSON preserves it, but a broken serializer might lose precision.
    assert deserialized == event


# ── _build_responses_args ───────────────────────────────────────


def test_build_responses_args_basic() -> None:
    """
    Basic call with model, tools, and no reasoning effort.
    """
    result = _build_responses_args(
        model="openai/gpt-4o",
        tools=[{"type": "function", "name": "search"}],
        extra={"temperature": 0.7},
    )

    # Model must be in kwargs — it's required by responses.create().
    assert result.kwargs["model"] == "openai/gpt-4o"
    # Tools must be in kwargs when non-empty.
    assert result.kwargs["tools"] == [{"type": "function", "name": "search"}]
    # Extra kwargs must pass through.
    assert result.kwargs["temperature"] == 0.7
    # No reasoning_effort → reasoning must be None.
    assert result.reasoning is None


def test_build_responses_args_with_reasoning() -> None:
    """
    When ``reasoning_effort`` is in extra, it must be extracted and
    mapped to the ``reasoning`` parameter with ``summary="detailed"``.
    """
    result = _build_responses_args(
        model="openai/o3",
        tools=[],
        extra={"reasoning_effort": "high", "temperature": 0.5},
    )

    # reasoning_effort must be extracted from kwargs — not passed through
    # as a raw kwarg, which would be invalid for responses.create().
    assert "reasoning_effort" not in result.kwargs
    # Mapped reasoning must include summary="detailed" for streaming.
    assert result.reasoning == {"effort": "high", "summary": "detailed"}
    # Other extra kwargs must still pass through.
    assert result.kwargs["temperature"] == 0.5


def test_build_responses_args_no_tools_omits_tools_key() -> None:
    """
    When tools list is empty, the ``tools`` key must be absent from kwargs.

    Sending ``tools=[]`` to some providers causes errors.
    """
    result = _build_responses_args(
        model="openai/gpt-4o",
        tools=[],
        extra={},
    )

    # Empty tools must NOT be included — some providers reject tools=[].
    assert "tools" not in result.kwargs


# ── _extract_tool_calls ─────────────────────────────────────────


def test_extract_tool_calls_from_response() -> None:
    """
    Tool calls in the response output must be extracted with parsed arguments.
    """
    response = _make_tool_call_response(
        call_id="call_123",
        name="get_weather",
        arguments={"city": "London"},
    )

    result = _extract_tool_calls(response)

    # Exactly one tool call must be extracted from a single-tool response.
    assert len(result) == 1
    assert result[0].call_id == "call_123"
    assert result[0].name == "get_weather"
    # Arguments must be parsed from JSON string → dict.
    assert result[0].arguments == {"city": "London"}


def test_extract_tool_calls_empty_for_text_only() -> None:
    """
    A text-only response must produce zero tool calls.
    """
    response = _make_text_response("Hello!")

    result = _extract_tool_calls(response)

    # No FunctionCallOutput items → empty list.
    assert result == []


# ── _extract_text ────────────────────────────────────────────────


def test_extract_text_from_message() -> None:
    """
    Text must be extracted from a MessageOutput with OutputText.
    """
    response = _make_text_response("Hello, world!")

    result = _extract_text(response)

    assert result == "Hello, world!"


def test_extract_text_returns_none_for_tool_only() -> None:
    """
    A response with only tool calls and no MessageOutput must return None.
    """
    response = _make_tool_call_response("call_1", "search", {"q": "test"})

    result = _extract_text(response)

    assert result is None


# ── _extract_native_tool_items ──────────────────────────────────


def test_extract_native_tool_items() -> None:
    """
    Native tool outputs in the response must be wrapped as NativeToolOutput events.
    """
    native_data = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    response = Response(
        output=[LLMNativeToolOutput(data=native_data)],
        model="test-model",
    )

    result = _extract_native_tool_items(response)

    # The raw dict must be preserved in the NativeToolOutput wrapper.
    assert len(result) == 1
    assert result[0].item == native_data


def test_extract_native_tool_items_empty() -> None:
    """
    A response with no native tool outputs must return an empty list.
    """
    response = _make_text_response("Just text.")

    result = _extract_native_tool_items(response)

    assert result == []


# ── _consume_stream ──────────────────────────────────────────────


def test_consume_stream_text_deltas() -> None:
    """
    Text delta events in the stream must yield TextChunk events.
    """
    response = _make_text_response("Hello!")
    stream: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="Hel"),
        ResponseTextDeltaEvent(delta="lo!"),
        ResponseCompletedEvent(response=response),
    ]

    events = list(_consume_stream(iter(stream)))

    # 2 text chunks + TurnComplete. If fewer, a delta was dropped.
    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    assert len(text_chunks) == 2
    assert text_chunks[0].text == "Hel"
    assert text_chunks[1].text == "lo!"

    # Final event must be TurnComplete with the assembled text.
    turn_complete = events[-1]
    assert isinstance(turn_complete, TurnComplete)
    assert turn_complete.text == "Hello!"


def test_consume_stream_reasoning_events() -> None:
    """
    All three reasoning event types must yield ReasoningChunk events.
    """
    response = _make_text_response("Done thinking.")
    stream: list[ResponseStreamEvent] = [
        ResponseReasoningStartedEvent(),
        ResponseReasoningTextDeltaEvent(delta="Let me think..."),
        ResponseReasoningSummaryTextDeltaEvent(delta="Summary: ok"),
        ResponseCompletedEvent(response=response),
    ]

    events = list(_consume_stream(iter(stream)))
    reasoning = [e for e in events if isinstance(e, ReasoningChunk)]

    # 3 reasoning events: started, text delta, summary delta.
    assert len(reasoning) == 3
    assert reasoning[0].event_type == "reasoning_started"
    assert reasoning[0].delta == ""
    assert reasoning[1].event_type == "reasoning_text"
    assert reasoning[1].delta == "Let me think..."
    assert reasoning[2].event_type == "reasoning_summary"
    assert reasoning[2].delta == "Summary: ok"


def test_consume_stream_native_tool_during_streaming() -> None:
    """
    NativeToolOutputAddedEvent during streaming must yield NativeToolOutput.
    """
    native_data = {"type": "web_search_call", "id": "ws_1"}
    response = _make_text_response("Found results.")
    stream: list[ResponseStreamEvent] = [
        NativeToolOutputAddedEvent(item=native_data),
        ResponseCompletedEvent(response=response),
    ]

    events = list(_consume_stream(iter(stream)))
    native_events = [e for e in events if isinstance(e, NativeToolOutput)]

    # Native tool output must be yielded during streaming, not just at completion.
    assert len(native_events) >= 1
    assert native_events[0].item == native_data


def test_consume_stream_tool_calls() -> None:
    """
    A response with tool calls must yield ToolCallRequested events
    and a TurnComplete with text=None.
    """
    response = _make_tool_call_response("call_1", "search", {"q": "test"})
    stream: list[ResponseStreamEvent] = [
        ResponseCompletedEvent(response=response),
    ]

    events = list(_consume_stream(iter(stream)))
    tool_events = [e for e in events if isinstance(e, ToolCallRequested)]

    # One tool call must be extracted from the completed response.
    assert len(tool_events) == 1
    assert tool_events[0].call_id == "call_1"
    assert tool_events[0].name == "search"
    assert tool_events[0].arguments == {"q": "test"}

    # TurnComplete.text must be None when tool calls are present —
    # the agent loop needs to execute tools, not emit text.
    turn_complete = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_complete) == 1
    assert turn_complete[0].text is None


def test_consume_stream_no_completed_event() -> None:
    """
    If the stream ends without ResponseCompletedEvent (e.g. mid-stream
    error), TurnComplete(text=None) must still be yielded.
    """
    stream: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="partial"),
    ]

    events = list(_consume_stream(iter(stream)))

    # Must still end with TurnComplete even if response.completed never arrived.
    # If missing, the workflow would hang waiting for turn completion.
    turn_complete = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_complete) == 1
    assert turn_complete[0].text is None


# ── _yield_final_events ──────────────────────────────────────────


def test_yield_final_events_text_only() -> None:
    """
    Text-only response must yield TurnComplete with the text.
    """
    response = _make_text_response("Hello!")

    events = list(_yield_final_events(response))

    # Final event must be TurnComplete with the text.
    assert events[-1] == TurnComplete(text="Hello!")


def test_yield_final_events_with_tool_calls() -> None:
    """
    Response with tool calls must yield ToolCallRequested events
    and TurnComplete(text=None).
    """
    response = _make_tool_call_response("call_1", "search", {"q": "x"})

    events = list(_yield_final_events(response))
    tool_events = [e for e in events if isinstance(e, ToolCallRequested)]

    assert len(tool_events) == 1
    assert tool_events[0].name == "search"
    # TurnComplete.text must be None when tool calls are present.
    assert events[-1] == TurnComplete(text=None)


def test_yield_final_events_with_native_tools() -> None:
    """
    Native tool items in the completed response must be yielded.
    """
    native_data = {"type": "web_search_call", "id": "ws_1"}
    response = Response(
        output=[
            LLMNativeToolOutput(data=native_data),
            MessageOutput(content=[OutputText(text="Results here.")]),
        ],
        model="test-model",
    )

    events = list(_yield_final_events(response))
    native = [e for e in events if isinstance(e, NativeToolOutput)]

    # Native tool items from the completed response must appear in events.
    assert len(native) == 1
    assert native[0].item == native_data


# ── _open_stream_with_retry ──────────────────────────────────────


def test_open_stream_with_retry_succeeds_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On success, the stream iterator is returned without retry.
    """
    expected_stream = iter([ResponseTextDeltaEvent(delta="hi")])

    monkeypatch.setattr(
        "agent_plane.runtime.executor._create_stream",
        lambda *_args, **_kw: expected_stream,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    result = _open_stream_with_retry(
        input_items=[],
        instructions="test",
        args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
        connection=None,
        timeout=30,
        retry_config=RetryConfig(max_attempts=3, backoff_base=0.001, backoff_max=0.01),
    )

    # Must return the same iterator object from _create_stream.
    assert result is expected_stream


def test_open_stream_with_retry_retries_on_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Transient errors must be retried up to max_attempts, then succeed.
    """
    call_count = 0
    expected_stream = iter([ResponseTextDeltaEvent(delta="ok")])

    def _failing_then_succeeding(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Fail with a retryable 429 on the first call, succeed on the second.

        :returns: Stream iterator on the second call.
        """
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            import httpx

            request = httpx.Request("POST", "http://test")
            response = httpx.Response(429, text="rate limited", request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return expected_stream

    monkeypatch.setattr(
        "agent_plane.runtime.executor._create_stream",
        _failing_then_succeeding,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    result = _open_stream_with_retry(
        input_items=[],
        instructions="test",
        args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
        connection=None,
        timeout=30,
        retry_config=RetryConfig(
            max_attempts=3,
            backoff_base=0.001,
            backoff_max=0.01,
            status_codes=[429],
        ),
    )

    # 2 calls: first fails with 429 (retried), second succeeds.
    assert call_count == 2
    assert result is expected_stream


def test_open_stream_with_retry_raises_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Permanent LLM errors (e.g. 401 auth failure) must not be retried.
    """

    def _permanent_failure(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Always raise a 401 auth error.

        :raises httpx.HTTPStatusError: 401 Unauthorized.
        """
        import httpx

        request = httpx.Request("POST", "http://test")
        response = httpx.Response(401, text="unauthorized", request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(
        "agent_plane.runtime.executor._create_stream",
        _permanent_failure,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    # 401 is not in the retryable list → PermanentLLMError must be raised
    # immediately without exhausting retries.
    with pytest.raises(PermanentLLMError):
        _open_stream_with_retry(
            input_items=[],
            instructions="test",
            args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
            connection=None,
            timeout=30,
            retry_config=RetryConfig(
                max_attempts=3,
                backoff_base=0.001,
                backoff_max=0.01,
            ),
        )


def test_open_stream_with_retry_raises_context_window_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ContextWindowExceededError must propagate immediately (not retried).
    """

    def _context_overflow(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Raise a context window exceeded error.

        :raises ContextWindowExceededError: Always.
        """
        raise ContextWindowExceededError(
            "too many tokens",
            code="context_length_exceeded",
            max_context_tokens=128000,
            actual_tokens=142000,
        )

    monkeypatch.setattr(
        "agent_plane.runtime.executor._create_stream",
        _context_overflow,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    with pytest.raises(ContextWindowExceededError) as exc_info:
        _open_stream_with_retry(
            input_items=[],
            instructions="test",
            args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
            connection=None,
            timeout=30,
            retry_config=RetryConfig(max_attempts=3, backoff_base=0.001, backoff_max=0.01),
        )

    # Token counts must be preserved through the raise chain.
    assert exc_info.value.max_context_tokens == 128000
    assert exc_info.value.actual_tokens == 142000


# ── _run_streaming_turn ──────────────────────────────────────────


def test_run_streaming_turn_yields_events_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    On a successful stream, events from _consume_stream must be yielded.
    """
    response = _make_text_response("Hi!")
    stream: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="Hi!"),
        ResponseCompletedEvent(response=response),
    ]

    monkeypatch.setattr(
        "agent_plane.runtime.executor._open_stream_with_retry",
        lambda *_args, **_kw: iter(stream),
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    events = list(
        _run_streaming_turn(
            task_id="task_1",
            input_items=[],
            instructions="test",
            args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
            connection=None,
            timeout=30,
            retry_config=RetryConfig(max_attempts=1),
        )
    )

    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    # One text delta must be yielded from the stream.
    assert len(text_chunks) == 1
    assert text_chunks[0].text == "Hi!"

    # Must end with TurnComplete carrying the full text.
    assert events[-1] == TurnComplete(text="Hi!")


def test_run_streaming_turn_context_window_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ContextWindowExceededError must yield a ContextWindowExceeded event.
    """

    def _overflow(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Raise context window exceeded.

        :raises ContextWindowExceededError: Always.
        """
        raise ContextWindowExceededError(
            "too big",
            code="context_length_exceeded",
            max_context_tokens=128000,
            actual_tokens=150000,
        )

    monkeypatch.setattr(
        "agent_plane.runtime.executor._open_stream_with_retry",
        _overflow,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    events = list(
        _run_streaming_turn(
            task_id="task_1",
            input_items=[],
            instructions="test",
            args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
            connection=None,
            timeout=30,
            retry_config=RetryConfig(max_attempts=1),
        )
    )

    # Must yield exactly one ContextWindowExceeded event — no TurnComplete.
    # If a TurnComplete also appears, the caller would try to process a
    # response that doesn't exist.
    assert len(events) == 1
    assert isinstance(events[0], ContextWindowExceeded)
    assert events[0].max_tokens == 128000
    assert events[0].actual_tokens == 150000


def test_run_streaming_turn_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    PermanentLLMError must yield an ExecutorError event.
    """

    def _auth_fail(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Raise a permanent auth error.

        :raises PermanentLLMError: Always.
        """
        raise PermanentLLMError("unauthorized", code="401")

    monkeypatch.setattr(
        "agent_plane.runtime.executor._open_stream_with_retry",
        _auth_fail,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    events = list(
        _run_streaming_turn(
            task_id="task_1",
            input_items=[],
            instructions="test",
            args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
            connection=None,
            timeout=30,
            retry_config=RetryConfig(max_attempts=1),
        )
    )

    # Must yield exactly one ExecutorError — no TurnComplete.
    assert len(events) == 1
    assert isinstance(events[0], ExecutorError)
    assert "unauthorized" in events[0].message
    assert events[0].code == "401"


def test_run_streaming_turn_retryable_error_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When all retries are exhausted, RetryableLLMError must yield ExecutorError.
    """

    def _retryable_fail(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Raise a retryable error (simulating exhausted retries).

        :raises RetryableLLMError: Always.
        """
        raise RetryableLLMError("rate limited", code="429")

    monkeypatch.setattr(
        "agent_plane.runtime.executor._open_stream_with_retry",
        _retryable_fail,
    )

    from agent_plane.runtime.executor import _ResponsesCallArgs

    events = list(
        _run_streaming_turn(
            task_id="task_1",
            input_items=[],
            instructions="test",
            args=_ResponsesCallArgs(kwargs={"model": "test"}, reasoning=None),
            connection=None,
            timeout=30,
            retry_config=RetryConfig(max_attempts=1),
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], ExecutorError)
    assert events[0].code == "429"


# ── DefaultExecutor.from_spec ─────────────────────────────────


def test_default_executor_from_spec() -> None:
    """
    ``from_spec`` must extract the LLM config from the AgentSpec.
    """
    spec = AgentSpec(
        spec_version=1,
        name="test-agent",
        llm=LLMConfig(model="openai/gpt-4o", request_timeout=120),
    )

    executor = DefaultExecutor.from_spec(spec)

    # The executor's internal config must match the spec's LLM config.
    assert executor._llm_config.model == "openai/gpt-4o"
    assert executor._llm_config.request_timeout == 120


def test_default_executor_from_spec_asserts_on_missing_llm() -> None:
    """
    ``from_spec`` must raise AssertionError when spec.llm is None.
    """
    spec = AgentSpec(spec_version=1, name="no-llm-agent")

    with pytest.raises(AssertionError):
        DefaultExecutor.from_spec(spec)


# ── DefaultExecutor.max_context_tokens ──────────────────────────


def test_max_context_tokens_returns_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the model is known, max_context_tokens returns the window size.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.executor._get_model_context_window",
        lambda model: 128000,
    )

    executor = DefaultExecutor(
        llm_config=LLMConfig(model="openai/gpt-4o"),
    )

    # Must return the value from the model registry lookup.
    assert executor.max_context_tokens() == 128000


def test_max_context_tokens_returns_none_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the model is not in the registry, max_context_tokens returns None.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.executor._get_model_context_window",
        lambda model: None,
    )

    executor = DefaultExecutor(
        llm_config=LLMConfig(model="unknown/custom-model"),
    )

    # None signals the workflow to skip compaction and @step wrapping.
    assert executor.max_context_tokens() is None


# ── DefaultExecutor.run_turn ─────────────────────────────────────


def test_run_turn_yields_events(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: LLMConfig,
    executor_context: ExecutorContext,
) -> None:
    """
    ``run_turn`` must delegate to ``_run_streaming_turn`` and yield events.
    """
    response = _make_text_response("Answer!")
    stream: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="Answer!"),
        ResponseCompletedEvent(response=response),
    ]

    monkeypatch.setattr(
        "agent_plane.runtime.executor._create_stream",
        lambda *_args, **_kw: iter(stream),
    )

    executor = DefaultExecutor(llm_config=llm_config)
    events = list(
        executor.run_turn(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
            system_prompt="You are helpful.",
            llm_config=llm_config,
            context=executor_context,
        )
    )

    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    # One text chunk from the stream delta.
    assert len(text_chunks) == 1
    assert text_chunks[0].text == "Answer!"

    # Must end with TurnComplete carrying the full assembled text.
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].text == "Answer!"


def test_run_turn_context_window_overflow(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: LLMConfig,
    executor_context: ExecutorContext,
) -> None:
    """
    Context window overflow during run_turn must yield ContextWindowExceeded.
    """

    def _overflow(*_args: Any, **_kw: Any) -> Iterator[ResponseStreamEvent]:
        """
        Raise context window exceeded.

        :raises ContextWindowExceededError: Always.
        """
        raise ContextWindowExceededError(
            "overflow",
            code="context_length_exceeded",
            max_context_tokens=100000,
            actual_tokens=120000,
        )

    monkeypatch.setattr(
        "agent_plane.runtime.executor._create_stream",
        _overflow,
    )

    executor = DefaultExecutor(llm_config=llm_config)
    events = list(
        executor.run_turn(
            messages=[{"role": "user", "content": "Long context"}],
            tools=[],
            system_prompt="system",
            llm_config=llm_config,
            context=executor_context,
        )
    )

    # Must yield ContextWindowExceeded so the workflow can compact and retry.
    assert len(events) == 1
    assert isinstance(events[0], ContextWindowExceeded)
    assert events[0].max_tokens == 100000
    assert events[0].actual_tokens == 120000
