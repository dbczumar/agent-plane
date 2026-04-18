"""
Integration tests for telemetry instrumentation in the agent
execution path.

Exercises ``DefaultExecutor.run_turn`` and ``_call_tool`` with a
real in-memory span exporter and real LLM/tool stub objects
(NOT ``MagicMock``) so the tests verify the full path from span
creation → attribute setting → exporter output.

See ``test_telemetry.py`` for pure-helper unit tests. Full
workflow-level tests (mocked LLM driving the whole agent loop)
are out of scope for this module — they live in
``test_workflow.py``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from agent_plane.llms.types import (
    FunctionCallOutput,
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    Usage,
)
from agent_plane.runtime import telemetry
from agent_plane.runtime.executors import ExecutorContext, TurnComplete
from agent_plane.runtime.executors.default import DefaultExecutor
from agent_plane.spec.types import LLMConfig, RetryConfig

_RESP_HEX = "abcdef0123456789abcdef0123456789"
_RESP_ID = f"resp_{_RESP_HEX}"


# ── Fixtures ────────────────────────────────────────────


@pytest.fixture
def in_memory_exporter() -> Iterator[InMemorySpanExporter]:
    """
    Install a fresh SDK TracerProvider with an in-memory exporter
    for one test.

    Forces unified mode so MLflow shares the global provider and
    resets MLflow's singleton trace manager so previous tests'
    state does not leak.
    """
    os.environ["MLFLOW_USE_DEFAULT_TRACER_PROVIDER"] = "false"
    telemetry._initialized = False  # type: ignore[attr-defined]

    from mlflow.tracing.trace_manager import InMemoryTraceManager

    trace_manager_instance = getattr(
        InMemoryTraceManager, "_instance", None
    )
    if trace_manager_instance is not None:
        trace_manager_instance._traces.clear()  # type: ignore[attr-defined]
        trace_manager_instance._otel_id_to_mlflow_trace_id.clear()  # type: ignore[attr-defined]

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = True  # type: ignore[attr-defined]

    from mlflow.tracing.provider import provider as mlflow_provider_wrapper

    mlflow_provider_wrapper._global_provider_init_once._done = False  # type: ignore[attr-defined]

    import mlflow.tracing

    mlflow.tracing.enable()

    yield exporter
    exporter.clear()


def _read_attr(span: Any, key: str) -> Any:
    """
    Decode an attribute from an OTel span, handling MLflow's
    JSON wrapping.

    :param span: An OTel ``ReadableSpan``.
    :param key: The attribute key to read.
    :returns: The decoded attribute value, or ``None`` if not set.
    """
    import json

    raw = span.attributes.get(key) if span.attributes else None
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


# ── Fake LLM client ─────────────────────────────────────


class _FakeResponsesAPI:
    """
    Stand-in for ``llms.Client.responses`` that yields a canned
    streaming response.

    Constructed with a pre-built ``Response`` and a list of
    pre-built streaming events. Returns an async iterator of the
    events when ``create(stream=True)`` is called.

    Uses real SDK types (``Response``, ``ResponseTextDeltaEvent``,
    ``ResponseCompletedEvent``) — NOT ``MagicMock`` — so the
    ``isinstance`` checks in ``_consume_stream`` traverse the
    intended code paths. See the testing skill for the rationale
    (MagicMock silently returns MagicMock for any attribute and
    hides broken type checks).
    """

    def __init__(
        self,
        *,
        events: list[ResponseStreamEvent],
    ) -> None:
        self._events = events
        self.create_calls = 0

    async def create(self, **kwargs: Any) -> AsyncIterator[ResponseStreamEvent]:
        """
        Yield the pre-built events one by one. Ignores all kwargs.

        :param kwargs: Ignored — accepted so the signature matches
            ``llms.Client.responses.create``.
        :returns: An async iterator over the canned events.
        """
        self.create_calls += 1

        async def _iter() -> AsyncIterator[ResponseStreamEvent]:
            for event in self._events:
                yield event

        return _iter()


class _FakeLLMClient:
    """
    Stand-in for ``llms.Client`` that holds a :class:`_FakeResponsesAPI`.

    The default executor reaches through ``_get_llm_client().responses``,
    so we only need to expose a ``.responses`` attribute.
    """

    def __init__(self, responses: _FakeResponsesAPI) -> None:
        self.responses = responses


def _build_completed_response(
    *,
    text: str = "Hi!",
    model: str = "test-model",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Response:
    """
    Build a real :class:`Response` with a text message output.

    :param text: The assistant text content.
    :param model: The model identifier on the response.
    :param input_tokens: Usage input tokens.
    :param output_tokens: Usage output tokens.
    :returns: A fully-constructed ``Response`` (no MagicMock).
    """
    return Response(
        output=[MessageOutput(content=[OutputText(text=text)])],
        model=model,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _install_fake_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    events: list[ResponseStreamEvent],
) -> _FakeResponsesAPI:
    """
    Patch the default executor's lazy LLM client singleton to
    return a fake that yields the given events.

    Returns the fake API so the test can assert on
    ``create_calls``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param events: Streaming events for the fake to yield.
    :returns: The installed ``_FakeResponsesAPI``.
    """
    fake_api = _FakeResponsesAPI(events=events)
    fake_client = _FakeLLMClient(fake_api)
    monkeypatch.setattr(
        "agent_plane.runtime.executors.default._llm_client",
        fake_client,
    )
    return fake_api


# ── DefaultExecutor.run_turn integration ───────────────


@pytest.mark.asyncio
async def test_run_turn_emits_chat_span_with_usage(
    in_memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """
    Running a turn through ``DefaultExecutor.run_turn`` inside a
    trace context emits one ``chat`` span with the correct model,
    provider, conversation ID, and token usage. This is the full
    path exercised on every LLM call.
    """
    completed = _build_completed_response(
        text="Hello world", input_tokens=123, output_tokens=45
    )
    events: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="Hello world"),
        ResponseCompletedEvent(response=completed),
    ]
    _install_fake_llm_client(monkeypatch, events)

    executor = DefaultExecutor(
        llm_config=LLMConfig(
            model="openai/gpt-5.4",
            extra={"temperature": 0.7},
            retry=RetryConfig(max_attempts=1),
        )
    )
    context = ExecutorContext(
        task_id=_RESP_ID,
        conversation_id="conv_xyz",
        storage_dir=tmp_path,
        call_tool=None,  # type: ignore[arg-type]
    )

    with telemetry.trace_context_for_response(response_id=_RESP_ID):
        events_out = [
            e
            async for e in executor.run_turn(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                system_prompt="You are helpful.",
                llm_config=executor._llm_config,  # type: ignore[attr-defined]
                context=context,
            )
        ]

    # Must yield a TurnComplete at the end with the accumulated text
    # so the workflow gets its final assistant response.
    final = events_out[-1]
    assert isinstance(final, TurnComplete)
    assert final.text == "Hello world", (
        f"expected final text 'Hello world', got {final.text!r} — "
        "the fake stream events should have flowed through _consume_stream."
    )

    # Exactly one chat span must be emitted per run_turn call.
    chat_spans = [
        s
        for s in in_memory_exporter.get_finished_spans()
        if s.name.startswith("chat ")
    ]
    assert len(chat_spans) == 1, (
        f"expected 1 chat span, got {len(chat_spans)} — "
        "run_turn should wrap each LLM call in exactly one span."
    )
    span = chat_spans[0]

    # Trace ID must match the response ID hex — proves the span
    # inherited the parent trace context set by
    # trace_context_for_response.
    trace_id = format(span.context.trace_id, "032x")
    assert trace_id == _RESP_HEX, (
        f"expected trace_id {_RESP_HEX!r}, got {trace_id!r} — "
        "chat span did not inherit the parent trace context."
    )

    # Model and provider must be parsed out of the model string
    # and recorded on the span (as mlflow.llm.{model,provider}
    # which translate to gen_ai.request.model / gen_ai.provider.name
    # on OTLP export).
    from mlflow.tracing.constant import SpanAttributeKey

    assert _read_attr(span, SpanAttributeKey.MODEL) == "gpt-5.4"
    assert _read_attr(span, SpanAttributeKey.MODEL_PROVIDER) == "openai"

    # gen_ai.conversation.id must be set to the executor context's
    # conversation_id so GenAI-aware backends can correlate turns.
    assert (
        _read_attr(span, "gen_ai.conversation.id") == "conv_xyz"
    ), (
        "chat span should carry gen_ai.conversation.id for "
        "cross-turn correlation."
    )

    # Temperature from extra must be recorded. Other request
    # params (top_p, max_tokens) are absent from extra so they
    # must NOT be set (we don't invent defaults).
    assert _read_attr(span, "gen_ai.request.temperature") == 0.7
    assert _read_attr(span, "gen_ai.request.top_p") is None

    # Token usage must be recorded from the completed Response's
    # Usage object — this proves the TurnComplete event carried
    # the usage dict and record_llm_usage was called.
    usage_payload = _read_attr(span, SpanAttributeKey.CHAT_USAGE)
    assert usage_payload is not None, (
        "CHAT_USAGE attribute missing — run_turn's TurnComplete "
        "handler did not call record_llm_usage."
    )
    assert usage_payload["input_tokens"] == 123, (
        f"expected input_tokens=123, got {usage_payload['input_tokens']!r} — "
        "the usage from the fake Response did not propagate through "
        "_yield_final_events → TurnComplete → record_llm_usage."
    )
    assert usage_payload["output_tokens"] == 45, (
        f"expected output_tokens=45, got {usage_payload['output_tokens']!r} — "
        "same propagation path as input_tokens; a wrong value here "
        "indicates _yield_final_events is dropping Usage.output_tokens."
    )

    # finish_reasons must be ["stop"] for a text-only response.
    # _yield_final_events derives this from the absence of tool
    # calls in the response output.
    finish_reasons = _read_attr(span, "gen_ai.response.finish_reasons")
    assert finish_reasons == ["stop"], (
        f"expected finish_reasons=['stop'], got {finish_reasons!r}"
    )


@pytest.mark.asyncio
async def test_run_turn_content_capture_records_inputs_and_outputs(
    in_memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """
    With ``AGENT_PLANE_OTEL_CAPTURE_CONTENT=true``, ``run_turn``
    records the input messages and output text via MLflow's
    ``set_inputs`` / ``set_outputs`` APIs. MLflow translates
    these to ``gen_ai.input.messages`` / ``gen_ai.output.messages``
    on OTLP export; the raw storage is
    ``mlflow.spanInputs`` / ``mlflow.spanOutputs``.
    """
    monkeypatch.setenv("AGENT_PLANE_OTEL_CAPTURE_CONTENT", "true")
    monkeypatch.setattr(telemetry, "_capture_content", True)

    completed = _build_completed_response(
        text="The answer is 42.",
        input_tokens=10,
        output_tokens=8,
    )
    events: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="The answer is 42."),
        ResponseCompletedEvent(response=completed),
    ]
    _install_fake_llm_client(monkeypatch, events)

    executor = DefaultExecutor(
        llm_config=LLMConfig(
            model="openai/gpt-5.4", retry=RetryConfig(max_attempts=1)
        )
    )
    context = ExecutorContext(
        task_id=_RESP_ID,
        conversation_id="conv_content",
        storage_dir=tmp_path,
        call_tool=None,  # type: ignore[arg-type]
    )

    with telemetry.trace_context_for_response(response_id=_RESP_ID):
        async for _ in executor.run_turn(
            messages=[{"role": "user", "content": "what's the answer?"}],
            tools=[],
            system_prompt="You are helpful.",
            llm_config=executor._llm_config,  # type: ignore[attr-defined]
            context=context,
        ):
            pass

    chat_spans = [
        s
        for s in in_memory_exporter.get_finished_spans()
        if s.name.startswith("chat ")
    ]
    assert len(chat_spans) == 1
    span = chat_spans[0]

    # Inputs payload must contain the messages the executor was
    # called with — proves span.set_inputs({"messages": ...}) fired.
    from mlflow.tracing.constant import SpanAttributeKey

    inputs = _read_attr(span, SpanAttributeKey.INPUTS)
    assert inputs is not None, (
        "mlflow.spanInputs missing — set_inputs was not called "
        "despite content capture being enabled."
    )
    assert inputs["messages"] == [
        {"role": "user", "content": "what's the answer?"}
    ], (
        f"inputs.messages = {inputs.get('messages')!r} — "
        "messages did not round-trip through set_inputs."
    )
    assert inputs["system"] == "You are helpful."
    assert inputs["model"] == "openai/gpt-5.4"

    # Outputs payload must contain the accumulated text — proves
    # span.set_outputs({"text": ...}) fired at TurnComplete.
    outputs = _read_attr(span, SpanAttributeKey.OUTPUTS)
    assert outputs is not None, (
        "mlflow.spanOutputs missing — set_outputs was not called "
        "on TurnComplete despite content capture being enabled."
    )
    assert outputs["text"] == "The answer is 42.", (
        f"outputs.text = {outputs.get('text')!r} — "
        "the accumulated stream text did not make it into set_outputs."
    )
    assert outputs["finish_reasons"] == ["stop"]


@pytest.mark.asyncio
async def test_run_turn_content_capture_disabled_omits_inputs_outputs(
    in_memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """
    With content capture disabled (default), the chat span must
    NOT have ``mlflow.spanInputs`` / ``mlflow.spanOutputs``
    populated — message content may contain PII, so we only
    record attribute metadata (model, usage, finish reasons).
    """
    monkeypatch.setattr(telemetry, "_capture_content", False)

    completed = _build_completed_response(text="hi")
    events: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent(delta="hi"),
        ResponseCompletedEvent(response=completed),
    ]
    _install_fake_llm_client(monkeypatch, events)

    executor = DefaultExecutor(
        llm_config=LLMConfig(
            model="openai/gpt-5.4", retry=RetryConfig(max_attempts=1)
        )
    )
    context = ExecutorContext(
        task_id=_RESP_ID,
        conversation_id="conv_nocapture",
        storage_dir=tmp_path,
        call_tool=None,  # type: ignore[arg-type]
    )

    with telemetry.trace_context_for_response(response_id=_RESP_ID):
        async for _ in executor.run_turn(
            messages=[{"role": "user", "content": "secret prompt"}],
            tools=[],
            system_prompt="system",
            llm_config=executor._llm_config,  # type: ignore[attr-defined]
            context=context,
        ):
            pass

    chat_spans = [
        s
        for s in in_memory_exporter.get_finished_spans()
        if s.name.startswith("chat ")
    ]
    assert len(chat_spans) == 1
    from mlflow.tracing.constant import SpanAttributeKey

    # With content capture disabled, set_inputs/set_outputs are
    # NOT called — the attributes must be absent.
    assert _read_attr(chat_spans[0], SpanAttributeKey.INPUTS) is None, (
        "mlflow.spanInputs should be absent when content capture "
        "is disabled — leaking 'secret prompt' into the span would "
        "violate the opt-in policy."
    )
    assert _read_attr(chat_spans[0], SpanAttributeKey.OUTPUTS) is None


@pytest.mark.asyncio
async def test_run_turn_emits_tool_call_events(
    in_memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """
    When the LLM response contains a ``FunctionCallOutput``, the
    chat span's ``finish_reasons`` is ``["tool_calls"]`` — the
    executor derives this from the presence of tool calls in the
    completed response.
    """
    completed = Response(
        output=[
            FunctionCallOutput(
                call_id="call_1",
                name="get_weather",
                arguments='{"city": "Paris"}',
            )
        ],
        model="test-model",
        usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
    )
    events: list[ResponseStreamEvent] = [
        ResponseCompletedEvent(response=completed),
    ]
    _install_fake_llm_client(monkeypatch, events)

    executor = DefaultExecutor(
        llm_config=LLMConfig(
            model="openai/gpt-5.4", retry=RetryConfig(max_attempts=1)
        )
    )
    context = ExecutorContext(
        task_id=_RESP_ID,
        conversation_id="conv_tool",
        storage_dir=tmp_path,
        call_tool=None,  # type: ignore[arg-type]
    )

    with telemetry.trace_context_for_response(response_id=_RESP_ID):
        async for _ in executor.run_turn(
            messages=[{"role": "user", "content": "weather?"}],
            tools=[],
            system_prompt="system",
            llm_config=executor._llm_config,  # type: ignore[attr-defined]
            context=context,
        ):
            pass

    chat_spans = [
        s
        for s in in_memory_exporter.get_finished_spans()
        if s.name.startswith("chat ")
    ]
    assert len(chat_spans) == 1
    # Tool-call responses must have finish_reasons=["tool_calls"]
    # so operators can distinguish them from text-only completions.
    finish_reasons = _read_attr(
        chat_spans[0], "gen_ai.response.finish_reasons"
    )
    assert finish_reasons == ["tool_calls"], (
        f"expected ['tool_calls'], got {finish_reasons!r} — "
        "_yield_final_events must detect tool calls in the response."
    )
