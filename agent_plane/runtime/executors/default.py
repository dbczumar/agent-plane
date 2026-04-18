"""DefaultExecutor: Responses API via litellm.

Calls ``responses.create(stream=True)`` and yields executor events
as tokens arrive. Does not handle tools internally — yields
``ToolCallRequested`` for each tool call and lets the workflow
execute via ``@step``.

Retry wraps only stream creation (the HTTP request). Once a stream
is open, events yield in real-time. Mid-stream failures are fatal
and surface as ``ExecutorError``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from agent_plane.llms.errors import (
    ContextWindowExceededError,
    PermanentLLMError,
    RetryableLLMError,
)
from agent_plane.llms.types import (
    FunctionCallOutput,
    MessageOutput,
    NativeToolOutputAddedEvent,
    ResponseCompletedEvent,
    ResponseReasoningStartedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from agent_plane.llms.types import (
    NativeToolOutput as LLMNativeToolOutput,
)
from agent_plane.llms.types import Response as LLMResponse
from agent_plane.runtime.executors.base import (
    ContextWindowExceeded,
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    NativeToolOutput,
    ReasoningChunk,
    TextChunk,
    ToolCallRequested,
    TurnComplete,
)
from agent_plane.runtime.llm_retry import classify_llm_error, compute_backoff_delay
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig, RetryConfig

_logger = logging.getLogger(__name__)


# Lazy singleton — created on first use so imports don't fail
# when provider API keys are not yet configured.
_llm_client: Any = None


def _get_llm_client() -> Any:
    """
    Return the shared LLM client, creating it on first use.

    :returns: The ``agent_plane.llms.Client`` singleton.
    """
    global _llm_client
    if _llm_client is None:
        from agent_plane.llms import Client

        _llm_client = Client()
    return _llm_client


def _get_model_context_window(model: str) -> int | None:
    """
    Look up the model's context window from the litellm registry.

    If the ``AP_CONTEXT_WINDOW_OVERRIDE`` environment variable is set,
    its integer value is returned instead of querying litellm. This
    supports custom/self-hosted models not in litellm's registry and
    enables e2e testing of compaction with small windows.

    :param model: The model identifier, e.g. ``"openai/gpt-4o"``.
    :returns: Context window size in tokens, or ``None`` if unknown.
    """
    override = os.environ.get("AP_CONTEXT_WINDOW_OVERRIDE")
    if override is not None:
        return int(override)
    try:
        import litellm

        info = litellm.get_model_info(model)
        return info.get("max_input_tokens") if info else None
    except Exception:
        return None


@dataclass
class _ResponsesCallArgs:
    """
    Parsed arguments for a ``client.responses.create()`` call.

    :param kwargs: Direct kwargs including ``"model"`` and optionally
        ``"tools"``.
    :param reasoning: The ``reasoning`` parameter, e.g.
        ``{"effort": "high", "summary": "detailed"}``, or ``None``.
    """

    kwargs: dict[str, Any]
    reasoning: dict[str, str] | None


def _build_responses_args(
    model: str,
    tools: list[dict[str, Any]],
    extra: dict[str, Any],
) -> _ResponsesCallArgs:
    """
    Build kwargs and reasoning config for ``responses.create()``.

    Extracts ``reasoning_effort`` from ``extra`` and maps it to the
    Responses API ``reasoning`` parameter. All other ``extra`` keys
    are passed through as-is.

    :param model: The model identifier, e.g. ``"gpt-5.4"``.
    :param tools: OpenAI-format tool schemas. Empty list if no tools.
    :param extra: Additional LLM config kwargs. ``reasoning_effort``
        is extracted and mapped.
    :returns: Parsed call arguments.
    """
    remaining = dict(extra)
    reasoning_effort = remaining.pop("reasoning_effort", None)
    reasoning: dict[str, str] | None = None
    if reasoning_effort:
        # summary="detailed" enables reasoning summary streaming events.
        reasoning = {"effort": reasoning_effort, "summary": "detailed"}

    kwargs: dict[str, Any] = {"model": model, **remaining}
    if tools:
        kwargs["tools"] = tools
    return _ResponsesCallArgs(kwargs=kwargs, reasoning=reasoning)


def _extract_tool_calls(
    response: LLMResponse,
) -> list[ToolCallRequested]:
    """
    Extract tool calls from a completed LLM response.

    :param response: The completed ``Response`` from the LLM.
    :returns: List of ``ToolCallRequested`` events. Empty list
        if the response contained no tool calls.
    """
    result: list[ToolCallRequested] = []
    for item in response.output:
        if isinstance(item, FunctionCallOutput):
            result.append(
                ToolCallRequested(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=json.loads(item.arguments),
                )
            )
    return result


def _extract_text(response: LLMResponse) -> str | None:
    """
    Extract the assistant's text from a completed LLM response.

    :param response: The completed ``Response`` from the LLM.
    :returns: The text content, or ``None`` if no text.
    """
    for item in response.output:
        if isinstance(item, MessageOutput):
            for part in item.content:
                if part.type == "output_text" and part.text:
                    return part.text
    return None


def _extract_native_tool_items(
    response: LLMResponse,
) -> list[NativeToolOutput]:
    """
    Extract provider-native tool outputs from a completed response.

    :param response: The completed ``Response`` from the LLM.
    :returns: List of ``NativeToolOutput`` events.
    """
    result: list[NativeToolOutput] = []
    for item in response.output:
        if isinstance(item, LLMNativeToolOutput):
            result.append(NativeToolOutput(item=item.data))
    return result


class DefaultExecutor(Executor):
    """
    Executor backed by the Responses API via litellm.

    Calls ``responses.create(stream=True)`` and yields executor
    events as tokens arrive. Does not handle tools internally —
    yields ``ToolCallRequested`` for each tool call and lets the
    workflow execute via ``@step``.

    Retry wraps only stream creation (the HTTP request). Once a
    stream is open, events yield in real-time. Mid-stream failures
    are fatal and surface as ``ExecutorError``.

    :param llm_config: Default LLM configuration from the agent spec.
    """

    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm_config = llm_config

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from the agent spec's LLM config.

        :param spec: Agent spec with a non-None llm field.
        :returns: Configured DefaultExecutor.
        """
        assert spec.llm is not None
        return cls(llm_config=spec.llm)

    def max_context_tokens(self) -> int | None:
        """
        Return the model's known context window limit.

        The workflow uses this for proactive compaction and to
        decide whether to wrap ``run_turn()`` in a ``@step``.

        :returns: Token limit from the model registry, or ``None``
            if the model is not recognized.
        """
        return _get_model_context_window(self._llm_config.model)

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> AsyncIterator[ExecutorEvent]:
        """
        One async streaming LLM call with retry.

        Yields ``TextChunk``, ``ReasoningChunk``, and
        ``NativeToolOutput`` events as tokens arrive. After the
        stream completes, yields ``ToolCallRequested`` for each
        tool call and a final ``TurnComplete``.

        On context window overflow, yields
        ``ContextWindowExceeded`` instead (no ``TurnComplete``).
        On permanent/exhausted errors, yields ``ExecutorError``.

        An MLflow ``LLM`` span covers the full LLM call. Inputs
        (messages, model config) and outputs (text, tool calls)
        are recorded when content capture is enabled — MLflow
        translates them to ``gen_ai.input.messages`` /
        ``gen_ai.output.messages`` on OTLP export. Usage and
        response metadata are recorded from the terminal
        ``TurnComplete`` event.

        :param messages: Pre-compacted conversation history.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: System instructions.
        :param llm_config: LLM configuration (may have
            per-request reasoning overrides applied).
        :param context: Agent-plane capabilities and identifiers.
        """
        import mlflow
        from mlflow.entities import SpanType

        from agent_plane.runtime import telemetry

        args = _build_responses_args(
            llm_config.model,
            tools,
            llm_config.extra,
        )
        provider, model_name = telemetry.parse_provider_name(llm_config.model)
        span_name = f"chat {model_name}" if model_name else "chat"
        with mlflow.start_span(span_name, span_type=SpanType.CHAT_MODEL) as span:
            # MLflow stores these as mlflow.llm.* and translates
            # to gen_ai.request.* / gen_ai.provider.name on OTLP export.
            from mlflow.tracing.constant import SpanAttributeKey

            span.set_attribute(SpanAttributeKey.MODEL, model_name)
            span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, provider)
            span.set_attribute("gen_ai.conversation.id", context.conversation_id)
            # Common request params — only record when present to
            # avoid invented defaults. ``extra`` is heterogeneous.
            if (max_tokens := llm_config.extra.get("max_completion_tokens")) is not None:
                span.set_attribute("gen_ai.request.max_tokens", max_tokens)
            elif (max_tokens := llm_config.extra.get("max_tokens")) is not None:
                span.set_attribute("gen_ai.request.max_tokens", max_tokens)
            if (temperature := llm_config.extra.get("temperature")) is not None:
                span.set_attribute("gen_ai.request.temperature", temperature)
            if (top_p := llm_config.extra.get("top_p")) is not None:
                span.set_attribute("gen_ai.request.top_p", top_p)
            if (reasoning_effort := llm_config.extra.get("reasoning_effort")) is not None:
                span.set_attribute("openai.request.reasoning_effort", reasoning_effort)

            if telemetry.should_capture_content():
                # MLflow auto-translates span.inputs containing a
                # "messages" field to gen_ai.input.messages on OTLP
                # export, so no hand-rolled format conversion needed.
                span.set_inputs(
                    {
                        "messages": messages,
                        "model": llm_config.model,
                        "system": system_prompt,
                        "tools": tools,
                    }
                )

            output_texts: list[str] = []
            async for event in _run_streaming_turn(
                task_id=context.task_id,
                input_items=messages,
                instructions=system_prompt,
                args=args,
                connection=llm_config.connection,
                timeout=llm_config.request_timeout,
                retry_config=llm_config.retry,
                chat_span=span,
            ):
                if isinstance(event, TextChunk):
                    output_texts.append(event.text)
                if isinstance(event, TurnComplete):
                    if event.usage is not None:
                        telemetry.record_llm_usage(span, event.usage)
                    if event.response_model is not None:
                        span.set_attribute("gen_ai.response.model", event.response_model)
                    if event.response_id is not None:
                        span.set_attribute("gen_ai.response.id", event.response_id)
                    if event.finish_reasons:
                        span.set_attribute(
                            "gen_ai.response.finish_reasons",
                            event.finish_reasons,
                        )
                    if telemetry.should_capture_content():
                        # Let MLflow translate the output payload
                        # to gen_ai.output.messages on export.
                        span.set_outputs(
                            {
                                "text": "".join(output_texts) or event.text,
                                "finish_reasons": event.finish_reasons or [],
                            }
                        )
                yield event


async def _create_stream(
    input_items: list[dict[str, Any]],
    instructions: str,
    args: _ResponsesCallArgs,
    connection: dict[str, str] | None,
    timeout: int,
) -> AsyncIterator[ResponseStreamEvent]:
    """
    Create an async streaming LLM response.

    Separated from retry logic so only the HTTP request (not
    stream iteration) is retried. Once we have a stream,
    mid-stream failures are fatal.

    :param input_items: Responses API input items.
    :param instructions: System instructions.
    :param args: Parsed call arguments (model, tools,
        reasoning).
    :param connection: Provider connection overrides.
    :param timeout: Request timeout in seconds.
    :returns: An async iterator of raw streaming events.
    """
    from typing import cast

    result = await _get_llm_client().responses.create(
        input=input_items,
        instructions=instructions,
        reasoning=args.reasoning,
        stream=True,
        connection_params=connection,
        timeout=timeout,
        **args.kwargs,
    )
    # create() returns AsyncIterator when stream=True.
    return cast(AsyncIterator[ResponseStreamEvent], result)


async def _run_streaming_turn(
    task_id: str,
    input_items: list[dict[str, Any]],
    instructions: str,
    args: _ResponsesCallArgs,
    connection: dict[str, str] | None,
    timeout: int,
    retry_config: RetryConfig,
    chat_span: Any | None = None,
) -> AsyncIterator[ExecutorEvent]:
    """
    Execute the async streaming LLM call with retry and yield
    events.

    Retry wraps stream creation only — once a stream is opened,
    events are yielded in real-time. Mid-stream failures are
    fatal.

    Catches ``ContextWindowExceededError``,
    ``PermanentLLMError``, and ``RetryableLLMError`` from stream
    creation and converts them to executor event types. When a
    ``chat_span`` is provided and an exception is raised, we add
    a ``gen_ai.retry`` event for retries and mark the span as
    failed for terminal errors.

    :param task_id: Task identifier for logging.
    :param input_items: Responses API input items.
    :param instructions: System instructions.
    :param args: Parsed call arguments (model, tools,
        reasoning).
    :param connection: Provider connection overrides.
    :param timeout: Request timeout in seconds.
    :param retry_config: Retry policy.
    :param chat_span: Optional chat span to annotate with retry
        events and error status. ``None`` skips annotation (the
        no-op tracer path).
    """
    from agent_plane.runtime import telemetry

    try:
        stream = await _open_stream_with_retry(
            input_items,
            instructions,
            args,
            connection,
            timeout,
            retry_config,
            chat_span=chat_span,
        )
    except ContextWindowExceededError as exc:
        if chat_span is not None:
            telemetry.record_error(chat_span, exc)
        yield ContextWindowExceeded(
            max_tokens=exc.max_context_tokens,
            actual_tokens=exc.actual_tokens,
        )
        return
    except (PermanentLLMError, RetryableLLMError) as exc:
        _logger.error(
            "LLM call failed for task %s: %s",
            task_id,
            exc,
        )
        if chat_span is not None:
            telemetry.record_error(chat_span, exc)
        yield ExecutorError(message=str(exc), code=exc.code)
        return

    async for event in _consume_stream(stream):
        yield event


async def _open_stream_with_retry(
    input_items: list[dict[str, Any]],
    instructions: str,
    args: _ResponsesCallArgs,
    connection: dict[str, str] | None,
    timeout: int,
    retry_config: RetryConfig,
    chat_span: Any | None = None,
) -> AsyncIterator[ResponseStreamEvent]:
    """
    Open an async LLM stream with retry on transient failures.

    Returns the open async stream iterator on success. On
    permanent or exhausted-retry failure, raises the classified
    error — the caller catches and converts to executor events.

    :param input_items: Responses API input items.
    :param instructions: System instructions.
    :param args: Parsed call arguments.
    :param connection: Provider connection overrides.
    :param timeout: Request timeout in seconds.
    :param retry_config: Retry policy.
    :param chat_span: Optional chat span to record retry events on.
    :returns: Open async stream iterator.
    :raises ContextWindowExceededError: On context overflow.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retries exhausted.
    """
    last_error: RetryableLLMError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return await _create_stream(
                input_items,
                instructions,
                args,
                connection,
                timeout,
            )
        except ContextWindowExceededError:
            # Re-raise directly — classify_llm_error doesn't
            # preserve the subclass and would wrap it in a
            # generic PermanentLLMError, losing token counts.
            raise
        except Exception as exc:
            classified = classify_llm_error(
                exc,
                retry_config.status_codes,
            )
            if isinstance(classified, PermanentLLMError):
                raise classified from exc

            last_error = classified
            if attempt + 1 < retry_config.max_attempts:
                delay = compute_backoff_delay(
                    attempt_index=attempt,
                    backoff_base=retry_config.backoff_base,
                    backoff_max=retry_config.backoff_max,
                )
                _logger.info(
                    "LLM stream retry %d/%d after %.1fs: %s",
                    attempt + 2,
                    retry_config.max_attempts,
                    delay,
                    classified,
                )
                if chat_span is not None:
                    chat_span.add_event(
                        "gen_ai.retry",
                        attributes={
                            "attempt": attempt + 1,
                            "max_attempts": retry_config.max_attempts,
                            "error.type": type(exc).__name__,
                            "error.message": str(exc),
                            "backoff_seconds": delay,
                        },
                    )
                await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


async def _consume_stream(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[ExecutorEvent]:
    """
    Iterate an async LLM stream and yield executor events in
    real-time.

    Streaming events (text deltas, reasoning, native tool
    outputs) are yielded immediately as they arrive. After the
    stream completes, tool calls and turn complete are yielded
    from the accumulated response.

    :param stream: The open async streaming response iterator.
    """
    completed_response: LLMResponse | None = None

    async for event in stream:
        if isinstance(event, ResponseTextDeltaEvent):
            yield TextChunk(text=event.delta)
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            yield ReasoningChunk(
                delta=event.delta,
                event_type="reasoning_text",
            )
        elif isinstance(
            event,
            ResponseReasoningSummaryTextDeltaEvent,
        ):
            yield ReasoningChunk(
                delta=event.delta,
                event_type="reasoning_summary",
            )
        elif isinstance(event, ResponseReasoningStartedEvent):
            yield ReasoningChunk(
                delta="",
                event_type="reasoning_started",
            )
        elif isinstance(event, NativeToolOutputAddedEvent):
            yield NativeToolOutput(item=event.item)
        elif isinstance(event, ResponseCompletedEvent):
            completed_response = event.response

    if completed_response is not None:
        for evt in _yield_final_events(completed_response):
            yield evt
    else:
        # Stream ended without response.completed (e.g. error).
        yield TurnComplete(text=None)


def _yield_final_events(
    response: LLMResponse,
) -> Iterator[ExecutorEvent]:
    """
    Yield tool call and turn complete events from a completed
    response.

    Native tool outputs from the completed response are also
    yielded (they may not all appear during streaming). The
    terminal ``TurnComplete`` carries usage, response metadata,
    and finish reasons so the workflow instrumentation layer can
    annotate the chat span without re-parsing the response.

    :param response: The completed LLM response.
    """
    yield from _extract_native_tool_items(response)

    tool_calls = _extract_tool_calls(response)
    yield from tool_calls

    text = _extract_text(response)
    usage_dict: dict[str, Any] | None = None
    if response.usage is not None:
        usage_dict = {}
        if response.usage.input_tokens is not None:
            usage_dict["input_tokens"] = response.usage.input_tokens
        if response.usage.output_tokens is not None:
            usage_dict["output_tokens"] = response.usage.output_tokens
    finish_reasons = _extract_finish_reasons(response)
    yield TurnComplete(
        text=text if not tool_calls else None,
        usage=usage_dict,
        response_model=response.model,
        # LLMResponse does not expose a response-level ID field
        # yet; when adapters start surfacing it, add it here.
        response_id=None,
        finish_reasons=finish_reasons,
    )


def _extract_finish_reasons(response: LLMResponse) -> list[str] | None:
    """
    Extract finish reasons from a completed LLM response.

    The Responses API uses ``stop`` for natural completion and
    ``tool_calls`` when the turn ended in tool dispatch. We derive
    the reason from response content rather than a dedicated field
    because :class:`LLMResponse` does not expose one — tool-call
    presence is the authoritative signal.

    :param response: The completed LLM response.
    :returns: A single-element list with the finish reason, or
        ``None`` when the reason cannot be inferred.
    """
    has_tool_calls = any(isinstance(item, FunctionCallOutput) for item in response.output)
    if has_tool_calls:
        return ["tool_calls"]
    has_text = any(isinstance(item, MessageOutput) for item in response.output)
    if has_text:
        return ["stop"]
    return None
