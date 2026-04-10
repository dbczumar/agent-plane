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

import json
import logging
import os
import time
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
        One streaming LLM call with retry (async generator).

        Yields ``TextChunk``, ``ReasoningChunk``, and
        ``NativeToolOutput`` events as tokens arrive. After the
        stream completes, yields ``ToolCallRequested`` for each
        tool call and a final ``TurnComplete``.

        On context window overflow, yields
        ``ContextWindowExceeded`` instead (no ``TurnComplete``).
        On permanent/exhausted errors, yields ``ExecutorError``.

        The internal LLM stream is sync (Phase 2 will make the
        client async). Each ``yield`` gives the event loop a
        chance to schedule other work.

        :param messages: Pre-compacted conversation history.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: System instructions.
        :param llm_config: LLM configuration (may have
            per-request reasoning overrides applied).
        :param context: Agent-plane capabilities and identifiers.
        """
        args = _build_responses_args(
            llm_config.model,
            tools,
            llm_config.extra,
        )
        for event in _run_streaming_turn(
            task_id=context.task_id,
            input_items=messages,
            instructions=system_prompt,
            args=args,
            connection=llm_config.connection,
            timeout=llm_config.request_timeout,
            retry_config=llm_config.retry,
        ):
            yield event


def _create_stream(
    input_items: list[dict[str, Any]],
    instructions: str,
    args: _ResponsesCallArgs,
    connection: dict[str, str] | None,
    timeout: int,
) -> Iterator[ResponseStreamEvent]:
    """
    Create a streaming LLM response.

    Separated from retry logic so only the HTTP request (not stream
    iteration) is retried. Once we have a stream, mid-stream failures
    are fatal.

    :param input_items: Responses API input items.
    :param instructions: System instructions.
    :param args: Parsed call arguments (model, tools, reasoning).
    :param connection: Provider connection overrides.
    :param timeout: Request timeout in seconds.
    :returns: An iterator of raw streaming events.
    """
    from typing import cast

    return cast(
        Iterator[ResponseStreamEvent],
        _get_llm_client().responses.create(
            input=input_items,
            instructions=instructions,
            reasoning=args.reasoning,
            stream=True,
            connection_params=connection,
            timeout=timeout,
            **args.kwargs,
        ),
    )


def _run_streaming_turn(
    task_id: str,
    input_items: list[dict[str, Any]],
    instructions: str,
    args: _ResponsesCallArgs,
    connection: dict[str, str] | None,
    timeout: int,
    retry_config: RetryConfig,
) -> Iterator[ExecutorEvent]:
    """
    Execute the streaming LLM call with retry and yield events.

    Retry wraps stream creation only — once a stream is opened,
    events are yielded in real-time. Mid-stream failures are fatal.

    Catches ``ContextWindowExceededError``, ``PermanentLLMError``,
    and ``RetryableLLMError`` from stream creation and converts
    them to the corresponding executor event types.

    :param task_id: Task identifier for logging.
    :param input_items: Responses API input items.
    :param instructions: System instructions.
    :param args: Parsed call arguments (model, tools, reasoning).
    :param connection: Provider connection overrides.
    :param timeout: Request timeout in seconds.
    :param retry_config: Retry policy.
    """
    try:
        stream = _open_stream_with_retry(
            input_items,
            instructions,
            args,
            connection,
            timeout,
            retry_config,
        )
    except ContextWindowExceededError as exc:
        yield ContextWindowExceeded(
            max_tokens=exc.max_context_tokens,
            actual_tokens=exc.actual_tokens,
        )
        return
    except (PermanentLLMError, RetryableLLMError) as exc:
        _logger.error("LLM call failed for task %s: %s", task_id, exc)
        yield ExecutorError(message=str(exc), code=exc.code)
        return

    yield from _consume_stream(stream)


def _open_stream_with_retry(
    input_items: list[dict[str, Any]],
    instructions: str,
    args: _ResponsesCallArgs,
    connection: dict[str, str] | None,
    timeout: int,
    retry_config: RetryConfig,
) -> Iterator[ResponseStreamEvent]:
    """
    Open an LLM stream with retry on transient connection failures.

    Returns the open stream iterator on success. On permanent or
    exhausted-retry failure, raises the classified error — the
    caller catches and converts to executor events.

    :param input_items: Responses API input items.
    :param instructions: System instructions.
    :param args: Parsed call arguments.
    :param connection: Provider connection overrides.
    :param timeout: Request timeout in seconds.
    :param retry_config: Retry policy.
    :returns: Open stream iterator.
    :raises ContextWindowExceededError: On context overflow.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retries exhausted.
    """
    last_error: RetryableLLMError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return _create_stream(
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
                time.sleep(delay)

    assert last_error is not None
    raise last_error


def _consume_stream(
    stream: Iterator[ResponseStreamEvent],
) -> Iterator[ExecutorEvent]:
    """
    Iterate a raw LLM stream and yield executor events in real-time.

    Streaming events (text deltas, reasoning, native tool outputs)
    are yielded immediately as they arrive. After the stream
    completes, tool calls and turn complete are yielded from the
    accumulated response.

    :param stream: The open streaming response iterator.
    """
    completed_response: LLMResponse | None = None

    for event in stream:
        if isinstance(event, ResponseTextDeltaEvent):
            yield TextChunk(text=event.delta)
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            yield ReasoningChunk(
                delta=event.delta,
                event_type="reasoning_text",
            )
        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
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
        yield from _yield_final_events(completed_response)
    else:
        # Stream ended without response.completed (e.g. error).
        yield TurnComplete(text=None)


def _yield_final_events(
    response: LLMResponse,
) -> Iterator[ExecutorEvent]:
    """
    Yield tool call and turn complete events from a completed response.

    Native tool outputs from the completed response are also yielded
    (they may not all appear during streaming).

    :param response: The completed LLM response.
    """
    yield from _extract_native_tool_items(response)

    tool_calls = _extract_tool_calls(response)
    yield from tool_calls

    text = _extract_text(response)
    yield TurnComplete(text=text if not tool_calls else None)
