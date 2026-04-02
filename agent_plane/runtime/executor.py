"""Executor ABC and DefaultExecutor implementation.

The Executor decouples the "call LLM + interpret response" concern
from the workflow's "persist, stream, compact, steer" concerns.

``DefaultExecutor`` wraps the existing litellm-based Responses API
path. It calls ``responses.create(stream=True)``, yields executor
events as tokens arrive, and handles retry at the stream-creation
level. It has no DBOS awareness — the workflow wraps calls in
``@step`` for durability.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
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
from agent_plane.runtime.llm_retry import classify_llm_error, compute_backoff_delay
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig, RetryConfig

_logger = logging.getLogger(__name__)


# ── Event types ────────────────────────────────────────────


@dataclass
class TextChunk:
    """
    A streamed text token from the model.

    :param text: The incremental text fragment, e.g. ``"Hello"``.
    """

    text: str


@dataclass
class ReasoningChunk:
    """
    A streamed reasoning token from the model.

    Gated by ``reasoning_effort`` in the LLM config.

    :param delta: The incremental reasoning text, e.g. ``"Let me think"``.
        Empty string for ``"reasoning_started"`` events.
    :param event_type: One of ``"reasoning_text"``,
        ``"reasoning_summary"``, or ``"reasoning_started"``.
    """

    delta: str
    event_type: str


@dataclass
class NativeToolOutput:
    """
    A provider-native tool output (e.g. ``web_search_call`` result).

    Not dispatched locally — flows through to the client as-is.

    :param item: The raw output dict from the provider, e.g.
        ``{"type": "web_search_call", "id": "ws_1", ...}``.
    """

    item: dict[str, Any]


@dataclass
class ToolCallRequested:
    """
    The executor wants the workflow to execute a tool.

    The workflow executes the tool via ``_call_tool()`` (``@step``),
    appends a tool_result message, and calls ``run_turn()`` again.

    :param call_id: Identifier for this call, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"web_search"``.
    :param arguments: Parsed tool arguments dict.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallObserved:
    """
    The executor ran a tool internally. Workflow just persists and streams.

    Emitted by internal executors (Claude SDK) after each tool the
    harness executed autonomously.

    :param call_id: Identifier, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"Bash"``.
    :param arguments: Parsed tool arguments dict.
    :param result: The tool's output string.
    :param status: ``"success"`` | ``"error"`` | ``"blocked"``.
    :param duration_ms: Wall-clock time the tool took, e.g. ``342.1``.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    status: str
    duration_ms: float


@dataclass
class TurnComplete:
    """
    The executor has finished its turn.

    :param text: The assistant's text response, or ``None`` if the turn
        ended with tool calls only.
    """

    text: str | None


@dataclass
class ContextWindowExceeded:
    """
    The executor hit a context window overflow.

    The workflow compacts messages and retries ``run_turn()``.

    :param max_tokens: The model's context window size, e.g. ``128000``.
    :param actual_tokens: The prompt size that triggered overflow,
        e.g. ``131072``.
    """

    max_tokens: int
    actual_tokens: int


@dataclass
class ExecutorError:
    """
    An unrecoverable executor failure. No retry.

    :param message: Human-readable error description.
    :param code: Machine-readable error code, e.g. ``"auth_failed"``.
    """

    message: str
    code: str | None = None


@dataclass
class ToolResult:
    """
    Result of a tool call executed via ``await_tool_output``.

    :param content: The tool's output string.
    :param status: ``"success"`` or ``"error"``.
    """

    content: str
    status: str


ExecutorEvent = (
    TextChunk
    | ReasoningChunk
    | NativeToolOutput
    | ToolCallRequested
    | ToolCallObserved
    | TurnComplete
    | ContextWindowExceeded
    | ExecutorError
)


# ── Event serialization ───────────────────────────────────


def event_to_dict(event: ExecutorEvent) -> dict[str, Any]:
    """
    Serialize an executor event to a JSON-safe dict.

    Used by ``_checkpointed_turn`` to cache the event list for
    DBOS replay. Each dict has a ``"type"`` key matching the
    event class name.

    :param event: The executor event to serialize.
    :returns: A JSON-serializable dict, e.g.
        ``{"type": "TextChunk", "text": "Hello"}``.
    """
    if isinstance(event, TextChunk):
        return {"type": "TextChunk", "text": event.text}
    if isinstance(event, ReasoningChunk):
        return {
            "type": "ReasoningChunk",
            "delta": event.delta,
            "event_type": event.event_type,
        }
    if isinstance(event, NativeToolOutput):
        return {"type": "NativeToolOutput", "item": event.item}
    if isinstance(event, ToolCallRequested):
        return {
            "type": "ToolCallRequested",
            "call_id": event.call_id,
            "name": event.name,
            "arguments": event.arguments,
        }
    if isinstance(event, ToolCallObserved):
        return {
            "type": "ToolCallObserved",
            "call_id": event.call_id,
            "name": event.name,
            "arguments": event.arguments,
            "result": event.result,
            "status": event.status,
            "duration_ms": event.duration_ms,
        }
    if isinstance(event, TurnComplete):
        return {"type": "TurnComplete", "text": event.text}
    if isinstance(event, ContextWindowExceeded):
        return {
            "type": "ContextWindowExceeded",
            "max_tokens": event.max_tokens,
            "actual_tokens": event.actual_tokens,
        }
    if isinstance(event, ExecutorError):
        return {
            "type": "ExecutorError",
            "message": event.message,
            "code": event.code,
        }
    raise ValueError(f"Unknown event type: {type(event)}")


_EVENT_CONSTRUCTORS: dict[str, type[ExecutorEvent]] = {
    "TextChunk": TextChunk,
    "ReasoningChunk": ReasoningChunk,
    "NativeToolOutput": NativeToolOutput,
    "ToolCallRequested": ToolCallRequested,
    "ToolCallObserved": ToolCallObserved,
    "TurnComplete": TurnComplete,
    "ContextWindowExceeded": ContextWindowExceeded,
    "ExecutorError": ExecutorError,
}


def dict_to_event(data: dict[str, Any]) -> ExecutorEvent:
    """
    Deserialize a dict (from DBOS cache) back to an executor event.

    Inverse of :func:`event_to_dict`.

    :param data: A dict with a ``"type"`` key, e.g.
        ``{"type": "TextChunk", "text": "Hello"}``.
    :returns: The corresponding executor event instance.
    """
    event_type = data["type"]
    cls = _EVENT_CONSTRUCTORS.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type}")
    # All event dataclasses use keyword-only fields that match
    # the dict keys (minus "type").
    fields = {k: v for k, v in data.items() if k != "type"}
    return cls(**fields)


# ── ExecutorContext ─────────────────────────────────────────


@dataclass
class ExecutorContext:
    """
    Capabilities and identifiers agent-plane provides to executors.

    Constructed by the workflow once per task and passed to
    ``run_turn()`` and lifecycle hooks. Extensible — new capabilities
    are added as fields, no signature changes needed.

    :param task_id: Current task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: Current conversation identifier,
        e.g. ``"conv_abc123"``.
    :param storage_dir: Scoped persistent directory for this
        conversation. The workflow manages artifact store I/O.
    :param await_tool_output: Submit a tool call for client-side
        execution. Blocks until the client returns a result.
    """

    task_id: str
    conversation_id: str
    storage_dir: Path
    await_tool_output: Callable[[ToolCallRequested], ToolResult]


# ── Executor ABC ───────────────────────────────────────────


class Executor(abc.ABC):
    """
    Abstract base for agent executors.

    Subclasses wrap a specific LLM backend or agent harness. The
    workflow calls ``run_turn()`` and consumes the event stream
    uniformly — no branching on executor type.

    Construction is standardized via ``from_spec()``. Each subclass
    extracts what it needs from the AgentSpec.
    """

    @classmethod
    @abc.abstractmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Construct an executor from an agent spec.

        :param spec: The parsed AgentSpec with a non-None llm field.
        :returns: A configured executor instance.
        """
        ...

    @abc.abstractmethod
    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        Run one executor turn and yield events.

        :param messages: Conversation history as Responses API input items.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: Assembled system instructions string.
        :param llm_config: LLM configuration (model, extra, connection,
            timeout, retry). May differ from the spec's config due to
            per-request overrides (e.g. reasoning effort).
        :param context: Capabilities and identifiers from agent-plane.
        """
        ...

    def on_task_start(self, context: ExecutorContext) -> None:
        """
        Called once at task start, after storage_dir has been restored.

        :param context: Capabilities and identifiers from agent-plane.
        """

    def on_task_end(self, context: ExecutorContext) -> None:
        """
        Called once at task end (in a finally block).

        :param context: Same context from on_task_start.
        """

    def max_context_tokens(self) -> int | None:
        """
        Context window limit, or None if managed internally.

        When an int: the workflow owns compaction and wraps
        ``run_turn()`` in a ``@step`` for DBOS durability.

        When None: the executor owns compaction. The workflow
        skips both compaction and the ``@step`` wrapper.

        :returns: Token limit (e.g. ``128000``) or None.
        """
        return None


# ── DefaultExecutor ────────────────────────────────────────


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

    :param model: The model identifier, e.g. ``"openai/gpt-4o"``.
    :returns: Context window size in tokens, or ``None`` if unknown.
    """
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

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        One streaming LLM call with retry.

        Yields ``TextChunk``, ``ReasoningChunk``, and
        ``NativeToolOutput`` events as tokens arrive. After the
        stream completes, yields ``ToolCallRequested`` for each
        tool call and a final ``TurnComplete``.

        On context window overflow, yields ``ContextWindowExceeded``
        instead (no ``TurnComplete``). On permanent/exhausted errors,
        yields ``ExecutorError``.

        :param messages: Pre-compacted conversation history.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: System instructions.
        :param llm_config: LLM configuration (may have per-request
            reasoning overrides applied).
        :param context: Agent-plane capabilities and identifiers.
        """
        args = _build_responses_args(
            llm_config.model,
            tools,
            llm_config.extra,
        )
        yield from _run_streaming_turn(
            task_id=context.task_id,
            input_items=messages,
            instructions=system_prompt,
            args=args,
            connection=llm_config.connection,
            timeout=llm_config.request_timeout,
            retry_config=llm_config.retry,
        )


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


# ── RemoteExecutor ────────────────────────────────────────


class RemoteExecutor(Executor):
    """
    Executor that delegates to a remote agent service over HTTP.

    The remote service manages its own agent loop, tools, prompt,
    and session state. Agent-plane sends messages, observes the SSE
    event stream, and persists events for durability and relay.

    Communicates via the ``POST /v1/turns`` REST protocol defined in
    ``designs/EXECUTOR_CONTRACT_FINAL.md``.

    :param endpoint: URL of the remote turn endpoint, e.g.
        ``"http://localhost:8000/v1/turns"``.
    :param request_timeout: Per-HTTP-call timeout in seconds,
        e.g. ``300``.
    """

    def __init__(
        self,
        endpoint: str,
        request_timeout: int = 300,
    ) -> None:
        self._endpoint = endpoint
        self._request_timeout = request_timeout

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from the agent spec's executor config.

        :param spec: Agent spec with ``executor.endpoint`` set.
        :returns: Configured RemoteExecutor.
        """
        assert spec.executor.endpoint is not None
        return cls(
            endpoint=spec.executor.endpoint,
            request_timeout=spec.executor.request_timeout or 300,
        )

    def max_context_tokens(self) -> int | None:
        """
        Remote service manages its own context window.

        :returns: None — workflow skips compaction and @step.
        """
        return None

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        POST to the remote service and consume the SSE stream.

        On 404 (session not found), retries once with full
        conversation history so the remote can rebuild its session.

        Uses httpx streaming so events are yielded in real-time
        as the remote service produces them.

        :param messages: Conversation history as input items.
        :param tools: Ignored — remote defines its own tools.
        :param system_prompt: Ignored — remote defines its prompt.
        :param llm_config: Ignored — remote defines its config.
        :param context: Agent-plane capabilities and identifiers.
        """
        import httpx

        new_messages = _extract_new_messages(messages)
        body: dict[str, Any] = {
            "conversation_id": context.conversation_id,
            "new_messages": new_messages,
        }
        headers = {"Accept": "text/event-stream"}
        timeout = httpx.Timeout(
            connect=30.0,
            read=float(self._request_timeout),
            write=30.0,
            pool=30.0,
        )

        with httpx.Client(timeout=timeout) as client:
            try:
                # First attempt — normal turn.
                with client.stream(
                    "POST",
                    self._endpoint,
                    json=body,
                    headers=headers,
                ) as response:
                    if response.status_code == 404:
                        # Consume and discard the 404 body so the
                        # connection is released for the retry.
                        response.read()
                    elif response.status_code != 200:
                        yield ExecutorError(
                            message=(f"Remote executor returned {response.status_code}"),
                            code="remote_error",
                        )
                        return
                    else:
                        yield from _consume_remote_sse_stream(
                            response,
                        )
                        return
            except Exception as exc:
                yield ExecutorError(
                    message=(f"Cannot connect to remote executor at {self._endpoint}: {exc}"),
                    code="connection_error",
                )
                return

            # 404 recovery — resend with full history.
            body["history"] = _messages_to_history(messages)
            try:
                with client.stream(
                    "POST",
                    self._endpoint,
                    json=body,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        yield ExecutorError(
                            message=(f"Remote executor returned {response.status_code}"),
                            code="remote_error",
                        )
                        return
                    yield from _consume_remote_sse_stream(
                        response,
                    )
            except Exception as exc:
                yield ExecutorError(
                    message=(f"Cannot connect to remote executor at {self._endpoint}: {exc}"),
                    code="connection_error",
                )
                return


def _extract_new_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract the most recent user message(s) for a remote turn.

    Takes the trailing sequence of non-assistant messages (user +
    tool results) from the conversation history. These are the
    "new" messages the remote hasn't seen yet.

    :param messages: Full Responses API input items.
    :returns: The trailing new messages.
    """
    # Walk backwards to find the last assistant message boundary.
    new: list[dict[str, Any]] = []
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "assistant":
            break
        new.append(msg)
    new.reverse()
    return new if new else messages[-1:]


def _messages_to_history(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert Responses API input items to the simplified history
    format for session recovery.

    :param messages: Full Responses API input items.
    :returns: Simplified history with role/content/tool_calls.
    """
    # For now, pass through as-is. The remote service is
    # responsible for interpreting the format.
    return list(messages)


def _consume_remote_sse_stream(
    response: Any,
) -> Iterator[ExecutorEvent]:
    """
    Parse SSE data lines from a streaming httpx response.

    Uses ``iter_lines()`` on a streaming response so events
    are yielded in real-time as the remote produces them.
    Heartbeat events are consumed silently (keepalive only).
    ``turn_complete`` and ``error`` are terminal.

    :param response: An httpx streaming response (from
        ``client.stream()`` context manager).
    """
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        evt_type = payload.get("type", "")

        if evt_type == "text_chunk":
            yield TextChunk(text=payload["text"])

        elif evt_type == "reasoning_chunk":
            yield ReasoningChunk(
                delta=payload.get("delta", ""),
                event_type=payload.get("event_type", "reasoning_text"),
            )

        elif evt_type == "tool_call_requested":
            yield ToolCallRequested(
                call_id=payload["call_id"],
                name=payload["name"],
                arguments=payload["arguments"],
            )

        elif evt_type == "tool_call_observed":
            yield ToolCallObserved(
                call_id=payload["call_id"],
                name=payload["name"],
                arguments=payload["arguments"],
                result=payload["result"],
                status=payload["status"],
                duration_ms=payload["duration_ms"],
            )

        elif evt_type == "turn_complete":
            yield TurnComplete(text=payload.get("text"))
            return

        elif evt_type == "heartbeat":
            continue

        elif evt_type == "error":
            yield ExecutorError(
                message=payload["message"],
                code=payload.get("code"),
            )
            return
