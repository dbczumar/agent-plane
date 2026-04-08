"""Agent execution workflow — the core agent loop.

Load agent → build prompt → call LLM → execute tools → repeat.
All durably checkpointed for crash recovery.
"""

from __future__ import annotations

import asyncio
import contextvars
import io
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from agent_plane.entities import (
    CompactionData,
    ConversationItem,
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
    NewConversationItem,
)
from agent_plane.entities.task import TERMINAL_STATUSES
from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.llms import Client as LLMClient
from agent_plane.llms.errors import (
    ContextWindowExceededError,
    PermanentLLMError,
    RetryableLLMError,
)
from agent_plane.llms.types import (
    FunctionCallOutput,
    MessageOutput,
    NativeToolOutput,
    NativeToolOutputAddedEvent,
    ResponseCompletedEvent,
    ResponseReasoningStartedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from agent_plane.llms.types import (
    Response as LLMResponse,
)
from agent_plane.runtime import (
    get_agent_cache,
    get_artifact_store,
    get_caps,
    get_conversation_store,
    get_file_store,
    get_task_store,
    get_tool_manager,
    set_tool_manager,
)
from agent_plane.runtime.compaction import (
    SummaryMetadata,
    _CompactionState,
    compact,
    compaction_to_history_items,
    count_tokens,
)
from agent_plane.runtime.content_resolver import resolve_content_references
from agent_plane.runtime.durability import (
    asyncio_wait,
    dbos_recv_async,
    get_workflow_id,
    step,
    workflow,
    write_stream,
)
from agent_plane.runtime.executor import (
    ContextWindowExceeded as ExecutorContextWindowExceeded,
)
from agent_plane.runtime.executor import (
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    ReasoningChunk,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
    dict_to_event,
    event_to_dict,
)
from agent_plane.runtime.executor import (
    NativeToolOutput as ExecutorNativeToolOutput,
)
from agent_plane.runtime.live_stream import close as _live_close
from agent_plane.runtime.live_stream import publish as _live_publish
from agent_plane.runtime.llm_retry import detail_to_dict, execute_with_retry
from agent_plane.runtime.prompt import build_instructions, history_to_input_items
from agent_plane.runtime.tool_retry import execute_tool_with_retry
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig, RetryConfig, ToolsConfig
from agent_plane.stores import ConversationStore, TaskStore
from agent_plane.tools import ToolManager
from agent_plane.tools.base import ToolContext
from agent_plane.tools.builtins import CheckSubAgentsTool, SpawnTool
from agent_plane.tools.client_specified import (
    ClientSideToolSpec,
    parse_client_side_tool_specs,
)

_logger = logging.getLogger(__name__)


def _monotonic() -> float:
    """
    Return the current monotonic time.

    Thin wrapper around ``time.monotonic()`` to allow test
    monkeypatching without interfering with the asyncio event
    loop, which also calls ``time.monotonic()`` internally.

    :returns: Monotonic time in seconds.
    """
    return time.monotonic()


_T = TypeVar("_T")


async def _to_thread(fn: Callable[[], _T]) -> _T:
    """
    Run a sync callable in the thread pool, propagating ``ContextVar`` values.

    Uses ``contextvars.copy_context()`` so that workflow-scoped state
    (e.g. ``_tool_manager_var``) is visible inside the thread pool
    thread. Without this, ``run_in_executor`` runs in a bare thread
    that has no access to the async task's context.

    :param fn: A zero-argument callable to execute.
    :returns: The callable's return value.
    """
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, ctx.run, fn)


# Hard upper bound on LLM turns per execution. Prevents runaway loops.
# See designs/AGENTLOOP.md "Not Yet" for making this configurable.
_MAX_ITERATIONS = 1000

# Lazy singleton — created on first LLM call so import doesn't
# fail when provider API keys are not yet set.
_llm_client: LLMClient | None = None


def _get_llm_client() -> LLMClient:
    """Return the shared LLM client, creating it on first use."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _create_executor(spec: AgentSpec) -> Executor:
    """
    Create an executor from the agent spec.

    Dispatches based on ``spec.executor.type``:
    ``"llm"`` (default) uses ``DefaultExecutor``,
    ``"claude_sdk"`` uses ``ClaudeAgentsExecutor``,
    ``"remote"`` uses ``RemoteExecutor``.

    :param spec: Agent spec. Must have ``llm`` set for the
        ``"llm"`` type, or ``executor.endpoint`` for ``"remote"``.
    :returns: A configured executor instance.
    """
    executor_type = spec.executor.type
    if executor_type == "claude_sdk":
        from agent_plane.runtime.executors.claude import (
            ClaudeAgentsExecutor,
        )

        return ClaudeAgentsExecutor.from_spec(spec)

    if executor_type == "remote":
        from agent_plane.runtime.executor import RemoteExecutor

        return RemoteExecutor.from_spec(spec)

    if executor_type != "llm":
        raise AgentPlaneError(
            f"Unknown executor type: {executor_type!r}. Must be 'llm', 'claude_sdk', or 'remote'.",
            code=ErrorCode.INVALID_INPUT,
        )

    from agent_plane.runtime.executor import DefaultExecutor

    return DefaultExecutor.from_spec(spec)


def _write_output(task_id: str, event: dict[str, Any]) -> None:
    """
    Write an event to both the durable stream and the live
    (real-time) stream.

    Safe to call from both sync threads (inside ``run_in_executor``)
    and async contexts. When called from an async context (no DBOS
    step context), the durable ``write_stream`` is skipped — SSE
    events still reach clients via ``_live_publish`` (the real-time
    in-process channel). Durable SSE replay on crash recovery uses
    the conversation store items, not the output stream.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param event: The event dict to write, e.g.
        ``{"type": "response.output_text.delta",
        "delta": "Hello"}``.
    """
    try:
        write_stream("output", event)
    except RuntimeError:
        # Called from async context where DBOS sync API is
        # unavailable (no step context on the event loop thread).
        # _live_publish still delivers the event in real-time.
        pass
    _live_publish(task_id, event)


async def _close_output(task_id: str) -> None:
    """
    Close both the durable stream and the live stream.

    Must be async because it runs inside the async workflow,
    and DBOS raises if ``close_stream`` (sync) is called while
    an event loop is running.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    """
    from agent_plane.runtime.durability import close_stream_async

    await close_stream_async("output")
    _live_close(task_id)


# ── Responses API helpers ─────────────────────────────────


@dataclass
class _AgentLoopResult:
    """
    Typed result returned by the agent loop and all its terminal
    helper functions.

    Converted to a plain JSON-serializable dict at the workflow
    boundary via :meth:`to_dict`.

    :param status: Terminal task status, one of ``"completed"``,
        ``"incomplete"``, or ``"failed"``, e.g. ``"completed"``.
    :param output: Accumulated API-format output items from the loop.
    :param completed_at: Unix timestamp of completion. ``None`` for
        non-completed results.
    :param error: Error details dict for failed results,
        e.g. ``{"code": "configuration_error", "message": "..."}``.
        ``None`` for non-failed results.
    :param incomplete_details: Details dict for incomplete results,
        e.g. ``{"reason": "max_iterations"}``. ``None`` for
        non-incomplete results.
    """

    status: str
    output: list[dict[str, Any]]
    completed_at: int | None = None
    error: dict[str, str] | None = None
    incomplete_details: dict[str, str] | None = None

    def to_dict(self, task_id: str) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dict for the workflow return value.

        :param task_id: The task identifier, e.g. ``"task_abc123"``.
        :returns: A dict with ``"task_id"``, ``"status"``, ``"output"``,
            and optional ``"completed_at"``, ``"error"``,
            ``"incomplete_details"`` keys.
        """
        out: dict[str, Any] = {
            "task_id": task_id,
            "status": self.status,
            "output": self.output,
        }
        if self.completed_at is not None:
            out["completed_at"] = self.completed_at
        if self.error is not None:
            out["error"] = self.error
        if self.incomplete_details is not None:
            out["incomplete_details"] = self.incomplete_details
        return out


@dataclass
class _ClientToolCallsPending:
    """
    Returned by ``_handle_tool_calls`` when the LLM has invoked one or
    more client-side tools. Signals the agent loop to complete the
    response and return the ``function_call`` items to the caller.

    :param last_seen: The ID of the last persisted ``function_call``
        item. Used as the inbox-close cursor so that
        ``close_inbox`` sees no new items and atomically closes,
        e.g. ``"item_abc123"``.
    :param client_call_ids: Call IDs of the client-side tool calls,
        e.g. ``["call_abc123", "call_def456"]``. Used by the park
        mechanism to register pending tool calls for sub-agents.
    """

    last_seen: str
    client_call_ids: list[str]


@dataclass
class _ToolCall:
    """
    A single tool invocation requested by the LLM.

    Extracted from the raw ``llm_resp`` dict at the
    :func:`_get_tool_calls` boundary and used throughout
    the tool execution pipeline. The raw dicts remain in
    ``llm_resp`` for checkpoint serialization; this dataclass
    is the typed representation used by workflow logic.

    :param call_id: The unique call ID assigned by the LLM,
        e.g. ``"call_abc123"``.
    :param name: The tool function name, e.g.
        ``"load_skill"`` or ``"get_weather"``.
    :param arguments: JSON-encoded arguments string from the
        LLM, e.g. ``'{"city": "San Francisco"}'``.
    """

    call_id: str
    name: str
    arguments: str


@dataclass
class _SteeringRetry:
    """
    Returned by ``_handle_final_response`` when late steering
    messages were found and the agent loop should continue.

    :param last_seen: The store cursor to use on the next
        iteration — the ID of the item with the highest
        store position among all items processed, e.g.
        ``"msg_abc123"``.
    """

    last_seen: str


@dataclass
class _ResponsesCallArgs:
    """
    Parsed arguments for a ``client.responses.create()`` call.

    :param kwargs: Direct kwargs passed to ``responses.create()``
        (includes ``"model"`` and optionally ``"tools"``).
    :param reasoning: The ``reasoning`` parameter for the Responses
        API, e.g. ``{"effort": "high", "summary": "concise"}``, or
        ``None`` if reasoning is not configured.
    """

    kwargs: dict[str, Any]
    reasoning: dict[str, str] | None


def _apply_request_reasoning(
    llm_config: LLMConfig,
    reasoning: dict[str, str] | None,
) -> LLMConfig:
    """Merge per-request reasoning into the agent's LLM config.

    When the request includes ``reasoning.effort``, it overrides
    the agent spec's ``reasoning_effort`` in ``extra``. Returns
    a new :class:`LLMConfig` with the merged extra dict — the
    original is not mutated.

    :param llm_config: The agent spec's LLM config.
    :param reasoning: Per-request reasoning dict, e.g.
        ``{"effort": "high"}``, or ``None``.
    :returns: A (possibly new) :class:`LLMConfig` with the
        merged reasoning effort.
    """
    if reasoning is None:
        return llm_config
    effort = reasoning.get("effort")
    if effort is None:
        return llm_config
    merged_extra = {**llm_config.extra, "reasoning_effort": effort}
    return LLMConfig(
        model=llm_config.model,
        extra=merged_extra,
        connection=llm_config.connection,
        request_timeout=llm_config.request_timeout,
        retry=llm_config.retry,
    )


def _build_responses_args(
    model: str,
    tools: list[dict[str, Any]],
    extra: dict[str, Any],
) -> _ResponsesCallArgs:
    """
    Build kwargs and reasoning config for ``client.responses.create()``.

    Extracts ``reasoning_effort`` from ``extra`` and maps it to the
    Responses API ``reasoning`` parameter (``{"effort": ...,
    "summary": "concise"}``). All other ``extra`` keys are passed
    through as-is.

    :param model: The model identifier, e.g. ``"gpt-5.4"``.
    :param tools: OpenAI-format tool schemas. Empty list if no tools.
    :param extra: Additional LLM config kwargs. ``reasoning_effort``
        is extracted and mapped; the remainder is included in kwargs.
    :returns: A :class:`_ResponsesCallArgs` with ``kwargs`` and
        ``reasoning`` fields.
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


def _response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    """
    Extract text, tool calls, and native tool items from a
    Responses API ``Response`` into a JSON-serializable dict.

    :param resp: A completed ``llms.types.Response`` object from
        ``client.responses.create(stream=False)``.
    :returns: A dict with ``"model"`` (str or None), ``"text"``
        (str or None), ``"tool_calls"`` (list of
        ``{"call_id", "name", "arguments"}`` dicts), and
        ``"native_tool_items"`` (list of raw dicts for
        provider-native tools like ``web_search_call``).
    """
    text: str | None = None
    tool_calls: list[dict[str, Any]] = []
    # Raw dicts for provider-native tool outputs (e.g. web_search_call).
    # These are not dispatched locally — they flow through to the client.
    native_tool_items: list[dict[str, Any]] = []

    for item in resp.output:
        if isinstance(item, MessageOutput):
            for part in item.content:
                if part.type == "output_text" and part.text:
                    text = part.text
                    break
        elif isinstance(item, FunctionCallOutput):
            tool_calls.append(
                {
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
            )
        elif isinstance(item, NativeToolOutput):
            native_tool_items.append(item.data)

    return {
        "model": resp.model,
        "text": text,
        "tool_calls": tool_calls,
        "native_tool_items": native_tool_items,
    }


# ── Checkpointed steps ───────────────────────────────────


@step()
async def _call_llm(
    task_id: str,
    input_items: list[dict[str, Any]],
    instructions: str,
    model: str,
    tools: list[dict[str, Any]],
    extra: dict[str, Any],
    connection: dict[str, str] | None = None,
    timeout: int | None = None,
    retry_config: RetryConfig | None = None,
) -> dict[str, Any]:
    """
    Call the LLM via the Responses API (non-streaming) with retry.

    Retries are handled inside this ``@step`` boundary so they
    don't cause duplicate checkpoints. Blocking LLM I/O runs in
    the thread pool via ``run_in_executor`` to avoid blocking
    the async event loop.

    :param task_id: The task identifier for SSE event emission,
        e.g. ``"task_abc123"``.
    :param input_items: Responses API input items (conversation
        history), e.g. ``[{"role": "user", "content": "Hello"}]``.
    :param instructions: System instructions string passed as
        ``instructions`` to the Responses API.
    :param model: The model identifier, e.g. ``"gpt-5.4"``.
    :param tools: OpenAI-format tool schemas for the agent's
        available tools. Empty list if no tools.
    :param extra: Additional kwargs from the agent's LLM config,
        e.g. ``{"temperature": 0.7}``. ``reasoning_effort`` is
        extracted and mapped to the ``reasoning`` parameter.
    :param connection: Per-provider connection overrides, e.g.
        ``{"api_key": "...", "base_url": "..."}``. ``None`` uses
        environment variable defaults.
    :param timeout: Request timeout in seconds. ``None`` uses the
        adapter's default (120s non-streaming, 300s streaming).
    :param retry_config: Retry policy. ``None`` means no retry
        (single attempt).
    :returns: A JSON-serializable dict with ``"model"``, ``"text"``,
        and ``"tool_calls"`` keys.
    :raises PermanentLLMError: On non-retryable LLM errors.
    :raises RetryableLLMError: When all retry attempts are exhausted.
    """
    args = _build_responses_args(model, tools, extra)

    def _blocking_call() -> dict[str, Any]:
        """Execute the non-streaming LLM call in a thread."""

        def do_call() -> dict[str, Any]:
            resp = cast(
                LLMResponse,
                _get_llm_client().responses.create(
                    input=input_items,
                    instructions=instructions,
                    reasoning=args.reasoning,
                    connection_params=connection,
                    timeout=timeout,
                    **args.kwargs,
                ),
            )
            return _response_to_dict(resp)

        effective_retry = retry_config or RetryConfig(max_attempts=1)
        return execute_with_retry(
            do_call,
            effective_retry,
            on_retry=lambda event: _write_output(task_id, event),
        )

    return await _to_thread(_blocking_call)


@step()
async def _call_llm_streaming(
    task_id: str,
    input_items: list[dict[str, Any]],
    instructions: str,
    model: str,
    tools: list[dict[str, Any]],
    extra: dict[str, Any],
    connection: dict[str, str] | None = None,
    timeout: int | None = None,
    retry_config: RetryConfig | None = None,
) -> dict[str, Any]:
    """
    Call the LLM via the Responses API with streaming and retry.

    Emits ``response.output_text.delta`` and reasoning delta events
    for each chunk, then returns the full accumulated response in
    the same dict format as :func:`_call_llm`.

    This is a ``@step`` so the result is durably checkpointed.
    On crash recovery the cached response is returned without
    re-executing the LLM call. Retries are internal to this step.
    Blocking LLM I/O runs in the thread pool via
    ``run_in_executor``.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param input_items: Responses API input items (conversation
        history), e.g. ``[{"role": "user", "content": "Hello"}]``.
    :param instructions: System instructions string passed as
        ``instructions`` to the Responses API.
    :param model: The model identifier, e.g. ``"gpt-5.4"``.
    :param tools: OpenAI-format tool schemas for the agent's
        available tools. Empty list if no tools.
    :param extra: Additional kwargs from the agent's LLM config,
        e.g. ``{"temperature": 0.7}``. ``reasoning_effort`` is
        extracted and mapped to the ``reasoning`` parameter.
    :param connection: Per-provider connection overrides, e.g.
        ``{"api_key": "...", "base_url": "..."}``. ``None`` uses
        environment variable defaults.
    :param timeout: Request timeout in seconds. ``None`` uses the
        adapter's default (120s non-streaming, 300s streaming).
    :param retry_config: Retry policy. ``None`` means no retry
        (single attempt).
    :returns: The accumulated response dict in the same shape
        as :func:`_call_llm`.
    :raises PermanentLLMError: On non-retryable LLM errors.
    :raises RetryableLLMError: When all retry attempts are exhausted.
    """
    args = _build_responses_args(model, tools, extra)

    def _blocking_call() -> dict[str, Any]:
        """Execute the streaming LLM call in a thread."""

        def do_call() -> dict[str, Any]:
            stream_resp = cast(
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
            return _accumulate_stream(task_id, stream_resp)

        effective_retry = retry_config or RetryConfig(max_attempts=1)
        return execute_with_retry(
            do_call,
            effective_retry,
            on_retry=lambda event: _write_output(task_id, event),
        )

    return await _to_thread(_blocking_call)


# SSE event types emitted for reasoning content
_REASONING_TEXT_EVENT = "response.reasoning_text.delta"
_REASONING_SUMMARY_EVENT = "response.reasoning_summary_text.delta"
_REASONING_STARTED_EVENT = "response.reasoning.started"


def _accumulate_stream(
    task_id: str,
    stream_resp: Iterator[ResponseStreamEvent],
) -> dict[str, Any]:
    """
    Consume a Responses API streaming response, emit text and
    reasoning delta events via :func:`_write_output` (durable +
    live stream), and return the full response dict.

    Emitted SSE event types:
    - ``response.output_text.delta`` — visible text tokens
    - ``response.reasoning_text.delta`` — full reasoning tokens
      (model-dependent; gated by ``reasoning_effort``)
    - ``response.reasoning_summary_text.delta`` — reasoning summary
      tokens (enabled when ``reasoning.summary`` is set; requires
      OpenAI org verification)

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param stream_resp: The Responses API streaming response to
        iterate over.
    :returns: The accumulated response dict in the same shape as
        :func:`_call_llm`.
    """
    completed_response: LLMResponse | None = None

    for event in stream_resp:
        if isinstance(event, ResponseReasoningStartedEvent):
            _write_output(task_id, {"type": _REASONING_STARTED_EVENT})
        elif isinstance(event, ResponseTextDeltaEvent):
            _write_output(
                task_id,
                {"type": "response.output_text.delta", "delta": event.delta},
            )
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            _write_output(
                task_id,
                {"type": _REASONING_TEXT_EVENT, "delta": event.delta},
            )
        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            _write_output(
                task_id,
                {"type": _REASONING_SUMMARY_EVENT, "delta": event.delta},
            )
        elif isinstance(event, NativeToolOutputAddedEvent):
            _write_output(
                task_id,
                {"type": "response.output_item.done", "item": event.item},
            )
        elif isinstance(event, ResponseCompletedEvent):
            completed_response = event.response

    if completed_response is not None:
        return _response_to_dict(completed_response)

    # Stream completed without a response.completed event (e.g. error).
    # Return an empty response so the loop can handle it gracefully.
    return {
        "model": None,
        "text": None,
        "tool_calls": [],
        "native_tool_items": [],
    }


# ── Executor @step wrapper ────────────────────────────────


# Maps executor reasoning event_type to SSE event type string.
_EXECUTOR_REASONING_SSE_TYPES: dict[str, str] = {
    "reasoning_text": _REASONING_TEXT_EVENT,
    "reasoning_summary": _REASONING_SUMMARY_EVENT,
    "reasoning_started": _REASONING_STARTED_EVENT,
}


@step()
async def _checkpointed_turn(
    task_id: str,
    executor: Executor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
) -> list[dict[str, Any]]:
    """
    Run one executor turn inside a DBOS checkpoint.

    Eagerly consumes the executor's event generator. Streaming
    events (text, reasoning, native tool output) are emitted to
    the live SSE stream via ``_write_output`` as they arrive.
    All events are serialized and returned so DBOS can cache
    them — on crash replay, the cached list is returned without
    re-calling the executor. Blocking executor iteration runs
    in the thread pool via ``run_in_executor``.

    :param task_id: Task identifier for SSE routing, e.g.
        ``"task_abc123"``.
    :param executor: The executor to run.
    :param messages: Conversation history as Responses API
        input items.
    :param tools: OpenAI-format tool schemas.
    :param system_prompt: Assembled system instructions.
    :param llm_config: LLM configuration (model, extra,
        connection, timeout, retry).
    :param context: Agent-plane capabilities and identifiers.
    :returns: Serialized event list (DBOS-cached on replay).
    """

    def _blocking_turn() -> list[dict[str, Any]]:
        """Consume executor events in a thread."""
        events: list[dict[str, Any]] = []
        for event in executor.run_turn(
            messages,
            tools,
            system_prompt,
            llm_config,
            context,
        ):
            _emit_executor_streaming_event(task_id, event)
            events.append(event_to_dict(event))
        return events

    return await _to_thread(_blocking_turn)


def _event_to_sse_dict(event: ExecutorEvent) -> dict[str, Any] | None:
    """
    Convert a streaming executor event to an SSE event dict.

    Returns ``None`` for event types that are not streamed (e.g.
    ``TurnComplete``, ``ToolCallObserved``) — those are handled
    by the loop body, not the SSE stream.

    :param event: The executor event to convert.
    :returns: An SSE-ready dict, or ``None`` if not streamable.
    """
    if isinstance(event, TextChunk):
        return {"type": "response.output_text.delta", "delta": event.text}
    if isinstance(event, ReasoningChunk):
        sse_type = _EXECUTOR_REASONING_SSE_TYPES[event.event_type]
        payload: dict[str, Any] = {"type": sse_type}
        if event.delta:
            payload["delta"] = event.delta
        return payload
    if isinstance(event, ExecutorNativeToolOutput):
        return {"type": "response.output_item.done", "item": event.item}
    return None


def _observed_tool_call_sse_dicts(
    event: ToolCallObserved,
) -> list[dict[str, Any]]:
    """
    Convert a ``ToolCallObserved`` to two SSE event dicts.

    The first is a ``function_call`` item (the tool invocation),
    the second is a ``function_call_output`` item (the result).
    Both are wrapped in ``response.output_item.done`` envelopes.

    :param event: The observed tool call event.
    :returns: Two-element list of SSE-ready dicts.
    """
    return [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": event.call_id,
                "name": event.name,
                "arguments": json.dumps(event.arguments),
                "status": "completed",
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call_output",
                "call_id": event.call_id,
                "output": event.result,
            },
        },
    ]


def _emit_executor_streaming_event(
    task_id: str,
    event: ExecutorEvent,
) -> None:
    """
    Emit an SSE event via the durable + live stream.

    Requires DBOS step context (used inside ``@step`` for
    workflow-managed executors). For executor-managed executors
    that skip ``@step``, use ``_emit_executor_live_only``
    instead.

    :param task_id: Task identifier for SSE routing, e.g.
        ``"task_abc123"``.
    :param event: The executor event to potentially emit.
    """
    if isinstance(event, ToolCallObserved):
        for obs_sse in _observed_tool_call_sse_dicts(event):
            _write_output(task_id, obs_sse)
        return
    sse = _event_to_sse_dict(event)
    if sse is not None:
        _write_output(task_id, sse)


def _emit_executor_live_only(
    task_id: str,
    event: ExecutorEvent,
) -> None:
    """
    Emit an SSE event via the live stream only (no durable stream).

    Used by ``_consume_executor_live`` for executor-managed
    executors (``max_context_tokens() is None``) that run outside
    ``@step()``. ``write_stream`` requires a DBOS step context,
    which is not available on this path. ``_live_publish`` is
    thread-safe and context-free.

    :param task_id: Task identifier for SSE routing, e.g.
        ``"task_abc123"``.
    :param event: The executor event to potentially emit.
    """
    if isinstance(event, ToolCallObserved):
        for obs_sse in _observed_tool_call_sse_dicts(event):
            _live_publish(task_id, obs_sse)
        return
    sse = _event_to_sse_dict(event)
    if sse is not None:
        _live_publish(task_id, sse)


# ── Executor turn → response dict bridge ──────────────────


@dataclass
class _ContextWindowOverflow:
    """
    Signal from ``_run_executor_turn`` that the executor hit
    a context window overflow.

    The caller compacts messages and retries. Not an exception —
    it is returned, not raised, so the caller can decide how
    to handle it without unwinding the stack.

    :param max_tokens: Model context window size, e.g. ``128000``.
    :param actual_tokens: Token count that triggered the overflow,
        e.g. ``142000``.
    """

    max_tokens: int
    actual_tokens: int


async def _run_executor_turn(
    task_id: str,
    executor: Executor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
) -> dict[str, Any] | _ContextWindowOverflow:
    """
    Run one executor turn and convert to a response dict.

    Workflow-managed executors (``max_context_tokens() is not None``)
    go through ``_checkpointed_turn`` for DBOS durability + live SSE.
    Executor-managed ones iterate directly with SSE emission here.

    :param task_id: Task identifier for SSE routing, e.g.
        ``"task_abc123"``.
    :param executor: The executor to run.
    :param messages: Responses API input items.
    :param tools: OpenAI-format tool schemas.
    :param system_prompt: Assembled system instructions.
    :param llm_config: LLM configuration.
    :param context: Agent-plane capabilities and identifiers.
    :returns: A response dict compatible with existing handlers,
        or ``_ContextWindowOverflow`` if compaction is needed.
    :raises PermanentLLMError: On unrecoverable executor errors.
    """
    if executor.max_context_tokens() is not None:
        raw = await _checkpointed_turn(
            task_id,
            executor,
            messages,
            tools,
            system_prompt,
            llm_config,
            context,
        )
        events: list[ExecutorEvent] = [dict_to_event(e) for e in raw]
    else:
        events = await _consume_executor_live(
            task_id, executor, messages, tools, system_prompt, llm_config, context
        )

    return _events_to_response_dict(events, llm_config.model)


async def _consume_executor_live(
    task_id: str,
    executor: Executor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
) -> list[ExecutorEvent]:
    """
    Consume an executor turn with live SSE emission.

    Used for executor-managed executors (``max_context_tokens()``
    returns ``None``) that skip the ``@step`` wrapper. Blocking
    executor iteration runs in the thread pool.

    :param task_id: Task identifier for SSE routing.
    :param executor: The executor to run.
    :param messages: Responses API input items.
    :param tools: OpenAI-format tool schemas.
    :param system_prompt: Assembled system instructions.
    :param llm_config: LLM configuration.
    :param context: Agent-plane capabilities and identifiers.
    :returns: Collected list of all executor events.
    """

    def _blocking_consume() -> list[ExecutorEvent]:
        """Consume executor events in a thread."""
        events: list[ExecutorEvent] = []
        for event in executor.run_turn(
            messages,
            tools,
            system_prompt,
            llm_config,
            context,
        ):
            _logger.info("executor event: %s", type(event).__name__)
            _emit_executor_live_only(task_id, event)
            events.append(event)
        _logger.info("total executor events: %d", len(events))
        return events

    return await _to_thread(_blocking_consume)


def _events_to_response_dict(
    events: list[ExecutorEvent],
    model: str,
) -> dict[str, Any] | _ContextWindowOverflow:
    """
    Convert collected executor events to a response dict
    compatible with existing handler functions.

    Returns ``_ContextWindowOverflow`` if a
    ``ContextWindowExceeded`` event is found. Raises
    ``PermanentLLMError`` on ``ExecutorError``.

    :param events: Collected executor events from a single turn.
    :param model: The model identifier, e.g. ``"openai/gpt-4o"``.
    :returns: Response dict with ``"model"``, ``"text"``,
        ``"tool_calls"``, ``"native_tool_items"``,
        ``"observed_tool_calls"`` keys, or
        ``_ContextWindowOverflow``.
    :raises PermanentLLMError: On executor error events.
    """
    tool_calls: list[dict[str, Any]] = []
    native_items: list[dict[str, Any]] = []
    observed: list[ToolCallObserved] = []
    turn_text: str | None = None

    for event in events:
        if isinstance(event, ToolCallRequested):
            tool_calls.append(
                {
                    "call_id": event.call_id,
                    "name": event.name,
                    # Existing handlers expect arguments as JSON string.
                    "arguments": json.dumps(event.arguments),
                }
            )
        elif isinstance(event, ExecutorNativeToolOutput):
            native_items.append(event.item)
        elif isinstance(event, ToolCallObserved):
            observed.append(event)
        elif isinstance(event, TurnComplete):
            turn_text = event.text
        elif isinstance(event, ExecutorContextWindowExceeded):
            return _ContextWindowOverflow(
                max_tokens=event.max_tokens,
                actual_tokens=event.actual_tokens,
            )
        elif isinstance(event, ExecutorError):
            raise PermanentLLMError(
                event.message,
                code=event.code or "executor_error",
            )

    return {
        "model": model,
        "text": turn_text,
        "tool_calls": tool_calls,
        "native_tool_items": native_items,
        "observed_tool_calls": observed,
    }


async def _executor_turn_with_compaction(
    task_id: str,
    executor: Executor,
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    compaction_state: _CompactionState,
    context: ExecutorContext,
    content_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Run an executor turn with proactive and reactive compaction.

    Same contract as ``_call_llm_maybe_compact``: returns a
    response dict on success, raises on unrecoverable error.
    Replaces ``_call_llm_maybe_compact`` in the agent loop when
    an executor is used.

    :param task_id: Task identifier, e.g. ``"task_abc123"``.
    :param executor: The executor to run.
    :param spec: The parsed AgentSpec.
    :param llm_config: LLM configuration.
    :param history: Conversation history as persisted items.
    :param instructions: Optional per-request instructions.
    :param tool_schemas: OpenAI-format tool schemas.
    :param compaction_state: Per-execution compaction state.
        Mutated in place when compaction triggers.
    :param context: Agent-plane capabilities and identifiers.
    :param content_cache: Per-task content reference cache.
    :returns: The response dict.
    :raises PermanentLLMError: On unrecoverable errors.
    """
    sys_instructions, messages, sys_tokens = _prepare_messages(
        spec,
        llm_config,
        history,
        instructions,
        tool_schemas,
        compaction_state,
        content_cache,
    )
    messages = _proactive_compact_if_needed(
        messages,
        history,
        sys_tokens,
        compaction_state,
        task_id,
    )
    result = await _run_executor_turn(
        task_id,
        executor,
        messages,
        tool_schemas,
        sys_instructions,
        llm_config,
        context,
    )
    if not isinstance(result, _ContextWindowOverflow):
        return result

    # Reactive compaction — compact and retry once.
    messages = _reactive_compact_from_overflow(
        messages,
        history,
        sys_tokens,
        result,
        compaction_state,
        task_id,
    )
    retry_result = await _run_executor_turn(
        task_id,
        executor,
        messages,
        tool_schemas,
        sys_instructions,
        llm_config,
        context,
    )
    if isinstance(retry_result, _ContextWindowOverflow):
        raise PermanentLLMError(
            f"Context window exceeded after compaction: "
            f"{retry_result.actual_tokens} > {retry_result.max_tokens}",
            code="context_length_exceeded",
        )
    return retry_result


def _prepare_messages(
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    compaction_state: _CompactionState,
    content_cache: dict[str, str] | None,
) -> tuple[str, list[dict[str, Any]], int]:
    """
    Build system instructions and Responses API input items.

    Resolves content references and counts system token budget.
    Extracted from ``_call_llm_maybe_compact`` for reuse by the
    executor path.

    :param spec: The parsed AgentSpec.
    :param llm_config: LLM configuration.
    :param history: Conversation history as persisted items.
    :param instructions: Optional per-request instructions.
    :param tool_schemas: OpenAI-format tool schemas.
    :param compaction_state: Per-execution compaction state.
    :param content_cache: Per-task content reference cache.
    :returns: Tuple of (system_instructions, messages, sys_tokens).
    """
    sys_instructions = build_instructions(spec, instructions, tool_schemas)
    file_store = get_file_store()
    artifact_store = get_artifact_store()
    resolved = history
    if file_store is not None and artifact_store is not None:
        resolved = resolve_content_references(
            history,
            file_store,
            artifact_store,
            content_cache,
        )
    messages = history_to_input_items(resolved)
    sys_tokens = count_tokens(
        [{"role": "system", "content": sys_instructions}],
        compaction_state.model,
    )
    return sys_instructions, messages, sys_tokens


def _reactive_compact_from_overflow(
    messages: list[dict[str, Any]],
    history: list[ConversationItem],
    sys_tokens: int,
    overflow: _ContextWindowOverflow,
    compaction_state: _CompactionState,
    task_id: str,
) -> list[dict[str, Any]]:
    """
    React to a context window overflow by compacting messages.

    Mirrors ``_reactive_compact`` but takes a
    ``_ContextWindowOverflow`` signal instead of a
    ``ContextWindowExceededError`` exception.

    :param messages: The messages that triggered the overflow.
    :param history: Conversation history for boundary detection.
    :param sys_tokens: System and tool schema token budget.
    :param overflow: The overflow signal from the executor.
    :param compaction_state: Per-execution state. Mutated in place.
    :param task_id: Task identifier for SSE event emission.
    :returns: Compacted messages list ready for retry.
    :raises PermanentLLMError: If tiktoken estimate diverges from
        the reported token count by more than 30%.
    """
    compaction_state.context_window = overflow.max_tokens
    our_estimate = count_tokens(messages, compaction_state.model) + sys_tokens
    if overflow.actual_tokens > 0:
        ratio = our_estimate / overflow.actual_tokens
        if not (0.7 <= ratio <= 1.3):
            _logger.warning(
                "tiktoken estimate %d diverges from reported %d "
                "(ratio %.2f) for task %s — raising PermanentLLMError",
                our_estimate,
                overflow.actual_tokens,
                ratio,
                task_id,
            )
            raise PermanentLLMError(
                f"Context window exceeded: {overflow.actual_tokens} > {overflow.max_tokens}",
                code="context_length_exceeded",
            )
    result = compact(
        messages,
        history,
        config=compaction_state.config,
        context_window=overflow.max_tokens,
        system_token_budget=sys_tokens,
        model=compaction_state.model,
        task_id=task_id,
        llm_client=_get_llm_client(),
        connection=compaction_state.connection,
    )
    if result.summary_metadata is not None:
        compaction_state.last_summary = result.summary_metadata
    return result.messages


@step()
async def _call_tool(
    task_id: str,
    agent_id: str,
    tool_name: str,
    arguments: str,
    timeout: int,
    retry_config: RetryConfig,
) -> str:
    """
    Route a tool call to the current workflow's ToolManager
    with timeout enforcement and retry.

    Constructs a :class:`ToolContext` from the serializable
    ``task_id`` and ``agent_id`` parameters so the context
    survives DBOS replay. Blocking tool execution runs in the
    thread pool via ``run_in_executor``.

    Retries are handled inside this ``@step`` boundary so they
    don't cause duplicate checkpoints. On exhausted retries,
    an error string is returned (not raised) so the LLM can
    decide how to proceed.

    :param task_id: The task identifier for SSE event emission,
        e.g. ``"task_abc123"``.
    :param agent_id: The registered agent ID,
        e.g. ``"ag_xyz789"``.
    :param tool_name: The tool function name, e.g.
        ``"load_skill"``.
    :param arguments: JSON-encoded arguments string from the
        LLM, e.g. ``'{"name": "summarize"}'``.
    :param timeout: Per-call timeout in seconds, e.g. ``60``.
    :param retry_config: Retry policy for this tool.
    :returns: The tool's string result, or an error string
        if all retries are exhausted.
    """

    def _blocking_call() -> str:
        """Execute the tool call in a thread."""
        mgr = get_tool_manager()
        ctx = ToolContext(task_id=task_id, agent_id=agent_id)
        tool = mgr.get_tool(tool_name)
        # Inject client-side tool schemas into spawn arguments so
        # sub-agents know which client tools are available.
        effective_args = arguments
        if tool_name == SpawnTool.name():
            effective_args = _inject_client_tools(
                arguments,
                mgr.get_client_tool_schemas(),
            )
        return execute_tool_with_retry(
            tool_name=tool_name,
            call_fn=lambda: mgr.call_tool(
                tool_name,
                effective_args,
                ctx,
            ),
            timeout=timeout,
            retry_config=retry_config,
            on_event=lambda event: _write_output(task_id, event),
            cancel_fn=tool.cancel if tool is not None else None,
        )

    return await _to_thread(_blocking_call)


def _inject_client_tools(
    arguments: str,
    client_tool_schemas: list[dict[str, Any]],
) -> str:
    """
    Inject client-side tool schemas into spawn arguments JSON.

    Adds a ``client_tools`` key so SpawnTool can propagate
    them to sub-agents without needing access to the
    ToolManager or ContextVars.

    :param arguments: Original JSON-encoded arguments from the
        LLM, e.g. ``'{"agents": [...]}'``.
    :param client_tool_schemas: OpenAI-format tool schemas,
        e.g. ``[{"type": "function", "function": {...}}]``.
    :returns: Updated JSON string with ``_client_tools`` added.
    """
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments
    args["client_tools"] = client_tool_schemas
    return json.dumps(args)


# ── Output helpers ────────────────────────────────────────


def _item_to_output(item: ConversationItem) -> dict[str, Any]:
    """
    Convert a persisted ConversationItem to the API output
    format. Mirrors ``_to_api_item()`` in conversations.py —
    see designs/LOOPGAPS.md.

    :param item: The persisted conversation item to convert.
    :returns: A flat dict with item fields suitable for the
        API response.
    """
    return {
        "id": item.id,
        "response_id": item.response_id,
        "type": item.type,
        "status": item.status,
        **item.data.model_dump(exclude_none=True, by_alias=True),
    }


def _has_tool_calls(llm_resp: dict[str, Any]) -> bool:
    """
    Check whether the LLM response contains tool calls.

    :param llm_resp: The LLM response dict (from
        :func:`_call_llm` or :func:`_call_llm_streaming`).
    :returns: ``True`` if the response has a non-empty
        ``tool_calls`` list.
    """
    return bool(llm_resp["tool_calls"])


def _get_tool_calls(
    llm_resp: dict[str, Any],
) -> list[_ToolCall]:
    """
    Extract the tool call list from the LLM response.

    Converts raw dicts (kept in ``llm_resp`` for checkpoint
    serialization) into typed :class:`_ToolCall` instances for
    use in the workflow pipeline.

    :param llm_resp: The LLM response dict (from
        :func:`_call_llm` or :func:`_call_llm_streaming`).
    :returns: List of :class:`_ToolCall` instances. Empty list
        if no tool calls.
    """
    raw: list[dict[str, Any]] = llm_resp["tool_calls"]
    return [
        _ToolCall(
            call_id=tc["call_id"],
            name=tc["name"],
            arguments=tc["arguments"],
        )
        for tc in raw
    ]


def _get_text_content(llm_resp: dict[str, Any]) -> str | None:
    """
    Extract text content from the LLM response.

    :param llm_resp: The LLM response dict (from
        :func:`_call_llm` or :func:`_call_llm_streaming`).
    :returns: The assistant's text content, or ``None`` if
        the response contained no text.
    """
    content: str | None = llm_resp["text"]
    return content


# ── Pagination helper ─────────────────────────────────────


def fetch_all_items(
    conv_store: ConversationStore,
    conversation_id: str,
    after: str | None = None,
) -> list[ConversationItem]:
    """
    Fetch all conversation items starting after the given
    cursor, paginating through every page until ``has_more``
    is ``False``.

    :param conv_store: The ConversationStore to query.
    :param conversation_id: The conversation to fetch items
        from, e.g. ``"conv_abc123"``.
    :param after: Cursor item ID to start after, or ``None``
        to fetch from the beginning.
    :returns: All items in chronological order after the
        cursor.
    """
    all_items: list[ConversationItem] = []
    cursor = after
    while True:
        page = conv_store.list_items(conversation_id, after=cursor)
        all_items.extend(page.data)
        if not page.has_more:
            break
        # Advance cursor to the last item of this page
        cursor = page.last_id
    return all_items


def _emit_native_tool_items(
    task_id: str,
    llm_resp: dict[str, Any],
    output_items: list[dict[str, Any]],
) -> None:
    """
    Append provider-native tool output items to *output_items*
    and stream them to SSE consumers.

    Native tool items (e.g. ``web_search_call``) are executed
    server-side by the LLM provider. They are included in the
    API response output but NOT persisted to the conversation
    store (they are ephemeral provider-specific items).

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param llm_resp: The LLM response dict from
        :func:`_response_to_dict`.
    :param output_items: Mutable list of API-format output dicts
        (modified in-place).
    """
    for item_dict in llm_resp.get("native_tool_items", []):
        output_items.append(item_dict)
        _write_output(
            task_id,
            {
                "type": "response.output_item.done",
                "item": item_dict,
                "output_index": len(output_items) - 1,
            },
        )


def _build_observed_tool_items(
    task_id: str,
    agent_name: str,
    observed: list[ToolCallObserved],
) -> list[NewConversationItem]:
    """
    Build conversation items for executor-observed tool calls.

    Each observation produces a ``function_call`` / ``function_call_output``
    pair. Mirrors the structure used by ``_build_function_call_items`` for
    workflow-managed tool calls.

    :param task_id: The task identifier used as ``response_id``,
        e.g. ``"task_abc123"``.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``. Used as the ``agent`` field in
        ``FunctionCallData``.
    :param observed: List of observed tool call events.
    :returns: Alternating function_call / function_call_output items.
    """
    items: list[NewConversationItem] = []
    for obs in observed:
        items.append(
            NewConversationItem(
                type="function_call",
                response_id=task_id,
                data=FunctionCallData(
                    agent=agent_name,
                    name=obs.name,
                    arguments=json.dumps(obs.arguments),
                    call_id=obs.call_id,
                ),
            )
        )
        items.append(
            NewConversationItem(
                type="function_call_output",
                response_id=task_id,
                data=FunctionCallOutputData(
                    call_id=obs.call_id,
                    output=obs.result,
                ),
            )
        )
    return items


def _persist_observed_tool_calls(
    task_id: str,
    conversation_id: str,
    agent_name: str,
    llm_resp: dict[str, Any],
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> str | None:
    """
    Persist executor-observed tool calls to the conversation store.

    SSE emission already happened during the turn via
    ``_emit_executor_live_only`` or ``_emit_executor_streaming_event``,
    so this function only persists and extends ``history`` /
    ``output_items`` — it does NOT re-emit SSE.

    :param task_id: The task identifier used as ``response_id``,
        e.g. ``"task_abc123"``.
    :param conversation_id: The conversation to append to,
        e.g. ``"conv_abc123"``.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param llm_resp: The response dict from
        ``_events_to_response_dict``. The ``"observed_tool_calls"``
        key is present for executor-managed turns; absent for
        checkpointed turns that use ``_executor_turn_with_compaction``.
    :param history: Mutable conversation history. Persisted items
        are appended in place.
    :param output_items: Mutable list of API-format output dicts.
        Persisted items are appended in place.
    :param conv_store: The ConversationStore to persist to.
    :returns: The ID of the last persisted item (new cursor),
        or ``None`` if nothing was persisted.
    """
    # Key is present for executor-managed turns (_events_to_response_dict)
    # but absent for checkpointed turns (_executor_turn_with_compaction).
    observed: list[ToolCallObserved] = llm_resp.get("observed_tool_calls", [])
    if not observed:
        return None

    new_items = _build_observed_tool_items(task_id, agent_name, observed)
    persisted = conv_store.append(conversation_id, new_items)
    for item in persisted:
        history.append(item)
        output_items.append(_item_to_output(item))
    return persisted[-1].id


# ── Extracted helpers ─────────────────────────────────────


def _persist_and_stream(
    task_id: str,
    conv_store: ConversationStore,
    conversation_id: str,
    new_items: list[NewConversationItem],
    output_items: list[dict[str, Any]],
) -> list[ConversationItem]:
    """
    Append items to conversation, convert to output format,
    and write each to the stream. Mutates ``output_items``
    in place. Returns the persisted ConversationItem list.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conv_store: The ConversationStore to persist to.
    :param conversation_id: The conversation to append to,
        e.g. ``"conv_abc123"``.
    :param new_items: Items to persist and stream.
    :param output_items: Mutable list of API-format output
        dicts. New items are appended in place.
    :returns: The persisted ConversationItem list with
        store-assigned IDs.
    """
    persisted = conv_store.append(conversation_id, new_items)
    for item in persisted:
        api_item = _item_to_output(item)
        output_items.append(api_item)
        _write_output(
            task_id,
            {
                "type": "response.output_item.done",
                "item": api_item,
                "output_index": len(output_items) - 1,
            },
        )
    return persisted


def _build_assistant_item(
    task_id: str,
    agent_name: str,
    text: str | None,
    annotations: list[dict[str, Any]] | None = None,
) -> NewConversationItem:
    """
    Build the NewConversationItem for the final assistant
    text message.

    :param task_id: The task identifier used as the
        ``response_id``, e.g. ``"task_abc123"``.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param text: The assistant's text content, or ``None``
        if the LLM produced no text.
    :param annotations: Optional list of file citation
        annotations for files the agent produced, e.g.
        ``[{"type": "file_citation", "file_id": "file_abc123",
        "filename": "chart.png", "content_type": "image/png"}]``.
    :returns: A NewConversationItem ready for persistence.
    """
    output_text_block: dict[str, Any] = {
        "type": "output_text",
        "text": text,
    }
    if annotations:
        output_text_block["annotations"] = annotations
    return NewConversationItem(
        type="message",
        response_id=task_id,
        data=MessageData(
            role="assistant",
            content=[output_text_block],
            agent=agent_name,
        ),
    )


def _handle_final_response(
    task_id: str,
    conversation_id: str,
    llm_resp: dict[str, Any],
    agent_name: str,
    last_seen: str | None,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    task_store: TaskStore,
    conv_store: ConversationStore,
) -> _AgentLoopResult | _SteeringRetry:
    """
    Handle the no-tool-calls path using persist-first-then-check.

    Persists the assistant response BEFORE checking the steering
    inbox. This prevents ghost tokens: since we already streamed
    tokens to SSE consumers, we must commit the response so those
    tokens correspond to a real persisted message. If late steering
    messages arrived during streaming, we continue the loop — the
    LLM will generate a follow-up addressing the new input,
    producing two valid committed messages instead of one spliced
    ghost.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param llm_resp: The LLM response dict containing the
        final text reply.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param last_seen: The ID of the last conversation item
        the agent has seen, or ``None``.
    :param history: Mutable conversation history. Extended
        in place if late messages arrive.
    :param output_items: Mutable list of API-format output
        dicts. Extended in place.
    :param task_store: The TaskStore for inbox operations.
    :param conv_store: The ConversationStore for persistence.
    :returns: A completed :class:`_AgentLoopResult` when the
        response is finalized, or a :class:`_SteeringRetry`
        when late messages arrived and the caller should
        continue the loop.
    """
    # ── Step 1: Persist first ──────────────────────────────
    # Commit the assistant message BEFORE checking the inbox.
    # Tokens were already streamed to SSE consumers, so this
    # message must exist in the conversation regardless of
    # whether late steering messages arrived.
    text = _get_text_content(llm_resp)
    item = _build_assistant_item(task_id, agent_name, text)
    persisted = _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        [item],
        output_items,
    )

    # ── Step 2: Check steering inbox ───────────────────────
    # Use the ORIGINAL last_seen (from before the LLM call),
    # not the newly-persisted item's ID. This ensures we
    # detect any steered messages that were delivered while
    # the LLM was streaming — those messages have positions
    # between last_seen and the assistant message we just
    # persisted.
    steered = _check_steering_inbox(
        task_id,
        conversation_id,
        last_seen,
        persisted,
        task_store,
    )
    if steered:
        # Late messages arrived during the LLM call. Add the
        # persisted assistant response first (it answers the
        # original input), then the steered user messages (new
        # input for the next iteration). This matches
        # conversational order.
        history.extend(persisted)
        history.extend(steered)
        return _SteeringRetry(last_seen=persisted[-1].id)

    # ── Step 3: Close inbox ────────────────────────────────
    # Step 2 found only our own persisted items, so the inbox
    # is still open. Call close_inbox with the persisted
    # item's cursor. A steering message could arrive between
    # steps 2 and 3, so we must check the return value: if
    # close_inbox returns items, a late message snuck in.
    final_late = task_store.close_inbox(
        task_id,
        conversation_id,
        persisted[-1].id,
    )
    if final_late:
        # A steering message arrived between steps 2 and 3.
        # The inbox is still open (close_inbox returned items
        # instead of closing). Retry the loop.
        history.extend(persisted)
        history.extend(final_late)
        return _SteeringRetry(last_seen=persisted[-1].id)

    return _AgentLoopResult(
        status="completed",
        output=output_items,
        completed_at=int(time.time()),
    )


def _check_steering_inbox(
    task_id: str,
    conversation_id: str,
    last_seen: str | None,
    persisted: list[ConversationItem],
    task_store: TaskStore,
) -> list[ConversationItem]:
    """
    Check for steered messages that arrived during the LLM call.

    Calls ``close_inbox`` with the pre-LLM cursor to detect
    messages delivered between the LLM call start and the
    assistant message persist. Filters out our own persisted
    items so only genuine steered user messages are returned.

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param last_seen: Cursor from before the LLM call — the
        ID of the last item the agent had seen.
    :param persisted: Items we just persisted (the assistant
        message). Used to filter out own items from the
        close_inbox return value.
    :param task_store: The TaskStore for inbox operations.
    :returns: List of steered conversation items (may be
        empty if no steering messages arrived).
    """
    late = task_store.close_inbox(
        task_id,
        conversation_id,
        last_seen,
    )
    own_ids = {ci.id for ci in persisted}
    return [ci for ci in late if ci.id not in own_ids]


def _build_function_call_items(
    task_id: str,
    agent_name: str,
    tool_calls: list[_ToolCall],
) -> list[NewConversationItem]:
    """
    Build NewConversationItem list for ``function_call``
    entries.

    :param task_id: The task identifier used as the
        ``response_id``, e.g. ``"task_abc123"``.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param tool_calls: Typed tool calls from the LLM response.
    :returns: A list of NewConversationItem instances ready
        for persistence.
    """
    fc_new_items: list[NewConversationItem] = []
    for tc in tool_calls:
        fc_new_items.append(
            NewConversationItem(
                type="function_call",
                response_id=task_id,
                data=FunctionCallData(
                    agent=agent_name,
                    name=tc.name,
                    arguments=tc.arguments,
                    call_id=tc.call_id,
                ),
            )
        )
    return fc_new_items


async def _execute_tools(
    task_id: str,
    conversation_id: str,
    tool_calls: list[_ToolCall],
    tools_config: ToolsConfig,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
    agent_id: str,
) -> str:
    """
    Execute tool calls in parallel and persist output in call order.

    Launches all tool calls concurrently via ``asyncio_wait``.
    Each call runs as a DBOS-checkpointed async step with the
    blocking work in the thread pool. Results are collected and
    persisted in the original call order so the LLM sees
    ``function_call_output`` items matching their ``function_call``
    items.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param tool_calls: Typed tool calls to execute.
    :param tools_config: The agent's global tools config with
        default timeout and retry policy.
    :param history: Mutable conversation history. Extended
        in place with tool output items.
    :param output_items: Mutable list of API-format output
        dicts. Extended in place.
    :param conv_store: The ConversationStore for persistence.
    :param agent_id: The registered agent ID, passed through
        to :class:`ToolContext`, e.g. ``"ag_abc123"``.
    :returns: The ID of the last persisted tool output item.
    """
    # Launch all tool calls concurrently as async tasks.
    tasks: list[asyncio.Task[str]] = [
        asyncio.ensure_future(
            _call_tool(
                task_id,
                agent_id,
                tc.name,
                tc.arguments,
                tools_config.timeout,
                tools_config.retry,
            )
        )
        for tc in tool_calls
    ]
    # DBOS.asyncio_wait checkpoints task completion state for
    # deterministic recovery on replay. Cast needed because list
    # is invariant and asyncio_wait expects list[Awaitable[Any]].
    await asyncio_wait(
        cast(list[Any], tasks),
        return_when=asyncio.ALL_COMPLETED,
    )

    # Map results back by position (tasks and tool_calls share
    # the same order).
    results: dict[str, str] = {
        tool_calls[i].call_id: tasks[i].result() for i in range(len(tool_calls))
    }

    # Persist in original call order so the LLM sees outputs
    # matching the function_call item sequence.
    last_seen: str | None = None
    for tc in tool_calls:
        fco_items = _persist_and_stream(
            task_id,
            conv_store,
            conversation_id,
            [
                NewConversationItem(
                    type="function_call_output",
                    response_id=task_id,
                    data=FunctionCallOutputData(
                        call_id=tc.call_id,
                        output=results[tc.call_id],
                    ),
                ),
            ],
            output_items,
        )
        history.extend(fco_items)
        last_seen = fco_items[-1].id
    # tool_calls is always non-empty when this function is called
    assert last_seen is not None
    return last_seen


@dataclass
class _ToolCallSplit:
    """
    Result of partitioning a tool call batch into server-side
    and client-side groups.

    :param server: Tool calls that must be executed server-side
        (MCP, skills, etc.).
    :param client: Tool calls that must be executed client-side,
        e.g. ``[_ToolCall(call_id="call_1", name="Read", ...)]``.
    :param has_client: ``True`` if the batch contains at least
        one client-side tool call.
    """

    server: list[_ToolCall]
    client: list[_ToolCall]
    has_client: bool


def _split_tool_calls(
    tool_calls: list[_ToolCall],
    tool_mgr: ToolManager,
) -> _ToolCallSplit:
    """
    Partition a batch of tool calls into server-side and client-side.

    Server-side tools (MCP, skills) need execution. Client-side
    tools are returned to the caller as ``function_call`` items
    without server-side execution.

    :param tool_calls: Typed tool calls from the LLM response.
    :param tool_mgr: The ToolManager that knows which tools are
        client-side via ``is_client_side_tool()``.
    :returns: A :class:`_ToolCallSplit` with the server-side tool
        calls and a flag indicating client-side presence.
    """
    server: list[_ToolCall] = []
    client: list[_ToolCall] = []
    for tc in tool_calls:
        if tool_mgr.is_client_side_tool(tc.name):
            client.append(tc)
        else:
            server.append(tc)
    return _ToolCallSplit(server=server, client=client, has_client=bool(client))


async def _handle_tool_calls(
    task_id: str,
    conversation_id: str,
    llm_resp: dict[str, Any],
    agent_name: str,
    agent_id: str,
    tools_config: ToolsConfig,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
    tool_mgr: ToolManager,
) -> str | _ClientToolCallsPending:
    """
    Handle the tool execution path: build ``function_call`` items,
    persist them, execute server-side tools, and signal the loop
    to complete if client-side tools are present.

    When a batch contains both server-side and client-side tools,
    the server-side tools are executed and their
    ``function_call_output`` items are persisted. The client-side
    ``function_call`` items are left unexecuted — the caller
    handles them externally. A :class:`_ClientToolCallsPending`
    is returned so the loop completes the response.

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param llm_resp: The LLM response dict containing tool calls.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param agent_id: The registered agent ID, passed through
        to :class:`ToolContext`, e.g. ``"ag_abc123"``.
    :param tools_config: The agent's global tools config with
        default timeout and retry policy.
    :param history: Mutable conversation history. Extended in place
        with function call and output items.
    :param output_items: Mutable list of API-format output dicts.
        Extended in place.
    :param conv_store: The ConversationStore for persistence.
    :param tool_mgr: The ToolManager for this workflow, used to
        detect client-side tools and dispatch server-side tools.
    :returns: The ID of the last persisted item on the server-side
        execution path, or a :class:`_ClientToolCallsPending` on
        the client-side path.
    """
    tool_calls = _get_tool_calls(llm_resp)
    fc_new_items = _build_function_call_items(task_id, agent_name, tool_calls)

    fc_items = _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        fc_new_items,
        output_items,
    )
    history.extend(fc_items)

    split = _split_tool_calls(tool_calls, tool_mgr)

    # Always execute server-side tools — even in mixed batches
    last_seen = fc_items[-1].id
    if split.server:
        last_seen = await _execute_tools(
            task_id,
            conversation_id,
            split.server,
            tools_config,
            history,
            output_items,
            conv_store,
            agent_id=agent_id,
        )

    if split.has_client:
        # Client-side function_call items are already persisted
        # and streamed. Server-side tool outputs (if any) are
        # also persisted. Complete the response so the caller
        # can handle the client-side calls externally.
        return _ClientToolCallsPending(
            last_seen=last_seen,
            client_call_ids=[tc.call_id for tc in split.client],
        )

    return last_seen


def _track_spawn_collect(
    output_items: list[dict[str, Any]],
    spawned_ids: set[str],
    collected_ids: set[str],
) -> None:
    """
    Scan output items for spawn/collect tool results and update
    the tracking sets.

    Parses ``function_call_output`` items whose corresponding
    ``function_call`` was ``spawn_sub_agents`` or
    ``check_sub_agents`` to extract response IDs. Only marks
    sub-agents as collected when their check result shows a
    terminal status.

    :param output_items: The accumulated output items list.
    :param spawned_ids: Mutable set of spawned response IDs.
    :param collected_ids: Mutable set of collected response IDs.
    """
    # Build call_id → tool_name lookup from function_call items.
    call_names: dict[str, str] = {}
    for item in output_items:
        if item.get("type") == "function_call":
            call_names[item.get("call_id", "")] = item.get("name", "")

    for item in output_items:
        if item.get("type") != "function_call_output":
            continue
        call_id = item.get("call_id", "")
        name = call_names.get(call_id, "")
        output_str = item.get("output", "")
        if not output_str:
            continue
        try:
            parsed = json.loads(output_str)
        except (json.JSONDecodeError, TypeError):
            continue
        # Match both bare names ("spawn_sub_agents") and MCP-prefixed
        # names ("mcp__agent_plane__spawn_sub_agents") so tracking
        # works for both the default executor path (bare) and the
        # Claude SDK executor path (MCP-prefixed observed tool calls).
        bare_name = name.rsplit("__", 1)[-1] if "__" in name else name
        if bare_name == SpawnTool.name():
            for rid in parsed.get("response_ids", []):
                spawned_ids.add(rid)
        elif bare_name == CheckSubAgentsTool.name():
            for r in parsed.get("results", []):
                rid = r.get("response_id", "")
                status = r.get("status")
                # Only mark as collected when terminal — in-progress
                # sub-agents still need auto-collect at turn end.
                if rid and status in TERMINAL_STATUSES:
                    collected_ids.add(rid)


def _recover_spawn_state(
    history: list[ConversationItem],
) -> tuple[set[str], set[str]]:
    """
    Reconstruct ``spawned_ids`` and ``collected_ids`` from the
    conversation history.

    When client-side tools cause ``_complete_for_client_tools``,
    each tool-call round-trip creates a NEW response (workflow).
    The new workflow starts with empty ``spawned_ids`` and
    ``collected_ids``. Without recovery, auto-collect never runs
    for sub-agents spawned in previous responses.

    Scans ``function_call`` / ``function_call_output`` items
    in history and also auto-collected-results system messages.

    :param history: The loaded conversation history.
    :returns: ``(spawned_ids, collected_ids)`` reconstructed
        from the conversation.
    """
    # Build call_id → tool_name from function_call items.
    call_names: dict[str, str] = {}
    for item in history:
        if item.type == "function_call" and isinstance(item.data, FunctionCallData):
            call_names[item.data.call_id] = item.data.name

    spawned: set[str] = set()
    collected: set[str] = set()

    for item in history:
        if item.type == "function_call_output" and isinstance(item.data, FunctionCallOutputData):
            name = call_names.get(item.data.call_id, "")
            output_str = item.data.output
            if not output_str:
                continue
            try:
                parsed = json.loads(output_str)
            except (json.JSONDecodeError, TypeError):
                continue
            bare_name = name.rsplit("__", 1)[-1] if "__" in name else name
            if bare_name == SpawnTool.name():
                for rid in parsed.get("response_ids", []):
                    spawned.add(rid)
            elif bare_name == CheckSubAgentsTool.name():
                for r in parsed.get("results", []):
                    rid = r.get("response_id", "")
                    status = r.get("status")
                    if rid and status in TERMINAL_STATUSES:
                        collected.add(rid)

        # Auto-collected results are persisted as user messages
        # with "[System: auto-collected sub-agent results]".
        if item.type == "message" and isinstance(item.data, MessageData):
            for block in item.data.content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text", "")
                if not text.startswith("[System: auto-collected sub-agent results]"):
                    continue
                # Parse the JSON after the prefix line.
                json_part = text.split("\n", 1)
                if len(json_part) < 2:
                    continue
                try:
                    payload = json.loads(json_part[1])
                except (json.JSONDecodeError, TypeError):
                    continue
                for r in payload.get("results", []):
                    rid = r.get("response_id", "")
                    if rid:
                        collected.add(rid)

    return spawned, collected


@dataclass
class _AutoCollectResult:
    """
    Result of :func:`_auto_collect_sub_agents`.

    :param last_seen: The **original** store cursor (unchanged
        from the input). Not advanced past the injected system
        message so that ``_sync_history`` on the next loop
        iteration picks up both the message and any steering
        that arrived during the poll, e.g. ``"msg_abc123"``.
    :param collected_ids: The set of sub-agent response IDs
        that actually reached a terminal status and were
        collected. May be a subset of the requested IDs if
        a steering message interrupted the wait.
    """

    last_seen: str
    collected_ids: set[str]


# Interval between polls when waiting for sub-agents during
# auto-collect. Short enough to detect steering promptly.
_COLLECT_POLL_S = 0.5


async def _persist_text_before_auto_collect(
    task_id: str,
    conversation_id: str,
    llm_resp: dict[str, Any],
    agent_name: str,
    uncollected_ids: list[str],
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> _AutoCollectResult:
    """
    Persist the LLM text then run auto-collect.

    Without this, the LLM's text is streamed to SSE but never
    persisted (ghost tokens). Subsequent LLM calls do not see
    the text in the conversation, causing the LLM to re-generate
    or ignore it. This is especially damaging for steering:
    the LLM's response to a steering message is lost, so
    the next LLM call sees only the auto-collect results and
    ignores the steering instruction.

    :param task_id: The parent task ID, e.g. ``"task_abc123"``.
    :param conversation_id: The conversation ID.
    :param llm_resp: The LLM response dict with text content.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param uncollected_ids: Sub-agent response IDs to collect.
    :param history: Mutable conversation history list.
    :param output_items: Mutable output items list.
    :param conv_store: ConversationStore for persistence.
    :returns: An :class:`_AutoCollectResult` from auto-collect.
    """
    text = _get_text_content(llm_resp)
    item = _build_assistant_item(task_id, agent_name, text)
    persisted = _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        [item],
        output_items,
    )
    history.extend(persisted)
    return await _auto_collect_sub_agents(
        task_id,
        conversation_id,
        uncollected_ids,
        # Cursor after the persisted text — steering detection
        # starts here so only messages arriving DURING
        # auto-collect are detected.
        persisted[-1].id,
        output_items,
        conv_store,
    )


async def _auto_collect_sub_agents(
    task_id: str,
    conversation_id: str,
    response_ids: list[str],
    last_seen: str,
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> _AutoCollectResult:
    """
    Auto-collect outstanding sub-agents before turn completion.

    Polls sub-agents with short timeouts, checking for steering
    messages between each poll. If all sub-agents finish, injects
    their results as a synthetic message. If a steering message
    arrives first, injects partial results (for any sub-agents
    that completed so far) and returns early so the agent loop
    can process the steering.

    Does NOT modify ``history`` — the caller re-enters the loop
    with ``continue``, and ``_sync_history`` picks up both the
    injected results and any steering messages from the store.

    :param task_id: The parent task ID.
    :param conversation_id: The parent's conversation ID.
    :param response_ids: Sub-agent response IDs to collect.
    :param last_seen: The current store cursor for detecting
        new items (steering), e.g. ``"msg_abc123"``.
    :param output_items: Mutable output items list.
    :param conv_store: ConversationStore for persistence.
    :returns: An :class:`_AutoCollectResult` with the
        **original** cursor and the set of actually-collected
        IDs.
    """
    results, collected = await _poll_subagents_with_steering_check(
        response_ids,
        conversation_id,
        last_seen,
        conv_store,
    )
    return _inject_collect_results(
        task_id,
        conversation_id,
        last_seen,
        results,
        collected,
        output_items,
        conv_store,
    )


async def _poll_subagents_with_steering_check(
    response_ids: list[str],
    conversation_id: str,
    last_seen: str,
    conv_store: ConversationStore,
) -> tuple[list[dict[str, str]], set[str]]:
    """
    Poll sub-agents for terminal status, checking for steering
    between each poll cycle.

    :param response_ids: Sub-agent response IDs to wait for.
    :param conversation_id: Parent's conversation ID for
        steering detection.
    :param last_seen: Store cursor for detecting new items.
    :param conv_store: ConversationStore for steering checks.
    :returns: ``(results, collected_ids)`` — results dicts for
        sub-agents that finished, and their IDs.
    """

    task_store = get_task_store()
    pending = set(response_ids)
    results: list[dict[str, str]] = []
    collected: set[str] = set()

    poll_count = 0
    while pending:
        newly_done = await _check_terminal(
            pending,
            task_store,
            results,
        )
        collected.update(newly_done)
        pending -= newly_done

        if not pending:
            _logger.debug(
                "auto-collect: all sub-agents terminal after %d polls",
                poll_count,
            )
            break

        # Check for steering messages before sleeping.
        new_items = fetch_all_items(
            conv_store,
            conversation_id,
            after=last_seen,
        )
        _logger.debug(
            "auto-collect poll %d: pending=%s, new_items=%d, last_seen=%s",
            poll_count,
            pending,
            len(new_items),
            last_seen,
        )
        if new_items:
            _logger.debug(
                "steering detected during auto-collect, returning %d partial results",
                len(results),
            )
            break

        poll_count += 1
        await asyncio.sleep(_COLLECT_POLL_S)

    return results, collected


async def _check_terminal(
    pending: set[str],
    task_store: TaskStore,
    results: list[dict[str, str]],
) -> set[str]:
    """
    Check for sub-agents that reached terminal status.

    :param pending: Set of sub-agent IDs still being waited on.
    :param task_store: For fetching current task status.
    :param results: Mutable list — appends a result dict for
        each newly-terminal sub-agent.
    :returns: Set of IDs that became terminal this poll cycle.
    """
    from agent_plane.tools.builtins.spawn import _task_to_result

    newly_done: set[str] = set()
    for rid in list(pending):

        def _get(task_id: str = rid) -> Any:
            return task_store.get_sync(task_id)

        task = await _to_thread(_get)
        if task is not None and task.status in TERMINAL_STATUSES:
            results.append(_task_to_result(task))
            newly_done.add(rid)
    return newly_done


def _inject_collect_results(
    task_id: str,
    conversation_id: str,
    last_seen: str,
    results: list[dict[str, str]],
    collected: set[str],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> _AutoCollectResult:
    """
    Persist collected sub-agent results as a system message.

    If no sub-agents completed, returns the original cursor
    without persisting anything.

    Returns the **original** ``last_seen`` (not the newly
    persisted item's ID) so that ``_sync_history`` on the next
    loop iteration picks up both the collected-results message
    AND any steering messages that arrived during the poll.
    Advancing the cursor here would skip steering messages
    whose positions fall between the original cursor and the
    injected system message.

    :param task_id: The parent task ID.
    :param conversation_id: The parent's conversation ID.
    :param last_seen: Current store cursor — returned as-is.
    :param results: Result dicts from completed sub-agents.
    :param collected: IDs of completed sub-agents.
    :param output_items: Mutable output items list.
    :param conv_store: ConversationStore for persistence.
    :returns: An :class:`_AutoCollectResult` with the
        **original** cursor.
    """
    if not results:
        return _AutoCollectResult(
            last_seen=last_seen,
            collected_ids=collected,
        )

    result_json = json.dumps({"results": results})
    new_items = [
        NewConversationItem(
            type="message",
            response_id=task_id,
            data=MessageData(
                role="user",
                content=[
                    {
                        "type": "input_text",
                        "text": ("[System: auto-collected sub-agent results]\n" + result_json),
                    },
                ],
            ),
        ),
    ]
    # Persist and stream to SSE/output_items, but do NOT add
    # to history — _sync_history handles that on the next
    # iteration, which also picks up any steering messages.
    _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        new_items,
        output_items,
    )
    return _AutoCollectResult(
        last_seen=last_seen,
        collected_ids=collected,
    )


async def _park_for_client_tools(
    task_id: str,
    conversation_id: str,
    root_task_id: str,
    client_call_ids: list[str],
    last_seen: str,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    task_store: TaskStore,
    conv_store: ConversationStore,
) -> str:
    """
    Park a sub-agent workflow while client-side tool calls are
    tunneled to the root response's client.

    Registers pending tool call rows, publishes ``function_call``
    items to the root task's SSE stream, polls until all results
    are delivered via PATCH, then injects ``function_call_output``
    items into the sub-agent's conversation so the loop can
    continue.

    :param task_id: The sub-agent's task ID, e.g.
        ``"task_sub1"``.
    :param conversation_id: The sub-agent's conversation ID.
    :param root_task_id: The root task ID for tunneling,
        e.g. ``"task_root1"``.
    :param client_call_ids: Call IDs of client-side tool calls
        to wait for, e.g. ``["call_abc123"]``.
    :param last_seen: Current cursor (last persisted item ID).
    :param history: Mutable conversation history.
    :param output_items: Mutable output items list.
    :param task_store: TaskStore for pending tool call ops.
    :param conv_store: ConversationStore for persistence.
    :returns: Updated ``last_seen`` cursor after injecting
        tool results.
    """
    # Build a lookup from call_id → function_call item so we
    # can extract tool_name and arguments for storage.
    client_id_set = set(client_call_ids)
    fc_by_call_id: dict[str, dict[str, Any]] = {
        item["call_id"]: item
        for item in output_items
        if item.get("type") == "function_call" and item.get("call_id") in client_id_set
    }

    # 1. Register pending tool calls so the PATCH endpoint
    #    can route results back to this sub-agent.
    for call_id in client_call_ids:
        fc = fc_by_call_id.get(call_id, {})
        task_store.create_pending_tool_call(
            call_id=call_id,
            root_task_id=root_task_id,
            task_id=task_id,
            tool_name=str(fc.get("name", "")),
            arguments=str(fc.get("arguments", "{}")),
        )

    # 2. Publish function_call items to the root task's SSE
    #    stream with status "action_required" so the client
    #    knows to execute and PATCH them back.
    for item in fc_by_call_id.values():
        tunneled = {**item, "status": "action_required"}
        _write_output(
            root_task_id,
            {
                "type": "response.output_item.done",
                "item": tunneled,
            },
        )

    # 3. Wait for the PATCH handler to signal that all pending
    #    calls are completed. Uses DBOS recv which yields the
    #    thread (no polling) — the PATCH handler calls
    #    DBOS.send to wake us.
    await _wait_for_pending_calls(client_call_ids)

    # 4. Fetch completed results and inject as
    #    function_call_output items into the conversation.
    completed = task_store.list_pending_tool_calls(task_id=task_id, status="completed")
    results_by_call_id = {ptc.call_id: ptc.result for ptc in completed}

    fco_new_items: list[NewConversationItem] = []
    for call_id in client_call_ids:
        result = results_by_call_id.get(call_id, "")
        fco_new_items.append(
            NewConversationItem(
                type="function_call_output",
                response_id=task_id,
                data=FunctionCallOutputData(
                    call_id=call_id,
                    output=result or "",
                ),
            ),
        )

    fco_items = _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        fco_new_items,
        output_items,
    )
    history.extend(fco_items)
    return fco_items[-1].id


async def _wait_for_pending_calls(
    call_ids: list[str],
) -> None:
    """
    Wait until all pending tool calls are completed.

    Uses ``DBOS.recv_async`` per call_id — each ``recv`` yields
    the event loop until the PATCH handler calls
    ``DBOS.send(workflow_id, call_id, topic="tool_result")``.

    :param call_ids: Call IDs to wait for.
    """
    for _call_id in call_ids:
        # Each recv awaits until the corresponding send
        # arrives. Yields the event loop so other workflows
        # can make progress.
        await dbos_recv_async(topic="tool_result", timeout_seconds=600)


def _complete_for_client_tools(
    task_id: str,
    conversation_id: str,
    fc_last_seen: str,
    output_items: list[dict[str, Any]],
    task_store: TaskStore,
) -> _AgentLoopResult:
    """
    Close the steering inbox and return a completed result for
    client-side tool calls.

    Called when ``_handle_tool_calls`` returns a
    :class:`_ClientToolCallsPending`. The ``function_call`` items
    are already persisted and streamed. This function closes the
    inbox at the post-persist cursor so ``try_deliver`` cannot
    inject messages after the response completes.

    Steering note: unlike ``_handle_final_response``, this uses
    the post-persist cursor (``fc_last_seen``) — so a steered
    message delivered during LLM streaming won't be detected by
    ``close_inbox``. This is acceptable because the response
    contains client-side tool calls: the client MUST send tool
    results via ``previous_response_id`` to continue, and that
    next request loads the full conversation history including
    the steered message. The LLM addresses it on that turn.
    Retrying the loop here is impossible — there are no tool
    results yet, so the LLM cannot proceed.

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param fc_last_seen: ID of the last persisted ``function_call``
        item. Used as the cursor for ``close_inbox``, e.g.
        ``"item_abc123"``.
    :param output_items: Accumulated output items to include in the
        response.
    :param task_store: The TaskStore for inbox close operations.
    :returns: A completed :class:`_AgentLoopResult`.
    """
    task_store.close_inbox(task_id, conversation_id, fc_last_seen)
    return _AgentLoopResult(
        status="completed",
        output=output_items,
        completed_at=int(time.time()),
    )


def _sync_steered_after_tools(
    conv_store: ConversationStore,
    conversation_id: str,
    pre_tool_last_seen: str | None,
    post_tool_last_seen: str,
    history: list[ConversationItem],
) -> str:
    """
    Pick up steered messages that arrived during tool execution.

    ``try_deliver`` assigns ``position = MAX(position) + 1`` at
    delivery time. If a steered message arrives between the
    function_call persist and the function_call_output persist,
    its position is interleaved among tool items. Using the
    post-tool ``last_seen`` (highest tool output position) for
    the next ``_sync_history`` would skip it.

    This function fetches all items newer than
    ``pre_tool_last_seen``, filters out items already in
    ``history`` (the tool items we just persisted), and appends
    any remaining steered messages.

    :param conv_store: The ConversationStore to query.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param pre_tool_last_seen: Cursor from before tool
        execution started.
    :param post_tool_last_seen: Cursor from after tool
        execution finished (the last tool output's ID).
    :param history: Mutable conversation history. Extended
        in place if steered messages are found.
    :returns: The effective ``last_seen`` — always
        ``post_tool_last_seen``, since tool outputs have the
        highest positions even if steered messages were
        interleaved.
    """
    if pre_tool_last_seen is None:
        return post_tool_last_seen

    all_new = fetch_all_items(
        conv_store,
        conversation_id,
        after=pre_tool_last_seen,
    )
    known_ids = {ci.id for ci in history}
    steered = [ci for ci in all_new if ci.id not in known_ids]
    if steered:
        history.extend(steered)
    return post_tool_last_seen


def _sync_history(
    conv_store: ConversationStore,
    conversation_id: str,
    last_seen: str | None,
    history: list[ConversationItem],
) -> str | None:
    """
    Check for new conversation items since *last_seen* and extend
    history with non-compaction items.

    Advances ``last_seen`` to the highest position seen — including
    compaction items — so they are not re-fetched on the next sync.
    Compaction items are excluded from *history* because they are
    metadata, not conversation content for prompt construction.

    :param conv_store: The ConversationStore to query.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param last_seen: The ID of the last item the agent has
        seen, or ``None`` if no items have been seen yet.
    :param history: Mutable conversation history. Extended
        in place with non-compaction items only.
    :returns: The updated ``last_seen`` ID (the highest position
        fetched), or the original value if no new items were found.
    """
    if last_seen is not None:
        new_items = fetch_all_items(
            conv_store,
            conversation_id,
            after=last_seen,
        )
        if new_items:
            # Advance last_seen past compaction items so we don't
            # re-fetch them on subsequent syncs.
            content_items = [i for i in new_items if i.type != "compaction"]
            if content_items:
                history.extend(content_items)
            return new_items[-1].id
    return last_seen


async def _invoke_llm_streaming(
    task_id: str,
    messages: list[dict[str, Any]],
    sys_instructions: str,
    llm_config: LLMConfig,
    tool_schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Call ``_call_llm_streaming`` with unpacked :class:`LLMConfig` fields.

    Thin wrapper that extracts ``model``, ``extra``, ``connection``,
    ``timeout``, and ``retry`` from *llm_config* so callers don't
    repeat the same kwarg extraction.

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param messages: Pre-built Responses API input items
        (output of ``history_to_input_items`` or ``compact()``).
    :param sys_instructions: Assembled system prompt string.
    :param llm_config: The agent's LLM configuration.
    :param tool_schemas: OpenAI-format tool schemas.
    :returns: Accumulated LLM response dict.
    :raises ContextWindowExceededError: When the prompt exceeds
        the model's context window.
    :raises PermanentLLMError: On non-retryable LLM errors.
    :raises RetryableLLMError: When all retry attempts are exhausted.
    """
    return await _call_llm_streaming(
        task_id,
        messages,
        sys_instructions,
        llm_config.model,
        tool_schemas,
        llm_config.extra,
        llm_config.connection,
        llm_config.request_timeout,
        llm_config.retry,
    )


def _find_latest_compaction_item(
    conv_store: ConversationStore,
    conversation_id: str,
) -> ConversationItem | None:
    """
    Return the most recently appended compaction item for a
    conversation, or ``None`` if none exists.

    Uses a descending ``limit=1`` query so only one row is read
    regardless of total conversation length.

    :param conv_store: The ConversationStore to query.
    :param conversation_id: The conversation to search,
        e.g. ``"conv_abc123"``.
    :returns: The latest compaction item, or ``None``.
    """
    page = conv_store.list_items(
        conversation_id,
        type="compaction",
        order="desc",
        limit=1,
    )
    return page.data[0] if page.data else None


def _load_initial_history(
    conv_store: ConversationStore,
    conversation_id: str,
) -> list[ConversationItem]:
    """
    Load the conversation history for the start of an execution.

    When a compaction item exists, only the items AFTER the
    summary's coverage boundary are loaded — the synthetic
    summary pair replaces the older items the LLM does not need
    to see verbatim. This bounds the load to O(items since last
    compaction), not O(total conversation length).

    When no compaction item exists, the full conversation is
    loaded (existing behaviour).

    :param conv_store: The ConversationStore to query.
    :param conversation_id: The conversation to load,
        e.g. ``"conv_abc123"``.
    :returns: History list ready for prompt construction. May
        begin with a synthetic summary pair from
        :func:`compaction_to_history_items`.
    """
    compaction_item = _find_latest_compaction_item(conv_store, conversation_id)
    if compaction_item is None:
        return fetch_all_items(conv_store, conversation_id)
    assert isinstance(compaction_item.data, CompactionData)
    # Load after last_item_id, NOT after the compaction item itself.
    # The compaction item may be appended after additional output
    # items that the summary does not cover — using last_item_id
    # ensures those post-summary items are included.
    recent_items = fetch_all_items(
        conv_store,
        conversation_id,
        after=compaction_item.data.last_item_id,
    )
    # Filter compaction items — they are metadata, not conversation
    # content the LLM should receive verbatim.
    content_items = [i for i in recent_items if i.type != "compaction"]
    return compaction_to_history_items(compaction_item) + content_items


def _proactive_compact_if_needed(
    messages: list[dict[str, Any]],
    history: list[ConversationItem],
    sys_tokens: int,
    compaction_state: _CompactionState,
    task_id: str,
) -> list[dict[str, Any]]:
    """
    Proactively compact *messages* before an LLM call when the
    estimated token count exceeds the trigger threshold.

    Only fires when ``compaction_state.context_window`` is set —
    i.e. after the first reactive overflow has revealed the model's
    limit. Returns *messages* unchanged when below the threshold.

    :param messages: The Responses API input items to check
        (output of ``history_to_input_items``).
    :param history: Conversation history items, passed through
        to :func:`compact` for boundary detection.
    :param sys_tokens: Tokens consumed by system instructions
        and tool schemas, subtracted from the window budget.
    :param compaction_state: Per-execution compaction state.
        Mutated in place: ``last_summary`` is updated if Layer 2
        triggers.
    :param task_id: Task identifier for SSE event emission.
    :returns: The (possibly compacted) messages list.
    """
    if compaction_state.context_window is None:
        return messages
    config = compaction_state.config
    threshold = config.trigger_threshold if config else 0.8
    budget = int(compaction_state.context_window * threshold)
    if count_tokens(messages, compaction_state.model) + sys_tokens <= budget:
        return messages
    result = compact(
        messages,
        history,
        config=config,
        context_window=compaction_state.context_window,
        system_token_budget=sys_tokens,
        model=compaction_state.model,
        task_id=task_id,
        llm_client=_get_llm_client(),
        connection=compaction_state.connection,
    )
    if result.summary_metadata is not None:
        compaction_state.last_summary = result.summary_metadata
    return result.messages


def _reactive_compact(
    messages: list[dict[str, Any]],
    history: list[ConversationItem],
    sys_tokens: int,
    exc: ContextWindowExceededError,
    compaction_state: _CompactionState,
    task_id: str,
) -> list[dict[str, Any]]:
    """
    React to a ``ContextWindowExceededError`` by validating with
    tiktoken, caching the discovered context window, and compacting.

    The tiktoken estimate must be within ~30% of the error's reported
    token count. If they diverge more than that, the error may be
    misclassified — it is re-raised as ``PermanentLLMError`` to avoid
    entering a pointless compact-retry loop.

    :param messages: The messages that triggered the overflow.
    :param history: Conversation history for boundary detection.
    :param sys_tokens: System and tool schema token budget.
    :param exc: The ``ContextWindowExceededError`` to react to.
    :param compaction_state: Per-execution state. ``context_window``
        and ``last_summary`` are mutated in place.
    :param task_id: Task identifier for SSE event emission.
    :returns: Compacted messages list ready for LLM retry.
    :raises PermanentLLMError: If tiktoken estimate diverges from
        the reported token count by more than 30%, indicating the
        error may be misclassified.
    """
    compaction_state.context_window = exc.max_context_tokens
    our_estimate = count_tokens(messages, compaction_state.model) + sys_tokens
    if exc.actual_tokens > 0:
        ratio = our_estimate / exc.actual_tokens
        if not (0.7 <= ratio <= 1.3):
            _logger.warning(
                "tiktoken estimate %d diverges from reported %d (ratio %.2f) "
                "for task %s — re-raising as PermanentLLMError",
                our_estimate,
                exc.actual_tokens,
                ratio,
                task_id,
            )
            raise PermanentLLMError(str(exc), code=exc.code, detail=exc.detail) from exc
    result = compact(
        messages,
        history,
        config=compaction_state.config,
        context_window=exc.max_context_tokens,
        system_token_budget=sys_tokens,
        model=compaction_state.model,
        task_id=task_id,
        llm_client=_get_llm_client(),
        connection=compaction_state.connection,
    )
    if result.summary_metadata is not None:
        compaction_state.last_summary = result.summary_metadata
    return result.messages


async def _call_llm_maybe_compact(
    task_id: str,
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    compaction_state: _CompactionState,
    content_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Call the LLM for one iteration with proactive and reactive
    compaction, plus SSE error event emission on failure.

    **Proactive path** (after the first overflow reveals the window):
    estimate tokens with tiktoken; if over threshold, compact before
    calling the LLM.

    **Reactive path** (first overflow, or proactive check missed):
    catch ``ContextWindowExceededError``, validate with tiktoken,
    compact, retry once. All other LLM errors emit a ``response.error``
    SSE event and re-raise immediately.

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param spec: The parsed AgentSpec for the executing agent.
    :param llm_config: The agent's LLM configuration.
    :param history: Conversation history as persisted items.
    :param instructions: Optional per-request instructions.
    :param tool_schemas: OpenAI-format tool schemas.
    :param compaction_state: Per-execution compaction state.
        Mutated in place when compaction triggers.
    :param content_cache: Per-task cache mapping ``file_id``
        to base64-encoded content, avoiding redundant artifact
        store fetches across iterations.
    :returns: The LLM response dict.
    :raises PermanentLLMError: On non-retryable errors or
        misclassified overflow.
    :raises RetryableLLMError: When all retries are exhausted.
    """
    sys_instructions = build_instructions(spec, instructions, tool_schemas)
    file_store = get_file_store()
    artifact_store = get_artifact_store()
    resolved = history
    if file_store is not None and artifact_store is not None:
        resolved = resolve_content_references(history, file_store, artifact_store, content_cache)
    messages = history_to_input_items(resolved)
    sys_tokens = count_tokens(
        [{"role": "system", "content": sys_instructions}],
        compaction_state.model,
    )
    messages = _proactive_compact_if_needed(
        messages, resolved, sys_tokens, compaction_state, task_id
    )
    try:
        return await _invoke_llm_streaming(
            task_id, messages, sys_instructions, llm_config, tool_schemas
        )
    except ContextWindowExceededError as exc:
        messages = _reactive_compact(
            messages, resolved, sys_tokens, exc, compaction_state, task_id
        )
        try:
            return await _invoke_llm_streaming(
                task_id, messages, sys_instructions, llm_config, tool_schemas
            )
        except (RetryableLLMError, PermanentLLMError) as inner_exc:
            _emit_llm_error_event(task_id, inner_exc)
            raise
    except (RetryableLLMError, PermanentLLMError) as exc:
        _emit_llm_error_event(task_id, exc)
        raise


def _maybe_persist_compaction_item(
    summary: SummaryMetadata,
    task_id: str,
    conversation_id: str,
    conv_store: ConversationStore,
) -> None:
    """
    Persist a compaction item for the current execution, unless one
    already exists (idempotent append for crash-recovery safety).

    The ``response_id`` on the item is the task ID, which is unique
    per execution. On crash recovery DBOS replays the workflow tail —
    the check-before-write prevents a duplicate compaction item from
    being appended.

    :param summary: The :class:`SummaryMetadata` from Layer 2.
    :param task_id: The task identifier used as the item's
        ``response_id``, e.g. ``"task_abc123"``.
    :param conversation_id: The conversation to append to,
        e.g. ``"conv_abc123"``.
    :param conv_store: The ConversationStore to append to.
    """
    existing = conv_store.list_items(
        conversation_id,
        type="compaction",
        order="desc",
        limit=1,
    )
    if existing.data and existing.data[0].response_id == task_id:
        # Already persisted — idempotent on crash recovery replay.
        return
    conv_store.append(
        conversation_id,
        [
            NewConversationItem(
                type="compaction",
                response_id=task_id,
                data=CompactionData(
                    summary=summary.text,
                    last_item_id=summary.last_item_id,
                    model=summary.model,
                    token_count=summary.token_count,
                ),
            )
        ],
    )


def _emit_llm_error_event(
    task_id: str,
    exc: RetryableLLMError | PermanentLLMError,
) -> None:
    """
    Emit a ``response.error`` SSE event for a terminal LLM failure.

    :param task_id: The task identifier for event routing.
    :param exc: The classified LLM error.
    """
    detail_dict = detail_to_dict(exc.detail) if exc.detail else None
    _write_output(
        task_id,
        {
            "type": "response.error",
            "source": "llm",
            "error": {
                "code": exc.code,
                "message": str(exc),
                "detail": detail_dict,
            },
        },
    )


def _handle_execution_timeout(
    task_id: str,
    output_items: list[dict[str, Any]],
    execution_timeout: int,
) -> _AgentLoopResult:
    """
    Handle execution timeout: emit SSE error event and return
    incomplete result.

    :param task_id: The task identifier for SSE event emission.
    :param output_items: Accumulated output items so far.
    :param execution_timeout: The timeout that was exceeded,
        in seconds, e.g. ``3600``.
    :returns: An incomplete :class:`_AgentLoopResult` with
        ``"execution_timeout"`` reason.
    """
    _write_output(
        task_id,
        {
            "type": "response.error",
            "source": "execution",
            "error": {
                "code": "execution_timeout",
                "message": (f"Wall-clock deadline exceeded after {execution_timeout}s"),
                "detail": None,
            },
        },
    )
    return _AgentLoopResult(
        status="incomplete",
        output=output_items,
        incomplete_details={"reason": "execution_timeout"},
    )


_EXECUTOR_STORAGE_KEY_PREFIX = "executor_storage"

# Base directory for executor storage on disk. Each (conversation, agent)
# pair gets a stable subdirectory that persists across tasks.
_EXECUTOR_STORAGE_BASE = Path.home() / ".agent-plane" / "executor_storage"


def _storage_artifact_key(
    conversation_id: str,
    agent_name: str,
) -> str:
    """
    Artifact store key for an executor's storage snapshot.

    :param conversation_id: e.g. ``"conv_abc123"``.
    :param agent_name: e.g. ``"research-agent"``.
    :returns: Key string, e.g.
        ``"executor_storage/conv_abc123/research-agent.tar.gz"``.
    :raises ValueError: If either argument would escape the expected
        directory tree (path traversal).
    """
    from pathlib import PurePosixPath

    for label, value in [("conversation_id", conversation_id), ("agent_name", agent_name)]:
        # PurePosixPath.name strips directory components; if it differs
        # from the original, the value contains separators or "..".
        if PurePosixPath(value).name != value:
            raise ValueError(f"{label} contains path traversal characters: {value!r}")
    return f"{_EXECUTOR_STORAGE_KEY_PREFIX}/{conversation_id}/{agent_name}.tar.gz"


def _get_or_restore_executor_storage(
    conversation_id: str,
    agent_name: str,
) -> Path:
    """
    Return a stable storage directory for ``(conversation_id, agent_name)``.

    Idempotent: if the directory already exists on disk with content,
    it is returned as-is (no artifact store round-trip). Otherwise,
    the directory is created and populated from the artifact store
    snapshot if one exists.

    The directory is never deleted between tasks — it lives for as
    long as the conversation is active.

    :param conversation_id: e.g. ``"conv_abc123"``.
    :param agent_name: e.g. ``"research-agent"``.
    :returns: Path to the storage directory.
    """
    storage_dir = _EXECUTOR_STORAGE_BASE / conversation_id / agent_name
    storage_dir.mkdir(parents=True, exist_ok=True)

    # If the directory already has content, the previous task left it
    # intact — no need to restore from the artifact store.
    if any(storage_dir.iterdir()):
        return storage_dir

    artifact_store = get_artifact_store()
    if artifact_store is None:
        return storage_dir

    key = _storage_artifact_key(conversation_id, agent_name)
    if not artifact_store.exists(key):
        return storage_dir

    import tarfile

    snapshot = artifact_store.get(key)
    with tarfile.open(fileobj=io.BytesIO(snapshot), mode="r:gz") as tf:
        tf.extractall(storage_dir)  # noqa: S202 — trusted internal data
    _logger.debug(
        "restored executor storage for %s/%s from artifact store",
        conversation_id,
        agent_name,
    )
    return storage_dir


def _persist_executor_storage(
    conversation_id: str,
    agent_name: str,
    storage_dir: Path,
) -> None:
    """
    Snapshot the executor storage directory to the artifact store.

    Called after every task so the artifact store stays current for
    server restart recovery. The directory itself is NOT deleted.

    :param conversation_id: e.g. ``"conv_abc123"``.
    :param agent_name: e.g. ``"research-agent"``.
    :param storage_dir: The executor's working directory to snapshot.
    """
    artifact_store = get_artifact_store()
    if artifact_store is None:
        return

    if not any(storage_dir.iterdir()):
        return

    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for child in storage_dir.iterdir():
            tf.add(child, arcname=child.name)
    key = _storage_artifact_key(conversation_id, agent_name)
    artifact_store.put(key, buf.getvalue())
    _logger.debug(
        "persisted executor storage for %s/%s (%d bytes)",
        conversation_id,
        agent_name,
        buf.tell(),
    )


def _build_executor_context(
    task_id: str,
    conversation_id: str,
    storage_dir: Path,
    root_task_id: str | None,
    task_store: TaskStore,
    tool_mgr: ToolManager,
    agent_id: str,
) -> ExecutorContext:
    """
    Build an :class:`ExecutorContext` for an agent execution.

    The ``await_tool_output`` callback bridges client-side tool
    calls from internal executors (Claude SDK) to the client.
    The ``call_server_tool`` callback lets executors call
    agent-plane server-side tools (e.g. ``spawn_sub_agents``)
    directly without client tunneling.

    :param task_id: Current task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: Current conversation identifier,
        e.g. ``"conv_abc123"``.
    :param storage_dir: Per-conversation working directory for
        executors that need persistent local state (e.g. Claude
        SDK session transcripts).
    :param root_task_id: The root task ID for sub-agents, or
        ``None`` for root-level tasks. Determines which SSE
        stream receives the ``function_call`` event.
    :param task_store: Task store for pending tool call operations.
    :param tool_mgr: The ToolManager for server-side tool dispatch.
    :param agent_id: The agent ID for the ToolContext.
    :returns: A configured ExecutorContext.
    """
    await_tool_output = _build_await_tool_output(
        task_id,
        root_task_id,
        task_store,
    )
    tool_ctx = ToolContext(task_id=task_id, agent_id=agent_id)
    server_names = frozenset(tool_mgr.get_tool_names())

    def _call_tool(call: ToolCallRequested) -> ToolResult:
        """
        Route a tool call to server-side or client-side execution.

        Server-side tools (registered in ToolManager) are called
        directly. Everything else is tunneled to the client via
        the parking/polling mechanism.

        :param call: The tool call to execute.
        :returns: The tool's result.
        """
        if call.name in server_names:
            result_str = tool_mgr.call_tool(
                call.name,
                json.dumps(call.arguments),
                tool_ctx,
            )
            return ToolResult(content=result_str, status="completed")
        return await_tool_output(call)

    return ExecutorContext(
        task_id=task_id,
        conversation_id=conversation_id,
        storage_dir=storage_dir,
        call_tool=_call_tool,
    )


# ── await_tool_output implementation ───────────────────────


# Polling interval for client tool result delivery.
_TOOL_POLL_INTERVAL_SECONDS = 0.5
# Maximum time to wait for a client tool result.
_TOOL_POLL_TIMEOUT_SECONDS = 600


def _build_await_tool_output(
    task_id: str,
    root_task_id: str | None,
    task_store: TaskStore,
) -> Callable[[ToolCallRequested], ToolResult]:
    """
    Build a callback that parks a client-side tool call until
    the client delivers a result via PATCH.

    Used by internal executors (Claude SDK) whose tool handlers
    need to block until the client completes the tool. The
    callback runs in the executor's thread (not the DBOS workflow
    thread), so it uses direct DB operations and live SSE publish
    instead of DBOS primitives.

    :param task_id: The current task ID, used as the sub-agent ID
        in pending tool call rows, e.g. ``"task_sub2"``.
    :param root_task_id: The root task ID for SSE routing, or
        ``None`` for root-level tasks (publishes to own stream).
    :param task_store: Task store for pending tool call operations.
    :returns: A blocking callback suitable for
        ``ExecutorContext.call_tool``.
    """
    # Sub-agents tunnel to the root task's stream; root tasks
    # publish to their own.
    publish_target = root_task_id if root_task_id is not None else task_id

    def _callback(call: ToolCallRequested) -> ToolResult:
        """
        Register, publish, and wait for a client-side tool call.

        :param call: The tool call to park.
        :returns: The client's tool result.
        """
        _register_client_tool_call(
            call,
            publish_target,
            task_id,
            task_store,
        )
        _publish_client_tool_call(call, publish_target)
        return _poll_for_tool_result(call.call_id, task_store)

    return _callback


def _register_client_tool_call(
    call: ToolCallRequested,
    root_task_id: str,
    task_id: str,
    task_store: TaskStore,
) -> None:
    """
    Insert a pending tool call row so the PATCH handler can
    route the client's result back.

    :param call: The tool call to register.
    :param root_task_id: The root task ID for the pending row,
        e.g. ``"task_root1"``.
    :param task_id: The parked task's ID (may be the same as
        root_task_id for root-level tasks).
    :param task_store: Task store for pending tool call operations.
    """
    task_store.create_pending_tool_call(
        call_id=call.call_id,
        root_task_id=root_task_id,
        task_id=task_id,
        tool_name=call.name,
        arguments=json.dumps(call.arguments),
    )


def _publish_client_tool_call(
    call: ToolCallRequested,
    publish_task_id: str,
) -> None:
    """
    Publish a ``function_call`` item to the SSE stream so the
    client knows to execute the tool.

    Uses ``_live_publish`` directly instead of ``_write_output``
    because this runs in the executor's thread, not the DBOS
    workflow thread. The DBOS durable stream is not written — if
    the client reconnects, the GET endpoint queries
    ``pending_tool_calls`` for in-flight calls.

    :param call: The tool call to publish.
    :param publish_task_id: The task ID whose SSE stream
        receives the event, e.g. ``"task_root1"``.
    """
    _live_publish(
        publish_task_id,
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
                "status": "action_required",
            },
        },
    )


def _poll_for_tool_result(
    call_id: str,
    task_store: TaskStore,
) -> ToolResult:
    """
    Poll the task store until the pending tool call is completed.

    The PATCH handler updates the row's status to ``"completed"``
    and writes the result. This function polls until that update
    is visible. Runs in the executor's thread pool (not the DBOS
    thread pool), so ``time.sleep`` does not cause workflow thread
    exhaustion.

    :param call_id: The tool call ID to wait for,
        e.g. ``"call_abc123"``.
    :param task_store: Task store for querying pending tool calls.
    :returns: The client's tool result, or a timeout error.
    """
    deadline = _monotonic() + _TOOL_POLL_TIMEOUT_SECONDS
    while _monotonic() < deadline:
        rows = task_store.list_pending_tool_calls(
            call_id=call_id,
            status="completed",
        )
        if rows:
            result_text = rows[0].result
            if result_text is None:
                # complete_pending_tool_call always sets a non-None
                # result string. None here means the row was somehow
                # completed without a result — fail loud.
                raise ValueError(f"Pending tool call {call_id} completed with None result")
            return ToolResult(
                content=result_text,
                status="success",
            )
        time.sleep(_TOOL_POLL_INTERVAL_SECONDS)
    return ToolResult(
        content=(
            f"Timeout: client did not deliver result for"
            f" {call_id} within {_TOOL_POLL_TIMEOUT_SECONDS}s"
        ),
        status="error",
    )


# ── The agent loop ────────────────────────────────────────


async def _run_agent_loop(
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    agent_name: str,
    agent_id: str,
    instructions: str | None,
    tool_mgr: ToolManager,
    executor: Executor,
    reasoning: dict[str, str] | None = None,
) -> _AgentLoopResult:
    """
    Core agent loop: load history, call LLM with optional compaction,
    dispatch to final response or tool call handler.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param spec: The parsed AgentSpec (must have a non-None
        ``llm`` field).
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param agent_id: The registered agent ID. Injected into
        ``spawn_sub_agents`` arguments so sub-agents can load
        the root spec, e.g. ``"ag_abc123"``.
    :param instructions: Optional per-request instructions
        to include in the system message.
    :param tool_mgr: The ToolManager for this workflow.
    :param executor: The executor for LLM calls. Constructed
        by the caller via ``Executor.from_spec()``.
    :param reasoning: Optional per-request reasoning config,
        e.g. ``{"effort": "high"}``. When provided, the
        ``effort`` value overrides the agent spec's
        ``reasoning_effort``.
    :returns: A :class:`_AgentLoopResult` describing the
        terminal state of the loop.
    """
    tool_mgr.start()
    tool_schemas = tool_mgr.get_tool_schemas()
    conv_store = get_conversation_store()
    task_store = get_task_store()
    # Determine if this is a sub-agent (has root_task_id).
    # Sub-agents park when hitting client tools instead of
    # completing — the park mechanism tunnels tool calls to
    # the root response's client.
    task_row = await _to_thread(lambda: task_store.get_sync(task_id))
    root_task_id: str | None = task_row.root_task_id if task_row else None
    # Load history, using the latest compaction item as a cursor
    # to avoid loading the full conversation on long-running agents.
    history = _load_initial_history(conv_store, conversation_id)
    last_seen = history[-1].id if history else None
    output_items: list[dict[str, Any]] = []
    # For remote executors, spec.llm may be None — they ignore it.
    # Provide a stub LLMConfig so downstream code doesn't break.
    if spec.llm is not None:
        llm_config = _apply_request_reasoning(spec.llm, reasoning)
    else:
        llm_config = LLMConfig(model="remote")
    tools_config = spec.tools
    # Per-task cache for resolved file_id → base64 content.
    # Shared across iterations so the same file is fetched and
    # encoded only once per task execution.
    content_cache: dict[str, str] = {}
    # Resolve execution timeout: min(spec, runtime cap)
    caps = get_caps()
    execution_timeout = min(spec.executor.timeout, caps.execution_timeout)
    max_iterations = spec.executor.max_iterations
    start_time = _monotonic()
    # Track spawned sub-agent response IDs for auto-collect.
    # When the LLM produces a final response without collecting,
    # the loop auto-collects outstanding sub-agents before
    # completing the turn.
    # Recover state from history so sub-agents spawned in
    # previous responses (before a client-tool round-trip) are
    # still tracked for auto-collect.
    spawned_ids, collected_ids = _recover_spawn_state(history)
    # Per-execution compaction state. context_window is seeded from
    # the executor's known limit (if any) so proactive compaction
    # can fire from the first iteration. Falls back to None when the
    # executor doesn't know its window (discovered on first overflow).
    compaction_state = _CompactionState(
        context_window=executor.max_context_tokens(),
        last_summary=None,
        config=spec.compaction,
        model=llm_config.model,
        connection=llm_config.connection,
    )
    # Stable storage dir scoped to (conversation, agent). Reused across
    # tasks — only restored from artifact store if empty (first task or
    # server restart).
    storage_dir = _get_or_restore_executor_storage(
        conversation_id,
        agent_name,
    )
    executor_context = _build_executor_context(
        task_id,
        conversation_id,
        storage_dir,
        root_task_id,
        task_store,
        tool_mgr,
        agent_id,
    )
    executor.on_task_start(executor_context)

    try:
        for iteration in range(max_iterations):
            # Check execution timeout at the top of each iteration.
            elapsed = _monotonic() - start_time
            if elapsed >= execution_timeout:
                return _handle_execution_timeout(task_id, output_items, execution_timeout)

            _logger.debug(
                "agent loop iteration %d for task %s",
                iteration,
                task_id,
            )
            last_seen = _sync_history(
                conv_store,
                conversation_id,
                last_seen,
                history,
            )
            llm_resp = await _executor_turn_with_compaction(
                task_id,
                executor,
                spec,
                llm_config,
                history,
                instructions,
                tool_schemas,
                compaction_state,
                executor_context,
                content_cache,
            )

            # Emit provider-native tool items (e.g. web_search_call) to
            # output before handling function calls or final response.
            _emit_native_tool_items(task_id, llm_resp, output_items)

            # Persist executor-observed tool calls (e.g. Claude SDK
            # running tools internally). SSE was already emitted
            # during the turn; this only persists to the store.
            # Advance last_seen past the persisted items so
            # _check_steering_inbox (inside _handle_final_response)
            # doesn't treat them as late steered messages.
            obs_cursor = _persist_observed_tool_calls(
                task_id,
                conversation_id,
                agent_name,
                llm_resp,
                history,
                output_items,
                conv_store,
            )
            if obs_cursor is not None:
                last_seen = obs_cursor
                # Track spawn/collect from observed tool calls so
                # auto-collect sees sub-agents spawned by executor-
                # managed turns (e.g. Claude SDK calling
                # spawn_sub_agents via MCP).
                _track_spawn_collect(
                    output_items,
                    spawned_ids,
                    collected_ids,
                )

            if not _has_tool_calls(llm_resp):
                # Auto-collect outstanding sub-agents before completing.
                uncollected = spawned_ids - collected_ids
                if uncollected:
                    # last_seen is always set after the first
                    # iteration (spawn persisted items).
                    assert last_seen is not None
                    # Persist the LLM text BEFORE auto-collect.
                    # Without this, text is streamed to SSE but
                    # never committed — ghost tokens that break
                    # steering (the LLM's steering response is
                    # lost and subsequent calls ignore the steer).
                    collect_result = await _persist_text_before_auto_collect(
                        task_id,
                        conversation_id,
                        llm_resp,
                        agent_name,
                        list(uncollected),
                        history,
                        output_items,
                        conv_store,
                    )
                    last_seen = collect_result.last_seen
                    # Only mark actually-collected IDs — partial
                    # collection (steering interrupt) leaves the
                    # rest for the next auto-collect cycle.
                    collected_ids.update(
                        collect_result.collected_ids,
                    )
                    # Re-run the LLM so it can see collected
                    # results and any steered messages.
                    continue
                result = _handle_final_response(
                    task_id,
                    conversation_id,
                    llm_resp,
                    agent_name,
                    last_seen,
                    history,
                    output_items,
                    task_store,
                    conv_store,
                )
                if isinstance(result, _SteeringRetry):
                    # Late steered messages arrived during streaming.
                    # _handle_final_response persisted the assistant
                    # response and appended both it and the steered
                    # messages to history. Use the cursor from the
                    # retry (the assistant message's ID, which has
                    # the highest store position) so _sync_history
                    # doesn't re-fetch already-processed items.
                    last_seen = result.last_seen
                    continue
                return result

            # Save the pre-tool last_seen so we can detect steered
            # messages that arrived during tool execution. Tool
            # outputs get positions after the steered message, so
            # using the post-tool last_seen would skip it.
            pre_tool_last_seen = last_seen
            handle_result = await _handle_tool_calls(
                task_id,
                conversation_id,
                llm_resp,
                agent_name,
                agent_id,
                tools_config,
                history,
                output_items,
                conv_store,
                tool_mgr,
            )
            if isinstance(handle_result, _ClientToolCallsPending):
                if root_task_id is not None:
                    # Sub-agent: park and wait for client to deliver
                    # tool results via PATCH on the root response.
                    last_seen = await _park_for_client_tools(
                        task_id,
                        conversation_id,
                        root_task_id,
                        handle_result.client_call_ids,
                        handle_result.last_seen,
                        history,
                        output_items,
                        task_store,
                        conv_store,
                    )
                    continue
                # Top-level task: return function_call items to the
                # caller and complete without server-side execution.
                return _complete_for_client_tools(
                    task_id,
                    conversation_id,
                    handle_result.last_seen,
                    output_items,
                    task_store,
                )
            last_seen = handle_result
            # Track spawned/collected sub-agent IDs for auto-collect.
            _track_spawn_collect(output_items, spawned_ids, collected_ids)
            # Check for steered messages that arrived between the
            # LLM call and tool completion. Use the pre-tool cursor
            # to catch messages with positions interleaved among
            # tool call items.
            last_seen = _sync_steered_after_tools(
                conv_store,
                conversation_id,
                pre_tool_last_seen,
                last_seen,
                history,
            )

        # Hit max iterations without a final response
        return _AgentLoopResult(
            status="incomplete",
            output=output_items,
            incomplete_details={"reason": "max_iterations"},
        )
    finally:
        executor.on_task_end(executor_context)
        # Snapshot to artifact store for server restart recovery.
        # The directory itself stays on disk for the next task.
        _persist_executor_storage(
            conversation_id,
            agent_name,
            storage_dir,
        )
        # Persist a compaction item if Layer 2 ran during this
        # execution. Idempotent — safe to call on crash recovery
        # replay because _maybe_persist_compaction_item checks
        # for an existing item with the same response_id first.
        if compaction_state.last_summary is not None:
            _maybe_persist_compaction_item(
                compaction_state.last_summary,
                task_id,
                conversation_id,
                conv_store,
            )


def _find_spec_by_name(
    spec: AgentSpec,
    name: str,
) -> AgentSpec | None:
    """
    Recursively search the spec tree for a sub-agent by name.

    Sub-agent names are validated to be unique across the entire
    spec tree, so this always finds at most one match.

    :param spec: The root agent spec to search.
    :param name: The sub-agent name to find,
        e.g. ``"researcher"``.
    :returns: The matching sub-agent spec, or ``None`` if not
        found.
    """
    for sa in spec.sub_agents:
        if sa.name == name:
            return sa
        found = _find_spec_by_name(sa, name)
        if found is not None:
            return found
    return None


async def _resolve_agent_spec_for_task(
    task_id: str,
    root_spec: AgentSpec,
) -> AgentSpec:
    """
    Resolve the effective :class:`AgentSpec` for a workflow execution.

    Every workflow execution runs with the root agent's spec (loaded
    once from the bundle at workflow start). For sub-agent workflows the
    root spec contains the full spec tree, so the sub-agent's own
    config lives inside ``root_spec.sub_agents``. This function
    determines whether the current execution is a top-level task or a
    spawned sub-agent, and returns the correct slice of the tree.

    For top-level tasks (``root_task_id IS NULL``): returns
    ``root_spec`` unchanged — the workflow *is* the root agent.

    For sub-agent tasks (``root_task_id IS NOT NULL``): looks up
    ``task.agent_name`` in the spec tree and returns the matching
    nested :class:`AgentSpec`. The sub-agent's ``spec.name`` is
    identical to ``task.agent_name`` by construction (SpawnTool
    validates the name against the spec before creating the task).

    :param task_id: The task identifier,
        e.g. ``"task_abc123"``.
    :param root_spec: The root agent's parsed spec, which contains
        the full sub-agent spec tree.
    :returns: The :class:`AgentSpec` to use for this execution.
    :raises LookupError: If the task row is missing, or the
        sub-agent name recorded on the task is not found in the
        spec tree.
    """
    task = await _to_thread(lambda: get_task_store().get_sync(task_id))
    if task is None:
        raise LookupError(f"task {task_id!r} not found")

    if task.root_task_id is None:
        # Top-level task — this workflow IS the root agent
        return root_spec

    # Sub-agent — find spec by agent_name in the tree
    sub_spec = _find_spec_by_name(root_spec, task.agent_name)
    if sub_spec is None:
        raise LookupError(f"sub-agent {task.agent_name!r} not found in spec tree")
    return sub_spec


@workflow()
async def agent_execution_workflow(
    agent_id: str,
    conversation_id: str,
    previous_response_id: str | None = None,
    instructions: str | None = None,
    reasoning: dict[str, str] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    The real agent execution loop.

    Loads the agent, builds a prompt from conversation history,
    calls the LLM, executes tool calls, and repeats until the
    LLM produces a final text response or we hit the iteration
    limit.

    ``previous_response_id``, ``reasoning``, and ``tools`` are
    persisted as workflow inputs and restored on crash recovery.

    :param agent_id: Unique agent identifier, e.g.
        ``"ag_abc123"``.
    :param conversation_id: The conversation to execute in,
        e.g. ``"conv_abc123"``.
    :param previous_response_id: The response ID of the
        previous turn, or ``None`` for the first turn.
        Persisted for recovery; not used by the loop.
    :param instructions: Optional per-request instructions
        to include in the system message.
    :param reasoning: Optional reasoning configuration dict,
        e.g. ``{"effort": "high"}``. When provided, the
        ``effort`` value overrides the agent spec's
        ``reasoning_effort`` for this execution.
    :param tools: Optional list of client-specified tool dicts in
        standard OpenAI function format. When the LLM invokes one,
        the ``function_call`` output items are returned to the caller
        (the response completes) rather than being executed
        server-side. Persisted for recovery. ``None`` and ``[]``
        are equivalent (no client tools), e.g.
        ``[{"type": "function", "function": {"name": "...",
        "description": "...", "parameters": {...}}}]``.
    :returns: A result dict with ``"task_id"``,
        ``"status"``, and ``"output"`` keys.
    """
    task_id = get_workflow_id()
    tool_mgr: ToolManager | None = None

    try:
        loaded = get_agent_cache().load(agent_id)
        root_spec = loaded.spec

        # Resolve spec for sub-agents: if root_task_id is set,
        # find the sub-agent spec by agent_name in the tree.
        spec = await _resolve_agent_spec_for_task(task_id, root_spec)
        # spec.name is None only for partially-constructed specs that
        # haven't been registered — fall back to agent_id for display.
        agent_name: str = spec.name or agent_id

        if spec.llm is None and spec.executor.type == "llm":
            return _AgentLoopResult(
                status="failed",
                output=[],
                error={
                    "code": "configuration_error",
                    "message": ("Agent spec has no LLM configuration"),
                },
            ).to_dict(task_id)

        client_tool_specs: list[ClientSideToolSpec] = parse_client_side_tool_specs(tools or [])
        tool_mgr = ToolManager(
            spec,
            client_tool_specs=client_tool_specs,
            workdir=loaded.workdir,
        )
        set_tool_manager(tool_mgr)
        executor = _create_executor(spec)

        result = await _run_agent_loop(
            task_id,
            conversation_id,
            spec,
            agent_name,
            agent_id,
            instructions,
            tool_mgr,
            executor,
            reasoning=reasoning,
        )
        return result.to_dict(task_id)
    except Exception:
        _logger.exception(
            "agent loop failed for task %s",
            task_id,
        )
        raise
    finally:
        await _close_output(task_id)
        if tool_mgr is not None:
            tool_mgr.shutdown()
        set_tool_manager(None)
