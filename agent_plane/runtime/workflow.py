"""Agent execution workflow — the core agent loop.

Load agent → build prompt → call LLM → execute tools → repeat.
All durably checkpointed by DBOS.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from agent_plane.entities import (
    ConversationItem,
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
    NewConversationItem,
)
from agent_plane.runtime import (
    get_agent_cache,
    get_agent_store,
    get_artifact_store,
    get_caps,
    get_conversation_store,
    get_file_store,
    get_task_store,
    get_tool_manager,
    set_tool_manager,
)
from agent_plane.runtime.content_resolver import resolve_content_references
from agent_plane.runtime.durability import (
    close_stream,
    get_workflow_id,
    step,
    workflow,
    write_stream,
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
from agent_plane.tools.client_specified import (
    ClientSideToolSpec,
    parse_client_side_tool_specs,
)
from llms import Client as LLMClient
from llms.errors import PermanentLLMError, RetryableLLMError
from llms.types import (
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
from llms.types import (
    Response as LLMResponse,
)

_logger = logging.getLogger(__name__)

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


def _write_output(task_id: str, event: dict[str, Any]) -> None:
    """
    Write an event to both DBOS (durable) and live stream
    (real-time).

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param event: The event dict to write, e.g.
        ``{"type": "response.output_text.delta",
        "delta": "Hello"}``.
    """
    write_stream("output", event)
    _live_publish(task_id, event)


def _close_output(task_id: str) -> None:
    """
    Close both the DBOS stream and the live stream.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    """
    close_stream("output")
    _live_close(task_id)


# ── Responses API helpers ─────────────────────────────────


@dataclass
class _AgentLoopResult:
    """
    Typed result returned by the agent loop and all its terminal
    helper functions.

    Converted to a plain JSON-serializable dict at the DBOS workflow
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
        Convert to a JSON-serializable dict for the DBOS workflow boundary.

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
    """

    last_seen: str


@dataclass
class _ToolCall:
    """
    A single tool invocation requested by the LLM.

    Extracted from the raw ``llm_resp`` dict at the
    :func:`_get_tool_calls` boundary and used throughout
    the tool execution pipeline. The raw dicts remain in
    ``llm_resp`` for DBOS checkpoint serialization; this
    dataclass is the typed representation used by workflow
    logic.

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


# ── DBOS-checkpointed steps ──────────────────────────────


@step()
def _call_llm(
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
    don't cause duplicate DBOS checkpoints.

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

    def do_call() -> dict[str, Any]:
        """Execute the non-streaming LLM call."""
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


@step()
def _call_llm_streaming(
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

    This is a ``@step`` so the result is checkpointed by DBOS.
    On crash recovery, DBOS returns the cached response without
    re-executing the LLM call. Retries are internal to this step.

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

    def do_call() -> dict[str, Any]:
        """Execute the streaming LLM call and accumulate."""
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
    reasoning delta events via :func:`_write_output` (DBOS + live
    stream), and return the full response dict.

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


@step()
def _call_tool(
    task_id: str,
    tool_name: str,
    arguments: str,
    timeout: int,
    retry_config: RetryConfig,
) -> str:
    """
    Route a tool call to the current workflow's ToolManager
    with timeout enforcement and retry.

    Retries are handled inside this ``@step`` boundary so they
    don't cause duplicate DBOS checkpoints. On exhausted retries,
    an error string is returned (not raised) so the LLM can
    decide how to proceed.

    :param task_id: The task identifier for SSE event emission,
        e.g. ``"task_abc123"``.
    :param tool_name: The tool function name, e.g.
        ``"load_skill"``.
    :param arguments: JSON-encoded arguments string from the
        LLM, e.g. ``'{"name": "summarize"}'``.
    :param timeout: Per-call timeout in seconds, e.g. ``60``.
    :param retry_config: Retry policy for this tool.
    :returns: The tool's string result, or an error string
        if all retries are exhausted.
    """
    mgr = get_tool_manager()
    return execute_tool_with_retry(
        tool_name=tool_name,
        call_fn=lambda: mgr.call_tool(tool_name, arguments),
        timeout=timeout,
        retry_config=retry_config,
        on_event=lambda event: _write_output(task_id, event),
    )


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

    Converts raw dicts (kept in ``llm_resp`` for DBOS checkpoint
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
    :returns: A NewConversationItem ready for persistence.
    """
    return NewConversationItem(
        type="message",
        response_id=task_id,
        data=MessageData(
            role="assistant",
            content=[
                {
                    "type": "output_text",
                    "text": text,
                }
            ],
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
    _logger.info(
        "[STEER-DEBUG] _handle_final_response: task=%s last_seen=%s",
        task_id, last_seen,
    )
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
    late = task_store.close_inbox(
        task_id,
        conversation_id,
        last_seen,
    )
    _logger.info(
        "[STEER-DEBUG] close_inbox returned %d items, persisted %d items",
        len(late), len(persisted),
    )
    for ci in late:
        _logger.info(
            "[STEER-DEBUG]   late item: id=%s type=%s role=%s",
            ci.id, ci.type,
            ci.data.role if hasattr(ci.data, "role") else "N/A",
        )

    # Filter out items we just persisted — close_inbox
    # returns ALL items newer than last_seen, including the
    # assistant message we committed in step 1. Exclude by
    # ID so steered user messages (which share the same
    # response_id as the running task) are preserved.
    own_ids = {ci.id for ci in persisted}
    steered = [ci for ci in late if ci.id not in own_ids]
    _logger.info(
        "[STEER-DEBUG] after filtering: %d steered items",
        len(steered),
    )

    if steered:
        # ── Step 3a: Late messages arrived ─────────────────
        # Add the persisted assistant response first (it
        # answers the original input), then the steered user
        # messages (new input for the next iteration). This
        # matches conversational order.
        history.extend(persisted)
        history.extend(steered)
        # The persisted assistant message has the highest
        # store position (appended AFTER steered messages
        # arrived during streaming). Use its ID as the
        # cursor so _sync_history doesn't re-fetch it.
        return _SteeringRetry(last_seen=persisted[-1].id)

    # ── Step 3b: No late messages — close inbox and finish ──
    # The first close_inbox call found items (our own persisted
    # output), so the inbox is still open. Call again with the
    # persisted item's ID so close_inbox sees zero new items
    # and atomically sets inbox_closed=True. This prevents
    # try_deliver from injecting orphaned messages after we
    # return "completed".
    task_store.close_inbox(
        task_id,
        conversation_id,
        persisted[-1].id,
    )
    return _AgentLoopResult(
        status="completed",
        output=output_items,
        completed_at=int(time.time()),
    )


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


def _execute_tools(
    task_id: str,
    conversation_id: str,
    tool_calls: list[_ToolCall],
    tools_config: ToolsConfig,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> str:
    """
    Execute each tool call with timeout/retry and persist output.

    Returns the ``last_seen`` ID after all tools complete.

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
    :returns: The ID of the last persisted tool output item.
    """
    last_seen: str | None = None
    for tc in tool_calls:
        result = _call_tool(
            task_id,
            tc.name,
            tc.arguments,
            tools_config.timeout,
            tools_config.retry,
        )

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
                        output=result,
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
    :param has_client: ``True`` if the batch contains at least
        one client-side tool call.
    """

    server: list[_ToolCall]
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
    has_client = False
    for tc in tool_calls:
        if tool_mgr.is_client_side_tool(tc.name):
            has_client = True
        else:
            server.append(tc)
    return _ToolCallSplit(server=server, has_client=has_client)


def _handle_tool_calls(
    task_id: str,
    conversation_id: str,
    llm_resp: dict[str, Any],
    agent_name: str,
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
        last_seen = _execute_tools(
            task_id,
            conversation_id,
            split.server,
            tools_config,
            history,
            output_items,
            conv_store,
        )

    if split.has_client:
        # Client-side function_call items are already persisted
        # and streamed. Server-side tool outputs (if any) are
        # also persisted. Complete the response so the caller
        # can handle the client-side calls externally.
        return _ClientToolCallsPending(last_seen=last_seen)

    return last_seen


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

    Known gap: unlike ``_handle_final_response``, this function
    uses the post-persist cursor (``fc_last_seen``) rather than
    the pre-LLM cursor. A steered message delivered during LLM
    streaming gets a position *between* those two cursors, so
    ``close_inbox`` won't see it. We can't retry the loop here
    because we must return the client-side tool calls immediately.
    The steered message is not lost — it is persisted in the
    conversation store and will be included in the prompt on the
    next request (when the client continues via
    ``previous_response_id``). See LOOPGAPS.md.

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
    Check steering inbox for new messages and extend history.
    Returns the updated ``last_seen`` ID.

    :param conv_store: The ConversationStore to query.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param last_seen: The ID of the last item the agent has
        seen, or ``None`` if no items have been seen yet.
    :param history: Mutable conversation history. Extended
        in place if new items are found.
    :returns: The updated ``last_seen`` ID, or the original
        value if no new items were found.
    """
    if last_seen is not None:
        new_items = fetch_all_items(
            conv_store,
            conversation_id,
            after=last_seen,
        )
        if new_items:
            history.extend(new_items)
            return new_items[-1].id
    return last_seen


def _call_llm_for_iteration(
    task_id: str,
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    *,
    stream: bool = False,
    content_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build prompt messages and call the LLM for one iteration.

    When ``stream=True``, emits
    ``response.output_text.delta`` events for each text chunk
    via :func:`_write_output` so SSE consumers see tokens
    incrementally. Falls back to the DBOS-checkpointed
    ``@step`` when ``stream=False`` (used for tool-call
    iterations where checkpointing matters more than
    token-level output).

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param spec: The parsed AgentSpec for the executing agent.
    :param llm_config: The agent's LLM configuration
        (model identifier and extra kwargs).
    :param history: Conversation history as persisted items.
    :param instructions: Optional per-request instructions.
    :param tool_schemas: OpenAI-format tool schemas for the
        agent's available tools.
    :param stream: If ``True``, stream tokens via SSE.
        If ``False``, use DBOS-checkpointed non-streaming.
    :param content_cache: Per-task cache mapping ``file_id``
        to base64-encoded content, avoiding redundant
        artifact store fetches across iterations.
    :returns: The LLM response dict.
    """
    sys_instructions = build_instructions(spec, instructions, tool_schemas)
    # Resolve file_id references to inline base64 content before
    # building the prompt. Skipped when stores are not configured.
    file_store = get_file_store()
    artifact_store = get_artifact_store()
    if file_store is not None and artifact_store is not None:
        history = resolve_content_references(history, file_store, artifact_store, content_cache)
    input_items = history_to_input_items(history)
    if stream:
        return _call_llm_streaming(
            task_id,
            input_items,
            sys_instructions,
            llm_config.model,
            tool_schemas,
            llm_config.extra,
            llm_config.connection,
            llm_config.timeout,
            llm_config.retry,
        )
    return _call_llm(
        task_id,
        input_items,
        sys_instructions,
        llm_config.model,
        tool_schemas,
        llm_config.extra,
        llm_config.connection,
        llm_config.timeout,
        llm_config.retry,
    )


def _call_llm_for_iteration_with_error_handling(
    task_id: str,
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    content_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Call the LLM for one iteration with error handling and SSE
    error event emission.

    Wraps :func:`_call_llm_for_iteration` to catch LLM errors,
    emit a ``response.error`` SSE event, and re-raise. This keeps
    the agent loop clean of error-handling boilerplate.

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param spec: The parsed AgentSpec for the executing agent.
    :param llm_config: The agent's LLM configuration.
    :param history: Conversation history as persisted items.
    :param instructions: Optional per-request instructions.
    :param tool_schemas: OpenAI-format tool schemas.
    :param content_cache: Per-task cache mapping ``file_id``
        to base64-encoded content (see
        :func:`_call_llm_for_iteration`).
    :returns: The LLM response dict.
    :raises PermanentLLMError: On non-retryable LLM errors.
    :raises RetryableLLMError: When all retries are exhausted.
    """
    try:
        return _call_llm_for_iteration(
            task_id,
            spec,
            llm_config,
            history,
            instructions,
            tool_schemas,
            stream=True,
            content_cache=content_cache,
        )
    except (RetryableLLMError, PermanentLLMError) as exc:
        _emit_llm_error_event(task_id, exc)
        raise


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


# ── The agent loop ────────────────────────────────────────


def _run_agent_loop(
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    agent_name: str,
    instructions: str | None,
    tool_mgr: ToolManager,
) -> _AgentLoopResult:
    """
    Core agent loop: load history, call LLM, dispatch to
    final response or tool call handler.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param spec: The parsed AgentSpec (must have a non-None
        ``llm`` field).
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param instructions: Optional per-request instructions
        to include in the system message.
    :param tool_mgr: The ToolManager for this workflow.
    :returns: A :class:`_AgentLoopResult` describing the
        terminal state of the loop.
    """
    tool_mgr.start()
    tool_schemas = tool_mgr.get_tool_schemas()
    conv_store = get_conversation_store()
    task_store = get_task_store()
    history = fetch_all_items(conv_store, conversation_id)
    last_seen = history[-1].id if history else None
    output_items: list[dict[str, Any]] = []
    # spec.llm is guaranteed non-None — checked by caller
    assert spec.llm is not None
    llm_config = spec.llm
    tools_config = spec.tools
    # Per-task cache for resolved file_id → base64 content.
    # Shared across iterations so the same file is fetched and
    # encoded only once per task execution.
    content_cache: dict[str, str] = {}
    # Resolve execution timeout: min(spec, runtime cap)
    caps = get_caps()
    execution_timeout = min(spec.execution.timeout, caps.execution_timeout)
    max_iterations = spec.execution.max_iterations
    start_time = time.monotonic()

    for iteration in range(max_iterations):
        # Check execution timeout at the top of each iteration.
        elapsed = time.monotonic() - start_time
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
        llm_resp = _call_llm_for_iteration_with_error_handling(
            task_id,
            spec,
            llm_config,
            history,
            instructions,
            tool_schemas,
            content_cache,
        )

        # Emit provider-native tool items (e.g. web_search_call) to
        # output before handling function calls or final response.
        _emit_native_tool_items(task_id, llm_resp, output_items)

        has_tools = _has_tool_calls(llm_resp)
        _logger.info(
            "[STEER-DEBUG] iteration: has_tool_calls=%s last_seen=%s "
            "history_len=%d",
            has_tools, last_seen, len(history),
        )
        if not has_tools:
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
        handle_result = _handle_tool_calls(
            task_id,
            conversation_id,
            llm_resp,
            agent_name,
            tools_config,
            history,
            output_items,
            conv_store,
            tool_mgr,
        )
        if isinstance(handle_result, _ClientToolCallsPending):
            # Client-side tool calls — return function_call items to
            # the caller and complete without server-side execution.
            return _complete_for_client_tools(
                task_id,
                conversation_id,
                handle_result.last_seen,
                output_items,
                task_store,
            )
        last_seen = handle_result
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


@workflow()
def agent_execution_workflow(
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
    stored by DBOS and restored on crash recovery.
    ``task_store.get()`` reads them back for the API response.

    :param agent_id: Unique agent identifier, e.g.
        ``"ag_abc123"``.
    :param conversation_id: The conversation to execute in,
        e.g. ``"conv_abc123"``.
    :param previous_response_id: The response ID of the
        previous turn, or ``None`` for the first turn.
        Stored by DBOS for recovery; not used by the loop.
    :param instructions: Optional per-request instructions
        to include in the system message.
    :param reasoning: Optional reasoning configuration dict,
        e.g. ``{"effort": "high"}``. Stored by DBOS for
        recovery; not yet consumed by the loop.
    :param tools: Optional list of client-specified tool dicts in
        standard OpenAI function format. When the LLM invokes one,
        the ``function_call`` output items are returned to the caller
        (the response completes) rather than being executed
        server-side. Stored by DBOS for recovery. ``None`` and ``[]``
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
        spec = loaded.spec

        agent = get_agent_store().get(agent_id)
        agent_name = agent.name if agent else agent_id

        if spec.llm is None:
            return _AgentLoopResult(
                status="failed",
                output=[],
                error={
                    "code": "configuration_error",
                    "message": "Agent spec has no LLM configuration",
                },
            ).to_dict(task_id)

        client_tool_specs: list[ClientSideToolSpec] = parse_client_side_tool_specs(tools or [])
        tool_mgr = ToolManager(
            spec,
            client_tool_specs=client_tool_specs,
            workdir=loaded.workdir,
        )
        set_tool_manager(tool_mgr)

        result = _run_agent_loop(
            task_id,
            conversation_id,
            spec,
            agent_name,
            instructions,
            tool_mgr,
        )
        return result.to_dict(task_id)
    except Exception:
        _logger.exception(
            "agent loop failed for task %s",
            task_id,
        )
        raise
    finally:
        _close_output(task_id)
        if tool_mgr is not None:
            tool_mgr.shutdown()
        set_tool_manager(None)
