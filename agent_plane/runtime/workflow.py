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
    get_caps,
    get_conversation_store,
    get_task_store,
    get_tool_manager,
    set_tool_manager,
)
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
from llms import Client as LLMClient
from llms.errors import PermanentLLMError, RetryableLLMError
from llms.types import (
    FunctionCallOutput,
    MessageOutput,
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
# See AGENTLOOP.md "Not Yet" for making this configurable.
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
    Extract text and tool calls from a Responses API ``Response``
    object into a JSON-serializable dict.

    :param resp: A completed ``llms.types.Response`` object from
        ``client.responses.create(stream=False)``.
    :returns: A dict with ``"model"`` (str or None), ``"text"``
        (str or None), and ``"tool_calls"`` (list of
        ``{"call_id", "name", "arguments"}`` dicts).
    """
    text: str | None = None
    tool_calls: list[dict[str, Any]] = []

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

    return {"model": resp.model, "text": text, "tool_calls": tool_calls}


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
        elif isinstance(event, ResponseCompletedEvent):
            completed_response = event.response

    if completed_response is not None:
        return _response_to_dict(completed_response)

    # Stream completed without a response.completed event (e.g. error).
    # Return an empty response so the loop can handle it gracefully.
    return {"model": None, "text": None, "tool_calls": []}


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
    see LOOPGAPS.md.

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
) -> list[dict[str, Any]]:
    """
    Extract the tool call list from the LLM response.

    :param llm_resp: The LLM response dict (from
        :func:`_call_llm` or :func:`_call_llm_streaming`).
    :returns: List of tool call dicts, each containing
        ``"call_id"``, ``"name"``, and ``"arguments"`` keys.
        Empty list if no tool calls.
    """
    tool_calls: list[dict[str, Any]] = llm_resp["tool_calls"]
    return tool_calls


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
) -> dict[str, Any] | _SteeringRetry:
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
    :returns: A result dict with ``"task_id"``,
        ``"status"``, and ``"output"`` when the response is
        finalized. A ``_SteeringRetry`` when late messages
        arrived and the caller should continue the loop.
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
    late = task_store.close_inbox(
        task_id,
        conversation_id,
        last_seen,
    )

    # Filter out items we just persisted — close_inbox
    # returns ALL items newer than last_seen, including the
    # assistant message we committed in step 1. Exclude by
    # ID so steered user messages (which share the same
    # response_id as the running task) are preserved.
    own_ids = {ci.id for ci in persisted}
    steered = [ci for ci in late if ci.id not in own_ids]

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
    return {
        "task_id": task_id,
        "status": "completed",
        "output": output_items,
        "completed_at": int(time.time()),
    }


def _build_function_call_items(
    task_id: str,
    agent_name: str,
    tool_calls: list[dict[str, Any]],
) -> list[NewConversationItem]:
    """
    Build NewConversationItem list for ``function_call``
    entries.

    :param task_id: The task identifier used as the
        ``response_id``, e.g. ``"task_abc123"``.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param tool_calls: Tool call dicts from the LLM response,
        each containing ``"id"`` and ``"function"`` keys.
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
                    name=tc["name"],
                    arguments=tc["arguments"],
                    call_id=tc["call_id"],
                ),
            )
        )
    return fc_new_items


def _execute_tools(
    task_id: str,
    conversation_id: str,
    tool_calls: list[dict[str, Any]],
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
    :param tool_calls: Tool call dicts from the LLM response,
        each containing ``"call_id"``, ``"name"``, and
        ``"arguments"`` keys.
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
            tc["name"],
            tc["arguments"],
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
                        call_id=tc["call_id"],
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


def _handle_tool_calls(
    task_id: str,
    conversation_id: str,
    llm_resp: dict[str, Any],
    agent_name: str,
    tools_config: ToolsConfig,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> str:
    """
    Handle the tool execution path: build ``function_call``
    items, persist them, execute each tool, and persist
    outputs. Returns the updated ``last_seen`` ID.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param llm_resp: The LLM response dict containing tool
        calls.
    :param agent_name: The agent's registered name, e.g.
        ``"research-agent"``.
    :param tools_config: The agent's global tools config with
        default timeout and retry policy.
    :param history: Mutable conversation history. Extended
        in place with function call and output items.
    :param output_items: Mutable list of API-format output
        dicts. Extended in place.
    :param conv_store: The ConversationStore for persistence.
    :returns: The ID of the last persisted item.
    """
    # Tool calls — persist function_call items
    tool_calls = _get_tool_calls(llm_resp)
    fc_new_items = _build_function_call_items(
        task_id,
        agent_name,
        tool_calls,
    )

    fc_items = _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        fc_new_items,
        output_items,
    )
    history.extend(fc_items)

    # Execute each tool call and persist output
    return _execute_tools(
        task_id,
        conversation_id,
        tool_calls,
        tools_config,
        history,
        output_items,
        conv_store,
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
    :returns: The LLM response dict.
    """
    sys_instructions = build_instructions(spec, instructions, tool_schemas)
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
) -> dict[str, Any]:
    """
    Handle execution timeout: emit SSE error event and return
    incomplete result.

    :param task_id: The task identifier for SSE event emission.
    :param output_items: Accumulated output items so far.
    :param execution_timeout: The timeout that was exceeded,
        in seconds, e.g. ``3600``.
    :returns: An incomplete result dict with
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
    return {
        "task_id": task_id,
        "status": "incomplete",
        "output": output_items,
        "incomplete_details": {"reason": "execution_timeout"},
    }


# ── The agent loop ────────────────────────────────────────


def _run_agent_loop(
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    agent_name: str,
    instructions: str | None,
    tool_mgr: ToolManager,
) -> dict[str, Any]:
    """
    Core agent loop: load history, call LLM, dispatch to
    final response or tool call handler. Returns result dict.

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
    :returns: A result dict with ``"task_id"``,
        ``"status"``, and ``"output"`` keys.
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
        )

        if not _has_tool_calls(llm_resp):
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
        last_seen = _handle_tool_calls(
            task_id,
            conversation_id,
            llm_resp,
            agent_name,
            tools_config,
            history,
            output_items,
            conv_store,
        )
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
    return {
        "task_id": task_id,
        "status": "incomplete",
        "output": output_items,
        "incomplete_details": {"reason": "max_iterations"},
    }


@workflow()
def agent_execution_workflow(
    agent_id: str,
    conversation_id: str,
    previous_response_id: str | None = None,
    instructions: str | None = None,
    reasoning: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    The real agent execution loop.

    Loads the agent, builds a prompt from conversation history,
    calls the LLM, executes tool calls, and repeats until the
    LLM produces a final text response or we hit the iteration
    limit.

    ``previous_response_id`` and ``reasoning`` are not used by
    the loop but must be in the signature — DBOS checkpoints
    all workflow inputs and restores them on recovery.
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
    :returns: A result dict with ``"task_id"``,
        ``"status"``, and ``"output"`` keys.
    """
    task_id = get_workflow_id()
    tool_mgr: ToolManager | None = None

    try:
        loaded = get_agent_cache().load(agent_id)
        spec = loaded.spec
        work_dir = loaded.workdir

        agent = get_agent_store().get(agent_id)
        agent_name = agent.name if agent else agent_id

        if spec.llm is None:
            return {
                "task_id": task_id,
                "status": "failed",
                "output": [],
                "error": {
                    "code": "configuration_error",
                    "message": "Agent spec has no LLM configuration",
                },
            }

        tool_mgr = ToolManager(spec, work_dir)
        set_tool_manager(tool_mgr)

        return _run_agent_loop(
            task_id,
            conversation_id,
            spec,
            agent_name,
            instructions,
            tool_mgr,
        )
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
