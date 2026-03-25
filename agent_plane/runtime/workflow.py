"""Agent execution workflow — the core agent loop.

Load agent → build prompt → call LLM → execute tools → repeat.
All durably checkpointed by DBOS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import openai
from openai import Stream
from openai.types.responses import (
    Response as OpenAIResponse,
)
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.shared_params import Reasoning as OpenAIReasoning

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
from agent_plane.runtime.prompt import build_instructions, history_to_input_items
from agent_plane.runtime.tool_manager import ToolManager
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig
from agent_plane.stores import ConversationStore, TaskStore

_logger = logging.getLogger(__name__)

# Hard upper bound on LLM turns per execution. Prevents runaway loops.
# See AGENTLOOP.md "Not Yet" for making this configurable.
_MAX_ITERATIONS = 1000

# Lazy singleton — created on first LLM call so import doesn't
# fail when OPENAI_API_KEY is not yet set.
_openai_client: openai.OpenAI | None = None


def _get_openai_client() -> openai.OpenAI:
    """Return the shared OpenAI client, creating it on first use."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


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
    reasoning: OpenAIReasoning | None


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
    reasoning: OpenAIReasoning | None = None
    if reasoning_effort:
        # summary="concise" enables reasoning summary streaming events.
        reasoning = OpenAIReasoning(effort=reasoning_effort, summary="concise")

    kwargs: dict[str, Any] = {"model": model, **remaining}
    if tools:
        kwargs["tools"] = tools
    return _ResponsesCallArgs(kwargs=kwargs, reasoning=reasoning)


def _response_to_dict(resp: OpenAIResponse) -> dict[str, Any]:
    """
    Extract text and tool calls from a Responses API ``Response``
    object into a JSON-serializable dict.

    :param resp: A completed ``openai.types.responses.Response``
        object from ``client.responses.create(stream=False)``.
    :returns: A dict with ``"model"`` (str or None), ``"text"``
        (str or None), and ``"tool_calls"`` (list of
        ``{"call_id", "name", "arguments"}`` dicts).
    """
    text: str | None = None
    tool_calls: list[dict[str, Any]] = []

    for item in resp.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text" and part.text:
                    text = part.text
                    break
        elif item.type == "function_call":
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
    input_items: list[dict[str, Any]],
    instructions: str,
    model: str,
    tools: list[dict[str, Any]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """
    Call the LLM via the Responses API (non-streaming). Returns the
    accumulated response as a JSON-serializable dict for DBOS
    checkpointing.

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
    :returns: A JSON-serializable dict with ``"model"``, ``"text"``,
        and ``"tool_calls"`` keys.
    """
    args = _build_responses_args(model, tools, extra)
    # cast: responses.create() without stream returns OpenAIResponse
    resp = cast(
        OpenAIResponse,
        _get_openai_client().responses.create(
            # cast: our dict list is structurally compatible with the TypedDict union
            input=cast(Any, input_items),
            instructions=instructions,
            reasoning=args.reasoning,
            **args.kwargs,
        ),
    )
    return _response_to_dict(resp)


def _call_llm_streaming(
    task_id: str,
    input_items: list[dict[str, Any]],
    instructions: str,
    model: str,
    tools: list[dict[str, Any]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """
    Call the LLM via the Responses API with streaming enabled.
    Emits ``response.output_text.delta`` and reasoning delta events
    for each chunk, then returns the full accumulated response in the
    same dict format as :func:`_call_llm`.

    NOT a ``@step`` — streaming is incompatible with DBOS
    checkpointing since we emit side effects
    (``write_stream``) during execution.

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
    :returns: The accumulated response dict in the same shape
        as :func:`_call_llm`.
    """
    args = _build_responses_args(model, tools, extra)
    # cast: responses.create() with stream=True returns Stream[ResponseStreamEvent]
    stream_resp = cast(
        Stream[ResponseStreamEvent],
        _get_openai_client().responses.create(
            # cast: our dict list is structurally compatible with the TypedDict union
            input=cast(Any, input_items),
            instructions=instructions,
            reasoning=args.reasoning,
            stream=True,
            **args.kwargs,
        ),
    )
    return _accumulate_stream(task_id, stream_resp)


# SSE event types emitted for reasoning content
_REASONING_TEXT_EVENT = "response.reasoning_text.delta"
_REASONING_SUMMARY_EVENT = "response.reasoning_summary_text.delta"


def _accumulate_stream(
    task_id: str,
    stream_resp: Stream[ResponseStreamEvent],
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
      tokens (enabled when ``reasoning.summary`` is set)

    :param task_id: The task identifier, e.g. ``"task_abc123"``.
    :param stream_resp: The Responses API streaming response to
        iterate over.
    :returns: The accumulated response dict in the same shape as
        :func:`_call_llm`.
    """
    completed_response: OpenAIResponse | None = None

    for event in stream_resp:
        # Use isinstance for type narrowing — string comparison on
        # the discriminant field doesn't narrow the union for mypy.
        if isinstance(event, ResponseTextDeltaEvent):
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
def _call_tool(tool_name: str, arguments: str) -> str:
    """
    Route a tool call to the current workflow's ToolManager.

    :param tool_name: The tool function name, e.g.
        ``"load_skill"``.
    :param arguments: JSON-encoded arguments string from the
        LLM, e.g. ``'{"name": "summarize"}'``.
    :returns: The tool's string result.
    """
    mgr = get_tool_manager()
    return mgr.call_tool(tool_name, arguments)


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
) -> dict[str, Any] | None:
    """
    Handle the no-tool-calls path: close inbox, check for
    late messages, persist assistant message, stream it, and
    return the result dict.

    Returns ``None`` if late messages arrived and the caller
    should continue the loop.

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
        ``"status"``, and ``"output"`` if the response was
        finalized, or ``None`` if the loop should continue.
    """
    # Final text response — check steering inbox
    late = task_store.close_inbox(
        task_id,
        conversation_id,
        last_seen,
    )
    if late:
        # New messages arrived — add to history and retry
        history.extend(late)
        return None

    # Persist assistant message to conversation
    text = _get_text_content(llm_resp)
    item = _build_assistant_item(task_id, agent_name, text)
    _persist_and_stream(
        task_id,
        conv_store,
        conversation_id,
        [item],
        output_items,
    )

    return {
        "task_id": task_id,
        "status": "completed",
        "output": output_items,
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
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
) -> str:
    """
    Execute each tool call and persist its output.
    Returns the ``last_seen`` ID after all tools complete.

    :param task_id: The task identifier, e.g.
        ``"task_abc123"``.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc123"``.
    :param tool_calls: Tool call dicts from the LLM response,
        each containing ``"id"`` and ``"function"`` keys.
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
            tc["name"],
            tc["arguments"],
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
        history,
        output_items,
        conv_store,
    )


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
        )
    return _call_llm(
        input_items,
        sys_instructions,
        llm_config.model,
        tool_schemas,
        llm_config.extra,
    )


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

    for iteration in range(_MAX_ITERATIONS):
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
        llm_resp = _call_llm_for_iteration(
            task_id,
            spec,
            llm_config,
            history,
            instructions,
            tool_schemas,
            # Stream tokens so SSE consumers see incremental output.
            stream=True,
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
            if result is not None:
                return result
            # Late messages arrived — _handle_final_response
            # extended history with non-empty late items
            last_seen = history[-1].id
            continue

        last_seen = _handle_tool_calls(
            task_id,
            conversation_id,
            llm_resp,
            agent_name,
            history,
            output_items,
            conv_store,
        )

    # Hit max iterations without a final response
    return {
        "task_id": task_id,
        "status": "incomplete",
        "output": output_items,
        "incomplete_details": {"reason": "max_output_tokens"},
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
