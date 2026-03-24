"""Agent execution workflow — the core agent loop.

Load agent → build prompt → call LLM → execute tools → repeat.
All durably checkpointed by DBOS.
"""

from __future__ import annotations

import logging
from typing import Any

import litellm

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
from agent_plane.runtime.prompt import build_messages
from agent_plane.runtime.tool_manager import ToolManager
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig
from agent_plane.stores import ConversationStore, TaskStore

_logger = logging.getLogger(__name__)

# Hard upper bound on LLM turns per execution. Prevents runaway loops.
# See AGENTLOOP.md "Not Yet" for making this configurable.
_MAX_ITERATIONS = 32


def _write_output(task_id: str, event: dict[str, Any]) -> None:
    """Write an event to both DBOS (durable) and live stream (real-time)."""
    write_stream("output", event)
    _live_publish(task_id, event)


def _close_output(task_id: str) -> None:
    """Close both the DBOS stream and the live stream."""
    close_stream("output")
    _live_close(task_id)


# ── DBOS-checkpointed steps ──────────────────────────────


@step()
def _call_llm(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """
    Call the LLM via litellm (non-streaming). Returns the full response
    as a dict (JSON-serializable for DBOS checkpointing).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        # litellm passes this through to providers that support it
        kwargs["reasoning_effort"] = reasoning_effort

    resp = litellm.completion(**kwargs)
    # litellm.completion returns ModelResponse which has model_dump()
    result: dict[str, Any] = resp.model_dump()
    return result


def _call_llm_streaming(
    task_id: str,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """
    Call the LLM via litellm with streaming enabled. Emits
    response.output_text.delta events for each text chunk, then
    returns the full accumulated response in the same dict format
    as _call_llm.

    NOT a @step — streaming is incompatible with DBOS checkpointing
    since we emit side effects (write_stream) during execution.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    stream_resp = litellm.completion(**kwargs)

    accumulated = _accumulate_stream(task_id, stream_resp)
    return accumulated


def _accumulate_tool_call_delta(
    tc_delta: dict[str, Any],
    tool_calls_by_index: dict[int, dict[str, Any]],
) -> None:
    """Merge a single streamed tool_call delta into the accumulator."""
    idx: int = tc_delta["index"]
    if idx not in tool_calls_by_index:
        # Empty strings are intentional — they're concatenation
        # accumulators, not sentinels. Each chunk appends to them.
        tool_calls_by_index[idx] = {
            "id": tc_delta.get("id", ""),
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
    entry = tool_calls_by_index[idx]
    fn_delta = tc_delta.get("function", {})
    if fn_delta.get("name"):
        entry["function"]["name"] += fn_delta["name"]
    if fn_delta.get("arguments"):
        entry["function"]["arguments"] += fn_delta["arguments"]
    if tc_delta.get("id"):
        entry["id"] = tc_delta["id"]


def _build_accumulated_response(
    text_parts: list[str],
    tool_calls_by_index: dict[int, dict[str, Any]],
    model: str | None,
    finish_reason: str | None,
) -> dict[str, Any]:
    """Build a response dict matching _call_llm's return shape."""
    full_text = "".join(text_parts) if text_parts else None
    message: dict[str, Any] = {"role": "assistant", "content": full_text}
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return {
        "model": model,
        "choices": [{"message": message, "finish_reason": finish_reason}],
    }


def _accumulate_stream(
    task_id: str,
    stream_resp: litellm.CustomStreamWrapper,
) -> dict[str, Any]:
    """
    Consume a litellm streaming response, emit text deltas via
    _write_output (DBOS + live stream), and return the full
    response dict.
    """
    text_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    model: str | None = None
    finish_reason: str | None = None

    for chunk in stream_resp:
        chunk_dict: dict[str, Any] = chunk.model_dump()
        if model is None:
            model = chunk_dict.get("model")

        delta = chunk_dict["choices"][0].get("delta", {})
        choice_finish = chunk_dict["choices"][0].get("finish_reason")
        if choice_finish is not None:
            finish_reason = choice_finish

        text_delta: str | None = delta.get("content")
        if text_delta:
            text_parts.append(text_delta)
            _write_output(
                task_id,
                {
                    "type": "response.output_text.delta",
                    "delta": text_delta,
                },
            )

        for tc_delta in delta.get("tool_calls") or []:
            _accumulate_tool_call_delta(tc_delta, tool_calls_by_index)

    return _build_accumulated_response(
        text_parts,
        tool_calls_by_index,
        model,
        finish_reason,
    )


@step()
def _call_tool(tool_name: str, arguments: str) -> str:
    """
    Route a tool call to the current workflow's ToolManager.
    """
    mgr = get_tool_manager()
    return mgr.call_tool(tool_name, arguments)


# ── Output helpers ────────────────────────────────────────


def _item_to_output(item: ConversationItem) -> dict[str, Any]:
    """
    Convert a persisted ConversationItem to the API output format.
    Mirrors _to_api_item() in conversations.py — see LOOPGAPS.md.
    """
    return {
        "id": item.id,
        "response_id": item.response_id,
        "type": item.type,
        "status": item.status,
        **item.data.model_dump(exclude_none=True, by_alias=True),
    }


def _has_tool_calls(llm_resp: dict[str, Any]) -> bool:
    """Check whether the LLM response contains tool calls."""
    msg = llm_resp["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")
    return bool(tool_calls)


def _get_tool_calls(
    llm_resp: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract the tool call list from the LLM response."""
    msg = llm_resp["choices"][0]["message"]
    calls: list[dict[str, Any]] = msg.get("tool_calls", [])
    return calls


def _get_text_content(llm_resp: dict[str, Any]) -> str | None:
    """Extract text content from the LLM response."""
    msg = llm_resp["choices"][0]["message"]
    content: str | None = msg.get("content")
    return content


# ── Pagination helper ─────────────────────────────────────


def fetch_all_items(
    conv_store: ConversationStore,
    conversation_id: str,
    after: str | None = None,
) -> list[ConversationItem]:
    """
    Fetch all conversation items starting after the given cursor,
    paginating through every page until has_more is False.
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
    and write each to the stream. Mutates output_items in place.
    Returns the persisted ConversationItem list.
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
    Handle the no-tool-calls path: close inbox, check for late
    messages, persist assistant message, stream it, and return
    the result dict. Returns None if late messages arrived and
    the caller should continue the loop.
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

    _close_output(task_id)
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
    Build NewConversationItem list for function_call entries.
    """
    fc_new_items: list[NewConversationItem] = []
    for tc in tool_calls:
        fc_new_items.append(
            NewConversationItem(
                type="function_call",
                response_id=task_id,
                data=FunctionCallData(
                    agent=agent_name,
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    call_id=tc["id"],
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
    Returns the last_seen id after all tools complete.
    """
    last_seen: str | None = None
    for tc in tool_calls:
        result = _call_tool(
            tc["function"]["name"],
            tc["function"]["arguments"],
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
                        call_id=tc["id"],
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
    Handle the tool execution path: build function_call items,
    persist them, execute each tool, and persist outputs.
    Returns the updated last_seen id.
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
    Returns the updated last_seen id.
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

    When stream=True, emits response.output_text.delta events for
    each text chunk via _write_output so SSE consumers see tokens
    incrementally. Falls back to the DBOS-checkpointed @step when
    stream=False (used for tool-call iterations where checkpointing
    matters more than token-level output).
    """
    messages = build_messages(
        spec,
        history,
        instructions,
        tool_schemas,
    )
    if stream:
        return _call_llm_streaming(
            task_id,
            messages,
            llm_config.model,
            tool_schemas,
            llm_config.max_completion_tokens,
            llm_config.reasoning_effort,
        )
    return _call_llm(
        messages,
        llm_config.model,
        tool_schemas,
        llm_config.max_completion_tokens,
        llm_config.reasoning_effort,
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
    _close_output(task_id)
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

    previous_response_id and reasoning are not used by the loop
    but must be in the signature — DBOS checkpoints all workflow
    inputs and restores them on recovery. task_store.get()
    reads them back for the API response.
    """
    task_id = get_workflow_id()

    # Phase 1: Load agent
    loaded = get_agent_cache().load(agent_id)
    spec = loaded.spec
    work_dir = loaded.workdir

    # Resolve the agent's registered name for output items.
    agent = get_agent_store().get(agent_id)
    agent_name = agent.name if agent else agent_id

    if spec.llm is None:
        _close_output(task_id)
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

    try:
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
        tool_mgr.shutdown()
        set_tool_manager(None)
