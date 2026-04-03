"""Layered conversation history compaction for LLM context management.

Compaction fires when the estimated prompt token count approaches the
model's context window. Three layers are applied in order, from
least-lossy to most-lossy:

1. Surgical clearing — tool result bodies and binary content blocks
   outside the recent window are replaced with markers.
2. LLM summarization — a @step LLM call summarises all messages
   outside the recent window into a single summary pair.
3. Truncation — oldest messages are dropped when layers 1+2
   are still insufficient (emergency fallback).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import tiktoken

from agent_plane.entities import (
    CompactionData,
    ConversationItem,
    MessageData,
)
from agent_plane.runtime.durability import step
from agent_plane.spec.types import CompactionConfig

_logger = logging.getLogger(__name__)

# Marker written into cleared tool result bodies.
_TOOL_RESULT_CLEARED = "[Previous tool result cleared — re-call tool if needed]"

# Marker written into cleared binary content block payloads.
_BINARY_CONTENT_CLEARED = (
    "[binary content removed for context management — use file_id to retrieve]"
)

# Default compaction settings when AgentSpec.compaction is None.
_DEFAULT_TRIGGER_THRESHOLD: float = 0.8
_DEFAULT_RECENT_WINDOW: int = 5


@dataclass
class SummaryMetadata:
    """
    Metadata from a Layer 2 summarization, passed from
    :func:`compact` to the workflow's end-of-execution
    persistence step.

    :param text: The LLM-generated summary text.
    :param last_item_id: The ID of the last conversation item
        covered by this summary, e.g. ``"msg_abc123"``.
    :param model: The model used for summarization, e.g.
        ``"openai/gpt-4o"``.
    :param token_count: Approximate token count of the summary
        text, e.g. ``342``.
    """

    text: str
    last_item_id: str
    model: str
    token_count: int


@dataclass
class CompactionResult:
    """
    Result of running :func:`compact` on a messages list.

    :param messages: The compacted messages list, ready to pass
        to the LLM.
    :param summary_metadata: Present only when Layer 2
        (summarization) was triggered. Contains the summary text
        and the ``last_item_id`` of the last item covered.
        ``None`` when only Layer 1 or Layer 3 applied, or when
        summarization failed and Layer 3 was used as fallback.
    """

    messages: list[dict[str, Any]]
    summary_metadata: SummaryMetadata | None


@dataclass
class _CompactionState:
    """
    Per-execution compaction state maintained in the agent loop.

    :param context_window: Cached context window size discovered
        from the first ContextWindowExceededError,
        e.g. ``128000``. ``None`` until the first overflow occurs.
    :param last_summary: Metadata from the most recent Layer 2
        summarization during this execution, or ``None`` if no
        summarization has occurred yet.
    :param config: The compaction config from the agent spec, or
        ``None`` to use defaults.
    :param model: The LLM model string used for tiktoken estimation,
        e.g. ``"openai/gpt-4o"``.
    """

    context_window: int | None
    last_summary: SummaryMetadata | None
    config: CompactionConfig | None
    model: str


def count_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """
    Estimate the token count for a messages list using tiktoken.

    Used as a sanity check against provider-reported token counts
    (within ~30%) and for proactive threshold checks. Not used as
    the authoritative count — tiktoken is ~85-95% accurate for
    non-OpenAI models, and the 20% headroom from
    ``trigger_threshold`` absorbs the difference.

    :param messages: The messages list to count tokens for.
    :param model: The LLM model string, e.g. ``"openai/gpt-4o"``.
        Used to select the appropriate tiktoken encoding; falls
        back to ``cl100k_base`` for unknown models.
    :returns: Approximate token count for the serialised messages.
    """
    # Strip provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o")
    # so tiktoken can look up the model encoding.
    bare_model = model.split("/", 1)[-1] if "/" in model else model
    try:
        enc = tiktoken.encoding_for_model(bare_model)
    except KeyError:
        # Unknown model — fall back to the most common encoding.
        enc = tiktoken.get_encoding("cl100k_base")
    text = json.dumps(messages, ensure_ascii=False)
    return len(enc.encode(text))


def _find_recent_boundary(
    history: list[ConversationItem],
    recent_window: int,
) -> int:
    """
    Find the index in *history* where the recent window begins.

    The recent window covers the last *recent_window* LLM response
    groups. One group = one assistant message or one function_call
    item (both mark an LLM response boundary). Items at or after
    the returned index are protected from compaction.

    :param history: The full conversation history list.
    :param recent_window: Number of LLM response groups to protect,
        e.g. ``5``.
    :returns: The index of the first item inside the recent window.
        Returns ``0`` if the history has fewer groups than the window
        size (protect everything).
    """
    groups_seen = 0
    for i in range(len(history) - 1, -1, -1):
        item = history[i]
        is_assistant_msg = (
            item.type == "message"
            and isinstance(item.data, MessageData)
            and item.data.role == "assistant"
        )
        is_function_call = item.type == "function_call"
        if is_assistant_msg or is_function_call:
            groups_seen += 1
            if groups_seen >= recent_window:
                return i
    return 0


def _clear_tool_results(
    messages: list[dict[str, Any]],
    protect_from: int,
) -> list[dict[str, Any]]:
    """
    Replace tool result bodies outside the recent window with a
    clearing marker.

    The function_call / function_call_output pair structure is
    preserved — no orphaned tool calls are created. Only the
    ``output`` field of ``function_call_output`` items is
    replaced.

    :param messages: The messages list to process (modified in place).
    :param protect_from: Index of the first message in the recent
        window. Messages at indices < *protect_from* are eligible
        for clearing.
    :returns: The same list (modified in place) for convenience.
    """
    for i, msg in enumerate(messages):
        if i >= protect_from:
            break
        if msg.get("type") == "function_call_output":
            msg["output"] = _TOOL_RESULT_CLEARED
    return messages


def _clear_binary_content(
    messages: list[dict[str, Any]],
    protect_from: int,
) -> list[dict[str, Any]]:
    """
    Replace binary payload data in image/file content blocks
    outside the recent window with a clearing marker.

    The ``file_id`` is preserved so the agent can re-fetch the
    content if needed. Text content blocks within the same message
    are untouched.

    :param messages: The messages list to process (modified in place).
    :param protect_from: Index of the first message in the recent
        window. Messages at indices < *protect_from* are eligible
        for clearing.
    :returns: The same list (modified in place) for convenience.
    """
    for i, msg in enumerate(messages):
        if i >= protect_from:
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") in ("image", "file")
                and "data" in block
            ):
                block["data"] = _BINARY_CONTENT_CLEARED
    return messages


def _truncate_oldest(
    messages: list[dict[str, Any]],
    budget: int,
    model: str,
) -> list[dict[str, Any]]:
    """
    Emergency Layer 3: drop oldest messages until the token count
    fits within *budget*.

    Preserves tool call pair integrity — never drops a
    ``function_call`` without also dropping its matching
    ``function_call_output``, and vice versa. Drops from the front
    of the list.

    :param messages: The messages list to truncate.
    :param budget: Maximum token count for the returned list,
        e.g. ``102400``.
    :param model: LLM model string for token counting.
    :returns: A new messages list with oldest items dropped.
    """
    result = list(messages)
    while result and count_tokens(result, model) > budget:
        drop_count = _pair_aware_drop_count(result)
        if drop_count == 0:
            break
        result = result[drop_count:]
    return result


def _pair_aware_drop_count(messages: list[dict[str, Any]]) -> int:
    """
    Return how many items to drop from the front to avoid
    orphaning a tool call pair.

    If the first item is a ``function_call`` and the second is its
    matching ``function_call_output``, both are dropped together.
    Otherwise, a single item is dropped.

    :param messages: The messages list (must be non-empty).
    :returns: Number of items to drop (1 or 2), or 0 if the list
        is empty.
    """
    if not messages:
        return 0
    if (
        len(messages) >= 2
        and messages[0].get("type") == "function_call"
        and messages[1].get("type") == "function_call_output"
        and messages[0].get("call_id") == messages[1].get("call_id")
    ):
        return 2
    return 1


@step()
def summarize_history(
    messages_to_summarize: list[dict[str, Any]],
    llm_client: Any,  # llms.Client — typed as Any to avoid circular import
    model: str,
) -> dict[str, Any]:
    """
    Layer 2: call the LLM to summarise conversation messages.

    Decorated with ``@step`` so DBOS checkpoints the result — on
    crash recovery the summary is replayed from cache without
    re-calling the LLM.

    :param messages_to_summarize: The messages outside the recent
        window to summarise, as Responses API input dicts. By the
        time this is called, Layer 1 has already cleared binary
        content blocks and tool result bodies from these messages.
    :param llm_client: The LLM client to use, e.g. an instance of
        ``llms.Client``.
    :param model: The LLM model string, e.g. ``"openai/gpt-4o"``.
    :returns: A dict with ``"text"`` (the summary) and
        ``"token_count"`` (approximate token count).
    """
    system_prompt = _build_summarization_prompt(messages_to_summarize)
    resp = llm_client.responses.create(
        model=model,
        input=messages_to_summarize,
        instructions=system_prompt,
        tools=[],
    )
    summary_text = _extract_summary_text(resp)
    token_count = count_tokens([{"role": "assistant", "content": summary_text}], model)
    return {"text": summary_text, "token_count": token_count}


def _build_summarization_prompt(messages_to_summarize: list[dict[str, Any]]) -> str:
    """
    Build the system prompt for Layer 2 summarization.

    Detects whether the input starts with a prior summary pair
    (progressive summarization) and prepends a continuation
    instruction if so.

    :param messages_to_summarize: The messages that will be
        summarized. Inspected to detect a prior summary header.
    :returns: The assembled system prompt string.
    """
    base_prompt = (
        "Summarize the conversation above so that a future assistant can continue\n"
        "the work without access to the original messages.\n\n"
        "Include: the user's goals, key decisions and why they were made, tool\n"
        "results that matter going forward (paths, values, errors), and any\n"
        "outstanding commitments or next steps.\n\n"
        "Exclude: verbose tool output, redundant exchanges, and intermediate\n"
        "reasoning that led to a final decision — keep the decision, not the path.\n\n"
        "Do not incorporate knowledge from outside this conversation. Do not\n"
        "invent facts. Write in plain text with no markup."
    )
    first_content = _extract_first_text(messages_to_summarize)
    if "[This is an automatically generated summary" in first_content:
        return (
            "The conversation starts with a summary of earlier context. "
            "Incorporate it into your new summary — do not discard it.\n\n"
        ) + base_prompt
    return base_prompt


def _extract_first_text(messages: list[dict[str, Any]]) -> str:
    """
    Extract the text content from the first message in a list.

    Used to detect progressive summarization by checking whether
    the first message starts with a prior summary header.

    :param messages: The messages list to inspect.
    :returns: The text of the first content block, or ``""`` if
        the list is empty or the first message has no text.
    """
    if not messages:
        return ""
    first_msg = messages[0]
    content = first_msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("input_text", "text"):
                text = block.get("text", "")
                return text if isinstance(text, str) else ""
    if isinstance(content, str):
        return content
    return ""


def _extract_summary_text(resp: Any) -> str:
    """
    Extract plain text from an LLM Responses API response object.

    Iterates over ``resp.output`` items and concatenates all text
    blocks found in their ``content`` attributes.

    :param resp: The response object from ``llm_client.responses.create()``.
    :returns: Concatenated summary text, or ``""`` if no text blocks
        are present.
    """
    summary_text = ""
    for output_item in resp.output:
        if hasattr(output_item, "content"):
            for block in output_item.content:
                if hasattr(block, "text"):
                    summary_text += block.text
    return summary_text


def compaction_to_history_items(
    compaction_item: ConversationItem,
) -> list[ConversationItem]:
    """
    Convert a compaction item into a synthetic user + assistant
    message pair for inclusion at the front of conversation history.

    The pair preserves natural turn-taking structure: a synthetic
    user message requests a summary, and a synthetic assistant
    message provides it. This avoids attribution confusion —
    the LLM knows it produced a summary (not a real prior response).

    The synthetic items are NOT persisted to the conversation store;
    they exist only in the in-memory history list for prompt
    construction.

    :param compaction_item: The compaction item from the store,
        with ``type="compaction"`` and
        ``data`` of type :class:`~agent_plane.entities.CompactionData`.
    :returns: Two :class:`~agent_plane.entities.ConversationItem`
        instances: a ``role=user`` message requesting the summary
        and a ``role=assistant`` message containing it.
    """
    assert isinstance(compaction_item.data, CompactionData)
    data = compaction_item.data

    synthetic_user_content = (
        "[This is an automatically generated summary of the prior conversation "
        "context. The original messages are available but not included in this "
        "prompt for brevity.]\n\n"
        "Please provide a summary of our conversation so far."
    )
    user_item = ConversationItem(
        id=f"{compaction_item.id}_user",
        type="message",
        status="completed",
        response_id=compaction_item.response_id,
        created_at=compaction_item.created_at,
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": synthetic_user_content}],
        ),
    )
    assistant_item = ConversationItem(
        id=f"{compaction_item.id}_assistant",
        type="message",
        status="completed",
        response_id=compaction_item.response_id,
        created_at=compaction_item.created_at,
        data=MessageData(
            role="assistant",
            content=[{"type": "output_text", "text": data.summary}],
            agent=data.model,
        ),
    )
    return [user_item, assistant_item]


def compact(
    messages: list[dict[str, Any]],
    history: list[ConversationItem],
    *,
    config: CompactionConfig | None,
    context_window: int,
    system_token_budget: int,
    model: str,
    task_id: str,
    llm_client: Any,  # llms.Client — typed as Any to avoid circular import
) -> CompactionResult:
    """
    Apply layered compaction to a messages list to fit within the
    context window budget.

    Layers are applied in order from least-lossy to most-lossy:

    1. **Layer 1** — Clear tool result bodies and binary content
       blocks outside the recent window (fast, no LLM call).
    2. **Layer 2** — LLM summarization of messages outside the
       recent window (slow, checkpointed @step).
    3. **Layer 3** — Truncate oldest messages (emergency fallback).

    The in-memory *history* list is never modified — only the
    *messages* copy passed to the LLM is compacted.

    :param messages: The messages list to compact. This is a copy
        — the original history is not modified.
    :param history: The original conversation history items, used
        to find ``last_item_id`` for the summary.
    :param config: Compaction configuration from the agent spec.
        ``None`` uses defaults.
    :param context_window: The model's context window size in tokens,
        e.g. ``128000``.
    :param system_token_budget: Tokens already consumed by the system
        prompt and tool schemas, subtracted from the window budget.
    :param model: The LLM model string, e.g. ``"openai/gpt-4o"``.
    :param task_id: The task identifier for SSE event emission.
    :param llm_client: The LLM client instance for Layer 2
        summarization.
    :returns: A :class:`CompactionResult` with the compacted messages
        and optional summary metadata.
    """
    trigger_threshold = config.trigger_threshold if config else _DEFAULT_TRIGGER_THRESHOLD
    recent_window = config.recent_window if config else _DEFAULT_RECENT_WINDOW
    # Budget = fraction of context window minus system/tool tokens.
    budget = int(context_window * trigger_threshold) - system_token_budget

    # Deep-copy messages so Layer 1 modifications don't affect the
    # caller's list.
    working = _deep_copy_messages(messages)

    history_boundary = _find_recent_boundary(history, recent_window)
    msg_boundary = _history_idx_to_msg_idx(history, history_boundary)

    # --- Layer 1 ---
    _clear_tool_results(working, msg_boundary)
    _clear_binary_content(working, msg_boundary)

    if count_tokens(working, model) <= budget:
        return CompactionResult(messages=working, summary_metadata=None)

    # --- Layer 2 ---
    summary_metadata = _run_layer2(
        working,
        history,
        history_boundary,
        msg_boundary,
        budget,
        model,
        task_id,
        llm_client,
    )
    if summary_metadata is not None:
        summary_messages = _summary_to_messages(summary_metadata)
        recent_messages = working[msg_boundary:]
        compacted = summary_messages + recent_messages
        if count_tokens(compacted, model) <= budget:
            return CompactionResult(
                messages=compacted,
                summary_metadata=summary_metadata,
            )
        # Summary + recent still exceeds budget — fall through to Layer 3.
        working = compacted

    # --- Layer 3 ---
    _logger.warning(
        "Layer 3 truncation triggered for task %s — context still exceeds budget after layers 1+2",
        task_id,
    )
    truncated = _truncate_oldest(working, budget, model)
    return CompactionResult(messages=truncated, summary_metadata=summary_metadata)


def _deep_copy_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return a deep copy of the messages list so Layer 1 clearing
    does not mutate the caller's list.

    :param messages: The messages list to copy.
    :returns: A deep copy.
    """
    result: list[dict[str, Any]] = json.loads(json.dumps(messages))
    return result


def _history_idx_to_msg_idx(
    history: list[ConversationItem],
    history_idx: int,
) -> int:
    """
    Map a history index to the corresponding messages list index.

    ``history_to_input_items`` skips reasoning items, so the
    messages list may be shorter than the history list. This
    function counts non-reasoning items up to *history_idx*.

    :param history: The full conversation history.
    :param history_idx: The index in *history* to map.
    :returns: The corresponding index in the messages list.
    """
    msg_idx = 0
    for i, item in enumerate(history):
        if i >= history_idx:
            break
        if item.type != "reasoning":
            msg_idx += 1
    return msg_idx


def _run_layer2(
    messages: list[dict[str, Any]],
    history: list[ConversationItem],
    history_boundary: int,
    msg_boundary: int,
    budget: int,
    model: str,
    task_id: str,
    llm_client: Any,
) -> SummaryMetadata | None:
    """
    Attempt Layer 2 LLM summarisation.

    Emits a ``response.compaction.in_progress`` SSE event before
    the LLM call. Returns ``None`` and falls through to Layer 3 if
    the LLM call fails.

    :param messages: The working messages list (after Layer 1).
    :param history: The original conversation history items.
    :param history_boundary: The boundary index in *history*.
    :param msg_boundary: The boundary index in *messages*.
    :param budget: Token budget for the compacted result.
    :param model: LLM model string.
    :param task_id: Task identifier for SSE event emission.
    :param llm_client: LLM client instance.
    :returns: :class:`SummaryMetadata` on success, ``None`` on failure.
    """
    _emit_compaction_event(task_id)

    to_summarize = messages[:msg_boundary]
    # If too large for the model, apply Layer 1 clearing to the
    # summarization input too.
    if count_tokens(to_summarize, model) > budget:
        to_summarize = _deep_copy_messages(to_summarize)
        _clear_tool_results(to_summarize, len(to_summarize))
        _clear_binary_content(to_summarize, len(to_summarize))

    try:
        result = summarize_history(to_summarize, llm_client, model)
    except Exception:
        _logger.warning(
            "Layer 2 summarisation failed for task %s — falling back to Layer 3",
            task_id,
            exc_info=True,
        )
        return None

    last_item_id = _find_last_summarized_item_id(history, history_boundary)
    if last_item_id is None:
        return None

    return SummaryMetadata(
        text=result["text"],
        last_item_id=last_item_id,
        model=model,
        token_count=result["token_count"],
    )


def _find_last_summarized_item_id(
    history: list[ConversationItem],
    history_boundary: int,
) -> str | None:
    """
    Find the ID of the last history item included in the summary.

    This is the item at ``history[history_boundary - 1]``, skipping
    any synthetic items (those without a real store ID). Synthetic
    items are identified by the ``_user`` or ``_assistant`` suffix
    added by :func:`compaction_to_history_items`.

    :param history: The conversation history items.
    :param history_boundary: The boundary index (exclusive).
    :returns: The last real item ID, or ``None`` if no real items
        exist before the boundary.
    """
    for i in range(history_boundary - 1, -1, -1):
        item = history[i]
        # Skip synthetic items (IDs with _user / _assistant suffix
        # added by compaction_to_history_items).
        if not item.id.endswith(("_user", "_assistant")):
            return item.id
    return None


def _summary_to_messages(
    summary: SummaryMetadata,
) -> list[dict[str, Any]]:
    """
    Convert a :class:`SummaryMetadata` into the synthetic
    user + assistant message pair for inclusion in the prompt.

    :param summary: The summary metadata from Layer 2.
    :returns: Two message dicts: user request and assistant summary.
    """
    user_text = (
        "[This is an automatically generated summary of the prior conversation "
        "context. The original messages are available but not included in this "
        "prompt for brevity.]\n\n"
        "Please provide a summary of our conversation so far."
    )
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": summary.text},
    ]


def _emit_compaction_event(task_id: str) -> None:
    """
    Emit a ``response.compaction.in_progress`` SSE event.

    Called immediately before the Layer 2 LLM call so clients can
    display a progress indicator.

    :param task_id: The task identifier for routing the SSE event.
    """
    # Imported locally to avoid circular imports at module level.
    from agent_plane.runtime.durability import write_stream
    from agent_plane.runtime.live_stream import publish as _live_publish

    event: dict[str, str] = {"type": "response.compaction.in_progress"}
    write_stream("output", event)
    _live_publish(task_id, event)
