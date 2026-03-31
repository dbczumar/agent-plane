"""Translate agent-plane SSE events to ACP session/update dicts.

Agent-plane emits OpenResponses-style SSE events. Toad expects ACP
``session/update`` notifications with specific ``sessionUpdate`` types.
This module maps between the two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Handler function signature: (translator, data) -> list of ACP updates
_HandlerFn = Callable[["EventTranslator", dict[str, object]], list[dict[str, object]]]


@dataclass
class ToolCallAccumulator:
    """Accumulates streamed function call argument deltas.

    :param call_id: The ``call_id`` from the function_call item.
    :param name: The tool/function name.
    :param arguments: Accumulated argument string so far, or
        ``None`` if no deltas have arrived yet.
    """

    call_id: str
    name: str
    arguments: str | None = None


@dataclass
class EventTranslator:
    """Stateful translator from agent-plane SSE to ACP updates.

    Accumulates tool call argument deltas and tracks the last
    response ID and conversation ID for conversation continuity.

    :param last_response_id: The most recent ``response.id`` seen,
        used for ``previous_response_id`` in follow-up requests.
    :param last_conversation_id: The conversation ID from the most
        recent ``response.completed`` event, e.g. ``"conv_abc123"``.
    :param stop_reason: Terminal status for the current prompt,
        e.g. ``"end_turn"``, ``"max_tokens"``, ``"cancelled"``.
        Reset to ``None`` before each prompt via
        :meth:`reset_for_prompt`.
    """

    last_response_id: str | None = None
    last_conversation_id: str | None = None
    stop_reason: str | None = None
    _pending_tool_calls: dict[str, ToolCallAccumulator] = field(default_factory=dict)
    _seen_function_calls: dict[str, dict[str, str]] = field(default_factory=dict)
    _seen_function_outputs: set[str] = field(default_factory=set)

    def reset_for_prompt(self) -> None:
        """Clear per-prompt state before a new prompt starts.

        Resets ``stop_reason`` and tool call tracking so the next
        prompt starts fresh. Does not clear ``last_response_id``
        or ``last_conversation_id`` — those persist across prompts.
        """
        self.stop_reason = None
        self._seen_function_calls.clear()
        self._seen_function_outputs.clear()

    @property
    def pending_client_tool_calls(
        self,
    ) -> list[dict[str, str]]:
        """Function calls that had no matching output in the stream.

        After a response completes, any call_id in
        ``_seen_function_calls`` but not in ``_seen_function_outputs``
        is a client-side tool call that the adapter must execute.

        :returns: List of dicts with ``call_id``, ``name``, and
            ``arguments`` for each pending call.
        """
        pending: list[dict[str, str]] = []
        for call_id, info in self._seen_function_calls.items():
            if call_id not in self._seen_function_outputs:
                pending.append(info)
        return pending

    def translate(self, event_type: str, data: dict[str, object]) -> list[dict[str, object]]:
        """Translate one SSE event into zero or more ACP updates.

        :param event_type: The SSE ``event:`` value, e.g.
            ``"response.output_text.delta"``.
        :param data: The parsed JSON ``data:`` payload.
        :returns: A list of ACP ``sessionUpdate`` dicts to send as
            ``session/update`` notifications. May be empty if the
            event has no ACP equivalent.
        """
        handler = _EVENT_MAP.get(event_type)
        if handler is not None:
            return handler(self, data)
        return []


def _on_text_delta(
    _translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.output_text.delta``.

    :param _translator: The translator instance (unused for this
        stateless event).
    :param data: SSE payload containing ``"delta"`` text.
    :returns: Single ``agent_message_chunk`` update.
    """
    # delta is required per OpenResponses SSE spec
    delta = str(data["delta"])
    return [
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": delta},
        }
    ]


def _on_reasoning_delta(
    _translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.reasoning_summary.delta``.

    Maps reasoning/summary deltas to Toad's
    ``agent_thought_chunk`` type.

    :param _translator: The translator instance (unused).
    :param data: SSE payload containing ``"delta"`` text.
    :returns: Single ``agent_thought_chunk`` update.
    """
    # delta is required per OpenResponses SSE spec
    delta = str(data["delta"])
    return [
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": delta},
        }
    ]


def _on_item_done(translator: EventTranslator, data: dict[str, object]) -> list[dict[str, object]]:
    """Handle ``response.output_item.done``.

    Emits ``tool_call`` updates for function calls and
    ``tool_call_update`` for function call outputs.

    :param translator: The translator instance (tracks pending
        tool calls).
    :param data: SSE payload containing the completed ``"item"``
        dict.
    :returns: Zero or more ACP updates depending on item type.
    """
    item = data.get("item")
    if not isinstance(item, dict):
        return []

    item_type = item.get("type")
    updates: list[dict[str, object]] = []

    if item_type == "function_call":
        # call_id, name, arguments are required per OpenResponses spec
        call_id = str(item["call_id"])
        name = str(item["name"])
        arguments = str(item["arguments"])
        # Track for pending client-side tool detection
        translator._seen_function_calls[call_id] = {
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
        updates.append(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": name,
                "kind": "other",
                "status": "in_progress",
            }
        )
    elif item_type == "function_call_output":
        output_call_id = item.get("call_id")
        if output_call_id is not None:
            translator._seen_function_outputs.add(str(output_call_id))
        updates.extend(_on_function_call_output_item(item))
    elif item_type not in ("message", None):
        # Native tool items (web_search_call, file_search_call, etc.)
        updates.extend(_on_native_tool_item(item))

    return updates


def _on_function_call_output_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Emit a ``tool_call_update`` for a function_call_output item.

    :param item: The completed item dict with ``call_id`` and
        ``output`` fields.
    :returns: Single ``tool_call_update`` with ``"completed"``
        status.
    """
    # call_id, output are required per OpenResponses spec
    call_id = str(item["call_id"])
    output = str(item["output"])
    return [
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "completed",
            "content": [
                {
                    "type": "content",
                    "content": {"type": "text", "text": output},
                }
            ],
        }
    ]


def _on_response_completed(
    translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.completed``.

    Captures the response ID and conversation ID for continuity,
    and sets ``stop_reason`` to ``"end_turn"``.

    :param translator: The translator instance (stores
        ``last_response_id`` and ``last_conversation_id``).
    :param data: SSE payload containing the ``"response"`` dict.
    :returns: Empty list (no ACP update emitted).
    """
    translator.stop_reason = "end_turn"
    response = data.get("response")
    if isinstance(response, dict):
        resp_id = response.get("id")
        if resp_id is not None:
            translator.last_response_id = str(resp_id)
        _extract_conversation_id(translator, response)
    return []


def _extract_conversation_id(
    translator: EventTranslator,
    response: dict[str, object],
) -> None:
    """Extract conversation ID from a response dict.

    Looks for ``response["conversation"]["id"]`` and stores it
    on the translator.

    :param translator: The translator to update.
    :param response: The ``"response"`` dict from an SSE payload.
    """
    conversation = response.get("conversation")
    if isinstance(conversation, dict):
        conv_id = conversation.get("id")
        if conv_id is not None:
            translator.last_conversation_id = str(conv_id)


# Maps agent-plane incomplete_details.reason to ACP stop reasons.
_INCOMPLETE_REASON_MAP: dict[str, str] = {
    "max_iterations": "max_turn_requests",
    "execution_timeout": "max_turn_requests",
    "max_output_tokens": "max_tokens",
}


def _on_response_failed(
    translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.failed``.

    Sets ``stop_reason`` to ``"end_turn"`` and captures IDs.

    :param translator: The translator instance.
    :param data: SSE payload containing the ``"response"`` dict.
    :returns: Empty list (no ACP update emitted).
    """
    translator.stop_reason = "end_turn"
    response = data.get("response")
    if isinstance(response, dict):
        resp_id = response.get("id")
        if resp_id is not None:
            translator.last_response_id = str(resp_id)
        _extract_conversation_id(translator, response)
    return []


def _on_response_incomplete(
    translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.incomplete``.

    Maps ``incomplete_details.reason`` to an ACP stop reason via
    :data:`_INCOMPLETE_REASON_MAP`.

    :param translator: The translator instance.
    :param data: SSE payload containing the ``"response"`` dict
        with ``incomplete_details``.
    :returns: Empty list (no ACP update emitted).
    """
    response = data.get("response")
    reason = _extract_incomplete_reason(response)
    if reason is not None:
        translator.stop_reason = _INCOMPLETE_REASON_MAP.get(reason, "end_turn")
    else:
        translator.stop_reason = "end_turn"
    if isinstance(response, dict):
        resp_id = response.get("id")
        if resp_id is not None:
            translator.last_response_id = str(resp_id)
        _extract_conversation_id(translator, response)
    return []


def _extract_incomplete_reason(
    response: object,
) -> str | None:
    """Extract the reason string from incomplete_details.

    :param response: The ``"response"`` value from the SSE payload.
    :returns: The reason string, or ``None`` if not found.
    """
    if not isinstance(response, dict):
        return None
    details = response.get("incomplete_details")
    if isinstance(details, dict):
        reason = details.get("reason")
        if reason is not None:
            return str(reason)
    return None


def _on_response_cancelled(
    translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.cancelled``.

    :param translator: The translator instance.
    :param data: SSE payload (response dict may be present).
    :returns: Empty list (no ACP update emitted).
    """
    translator.stop_reason = "cancelled"
    response = data.get("response")
    if isinstance(response, dict):
        resp_id = response.get("id")
        if resp_id is not None:
            translator.last_response_id = str(resp_id)
        _extract_conversation_id(translator, response)
    return []


def _on_response_error(
    _translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.error``.

    Emits an ``agent_message_chunk`` with the error text so the
    user sees it in the chat.

    :param _translator: The translator instance (unused).
    :param data: SSE payload containing ``"message"`` or
        ``"error"`` text.
    :returns: Single ``agent_message_chunk`` with error text.
    """
    # SSE error events may use "message" or "error" key; show
    # empty bracket prefix if neither is present so the user
    # still sees something happened.
    message = str(data.get("message") or data.get("error") or "Unknown error")
    return [
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": f"[Error] {message}"},
        }
    ]


def _on_response_retry(
    _translator: EventTranslator, data: dict[str, object]
) -> list[dict[str, object]]:
    """Handle ``response.retry``.

    Emits an ``agent_thought_chunk`` so the user sees retry info
    as internal reasoning.

    :param _translator: The translator instance (unused).
    :param data: SSE payload containing ``"message"`` or retry
        details.
    :returns: Single ``agent_thought_chunk`` with retry info.
    """
    # "message" is optional on retry events; "Retrying..." is the
    # ACP-visible placeholder when the server omits details.
    message = str(data.get("message") or "Retrying...")
    return [
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": f"[Retry] {message}"},
        }
    ]


def _on_native_tool_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Emit a completed ``tool_call`` for a native tool item.

    Native tools (e.g. ``web_search_call``, ``file_search_call``)
    are not function_call/function_call_output — they are
    platform-provided tool invocations that complete inline.

    :param item: The completed item dict from the SSE payload.
    :returns: Single ``tool_call`` update with ``"completed"``
        status.
    """
    item_type = str(item.get("type", ""))
    # Use item id as toolCallId; fall back to type for uniqueness
    call_id = str(item.get("id", item_type))
    return [
        {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": item_type,
            "kind": "other",
            "status": "completed",
        }
    ]


# Map of agent-plane SSE event type -> handler function.
# Events not in this map are silently ignored.
_EVENT_MAP: dict[str, _HandlerFn] = {
    "response.output_text.delta": _on_text_delta,
    "response.reasoning.delta": _on_reasoning_delta,
    "response.reasoning_summary.delta": _on_reasoning_delta,
    "response.reasoning_summary_text.delta": _on_reasoning_delta,
    "response.output_item.done": _on_item_done,
    "response.completed": _on_response_completed,
    "response.failed": _on_response_failed,
    "response.incomplete": _on_response_incomplete,
    "response.cancelled": _on_response_cancelled,
    "response.error": _on_response_error,
    "response.retry": _on_response_retry,
}
