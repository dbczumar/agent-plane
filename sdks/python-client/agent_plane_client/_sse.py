"""SSE frame parser — converts raw byte chunks into typed events.

Handles the ``event:`` / ``data:`` / ``[DONE]`` framing from the
server's ``text/event-stream`` responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ._events import (
    NATIVE_TOOL_TYPES,
    RESERVED_APPROVAL_TOOL_NAME,
    ApprovalRequest,
    CompactionInProgress,
    ErrorEvent,
    MessageDone,
    NativeToolCall,
    OutputFileDone,
    ReasoningDelta,
    ReasoningStarted,
    ReasoningSummaryDelta,
    ResponseCancelled,
    ResponseCompleted,
    ResponseCreated,
    ResponseFailed,
    ResponseIncomplete,
    ResponseInProgress,
    ResponseQueued,
    RetryEvent,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolResult,
)
from ._types import ErrorInfo, Response

_log = logging.getLogger("agent_plane_client.sse")


async def parse_sse_stream(
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[StreamEvent]:
    """Parse an SSE byte stream into typed events.

    :param byte_stream: Raw bytes from ``httpx.Response.aiter_bytes()``.
    :yields: Typed :class:`StreamEvent` instances.
    """
    buf = ""
    current_event: str | None = None

    async for chunk in byte_stream:
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")

            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: ") and current_event is not None:
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    current_event = None
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    _log.warning("Failed to parse SSE data: %s", data_str[:200])
                    current_event = None
                    continue
                event = _parse_event(current_event, data)
                if event is not None:
                    yield event
                current_event = None
            elif line == "":
                current_event = None


def _normalize_event_type(event_type: str) -> str:
    """Normalize event type to handle server enum rendering.

    The server builds terminal events as ``f"response.{task.status}"``
    where ``task.status`` may be a Python enum (rendering as
    ``response.TaskStatus.COMPLETED``) instead of the expected
    ``response.completed``. Normalize by extracting and lowercasing
    the enum value.
    """
    if ".TaskStatus." in event_type:
        # "response.TaskStatus.COMPLETED" → "response.completed"
        parts = event_type.split(".")
        status = parts[-1].lower()
        return f"response.{status}"
    return event_type


def _parse_event(event_type: str, data: dict[str, Any]) -> StreamEvent | None:
    """Convert a raw SSE event type + JSON data into a typed event."""
    event_type = _normalize_event_type(event_type)

    # Response lifecycle
    if event_type == "response.created":
        return ResponseCreated(response=_parse_response(data))
    if event_type == "response.queued":
        return ResponseQueued(response=_parse_response(data))
    if event_type == "response.in_progress":
        return ResponseInProgress(response=_parse_response(data))
    if event_type == "response.completed":
        return ResponseCompleted(response=_parse_response(data))
    if event_type == "response.failed":
        return ResponseFailed(response=_parse_response(data))
    if event_type == "response.incomplete":
        resp = _parse_response(data)
        reason = ""
        if resp.incomplete_details is not None:
            reason = resp.incomplete_details.reason
        return ResponseIncomplete(response=resp, reason=reason)
    if event_type == "response.cancelled":
        return ResponseCancelled(response=_parse_response(data))

    # Text streaming
    if event_type == "response.output_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            return TextDelta(delta=delta)
        return None

    # Reasoning
    if event_type == "response.reasoning.started":
        return ReasoningStarted()
    if event_type == "response.reasoning_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            return ReasoningDelta(delta=delta)
        return None
    if event_type == "response.reasoning_summary_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            return ReasoningSummaryDelta(delta=delta)
        return None

    # Output items
    if event_type == "response.output_item.done":
        return _parse_output_item(data)

    # File output
    if event_type == "response.output_file.done":
        return OutputFileDone(
            file_id=str(data.get("file_id", "")),
            filename=str(data["filename"]) if data.get("filename") is not None else None,
            content_type=str(data["content_type"])
            if data.get("content_type") is not None
            else None,
        )

    # Retry
    if event_type == "response.retry":
        return RetryEvent(
            source=str(data.get("source", "")),
            tool_name=str(data["tool_name"]) if data.get("tool_name") is not None else None,
            attempt=int(data.get("attempt", 0)),
            max_attempts=int(data.get("max_attempts", 0)),
            delay_seconds=float(data.get("delay_seconds", 0.0)),
            error=_parse_error_info(data.get("error", {})),
        )

    # Error
    if event_type == "response.error":
        return ErrorEvent(
            source=str(data.get("source", "")),
            tool_name=str(data["tool_name"]) if data.get("tool_name") is not None else None,
            error=_parse_error_info(data.get("error", {})),
        )

    # Compaction
    if event_type == "response.compaction.in_progress":
        return CompactionInProgress()

    # Unknown event — skip gracefully for forward-compatibility
    _log.debug("Skipping unknown SSE event type: %s", event_type)
    return None


def _parse_output_item(data: dict[str, Any]) -> StreamEvent | None:
    """Parse a ``response.output_item.done`` event into a typed event."""
    item = data.get("item")
    if not isinstance(item, dict):
        return None

    item_type = item.get("type", "")

    if item_type == "function_call":
        args_str = str(item.get("arguments", "{}"))
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {}
        name = str(item.get("name", ""))
        call_id = str(item.get("call_id", ""))
        # Reserved-name carve-out: policy ASKs arrive as a
        # synthetic ``function_call`` named ``request_approval``.
        # Surface them as a distinct event type so the stream
        # consumer never feeds them into a ToolHandler — the
        # verdict goes through ``submit_approval`` instead.
        if name == RESERVED_APPROVAL_TOOL_NAME:
            return ApprovalRequest(
                call_id=call_id,
                reason=str(arguments.get("reason") or ""),
                policy_name=str(arguments.get("policy_name") or ""),
                phase=str(arguments.get("phase") or ""),
                content_preview=str(arguments.get("content_preview") or ""),
            )
        return ToolCall(
            name=name,
            arguments=arguments,
            call_id=call_id,
            status=str(item.get("status", "")),
            agent_name=str(item.get("model", "")),
        )

    if item_type == "function_call_output":
        return ToolResult(
            call_id=str(item.get("call_id", "")),
            output=str(item.get("output", "")),
        )

    if item_type == "message":
        content = item.get("content", [])
        return MessageDone(
            content=content if isinstance(content, list) else [],
        )

    if item_type in NATIVE_TOOL_TYPES:
        return NativeToolCall(
            tool_type=item_type,
            data=item,
        )

    # Compaction items, reasoning items, etc. — skip
    _log.debug("Skipping output item type: %s", item_type)
    return None


def _parse_response(data: dict[str, Any]) -> Response:
    """Extract the Response object from an SSE event payload."""
    resp_data = data.get("response")
    if isinstance(resp_data, dict):
        return Response.from_dict(resp_data)
    # Some events put fields at the top level
    return Response.from_dict(data)


def _parse_error_info(raw: Any) -> ErrorInfo:
    """Parse an ErrorInfo from a nested dict."""
    if isinstance(raw, dict):
        return ErrorInfo(
            code=str(raw.get("code", "")),
            message=str(raw.get("message", "")),
        )
    return ErrorInfo(code="", message=str(raw))
