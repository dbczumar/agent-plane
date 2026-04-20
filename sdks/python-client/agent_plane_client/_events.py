"""Typed event dataclasses for SSE stream events.

The client parses raw SSE frames into these types. Consumers
iterate over them via ``async for event in stream``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._types import ErrorInfo, Response

# ── Native tool type constants ───────────────────────────

NATIVE_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "computer_call",
        "image_generation_call",
        "mcp_call",
        "mcp_list_tools",
    }
)

# Reserved tool name for policy approval requests. The server
# emits a synthetic ``function_call`` item with this name when
# a policy returns ASK; the client must route it to an
# approval handler rather than a normal ToolHandler (it is not
# a real tool). See POLICIES.md §7 / §15.10.
RESERVED_APPROVAL_TOOL_NAME = "request_approval"


# ── Response lifecycle events ────────────────────────────


@dataclass
class ResponseCreated:
    """``response.created`` — always first (sequence 0)."""

    response: Response


@dataclass
class ResponseQueued:
    """``response.queued`` — only when ``background=True``."""

    response: Response


@dataclass
class ResponseInProgress:
    """``response.in_progress`` — execution started."""

    response: Response


@dataclass
class ResponseCompleted:
    """``response.completed`` — agent finished successfully."""

    response: Response


@dataclass
class ResponseFailed:
    """``response.failed`` — unrecoverable error."""

    response: Response


@dataclass
class ResponseIncomplete:
    """``response.incomplete`` — stopped early."""

    response: Response
    reason: str  # "max_iterations", "execution_timeout", etc.


@dataclass
class ResponseCancelled:
    """``response.cancelled`` — cancelled via POST /cancel."""

    response: Response


# ── Text streaming ───────────────────────────────────────


@dataclass
class TextDelta:
    """``response.output_text.delta`` — incremental text token."""

    delta: str


# ── Reasoning ────────────────────────────────────────────


@dataclass
class ReasoningStarted:
    """``response.reasoning.started`` — reasoning block opened."""

    pass


@dataclass
class ReasoningDelta:
    """``response.reasoning_text.delta`` — reasoning token."""

    delta: str


@dataclass
class ReasoningSummaryDelta:
    """``response.reasoning_summary_text.delta`` — summary token."""

    delta: str


# ── Parsed output items ─────────────────────────────────


@dataclass
class ToolCall:
    """A tool call from ``output_item.done`` (type ``function_call``)."""

    name: str
    arguments: dict[str, object]
    call_id: str
    status: str  # "completed", "action_required", "incomplete"
    agent_name: str  # "coder" or "coder.researcher"


@dataclass
class ToolResult:
    """A tool result from ``output_item.done`` (type ``function_call_output``)."""

    call_id: str
    output: str


@dataclass
class ApprovalRequest:
    """
    A policy ASK surfaced as a synthetic ``request_approval`` call.

    Parsed out of the ``function_call`` item whose ``name`` is
    :data:`RESERVED_APPROVAL_TOOL_NAME`. Consumers respond by
    calling ``client.responses.submit_approval(response_id,
    call_id, approved=...)`` — the server routes the verdict
    through the same PATCH endpoint client-side tool results use.

    :param call_id: Server-assigned call_id. Must be echoed back
        verbatim when submitting the verdict.
    :param reason: Combined reason string from the deciding ASK
        policies (``"; "``-joined per POLICIES.md §4).
    :param policy_name: Name of the deciding (first-in-YAML-
        order) ASKing policy, e.g. ``"approve_web_search"``.
    :param phase: Which enforcement point produced the ASK —
        one of ``"input"``, ``"tool_call"``, ``"tool_result"``,
        ``"output"``.
    :param content_preview: Truncated snapshot of the gated
        content. Safe to display verbatim in an approval UI.
    """

    call_id: str
    reason: str
    policy_name: str
    phase: str
    content_preview: str


@dataclass
class NativeToolCall:
    """A provider-native tool output (web_search, mcp, etc.)."""

    tool_type: str  # e.g. "web_search_call"
    data: dict[str, object]


@dataclass
class MessageDone:
    """The final assistant message from ``output_item.done`` (type ``message``)."""

    content: list[dict[str, object]] = field(default_factory=list)


# ── File output ──────────────────────────────────────────


@dataclass
class OutputFileDone:
    """``response.output_file.done`` — file artifact produced."""

    file_id: str
    filename: str | None = None
    content_type: str | None = None


# ── Error and retry ──────────────────────────────────────


@dataclass
class RetryEvent:
    """``response.retry`` — a retryable failure, will retry."""

    source: str  # "llm" or "tool"
    tool_name: str | None
    attempt: int
    max_attempts: int
    delay_seconds: float
    error: ErrorInfo


@dataclass
class ErrorEvent:
    """``response.error`` — an error during execution."""

    source: str  # "llm" or "tool"
    tool_name: str | None
    error: ErrorInfo


# ── Compaction ───────────────────────────────────────────


@dataclass
class CompactionInProgress:
    """``response.compaction.in_progress`` — server is compacting."""

    pass


# ── Async client-tool cancel (Phase 5) ───────────────────


@dataclass
class ClientTaskCancel:
    """
    ``response.client_task.cancel`` — Phase 5.

    Emitted by the server when an async client tool task
    (``kind="client_tool"``) was cancelled mid-flight, either
    via direct ``cancel_task`` or via parent-cancel
    propagation. The client should cancel the matching
    background asyncio task and (optionally) PATCH back
    ``async_tool_results`` with ``status="cancelled"`` so the
    parent's drain sees the terminal state.

    :param task_id: The server-issued client-tool task id from
        the original ``function_call_output`` handle, e.g.
        ``"resp_async_xyz"``.
    """

    task_id: str


# ── Union type for all events ────────────────────────────

StreamEvent = (
    ResponseCreated
    | ResponseQueued
    | ResponseInProgress
    | ResponseCompleted
    | ResponseFailed
    | ResponseIncomplete
    | ResponseCancelled
    | TextDelta
    | ReasoningStarted
    | ReasoningDelta
    | ReasoningSummaryDelta
    | ToolCall
    | ToolResult
    | NativeToolCall
    | MessageDone
    | OutputFileDone
    | RetryEvent
    | ErrorEvent
    | CompactionInProgress
    | ClientTaskCancel
)
