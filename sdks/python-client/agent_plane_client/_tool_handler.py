"""Tool handler, hooks, and context types for client-side tool execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ._types import ErrorInfo, Response

# ── Tool handler ─────────────────────────────────────────


@dataclass
class ToolCallInfo:
    """Context passed to ``ToolHandler.execute``."""

    name: str
    arguments: dict[str, object]
    call_id: str
    agent_name: str  # "coder" or "coder.researcher"
    response_id: str
    iteration: int  # Tool loop iteration (0-based)


@dataclass
class ToolHandler:
    """Client-side tool execution configuration.

    Passed to ``stream()`` to enable automatic tool execution.
    The client runs the full loop: stream -> detect tool calls ->
    call ``execute()`` -> send results -> continue streaming.
    """

    schemas: list[dict[str, object]]
    execute: Callable[[ToolCallInfo], Awaitable[str] | str]


# ── Hook context types ───────────────────────────────────


@dataclass
class ToolCallStartCtx:
    """Context for ``on_tool_call_start``."""

    name: str
    arguments: dict[str, object]
    call_id: str
    agent_name: str
    executed_by: str  # "client" or "server"


@dataclass
class ToolCallEndCtx:
    """Context for ``on_tool_call_end``."""

    name: str
    call_id: str
    agent_name: str
    output: str


@dataclass
class NativeToolCallCtx:
    """Context for ``on_native_tool_call``."""

    tool_type: str
    data: dict[str, object]


@dataclass
class ToolResultInfo:
    """A single tool result within ``ToolResultsReadyCtx``."""

    call_id: str
    name: str
    output: str
    agent_name: str


@dataclass
class ToolResultsReadyCtx:
    """Context for ``on_tool_results_ready``."""

    results: list[ToolResultInfo]
    iteration: int


@dataclass
class ReasoningStartCtx:
    """Context for ``on_reasoning_start``."""

    pass


@dataclass
class ReasoningEndCtx:
    """Context for ``on_reasoning_end``."""

    reasoning_text: str
    summary_text: str


@dataclass
class CompactionStartCtx:
    """Context for ``on_compaction_start``."""

    pass


@dataclass
class CompactionEndCtx:
    """Context for ``on_compaction_end``."""

    item: dict[str, object]


@dataclass
class MessageStartCtx:
    """Context for ``on_message_start``."""

    response_id: str


@dataclass
class MessageEndCtx:
    """Context for ``on_message_end``."""

    content: list[dict[str, object]]


@dataclass
class FileOutputCtx:
    """Context for ``on_file_output``."""

    file_id: str
    filename: str | None
    content_type: str | None


@dataclass
class RetryCtx:
    """Context for ``on_retry``."""

    source: str
    tool_name: str | None
    attempt: int
    max_attempts: int
    delay_seconds: float
    error: ErrorInfo


@dataclass
class ServerErrorCtx:
    """Context for ``on_server_error``."""

    source: str
    tool_name: str | None
    error: ErrorInfo


@dataclass
class TransportErrorCtx:
    """Context for ``on_transport_error``."""

    error: Exception


@dataclass
class SubAgentInfo:
    """Info about a single spawned sub-agent."""

    response_id: str
    agent_name: str


@dataclass
class SubAgentSpawnedCtx:
    """Context for ``on_sub_agent_spawned``."""

    parent_response_id: str
    sub_agents: list[SubAgentInfo] = field(default_factory=list)


@dataclass
class SubAgentCompletedCtx:
    """Context for ``on_sub_agent_completed``."""

    response_id: str
    agent_name: str
    status: str
    output_summary: str | None


@dataclass
class ResponseStartCtx:
    """Context for ``on_response_start``."""

    response: Response


@dataclass
class ResponseEndCtx:
    """Context for ``on_response_end``."""

    response: Response
    status: str  # "completed", "failed", "incomplete", "cancelled"


# ── Stream hooks ─────────────────────────────────────────

# Hook type alias: sync or async callable, or None.
_Hook = Callable[..., Awaitable[None] | None] | None


@dataclass
class StreamHooks:
    """Lifecycle hooks for stream events.

    Every class of event has a start/end hook pair where
    applicable. Hooks can be sync or async. All are optional.
    """

    # Tool calls (server-side and client-side)
    on_tool_call_start: Callable[[ToolCallStartCtx], Awaitable[None] | None] | None = None
    on_tool_call_end: Callable[[ToolCallEndCtx], Awaitable[None] | None] | None = None

    # Native tool calls (provider-executed)
    on_native_tool_call: Callable[[NativeToolCallCtx], Awaitable[None] | None] | None = None

    # Tool loop iteration
    on_tool_results_ready: Callable[[ToolResultsReadyCtx], Awaitable[None] | None] | None = None

    # Reasoning
    on_reasoning_start: Callable[[ReasoningStartCtx], Awaitable[None] | None] | None = None
    on_reasoning_end: Callable[[ReasoningEndCtx], Awaitable[None] | None] | None = None

    # Compaction
    on_compaction_start: Callable[[CompactionStartCtx], Awaitable[None] | None] | None = None
    on_compaction_end: Callable[[CompactionEndCtx], Awaitable[None] | None] | None = None

    # Message (assistant response)
    on_message_start: Callable[[MessageStartCtx], Awaitable[None] | None] | None = None
    on_message_end: Callable[[MessageEndCtx], Awaitable[None] | None] | None = None

    # File output
    on_file_output: Callable[[FileOutputCtx], Awaitable[None] | None] | None = None

    # Retry and error
    on_retry: Callable[[RetryCtx], Awaitable[None] | None] | None = None
    on_server_error: Callable[[ServerErrorCtx], Awaitable[None] | None] | None = None
    on_transport_error: Callable[[TransportErrorCtx], Awaitable[bool] | bool] | None = None

    # Sub-agent lifecycle
    on_sub_agent_spawned: Callable[[SubAgentSpawnedCtx], Awaitable[None] | None] | None = None
    on_sub_agent_completed: Callable[[SubAgentCompletedCtx], Awaitable[None] | None] | None = None

    # Response lifecycle
    on_response_start: Callable[[ResponseStartCtx], Awaitable[None] | None] | None = None
    on_response_end: Callable[[ResponseEndCtx], Awaitable[None] | None] | None = None
