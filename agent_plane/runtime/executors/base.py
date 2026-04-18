"""Executor ABC, event types, and serialization.

The Executor decouples the "call LLM + interpret response" concern
from the workflow's "persist, stream, compact, steer" concerns.

Event types are dataclasses yielded by ``run_turn()``. The workflow
consumes them uniformly — no branching on executor type.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typing_extensions import Self

from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig

# ── Event types ────────────────────────────────────────────


@dataclass
class TextChunk:
    """
    A streamed text token from the model.

    :param text: The incremental text fragment, e.g. ``"Hello"``.
    """

    text: str


@dataclass
class ReasoningChunk:
    """
    A streamed reasoning token from the model.

    Gated by ``reasoning_effort`` in the LLM config.

    :param delta: The incremental reasoning text, e.g. ``"Let me think"``.
        Empty string for ``"reasoning_started"`` events.
    :param event_type: One of ``"reasoning_text"``,
        ``"reasoning_summary"``, or ``"reasoning_started"``.
    """

    delta: str
    event_type: str


@dataclass
class NativeToolOutput:
    """
    A provider-native tool output (e.g. ``web_search_call`` result).

    Not dispatched locally — flows through to the client as-is.

    :param item: The raw output dict from the provider, e.g.
        ``{"type": "web_search_call", "id": "ws_1", ...}``.
    """

    item: dict[str, Any]


@dataclass
class ToolCallRequested:
    """
    The executor wants the workflow to execute a tool.

    The workflow executes the tool via ``_call_tool()`` (``@step``),
    appends a tool_result message, and calls ``run_turn()`` again.

    :param call_id: Identifier for this call, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"web_search"``.
    :param arguments: Parsed tool arguments dict.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallObserved:
    """
    The executor ran a tool internally. Workflow just persists and streams.

    Emitted by internal executors (Claude SDK) after each tool the
    harness executed autonomously.

    :param call_id: Identifier, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"Bash"``.
    :param arguments: Parsed tool arguments dict.
    :param result: The tool's output string.
    :param status: ``"success"`` | ``"error"`` | ``"blocked"``.
    :param duration_ms: Wall-clock time the tool took, e.g. ``342.1``.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    status: str
    duration_ms: float


@dataclass
class TurnComplete:
    """
    The executor has finished its turn.

    :param text: The assistant's text response, or ``None`` if the turn
        ended with tool calls only.
    :param usage: Token usage for this turn, e.g.
        ``{"input_tokens": 1523, "output_tokens": 847}``. ``None``
        when the executor does not report usage. Known keys:
        ``"input_tokens"``, ``"output_tokens"``,
        ``"cache_read_input_tokens"``,
        ``"cache_creation_input_tokens"``.
    :param response_model: The actual model identifier reported by
        the provider (may differ from the requested model), or
        ``None`` if unavailable.
    :param response_id: The provider-assigned response/completion
        identifier, or ``None`` if unavailable.
    :param finish_reasons: Finish reasons from the response, e.g.
        ``["stop"]`` or ``["tool_calls"]``. ``None`` when the
        executor cannot derive a reason (e.g. empty response
        output). Never an empty list — callers should treat
        absence as "unknown", not "zero reasons".
    """

    text: str | None
    usage: dict[str, Any] | None = None
    response_model: str | None = None
    response_id: str | None = None
    finish_reasons: list[str] | None = None


@dataclass
class ContextWindowExceeded:
    """
    The executor hit a context window overflow.

    The workflow compacts messages and retries ``run_turn()``.

    :param max_tokens: The model's context window size, e.g. ``128000``.
    :param actual_tokens: The prompt size that triggered overflow,
        e.g. ``131072``.
    """

    max_tokens: int
    actual_tokens: int


@dataclass
class TurnCancelled:
    """
    The executor's turn was cancelled before completing.

    Yielded when an ``asyncio.CancelledError`` interrupts ``run_turn()``
    mid-stream, or when the workflow detects cancellation between
    executor events. Carries partial output so the workflow can persist
    what was generated before the interruption.

    :param reason: Why the turn was cancelled, e.g. ``"user_cancelled"``.
    :param partial_text: Any text the model had streamed before the
        cancellation, or ``None`` if no text was emitted yet.
    """

    reason: str
    partial_text: str | None = None


@dataclass
class ExecutorError:
    """
    An unrecoverable executor failure. No retry.

    :param message: Human-readable error description.
    :param code: Machine-readable error code, e.g. ``"auth_failed"``.
    """

    message: str
    code: str | None = None


@dataclass
class ToolResult:
    """
    Result of a tool call executed via ``call_tool``.

    :param content: The tool's output string.
    :param status: ``"success"`` or ``"error"``.
    """

    content: str
    status: str


ExecutorEvent = (
    TextChunk
    | ReasoningChunk
    | NativeToolOutput
    | ToolCallRequested
    | ToolCallObserved
    | TurnComplete
    | TurnCancelled
    | ContextWindowExceeded
    | ExecutorError
)


# ── Event serialization ───────────────────────────────────


def event_to_dict(event: ExecutorEvent) -> dict[str, Any]:
    """
    Serialize an executor event to a JSON-safe dict.

    Used by ``_checkpointed_turn`` to cache the event list for
    DBOS replay. Each dict has a ``"type"`` key matching the
    event class name.

    :param event: The executor event to serialize.
    :returns: A JSON-serializable dict, e.g.
        ``{"type": "TextChunk", "text": "Hello"}``.
    """
    if isinstance(event, TextChunk):
        return {"type": "TextChunk", "text": event.text}
    if isinstance(event, ReasoningChunk):
        return {
            "type": "ReasoningChunk",
            "delta": event.delta,
            "event_type": event.event_type,
        }
    if isinstance(event, NativeToolOutput):
        return {"type": "NativeToolOutput", "item": event.item}
    if isinstance(event, ToolCallRequested):
        return {
            "type": "ToolCallRequested",
            "call_id": event.call_id,
            "name": event.name,
            "arguments": event.arguments,
        }
    if isinstance(event, ToolCallObserved):
        return {
            "type": "ToolCallObserved",
            "call_id": event.call_id,
            "name": event.name,
            "arguments": event.arguments,
            "result": event.result,
            "status": event.status,
            "duration_ms": event.duration_ms,
        }
    if isinstance(event, TurnComplete):
        return {
            "type": "TurnComplete",
            "text": event.text,
            # Usage/metadata fields are included unconditionally
            # (even when None). dict_to_event uses **fields so the
            # keyword args pass through cleanly; None matches the
            # dataclass defaults. Old cached events that predate
            # these fields still deserialize because dict_to_event
            # only passes keys that exist in the payload.
            "usage": event.usage,
            "response_model": event.response_model,
            "response_id": event.response_id,
            "finish_reasons": event.finish_reasons,
        }
    if isinstance(event, TurnCancelled):
        return {
            "type": "TurnCancelled",
            "reason": event.reason,
            "partial_text": event.partial_text,
        }
    if isinstance(event, ContextWindowExceeded):
        return {
            "type": "ContextWindowExceeded",
            "max_tokens": event.max_tokens,
            "actual_tokens": event.actual_tokens,
        }
    if isinstance(event, ExecutorError):
        return {
            "type": "ExecutorError",
            "message": event.message,
            "code": event.code,
        }
    raise ValueError(f"Unknown event type: {type(event)}")


_EVENT_CONSTRUCTORS: dict[str, type[ExecutorEvent]] = {
    "TextChunk": TextChunk,
    "ReasoningChunk": ReasoningChunk,
    "NativeToolOutput": NativeToolOutput,
    "ToolCallRequested": ToolCallRequested,
    "ToolCallObserved": ToolCallObserved,
    "TurnComplete": TurnComplete,
    "TurnCancelled": TurnCancelled,
    "ContextWindowExceeded": ContextWindowExceeded,
    "ExecutorError": ExecutorError,
}


def dict_to_event(data: dict[str, Any]) -> ExecutorEvent:
    """
    Deserialize a dict (from DBOS cache) back to an executor event.

    Inverse of :func:`event_to_dict`.

    :param data: A dict with a ``"type"`` key, e.g.
        ``{"type": "TextChunk", "text": "Hello"}``.
    :returns: The corresponding executor event instance.
    """
    event_type = data["type"]
    cls = _EVENT_CONSTRUCTORS.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type}")
    # All event dataclasses use keyword-only fields that match
    # the dict keys (minus "type").
    fields = {k: v for k, v in data.items() if k != "type"}
    return cls(**fields)


# ── ExecutorContext ─────────────────────────────────────────


@dataclass
class ExecutorContext:
    """
    Capabilities and identifiers agent-plane provides to executors.

    Constructed by the workflow once per task and passed to
    ``run_turn()`` and lifecycle hooks. Extensible — new capabilities
    are added as fields, no signature changes needed.

    :param task_id: Current task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: Current conversation identifier,
        e.g. ``"conv_abc123"``.
    :param storage_dir: Scoped persistent directory for this
        conversation. The workflow manages artifact store I/O.
    :param call_tool: Execute a tool by name. Async — uses
        ``asyncio.sleep`` for client-side tool polling so the
        event loop stays free. The workflow routes to server-side
        execution (ToolManager) if the tool is registered,
        otherwise tunnels to the client for execution.
    """

    task_id: str
    conversation_id: str
    storage_dir: Path
    call_tool: Callable[[ToolCallRequested], Awaitable[ToolResult]]


# ── Executor ABC ───────────────────────────────────────────


class Executor(abc.ABC):
    """
    Abstract base for agent executors.

    Subclasses wrap a specific LLM backend or agent harness. The
    workflow calls ``run_turn()`` and consumes the event stream
    uniformly — no branching on executor type.

    Construction is standardized via ``from_spec()``. Each subclass
    extracts what it needs from the AgentSpec.
    """

    @classmethod
    @abc.abstractmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Construct an executor from an agent spec.

        :param spec: The parsed AgentSpec with a non-None llm field.
        :returns: A configured executor instance.
        """
        ...

    @abc.abstractmethod
    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Run one executor turn and yield events.

        Async generator — callers consume with ``async for``.
        Yields events as they arrive (text deltas, tool calls,
        etc.) and terminates with ``TurnComplete`` or
        ``ExecutorError``.

        :param messages: Conversation history as Responses API
            input items.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: Assembled system instructions string.
        :param llm_config: LLM configuration (model, extra,
            connection, timeout, retry). May differ from the spec's
            config due to per-request overrides (e.g. reasoning
            effort).
        :param context: Capabilities and identifiers from
            agent-plane.
        """
        # Unreachable yield makes the type checker recognize this
        # as an async generator. Subclasses use ``yield`` directly.
        if False:  # pragma: no cover
            yield

    def on_task_start(self, context: ExecutorContext) -> None:
        """
        Called once at task start, after storage_dir has been restored.

        :param context: Capabilities and identifiers from agent-plane.
        """

    def on_task_end(self, context: ExecutorContext) -> None:
        """
        Called once at task end (in a finally block).

        :param context: Same context from on_task_start.
        """

    def max_context_tokens(self) -> int | None:
        """
        Context window limit, or None if managed internally.

        When an int: the workflow owns compaction and wraps
        ``run_turn()`` in a ``@step`` for DBOS durability.

        When None: the executor owns compaction. The workflow
        skips both compaction and the ``@step`` wrapper.

        :returns: Token limit (e.g. ``128000``) or None.
        """
        return None
