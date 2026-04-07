"""Executor package — public API for all executor types and events.

Import from this package rather than individual submodules::

    from agent_plane.runtime.executors import (
        Executor, DefaultExecutor, RemoteExecutor, ClaudeAgentsExecutor,
        TextChunk, TurnComplete, ExecutorEvent, event_to_dict, ...
    )
"""

from agent_plane.runtime.executors.base import (
    ContextWindowExceeded,
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    NativeToolOutput,
    ReasoningChunk,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
    dict_to_event,
    event_to_dict,
)
from agent_plane.runtime.executors.claude import ClaudeAgentsExecutor
from agent_plane.runtime.executors.default import (
    DefaultExecutor,
    _build_responses_args,
    _consume_stream,
    _extract_native_tool_items,
    _extract_text,
    _extract_tool_calls,
    _open_stream_with_retry,
    _ResponsesCallArgs,
    _run_streaming_turn,
    _yield_final_events,
)
from agent_plane.runtime.executors.remote import RemoteExecutor

__all__ = [
    "ContextWindowExceeded",
    "ClaudeAgentsExecutor",
    "DefaultExecutor",
    "Executor",
    "ExecutorContext",
    "ExecutorError",
    "ExecutorEvent",
    "NativeToolOutput",
    "ReasoningChunk",
    "RemoteExecutor",
    "TextChunk",
    "ToolCallObserved",
    "ToolCallRequested",
    "ToolResult",
    "TurnComplete",
    "dict_to_event",
    "event_to_dict",
    "_ResponsesCallArgs",
    "_build_responses_args",
    "_consume_stream",
    "_extract_native_tool_items",
    "_extract_text",
    "_extract_tool_calls",
    "_open_stream_with_retry",
    "_run_streaming_turn",
    "_yield_final_events",
]
