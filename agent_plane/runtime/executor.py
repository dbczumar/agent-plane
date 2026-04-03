"""Backwards-compatibility shim — imports moved to executors package.

All symbols re-exported so existing ``from agent_plane.runtime.executor
import ...`` statements continue to work.
"""

from agent_plane.runtime.executors import (
    ContextWindowExceeded,
    DefaultExecutor,
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    NativeToolOutput,
    ReasoningChunk,
    RemoteExecutor,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
    _build_responses_args,
    _consume_stream,
    _extract_native_tool_items,
    _extract_text,
    _extract_tool_calls,
    _open_stream_with_retry,
    _ResponsesCallArgs,
    _run_streaming_turn,
    _yield_final_events,
    dict_to_event,
    event_to_dict,
)

# _create_stream is monkeypatched by tests — must be importable here.
from agent_plane.runtime.executors.default import (
    _create_stream,
    _get_llm_client,
    _get_model_context_window,
)

__all__ = [
    "ContextWindowExceeded",
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
    "_ResponsesCallArgs",
    "_build_responses_args",
    "_consume_stream",
    "_create_stream",
    "_extract_native_tool_items",
    "_extract_text",
    "_extract_tool_calls",
    "_get_llm_client",
    "_get_model_context_window",
    "_open_stream_with_retry",
    "_run_streaming_turn",
    "_yield_final_events",
    "dict_to_event",
    "event_to_dict",
]
