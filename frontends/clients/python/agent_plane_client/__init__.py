"""agent-plane Python client library.

Usage::

    from agent_plane_client import AgentPlaneClient, LocalServer

    async with AgentPlaneClient(base_url="http://localhost:8080") as client:
        async for event in client.responses.stream(model="archer", input="hello"):
            print(event)
"""

from ._client import AgentPlaneClient
from ._errors import (
    AgentNotFoundError,
    AgentPlaneError,
    BundleInvalidError,
    ConflictError,
    ConversationNotFoundError,
    InvalidInputError,
    ResponseNotFoundError,
    ServerError,
    ToolCallDenied,
)
from ._events import (
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
from ._server import LocalServer
from ._session import Session
from ._tool_handler import (
    CompactionEndCtx,
    CompactionStartCtx,
    FileOutputCtx,
    MessageEndCtx,
    MessageStartCtx,
    NativeToolCallCtx,
    ReasoningEndCtx,
    ReasoningStartCtx,
    ResponseEndCtx,
    ResponseStartCtx,
    RetryCtx,
    ServerErrorCtx,
    StreamHooks,
    SubAgentCompletedCtx,
    SubAgentInfo,
    SubAgentSpawnedCtx,
    ToolCallEndCtx,
    ToolCallInfo,
    ToolCallStartCtx,
    ToolHandler,
    ToolResultInfo,
    ToolResultsReadyCtx,
    TransportErrorCtx,
)
from ._types import Agent, Conversation, ConversationRef, ErrorInfo, File, Response, Usage

__all__ = [
    # Client
    "AgentPlaneClient",
    "LocalServer",
    "Session",
    # Tool handler
    "ToolHandler",
    "ToolCallInfo",
    "ToolCallDenied",
    # Hooks
    "StreamHooks",
    "ToolCallStartCtx",
    "ToolCallEndCtx",
    "NativeToolCallCtx",
    "ToolResultInfo",
    "ToolResultsReadyCtx",
    "ReasoningStartCtx",
    "ReasoningEndCtx",
    "CompactionStartCtx",
    "CompactionEndCtx",
    "MessageStartCtx",
    "MessageEndCtx",
    "FileOutputCtx",
    "RetryCtx",
    "ServerErrorCtx",
    "TransportErrorCtx",
    "SubAgentInfo",
    "SubAgentSpawnedCtx",
    "SubAgentCompletedCtx",
    "ResponseStartCtx",
    "ResponseEndCtx",
    # Events
    "StreamEvent",
    "ResponseCreated",
    "ResponseQueued",
    "ResponseInProgress",
    "ResponseCompleted",
    "ResponseFailed",
    "ResponseIncomplete",
    "ResponseCancelled",
    "TextDelta",
    "ReasoningStarted",
    "ReasoningDelta",
    "ReasoningSummaryDelta",
    "ToolCall",
    "ToolResult",
    "NativeToolCall",
    "MessageDone",
    "OutputFileDone",
    "RetryEvent",
    "ErrorEvent",
    "CompactionInProgress",
    # Types
    "Response",
    "Agent",
    "File",
    "Conversation",
    "ConversationRef",
    "Usage",
    "ErrorInfo",
    # Errors
    "AgentPlaneError",
    "AgentNotFoundError",
    "ResponseNotFoundError",
    "ConversationNotFoundError",
    "InvalidInputError",
    "ConflictError",
    "BundleInvalidError",
    "ServerError",
]
