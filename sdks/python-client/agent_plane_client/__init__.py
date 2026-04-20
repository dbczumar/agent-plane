"""agent-plane client SDK — Python client for the agent-plane server API.

Headless HTTP/SSE client for invoking agents, tracking conversation
state, and consuming the response stream as either raw events or
semantic blocks. No UI or terminal dependencies — frontends layer
on top of this.

Usage::

    from agent_plane_client import AgentPlaneClient

    async with AgentPlaneClient(base_url="http://localhost:8080") as client:
        session = client.session(model="archer")
        async for event in session.send("hello"):
            ...

Or consume semantic blocks via :class:`BlockStream`::

    from agent_plane_client import BlockStream, pipe, skip_intermediate_ends

    stream = BlockStream()
    async for block in pipe(
        stream.stream(session, "hello"),
        skip_intermediate_ends(),
    ):
        ...
"""

from ._blocks import (
    AnyBlock,
    BlockContext,
    CompactionBlock,
    ErrorBlock,
    FileBlock,
    NativeToolBlock,
    ReasoningBlock,
    ReasoningChunk,
    ReasoningStartBlock,
    ResponseEndBlock,
    ResponseStartBlock,
    RetryBlock,
    StreamBlock,
    TextChunk,
    TextDone,
    ToolExecution,
    ToolGroup,
    ToolResultBlock,
)
from ._client import AgentPlaneClient
from ._errors import AgentPlaneError, ToolCallDenied
from ._events import RESERVED_APPROVAL_TOOL_NAME, ApprovalRequest
from ._query import QueryResult, QueryStream
from ._server import LocalServer
from ._session import Session
from ._stream import BlockStream
from ._tool_handler import (
    ApprovalRequestCtx,
    StreamHooks,
    ToolCallInfo,
    ToolHandler,
)
from ._transforms import (
    merge_text_across_iterations,
    only_agent,
    pipe,
    skip_blocks,
    skip_intermediate_ends,
)
from ._types import File
from .tools import ToolMetadata, ToolState, tool

__all__ = [
    "RESERVED_APPROVAL_TOOL_NAME",
    "AgentPlaneClient",
    "AgentPlaneError",
    "AnyBlock",
    "ApprovalRequest",
    "ApprovalRequestCtx",
    "BlockContext",
    "BlockStream",
    "CompactionBlock",
    "ErrorBlock",
    "File",
    "FileBlock",
    "LocalServer",
    "NativeToolBlock",
    "QueryResult",
    "QueryStream",
    "ReasoningBlock",
    "ReasoningChunk",
    "ReasoningStartBlock",
    "ResponseEndBlock",
    "ResponseStartBlock",
    "RetryBlock",
    "Session",
    "StreamBlock",
    "StreamHooks",
    "TextChunk",
    "TextDone",
    "ToolCallDenied",
    "ToolCallInfo",
    "ToolExecution",
    "ToolGroup",
    "ToolHandler",
    "ToolMetadata",
    "ToolResultBlock",
    "ToolState",
    "merge_text_across_iterations",
    "only_agent",
    "pipe",
    "skip_blocks",
    "skip_intermediate_ends",
    "tool",
]
