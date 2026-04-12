"""agent-plane UI SDK — Python client library for building frontends.

Usage::

    from agent_plane_ui_sdk import AgentPlaneClient, StreamRenderer
    from agent_plane_ui_sdk.terminal import RichBlockFormatter, TerminalHost
"""

from ._blocks import (
    AnyBlock,
    BlockContext,
    CompactionBlock,
    ErrorBlock,
    FileBlock,
    NativeToolBlock,
    ReasoningBlock,
    ReasoningStartBlock,
    RenderBlock,
    ResponseEndBlock,
    ResponseStartBlock,
    RetryBlock,
    TextChunk,
    TextDone,
    ToolExecution,
    ToolGroup,
)
from ._client import AgentPlaneClient
from ._errors import AgentPlaneError, ToolCallDenied
from ._renderer import StreamRenderer
from ._server import LocalServer
from ._session import Session
from ._tool_handler import StreamHooks, ToolCallInfo, ToolHandler
from ._transforms import (
    merge_text_across_iterations,
    only_agent,
    pipe,
    skip_blocks,
    skip_intermediate_ends,
)

__all__ = [
    "AgentPlaneClient",
    "AgentPlaneError",
    "AnyBlock",
    "BlockContext",
    "CompactionBlock",
    "ErrorBlock",
    "FileBlock",
    "LocalServer",
    "NativeToolBlock",
    "ReasoningBlock",
    "ReasoningStartBlock",
    "RenderBlock",
    "ResponseEndBlock",
    "ResponseStartBlock",
    "RetryBlock",
    "Session",
    "StreamHooks",
    "StreamRenderer",
    "TextChunk",
    "TextDone",
    "ToolCallDenied",
    "ToolCallInfo",
    "ToolExecution",
    "ToolGroup",
    "ToolHandler",
    "merge_text_across_iterations",
    "only_agent",
    "pipe",
    "skip_blocks",
    "skip_intermediate_ends",
]
