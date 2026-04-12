"""Render block types with context.

Every block carries a ``BlockContext`` describing which agent produced
it, at what depth, and in which turn. The simple case ignores context.
Multi-agent UIs route by ``block.ctx.agent``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ._types import Response


@dataclass
class BlockContext:
    """Metadata attached to every render block."""

    agent: str = ""
    depth: int = 0
    turn: int = 0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class RenderBlock:
    """Base for all render blocks."""

    ctx: BlockContext = field(default_factory=BlockContext)


# ── Response lifecycle ───────────────────────────────────


@dataclass
class ResponseStartBlock(RenderBlock):
    """The response has started."""

    model: str = ""
    response_id: str = ""


# ── Tool calls ───────────────────────────────────────────


@dataclass
class ToolExecution:
    """A single tool call paired with its result."""

    name: str = ""
    arguments: dict[str, object] = field(default_factory=dict)
    args_summary: str = ""
    call_id: str = ""
    agent_name: str = ""
    executed_by: str = "server"
    output: str | None = None


@dataclass
class ToolGroup(RenderBlock):
    """A batch of tool calls from one iteration."""

    executions: list[ToolExecution] = field(default_factory=list)
    iteration: int = 0


@dataclass
class ToolResultBlock(RenderBlock):
    """A tool result, emitted after the tool executes."""

    name: str = ""
    call_id: str = ""
    agent_name: str = ""
    output: str = ""


@dataclass
class NativeToolBlock(RenderBlock):
    """A provider-native tool output (web_search, mcp, etc.)."""

    tool_type: str = ""
    label: str = ""
    data: dict[str, object] = field(default_factory=dict)


# ── Text ─────────────────────────────────────────────────


@dataclass
class TextChunk(RenderBlock):
    """A flushed chunk of streamed text."""

    text: str = ""


@dataclass
class TextDone(RenderBlock):
    """Complete text from a text-streaming section."""

    full_text: str = ""
    has_code_blocks: bool = False


# ── Reasoning ────────────────────────────────────────────


@dataclass
class ReasoningStartBlock(RenderBlock):
    """Reasoning has started — show a thinking indicator."""

    pass


@dataclass
class ReasoningBlock(RenderBlock):
    """A completed reasoning/thinking block."""

    reasoning_text: str = ""
    summary_text: str = ""


# ── Status ───────────────────────────────────────────────


@dataclass
class ErrorBlock(RenderBlock):
    """An error during the response."""

    message: str = ""
    source: str = ""


@dataclass
class RetryBlock(RenderBlock):
    """The server is retrying."""

    source: str = ""
    attempt: int = 0
    max_attempts: int = 0
    delay_seconds: float = 0.0


@dataclass
class CompactionBlock(RenderBlock):
    """Conversation is being compacted."""

    pass


@dataclass
class FileBlock(RenderBlock):
    """A file artifact produced by the agent."""

    file_id: str = ""
    filename: str | None = None


@dataclass
class ResponseEndBlock(RenderBlock):
    """The response reached a terminal state."""

    status: str = ""
    response: Response | None = None


# Union of all block types (for type hints).
AnyBlock = (
    ResponseStartBlock
    | ToolGroup
    | ToolResultBlock
    | NativeToolBlock
    | TextChunk
    | TextDone
    | ReasoningStartBlock
    | ReasoningBlock
    | ErrorBlock
    | RetryBlock
    | CompactionBlock
    | FileBlock
    | ResponseEndBlock
)
