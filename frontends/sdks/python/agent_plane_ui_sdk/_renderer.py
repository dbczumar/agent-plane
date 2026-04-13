"""StreamRenderer — event dispatch state machine that emits semantic blocks.

Consumes the raw event stream from ``session.send()`` and produces
a stream of typed blocks with context. Each block carries a
``BlockContext`` identifying which agent produced it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ._blocks import (
    AnyBlock,
    BlockContext,
    CompactionBlock,
    ErrorBlock,
    FileBlock,
    NativeToolBlock,
    ReasoningBlock,
    ReasoningStartBlock,
    ResponseEndBlock,
    ResponseStartBlock,
    RetryBlock,
    TextChunk,
    TextDone,
    ToolExecution,
    ToolGroup,
    ToolResultBlock,
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
    TextDelta,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from ._session import Session


def _format_args_brief(name: str, arguments: dict[str, object]) -> str:
    """Format tool arguments for inline display."""
    if not arguments:
        return ""
    _KEYS = {
        "Read": "file_path",
        "Write": "file_path",
        "Edit": "file_path",
        "Bash": "command",
        "Glob": "pattern",
        "Grep": "pattern",
        "web_search": "query",
    }
    key = _KEYS.get(name)
    if key and key in arguments:
        s = str(arguments[key])
        if key == "file_path" and "/" in s:
            s = s.rsplit("/", 1)[-1]
        return s[:80] + "…" if len(s) > 80 else s
    try:
        s = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(arguments)
    return s[:80] + "…" if len(s) > 80 else s


def _format_native_label(tool_type: str, data: dict[str, object]) -> str:
    """Format a native tool label."""
    if tool_type == "web_search_call":
        action = data.get("action")
        if isinstance(action, dict):
            at = action.get("type", "")
            if at == "search":
                return f"web search: {str(action.get('query', ''))[:80]}"
            if at == "open_page":
                return f"web open: {str(action.get('url', ''))[:80]}"
        return "web search"
    if tool_type == "mcp_call":
        n = data.get("name", "")
        return f"mcp: {n}" if n else "mcp call"
    return tool_type.replace("_", " ")


class StreamRenderer:
    """Consumes a session event stream and emits semantic render blocks.

    :param text_flush_threshold: Min chars to buffer before flushing
        on a word boundary. Default 30.
    """

    def __init__(self, text_flush_threshold: int = 30) -> None:
        self._flush_threshold = text_flush_threshold

    async def stream(
        self,
        session: Session,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None = None,
    ) -> AsyncIterator[AnyBlock]:
        """Stream render blocks for one turn."""
        in_reasoning = False
        reasoning_text = ""
        summary_text = ""
        in_text = False
        accumulated = ""
        full_text = ""
        pending_tools: dict[str, ToolExecution] = {}
        agent: str | None = None
        turn = 0
        started = False

        def _ctx() -> BlockContext:
            depth = agent.count(".") if agent else 0
            return BlockContext(agent=agent, depth=depth, turn=turn)

        async for event in session.send(input, files=files):
            # ── Response lifecycle ───────────────────
            if isinstance(event, ResponseCreated):
                # Tool calls were already yielded immediately. Emit
                # result-only groups for tools that got output between
                # ResponseCompleted and this ResponseCreated.
                for ex in list(pending_tools.values()):
                    if ex.output is not None:
                        yield ToolResultBlock(
                            name=ex.name,
                            call_id=ex.call_id,
                            agent_name=ex.agent_name,
                            output=ex.output,
                            ctx=_ctx(),
                        )
                pending_tools.clear()
                agent = event.response.model
                if not started:
                    started = True
                    yield ResponseStartBlock(
                        model=agent,
                        response_id=event.response.id,
                        ctx=_ctx(),
                    )
                else:
                    turn += 1

            elif isinstance(event, ResponseQueued | ResponseInProgress):
                pass

            # ── Reasoning ────────────────────────────
            elif isinstance(event, ReasoningStarted):
                in_reasoning = True
                reasoning_text = ""
                summary_text = ""
                yield ReasoningStartBlock(ctx=_ctx())

            elif isinstance(event, ReasoningDelta):
                reasoning_text += event.delta

            elif isinstance(event, ReasoningSummaryDelta):
                summary_text += event.delta

            # ── Text ─────────────────────────────────
            elif isinstance(event, TextDelta):
                if in_reasoning:
                    in_reasoning = False
                    yield ReasoningBlock(
                        reasoning_text=reasoning_text,
                        summary_text=summary_text,
                        ctx=_ctx(),
                    )
                # Emit results for tools that completed.
                for ex in list(pending_tools.values()):
                    if ex.output is not None:
                        yield ToolResultBlock(
                            name=ex.name,
                            call_id=ex.call_id,
                            agent_name=ex.agent_name,
                            output=ex.output,
                            ctx=_ctx(),
                        )
                pending_tools.clear()

                in_text = True
                accumulated += event.delta
                full_text += event.delta

                while "\n" in accumulated:
                    line, accumulated = accumulated.split("\n", 1)
                    yield TextChunk(text=line + "\n", ctx=_ctx())

                if len(accumulated) >= self._flush_threshold:
                    last_space = accumulated.rfind(" ")
                    if last_space > 0:
                        yield TextChunk(text=accumulated[: last_space + 1], ctx=_ctx())
                        accumulated = accumulated[last_space + 1 :]

            # ── Tool calls ───────────────────────────
            elif isinstance(event, ToolCall):
                if in_reasoning:
                    in_reasoning = False
                    yield ReasoningBlock(
                        reasoning_text=reasoning_text,
                        summary_text=summary_text,
                        ctx=_ctx(),
                    )
                if in_text:
                    if accumulated:
                        yield TextChunk(text=accumulated, ctx=_ctx())
                        accumulated = ""
                    yield TextDone(
                        full_text=full_text,
                        has_code_blocks="```" in full_text,
                        ctx=_ctx(),
                    )
                    in_text = False
                    full_text = ""

                execution = ToolExecution(
                    name=event.name,
                    arguments=event.arguments,
                    args_summary=_format_args_brief(event.name, event.arguments),
                    call_id=event.call_id,
                    agent_name=event.agent_name,
                    executed_by="server",
                )
                pending_tools[event.call_id] = execution
                # Yield immediately so the user sees the tool call
                # before execution. output=None means the formatter
                # shows the call line but no result panel.
                yield ToolGroup(executions=[execution], ctx=_ctx())

            elif isinstance(event, ToolResult):
                ex = pending_tools.get(event.call_id)
                if ex is not None:
                    ex.output = event.output
                    ex.executed_by = "client"

            # ── Native tools ─────────────────────────
            elif isinstance(event, NativeToolCall):
                yield NativeToolBlock(
                    tool_type=event.tool_type,
                    label=_format_native_label(event.tool_type, event.data),
                    data=event.data,
                    ctx=_ctx(),
                )

            # ── Message done ─────────────────────────
            elif isinstance(event, MessageDone):
                if in_reasoning:
                    in_reasoning = False
                    yield ReasoningBlock(
                        reasoning_text=reasoning_text,
                        summary_text=summary_text,
                        ctx=_ctx(),
                    )
                if in_text:
                    if accumulated:
                        yield TextChunk(text=accumulated, ctx=_ctx())
                        accumulated = ""
                    yield TextDone(
                        full_text=full_text,
                        has_code_blocks="```" in full_text,
                        ctx=_ctx(),
                    )
                    in_text = False
                    full_text = ""

            # ── Status events ────────────────────────
            elif isinstance(event, CompactionInProgress):
                yield CompactionBlock(ctx=_ctx())

            elif isinstance(event, RetryEvent):
                yield RetryBlock(
                    source=event.source,
                    attempt=event.attempt,
                    max_attempts=event.max_attempts,
                    delay_seconds=event.delay_seconds,
                    ctx=_ctx(),
                )

            elif isinstance(event, ErrorEvent):
                yield ErrorBlock(message=event.error.message, source=event.source, ctx=_ctx())

            elif isinstance(event, OutputFileDone):
                yield FileBlock(file_id=event.file_id, filename=event.filename, ctx=_ctx())

            # ── Terminal events ──────────────────────
            elif isinstance(
                event,
                ResponseCompleted | ResponseFailed | ResponseIncomplete | ResponseCancelled,
            ):
                if in_reasoning:
                    in_reasoning = False
                    yield ReasoningBlock(
                        reasoning_text=reasoning_text,
                        summary_text=summary_text,
                        ctx=_ctx(),
                    )
                if in_text:
                    if accumulated:
                        yield TextChunk(text=accumulated, ctx=_ctx())
                        accumulated = ""
                    yield TextDone(
                        full_text=full_text,
                        has_code_blocks="```" in full_text,
                        ctx=_ctx(),
                    )
                    in_text = False
                    full_text = ""

                yield ResponseEndBlock(
                    status=event.response.status,
                    response=event.response,
                    ctx=_ctx(),
                )
