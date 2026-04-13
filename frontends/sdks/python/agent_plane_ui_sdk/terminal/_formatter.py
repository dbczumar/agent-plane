"""RichBlockFormatter — converts render blocks to Rich renderables.

Each block type has a ``format_*`` method. Override any method to
customize rendering. The base class provides a polished terminal
treatment with panels, syntax highlighting, and a warm color scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from .._blocks import (
    AnyBlock,
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


@dataclass
class StreamingText:
    """Marker: the host should print this with ``end=""`` for streaming."""

    text: str


# Union of items a formatter can return.
FormattedItem = RenderableType | StreamingText


class RichBlockFormatter:
    """Converts render blocks to Rich renderables.

    :param accent_color: Warm accent for branding (default orange).
    :param code_theme: Pygments theme for code blocks.
    :param max_result_lines: Max lines in tool result panels.
    :param show_agent_labels: Prefix sub-agent blocks with agent name.
    """

    def __init__(
        self,
        *,
        accent_color: str = "#d87757",
        code_theme: str = "monokai",
        max_result_lines: int = 30,
        show_agent_labels: bool = False,
    ) -> None:
        self.accent = accent_color
        self.code_theme = code_theme
        self.max_result_lines = max_result_lines
        self.show_agent_labels = show_agent_labels

        # Derived styles.
        self.assistant = "bold green"
        self.muted = "#6a6a6a"
        self.warning = "#ffa500"
        self.error = "bold #ff6b80"
        self.success = "#4eba65"
        self.reasoning_style = "dim italic #8a8a8a"

    # ── Main dispatch ────────────────────────────────────

    def format(self, block: AnyBlock) -> list[FormattedItem]:
        """Format a block into display items."""
        items = self._dispatch(block)
        if self.show_agent_labels and block.ctx.depth > 0:
            label = Text.from_markup(f"   [{self.muted}][{block.ctx.agent}][/{self.muted}]")
            return [label, *items]
        return items

    def _dispatch(self, block: AnyBlock) -> list[FormattedItem]:
        if isinstance(block, ResponseStartBlock):
            return self.format_response_start(block)
        if isinstance(block, TextChunk):
            return self.format_text_chunk(block)
        if isinstance(block, TextDone):
            return self.format_text_done(block)
        if isinstance(block, ToolGroup):
            return self.format_tool_group(block)
        if isinstance(block, ToolResultBlock):
            return self.format_tool_result(block)
        if isinstance(block, NativeToolBlock):
            return self.format_native_tool(block)
        if isinstance(block, ReasoningStartBlock):
            return self.format_reasoning_start(block)
        if isinstance(block, ReasoningBlock):
            return self.format_reasoning(block)
        if isinstance(block, ErrorBlock):
            return self.format_error(block)
        if isinstance(block, RetryBlock):
            return self.format_retry(block)
        if isinstance(block, CompactionBlock):
            return self.format_compaction(block)
        if isinstance(block, FileBlock):
            return self.format_file(block)
        if isinstance(block, ResponseEndBlock):
            return self.format_response_end(block)
        return []

    # ── Override points ──────────────────────────────────

    def format_response_start(self, block: ResponseStartBlock) -> list[FormattedItem]:
        return [Text.from_markup(f"\n [{self.assistant}]◆ {block.model}[/{self.assistant}]")]

    def format_text_chunk(self, block: TextChunk) -> list[FormattedItem]:
        return [StreamingText(text=block.text)]

    def format_text_done(self, block: TextDone) -> list[FormattedItem]:
        if block.has_code_blocks:
            return [
                Padding(
                    Markdown(block.full_text, code_theme=self.code_theme),
                    (0, 1, 0, 3),
                )
            ]
        return []

    def format_tool_group(self, block: ToolGroup) -> list[FormattedItem]:
        items: list[FormattedItem] = []
        for ex in block.executions:
            items.append(self._tool_call_line(ex))
            if ex.output is not None:
                items.append(self._tool_result_panel(ex))
        return items

    def format_tool_result(self, block: ToolResultBlock) -> list[FormattedItem]:
        """Render a tool result panel (no call line — already displayed)."""
        ex = ToolExecution(
            name=block.name,
            # args_summary is not displayed for result-only panels,
            # but is required by ToolExecution.
            args_summary="",
            call_id=block.call_id,
            agent_name=block.agent_name,
            output=block.output,
        )
        return [self._tool_result_panel(ex)]

    def format_native_tool(self, block: NativeToolBlock) -> list[FormattedItem]:
        return [Text.from_markup(f"   [{self.accent}]⏵ {block.label}[/{self.accent}]")]

    def format_reasoning_start(self, block: ReasoningStartBlock) -> list[FormattedItem]:
        return [
            Text.from_markup(
                f"   [{self.accent}]·[/{self.accent}] [{self.muted}]thinking…[/{self.muted}]"
            )
        ]

    def format_reasoning(self, block: ReasoningBlock) -> list[FormattedItem]:
        text = block.summary_text or block.reasoning_text
        if text.strip():
            return [self._reasoning_panel(text)]
        return []

    def format_error(self, block: ErrorBlock) -> list[FormattedItem]:
        src = f"[{block.source}] " if block.source else ""
        return [
            Padding(
                Panel(
                    Text(f"{src}{block.message}", style=self.error),
                    border_style="#ff6b80",
                    box=box.ROUNDED,
                    padding=(0, 1),
                ),
                (0, 1, 0, 3),
            )
        ]

    def format_retry(self, block: RetryBlock) -> list[FormattedItem]:
        return [
            Text.from_markup(
                f"   [{self.warning}]↻ retrying {block.source}"
                f" ({block.attempt}/{block.max_attempts})…[/{self.warning}]"
            )
        ]

    def format_compaction(self, block: CompactionBlock) -> list[FormattedItem]:
        return [Text.from_markup(f"   [{self.muted}]◐ compacting…[/{self.muted}]")]

    def format_file(self, block: FileBlock) -> list[FormattedItem]:
        name = block.filename or block.file_id
        return [Text.from_markup(f"   [{self.success}]📎 {name}[/{self.success}]")]

    def format_response_end(self, block: ResponseEndBlock) -> list[FormattedItem]:
        if block.status == "completed":
            return []
        return [Text.from_markup(f"   [{self.warning}]{block.status}[/{self.warning}]")]

    # ── Non-block helpers ────────────────────────────────

    def welcome(self, model: str) -> FormattedItem:
        """The welcome banner."""
        return Panel(
            Text.from_markup(
                f"[{self.accent}]agent plane[/{self.accent}]"
                f"  [{self.muted}]·[/{self.muted}]  [bold]{model}[/bold]\n"
                f"[{self.muted}]Type a message to chat · F1 help"
                f" · Esc cancel · Ctrl+C exit[/{self.muted}]"
            ),
            box=box.ROUNDED,
            border_style=self.accent,
            padding=(0, 1),
        )

    def user_message(self, text: str) -> FormattedItem:
        """Format a user message with accent marker and gray background."""
        truncated = text
        lines = text.split("\n")
        if len(lines) > 4:
            truncated = "\n".join(lines[:4]) + f"\n… {len(lines) - 4} more lines"
        escaped = truncated.replace("[", "\\[").replace("]", "\\]")
        return Text.from_markup(
            f"\n [{self.accent}]❯[/{self.accent}] [on #1a1a1a]{escaped}[/on #1a1a1a]"
        )

    def goodbye(self) -> FormattedItem:
        """Goodbye message."""
        return Text.from_markup(f"\n  [{self.muted}]Goodbye.[/{self.muted}]\n")

    # ── Internal builders ────────────────────────────────

    def _tool_call_line(self, ex: ToolExecution) -> FormattedItem:
        color = self.accent
        prefix = ""
        if "." in ex.agent_name:
            prefix = f"[{self.muted}]{ex.agent_name} → [/{self.muted}]"
        args = ex.args_summary
        return Text.from_markup(f"   {prefix}[{color}]⏵ {ex.name}[/{color}][dim]({args})[/dim]")

    def _tool_result_panel(self, ex: ToolExecution) -> FormattedItem:
        output = ex.output or ""
        lines = output.split("\n")
        total = len(lines)
        if total > self.max_result_lines:
            visible = "\n".join(lines[: self.max_result_lines])
            footer = f"\n[{self.muted}]… {total - self.max_result_lines} more lines[/{self.muted}]"
        else:
            visible = output
            footer = ""
        first_line = lines[0][:80] if lines else ""
        escaped_fl = first_line.replace("[", "\\[").replace("]", "\\]")
        escaped_vis = visible.replace("[", "\\[").replace("]", "\\]")
        return Padding(
            Panel(
                Text.from_markup(f"[dim]{escaped_vis}{footer}[/dim]"),
                title=f"[dim]{escaped_fl}[/dim]",
                title_align="left",
                border_style=self.accent,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            (0, 1, 0, 3),
        )

    def _reasoning_panel(self, text: str) -> FormattedItem:
        lines = text.strip().split("\n")
        if len(lines) > 8:
            preview = "\n".join(lines[-8:])
            preview = f"[{self.muted}]… {len(lines) - 8} earlier lines[/{self.muted}]\n" + preview
        else:
            preview = "\n".join(lines)
        escaped = preview.replace("[", "\\[").replace("]", "\\]")
        return Padding(
            Panel(
                Text.from_markup(f"[{self.reasoning_style}]{escaped}[/{self.reasoning_style}]"),
                title=f"[{self.muted}]thinking[/{self.muted}]",
                title_align="left",
                border_style=self.muted,
                box=box.ROUNDED,
                padding=(0, 1),
            ),
            (0, 1, 0, 3),
        )
