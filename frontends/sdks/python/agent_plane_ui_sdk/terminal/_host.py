"""TerminalHost — manages terminal I/O with a pinned input bar.

Wraps prompt_toolkit. All output goes through ``output()`` which
handles Rich rendering through the stdout proxy. Background tasks
keep the prompt visible during streaming.
"""

from __future__ import annotations

import asyncio
import io
import os
import pathlib
import textwrap
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from wcwidth import wcswidth

from ._formatter import FormattedItem, StreamingText

# Image extensions recognized for inline display.
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})

# File extensions recognized for attachment.
_FILE_EXTENSIONS = (
    frozenset(
        {
            ".pdf",
            ".txt",
            ".csv",
            ".json",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".xml",
            ".yaml",
            ".yml",
            ".toml",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".rb",
            ".sh",
            ".sql",
        }
    )
    | _IMAGE_EXTENSIONS
)


@dataclass
class PendingAttachment:
    """A file queued for upload with the next message."""

    path: str
    is_image: bool


def _extract_file_paths(text: str) -> list[PendingAttachment]:
    """Detect file paths in pasted text (drag-and-drop).

    Terminals like iTerm2 and Kitty convert drag-and-drop into
    pasted text with file paths (possibly shell-escaped).
    Only checks whitespace-separated tokens — no shell parsing
    that could concatenate long text with filenames.
    """
    attachments: list[PendingAttachment] = []
    for token in text.split():
        token = token.strip("'\"")
        if len(token) > 512:
            continue
        if not any(token.endswith(ext) for ext in _FILE_EXTENSIONS):
            continue
        try:
            p = pathlib.Path(os.path.expanduser(token)).resolve()
        except (OSError, ValueError):
            continue
        if not p.is_file():
            continue
        is_image = p.suffix.lower() in _IMAGE_EXTENSIONS
        attachments.append(PendingAttachment(path=str(p), is_image=is_image))

    return attachments


def _strip_file_paths(text: str, attachments: list[PendingAttachment]) -> str:
    """
    Remove detected file paths from the input text.

    After extracting attachments, the raw pasted paths (possibly
    shell-escaped) should not appear in the message sent to the LLM.
    Returns the remaining text, stripped.

    :param text: The raw input line.
    :param attachments: Detected attachments with resolved paths.
    :returns: The text with file path tokens removed.
    """
    resolved_paths = {a.path for a in attachments}
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    remaining = []
    for token in tokens:
        cleaned = token.strip("'\"")
        p = pathlib.Path(os.path.expanduser(cleaned)).resolve()
        if str(p) in resolved_paths:
            continue
        remaining.append(token)
    return " ".join(remaining).strip()


def _display_width(text: str) -> int:
    """Visible width of text in terminal columns (handles CJK, emoji)."""
    w = wcswidth(text)
    return w if w >= 0 else len(text)


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 80


def _term_height() -> int:
    try:
        return os.get_terminal_size().lines
    except (ValueError, OSError):
        return 24


class TerminalHost:
    """Terminal I/O host with a pinned input bar.

    Usage::

        async def on_input(text: str) -> None:
            ...  # process input, call host.output()

        host = TerminalHost(model_name="coder")
        async with host:
            host.output(fmt.welcome("coder"))
            await host.run(on_input)

    :param prompt_marker: Character shown before the cursor.
    :param accent_color: Color for prompt bars and marker.
    :param history_file: Path for persistent input history.
    :param model_name: Shown in the bottom toolbar.
    """

    def __init__(
        self,
        *,
        prompt_marker: str = "❯",
        accent_color: str = "#d87757",
        history_file: str = "~/.agent-plane-history",
        model_name: str = "",
    ) -> None:
        self._marker = prompt_marker
        self._accent = accent_color
        self._model = model_name
        self._tasks: list[asyncio.Task[None]] = []
        self._console = Console(highlight=False)
        self._stream_start: float | None = None
        self._last_was_streaming: bool = False
        self._text_buffer: str = ""
        self._streamed_line_count: int = 0  # Lines printed from streaming text.
        self.text_indent: str = "   "  # Indent for streaming text lines.
        self.on_help: Callable[[], None] | None = None  # F1 callback.
        self._pending_attachments: list[PendingAttachment] = []

        style = PTStyle.from_dict(
            {
                "bar": accent_color,
                "prompt-marker": f"{accent_color} bold",
                "model-name": "#6a6a6a",
                "bottom-toolbar.key": accent_color,
            }
        )

        kb = KeyBindings()

        @kb.add("escape")
        def _on_escape(event: object) -> None:
            self.cancel()

        @kb.add("f1")
        def _on_help(event: object) -> None:
            if self.on_help is not None:
                self.on_help()

        self._prompt = PromptSession(
            history=FileHistory(os.path.expanduser(history_file)),
            style=style,
            erase_when_done=True,
            key_bindings=kb,
        )

    async def __aenter__(self) -> TerminalHost:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.cancel()

    @property
    def pending_attachments(self) -> list[PendingAttachment]:
        """Files queued for the next message."""
        return self._pending_attachments

    def take_attachments(self) -> list[PendingAttachment]:
        """Take and clear pending attachments."""
        attachments = self._pending_attachments
        self._pending_attachments = []
        return attachments

    async def run(self, handler: Callable[..., Awaitable[None]]) -> None:
        """Run the input loop.

        Uses the alternate screen buffer so the prompt stays pinned
        at the bottom on terminal resize. Output scrolls above the
        prompt. On exit, the alternate buffer is discarded and the
        original terminal content is restored.

        Calls ``handler(text)`` as a background task for each input.
        The prompt re-renders immediately so the bar stays visible.
        If the user types while a handler is running, a new task
        starts (for steering). Escape cancels all tasks.
        """

        async def _toolbar_ticker() -> None:
            while True:
                if self._stream_start is not None:
                    if self._prompt.app:
                        self._prompt.app.invalidate()
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0.5)

        ticker = asyncio.create_task(_toolbar_ticker())
        try:
            with patch_stdout(raw=True):
                while True:
                    try:
                        line = await self._read_input()
                    except (EOFError, KeyboardInterrupt):
                        break

                    if line is None:
                        continue

                    line = line.strip()

                    # Detect file paths from drag-and-drop paste.
                    attachments = _extract_file_paths(line)
                    if attachments:
                        self._pending_attachments.extend(attachments)
                        # Invalidate so the paperclip renders immediately.
                        if self._prompt.app:
                            self._prompt.app.invalidate()
                        # If ONLY file paths and no other text, queue
                        # and wait for a message.
                        att_paths = {a.path for a in attachments}
                        non_file_tokens = [
                            t
                            for t in line.split()
                            if str(pathlib.Path(os.path.expanduser(t.strip("'\""))).resolve())
                            not in att_paths
                        ]
                        if not non_file_tokens:
                            continue

                    # Allow empty text when attachments are pending (the
                    # user dropped a file then hit Enter without typing).
                    if not line and not self._pending_attachments:
                        continue

                    # Clear attachments before starting the handler.
                    files = self.take_attachments()
                    task = asyncio.create_task(handler(line, files))
                    self._tasks.append(task)
                    task.add_done_callback(
                        lambda t: self._tasks.remove(t) if t in self._tasks else None
                    )
        finally:
            ticker.cancel()

    def output(self, item: FormattedItem | None) -> None:
        """Display a formatted item above the pinned prompt.

        - ``StreamingText``: printed with ``end=""`` for live streaming.
        - Rich renderables: rendered to ANSI via a temp console, printed.
        - ``None``: ignored.
        """
        if item is None:
            return
        if isinstance(item, StreamingText):
            self._text_buffer += item.text
            # Flush complete lines (LLM-produced newlines).
            while "\n" in self._text_buffer:
                line, self._text_buffer = self._text_buffer.split("\n", 1)
                self._print_text_line(line)
            # Flush when buffer fills a terminal line. Each flushed
            # line is full-width with consistent indent — no jagged
            # short lines, no terminal word-wrap without indent.
            available = max(20, _term_width() - _display_width(self.text_indent))
            while _display_width(self._text_buffer) >= available:
                wrap_at = self._text_buffer.rfind(" ", 0, available)
                if wrap_at <= 0:
                    wrap_at = available
                line = self._text_buffer[:wrap_at]
                self._text_buffer = self._text_buffer[wrap_at:].lstrip()
                print(f"{self.text_indent}{line}", flush=True)
                self._streamed_line_count += 1
            self._last_was_streaming = True
            return
        # Flush any remaining streaming text buffer (partial line).
        if self._text_buffer:
            buf = self._text_buffer
            self._text_buffer = ""
            if buf.strip():
                print(f"{self.text_indent}{buf}", flush=True)
                self._streamed_line_count += 1
            else:
                print(flush=True)
                self._streamed_line_count += 1
        if self._last_was_streaming:
            self._last_was_streaming = False
        # Non-streaming output — reset streamed line counter.
        # (clear_streamed_text must be called before this if needed.)
        self._streamed_line_count = 0
        # Render Rich content to ANSI string, print through proxy.
        buf = io.StringIO()
        temp = Console(
            file=buf,
            force_terminal=True,
            width=_term_width(),
            highlight=False,
        )
        temp.print(item)
        print(buf.getvalue(), end="", flush=True)

    def _print_text_line(self, text: str) -> None:
        """Print a line of streaming text, wrapped and indented."""
        if not text.strip():
            print(flush=True)
            self._streamed_line_count += 1
            return
        width = _term_width()
        indent = self.text_indent
        available = max(20, width - _display_width(indent))
        wrapped = textwrap.fill(
            text,
            width=available,
            initial_indent=indent,
            subsequent_indent=indent,
        )
        self._streamed_line_count += wrapped.count("\n") + 1
        print(wrapped, flush=True)

    def clear_streamed_text(self) -> None:
        """Clear the previously streamed text lines using ANSI escapes.

        Call this before outputting a re-render (e.g., markdown) to
        avoid showing duplicate content.
        """
        if self._streamed_line_count > 0:
            # Move cursor up and clear each line.
            for _ in range(self._streamed_line_count):
                print("\033[A\033[2K", end="", flush=True)
            self._streamed_line_count = 0

    @property
    def is_busy(self) -> bool:
        """True if any handler task is running."""
        return any(not t.done() for t in self._tasks)

    def start_timer(self) -> None:
        """Start the elapsed timer shown in the toolbar."""
        import time as _time

        self._stream_start = _time.monotonic()

    def stop_timer(self) -> None:
        """Stop the elapsed timer."""
        self._stream_start = None

    def cancel(self) -> None:
        """Cancel all running handler tasks."""
        self.stop_timer()
        for task in self._tasks:
            if not task.done():
                task.cancel()

    async def _read_input(self) -> str | None:
        prompt = self.build_prompt()
        toolbar = self.build_toolbar
        line = await self._prompt.prompt_async(
            prompt,
            bottom_toolbar=toolbar,
            multiline=False,
        )
        return line

    def build_prompt(self) -> FormattedText:
        width = _term_width()
        bar = "─" * width
        parts: list[tuple[str, str]] = [("class:bar", bar + "\n")]
        # Show pending attachments above the input.
        for i, att in enumerate(self._pending_attachments):
            name = pathlib.Path(att.path).name
            if att.is_image:
                parts.append(("class:prompt-marker", f" [Image #{i + 1}] "))
                parts.append(("class:model-name", f"{name}\n"))
            else:
                parts.append(("class:prompt-marker", " 📎 "))
                parts.append(("class:model-name", f"{name}\n"))
        parts.append(("class:prompt-marker", f" {self._marker} "))
        return FormattedText(parts)

    def build_toolbar(self) -> FormattedText:
        import time as _time

        if self._stream_start is not None:
            elapsed = _time.monotonic() - self._stream_start
            status = f"streaming… {elapsed:.0f}s"
        elif self.is_busy:
            status = "streaming…"
        else:
            status = "ready"
        parts = f" {self._model} · {status} "
        hints = " esc cancel · ctrl+c exit "
        width = _term_width()
        bar_right = max(0, width - 2 - len(parts) - len(hints))
        return FormattedText(
            [
                ("class:bar", "──"),
                ("class:model-name", parts),
                ("class:bottom-toolbar.key", hints),
                ("class:bar", "─" * bar_right),
            ]
        )
