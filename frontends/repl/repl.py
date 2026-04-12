#!/usr/bin/env python
"""Rich-based REPL for agent-plane.

Usage:
    python repl.py <agent-dir-or-tarball>
    python repl.py <agent-dir-or-tarball> --client-tools coder
    python repl.py --server http://remote:8000 <agent-name>

Features:
    - Streaming text with progressive markdown rendering (Rich Live)
    - prompt_toolkit input with persistent history and multi-line
    - Steering: type while the agent is streaming to redirect it
    - Ctrl+C to cancel a running response
    - Slash commands for conversation management
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from types import ModuleType

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console

# Add the client library to the path.
_CLIENT_LIB = str(pathlib.Path(__file__).resolve().parent.parent / "clients" / "python")
if _CLIENT_LIB not in sys.path:
    sys.path.insert(0, _CLIENT_LIB)

import renderer  # noqa: E402
from agent_plane_client import (  # noqa: E402
    AgentPlaneClient,
    CompactionInProgress,
    ErrorEvent,
    LocalServer,
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
    Session,
    StreamHooks,
    TextDelta,
    ToolCall,
    ToolCallInfo,
    ToolHandler,
    ToolResult,
)
from commands import handle_command  # noqa: E402
from renderer import (  # noqa: E402
    render_error,
    render_goodbye,
    render_server_ready,
    render_server_starting,
    render_welcome,
)

console = Console()

_HISTORY_PATH = os.path.expanduser("~/.agent-plane-repl-history")


def rprint(*args: object, **kwargs: object) -> None:
    """Print Rich content through plain print() for patch_stdout compatibility.

    Renders Rich content (panels, styled text, markdown) to a string
    with ANSI codes, then writes it via ``print(flush=True)`` which
    goes through prompt_toolkit's stdout proxy and appears above the
    pinned prompt immediately.

    Accepts both Rich markup strings and Rich renderables (Panel,
    Padding, Markdown, etc).
    """
    import io as _io

    buf = _io.StringIO()
    try:
        width = os.get_terminal_size().columns
    except (ValueError, OSError):
        width = 80
    temp = Console(file=buf, force_terminal=True, width=width, highlight=False)
    temp.print(*args, **kwargs)
    output = buf.getvalue()
    print(output, end="", flush=True)


# ── Tool set loading ─────────────────────────────────────


def _load_tool_set(name: str) -> ModuleType:
    """Load a client-side tool set module by name."""
    frontends_dir = str(
        pathlib.Path(__file__).resolve().parent.parent.parent / "examples" / "frontends"
    )
    if frontends_dir not in sys.path:
        sys.path.insert(0, frontends_dir)
    from tool_sets import get_tool_set

    return get_tool_set(name)


def _make_tool_handler(tool_set: ModuleType) -> ToolHandler:
    """Create a ToolHandler from a tool set module."""

    def execute(call: ToolCallInfo) -> str:
        return tool_set.execute_tool(call.name, call.arguments)

    return ToolHandler(schemas=tool_set.TOOLS, execute=execute)


# ── Main REPL ────────────────────────────────────────────


# ── prompt_toolkit styling ───────────────────────────────

# Style for the prompt input area — accent-colored bars.
_PT_STYLE = PTStyle.from_dict(
    {
        "bar": "#d87757",
        "prompt-marker": "#d87757 bold",
        "model-name": "#6a6a6a",
        "bottom-toolbar": "bg:#1a1a1a #6a6a6a",
        "bottom-toolbar.key": "#d87757",
    }
)


def _get_terminal_width() -> int:
    """Get the terminal width, defaulting to 80."""
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 80


def _get_terminal_height() -> int:
    """Get the terminal height, defaulting to 24."""
    try:
        return os.get_terminal_size().lines
    except (ValueError, OSError):
        return 24


def _make_prompt_text(model: str) -> FormattedText:
    """Build the styled prompt with a top bar.

    Looks like:
        ───────────────────────────────
         ❯
    """
    width = _get_terminal_width()
    bar = "─" * width
    return FormattedText(
        [
            ("class:bar", bar),
            ("", "\n"),
            ("class:prompt-marker", " ❯ "),
        ]
    )


def _make_bottom_toolbar(model: str, session_obj: Session, is_busy: bool = False) -> FormattedText:
    """Build the bottom toolbar with model name and hints.

    Looks like:
        ───── coder · ready · ctrl+c exit ─────
    """
    status = "streaming…" if is_busy else "ready"
    parts = f" {model} · {status} "
    hints = " esc cancel · ctrl+c exit "
    width = _get_terminal_width()
    bar_left_len = 2
    bar_right_len = max(0, width - bar_left_len - len(parts) - len(hints))
    return FormattedText(
        [
            ("class:bar", "─" * bar_left_len),
            ("class:model-name", parts),
            ("class:bottom-toolbar.key", hints),
            ("class:bar", "─" * bar_right_len),
        ]
    )


class Repl:
    """The main REPL — prompt always visible, streaming in background.

    ``patch_stdout(raw=True)`` wraps the entire loop. ``prompt_async``
    runs continuously so the input bar is always at the bottom.
    Streaming runs as a background task. All output goes through
    ``rprint()`` which renders Rich content to ANSI strings and writes
    via ``print(flush=True)`` — this goes through prompt_toolkit's
    stdout proxy and appears above the pinned prompt immediately.
    """

    def __init__(
        self,
        client: AgentPlaneClient,
        agent_name: str,
        tool_handler: ToolHandler | None,
    ) -> None:
        self._client = client
        self._tool_handler = tool_handler
        self._hooks = StreamHooks()
        self._session = client.session(
            model=agent_name,
            tool_handler=tool_handler,
            hooks=self._hooks,
        )
        self._state: dict = {}
        self._stream_task: asyncio.Task[None] | None = None

        # Escape cancels the running stream.
        kb = KeyBindings()

        @kb.add("escape")
        def _on_escape(event: object) -> None:
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
                asyncio.ensure_future(self._cancel_stream())

        self._prompt = PromptSession(
            history=FileHistory(_HISTORY_PATH),
            style=_PT_STYLE,
            erase_when_done=True,
            key_bindings=kb,
        )

    async def run(self) -> None:
        """Run the REPL loop — prompt is always visible."""
        from prompt_toolkit.patch_stdout import patch_stdout

        with patch_stdout(raw=True):
            while True:
                try:
                    line = await self._read_input()
                except (EOFError, KeyboardInterrupt):
                    if self._stream_task and not self._stream_task.done():
                        self._stream_task.cancel()
                        try:
                            await self._stream_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        await self._session.cancel()
                    render_goodbye(console)
                    break

                if line is None or not line.strip():
                    continue
                line = line.strip()

                # Slash commands.
                if line.startswith("/"):
                    handled = await handle_command(
                        line, console, self._client, self._session, self._state
                    )
                    if self._state.get("quit"):
                        break
                    if self._state.get("switch_model"):
                        new_model = self._state.pop("switch_model")
                        self._session = self._client.session(
                            model=new_model,
                            tool_handler=self._tool_handler,
                            hooks=self._hooks,
                        )
                    if handled:
                        continue

                # Show user message.
                user_text = renderer.escape_markup(renderer.truncate_user_text(line))
                rprint(
                    f" [{renderer.ACCENT}]❯[/{renderer.ACCENT}]"
                    f" [on #1a1a1a]{user_text}[/on #1a1a1a]"
                )

                # If a stream is running, this is steering.
                if self._stream_task and not self._stream_task.done():
                    asyncio.create_task(self._steer(line))
                    continue

                # Start streaming in background — prompt immediately
                # re-renders so the bar never disappears.
                self._stream_task = asyncio.create_task(self._run_turn(line))

    async def _read_input(self) -> str | None:
        """Read input with styled prompt bars."""
        model = self._session.model
        prompt_text = _make_prompt_text(model)
        is_busy = self._stream_task is not None and not self._stream_task.done()

        def toolbar() -> FormattedText:
            return _make_bottom_toolbar(model, self._session, is_busy)

        line = await self._prompt.prompt_async(
            prompt_text,
            bottom_toolbar=toolbar,
            multiline=False,
        )
        return line

    async def _cancel_stream(self) -> None:
        """Cancel the running stream and notify the server."""
        try:
            await self._session.cancel()
            rprint(f"   [{renderer.WARNING}]Cancelled.[/{renderer.WARNING}]")
        except Exception:
            pass

    async def _steer(self, text: str) -> None:
        """Send a steering message."""
        try:
            async for _ in self._session.send(text):
                pass
        except Exception as exc:
            rprint(f"   [{renderer.WARNING}]Steering failed: {exc}[/{renderer.WARNING}]")

    async def _run_turn(self, text: str) -> None:
        """Stream a full turn. Output goes through rprint() above the bar."""
        files = self._state.pop("pending_files", None)
        accumulated_text = ""  # Unflushed remainder (partial line).
        full_text = ""  # Full text for code-block re-render.
        in_text = False
        in_reasoning = False
        reasoning_text = ""
        summary_text = ""
        model = self._session.model
        header_printed = False

        def _print_header() -> None:
            nonlocal header_printed
            if not header_printed:
                header_printed = True
                rprint(f"\n [{renderer.ASSISTANT}]◆ {model}[/{renderer.ASSISTANT}]")

        try:
            async for event in self._session.send(text, files=files):
                if isinstance(event, (ResponseCreated, ResponseQueued, ResponseInProgress)):
                    pass

                elif isinstance(event, ReasoningStarted):
                    _print_header()
                    in_reasoning = True
                    reasoning_text = ""
                    summary_text = ""
                    rprint(
                        f"   [{renderer.ACCENT}]·[/{renderer.ACCENT}] "
                        f"[{renderer.MUTED}]thinking…[/{renderer.MUTED}]"
                    )

                elif isinstance(event, ReasoningDelta):
                    reasoning_text += event.delta

                elif isinstance(event, ReasoningSummaryDelta):
                    summary_text += event.delta

                elif isinstance(event, TextDelta):
                    if in_reasoning:
                        in_reasoning = False
                        rprint(renderer._build_reasoning_panel(reasoning_text, summary_text))
                    if not in_text:
                        _print_header()
                        in_text = True
                    accumulated_text += event.delta
                    full_text += event.delta
                    # Flush complete lines immediately.
                    while "\n" in accumulated_text:
                        line_out, accumulated_text = accumulated_text.split("\n", 1)
                        rprint(f"   {renderer.escape_markup(line_out)}")
                    # For text without newlines: flush at word boundaries
                    # once we've buffered ~30+ chars. This gives a few
                    # words per flush — fast enough to feel real-time,
                    # long enough to not be one-word-per-line.
                    if len(accumulated_text) >= 30:
                        last_space = accumulated_text.rfind(" ")
                        if last_space > 0:
                            rprint(f"   {renderer.escape_markup(accumulated_text[:last_space])}")
                            accumulated_text = accumulated_text[last_space + 1 :]

                elif isinstance(event, ToolCall):
                    _print_header()
                    if in_reasoning:
                        in_reasoning = False
                        rprint(renderer._build_reasoning_panel(reasoning_text, summary_text))
                    if in_text:
                        in_text = False
                        if accumulated_text.strip():
                            rprint(f"   {renderer.escape_markup(accumulated_text)}")
                        accumulated_text = ""
                    rprint(
                        renderer._build_tool_call_line(
                            event.name, event.arguments, event.agent_name
                        )
                    )

                elif isinstance(event, ToolResult):
                    rprint(renderer.build_tool_result_panel("", event.output))

                elif isinstance(event, NativeToolCall):
                    _print_header()
                    rprint(renderer._build_native_tool_line(event.tool_type, event.data))

                elif isinstance(event, MessageDone):
                    if in_reasoning:
                        in_reasoning = False
                        rprint(renderer._build_reasoning_panel(reasoning_text, summary_text))
                    if in_text:
                        in_text = False
                        # Flush remaining partial line.
                        if accumulated_text.strip():
                            rprint(f"   {renderer.escape_markup(accumulated_text)}")
                        # Re-render full text as rich markdown if it
                        # contains code blocks (syntax highlighting).
                        if full_text.strip() and "```" in full_text:
                            rprint(renderer._build_message_text(full_text))
                    accumulated_text = ""
                    full_text = ""

                elif isinstance(event, OutputFileDone):
                    rprint(
                        f"   [{renderer.SUCCESS}]📎 "
                        f"{event.filename or event.file_id}"
                        f"[/{renderer.SUCCESS}]"
                    )

                elif isinstance(event, CompactionInProgress):
                    rprint(f"   [{renderer.MUTED}]◐ compacting conversation…[/{renderer.MUTED}]")

                elif isinstance(event, RetryEvent):
                    rprint(
                        f"   [{renderer.WARNING}]↻ retrying {event.source} "
                        f"(attempt {event.attempt}/{event.max_attempts})…[/{renderer.WARNING}]"
                    )

                elif isinstance(event, ErrorEvent):
                    rprint(
                        f"   [{renderer.ERROR}]Error [{event.source}]:"
                        f" {event.error.message}[/{renderer.ERROR}]"
                    )

                elif isinstance(event, ResponseCompleted):
                    if in_reasoning:
                        in_reasoning = False
                        rprint(renderer._build_reasoning_panel(reasoning_text, summary_text))
                    if in_text:
                        in_text = False
                        if accumulated_text.strip():
                            rprint(f"   {renderer.escape_markup(accumulated_text)}")
                        if full_text.strip() and "```" in full_text:
                            rprint(renderer._build_message_text(full_text))
                    accumulated_text = ""
                    full_text = ""

                elif isinstance(event, ResponseFailed):
                    _print_header()
                    in_text = False
                    err = event.response.error
                    rprint(
                        f"   [{renderer.ERROR}]Error: "
                        f"{err.message if err else 'Unknown error'}"
                        f"[/{renderer.ERROR}]"
                    )

                elif isinstance(event, ResponseIncomplete):
                    if in_text:
                        in_text = False
                        if accumulated_text.strip():
                            rprint(f"   {renderer.escape_markup(accumulated_text)}")
                        accumulated_text = ""
                        full_text = ""
                    rprint(
                        f"   [{renderer.WARNING}]incomplete ({event.reason})[/{renderer.WARNING}]"
                    )

                elif isinstance(event, ResponseCancelled):
                    if in_text:
                        in_text = False
                    rprint(f"   [{renderer.WARNING}]cancelled[/{renderer.WARNING}]")

        except asyncio.CancelledError:
            pass  # _cancel_stream already printed the message.
        except Exception as exc:
            if in_text:
                print(flush=True)
            rprint(f"   [{renderer.ERROR}]Error: {exc}[/{renderer.ERROR}]")
        rprint("")  # Blank line after response.


# ── Entry point ──────────────────────────────────────────


def _parse_args() -> tuple[str | None, str, str | None]:
    """Parse CLI arguments. Returns (remote_server, agent_path_or_name, tool_set_name)."""
    args = sys.argv[1:]
    remote_server: str | None = None
    tool_set_name: str | None = None

    if "--server" in args:
        idx = args.index("--server")
        if idx + 1 < len(args):
            remote_server = args[idx + 1].rstrip("/")
            args = args[:idx] + args[idx + 2 :]
        else:
            console.print("[red]Error: --server requires a URL[/red]")
            sys.exit(1)

    if "--client-tools" in args:
        idx = args.index("--client-tools")
        if idx + 1 < len(args):
            tool_set_name = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            console.print("[red]Error: --client-tools requires a name[/red]")
            sys.exit(1)

    if not args:
        console.print(
            "Usage: python repl.py <agent-dir-or-tarball> [--client-tools NAME] [--server URL]"
        )
        sys.exit(1)

    return remote_server, args[0], tool_set_name


async def main() -> None:
    """Entry point — parse args, start server or connect, run REPL."""
    remote_server, agent_path_or_name, tool_set_name = _parse_args()

    tool_handler: ToolHandler | None = None
    if tool_set_name is not None:
        tool_set = _load_tool_set(tool_set_name)
        tool_handler = _make_tool_handler(tool_set)

    # Push everything to the bottom of the terminal.
    # Welcome box is ~5 lines, prompt is ~3 lines, we want a few
    # blank lines between welcome and prompt for breathing room.
    term_height = _get_terminal_height()
    # Reserve: welcome(5) + gap(3) + prompt area(3) = 11
    console.print("\n" * max(0, term_height - 11))

    if remote_server is not None:
        agent_name = agent_path_or_name
        async with AgentPlaneClient(base_url=remote_server) as client:
            agent = await client.agents.get_by_name(agent_name)
            if agent is None:
                render_error(console, f"Agent '{agent_name}' not found on {remote_server}")
                return
            render_welcome(console, agent_name)
            console.print("\n\n")
            repl = Repl(client, agent_name, tool_handler)
            await repl.run()
    else:
        agent_path = agent_path_or_name
        render_server_starting(console)
        async with LocalServer(agent_path=agent_path) as server:
            client = server.client
            agents = await client.agents.list()
            if not agents:
                render_error(console, "No agents found after server startup")
                return
            agent_name = agents[0].name
            render_server_ready(console, server.base_url)
            render_welcome(console, agent_name)
            console.print("\n\n")
            repl = Repl(client, agent_name, tool_handler)
            await repl.run()


if __name__ == "__main__":
    asyncio.run(main())
