"""Rich-based REPL for agent-plane — built on the UI SDK framework.

The public API is ``run_repl(client, agent_name, tool_handler)``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from agent_plane_ui_sdk import (
    AgentPlaneClient,
    ResponseEndBlock,
    ResponseStartBlock,
    StreamRenderer,
    ToolHandler,
    pipe,
    skip_intermediate_ends,
)
from agent_plane_ui_sdk.terminal import (
    RichBlockFormatter,
    TerminalHost,
)
from rich.text import Text


class TimedFormatter(RichBlockFormatter):  # type: ignore[misc]
    """Shows final elapsed time after response completes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._start_time: float | None = None

    def format_response_start(self, block: ResponseStartBlock) -> list[Any]:
        self._start_time = block.ctx.timestamp
        return super().format_response_start(block)

    def format_response_end(self, block: ResponseEndBlock) -> list[Any]:
        items = super().format_response_end(block)
        if self._start_time is not None:
            elapsed = block.ctx.timestamp - self._start_time
            items.append(Text.from_markup(f"   [{self.muted}]{elapsed:.1f}s[/{self.muted}]"))
            self._start_time = None
        return items


async def run_repl(
    client: AgentPlaneClient,
    agent_name: str,
    tool_handler: ToolHandler | None,
    *,
    initial_message: str | None = None,
) -> None:
    """The entire REPL — using the framework.

    :param client: Connected AgentPlaneClient.
    :param agent_name: Agent name (used for API calls).
    :param tool_handler: Optional client-side tool handler.
    :param initial_message: If set, auto-send this message on startup
        (e.g. a greeting prompt for onboarding).
    """
    ui_name = agent_name.replace("-", " ").replace("_", " ")
    session = client.session(model=agent_name, tool_handler=tool_handler)
    renderer = StreamRenderer()
    fmt = TimedFormatter(show_agent_labels=True)
    host = TerminalHost(model_name=ui_name)
    # Queued steering messages — displayed after current stream ends.
    pending_steers: list[str] = []
    is_streaming = False

    def show_help() -> None:
        from rich.text import Text

        lines = []
        for name, (desc, _) in COMMANDS.items():
            if name in ("/?", "/exit"):
                continue  # Skip aliases.
            lines.append(
                f"  [{fmt.accent}]{name}[/{fmt.accent}]  [{fmt.muted}]{desc}[/{fmt.muted}]"
            )
        host.output(Text.from_markup("\n".join(lines)))

    host.on_help = show_help

    async def on_input(text: str, attachments: list[Any] | None = None) -> None:
        nonlocal is_streaming

        if text.startswith("/"):
            await handle_slash_command(text, session, client, host, fmt)
            return

        files = [a.path for a in attachments] if attachments else None

        if is_streaming:
            # Another handler is actively streaming. Queue the display
            # and send the steer, but don't print yet — it would
            # interleave with the agent's output.
            pending_steers.append(text)
            async for _ in session.send(text, files=files):
                pass  # Steer yields nothing if delivered.
            return

        host.output(fmt.user_message(text))
        host.start_timer()
        await asyncio.sleep(0)
        is_streaming = True
        try:
            stream = pipe(
                renderer.stream(session, text, files=files),
                skip_intermediate_ends(),
            )
            from agent_plane_ui_sdk import TextDone

            async for block in stream:
                if isinstance(block, TextDone) and block.has_code_blocks:
                    host.clear_streamed_text()
                for item in fmt.format(block):
                    host.output(item)
                await asyncio.sleep(0)
        finally:
            is_streaming = False
            host.stop_timer()
            # Show queued steering messages now that the stream ended.
            for steer_text in pending_steers:
                host.output(fmt.user_message(steer_text))
            pending_steers.clear()

    async with host:
        host.output(fmt.welcome(ui_name))

        from agent_plane_ui_sdk.terminal import StreamingText

        host.output(StreamingText(text="\n\n\n"))
        if initial_message:
            # Auto-send the initial message (e.g. onboarding greeting).
            asyncio.create_task(on_input(initial_message))
        await host.run(on_input)
    host.output(fmt.goodbye())


def _clear_screen() -> None:
    """Clear visible content by scrolling it off screen."""

    try:
        height = os.get_terminal_size().lines
    except (ValueError, OSError):
        height = 24
    print("\n" * height, end="", flush=True)


# ── Slash commands ───────────────────────────────────────

# Single registry: name → (help string, handler).
# Handlers take (arg, session, client, host, fmt).

COMMANDS: dict[str, tuple[str, Any]] = {}


def _cmd(name: str, help_text: str) -> Any:
    """Decorator to register a slash command."""

    def _register(fn: Any) -> Any:
        COMMANDS[name] = (help_text, fn)
        return fn

    return _register


@_cmd("/help", "Show this help")
async def _cmd_help(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    lines = []
    for name, (desc, _) in COMMANDS.items():
        if name in ("/?", "/exit"):
            continue  # Skip aliases.
        lines.append(f"  [{fmt.accent}]{name}[/{fmt.accent}]  [{fmt.muted}]{desc}[/{fmt.muted}]")
    host.output(Text.from_markup("\n".join(lines)))


COMMANDS["/?"] = COMMANDS["/help"]


@_cmd("/new", "Start a new conversation")
async def _cmd_new(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    session.reset()
    _clear_screen()
    host.output(fmt.welcome(session._model))
    host.output(Text.from_markup(f"\n  [{fmt.muted}]New conversation.[/{fmt.muted}]"))


@_cmd("/switch", "List or switch conversations")
async def _cmd_switch(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from datetime import datetime

    from rich.table import Table
    from rich.text import Text

    if not arg:
        convos = await client.conversations.list(limit=20)
        if convos:
            table = Table(title="Switch to…")
            table.add_column("#", style="bold " + fmt.accent)
            table.add_column("ID", style="dim")
            table.add_column("Title")
            table.add_column("Created", style="dim")
            for i, c in enumerate(convos, 1):
                when = datetime.fromtimestamp(c.created_at).strftime("%b %d %H:%M")
                table.add_row(str(i), c.id, c.title or "(untitled)", when)
            host.output(table)
            host.output(Text.from_markup(f"  [{fmt.muted}]/switch <id> to resume[/{fmt.muted}]"))
        else:
            host.output(Text.from_markup(f"  [{fmt.muted}]No conversations.[/{fmt.muted}]"))
    else:
        try:
            items = await client.conversations.list_items(arg, limit=100)
            last_response_id = None
            for item in reversed(items):
                rid = item.get("response_id")
                if isinstance(rid, str):
                    last_response_id = rid
                    break
            if last_response_id:
                session.reset()
                session.resume_from_response(last_response_id)
                # Clear screen and show recent history in consistent style.
                # ~5 welcome + 2 label + 2*recent messages.
                _clear_screen()
                host.output(fmt.welcome(session._model))
                host.output(
                    Text.from_markup(
                        f"  [{fmt.muted}]Resumed conversation {arg[:16]}…[/{fmt.muted}]\n"
                    )
                )
                recent = items[-6:] if len(items) > 6 else items
                for item in recent:
                    _render_history_item(item, host, fmt)
            else:
                host.output(Text.from_markup(f"  [{fmt.muted}]Empty conversation.[/{fmt.muted}]"))
        except Exception as exc:
            host.output(Text.from_markup(f"  [bold red]Error: {exc}[/]"))


@_cmd("/history", "Show current conversation history")
async def _cmd_history(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    if not session.current_response_id:
        host.output(Text.from_markup(f"  [{fmt.muted}]No active conversation.[/{fmt.muted}]"))
        return
    try:
        resp = await client.responses.get(session.current_response_id)
        if resp.conversation:
            items = await client.conversations.list_items(resp.conversation.id, limit=50)
            for item in items:
                _render_history_item(item, host, fmt)
        else:
            host.output(Text.from_markup(f"  [{fmt.muted}]No conversation.[/{fmt.muted}]"))
    except Exception as exc:
        host.output(Text.from_markup(f"  [bold red]Error: {exc}[/]"))


@_cmd("/agents", "List available agents")
async def _cmd_agents(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.table import Table
    from rich.text import Text

    agents = await client.agents.list()
    if agents:
        table = Table(title="Agents")
        table.add_column("Name", style="bold")
        table.add_column("ID", style="dim")
        for a in agents:
            table.add_row(a.name, a.id)
        host.output(table)
    else:
        host.output(Text.from_markup(f"  [{fmt.muted}]No agents.[/{fmt.muted}]"))


@_cmd("/cancel", "Cancel the current response")
async def _cmd_cancel(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    resp = await session.cancel()
    if resp:
        host.output(Text.from_markup(f"  [{fmt.warning}]Cancelled {resp.id}[/{fmt.warning}]"))


@_cmd("/quit", "Exit")
async def _cmd_quit(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    raise EOFError


COMMANDS["/exit"] = COMMANDS["/quit"]


def _render_history_item(
    item: dict[str, Any],
    host: Any,
    fmt: Any = None,
) -> None:
    """Render a single conversation history item in consistent style."""
    from rich.text import Text

    if fmt is None:
        fmt = RichBlockFormatter()
    itype = item.get("type", "")
    if itype == "message":
        role = item.get("role", "")
        content = item.get("content", [])
        text_parts = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("input_text", "output_text"):
                    text_parts.append(str(b.get("text", "")))
        text = " ".join(text_parts)
        if role == "user":
            host.output(fmt.user_message(text))
        elif role == "assistant":
            model = item.get("model", "")
            host.output(Text.from_markup(f" [{fmt.assistant}]◆ {model}[/{fmt.assistant}]"))
            # Show the text with proper indentation.
            preview = text[:300]
            if len(text) > 300:
                preview += "…"
            for line in preview.split("\n"):
                if line.strip():
                    host.output(Text.from_markup(f"   [{fmt.muted}]{line}[/{fmt.muted}]"))
    elif itype == "function_call":
        name = item.get("name", "?")
        host.output(Text.from_markup(f"   [{fmt.accent}]⏵ {name}[/{fmt.accent}]"))


async def handle_slash_command(
    line: str,
    session: Any,
    client: AgentPlaneClient,
    host: Any,
    fmt: Any,
) -> None:
    """Dispatch a slash command from the registry."""
    from rich.text import Text

    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    entry = COMMANDS.get(cmd)
    if entry:
        _, handler = entry
        await handler(arg, session, client, host, fmt)
    else:
        host.output(
            Text.from_markup(
                f"  [{fmt.muted}]Unknown command: {cmd} · /help for list[/{fmt.muted}]"
            )
        )
