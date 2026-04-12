"""Slash command handlers for the REPL."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from agent_plane_client import AgentPlaneClient, Session


async def handle_command(
    line: str,
    console: Console,
    client: AgentPlaneClient,
    session: Session,
    state: dict,
) -> bool:
    """Handle a slash command. Returns True if handled, False otherwise."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        state["quit"] = True
        return True

    if cmd == "/new":
        session.reset()
        console.print("[dim]New conversation started.[/dim]")
        return True

    if cmd == "/cancel":
        if session.is_streaming:
            resp = await session.cancel()
            if resp is not None:
                console.print(f"[yellow]Cancelled response {resp.id}[/yellow]")
            else:
                console.print("[dim]No active response to cancel.[/dim]")
        else:
            console.print("[dim]No active response to cancel.[/dim]")
        return True

    if cmd == "/agents":
        agents = await client.agents.list()
        if not agents:
            console.print("[dim]No agents registered.[/dim]")
        else:
            table = Table(title="Agents")
            table.add_column("Name", style="bold")
            table.add_column("ID", style="dim")
            table.add_column("Description")
            for a in agents:
                table.add_row(a.name, a.id, a.description or "")
            console.print(table)
        return True

    if cmd == "/model":
        if not arg:
            console.print(f"[dim]Current model: {session.model}[/dim]")
        else:
            # Create a new session with the new model, preserving hooks.
            state["switch_model"] = arg
            console.print(f"[dim]Switched to model: {arg}[/dim]")
        return True

    if cmd == "/conversations":
        convos = await client.conversations.list(limit=20)
        if not convos:
            console.print("[dim]No conversations found.[/dim]")
        else:
            table = Table(title="Conversations")
            table.add_column("ID", style="dim")
            table.add_column("Title")
            for c in convos:
                table.add_row(c.id, c.title or "(untitled)")
            console.print(table)
        return True

    if cmd == "/resume":
        if not arg:
            console.print("[dim]Usage: /resume <conversation_id>[/dim]")
        else:
            try:
                items = await client.conversations.list_items(arg, limit=100)
                if items:
                    # Find the last response_id to continue from.
                    last_response_id = None
                    for item in reversed(items):
                        rid = item.get("response_id")
                        if isinstance(rid, str):
                            last_response_id = rid
                            break
                    if last_response_id is not None:
                        session.reset()
                        session.resume_from_response(last_response_id)
                        console.print(f"[dim]Resumed conversation {arg}[/dim]")
                    else:
                        console.print("[dim]No response ID found in conversation.[/dim]")
                else:
                    console.print("[dim]Conversation is empty.[/dim]")
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
        return True

    if cmd == "/history":
        if session.current_response_id is None:
            console.print("[dim]No active conversation.[/dim]")
        else:
            try:
                resp = await client.responses.get(session.current_response_id)
                if resp.conversation is not None:
                    items = await client.conversations.list_items(resp.conversation.id, limit=50)
                    for item in items:
                        _render_history_item(console, item)
                else:
                    console.print("[dim]No conversation found.[/dim]")
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
        return True

    if cmd == "/attach":
        if not arg:
            console.print("[dim]Usage: /attach <file_path>[/dim]")
        else:
            path = pathlib.Path(arg)
            if not path.is_file():
                console.print(f"[red]File not found: {arg}[/red]")
            else:
                state.setdefault("pending_files", []).append(str(path))
                console.print(f"  [dim]📎 attached: {path.name}[/dim]")
        return True

    return False


def _render_history_item(console: Console, item: dict) -> None:
    """Render a single conversation history item."""
    item_type = item.get("type", "")
    if item_type == "message":
        role = item.get("role", "")
        content = item.get("content", [])
        text_parts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("input_text", "output_text"):
                        text_parts.append(str(block.get("text", "")))
        text = " ".join(text_parts)[:200]
        if role == "user":
            console.print(f"  [bold cyan]you>[/bold cyan] {text}")
        elif role == "assistant":
            console.print(f"  [bold green]assistant>[/bold green] {text}")
    elif item_type == "function_call":
        name = item.get("name", "?")
        console.print(f"  [green]▸ {name}(...)[/green]")
    elif item_type == "function_call_output":
        output = str(item.get("output", ""))[:80]
        console.print(f"  [dim]  → {output}[/dim]")
