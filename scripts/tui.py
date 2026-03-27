#!/usr/bin/env python
"""Textual-based chat TUI for agent-plane.

Usage:
    python scripts/tui.py <agent-dir-or-tarball>
    python scripts/tui.py ./my-agent/
    python scripts/tui.py ./my-agent.tar.gz

Starts a temporary server, deploys the agent, and opens an
interactive chat TUI with streaming responses, markdown
rendering, and steering support.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass

import httpx
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

# ── Configuration ─────────────────────────────────────

PORT = 18400
BASE_URL = f"http://127.0.0.1:{PORT}"
# Set by main() after parsing the agent's config.yaml.
AGENT_NAME: str = "agent"


# ── Agent bundle ──────────────────────────────────────


def _load_agent_bundle(path: str) -> bytes:
    """
    Load an agent bundle from a directory or tarball path.

    If *path* is a directory, creates a ``.tar.gz`` from its
    contents. If *path* is already a tarball, reads it as-is.

    :param path: Filesystem path to an agent directory or
        ``.tar.gz`` file.
    :returns: Gzipped tar bytes ready for upload.
    :raises SystemExit: If the path doesn't exist or has no
        ``config.yaml``.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        print(f"Error: path not found: {path}")
        sys.exit(1)

    if p.is_file():
        return p.read_bytes()

    # Directory — bundle into a tarball
    if not (p / "config.yaml").exists():
        print(f"Error: no config.yaml in {path}")
        sys.exit(1)

    return _tar_directory(p)


def _tar_directory(root: pathlib.Path) -> bytes:
    """
    Create a ``.tar.gz`` from an agent image directory.

    :param root: Path to the agent directory containing
        ``config.yaml`` and optional ``skills/``, ``tools/``,
        etc.
    :returns: Gzipped tar bytes.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                arcname = str(file_path.relative_to(root))
                tf.add(str(file_path), arcname=arcname)
    return buf.getvalue()


def _extract_agent_name(bundle: bytes) -> str:
    """
    Read the agent name from ``config.yaml`` inside a tarball.

    :param bundle: Gzipped tar bytes.
    :returns: The agent name string.
    :raises SystemExit: If ``config.yaml`` is missing or has
        no ``name`` field.
    """
    import yaml  # type: ignore[import-untyped]

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        for member in tf.getmembers():
            if member.name == "config.yaml" or member.name.endswith("/config.yaml"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                config = yaml.safe_load(f.read())
                name = config.get("name")
                if not isinstance(name, str):
                    print("Error: config.yaml missing 'name' field")
                    sys.exit(1)
                return name
    print("Error: no config.yaml found in bundle")
    sys.exit(1)


# ── Server lifecycle ──────────────────────────────────


def wait_for_server(
    proc: subprocess.Popen[bytes],
    timeout: float = 15.0,
) -> None:
    """
    Poll until the server responds or timeout.

    :param proc: The server subprocess.
    :param timeout: Max seconds to wait.
    :raises RuntimeError: If the server exits or doesn't start.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"Server exited with code {proc.returncode}.\n{out[-3000:]}")
        try:
            resp = httpx.get(f"{BASE_URL}/v1/conversations", timeout=2.0)
            if resp.status_code in (200, 404):
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start within {timeout}s")


def register_agent(client: httpx.Client, bundle: bytes) -> str:
    """
    Register the chat agent, returning its ID.

    :param client: HTTP client.
    :param bundle: Agent tarball bytes.
    :returns: The agent ID string.
    """
    resp = client.post(
        f"{BASE_URL}/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    if resp.status_code == 409:
        resp = client.get(f"{BASE_URL}/api/agents")
        for agent in resp.json()["data"]:
            if agent["name"] == AGENT_NAME:
                return agent["id"]
        raise RuntimeError("Agent exists but not found in list")
    resp.raise_for_status()
    return resp.json()["id"]


# ── Streaming state ───────────────────────────────────


@dataclass
class _StreamAccumulator:
    """
    Accumulates SSE events for a single assistant turn.

    Collects text deltas into a buffer and tracks which
    sections (reasoning, summary) are active.

    :param text: Accumulated assistant text.
    :param reasoning: Accumulated reasoning text.
    :param summary: Accumulated reasoning summary text.
    :param in_reasoning: Whether a reasoning block is open.
    :param in_summary: Whether a summary block is open.
    :param had_text: Whether any text deltas arrived.
    """

    text: str = ""
    reasoning: str = ""
    summary: str = ""
    in_reasoning: bool = False
    in_summary: bool = False
    had_text: bool = False


# ── Message widgets ───────────────────────────────────


class UserMessage(Static):
    """
    A user message in the chat log.

    Styled with cyan bold prefix.
    """

    DEFAULT_CSS = """
    UserMessage {
        margin: 1 0 0 0;
        color: $text;
    }
    """


class AssistantMessage(Static):
    """
    An assistant message in the chat log.

    Updated token-by-token during streaming via
    ``update()``, then left in place as the final message.
    """

    DEFAULT_CSS = """
    AssistantMessage {
        margin: 0 0 1 0;
        color: $text;
    }
    """


class SystemInfo(Static):
    """
    A system info line (reasoning headers, tool calls, etc.).
    """

    DEFAULT_CSS = """
    SystemInfo {
        margin: 0 0 0 2;
        color: $text-muted;
    }
    """


# ── Textual App ───────────────────────────────────────


class ChatApp(App[None]):
    """
    Agent-plane chat TUI.

    Uses a VerticalScroll container with individual Static
    widgets per message. Assistant messages are updated
    token-by-token via ``Static.update()`` during streaming.
    """

    TITLE = "agent-plane"

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
        padding: 0 1;
    }

    #user-input {
        dock: bottom;
        height: auto;
        max-height: 5;
        margin: 0 1;
    }

    Collapsible {
        margin: 0 0 0 2;
        padding: 0;
    }

    Collapsible CollapsibleTitle {
        color: $text-muted;
    }

    Collapsible:focus-within CollapsibleTitle {
        color: $accent;
    }

    Collapsible Contents {
        padding: 0 0 0 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+r", "toggle_reasoning", "Reasoning"),
        Binding("escape", "toggle_browse", "Browse", show=False),
    ]

    def __init__(
        self,
        server_proc: subprocess.Popen[bytes],
        agent_id: str,
    ) -> None:
        """
        :param server_proc: The running server subprocess.
            Terminated on app exit.
        :param agent_id: The deployed agent's ID.
        """
        super().__init__()
        self._server_proc = server_proc
        self._agent_id = agent_id
        self._previous_response_id: str | None = None
        self._current_response_id: str | None = None
        self._streaming = False
        # The live Static widget being updated during streaming.
        self._live_widget: Static | None = None

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header()
        yield VerticalScroll(id="chat-scroll")
        yield Input(
            id="user-input",
            placeholder="Type a message… (Enter to send, Ctrl+C to quit)",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input on startup and show welcome."""
        self.sub_title = AGENT_NAME
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(
            SystemInfo(
                Text.from_markup(
                    f"[dim]Agent [bold]{AGENT_NAME}[/bold] ready · type below to chat[/dim]"
                )
            )
        )
        self.query_one("#user-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Handle Enter key in the input box.

        If the assistant is currently streaming, deliver the
        message as a steering request. Otherwise, start a new
        conversation turn.
        """
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        scroll = self.query_one("#chat-scroll", VerticalScroll)

        if self._streaming and self._current_response_id is not None:
            self._send_steering(text)
            scroll.mount(
                UserMessage(
                    Text.from_markup(
                        f"[bold cyan]you>[/bold cyan] [dim italic](steering)[/dim italic]"
                        f" {escape(text)}"
                    )
                )
            )
            scroll.scroll_end()
            return

        scroll.mount(UserMessage(Text.from_markup(f"[bold cyan]you>[/bold cyan] {escape(text)}")))
        scroll.scroll_end()
        self._start_stream(text)

    @work(exclusive=True, group="stream")
    async def _start_stream(self, user_input: str) -> None:
        """
        Stream a response from the agent in a background worker.

        Mounts a live ``AssistantMessage`` widget and updates it
        token-by-token as text deltas arrive. When the message
        completes, the widget stays as the final rendered message.

        :param user_input: The user's message text.
        """
        self._streaming = True
        self._current_response_id = None
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        acc = _StreamAccumulator()

        # Mount the live assistant widget for streaming output.
        live = AssistantMessage(
            Text.from_markup("[bold green]assistant>[/bold green] [dim]…[/dim]")
        )
        self._live_widget = live
        await scroll.mount(live)
        scroll.scroll_end()

        body: dict[str, object] = {
            "model": AGENT_NAME,
            "input": user_input,
            "stream": True,
        }
        if self._previous_response_id is not None:
            body["previous_response_id"] = self._previous_response_id

        try:
            await _run_sse_stream(self, scroll, live, acc, body)
        except httpx.HTTPStatusError as exc:
            current = self._live_widget or live
            current.update(
                Text.from_markup(
                    f"[bold red]Error {exc.response.status_code}:"
                    f"[/bold red] {escape(exc.response.text[:200])}"
                )
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            current = self._live_widget or live
            current.update(
                Text.from_markup(f"[bold red]Connection error:[/bold red] {escape(str(exc))}")
            )
        finally:
            self._streaming = False
            # Remove trailing placeholder widget if it never got content
            final = self._live_widget
            if final is not None and not acc.had_text:
                final.remove()
            self._live_widget = None
            if self._current_response_id is not None:
                self._previous_response_id = self._current_response_id

    @work(thread=True, group="steer")
    def _send_steering(self, text: str) -> None:
        """
        Deliver a steering message to the active stream.

        :param text: The user's steering text.
        """
        body: dict[str, object] = {
            "model": AGENT_NAME,
            "input": text,
            "previous_response_id": self._current_response_id,
            "stream": False,
            "background": True,
        }
        try:
            resp = httpx.post(
                f"{BASE_URL}/v1/responses",
                json=body,
                timeout=10.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            scroll.mount(SystemInfo(Text.from_markup("[dim red]steering failed[/dim red]")))

    def action_clear_log(self) -> None:
        """Clear the chat log (Ctrl+L)."""
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()

    def action_toggle_reasoning(self) -> None:
        """Toggle all reasoning/summary collapsibles (Ctrl+R)."""
        from textual.widgets import Collapsible

        for c in self.query(Collapsible):
            c.collapsed = not c.collapsed

    def action_toggle_browse(self) -> None:
        """
        Toggle between input mode and browse mode (Escape).

        In browse mode, Up/Down navigates collapsibles and
        Enter toggles them. Escape returns to input mode.
        """
        from textual.widgets import Collapsible

        inp = self.query_one("#user-input", Input)
        if self.focused is inp:
            # Enter browse mode: focus the last collapsible
            collapsibles = list(self.query(Collapsible))
            if collapsibles:
                collapsibles[-1].focus()
        else:
            # Return to input mode
            inp.focus()

    def on_key(self, event: events.Key) -> None:
        """
        Handle arrow keys and Enter in browse mode.

        When a ``Collapsible`` is focused, Up/Down moves between
        collapsibles and Enter toggles the focused one. Any other
        key returns focus to the input.

        :param event: The key event.
        """
        from textual.widgets import Collapsible

        focused = self.focused
        if not isinstance(focused, Collapsible):
            return

        collapsibles = list(self.query(Collapsible))
        if not collapsibles:
            return

        if event.key == "up":
            event.prevent_default()
            idx = _index_of(collapsibles, focused)
            if idx > 0:
                collapsibles[idx - 1].focus()
                collapsibles[idx - 1].scroll_visible()
        elif event.key == "down":
            event.prevent_default()
            idx = _index_of(collapsibles, focused)
            if idx < len(collapsibles) - 1:
                collapsibles[idx + 1].focus()
                collapsibles[idx + 1].scroll_visible()
        elif event.key == "enter":
            event.prevent_default()
            focused.collapsed = not focused.collapsed
        elif event.key != "escape":
            # Any other key returns to input
            event.prevent_default()
            self.query_one("#user-input", Input).focus()

    def on_unmount(self) -> None:
        """Shut down the server on exit."""
        self._server_proc.send_signal(signal.SIGINT)
        try:
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()


def _index_of(
    widgets: list[object],
    target: object,
) -> int:
    """
    Find the index of ``target`` in ``widgets``.

    :param widgets: List of widgets to search.
    :param target: The widget to find.
    :returns: Index of target, or 0 if not found.
    """
    for i, w in enumerate(widgets):
        if w is target:
            return i
    return 0


# ── SSE streaming ─────────────────────────────────────


_DEBUG_SSE = os.environ.get("DEBUG_SSE") == "1"


async def _run_sse_stream(
    app: ChatApp,
    scroll: VerticalScroll,
    live: AssistantMessage,
    acc: _StreamAccumulator,
    body: dict[str, object],
) -> None:
    """
    Open an SSE connection and dispatch events.

    Updates the live ``AssistantMessage`` widget on each text
    delta for real-time streaming output. Set ``DEBUG_SSE=1``
    to log all event types to ``/tmp/tui-sse.log``.

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param acc: The stream accumulator.
    :param body: The request body for ``/v1/responses``.
    """
    debug_file = open("/tmp/tui-sse.log", "a") if _DEBUG_SSE else None  # noqa: SIM115
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/responses",
            json=body,
        ) as resp:
            resp.raise_for_status()
            current_event: str | None = None
            buf = ""
            async for chunk in resp.aiter_bytes():
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: ") and current_event is not None:
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            current_event = None
                            continue
                        data = json.loads(data_str)
                        if debug_file is not None:
                            debug_file.write(f"{current_event}: {data_str[:200]}\n")
                            debug_file.flush()
                        # Rebind: _handle_sse mounts a new widget after
                        # each finalized message (e.g. steering follow-up).
                        live = _handle_sse(app, scroll, live, current_event, data, acc)
                        current_event = None
                    elif line == "":
                        current_event = None


def _handle_sse(
    app: ChatApp,
    scroll: VerticalScroll,
    live: AssistantMessage,
    event_type: str,
    data: dict[str, object],
    acc: _StreamAccumulator,
) -> AssistantMessage:
    """
    Dispatch a single SSE event.

    Text deltas update the live widget in-place for real-time
    streaming. Reasoning, tool calls, and completion events
    mount new widgets or finalize the live widget.

    Returns the current live widget — may be a newly mounted
    widget if the previous message was finalized (e.g. during
    steering, where multiple messages arrive on one stream).

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param event_type: SSE event name.
    :param data: Parsed JSON payload.
    :param acc: The stream accumulator.
    :returns: The active live widget (same or new).
    """
    if event_type == "response.created":
        _extract_response_id(data, app)

    elif event_type == "response.reasoning.started":
        if not acc.in_reasoning:
            live.update(
                Text.from_markup(
                    "[bold green]assistant>[/bold green] [dim cyan]thinking…[/dim cyan]"
                )
            )
            acc.in_reasoning = True

    elif event_type == "response.reasoning_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            acc.in_reasoning = True
            acc.reasoning += delta
            # Show reasoning progress in the live widget
            live.update(
                Text.from_markup(
                    "[bold green]assistant>[/bold green]"
                    " [dim cyan]thinking…[/dim cyan]\n"
                    f"[dim cyan]{escape(acc.reasoning[-200:])}[/dim cyan]"
                )
            )
            scroll.scroll_end()

    elif event_type == "response.reasoning_summary_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            acc.in_summary = True
            acc.summary += delta
            # Stream summary tokens visibly — for models like o4-mini,
            # the summary IS the only reasoning content available.
            label = "thinking" if not acc.reasoning else "summarizing"
            live.update(
                Text.from_markup(
                    f"[bold green]assistant>[/bold green]"
                    f" [dim yellow]{label}…[/dim yellow]\n"
                    f"[dim yellow]{escape(acc.summary[-200:])}[/dim yellow]"
                )
            )
            scroll.scroll_end()

    elif event_type == "response.output_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            acc.had_text = True
            acc.text += delta
            # Update the live widget with accumulated text
            live.update(
                Text.from_markup(f"[bold green]assistant>[/bold green] {escape(acc.text)}")
            )
            scroll.scroll_end()

    elif event_type == "response.output_item.done":
        live = _handle_item_done(app, scroll, live, data, acc)

    elif event_type == "response.completed":
        _extract_response_id(data, app)

    return live


def _extract_response_id(
    data: dict[str, object],
    app: ChatApp,
) -> None:
    """
    Extract response ID from created/completed events.

    :param data: SSE event payload.
    :param app: ChatApp to update with the response ID.
    """
    resp_data = data.get("response", {})
    if isinstance(resp_data, dict):
        rid = resp_data.get("id")
        if isinstance(rid, str):
            app._current_response_id = rid


def _handle_item_done(
    app: ChatApp,
    scroll: VerticalScroll,
    live: AssistantMessage,
    data: dict[str, object],
    acc: _StreamAccumulator,
) -> AssistantMessage:
    """
    Handle a completed output item.

    For messages: finalize the live widget with the full
    content including any reasoning sections, then mount a
    fresh widget for any subsequent message (e.g. from
    steering). For tool calls and results: mount new
    SystemInfo widgets.

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param data: The output_item.done payload.
    :param acc: The stream accumulator.
    :returns: The active live widget (same or new).
    """
    item = data.get("item")
    if not isinstance(item, dict):
        return live

    item_type = item.get("type")
    if item_type == "message":
        _finalize_message(scroll, live, item, acc)
        # Mount a fresh widget for any subsequent message on
        # the same stream (e.g. steered follow-up).
        new_live = AssistantMessage(
            Text.from_markup("[bold green]assistant>[/bold green] [dim]…[/dim]")
        )
        scroll.mount(new_live)
        app._live_widget = new_live
        return new_live
    elif item_type == "function_call":
        _mount_tool_call(scroll, item, before=live)
    elif item_type == "function_call_output":
        _mount_tool_result(scroll, item, before=live)
    return live


def _finalize_message(
    scroll: VerticalScroll,
    live: AssistantMessage,
    item: dict[str, object],
    acc: _StreamAccumulator,
) -> None:
    """
    Finalize the live assistant widget with completed content.

    Mounts reasoning/summary as collapsed ``Collapsible``
    widgets above the answer, then updates the live widget
    with the final text.

    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param item: The message output item dict.
    :param acc: The stream accumulator.
    """
    from textual.widgets import Collapsible

    # Mount collapsible sections before the live answer widget
    if acc.reasoning:
        scroll.mount(
            Collapsible(
                Static(Text.from_markup(f"[dim cyan]{escape(acc.reasoning)}[/dim cyan]")),
                title="reasoning",
                collapsed=True,
            ),
            before=live,
        )
    if acc.summary:
        scroll.mount(
            Collapsible(
                Static(
                    Text.from_markup(
                        f"[dim italic yellow]{escape(acc.summary)}[/dim italic yellow]"
                    )
                ),
                title="reasoning summary",
                collapsed=True,
            ),
            before=live,
        )

    final_text = _extract_final_text(acc, item)
    live.update(Text.from_markup(f"[bold green]assistant>[/bold green] {escape(final_text)}"))
    scroll.scroll_end()

    _reset_accumulator(acc)


def _extract_final_text(
    acc: _StreamAccumulator,
    item: dict[str, object],
) -> str:
    """
    Get the final text content from the accumulator or item.

    :param acc: The stream accumulator.
    :param item: The message output item dict.
    :returns: The final text string.
    """
    if acc.had_text:
        return acc.text

    # Fallback: extract from the full content payload
    content = item.get("content", [])
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        return "".join(text_parts)
    return acc.text


def _reset_accumulator(acc: _StreamAccumulator) -> None:
    """
    Reset accumulator for the next message in multi-output turns.

    :param acc: The stream accumulator to reset.
    """
    acc.text = ""
    acc.reasoning = ""
    acc.summary = ""
    acc.in_reasoning = False
    acc.in_summary = False
    acc.had_text = False


def _mount_tool_call(
    scroll: VerticalScroll,
    item: dict[str, object],
    before: Static | None = None,
) -> None:
    """
    Mount a tool call widget in the chat scroll.

    :param scroll: The scrollable container.
    :param item: The function_call output item dict.
    :param before: If provided, mount the widget before this
        widget in the DOM (keeps tool calls above the live
        assistant widget).
    """
    name = item.get("name", "?")
    args = item.get("arguments", "")
    widget = SystemInfo(
        Text.from_markup(f"[green]▸ {escape(str(name))}({escape(str(args)[:120])})[/green]")
    )
    if before is not None:
        scroll.mount(widget, before=before)
    else:
        scroll.mount(widget)
    scroll.scroll_end()


def _mount_tool_result(
    scroll: VerticalScroll,
    item: dict[str, object],
    before: Static | None = None,
) -> None:
    """
    Mount a tool result in a collapsed ``Collapsible``.

    :param scroll: The scrollable container.
    :param item: The function_call_output item dict.
    :param before: If provided, mount the widget before this
        widget in the DOM (keeps tool results above the live
        assistant widget).
    """
    from textual.widgets import Collapsible

    output = str(item.get("output", ""))
    # Show first line as the collapsible title
    first_line = output.split("\n", 1)[0][:80]
    widget = Collapsible(
        Static(Text.from_markup(f"[dim]{escape(output)}[/dim]")),
        title=f"result: {first_line}",
        collapsed=True,
    )
    if before is not None:
        scroll.mount(widget, before=before)
    else:
        scroll.mount(widget)
    scroll.scroll_end()


# ── Entry point ───────────────────────────────────────


def main() -> None:
    """
    Start server, deploy agent from directory or tarball, launch TUI.
    """
    global AGENT_NAME
    if len(sys.argv) < 2:
        print("Usage: python scripts/tui.py <agent-dir-or-tarball>")
        print()
        print("  agent-dir-or-tarball  Path to an agent image directory")
        print("                        (containing config.yaml) or a")
        print("                        .tar.gz bundle.")
        print()
        print("Examples:")
        print("  python scripts/tui.py ./my-agent/")
        print("  python scripts/tui.py ./my-agent.tar.gz")
        sys.exit(1)

    agent_path = sys.argv[1]
    bundle = _load_agent_bundle(agent_path)
    AGENT_NAME = _extract_agent_name(bundle)

    server_proc = _start_server()
    try:
        wait_for_server(server_proc)
        with httpx.Client() as client:
            agent_id = register_agent(client, bundle)
    except Exception:
        server_proc.kill()
        raise

    app = ChatApp(server_proc=server_proc, agent_id=agent_id)
    app.run()


def _start_server() -> subprocess.Popen[bytes]:
    """
    Launch a temporary agent-plane server.

    :returns: The server subprocess.
    """
    tmpdir = tempfile.mkdtemp(prefix="agent-plane-tui-")
    db_uri = f"sqlite:///{tmpdir}/chat.db"
    art_loc = f"{tmpdir}/artifacts"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--database-uri",
            db_uri,
            "--artifact-location",
            art_loc,
        ],
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    main()
