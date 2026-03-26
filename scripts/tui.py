#!/usr/bin/env python
"""Textual-based chat TUI for agent-plane.

Usage:
    python scripts/tui.py <API_KEY>
    python scripts/tui.py $(cat /tmp/mykey)

Starts a temporary server, deploys an example agent, and opens
an interactive chat TUI with streaming responses, markdown
rendering, and steering support.
"""

from __future__ import annotations

import io
import json
import os
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
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

# ── Configuration ─────────────────────────────────────

PORT = 18400
BASE_URL = f"http://127.0.0.1:{PORT}"
AGENT_NAME = "chat-agent"
MODEL = "o4-mini"
CONNECTION: dict[str, str] | None = None
SYSTEM_INSTRUCTIONS = "You are a helpful assistant. Be concise but thorough."


# ── Agent bundle ──────────────────────────────────────


def _add_tar_file(
    tf: tarfile.TarFile,
    name: str,
    data: bytes,
) -> None:
    """
    Add a file to the tarball.

    :param tf: Open tarfile to add the file to.
    :param name: Path within the tarball.
    :param data: Raw file content bytes.
    """
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def make_agent_bundle() -> bytes:
    """
    Create an agent tarball with system instructions.

    :returns: Gzipped tar bytes ready for upload.
    """
    connection_lines = ""
    if CONNECTION:
        connection_lines = "  connection:\n"
        for k, v in CONNECTION.items():
            connection_lines += f"    {k}: {v}\n"

    config_yaml = f"""\
spec_version: 1
name: {AGENT_NAME}
llm:
  model: {MODEL}
  reasoning_effort: high
{connection_lines}instructions: |
  {SYSTEM_INSTRUCTIONS}
""".encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_tar_file(tf, "config.yaml", config_yaml)
    return buf.getvalue()


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
        margin: 0 0 0 0;
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
        margin: 0 0 0 0;
        color: $text;
    }
    """


class SystemInfo(Static):
    """
    A system info line (reasoning headers, tool calls, etc.).
    """

    DEFAULT_CSS = """
    SystemInfo {
        margin: 0;
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
    SUB_TITLE = f"{AGENT_NAME} ({MODEL})"

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
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
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
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(
            SystemInfo(
                Text.from_markup(
                    f"[dim]Agent [bold]{AGENT_NAME}[/bold] ready "
                    f"· model [bold]{MODEL}[/bold] "
                    f"· type below to chat[/dim]"
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
                SystemInfo(
                    Text.from_markup(f"[dim italic]steered: {escape(text[:80])}[/dim italic]")
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
            live.update(
                Text.from_markup(
                    f"[bold red]Error {exc.response.status_code}:"
                    f"[/bold red] {escape(exc.response.text[:200])}"
                )
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            live.update(
                Text.from_markup(f"[bold red]Connection error:[/bold red] {escape(str(exc))}")
            )
        finally:
            self._streaming = False
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

    def on_unmount(self) -> None:
        """Shut down the server on exit."""
        self._server_proc.send_signal(signal.SIGINT)
        try:
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()


# ── SSE streaming ─────────────────────────────────────


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
    delta for real-time streaming output.

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param acc: The stream accumulator.
    :param body: The request body for ``/v1/responses``.
    """
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
                        _handle_sse(app, scroll, live, current_event, data, acc)
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
) -> None:
    """
    Dispatch a single SSE event.

    Text deltas update the live widget in-place for real-time
    streaming. Reasoning, tool calls, and completion events
    mount new widgets or finalize the live widget.

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param event_type: SSE event name.
    :param data: Parsed JSON payload.
    :param acc: The stream accumulator.
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
        _handle_item_done(app, scroll, live, data, acc)

    elif event_type == "response.completed":
        _extract_response_id(data, app)


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
) -> None:
    """
    Handle a completed output item.

    For messages: finalize the live widget with the full
    content including any reasoning sections. For tool calls
    and results: mount new SystemInfo widgets.

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param data: The output_item.done payload.
    :param acc: The stream accumulator.
    """
    item = data.get("item")
    if not isinstance(item, dict):
        return

    item_type = item.get("type")
    if item_type == "message":
        _finalize_message(scroll, live, item, acc)
    elif item_type == "function_call":
        _mount_tool_call(scroll, item)
    elif item_type == "function_call_output":
        _mount_tool_result(scroll, item)


def _finalize_message(
    scroll: VerticalScroll,
    live: AssistantMessage,
    item: dict[str, object],
    acc: _StreamAccumulator,
) -> None:
    """
    Finalize the live assistant widget with completed content.

    Builds a Rich Text with optional reasoning/summary sections
    above the main text, then calls ``live.update()`` to set
    the final content.

    :param scroll: The scrollable container.
    :param live: The live assistant message widget.
    :param item: The message output item dict.
    :param acc: The stream accumulator.
    """
    parts: list[str] = []

    if acc.reasoning:
        parts.append("[cyan]── reasoning ──────────────────[/cyan]")
        parts.append(f"[dim cyan]{escape(acc.reasoning)}[/dim cyan]")

    if acc.summary:
        parts.append("[yellow]── reasoning summary ──────────[/yellow]")
        parts.append(f"[dim italic yellow]{escape(acc.summary)}[/dim italic yellow]")

    if acc.reasoning or acc.summary:
        parts.append("[dim]── answer ─────────────────────[/dim]")

    # Get the final text content
    final_text = acc.text
    if not acc.had_text:
        # Fallback: extract from the full content
        content = item.get("content", [])
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    t = block.get("text")
                    if isinstance(t, str):
                        text_parts.append(t)
            final_text = "".join(text_parts)

    parts.append(f"[bold green]assistant>[/bold green] {escape(final_text)}")

    live.update(Text.from_markup("\n".join(parts)))
    scroll.scroll_end()

    # Reset for next message in multi-output turns
    acc.text = ""
    acc.reasoning = ""
    acc.summary = ""
    acc.in_reasoning = False
    acc.in_summary = False
    acc.had_text = False


def _mount_tool_call(
    scroll: VerticalScroll,
    item: dict[str, object],
) -> None:
    """
    Mount a tool call widget in the chat scroll.

    :param scroll: The scrollable container.
    :param item: The function_call output item dict.
    """
    name = item.get("name", "?")
    args = item.get("arguments", "")
    scroll.mount(
        SystemInfo(
            Text.from_markup(
                f"[bold green]── tool call"
                f" ──────────────────[/bold green]\n"
                f"  [green]{escape(str(name))}"
                f"({escape(str(args)[:200])})[/green]"
            )
        )
    )
    scroll.scroll_end()


def _mount_tool_result(
    scroll: VerticalScroll,
    item: dict[str, object],
) -> None:
    """
    Mount a tool result widget in the chat scroll.

    :param scroll: The scrollable container.
    :param item: The function_call_output item dict.
    """
    output = str(item.get("output", ""))
    display = output[:300]
    if len(output) > 300:
        display += "…"
    scroll.mount(
        SystemInfo(
            Text.from_markup(
                f"[dim green]── tool result"
                f" ────────────────[/dim green]\n"
                f"  [dim green]{escape(display)}[/dim green]\n"
                f"[dim green]─────────────────────────"
                f"──────[/dim green]"
            )
        )
    )
    scroll.scroll_end()


# ── Entry point ───────────────────────────────────────


def main() -> None:
    """
    Start server, deploy agent, launch TUI.
    """
    global CONNECTION
    if len(sys.argv) < 2:
        print("Usage: python scripts/tui.py <API_KEY>")
        print("       python scripts/tui.py $(cat /tmp/mykey)")
        sys.exit(1)

    api_key = sys.argv[1].strip()
    CONNECTION = {"api_key": api_key}

    tmpdir = tempfile.mkdtemp(prefix="agent-plane-tui-")
    db_uri = f"sqlite:///{tmpdir}/chat.db"
    art_loc = f"{tmpdir}/artifacts"

    server_proc = subprocess.Popen(
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

    try:
        wait_for_server(server_proc)
        client = httpx.Client()
        bundle = make_agent_bundle()
        agent_id = register_agent(client, bundle)
        client.close()
    except Exception:
        server_proc.kill()
        raise

    app = ChatApp(server_proc=server_proc, agent_id=agent_id)
    app.run()


if __name__ == "__main__":
    main()
