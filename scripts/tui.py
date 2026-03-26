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
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog

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
            raise RuntimeError(
                f"Server exited with code {proc.returncode}.\n{out[-3000:]}"
            )
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
    sections (reasoning, summary) are active. When the
    message completes, the full text is written to the
    RichLog as a single entry.

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


# ── Textual App ───────────────────────────────────────


class ChatApp(App[None]):
    """
    Agent-plane chat TUI.

    Provides a scrollable message log with streaming responses,
    tool call display, and reasoning block rendering. Supports
    steering (typing while the assistant is responding).
    """

    TITLE = "agent-plane"
    SUB_TITLE = f"{AGENT_NAME} ({MODEL})"

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-log {
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

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header()
        with Vertical():
            yield RichLog(
                id="chat-log",
                markup=True,
                wrap=True,
                highlight=False,
            )
            yield Input(
                id="user-input",
                placeholder="Type a message… (Enter to send, Ctrl+C to quit)",
            )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input on startup and show welcome."""
        log = self.query_one("#chat-log", RichLog)
        log.write(
            Text.from_markup(
                f"[dim]Agent [bold]{AGENT_NAME}[/bold] ready "
                f"· model [bold]{MODEL}[/bold] "
                f"· type below to chat[/dim]"
            )
        )
        log.write("")
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

        log = self.query_one("#chat-log", RichLog)

        if self._streaming and self._current_response_id is not None:
            self._send_steering(text)
            log.write(
                Text.from_markup(
                    f"[dim italic]steered: {escape(text[:80])}[/dim italic]"
                )
            )
            return

        log.write(
            Text.from_markup(f"[bold cyan]you>[/bold cyan] {escape(text)}")
        )
        self._start_stream(text)

    @work(exclusive=True, group="stream")
    async def _start_stream(self, user_input: str) -> None:
        """
        Stream a response from the agent in a background worker.

        Accumulates text deltas and writes the full response
        to the chat log when complete. Shows section headers
        for reasoning blocks and tool calls as they arrive.

        :param user_input: The user's message text.
        """
        self._streaming = True
        self._current_response_id = None
        log = self.query_one("#chat-log", RichLog)
        acc = _StreamAccumulator()

        body: dict[str, object] = {
            "model": AGENT_NAME,
            "input": user_input,
            "stream": True,
        }
        if self._previous_response_id is not None:
            body["previous_response_id"] = self._previous_response_id

        try:
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
                            elif (
                                line.startswith("data: ")
                                and current_event is not None
                            ):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    current_event = None
                                    continue
                                data = json.loads(data_str)
                                _handle_sse(
                                    log, current_event, data, acc, self
                                )
                                current_event = None
                            elif line == "":
                                current_event = None
        except httpx.HTTPStatusError as exc:
            log.write(
                Text.from_markup(
                    f"[bold red]Error {exc.response.status_code}:[/bold red]"
                    f" {escape(exc.response.text[:200])}"
                )
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            log.write(
                Text.from_markup(
                    f"[bold red]Connection error:[/bold red] {escape(str(exc))}"
                )
            )
        finally:
            self._streaming = False
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
            log = self.query_one("#chat-log", RichLog)
            log.write(Text.from_markup("[dim red]steering failed[/dim red]"))

    def action_clear_log(self) -> None:
        """Clear the chat log (Ctrl+L)."""
        self.query_one("#chat-log", RichLog).clear()

    def on_unmount(self) -> None:
        """Shut down the server on exit."""
        self._server_proc.send_signal(signal.SIGINT)
        try:
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()


# ── SSE event dispatch ────────────────────────────────


def _handle_sse(
    log: RichLog,
    event_type: str,
    data: dict[str, object],
    acc: _StreamAccumulator,
    app: ChatApp,
) -> None:
    """
    Dispatch a single SSE event to the chat log.

    Accumulates text in ``acc`` and writes completed sections
    to the RichLog. Reasoning and summary blocks are written
    when a new section begins or the message ends. Text
    deltas are accumulated and written as a single block
    when the output item completes.

    :param log: The RichLog widget to write to.
    :param event_type: SSE event name, e.g.
        ``"response.output_text.delta"``.
    :param data: Parsed JSON payload.
    :param acc: Accumulator for the current turn.
    :param app: The ChatApp instance for state updates.
    """
    if event_type == "response.created":
        _handle_response_created(data, app)

    elif event_type == "response.reasoning.started":
        if not acc.in_reasoning:
            log.write(Text.from_markup("[dim cyan]thinking…[/dim cyan]"))
            acc.in_reasoning = True

    elif event_type == "response.reasoning_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            if not acc.in_reasoning:
                acc.in_reasoning = True
            acc.reasoning += delta

    elif event_type == "response.reasoning_summary_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            if not acc.in_summary:
                acc.in_summary = True
            acc.summary += delta

    elif event_type == "response.output_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            acc.had_text = True
            acc.text += delta

    elif event_type == "response.output_item.done":
        _handle_output_item_done(log, data, acc)

    elif event_type == "response.completed":
        _handle_response_created(data, app)


def _handle_response_created(
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


def _handle_output_item_done(
    log: RichLog,
    data: dict[str, object],
    acc: _StreamAccumulator,
) -> None:
    """
    Handle a completed output item (message, tool call, tool result).

    Writes accumulated content to the log and resets the
    accumulator for the next item.

    :param log: The RichLog widget.
    :param data: The ``response.output_item.done`` payload.
    :param acc: The stream accumulator.
    """
    item = data.get("item")
    if not isinstance(item, dict):
        return

    item_type = item.get("type")

    if item_type == "message":
        _write_message(log, item, acc)
    elif item_type == "function_call":
        _write_tool_call(log, item)
    elif item_type == "function_call_output":
        _write_tool_result(log, item)


def _write_message(
    log: RichLog,
    item: dict[str, object],
    acc: _StreamAccumulator,
) -> None:
    """
    Write a completed assistant message to the log.

    Includes reasoning/summary sections if present, followed
    by the main text content.

    :param log: The RichLog widget.
    :param item: The message output item dict.
    :param acc: The stream accumulator with collected text.
    """
    # Write reasoning block if we collected any
    if acc.reasoning:
        log.write(
            Text.from_markup(
                "[cyan]── reasoning ──────────────────[/cyan]"
            )
        )
        log.write(Text(acc.reasoning, style="dim cyan"))

    # Write summary block if we collected any
    if acc.summary:
        log.write(
            Text.from_markup(
                "[yellow]── reasoning summary ──────────[/yellow]"
            )
        )
        log.write(Text(acc.summary, style="dim italic yellow"))

    # Section divider if reasoning was shown
    if acc.reasoning or acc.summary:
        log.write(
            Text.from_markup(
                "[dim]── answer ─────────────────────[/dim]"
            )
        )

    # Write the assistant's text
    if acc.had_text and acc.text:
        log.write(
            Text.from_markup(
                f"[bold green]assistant>[/bold green] {escape(acc.text)}"
            )
        )
    elif not acc.had_text:
        # Fallback: no deltas received, use full content
        content = item.get("content", [])
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "output_text"
                ):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                log.write(
                    Text.from_markup(
                        f"[bold green]assistant>[/bold green]"
                        f" {escape(''.join(parts))}"
                    )
                )

    log.write("")  # Blank line after message

    # Reset for next message in same turn (multi-output)
    acc.text = ""
    acc.reasoning = ""
    acc.summary = ""
    acc.in_reasoning = False
    acc.in_summary = False
    acc.had_text = False


def _write_tool_call(log: RichLog, item: dict[str, object]) -> None:
    """
    Write a tool call to the log.

    :param log: The RichLog widget.
    :param item: The function_call output item dict.
    """
    name = item.get("name", "?")
    args = item.get("arguments", "")
    log.write(
        Text.from_markup(
            f"[bold green]── tool call ──────────────────[/bold green]"
        )
    )
    log.write(
        Text.from_markup(
            f"  [green]{escape(str(name))}"
            f"({escape(str(args)[:200])})[/green]"
        )
    )


def _write_tool_result(log: RichLog, item: dict[str, object]) -> None:
    """
    Write a tool result to the log.

    :param log: The RichLog widget.
    :param item: The function_call_output item dict.
    """
    output = str(item.get("output", ""))
    display = output[:300]
    if len(output) > 300:
        display += "…"
    log.write(
        Text.from_markup(
            f"[dim green]── tool result ────────────────[/dim green]"
        )
    )
    log.write(
        Text.from_markup(f"  [dim green]{escape(display)}[/dim green]")
    )
    log.write(
        Text.from_markup(
            f"[dim green]───────────────────────────────[/dim green]"
        )
    )


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
