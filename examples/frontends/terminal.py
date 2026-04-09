#!/usr/bin/env python
"""Textual-based chat TUI for agent-plane.

Usage:
    python examples/frontends/terminal.py <agent-dir-or-tarball>
    python terminal.py ../agents/archer/
    python terminal.py ../agents/coder/ --client-tools coder

Starts a temporary server, deploys the agent, and opens an
interactive chat TUI with streaming responses, markdown
rendering, and steering support.

When ``--client-tools <name>`` is provided, loads a client-side tool
set from ``examples/frontends/tool_sets/<name>.py``. Tool schemas
are passed to the server, and ``function_call`` items are executed
locally by the TUI — results are sent back as
``function_call_output`` items to continue the conversation.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from types import ModuleType

import httpx
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

# ── Configuration ─────────────────────────────────────


def _find_free_port() -> int:
    """
    Find a free TCP port by binding to port 0 and reading the
    OS-assigned port number.

    :returns: An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = _find_free_port()
BASE_URL = f"http://127.0.0.1:{PORT}"
# Set by main() after parsing the agent's config.yaml.
AGENT_NAME: str = "agent"
# Maximum number of child widgets in the chat scroll before
# old widgets are pruned to prevent Textual DOM slowdown.
_MAX_SCROLL_WIDGETS: int = 200
# Maximum characters to store in a tool-result Collapsible.
# Even collapsed, large Static widgets slow the DOM.
_MAX_TOOL_RESULT_DISPLAY: int = 4000
# Provider-native tool output types rendered in the TUI.
_NATIVE_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "computer_call",
        "image_generation_call",
        "mcp_call",
        "mcp_list_tools",
    }
)
# File extensions recognized as images for inline display.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
    }
)


# ── File upload ───────────────────────────────────────


@dataclass
class _UploadedFile:
    """
    Metadata for a file uploaded to the server.

    :param file_id: Server-assigned file identifier, e.g.
        ``"file_abc123"``.
    :param filename: Original filename, e.g. ``"photo.png"``.
    :param content_type: MIME type, e.g. ``"image/png"``.
    """

    file_id: str
    filename: str
    content_type: str | None


def _upload_file(file_path: pathlib.Path) -> _UploadedFile:
    """
    Upload a local file to the agent-plane server.

    :param file_path: Path to the file on disk.
    :returns: An :class:`_UploadedFile` with the server-assigned
        ``file_id``.
    :raises httpx.HTTPStatusError: If the upload fails.
    """
    content_type = mimetypes.guess_type(str(file_path))[0]
    with open(file_path, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/v1/files",
            files={"file": (file_path.name, f, content_type)},
            timeout=30.0,
        )
    resp.raise_for_status()
    body = resp.json()
    return _UploadedFile(
        file_id=body["id"],
        filename=file_path.name,
        content_type=content_type,
    )


def _build_content_blocks(
    text: str,
    files: list[_UploadedFile],
) -> str | list[dict[str, object]]:
    """
    Build the ``input`` payload from user text and attached files.

    Returns a plain string when there are no file attachments
    (backward compatible). Returns a list of content blocks when
    files are present.

    :param text: The user's message text.
    :param files: Uploaded file references to attach.
    :returns: A string or list of content block dicts.
    """
    if not files:
        return text

    blocks: list[dict[str, object]] = []
    if text:
        blocks.append({"type": "input_text", "text": text})
    for f in files:
        if f.content_type and f.content_type.startswith("image/"):
            blocks.append(
                {
                    "type": "input_image",
                    "file_id": f.file_id,
                }
            )
        else:
            blocks.append(
                {
                    "type": "input_file",
                    "file_id": f.file_id,
                    "filename": f.filename,
                }
            )
    return blocks


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
            # Only match the root config.yaml, not sub-agent
            # configs like agents/researcher/config.yaml.
            if member.name == "config.yaml":
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
        # Agent with same name already exists — delete the stale
        # registration and re-upload so the latest bundle (with
        # updated AGENTS.md, tools, etc.) takes effect.
        list_resp = client.get(f"{BASE_URL}/api/agents")
        for agent in list_resp.json()["data"]:
            if agent["name"] == AGENT_NAME:
                client.delete(f"{BASE_URL}/api/agents/{agent['id']}")
                break
        resp = client.post(
            f"{BASE_URL}/api/agents",
            files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
        )
    if resp.is_error:
        raise RuntimeError(f"Agent upload failed ({resp.status_code}): {resp.text}")
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
    :param finalized: Whether ``_finalize_message`` completed for this turn.
    :param pending_tool_calls: Client-side ``function_call`` items
        collected during streaming. After the stream completes,
        the TUI executes these locally and sends results back.
    """

    text: str = ""
    reasoning: str = ""
    summary: str = ""
    in_reasoning: bool = False
    in_summary: bool = False
    had_text: bool = False
    finalized: bool = False
    # Separate widget for streaming text, mounted at the bottom
    # (below tool calls). The main live widget at the top only
    # shows status like "thinking…".
    text_widget: AssistantMessage | None = None
    # Client-side function_call items to execute after the
    # stream completes. Each dict has "call_id", "name",
    # "arguments" keys.
    pending_tool_calls: list[dict[str, object]] = field(
        default_factory=list,
    )
    # call_ids of server-side tools that already have a
    # function_call_output — used to skip them in
    # pending_tool_calls.
    _completed_call_ids: set[str] = field(default_factory=set)
    # Tunneled function_calls from sub-agents needing
    # immediate execution and PATCH. Drained by
    # _run_sse_stream after each SSE event.
    _tunneled_calls: list[dict[str, object]] = field(
        default_factory=list,
    )


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
        Binding("ctrl+v", "paste_files", "Paste", show=False),
        Binding("escape", "toggle_browse", "Cancel/Browse", show=False),
    ]

    def __init__(
        self,
        server_proc: subprocess.Popen[bytes],
        agent_id: str,
        auto_send: str | None = None,
        tool_set: ModuleType | None = None,
    ) -> None:
        """
        :param server_proc: The running server subprocess.
            Terminated on app exit.
        :param agent_id: The deployed agent's ID.
        :param auto_send: If set, auto-submit this message on
            startup (for automated testing).
        :param tool_set: Optional client-side tool set module
            with ``TOOLS`` (schemas) and ``execute_tool(name, args)``
            attributes. When provided, tool schemas are sent with
            each request and ``function_call`` items are executed
            locally.
        """
        super().__init__()
        self._server_proc = server_proc
        self._agent_id = agent_id
        self._auto_send = auto_send
        self._tool_set = tool_set
        # Set after auto-send response completes, triggers screenshot + exit
        self._auto_send_done = False
        self._previous_response_id: str | None = None
        self._current_response_id: str | None = None
        self._streaming = False
        # Set True when response.completed/failed arrives from the
        # server. Reset at the start of each new stream. This is
        # the authoritative signal that the response is done.
        self._response_terminal = True
        # The live Static widget being updated during streaming.
        self._live_widget: Static | None = None
        # Files attached to the next message (drag-and-drop or paste).
        self._pending_files: list[_UploadedFile] = []

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
        inp = self.query_one("#user-input", Input)
        inp.focus()
        # Auto-send for automated testing — send message and
        # screenshot + exit after the response completes.
        if self._auto_send is not None:
            msg = self._auto_send
            self._auto_send = None
            self._auto_send_done = True
            scroll.mount(
                UserMessage(Text.from_markup(f"[bold cyan]you>[/bold cyan] {escape(msg)}"))
            )
            scroll.scroll_end()
            self._start_stream(msg)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Handle Enter key in the input box.

        If the assistant is currently streaming, deliver the
        message as a steering request. Otherwise, start a new
        conversation turn. Attached files (from drag-and-drop
        or paste) are included as content blocks.
        """
        text = event.value.strip()
        if not text and not self._pending_files:
            return
        event.input.value = ""

        scroll = self.query_one("#chat-scroll", VerticalScroll)

        # Route to steering only if the server response is still
        # in progress. _response_terminal is set by _handle_sse
        # when response.completed/failed arrives — this is the
        # authoritative signal from the server, not an async flag
        # subject to worker lifecycle timing.
        if (
            not self._response_terminal
            and self._current_response_id is not None
            and self._streaming
        ):
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

        # Collect any inline file paths (e.g. /path/to/image.png
        # pasted by terminals that convert drag-and-drop to paths).
        attached = list(self._pending_files)
        self._pending_files.clear()
        attached.extend(_extract_file_paths(text))

        # Build display label.
        display = _user_display_label(text, attached)
        scroll.mount(UserMessage(Text.from_markup(display)))
        scroll.scroll_end()

        content = _build_content_blocks(_strip_file_paths(text), attached)
        self._start_stream(content)

    @work(exclusive=True, group="stream")
    async def _start_stream(
        self,
        user_input: str | list[dict[str, object]],
    ) -> None:
        """
        Stream a response from the agent in a background worker.

        Mounts a live ``AssistantMessage`` widget and updates it
        token-by-token as text deltas arrive. When the message
        completes, the widget stays as the final rendered message.

        When a client-side tool set is active, ``function_call``
        items are collected during streaming. After the stream
        ends, tools are executed locally and results sent back
        via a new request. This loops until the agent produces
        a final text response with no pending tool calls.

        :param user_input: The user's message text, or a list
            of content block dicts when files are attached.
        """
        self._streaming = True
        self._response_terminal = False
        self._current_response_id = None
        scroll = self.query_one("#chat-scroll", VerticalScroll)

        # The input for the current iteration — starts as user
        # text, becomes function_call_output list on tool loops.
        current_input: str | list[dict[str, object]] = user_input

        try:
            await self._stream_loop(scroll, current_input)
            # Mark streaming done immediately after the loop exits
            # successfully — before the finally block's cleanup.
            # Without this, there's a window between the last
            # visible text update and the finally block where
            # user input routes to steering instead of a new turn.
            self._streaming = False
        except httpx.HTTPStatusError as exc:
            # Streaming responses must be read before accessing .text.
            try:
                await exc.response.aread()
                detail = exc.response.text[:200]
            except Exception:
                detail = str(exc)
            error_widget = _ensure_live(self, scroll, self._live_widget)
            error_widget.update(
                Text.from_markup(
                    f"[bold red]Error {exc.response.status_code}:[/bold red] {escape(detail)}"
                )
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            error_widget = _ensure_live(self, scroll, self._live_widget)
            error_widget.update(
                Text.from_markup(f"[bold red]Connection error:[/bold red] {escape(str(exc))}")
            )
        except httpx.TimeoutException as exc:
            error_widget = _ensure_live(self, scroll, self._live_widget)
            error_widget.update(
                Text.from_markup(f"[bold red]Timeout:[/bold red] {escape(str(exc))}")
            )
        except Exception as exc:
            _logger.exception("unexpected error in SSE stream")
            error_widget = _ensure_live(self, scroll, self._live_widget)
            error_widget.update(
                Text.from_markup(f"[bold red]Error:[/bold red] {escape(str(exc))}")
            )
        finally:
            self._streaming = False
            try:
                status = self._live_widget
                if status is not None:
                    status.remove()
            except Exception:
                _logger.exception("error cleaning up status widget")
            self._live_widget = None
            if self._current_response_id is not None:
                self._previous_response_id = self._current_response_id
            if self._auto_send_done:
                self._save_and_exit()

    async def _stream_loop(
        self,
        scroll: VerticalScroll,
        current_input: str | list[dict[str, object]],
    ) -> None:
        """
        Stream responses, executing client-side tools in a loop.

        Each iteration streams one server response. If the response
        contains ``function_call`` items and a tool set is active,
        tools are executed locally and results sent back as the next
        input. The loop exits when no pending tool calls remain.

        :param scroll: The scrollable chat container.
        :param current_input: Initial user text or tool results
            from a previous iteration.
        """
        while True:
            acc = _StreamAccumulator()
            _open_widget_log()
            live = AssistantMessage(
                Text.from_markup(
                    "[bold green]assistant>[/bold green] [dim]…[/dim]",
                )
            )
            self._live_widget = live
            await scroll.mount(live)
            _wlog("MOUNT", "AssistantMessage", "status: assistant> …")
            scroll.scroll_end()

            body = self._build_request_body(current_input)
            await _run_sse_stream(self, scroll, live, acc, body)

            # Update previous_response_id for tool result continuations.
            if self._current_response_id is not None:
                self._previous_response_id = self._current_response_id

            # Prune old widgets to prevent DOM slowdown in
            # long conversations with many tool calls.
            _prune_old_widgets(scroll)

            # Filter out server-side calls that already have a
            # function_call_output — only client-side calls remain.
            acc.pending_tool_calls = [
                tc
                for tc in acc.pending_tool_calls
                if tc.get("call_id") not in acc._completed_call_ids
            ]

            if not acc.pending_tool_calls or self._tool_set is None:
                # Clean up the status widget if finalization ran.
                if acc.finalized and self._live_widget is not None:
                    self._live_widget = None
                break

            # Execute tools locally and send results back.
            current_input = await self._execute_pending_tools(
                scroll,
                acc.pending_tool_calls,
            )

    def _build_request_body(
        self,
        current_input: str | list[dict[str, object]],
    ) -> dict[str, object]:
        """
        Build the request body for ``POST /v1/responses``.

        :param current_input: User text or ``function_call_output``
            items from local tool execution.
        :returns: Request body dict with model, input, stream, and
            optionally tools and previous_response_id.
        """
        body: dict[str, object] = {
            "model": AGENT_NAME,
            "input": current_input,
            "stream": True,
        }
        if self._previous_response_id is not None:
            body["previous_response_id"] = self._previous_response_id
        if self._tool_set is not None:
            body["tools"] = self._tool_set.TOOLS
        return body

    async def _execute_pending_tools(
        self,
        scroll: VerticalScroll,
        tool_calls: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """
        Execute client-side tool calls and return results.

        Each tool is executed locally via the tool set's
        ``execute_tool`` function. Results are displayed in the
        TUI as collapsible widgets and returned as
        ``function_call_output`` items for the next request.

        :param scroll: The scrollable chat container.
        :param tool_calls: List of ``function_call`` item dicts,
            each with ``call_id``, ``name``, ``arguments``.
        :returns: List of ``function_call_output`` dicts to send
            as the next request's input.
        """
        assert self._tool_set is not None
        results: list[dict[str, object]] = []
        for fc in tool_calls:
            name = str(fc.get("name", ""))
            call_id = str(fc.get("call_id", ""))
            args_str = str(fc.get("arguments", "{}"))
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}
            output = self._tool_set.execute_tool(name, arguments)
            # Display result in TUI.
            _mount_tool_result(scroll, {"output": output})
            results.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
        return results

    def _save_and_exit(self) -> None:
        """
        Save a Textual screenshot to ``/tmp/tui-textual.svg``
        and exit. Used by auto-send mode for headless verification.
        """
        self.save_screenshot("/tmp/tui-textual.svg")
        self.exit()

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
                headers=_REMOTE_AUTH_HEADERS,
                # Server may be busy executing a long tool call
                # (e.g. npm install) — use a generous timeout.
                timeout=120.0,
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

    def action_paste_files(self) -> None:
        """
        Paste file paths from the system clipboard (Ctrl+V).

        Inspects clipboard text for file paths (one per line).
        Image and document files are uploaded to the server and
        queued as attachments for the next message.
        """
        try:
            import subprocess as _sp

            result = _sp.run(["pbpaste"], capture_output=True, text=True, timeout=2)
            clipboard = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # pbpaste missing (non-macOS) or timed out — nothing to paste.
            return

        if not clipboard:
            return

        files = _parse_clipboard_paths(clipboard)
        if not files:
            # Not file paths — let Textual handle as normal text paste.
            inp = self.query_one("#user-input", Input)
            inp.insert_text_at_cursor(clipboard)
            return

        scroll = self.query_one("#chat-scroll", VerticalScroll)
        for uploaded in files:
            self._pending_files.append(uploaded)
            scroll.mount(
                SystemInfo(
                    Text.from_markup(f"[dim]📎 attached: {escape(uploaded.filename)}[/dim]")
                )
            )
        scroll.scroll_end()

    def action_toggle_browse(self) -> None:
        """
        Handle Escape key — context-sensitive.

        If the assistant is currently streaming, cancel the
        in-progress response via the cancel API. Otherwise,
        toggle between input mode and browse mode.
        """
        if not self._response_terminal and self._current_response_id is not None:
            self._cancel_response()
            return

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

    @work(group="cancel")
    async def _cancel_response(self) -> None:
        """
        Cancel the in-progress response via the server API.

        Sends ``POST /v1/responses/{id}/cancel`` and displays
        a status message. Must be async (not thread) because
        ``scroll.mount()`` requires the Textual event loop.
        The SSE stream terminates naturally when the workflow
        observes the cancellation at its next checkpoint.
        """
        response_id = self._current_response_id
        if response_id is None:
            return
        # Immediately clear streaming state so the next user input starts a
        # new turn instead of routing as a steering message.
        self._streaming = False
        # Remove the stale "assistant> …" live widget so it doesn't linger.
        if self._live_widget is not None:
            try:
                self._live_widget.remove()
            except Exception:
                pass
            self._live_widget = None
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=_REMOTE_AUTH_HEADERS) as client:
                resp = await client.post(
                    f"{BASE_URL}/v1/responses/{response_id}/cancel",
                )
                resp.raise_for_status()
            await scroll.mount(
                SystemInfo(Text.from_markup("[dim yellow]⏹ response cancelled[/dim yellow]"))
            )
        except httpx.HTTPStatusError as exc:
            await scroll.mount(
                SystemInfo(
                    Text.from_markup(
                        f"[dim red]cancel failed: {exc.response.status_code}[/dim red]"
                    )
                )
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            await scroll.mount(
                SystemInfo(
                    Text.from_markup(f"[dim red]cancel failed: {escape(str(exc))}[/dim red]")
                )
            )

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

    def on_paste(self, event: events.Paste) -> None:
        """
        Handle paste events — detect file paths from drag-and-drop.

        Terminals like iTerm2 and kitty convert drag-and-drop into
        a paste event containing file paths. If the pasted text
        contains valid file paths, they are uploaded and attached
        to the next message. Otherwise the paste is forwarded to
        the input widget as normal text.

        :param event: The paste event with the pasted text.
        """
        text = event.text.strip()
        if not text:
            return

        files = _parse_clipboard_paths(text)
        if files:
            event.prevent_default()
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            for uploaded in files:
                self._pending_files.append(uploaded)
                scroll.mount(
                    SystemInfo(
                        Text.from_markup(f"[dim]📎 attached: {escape(uploaded.filename)}[/dim]")
                    )
                )
            scroll.scroll_end()

    def on_unmount(self) -> None:
        """Shut down the server on exit."""
        self._server_proc.send_signal(signal.SIGINT)
        try:
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()


def _prune_old_widgets(scroll: VerticalScroll) -> None:
    """
    Remove old widgets when the scroll container exceeds the cap.

    Keeps the most recent widgets and removes the oldest ones
    to prevent Textual DOM slowdown in long conversations.
    Inserts a "[earlier messages pruned]" marker at the top.

    :param scroll: The scrollable chat container.
    """
    children = list(scroll.children)
    if len(children) <= _MAX_SCROLL_WIDGETS:
        return
    # Remove the oldest widgets, keeping the most recent ones.
    to_remove = len(children) - _MAX_SCROLL_WIDGETS
    for child in children[:to_remove]:
        child.remove()
    # Add a marker so the user knows messages were pruned.
    if scroll.children:
        first = scroll.children[0]
        # Only add marker if one isn't already there.
        if not (
            isinstance(first, SystemInfo) and "[earlier messages pruned]" in str(first.renderable)
        ):
            scroll.mount(
                SystemInfo(Text.from_markup("[dim italic][earlier messages pruned][/dim italic]")),
                before=first,
            )


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


# ── File path helpers ─────────────────────────────────


def _is_supported_file(path: pathlib.Path) -> bool:
    """
    Check if a file path points to an uploadable file.

    :param path: The file path to check.
    :returns: ``True`` if the file exists and has a recognized
        extension (image or common document format).
    """
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    return suffix in _IMAGE_EXTENSIONS or suffix in {
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
    }


def _upload_paths(paths: list[pathlib.Path]) -> list[_UploadedFile]:
    """
    Upload a list of file paths, skipping any that fail.

    Each path is uploaded via ``_upload_file``. Network and I/O
    errors are logged and the path is skipped — the user sees the
    attachment confirmation only for successful uploads.

    :param paths: Resolved file paths to upload, e.g.
        ``[Path("/tmp/photo.png"), Path("report.pdf")]``.
    :returns: Successfully uploaded file references.
    """
    uploaded: list[_UploadedFile] = []
    for path in paths:
        try:
            uploaded.append(_upload_file(path))
        except (httpx.HTTPError, OSError) as exc:
            _logger.warning("failed to upload %s: %s", path, exc)
    return uploaded


def _parse_clipboard_paths(clipboard: str) -> list[_UploadedFile]:
    """
    Parse clipboard text for file paths and upload each one.

    Lines that resolve to existing files with supported extensions
    are uploaded. Non-file lines are ignored.

    :param clipboard: Raw clipboard text, potentially containing
        file paths separated by newlines.
    :returns: List of uploaded file references.
    """
    paths: list[pathlib.Path] = []
    for line in clipboard.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip surrounding quotes added by some terminals on
        # drag-and-drop (e.g. '/Users/me/my file.png').
        stripped = line.strip("'\"")
        path = pathlib.Path(stripped)
        if _is_supported_file(path):
            paths.append(path)
    return _upload_paths(paths)


def _shell_tokenize(text: str) -> list[str]:
    """
    Split user input into tokens, handling shell quoting and escapes.

    Terminals convert drag-and-drop into paths with shell escaping,
    e.g. ``'/Users/me/my file.png'`` or ``/Users/me/my\\ file.png``.
    ``shlex.split`` handles both. Falls back to naive whitespace
    split if the input has unbalanced quotes.

    :param text: Raw user input text.
    :returns: List of unescaped tokens.
    """
    try:
        return shlex.split(text)
    except ValueError:
        # Unbalanced quotes — fall back to whitespace split.
        return text.split()


def _extract_file_paths(text: str) -> list[_UploadedFile]:
    """
    Extract and upload file paths from user input text.

    Recognizes absolute paths (starting with ``/``) and relative
    paths with recognized file extensions. Handles shell-quoted
    and backslash-escaped paths from terminal drag-and-drop.

    :param text: The raw user input text.
    :returns: List of uploaded file references.
    """
    paths = [
        pathlib.Path(token)
        for token in _shell_tokenize(text)
        if _is_supported_file(pathlib.Path(token))
    ]
    return _upload_paths(paths)


def _strip_file_paths(text: str) -> str:
    """
    Remove file path tokens from user input text.

    Strips tokens that were recognized as uploadable files so
    the text sent to the LLM is clean. Uses shell-aware
    tokenization to match what ``_extract_file_paths`` detects.

    :param text: The raw user input text.
    :returns: Text with file path tokens removed.
    """
    kept = [t for t in _shell_tokenize(text) if not _is_supported_file(pathlib.Path(t))]
    return " ".join(kept)


def _user_display_label(
    text: str,
    files: list[_UploadedFile],
) -> str:
    """
    Build the display label for a user message with attachments.

    :param text: The user's message text (may contain file paths).
    :param files: Uploaded file references.
    :returns: Rich markup string for the user message widget.
    """
    clean_text = _strip_file_paths(text)
    parts = ["[bold cyan]you>[/bold cyan]"]
    if clean_text:
        parts.append(f" {escape(clean_text)}")
    for f in files:
        parts.append(f" [dim]📎 {escape(f.filename)}[/dim]")
    return "".join(parts)


# ── Widget audit log ──────────────────────────────────
#
# Writes a structured trace of every widget operation to
# /tmp/tui-widgets.log so layout issues can be debugged
# without screenshots.

_widget_log: io.TextIOWrapper | None = None


def _open_widget_log() -> None:
    """
    Open the widget audit log (cleared per stream).
    """
    global _widget_log
    _widget_log = open("/tmp/tui-widgets.log", "w")  # noqa: SIM115


def _wlog(op: str, widget_type: str, content: str, before: str | None = None) -> None:
    """
    Write a widget operation to the audit log.

    :param op: Operation name, e.g. ``"MOUNT"``, ``"UPDATE"``,
        ``"REMOVE"``.
    :param widget_type: Widget class name, e.g.
        ``"AssistantMessage"``.
    :param content: First 120 chars of the widget's content.
    :param before: If the mount used ``before=``, the type/content
        of the reference widget.
    """
    if _widget_log is None:
        return
    before_str = f" before={before}" if before else ""
    _widget_log.write(f"{op} {widget_type}{before_str}: {content[:120]}\n")
    _widget_log.flush()


# ── SSE streaming ─────────────────────────────────────


_DEBUG_SSE = os.environ.get("DEBUG_SSE") == "1"
_logger = logging.getLogger("tui")


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
    # Always log SSE events to help debug issues.
    debug_file = open("/tmp/tui-sse.log", "w")  # noqa: SIM115
    # Long read timeout: tool execution can pause SSE for minutes
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, headers=_REMOTE_AUTH_HEADERS) as client:
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
                        try:
                            live = _handle_sse(
                                app,
                                scroll,
                                live,
                                current_event,
                                data,
                                acc,
                            )
                        except Exception:
                            _logger.exception("error handling SSE event %s", current_event)
                            if debug_file is not None:
                                import traceback

                                debug_file.write(f"ERROR: {traceback.format_exc()}\n")
                                debug_file.flush()
                        # Fire off tunneled calls as background
                        # tasks — execute and PATCH results back
                        # while the SSE stream continues reading.
                        for tc in acc._tunneled_calls:
                            asyncio.ensure_future(_execute_and_patch_tool_call(app, scroll, tc))
                        acc._tunneled_calls.clear()
                        current_event = None
                    elif line == "":
                        current_event = None


def _handle_sse(
    app: ChatApp,
    scroll: VerticalScroll,
    live: AssistantMessage | None,
    event_type: str,
    data: dict[str, object],
    acc: _StreamAccumulator,
) -> AssistantMessage | None:
    """
    Dispatch a single SSE event.

    Text and reasoning deltas update the live widget in-place.
    Tool calls mount at the scroll end (below the live widget).
    On finalization the live widget is removed and the final
    text is mounted at the bottom, below all tool calls.

    The *live* widget may become ``None`` after a message is
    finalized — a fresh one is created via :func:`_ensure_live`
    if a follow-up message starts (e.g. from steering).

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget, or ``None``
        after a message was finalized.
    :param event_type: SSE event name.
    :param data: Parsed JSON payload.
    :param acc: The stream accumulator.
    :returns: The active live widget (same, new, or ``None``).
    """
    if event_type == "response.created":
        _extract_response_id(data, app)

    elif event_type == "response.compaction.in_progress":
        live = _ensure_live(app, scroll, live)
        live.update(
            Text.from_markup(
                "[bold green]assistant>[/bold green]"
                " [dim magenta]compacting conversation…[/dim magenta]"
            )
        )
        scroll.scroll_end()

    elif event_type == "response.reasoning.started":
        if not acc.in_reasoning:
            live = _ensure_live(app, scroll, live)
            live.update(
                Text.from_markup(
                    "[bold green]assistant>[/bold green] [dim cyan]thinking…[/dim cyan]"
                )
            )
            acc.in_reasoning = True

    elif event_type == "response.reasoning_text.delta":
        delta = data.get("delta")
        if isinstance(delta, str):
            live = _ensure_live(app, scroll, live)
            acc.in_reasoning = True
            acc.reasoning += delta
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
            live = _ensure_live(app, scroll, live)
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
            # Stream text into a separate widget at the bottom
            # (below tool calls). No "assistant>" prefix — the
            # status widget above already shows the header.
            if acc.text_widget is None:
                acc.text_widget = AssistantMessage(Text.from_markup(escape(acc.text)))
                scroll.mount(acc.text_widget)
                _wlog("MOUNT", "AssistantMessage", f"text: {acc.text[:80]}")
            else:
                acc.text_widget.update(Text.from_markup(escape(acc.text)))
            scroll.scroll_end()

    elif event_type == "response.output_item.done":
        live = _handle_item_done(app, scroll, live, data, acc)

    elif event_type in ("response.completed", "response.failed"):
        _extract_response_id(data, app)
        # Mark the response as terminal. This is the authoritative
        # signal from the server. on_input_submitted checks this
        # to decide between steering (response still running) and
        # new turn (response done). Unlike _streaming (which depends
        # on async worker lifecycle), this is set synchronously in
        # the SSE handler with no race.
        app._response_terminal = True

    return live


def _ensure_live(
    app: ChatApp,
    scroll: VerticalScroll,
    live: AssistantMessage | None,
) -> AssistantMessage:
    """Create the live assistant widget if it does not yet exist.

    Called lazily on the first text or reasoning delta so that
    tool calls mount at the scroll end (below the user message)
    before any ``assistant>`` placeholder appears.

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The existing live widget, or ``None``.
    :returns: The existing or newly created live widget.
    """
    if live is not None:
        return live
    widget = AssistantMessage(Text.from_markup("[bold green]assistant>[/bold green] [dim]…[/dim]"))
    app._live_widget = widget
    scroll.mount(widget)
    scroll.scroll_end()
    return widget


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
    live: AssistantMessage | None,
    data: dict[str, object],
    acc: _StreamAccumulator,
) -> AssistantMessage | None:
    """
    Handle a completed output item.

    For messages: finalize the live widget with the full
    content including any reasoning sections, then mount a
    fresh widget for any subsequent message (e.g. from
    steering). For tool calls and results: mount new
    SystemInfo widgets before the live widget (if it exists)
    or at the scroll end (if not yet created).

    :param app: The ChatApp instance.
    :param scroll: The scrollable container.
    :param live: The live assistant message widget, or ``None``
        if not yet created.
    :param data: The output_item.done payload.
    :param acc: The stream accumulator.
    :returns: The active live widget (same, new, or ``None``).
    """
    item = data.get("item")
    if not isinstance(item, dict):
        return live

    item_type = item.get("type")
    if item_type == "message":
        if live is not None:
            _finalize_message(scroll, live, item, acc)
        # Download files referenced by file_citation annotations.
        _download_annotated_files(scroll, item)
        # Reset live to None — a fresh widget will be created
        # lazily by _ensure_live if a follow-up message starts
        # (e.g. from steering). This avoids a visible flash of
        # a placeholder "assistant> …" widget.
        app._live_widget = None
        return None
    elif item_type == "function_call":
        # Mount at scroll end — below the live status widget,
        # so tool calls appear as part of the assistant's turn.
        _mount_tool_call(scroll, item)
        if app._tool_set is not None:
            fc_status = item.get("status", "")
            if fc_status == "action_required":
                # Tunneled call from a sub-agent — execute
                # immediately in background and PATCH result
                # back to the server. Mark as completed so
                # post-stream filtering skips it.
                acc._completed_call_ids.add(item.get("call_id", ""))
                acc._tunneled_calls.append(item)
            else:
                # Non-tunneled client tool — batch for
                # post-stream execution.
                acc.pending_tool_calls.append(item)
    elif item_type == "function_call_output":
        _mount_tool_result(scroll, item)
        # Track server-side completions so we can filter them
        # out of pending_tool_calls after the stream ends.
        call_id = item.get("call_id", "")
        if call_id:
            acc._completed_call_ids.add(call_id)
    elif item_type in _NATIVE_TOOL_TYPES:
        _mount_native_tool(scroll, item)
    return live


async def _execute_and_patch_tool_call(
    app: ChatApp,
    scroll: VerticalScroll,
    item: dict[str, object],
) -> None:
    """
    Execute a tunneled client-side tool call and PATCH the
    result back to the server.

    Called as a background task for ``action_required``
    function_call items from sub-agents. The result is
    submitted via ``PATCH /v1/responses/{root_id}`` so the
    parked sub-agent can resume.

    :param app: The ChatApp instance (for tool set access).
    :param scroll: The scrollable container for result display.
    :param item: The function_call item dict with ``call_id``,
        ``name``, ``arguments``.
    """
    assert app._tool_set is not None
    name = str(item.get("name", ""))
    call_id = str(item.get("call_id", ""))
    args_str = str(item.get("arguments", "{}"))
    try:
        arguments = json.loads(args_str)
    except json.JSONDecodeError:
        arguments = {}

    output = app._tool_set.execute_tool(name, arguments)
    _mount_tool_result(scroll, {"output": output})

    # PATCH result back to the root response so the parked
    # sub-agent can resume.
    root_id = app._current_response_id or app._previous_response_id
    if root_id is None:
        _logger.error("No response ID for PATCH — tunneled tool call lost")
        return
    async with httpx.AsyncClient(timeout=60.0, headers=_REMOTE_AUTH_HEADERS) as client:
        patch_resp = await client.patch(
            f"{BASE_URL}/v1/responses/{root_id}",
            json={
                "tool_results": [
                    {"call_id": call_id, "output": output},
                ],
            },
        )
        if patch_resp.status_code != 200:
            _logger.error(
                "PATCH failed for call_id %s: %s",
                call_id,
                patch_resp.text[:200],
            )


def _truncate_for_display(text: str) -> str:
    """
    Truncate text to ``_MAX_TOOL_RESULT_DISPLAY`` for DOM storage.

    :param text: The raw text to truncate.
    :returns: Truncated text with a marker if it exceeded the limit.
    """
    if len(text) <= _MAX_TOOL_RESULT_DISPLAY:
        return text
    return text[:_MAX_TOOL_RESULT_DISPLAY] + "\n… [truncated]"


def _mount_reasoning_collapsibles(
    scroll: VerticalScroll,
    acc: _StreamAccumulator,
    before: AssistantMessage,
) -> None:
    """
    Mount reasoning and summary collapsibles before the text widget.

    Content is truncated to prevent large DOM nodes from slowing
    Textual rendering.

    :param scroll: The scrollable container.
    :param acc: The stream accumulator with reasoning/summary text.
    :param before: The text widget to mount collapsibles before.
    """
    from textual.widgets import Collapsible

    if acc.reasoning:
        scroll.mount(
            Collapsible(
                Static(
                    Text.from_markup(
                        f"[dim cyan]{escape(_truncate_for_display(acc.reasoning))}[/dim cyan]"
                    )
                ),
                title="reasoning",
                collapsed=True,
            ),
            before=before,
        )
    if acc.summary:
        scroll.mount(
            Collapsible(
                Static(
                    Text.from_markup(
                        "[dim italic yellow]"
                        f"{escape(_truncate_for_display(acc.summary))}"
                        "[/dim italic yellow]"
                    )
                ),
                title="reasoning summary",
                collapsed=True,
            ),
            before=before,
        )


def _finalize_message(
    scroll: VerticalScroll,
    live: AssistantMessage,
    item: dict[str, object],
    acc: _StreamAccumulator,
) -> None:
    """
    Finalize the assistant's turn.

    Keeps the status widget as the ``assistant>`` header so
    tool calls and text appear grouped under it. Updates the
    text widget (or creates one) with the final content, and
    mounts reasoning collapsibles before the text.

    Final layout::

        assistant>
          ▸ tool_call(...)
          ▶ result: ...
          ▶ reasoning summary   (collapsible)
        Final text here...

    :param scroll: The scrollable container.
    :param live: The live status widget (becomes the header).
    :param item: The message output item dict.
    :param acc: The stream accumulator.
    """

    # Keep the status widget as the "assistant>" header.
    _wlog("UPDATE", "AssistantMessage", "status → assistant>")
    live.update(Text.from_markup("[bold green]assistant>[/bold green]"))

    final_text = _extract_final_text(acc, item)
    # Determine the text widget — either already streaming
    # or a fresh one we mount now.
    text_widget = acc.text_widget
    if text_widget is None:
        text_widget = AssistantMessage(Text.from_markup(escape(final_text)))
        scroll.mount(text_widget)
        _wlog("MOUNT", "AssistantMessage", f"final_text(new): {final_text[:80]}")

    _mount_reasoning_collapsibles(scroll, acc, text_widget)

    # Update text widget with final content (no "assistant>" prefix —
    # the status widget above already shows the header).
    _wlog("UPDATE", "AssistantMessage", f"final_text: {final_text[:80]}")
    text_widget.update(Text.from_markup(escape(final_text)))
    scroll.scroll_end()

    # Mark finalized BEFORE resetting other fields so the finally
    # block knows not to remove the status widget.
    acc.finalized = True
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
    acc.text_widget = None


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
    label = f"▸ {name}({str(args)[:80]})"
    widget = SystemInfo(
        Text.from_markup(f"[green]▸ {escape(str(name))}({escape(str(args)[:120])})[/green]")
    )
    if before is not None:
        scroll.mount(widget, before=before)
        _wlog("MOUNT", "SystemInfo", f"tool_call: {label}", before="ref")
    else:
        scroll.mount(widget)
        _wlog("MOUNT", "SystemInfo", f"tool_call: {label}")
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

    # Tool output may be absent if the tool produced no stdout/stderr.
    raw_output = item.get("output")
    output = str(raw_output) if raw_output is not None else "(no output)"
    # Truncate content stored in the DOM — even collapsed,
    # large Static widgets slow Textual rendering.
    display_output = _truncate_for_display(output)
    # Show first line as the collapsible title
    first_line = output.split("\n", 1)[0][:80]
    widget = Collapsible(
        Static(Text.from_markup(f"[dim]{escape(display_output)}[/dim]")),
        title=f"result: {first_line}",
        collapsed=True,
    )
    if before is not None:
        scroll.mount(widget, before=before)
        _wlog("MOUNT", "Collapsible", f"tool_result: {first_line}", before="ref")
    else:
        scroll.mount(widget)
        _wlog("MOUNT", "Collapsible", f"tool_result: {first_line}")
    scroll.scroll_end()


def _download_annotated_files(
    scroll: VerticalScroll,
    item: dict[str, object],
) -> None:
    """
    Download files referenced by ``file_citation`` annotations on
    a message's ``output_text`` blocks.

    For each annotation with a ``file_id``, downloads via
    ``GET /v1/files/{id}/content``, saves to ``./downloads/``,
    and opens with the system viewer.

    :param scroll: The scrollable container for status messages.
    :param item: A ``message`` output item dict.
    """
    content = item.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        annotations = block.get("annotations")
        if not isinstance(annotations, list):
            continue
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get("type") != "file_citation":
                continue
            file_id = ann.get("file_id")
            filename = ann.get("filename", "download")
            if not file_id:
                continue
            _download_single_file(scroll, file_id, filename)


def _download_single_file(
    scroll: VerticalScroll,
    file_id: str,
    filename: str,
) -> None:
    """
    Download a single file by ID, save it, and open with the
    system viewer.

    :param scroll: The scrollable container for status messages.
    :param file_id: The file store ID, e.g. ``"file_abc123"``.
    :param filename: The display filename, e.g. ``"chart.png"``.
    """
    try:
        resp = httpx.get(
            f"{BASE_URL}/v1/files/{file_id}/content",
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        scroll.mount(
            SystemInfo(
                Text.from_markup(
                    f"[dim red]download failed: {escape(str(exc))}[/dim red]",
                )
            )
        )
        return

    downloads = pathlib.Path("downloads")
    downloads.mkdir(exist_ok=True)
    dest = downloads / filename
    dest.write_bytes(resp.content)
    scroll.mount(
        SystemInfo(
            Text.from_markup(
                f"[dim green]saved: {escape(str(dest))}[/dim green]",
            )
        )
    )
    scroll.scroll_end()

    import platform
    import subprocess as _sp

    if platform.system() == "Darwin":
        _sp.Popen(["open", str(dest)])
    else:
        _sp.Popen(["xdg-open", str(dest)])


def _mount_native_tool(
    scroll: VerticalScroll,
    item: dict[str, object],
) -> None:
    """
    Mount a provider-native tool call widget (e.g. web search).

    :param scroll: The scrollable container.
    :param item: The native tool output item dict, e.g.
        ``{"type": "web_search_call", "status": "completed",
        "action": {"type": "search", "query": "..."}}``.
    """
    item_type = item.get("type", "unknown")
    label = _format_native_tool_label(item_type, item)
    widget = SystemInfo(Text.from_markup(f"[cyan]▸ {escape(label)}[/cyan]"))
    scroll.mount(widget)
    _wlog("MOUNT", "SystemInfo", f"native_tool: {label}")
    scroll.scroll_end()


def _format_native_tool_label(
    item_type: str,
    item: dict[str, object],
) -> str:
    """
    Build a human-readable label for a native tool output item.

    :param item_type: The item type, e.g. ``"web_search_call"``.
    :param item: The full item dict.
    :returns: A short label string for display.
    """
    if item_type == "web_search_call":
        action = item.get("action")
        if isinstance(action, dict):
            action_type = action.get("type", "")
            if action_type == "search":
                query = action.get("query", "")
                return f"web search: {str(query)[:100]}"
            elif action_type == "open_page":
                url = action.get("url", "")
                return f"web open: {str(url)[:100]}"
            elif action_type == "find_in_page":
                return "web find in page"
        return "web search"
    # Generic label for other native tool types.
    display_name = item_type.replace("_", " ")
    return f"{display_name}"


# Auth headers for remote server connections. Populated during
# main() and read by _run_sse_stream / _send_steering / etc.
# Ideally this would be on ChatApp, but it's also needed by
# module-level functions that don't have access to the app instance.
_REMOTE_AUTH_HEADERS: dict[str, str] = {}


# ── Entry point ───────────────────────────────────────


def main() -> None:
    """
    Start server, deploy agent from directory or tarball, launch TUI.

    Two modes:

    **Local mode** (default): starts a local agent-plane server,
    deploys the agent from the given directory, launches the TUI.
    First positional arg is the agent directory path.

    **Remote mode** (``--server <url>``): connects to an existing
    remote server (e.g. a Databricks App). No local server started.
    First positional arg is the agent name (not a path). For
    Databricks Apps, opens a browser for OAuth authentication.

    Accepts ``--auto-send "message"`` to auto-submit a message on
    startup (for automated testing without user interaction).
    Accepts ``--client-tools <name>`` to load a client-side tool set.
    """
    global AGENT_NAME, BASE_URL
    auto_send: str | None = None
    tool_set: ModuleType | None = None
    remote_server: str | None = None
    args = sys.argv[1:]

    # Parse --auto-send flag.
    if "--auto-send" in args:
        idx = args.index("--auto-send")
        if idx + 1 < len(args):
            auto_send = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            print("Error: --auto-send requires a message argument")
            sys.exit(1)

    # Parse --client-tools flag.
    if "--client-tools" in args:
        idx = args.index("--client-tools")
        if idx + 1 < len(args):
            tool_set = _load_tool_set(args[idx + 1])
            args = args[:idx] + args[idx + 2 :]
        else:
            print("Error: --client-tools requires a tool set name")
            sys.exit(1)

    # Parse --server flag for connecting to a remote server.
    # Use with Databricks Apps: the TUI opens a browser for OAuth
    # login, then uses the session token for all API calls.
    if "--server" in args:
        idx = args.index("--server")
        if idx + 1 < len(args):
            remote_server = args[idx + 1].rstrip("/")
            args = args[:idx] + args[idx + 2 :]
        else:
            print("Error: --server requires a URL argument")
            sys.exit(1)

    if not args:
        _print_usage()
        sys.exit(1)

    if remote_server is not None:
        # Remote mode: connect to existing server, no local server.
        # First arg is the agent name (not a path).
        AGENT_NAME = args[0]
        BASE_URL = remote_server

        # Authenticate to the remote server. For Databricks Apps,
        # this does a browser-based OAuth flow and returns headers
        # with the session token. For plain HTTP servers, returns
        # empty headers.
        from auth import authenticate

        auth_headers = authenticate(remote_server)
        _REMOTE_AUTH_HEADERS.update(auth_headers)

        with httpx.Client(timeout=30, headers=auth_headers) as client:
            agents_resp = client.get(f"{BASE_URL}/api/agents")
            agents_resp.raise_for_status()
            agent_id = None
            for agent in agents_resp.json()["data"]:
                if agent["name"] == AGENT_NAME:
                    agent_id = agent["id"]
                    break
            if agent_id is None:
                raise RuntimeError(f"Agent '{AGENT_NAME}' not found on server {remote_server}")

        app = ChatApp(
            server_proc=None,
            agent_id=agent_id,
            auto_send=auto_send,
            tool_set=tool_set,
        )
        app.run()
    else:
        # Local mode: start server, deploy agent.
        agent_path = args[0]
        bundle = _load_agent_bundle(agent_path)
        AGENT_NAME = _extract_agent_name(bundle)

        server_proc = _start_server(agent_path)
        try:
            wait_for_server(server_proc)
            with httpx.Client() as client:
                agents_resp = client.get(f"{BASE_URL}/api/agents")
                agents_resp.raise_for_status()
                agent_id = None
                for agent in agents_resp.json()["data"]:
                    if agent["name"] == AGENT_NAME:
                        agent_id = agent["id"]
                        break
                if agent_id is None:
                    raise RuntimeError(f"Agent '{AGENT_NAME}' not found after server startup")
        except Exception:
            server_proc.kill()
            raise

        app = ChatApp(
            server_proc=server_proc,
            agent_id=agent_id,
            auto_send=auto_send,
            tool_set=tool_set,
        )
        app.run()


def _load_tool_set(name: str) -> ModuleType:
    """
    Load a client-side tool set by name.

    Adds the ``examples/frontends/`` directory to ``sys.path``
    so that ``tool_sets.<name>`` can be imported regardless of
    the caller's working directory.

    :param name: Tool set name, e.g. ``"coder"``.
    :returns: The tool set module with ``TOOLS`` and
        ``execute_tool`` attributes.
    """
    scripts_dir = str(pathlib.Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from tool_sets import get_tool_set

    return get_tool_set(name)


def _print_usage() -> None:
    """
    Print CLI usage information and exit.
    """
    print(
        "Usage: python terminal.py <agent-dir-or-tarball> [--client-tools NAME] [--auto-send MSG]"
    )
    print()
    print("  agent-dir-or-tarball  Path to an agent image directory")
    print("                        (containing config.yaml) or a")
    print("                        .tar.gz bundle.")
    print()
    print("Options:")
    print("  --client-tools NAME          Load client-side tool set (e.g. 'coder')")
    print("  --auto-send MSG       Auto-submit MSG on startup (for testing)")
    print()
    print("Examples:")
    print("  python terminal.py ../agents/archer/")
    print("  python terminal.py ../agents/coder/ --client-tools coder")
    print("  python terminal.py ../agents/archer/ --auto-send 'say hello'")


def _start_server(agent_path: str) -> subprocess.Popen[bytes]:
    """
    Launch a temporary agent-plane server with a fresh DB.

    Uses ``--agent`` to pre-register the agent at startup,
    avoiding the separate HTTP upload step and stale cache issues.

    :param agent_path: Path to the agent directory or tarball.
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
            "--agent",
            agent_path,
        ],
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    main()
