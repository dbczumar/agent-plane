#!/usr/bin/env python
"""Interactive chat shell for agent-plane.

Usage:
    python scripts/chat.py <OPENAI_API_KEY>
    python scripts/chat.py $(cat /tmp/mykey)

Starts a temporary server, deploys an example agent, and opens an
interactive chat loop with streaming responses.
"""

from __future__ import annotations

import io
import json
import os
import readline  # noqa: F401 — enables arrow-key editing in input()
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field

import httpx

# ── Configuration ─────────────────────────────────────

PORT = 18400
BASE_URL = f"http://127.0.0.1:{PORT}"
AGENT_NAME = "chat-agent"
MODEL = "o4-mini"
CONNECTION: dict[str, str] | None = None
SYSTEM_INSTRUCTIONS = "You are a helpful assistant. Be concise but thorough."


# ── Helpers ───────────────────────────────────────────


def _add_tar_file(
    tf: tarfile.TarFile,
    name: str,
    data: bytes,
) -> None:
    """
    Add a file to the tarball.

    :param tf: Open tarfile to add the file to.
    :param name: Path within the tarball, e.g.
        ``"skills/code-review/SKILL.md"``.
    :param data: Raw file content bytes.
    """
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def make_agent_bundle() -> bytes:
    """
    Create an agent tarball with system instructions and
    a sample skill.
    """
    # Build connection block if configured.
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
{connection_lines}instructions: |
  {SYSTEM_INSTRUCTIONS}
""".encode()

    # Sample skill: code-review
    skill_md = b"""\
---
name: code-review
description: >-
  Reviews code snippets for quality, style, and correctness.
---

You are now in **code review mode**.

When the user provides code, analyze it for:
1. **Correctness** - logic errors, off-by-one, null handling
2. **Style** - naming, formatting, idiomatic patterns
3. **Performance** - unnecessary allocations, O(n^2)
4. **Security** - injection, unsanitized input, secrets

Provide specific, actionable feedback with line references.
Keep suggestions concise. Praise what's done well.
"""

    # Sample reference file for the skill
    style_guide = b"""\
# Style Guide Reference

## Naming
- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE

## Line Length
- Maximum 100 characters

## Imports
- Standard library first, then third-party, then local
- Alphabetical within each group
"""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_tar_file(tf, "config.yaml", config_yaml)
        _add_tar_file(
            tf,
            "skills/code-review/SKILL.md",
            skill_md,
        )
        _add_tar_file(
            tf,
            "skills/code-review/references/style-guide.md",
            style_guide,
        )
    return buf.getvalue()


def wait_for_server(proc: subprocess.Popen[bytes], timeout: float = 15.0) -> None:
    """Poll until the server responds or timeout."""
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
    """Register the chat agent. Returns agent ID."""
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


def _iter_sse_lines(resp: httpx.Response):
    """
    Yield SSE lines from a streaming response in real-time.

    httpx's iter_lines() buffers internally, which delays token-level
    events by up to a full chunk. iter_bytes() yields data as soon as
    it arrives from the socket, so we split lines manually.
    """
    buf = ""
    for chunk in resp.iter_bytes():
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line.rstrip("\r")
    if buf:
        yield buf.rstrip("\r")


@dataclass
class SSEParser:
    """
    Mutable state for incremental SSE parsing.

    :param on_response_id: Optional callback invoked with the
        response ID as soon as the ``response.created`` event
        arrives. Used by the chat loop to enable steering before
        the stream finishes.
    """

    current_event: str | None = field(default=None)
    current_data: str | None = field(default=None)
    response_id: str | None = field(default=None)
    # Callable[[str], None] | None — but we avoid importing
    # Callable just for this optional script-level callback.
    on_response_id: object = field(default=None)

    def feed(self, line: str) -> None:
        """Process one SSE line, dispatching on blank-line boundaries."""
        if line.startswith("event: "):
            self.current_event = line[7:]
        elif line.startswith("data: "):
            self.current_data = line[6:]
        elif line == "":
            if self.current_event is not None and self.current_data is not None:
                _handle_event(self.current_event, self.current_data)
                if self.response_id is None and self.current_event == "response.created":
                    payload = json.loads(self.current_data)
                    self.response_id = payload["response"]["id"]
                    if self.on_response_id is not None:
                        self.on_response_id(self.response_id)
            self.current_event = None
            self.current_data = None


def stream_response(
    client: httpx.Client,
    user_input: str,
    previous_response_id: str | None,
    on_response_id: object = None,
) -> str | None:
    """
    Send a message and stream the response.

    :param client: HTTP client.
    :param user_input: The user's message text.
    :param previous_response_id: ID of the previous response
        for conversation continuity.
    :param on_response_id: Optional callback invoked with the
        response ID as soon as ``response.created`` arrives.
    :returns: The response ID, or ``None`` on failure.
    """
    body: dict[str, object] = {
        "model": AGENT_NAME,
        "input": user_input,
        "stream": True,
    }
    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id

    parser = SSEParser(on_response_id=on_response_id)

    try:
        with client.stream(
            "POST",
            f"{BASE_URL}/v1/responses",
            json=body,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in _iter_sse_lines(resp):
                parser.feed(line)
    except httpx.RemoteProtocolError:
        # Server dropped the connection mid-stream (e.g. race in
        # workflow completion). The output text was already printed;
        # we just missed the terminal event. Continue the conversation.
        if parser.response_id is None:
            raise

    return parser.response_id


def send_steering_message(
    client: httpx.Client,
    text: str,
    previous_response_id: str,
) -> None:
    """
    Deliver a steering message to a running workflow.

    The server's ``_attempt_steering`` path checks that the
    previous response is still active and calls ``try_deliver``
    to inject the message into the workflow's inbox. The
    workflow picks it up on its next ``close_inbox`` check and
    makes another LLM call with the steered context.

    :param client: HTTP client.
    :param text: The user's steering text.
    :param previous_response_id: ID of the response currently
        being streamed, e.g. ``"resp_abc123"``.
    """
    body: dict[str, object] = {
        "model": AGENT_NAME,
        "input": text,
        "previous_response_id": previous_response_id,
        # Non-streaming background request — we don't need
        # to read the response body, just deliver the message.
        "stream": False,
        "background": True,
    }
    resp = client.post(
        f"{BASE_URL}/v1/responses",
        json=body,
        timeout=10.0,
    )
    resp.raise_for_status()


# Tracks whether text deltas were received for the current message,
# so we don't double-print text from response.output_item.done.
_had_text_deltas = False

# Tracks whether we've printed the section header for the current
# reasoning block so we only print it once per response.
_reasoning_text_started = False
_reasoning_summary_started = False


def _handle_event(event_type: str, data: str) -> None:
    """Process a single SSE event and print to terminal."""
    global _had_text_deltas, _reasoning_text_started, _reasoning_summary_started

    if event_type == "response.reasoning.started":
        # Reasoning began but content may be encrypted (org not verified).
        # Show a dim indicator so it's clear the model is thinking.
        sys.stdout.write("\x1b[2;36m[thinking...]\x1b[0m\n")
        sys.stdout.flush()
    elif event_type == "response.reasoning_text.delta":
        payload = json.loads(data)
        if not _reasoning_text_started:
            # Cyan header to make the reasoning block unmistakable
            sys.stdout.write("\x1b[36m── reasoning ──────────────────────────────────\x1b[0m\n")
            _reasoning_text_started = True
        # Dim cyan text for reasoning content
        sys.stdout.write(f"\x1b[2;36m{payload['delta']}\x1b[0m")
        sys.stdout.flush()
    elif event_type == "response.reasoning_summary_text.delta":
        payload = json.loads(data)
        if not _reasoning_summary_started:
            # Yellow header for the summary block
            sys.stdout.write("\x1b[33m── reasoning summary ──────────────────────────\x1b[0m\n")
            _reasoning_summary_started = True
        # Yellow italic text for the summary
        sys.stdout.write(f"\x1b[3;33m{payload['delta']}\x1b[0m")
        sys.stdout.flush()
    elif event_type == "response.output_text.delta":
        # Real-time text deltas — print each token immediately
        _had_text_deltas = True
        payload = json.loads(data)
        sys.stdout.write(payload["delta"])
        sys.stdout.flush()
    elif event_type == "response.output_item.done":
        payload = json.loads(data)
        item = payload["item"]
        if item.get("type") == "message":
            if not _had_text_deltas:
                # Fallback: no deltas received, print full text at once
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        sys.stdout.write(block["text"])
                        sys.stdout.flush()
            # Print closing separator if a reasoning block was open
            if _reasoning_text_started or _reasoning_summary_started:
                sys.stdout.write("\x1b[2m── answer ─────────────────────────────────────\x1b[0m\n")
                sys.stdout.flush()
            print()
            _had_text_deltas = False
            _reasoning_text_started = False
            _reasoning_summary_started = False
        elif item.get("type") == "function_call":
            name = item.get("name", "?")
            args = item.get("arguments", "")
            # Green header + call details
            sys.stdout.write(
                f"\x1b[32m── tool call ──────────────────────────────────\x1b[0m\n"
                f"  \x1b[32m{name}({args})\x1b[0m\n"
            )
            sys.stdout.flush()
        elif item.get("type") == "function_call_output":
            output = item.get("output", "")
            display = output[:300] + "..." if len(output) > 300 else output
            # Dim green header + result
            sys.stdout.write(
                f"\x1b[2;32m── tool result ────────────────────────────────\x1b[0m\n"
                f"  \x1b[2;32m{display}\x1b[0m\n"
                f"\x1b[2;32m───────────────────────────────────────────────\x1b[0m\n"
            )
            sys.stdout.flush()


# ── Main ──────────────────────────────────────────────


@dataclass
class _StreamState:
    """
    Shared state between the streaming thread and the input loop.

    :param streaming: True while the background thread is
        actively reading SSE events.
    :param response_id: Set by the streaming thread once the
        ``response.created`` event arrives. The input loop reads
        this to send steering messages.
    :param error: Set if the streaming thread encounters an error.
    """

    streaming: bool = field(default=False)
    response_id: str | None = field(default=None)
    error: str | None = field(default=None)
    done: threading.Event = field(default_factory=threading.Event)


def _stream_in_background(
    client: httpx.Client,
    user_input: str,
    previous_response_id: str | None,
    state: _StreamState,
) -> None:
    """
    Run ``stream_response`` in a thread, updating shared state.

    :param client: HTTP client.
    :param user_input: The user's message text.
    :param previous_response_id: ID of the previous response
        for conversation continuity.
    :param state: Shared state object for cross-thread
        communication.
    """
    state.error = None
    try:
        rid = stream_response(
            client,
            user_input,
            previous_response_id,
            # Callback fires on the streaming thread as soon as
            # response.created arrives, making state.response_id
            # available for steering before the stream finishes.
            on_response_id=lambda rid: setattr(state, "response_id", rid),
        )
        state.response_id = rid
    except httpx.HTTPStatusError as exc:
        state.error = f"Error {exc.response.status_code}: {exc.response.text}"
    except httpx.ConnectError:
        state.error = "Server connection lost"
    except httpx.RemoteProtocolError:
        state.error = "Server dropped connection"
    finally:
        state.streaming = False
        state.done.set()


def _run_chat_loop(client: httpx.Client) -> None:
    """
    Interactive chat loop with steering support.

    Streaming runs in a background thread so the main thread
    can accept ``input()`` at any time. If the user types while
    the assistant is streaming, the message is delivered as a
    steering request via ``try_deliver``. If the stream has
    already finished, the message starts a new turn.

    :param client: HTTP client connected to the server.
    """
    previous_response_id: str | None = None
    state = _StreamState()

    while True:
        # If a stream just finished, pick up its response_id
        # and show any error before prompting.
        if state.done.is_set() and not state.streaming:
            if state.error:
                print(f"\n[{state.error}]")
                if state.error == "Server connection lost":
                    break
            if state.response_id is not None:
                previous_response_id = state.response_id
            state.done.clear()

        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Steering: user typed while assistant is streaming ──
        if state.streaming and state.response_id is not None:
            try:
                send_steering_message(
                    client,
                    user_input,
                    state.response_id,
                )
                sys.stdout.write(f"\x1b[2m[steered: {user_input[:60]}]\x1b[0m\n")
                sys.stdout.flush()
            except httpx.HTTPStatusError as exc:
                print(f"\n[Steering failed: {exc.response.status_code}]")
            continue

        # ── Normal: wait for any prior stream, then start new ──
        if state.streaming:
            state.done.wait(timeout=120)
            if state.response_id is not None:
                previous_response_id = state.response_id

        state.done.clear()
        # Set streaming before starting the thread to avoid a
        # race where the main loop re-enters input() and checks
        # state.streaming before the thread has set it. Clear
        # response_id so steering checks don't use a stale ID
        # from the previous turn.
        state.streaming = True
        state.response_id = None

        print()
        sys.stdout.write("assistant> ")
        sys.stdout.flush()

        thread = threading.Thread(
            target=_stream_in_background,
            args=(client, user_input, previous_response_id, state),
            daemon=True,
        )
        thread.start()


def main() -> None:
    """
    Entry point: start server, deploy agent, run chat loop.
    """
    global CONNECTION
    if len(sys.argv) < 2:
        print("Usage: python scripts/chat.py <API_KEY>")
        print("       python scripts/chat.py $(cat /tmp/mykey)")
        sys.exit(1)

    api_key = sys.argv[1].strip()
    CONNECTION = {"api_key": api_key}

    with tempfile.TemporaryDirectory() as tmpdir:
        db_uri = f"sqlite:///{tmpdir}/chat.db"
        art_loc = f"{tmpdir}/artifacts"

        print("Starting server...")
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

            print(f"Agent deployed: {AGENT_NAME} ({agent_id})")
            print(f"Model: {MODEL}")
            print()
            print("Type your message and press Enter.")
            print("While the assistant is responding, type to steer it.")
            print("Ctrl-C to quit.")
            print("-" * 50)

            _run_chat_loop(client)

        finally:
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
