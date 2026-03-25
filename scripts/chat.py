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
import time
from dataclasses import dataclass, field

import httpx

# ── Configuration ─────────────────────────────────────

PORT = 18400
BASE_URL = f"http://127.0.0.1:{PORT}"
AGENT_NAME = "chat-agent"
MODEL = "gpt-5.4"
# MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTIONS = "You are a helpful assistant. Be concise but thorough."


# ── Helpers ───────────────────────────────────────────


def make_agent_bundle() -> bytes:
    """Create an agent tarball with system instructions."""
    config_yaml = f"""\
spec_version: 1
name: {AGENT_NAME}
llm:
  model: {MODEL}
  reasoning_effort: high
instructions: |
  {SYSTEM_INSTRUCTIONS}
""".encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_yaml)
        tf.addfile(info, io.BytesIO(config_yaml))
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
    """Mutable state for incremental SSE parsing."""

    current_event: str | None = field(default=None)
    current_data: str | None = field(default=None)
    response_id: str | None = field(default=None)

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
            self.current_event = None
            self.current_data = None


def stream_response(
    client: httpx.Client,
    user_input: str,
    previous_response_id: str | None,
) -> str | None:
    """
    Send a message and stream the response.
    Returns the response ID for conversation continuity.
    """
    body: dict[str, object] = {
        "model": AGENT_NAME,
        "input": user_input,
        "stream": True,
    }
    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id

    parser = SSEParser()

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


# Tracks whether text deltas were received for the current message,
# so we don't double-print text from response.output_item.done.
_had_text_deltas = False


def _handle_event(event_type: str, data: str) -> None:
    """Process a single SSE event and print to terminal."""
    global _had_text_deltas

    if event_type == "response.reasoning_text.delta":
        # Full reasoning tokens — dim display to distinguish from final answer
        payload = json.loads(data)
        sys.stdout.write(f"\x1b[2m{payload['delta']}\x1b[0m")
        sys.stdout.flush()
    elif event_type == "response.reasoning_summary_text.delta":
        # Reasoning summary tokens — shown before the final answer
        payload = json.loads(data)
        sys.stdout.write(f"\x1b[3m{payload['delta']}\x1b[0m")
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
            print()
            _had_text_deltas = False
        elif item.get("type") == "function_call":
            name = item.get("name", "?")
            args = item.get("arguments", "")
            print(f"  [tool call: {name}({args})]")
        elif item.get("type") == "function_call_output":
            output = item.get("output", "")
            display = output[:200] + "..." if len(output) > 200 else output
            print(f"  [tool result: {display}]")


# ── Main ──────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/chat.py <OPENAI_API_KEY>")
        print("       python scripts/chat.py $(cat /tmp/mykey)")
        sys.exit(1)

    api_key = sys.argv[1].strip()
    os.environ["OPENAI_API_KEY"] = api_key

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
            print("Type your message and press Enter. Ctrl-C to quit.")
            print("-" * 50)

            previous_response_id: str | None = None

            while True:
                try:
                    user_input = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    break

                if not user_input:
                    continue

                print()
                sys.stdout.write("assistant> ")
                sys.stdout.flush()

                try:
                    previous_response_id = stream_response(
                        client,
                        user_input,
                        previous_response_id,
                    )
                except httpx.HTTPStatusError as exc:
                    print(f"\n[Error {exc.response.status_code}: {exc.response.text}]")
                except httpx.ConnectError:
                    print("\n[Server connection lost]")
                    break

        finally:
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
