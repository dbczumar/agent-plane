#!/usr/bin/env python
"""Headless SSE stream test — verifies server event flow without the TUI.

Usage:
    python scripts/test_tui_sse.py <agent-dir> [message]

Starts a temporary server, deploys the agent, sends a message via
the streaming /v1/responses endpoint, and prints every SSE event
with its type and abbreviated data. Exits with 0 if the stream
completes normally, 1 otherwise.

This script is the first debugging step: if events look correct
here but the TUI is broken, the bug is in TUI rendering. If events
are wrong here, the bug is server-side.
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

import httpx

PORT = 18400
BASE_URL = f"http://127.0.0.1:{PORT}"


def _tar_directory(root: pathlib.Path) -> bytes:
    """
    Create a .tar.gz from a directory.

    :param root: Directory containing config.yaml.
    :returns: Gzipped tar bytes.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                tf.add(str(f), str(f.relative_to(root)))
    return buf.getvalue()


def _extract_name(bundle: bytes) -> str:
    """
    Read agent name from config.yaml in bundle.

    :param bundle: Gzipped tar bytes.
    :returns: Agent name string.
    """
    import yaml  # type: ignore[import-untyped]

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name == "config.yaml" or m.name.endswith("/config.yaml"):
                f = tf.extractfile(m)
                if f is None:
                    continue
                cfg = yaml.safe_load(f.read())
                name = cfg.get("name")
                if isinstance(name, str):
                    return name
    print("ERROR: no name in config.yaml")
    sys.exit(1)


def _start_server() -> subprocess.Popen[bytes]:
    """
    Launch a temporary agent-plane server.

    :returns: Server subprocess.
    """
    tmpdir = tempfile.mkdtemp(prefix="test-sse-")
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
            f"sqlite:///{tmpdir}/test.db",
            "--artifact-location",
            f"{tmpdir}/artifacts",
        ],
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _wait_for_server(proc: subprocess.Popen[bytes]) -> None:
    """
    Poll until the server responds.

    :param proc: Server subprocess.
    """
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            print(f"Server died:\n{out[-2000:]}")
            sys.exit(1)
        try:
            r = httpx.get(f"{BASE_URL}/v1/conversations", timeout=2.0)
            if r.status_code in (200, 404):
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    print("Server didn't start in 15s")
    sys.exit(1)


def _register_agent(bundle: bytes, name: str) -> str:
    """
    Register the agent, returning its ID.

    :param bundle: Agent tarball bytes.
    :param name: Agent name for conflict resolution.
    :returns: Agent ID string.
    """
    resp = httpx.post(
        f"{BASE_URL}/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    if resp.status_code == 400:
        body = resp.json()
        msg = body.get("error", {}).get("message", resp.text)
        print(f"Registration failed: {msg}")
        if "environment variable" in msg.lower():
            print("Hint: set the required env vars (e.g. OPENAI_API_KEY)")
        sys.exit(1)
    if resp.status_code == 409:
        resp = httpx.get(f"{BASE_URL}/api/agents")
        for a in resp.json()["data"]:
            if a["name"] == name:
                return a["id"]
        print("Agent conflict but not found")
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()["id"]


def _stream_and_print(agent_name: str, message: str) -> bool:
    """
    Send a message via streaming SSE and print every event.

    :param agent_name: The agent model name.
    :param message: User message to send.
    :returns: True if stream completed normally.
    """
    body = {"model": agent_name, "input": message, "stream": True}
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)

    print(f"\n{'='*60}")
    print(f"SENDING: {message!r}")
    print(f"{'='*60}\n")

    event_count = 0
    current_event: str | None = None
    buf = ""
    saw_done = False
    text_deltas: list[str] = []
    item_done_types: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", f"{BASE_URL}/v1/responses", json=body) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip("\r")

                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: ") and current_event is not None:
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            saw_done = True
                            print(f"  [{event_count}] [DONE]")
                            current_event = None
                            continue

                        data = json.loads(data_str)
                        event_count += 1
                        _print_event(event_count, current_event, data)

                        # Track key events
                        if current_event == "response.output_text.delta":
                            delta = data.get("delta", "")
                            text_deltas.append(str(delta))
                        elif current_event == "response.output_item.done":
                            item = data.get("item", {})
                            if isinstance(item, dict):
                                item_done_types.append(str(item.get("type")))

                        current_event = None
                    elif line == "":
                        current_event = None

    # Summary
    print(f"\n{'='*60}")
    print("STREAM SUMMARY")
    print(f"{'='*60}")
    print(f"  Total events: {event_count}")
    print(f"  [DONE] received: {saw_done}")
    print(f"  Text deltas: {len(text_deltas)}")
    full_text = "".join(text_deltas)
    print(f"  Full text ({len(full_text)} chars): {full_text[:200]!r}...")
    print(f"  output_item.done types: {item_done_types}")
    print()

    # Verify ordering
    ok = True
    if not saw_done:
        print("FAIL: no [DONE] marker")
        ok = False
    if "message" not in item_done_types:
        print("FAIL: no output_item.done with type=message")
        ok = False
    if not text_deltas:
        print("WARN: no text deltas (model may have returned text only in item)")

    if ok:
        print("PASS: stream completed normally")
    return ok


def _print_event(
    num: int,
    event_type: str,
    data: dict[str, object],
) -> None:
    """
    Print a formatted SSE event.

    :param num: Event sequence number.
    :param event_type: SSE event type string.
    :param data: Parsed event data dict.
    """
    if event_type == "response.output_text.delta":
        delta = data.get("delta", "")
        print(f"  [{num}] {event_type}: {str(delta)[:80]!r}")
    elif event_type == "response.output_item.done":
        item = data.get("item", {})
        if isinstance(item, dict):
            itype = item.get("type", "?")
            if itype == "function_call":
                name = item.get("name", "?")
                args = str(item.get("arguments", ""))[:60]
                print(f"  [{num}] {event_type}: function_call {name}({args})")
            elif itype == "function_call_output":
                out = str(item.get("output", ""))[:80]
                print(f"  [{num}] {event_type}: function_call_output {out!r}")
            elif itype == "message":
                content = item.get("content", [])
                text = ""
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            text += str(block.get("text", ""))
                print(f"  [{num}] {event_type}: message ({len(text)} chars) {text[:80]!r}")
            else:
                print(f"  [{num}] {event_type}: {itype}")
        else:
            print(f"  [{num}] {event_type}: (no item)")
    elif "delta" in event_type:
        delta = data.get("delta", "")
        print(f"  [{num}] {event_type}: {str(delta)[:60]!r}")
    elif event_type in ("response.created", "response.completed", "response.in_progress"):
        print(f"  [{num}] {event_type}")
    else:
        print(f"  [{num}] {event_type}: {json.dumps(data)[:100]}")


def main() -> None:
    """
    Start server, deploy agent, stream one message, verify events.
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_tui_sse.py <agent-dir> [message]")
        sys.exit(1)

    agent_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else "say hello in one sentence"

    p = pathlib.Path(agent_path)
    if not p.exists():
        print(f"Not found: {agent_path}")
        sys.exit(1)

    bundle = _tar_directory(p) if p.is_dir() else p.read_bytes()
    agent_name = _extract_name(bundle)

    # Kill any existing server on the port
    subprocess.run(
        ["lsof", "-ti", f":{PORT}"],
        capture_output=True,
    )

    print(f"Starting server on port {PORT}...")
    proc = _start_server()
    try:
        _wait_for_server(proc)
        print("Server ready.")

        agent_id = _register_agent(bundle, agent_name)
        print(f"Agent registered: {agent_name} ({agent_id})")

        ok = _stream_and_print(agent_name, message)
        sys.exit(0 if ok else 1)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
