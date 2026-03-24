#!/usr/bin/env python
"""Smoke test for the agent execution loop.

Usage:
    python scripts/smoke_test.py <OPENAI_API_KEY>
    python scripts/smoke_test.py $(cat /tmp/mykey)

Starts a temporary server, registers a minimal agent, sends a request
via POST /v1/responses, and verifies a real LLM response comes back.
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

import httpx

# ── Configuration ─────────────────────────────────────

PORT = 18321
BASE_URL = f"http://127.0.0.1:{PORT}"
AGENT_NAME = "smoke-test-agent"
# gpt-4o-mini is cheap and fast — good for smoke tests
MODEL = "gpt-4o-mini"
PROMPT = "Say exactly: 'Agent loop works!' and nothing else."


# ── Helpers ───────────────────────────────────────────


def make_agent_bundle() -> bytes:
    """Create a minimal agent tarball with config.yaml."""
    config_yaml = f"""\
spec_version: 1
name: {AGENT_NAME}
llm:
  model: {MODEL}
""".encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_yaml)
        tf.addfile(info, io.BytesIO(config_yaml))
    return buf.getvalue()


def wait_for_server(
    proc: subprocess.Popen[bytes],
    timeout: float = 15.0,
) -> None:
    """Poll until the server responds or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check if the process died before we could connect
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(
                f"Server process exited with code {proc.returncode}.\n"
                f"Output:\n{out[-3000:]}"
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
    """Register the smoke test agent. Returns agent ID."""
    resp = client.post(
        f"{BASE_URL}/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
        data={"name": AGENT_NAME},
    )
    if resp.status_code == 409:
        # Already exists from a previous run — that's fine
        resp = client.get(f"{BASE_URL}/api/agents")
        for agent in resp.json()["data"]:
            if agent["name"] == AGENT_NAME:
                return agent["id"]
        raise RuntimeError("Agent exists but not found in list")
    resp.raise_for_status()
    body = resp.json()
    print(f"  Registered agent: {body['name']} ({body['id']})")
    return body["id"]


def test_non_streaming(client: httpx.Client) -> None:
    """Test non-streaming mode (stream=false)."""
    print("\n--- Non-streaming test ---")
    resp = client.post(
        f"{BASE_URL}/v1/responses",
        json={
            "model": AGENT_NAME,
            "input": PROMPT,
            "stream": False,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()

    print(f"  Status: {body['status']}")
    print(f"  Model:  {body['model']}")

    assert body["status"] == "completed", f"Expected completed, got {body['status']}"
    assert body["model"] == AGENT_NAME
    assert len(body["output"]) >= 1, "No output items"
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["role"] == "assistant"

    # Extract text from content blocks
    content = body["output"][0].get("content", [])
    text = " ".join(b.get("text", "") for b in content if b.get("type") == "output_text")
    print(f"  Output: {text!r}")
    assert len(text) > 0, "Empty assistant response"

    # Verify conversation was created
    conv_id = body["conversation"]["id"]
    items_resp = client.get(f"{BASE_URL}/v1/conversations/{conv_id}/items")
    items_resp.raise_for_status()
    items = items_resp.json()["data"]
    print(f"  Conversation items: {len(items)}")
    # Should have: user message + assistant response
    assert len(items) >= 2, f"Expected >= 2 items, got {len(items)}"

    print("  PASSED")


def test_streaming(client: httpx.Client) -> None:
    """Test streaming mode (stream=true)."""
    print("\n--- Streaming test ---")

    # Use httpx streaming to read SSE events
    events: list[dict[str, str]] = []
    with client.stream(
        "POST",
        f"{BASE_URL}/v1/responses",
        json={
            "model": AGENT_NAME,
            "input": "What is 2+2? Reply with just the number.",
            "stream": True,
        },
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        current_event: str | None = None
        current_data: str | None = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                current_data = line[6:]
            elif line == "":
                if current_event is not None and current_data is not None:
                    events.append({"event": current_event, "data": current_data})
                current_event = None
                current_data = None

    event_types = [e["event"] for e in events]
    print(f"  Events received: {len(events)}")
    print(f"  Event types: {event_types}")

    assert "response.created" in event_types, "Missing response.created"
    assert "response.in_progress" in event_types, "Missing response.in_progress"
    assert "response.completed" in event_types, "Missing response.completed"

    # Parse the completed event to check the final response
    for e in events:
        if e["event"] == "response.completed":
            body = json.loads(e["data"])
            final = body["response"]
            print(f"  Final status: {final['status']}")
            assert final["status"] == "completed"
            assert len(final["output"]) >= 1
            break

    print("  PASSED")


# ── Main ──────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_test.py <OPENAI_API_KEY>")
        print("       python scripts/smoke_test.py $(cat /tmp/mykey)")
        sys.exit(1)

    api_key = sys.argv[1].strip()
    os.environ["OPENAI_API_KEY"] = api_key

    # Use a temp directory for the database and artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        db_uri = f"sqlite:///{tmpdir}/smoke.db"
        art_loc = f"{tmpdir}/artifacts"

        print(f"Starting server on port {PORT}...")
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "agent_plane.cli", "server",
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--database-uri", db_uri,
                "--artifact-location", art_loc,
            ],
            env={**os.environ},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_for_server(server_proc)
            print("Server is up!")

            client = httpx.Client()
            bundle = make_agent_bundle()
            register_agent(client, bundle)

            test_non_streaming(client)
            test_streaming(client)

            print("\n=== ALL SMOKE TESTS PASSED ===")

        except Exception:
            # Dump server output on failure for debugging
            if server_proc.stdout:
                print("\n--- Server output ---")
                # Read whatever is available without blocking
                server_proc.send_signal(signal.SIGINT)
                try:
                    out, _ = server_proc.communicate(timeout=5)
                    print(out.decode(errors="replace")[-3000:])
                except subprocess.TimeoutExpired:
                    pass
            raise

        finally:
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
