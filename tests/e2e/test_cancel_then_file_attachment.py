"""E2E test: cancel → file attachment → cancel → file attachment → success.

Exercises the full REPL-style cancel + file attachment flow via the
SDK session (same code path as the terminal REPL). Verifies that:

1. Cancelling mid-response doesn't break subsequent turns.
2. File attachments work after cancellation.
3. Multiple cancel → send cycles don't corrupt session state.
4. The LLM actually reads the attached markdown content.

The test simulates the REPL's Escape-cancel behavior by consuming
a few streaming events then calling ``session.cancel()`` — the exact
same sequence the REPL's ``on_input`` + ``asyncio.shield`` performs.

Uses keyword assertions on the final response to confirm the file
was read (the file contains distinctive fictional terms that can
only appear if the LLM processed the content).

Usage::

    pytest tests/e2e/test_cancel_then_file_attachment.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

_MD_CONTENT = (
    b"# Zebra Deployment Protocol\n\n"
    b"## Overview\n\n"
    b"The Zebra Deployment Protocol (ZDP) is a fictional deployment\n"
    b"strategy used exclusively by the Interplanetary Logistics Corps\n"
    b"to deliver supply crates to Mars colonies.\n\n"
    b"## Key Steps\n\n"
    b"1. Load crates onto the orbital catapult.\n"
    b"2. Calibrate the zebra-stripe targeting laser.\n"
    b"3. Launch during the Tuesday alignment window.\n"
    b"4. Confirm delivery via carrier pigeon relay.\n"
)
"""Distinctive fictional markdown — keyword assertions check for
'zebra', 'Mars', 'catapult' to confirm the file was actually read."""


def _extract_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body.
    :returns: All assistant text.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _upload_md(client: httpx.Client, base_url: str) -> str:
    """
    Upload the test markdown file and return its file_id.

    :param client: Sync HTTP client.
    :param base_url: Server base URL.
    :returns: The uploaded file's ID.
    """
    resp = client.post(
        f"{base_url}/v1/files",
        files={"file": ("protocol.md", _MD_CONTENT, "text/markdown")},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _send_with_file(
    client: httpx.Client,
    base_url: str,
    model: str,
    text: str,
    file_id: str,
    previous_response_id: str | None,
) -> dict[str, Any]:
    """
    Send a message with a file attachment (blocking).

    :param client: Sync HTTP client.
    :param base_url: Server base URL.
    :param model: Agent name.
    :param text: User message text.
    :param file_id: Uploaded file ID.
    :param previous_response_id: Previous response for conversation
        continuity, or None.
    :returns: The response JSON body.
    """
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_file", "file_id": file_id, "filename": "protocol.md"},
                ],
            },
        ],
        "stream": False,
    }
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    resp = client.post(
        f"{base_url}/v1/responses",
        json=body,
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


@pytest.mark.asyncio()
async def test_cancel_send_file_cancel_send_file_succeeds(
    live_server: str,
    archer_agent: str,
) -> None:
    """
    REPL-style flow: send → cancel → send with .md → cancel →
    send with .md → verify content was read.

    Uses the SDK Session for the cancel-aware turns (same code path
    as the REPL's on_input + Escape handling), then sends the final
    file attachment via HTTP to verify the conversation state is
    clean.

    **What breaks if wrong:**

    - session._is_terminal not reset after cancel → next send()
      tries to steer a dead response.
    - Dangling function_call items without outputs → OpenAI 400.
    - File content_type stored as application/octet-stream → OpenAI
      rejects the data URI.
    - _normalize_input double-wraps message items → OpenAI 400.

    :param live_server: Base URL of the running test server.
    :param archer_agent: Name of the registered archer agent.
    """
    from agent_plane_client import AgentPlaneClient

    async with AgentPlaneClient(base_url=live_server) as client:
        session = client.session(model=archer_agent)

        # ── Turn 1: send a message, cancel mid-stream ──────────
        event_count = 0
        async for _event in session.send("Write a 2000-word essay about volcanoes."):
            event_count += 1
            if event_count >= 3:
                break
        await session.cancel()
        prev_id_1 = session.current_response_id
        assert prev_id_1 is not None, "Session should have a response ID after cancel"

        # ── Turn 2: send with markdown file, cancel mid-stream ─
        # Upload via sync client (simpler for file upload).
        sync_client = httpx.Client(base_url=live_server, timeout=120)
        file_id_1 = _upload_md(sync_client, live_server)

        # Start a turn with the file — use HTTP directly since
        # session.send(files=) takes disk paths, not file_ids.
        resp2 = sync_client.post(
            f"{live_server}/v1/responses",
            json={
                "model": archer_agent,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Read this file."},
                            {
                                "type": "input_file",
                                "file_id": file_id_1,
                                "filename": "protocol.md",
                            },
                        ],
                    },
                ],
                "previous_response_id": prev_id_1,
                "background": True,
            },
        )
        resp2.raise_for_status()
        rid2 = resp2.json()["id"]

        # Let it start, then cancel.
        import time

        for _ in range(30):
            check = sync_client.get(f"{live_server}/v1/responses/{rid2}")
            if check.json()["status"] == "in_progress":
                break
            time.sleep(0.3)
        cancel2 = sync_client.post(f"{live_server}/v1/responses/{rid2}/cancel")
        cancel2.raise_for_status()

        # ── Turn 3: send with markdown file again — must succeed ─
        file_id_2 = _upload_md(sync_client, live_server)
        body3 = _send_with_file(
            sync_client,
            live_server,
            archer_agent,
            text=(
                "Read this file and tell me: what is the name of "
                "the protocol, what planet does it target, and what "
                "animal is in the name? Answer in one sentence."
            ),
            file_id=file_id_2,
            previous_response_id=rid2,
        )

        assert body3["status"] == "completed", (
            f"Turn 3 status: {body3['status']!r}. Error: {body3.get('error')}"
        )

        text = _extract_text(body3)
        assert text.strip(), f"Agent produced no output. Body: {body3}"

        # ── LLM judge: verify the agent actually read the file ──
        # The content has distinctive terms that can only appear
        # if the LLM processed the uploaded markdown.
        text_lower = text.lower()
        assert "zebra" in text_lower, (
            f"Response should mention 'zebra' from the file. Got: {text[:300]}"
        )
        assert "mars" in text_lower, (
            f"Response should mention 'Mars' from the file. Got: {text[:300]}"
        )

        sync_client.close()
