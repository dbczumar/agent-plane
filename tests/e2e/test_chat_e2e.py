"""E2E test for ``ap chat`` — local mode with archer.

Verifies that ``ap chat ./agent-dir/`` starts a server, opens the
REPL, and the agent responds. Since the REPL is interactive, we
test by directly calling the local mode components rather than
launching the full CLI.

Usage::

    pytest tests/e2e/test_chat_e2e.py \
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.conftest import find_free_port, wait_for_server

_ARCHER_DIR = Path(__file__).resolve().parents[2] / "examples" / "agents" / "archer"


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def test_chat_local_starts_server_and_agent_responds(
    llm_api_key: str,
) -> None:
    """
    ``ap chat ./agent-dir/`` starts a local server with the agent
    and the agent can respond to messages.

    Tests the server startup and agent registration path used by
    ``ap chat`` in local mode. Since the REPL itself is interactive,
    we verify the underlying server works by sending a direct HTTP
    request.

    **What breaks if this fails:**
    - _start_local_server broken → server doesn't boot.
    - Agent bundle not registered → 404 on responses.
    - Agent config invalid → 500 on responses.
    """
    from agent_plane.chat import (
        _start_local_server,
        _stop_server,
        _wait_for_server,
    )

    # Ensure the API key is in the environment for the server subprocess.
    os.environ["OPENAI_API_KEY"] = llm_api_key

    port = find_free_port()
    server_proc = _start_local_server(_ARCHER_DIR, port)

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(port)

        # Verify agent is registered.
        agents_resp = httpx.get(f"{base_url}/api/agents", timeout=10.0)
        agents_resp.raise_for_status()
        agents = agents_resp.json()["data"]
        assert len(agents) > 0, "No agents registered after server start."

        agent_name = agents[0]["name"]
        assert agent_name == "archer", f"Expected archer agent, got {agent_name!r}."

        # Send a message and verify the agent responds.
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": agent_name,
                "input": "Say hello briefly.",
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        body = resp.json()

        assert body["status"] == "completed", (
            f"Status: {body['status']!r}. Output: {body.get('output', [])}"
        )

        text = _extract_all_text(body)
        assert len(text) > 0, "Agent produced no text output."

    finally:
        _stop_server(server_proc)


def test_chat_remote_pick_agent(
    llm_api_key: str,
) -> None:
    """
    ``ap chat http://server`` can list and identify agents on a server.

    Tests the remote mode's agent discovery by starting a server with
    archer and verifying ``_pick_agent`` finds it.

    **What breaks if this fails:**
    - _pick_agent can't parse /api/agents response.
    - Agent name extraction broken.
    """
    from agent_plane.chat import _pick_agent, _start_local_server, _stop_server

    os.environ["OPENAI_API_KEY"] = llm_api_key

    port = find_free_port()
    server_proc = _start_local_server(_ARCHER_DIR, port)
    base_url = f"http://127.0.0.1:{port}"

    try:
        wait_for_server(base_url)

        # _pick_agent auto-selects when there's only one agent.
        agent_name = _pick_agent(base_url)
        assert agent_name == "archer", (
            f"Expected _pick_agent to return 'archer', got {agent_name!r}."
        )
    finally:
        _stop_server(server_proc)
