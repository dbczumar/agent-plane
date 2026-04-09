"""E2E test: agent installs dependencies from PyPI and npm in sandbox.

Verifies that the ``code_sandbox`` tool can install packages via
``pip install`` and ``npm install`` inside the per-conversation
workspace, and that the installed packages are usable by subsequent
commands within the same turn.

Uses the ``archer`` agent which has ``code_sandbox`` enabled.

Usage::

    pytest tests/e2e/test_sandbox_dependencies.py \
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _has_tool_call(body: dict[str, Any], name: str) -> bool:
    """
    Check if the response output contains a function_call with the
    given tool name.

    :param body: The terminal response body.
    :param name: Tool name to search for.
    :returns: True if found.
    """
    for item in body.get("output", []):
        if item.get("type") == "function_call" and item.get("name") == name:
            return True
    return False


def test_pip_install_and_use_package(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    The agent installs a PyPI package via ``pip install`` in the
    sandbox and uses it in a subsequent Python command.

    Uses ``cowsay`` — a tiny package with no C dependencies that
    installs in <2 seconds.

    :param http_client: HTTP client pointed at the live e2e server.
    :param archer_agent: The uploaded archer agent name.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use the code_sandbox tool to: "
                "1) pip install cowsay "
                "2) Run: python -c \"import cowsay; cowsay.cow('hello from agent-plane')\" "
                "Show me the output."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=300)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. "
        f"Error: {body.get('error')}. "
        f"The agent should complete after installing and running cowsay."
    )

    # The agent must have called code_sandbox at least once.
    assert _has_tool_call(body, "code_sandbox"), (
        "Expected at least one code_sandbox tool call. "
        "The agent may not have used the sandbox tool."
    )

    # The cowsay ASCII art must appear in the output — proves the
    # package was installed AND executed successfully. If pip fails
    # (SSL, network, etc.), the test fails — that's a broken
    # environment, not something to handle gracefully.
    text = _extract_all_text(body)
    all_output = " ".join(
        str(it.get("output", ""))
        for it in body.get("output", [])
        if it.get("type") == "function_call_output"
    )
    combined = (text + " " + all_output).lower()
    assert "hello from agent-plane" in combined, (
        f"Expected cowsay ASCII art with 'hello from agent-plane' "
        f"in output — proves pip install succeeded and the package "
        f"ran. Got: {combined[:500]}"
    )


def test_npm_install_and_use_package(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    The agent installs an npm package via ``npm install`` in the
    sandbox and uses it in a subsequent Node.js command.

    Uses ``cowsay`` (npm version) — tiny, no native deps.

    :param http_client: HTTP client pointed at the live e2e server.
    :param archer_agent: The uploaded archer agent name.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use the code_sandbox tool to: "
                "1) npm install cowsay "
                "2) Run: node -e \"const cowsay = require('cowsay'); "
                "console.log(cowsay.say({text: 'npm works'}))\" "
                "Show me the output."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=300)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. "
        f"Error: {body.get('error')}. "
        f"The agent should complete after npm install and node run."
    )

    assert _has_tool_call(body, "code_sandbox"), "Expected at least one code_sandbox tool call."

    text = _extract_all_text(body)
    all_output = " ".join(
        str(it.get("output", ""))
        for it in body.get("output", [])
        if it.get("type") == "function_call_output"
    )
    combined = (text + " " + all_output).lower()
    assert "npm works" in combined, (
        f"Expected cowsay output with 'npm works' — proves npm "
        f"install succeeded and node ran the package. "
        f"Got: {combined[:500]}"
    )
