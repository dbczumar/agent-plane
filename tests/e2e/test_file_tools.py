"""E2E test: list_files and download_file tools.

Verifies the full round-trip: agent creates a file with
code_sandbox, uploads it with upload_file, then uses list_files
to find it and download_file to retrieve it.

Usage::

    pytest tests/e2e/test_file_tools.py \
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all assistant output_text blocks.

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


def _has_tool_call(body: dict[str, Any], name: str) -> bool:
    """
    Check if a function_call with the given name exists in output.

    :param body: The terminal response body.
    :param name: Tool name to find.
    :returns: True if found.
    """
    return any(
        i.get("type") == "function_call" and i.get("name") == name for i in body.get("output", [])
    )


def test_list_files_finds_uploaded_file(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Agent uploads a file, then list_files finds it.

    :param http_client: HTTP client pointed at the live server.
    :param archer_agent: The registered archer agent name.
    """
    # Turn 1: create and upload
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use code_sandbox to create a file called "
                "test_data.txt containing 'Hello from agent-plane'. "
                "Then upload it with upload_file."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    body1 = poll_until_terminal(http_client, rid1, timeout=180)
    assert body1["status"] == "completed", f"Turn 1 failed: {body1.get('error')}"

    # Turn 2: list files only
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use the list_files tool to show me all uploaded "
                "files. Only use list_files, nothing else."
            ),
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    body2 = poll_until_terminal(http_client, rid2, timeout=180)
    assert body2["status"] == "completed", f"Turn 2 failed: {body2.get('error')}"

    assert _has_tool_call(body2, "list_files"), "Agent didn't call list_files"
    text = _extract_all_text(body2)
    assert "test_data" in text.lower(), f"list_files didn't find uploaded file: {text[:300]}"


def test_download_file_retrieves_content(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Agent uploads a file, then download_file retrieves its content.

    :param http_client: HTTP client pointed at the live server.
    :param archer_agent: The registered archer agent name.
    """
    # Turn 1: create and upload
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use code_sandbox to create a file called "
                "greeting.txt containing exactly 'HELLO_WORLD'. "
                "Then upload it with upload_file."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    body1 = poll_until_terminal(http_client, rid1, timeout=180)
    assert body1["status"] == "completed", f"Turn 1 failed: {body1.get('error')}"

    # Turn 2: download and show contents
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use download_file to download greeting.txt and tell me exactly what it contains."
            ),
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    body2 = poll_until_terminal(http_client, rid2, timeout=180)
    assert body2["status"] == "completed", f"Turn 2 failed: {body2.get('error')}"

    assert _has_tool_call(body2, "download_file"), "Agent didn't call download_file"
    text = _extract_all_text(body2)
    assert "hello_world" in text.lower(), f"Agent didn't show file contents: {text[:300]}"
