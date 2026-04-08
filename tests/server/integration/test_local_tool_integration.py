"""Integration tests for local Python tool execution through the full pipeline.

Creates an agent with a ``tools/python/word_count.py`` local tool, mocks
the LLM to call it, and verifies the tool runs in a subprocess (with or
without srt) and the result appears in the persisted conversation.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response

pytestmark = [pytest.mark.asyncio]

_AGENT_NAME = "local-tool-test-agent"

# ── Tool source code (embedded in the bundle) ───────────

_WORD_COUNT_SOURCE = '''\
"""Count words in text."""
from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "word_count",
        "description": "Count words in text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to count."},
            },
            "required": ["text"],
        },
    },
}

async def run(arguments: dict[str, Any]) -> str:
    """Count words and return JSON."""
    import json as _json
    text = arguments.get("text", "")
    return _json.dumps({"word_count": len(text.split())})
'''


# ── Bundle builder ───────────────────────────────────────


def _build_local_tool_agent_bundle() -> bytes:
    """
    Build an agent bundle with a local Python tool.

    :returns: Raw tar.gz bytes with config.yaml and
        tools/python/word_count.py.
    """
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {"model": _AGENT_NAME},
    }

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_text(tf, "config.yaml", yaml.dump(config))
        _add_text(tf, "tools/python/word_count.py", _WORD_COUNT_SOURCE)
    return buf.getvalue()


def _add_text(tf: tarfile.TarFile, name: str, content: str) -> None:
    """
    Add a text file to a tarball.

    :param tf: The open TarFile.
    :param name: Archive member name.
    :param content: Text content.
    """
    data = content.encode()
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


# ── Helpers ──────────────────────────────────────────────


async def _create_local_tool_agent(
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Upload an agent bundle with a local Python tool.

    :param client: HTTP client for the test server.
    :returns: The agent creation response JSON.
    """
    bundle = _build_local_tool_agent_bundle()
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent creation failed: {resp.status_code} {resp.text}"
    return resp.json()


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status.

    :param client: HTTP client for the server.
    :param response_id: The response/task ID to poll.
    :returns: The response JSON once completed.
    """
    for _ in range(200):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"Response {response_id} did not reach terminal status",
    )


async def _get_items(
    client: httpx.AsyncClient,
    conv_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all conversation items.

    :param client: HTTP client for the server.
    :param conv_id: The conversation ID.
    :returns: List of item dicts.
    """
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    result: list[dict[str, Any]] = resp.json()["data"]
    return result


# ── Tests ────────────────────────────────────────────────


async def test_local_tool_executes_in_subprocess(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A local Python tool registered from ``tools/python/`` executes
    in a subprocess and returns its result through the full pipeline.

    The mock LLM calls ``word_count``, the tool runs in a subprocess
    (with srt if available, otherwise plain), and the
    ``function_call_output`` item in the conversation contains the
    real tool result.

    **What breaks if wrong**: tool not registered (schema missing from
    LLM call), subprocess fails (fd 3 or stdout protocol error), or
    result not persisted to conversation.
    """
    await _create_local_tool_agent(client)

    # LLM call 1: call the word_count tool
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_wc_1",
                "name": "word_count",
                "arguments": json.dumps({"text": "one two three four five"}),
            },
        ],
    )
    # LLM call 2: after tool result, return final text
    mock_llm.add_call(text="The text has 5 words.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Count the words in 'one two three four five'",
    )
    assert result.status_code == 200, (
        f"Response creation failed: {result.status_code} {result.body}"
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    assert completed["status"] == "completed", (
        f"Expected completed, got {completed['status']}. Error: {completed.get('error')}"
    )

    # Verify the tool was called and the result is in the conversation
    items = await _get_items(client, conv_id)

    # Find the function_call_output item for word_count
    fco_items = [
        i
        for i in items
        if i.get("type") == "function_call_output" and i.get("call_id") == "call_wc_1"
    ]
    assert len(fco_items) == 1, (
        f"Expected 1 function_call_output for word_count, "
        f"got {len(fco_items)}. All items: {[i['type'] for i in items]}"
    )

    output = fco_items[0]["output"]
    parsed = json.loads(output)
    assert parsed["word_count"] == 5, (
        f"Expected word_count=5, got {parsed}. Tool may have failed or returned wrong result."
    )


async def test_local_tool_crash_does_not_kill_server(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A local tool that crashes (os._exit) returns an error string
    but the server remains healthy.

    **What breaks if wrong**: in-process execution would kill the
    server and this test would hang or error on the health check.
    """
    # Build a bundle with a crashing tool
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {"model": _AGENT_NAME},
    }
    crash_source = """\
from typing import Any
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "crasher",
        "description": "A tool that crashes.",
        "parameters": {"type": "object", "properties": {}},
    },
}
async def run(arguments: dict[str, Any]) -> str:
    import os; os._exit(1)
"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_text(tf, "config.yaml", yaml.dump(config))
        _add_text(tf, "tools/python/crasher.py", crash_source)
    bundle = buf.getvalue()

    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # LLM call 1: call the crashing tool
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_crash_1",
                "name": "crasher",
                "arguments": "{}",
            },
        ],
    )
    # LLM call 2: after error result, produce final text
    mock_llm.add_call(text="The tool crashed, but I can continue.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Call the crasher tool",
    )
    response_id = result.body["id"]
    completed = await _wait_for_completion(client, response_id)

    # Server is still alive — verify with a health check
    health = await client.get("/health")
    assert health.status_code == 200, "Server should still be healthy after tool crash"

    # The response should have completed (LLM recovered from the error)
    assert completed["status"] == "completed", (
        f"Expected completed (LLM handles tool error gracefully), got {completed['status']}"
    )
