"""E2E test: Claude SDK executor with client-side tools.

Verifies that the Claude SDK executor correctly parks client-side
tool calls through ``await_tool_output``. The client registers a
tool in ``POST /v1/responses``, the SDK calls it via the MCP
bridge, the call parks (``action_required``), the client PATCHes
the result, and the SDK continues.

Usage::

    pytest tests/e2e/test_claude_coder_client_tools.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from tests.e2e.conftest import poll_for_pending_tool_calls, poll_until_terminal

# A simple client-side tool: the agent asks what time it is.
_CLIENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time. Call this when you need to know what time it is."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


def test_claude_sdk_parks_client_tool_call(
    http_client: httpx.Client,
    claude_coder_agent: str,
) -> None:
    """
    The Claude SDK executor parks a client-side tool call and
    resumes after the client PATCHes the result.

    The agent is asked a question that requires the client-side
    ``get_current_time`` tool. The SDK calls it via the MCP bridge,
    the call appears as ``action_required`` on the response output,
    the client delivers the result via PATCH, and the agent
    completes with a response referencing the time.

    :param http_client: HTTP client pointed at the live e2e server.
    :param claude_coder_agent: The uploaded claude-coder agent name.
    """
    # Create a response with the client-side tool registered.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "What time is it right now? You MUST call the "
                "get_current_time tool to find out. Do not guess."
            ),
            "background": True,
            "tools": _CLIENT_TOOLS,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    # Poll for the pending tool call.
    pending = poll_for_pending_tool_calls(
        http_client,
        response_id,
        timeout=120,
    )

    # The SDK should have called get_current_time, which parked.
    assert len(pending) >= 1, (
        f"Expected at least 1 pending tool call (get_current_time), "
        f"got {len(pending)}. If 0, the SDK didn't call the client "
        f"tool — it may have been filtered from allowed_tools or "
        f"the MCP bridge didn't route it correctly."
    )

    time_call = None
    for fc in pending:
        if fc.get("name") == "get_current_time":
            time_call = fc
            break

    assert time_call is not None, (
        f"Expected a pending get_current_time call, but found: "
        f"{[fc.get('name') for fc in pending]}. The SDK may have "
        f"called a different tool instead."
    )

    # PATCH the result back.
    call_id = time_call["call_id"]
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    patch_resp = http_client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {
                    "call_id": call_id,
                    "output": f"The current time is {current_time}.",
                },
            ],
        },
    )
    assert patch_resp.status_code == 200, (
        f"PATCH failed: {patch_resp.status_code} {patch_resp.text}"
    )

    # Poll until the agent completes.
    body = poll_until_terminal(http_client, response_id, timeout=120)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. "
        f"Error: {body.get('error')}. "
        f"The agent should complete after receiving the tool result."
    )

    # The response should reference the time we provided.
    output_texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    output_texts.append(text)

    combined = " ".join(output_texts).lower()
    # The agent should mention the time in its response.
    assert any(fragment in combined for fragment in [current_time[:10], "time", "current"]), (
        f"Expected the agent to reference the time in its response. Got: {combined[:300]}"
    )
