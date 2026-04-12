"""End-to-end tests for the coder agent with sub-agents.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_coder_subagent.py \\
        --llm-api-key $LLM_API_KEY -v

Tests exercise:
- Sub-agent spawning with real LLM
- Client-side tool tunneling (park → poll → PATCH → resume)
- Auto-collect at turn end
- Full reviewer sub-agent workflow with real tool execution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

# Load the coder tool set for client-side tool execution.
from agent_plane.client_tools import get_tool_set as _get_tool_set
from tests.e2e.conftest import (
    poll_for_pending_tool_calls,
)

_tool_mod = _get_tool_set("coding")
TOOLS: list[dict[str, Any]] = _tool_mod.TOOLS
execute_tool = _tool_mod.execute_tool


def _create_response(
    client: httpx.Client,
    model: str,
    user_input: str,
) -> dict[str, Any]:
    """
    Create a background response with client-side tools.

    :param client: HTTP client.
    :param model: Agent name.
    :param user_input: User message.
    :returns: The response body dict.
    """
    resp = client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": user_input,
            "background": True,
            "tools": TOOLS,
        },
    )
    resp.raise_for_status()
    return resp.json()


def _handle_tunneled_calls(
    client: httpx.Client,
    response_id: str,
    pending: list[dict[str, Any]],
) -> None:
    """
    Execute tunneled tool calls and PATCH results back.

    :param client: HTTP client.
    :param response_id: Root response ID for PATCH.
    :param pending: List of action_required function_call items.
    """
    tool_results = []
    for fc in pending:
        name = fc["name"]
        call_id = fc["call_id"]
        arguments = json.loads(fc.get("arguments", "{}"))
        result = execute_tool(name, arguments)
        tool_results.append(
            {
                "call_id": call_id,
                "output": result,
            }
        )
    resp = client.patch(
        f"/v1/responses/{response_id}",
        json={"tool_results": tool_results},
    )
    assert resp.status_code == 200, f"PATCH failed: {resp.text[:300]}"


def _run_with_tunneling(
    client: httpx.Client,
    model: str,
    user_input: str,
) -> dict[str, Any]:
    """
    Create a response and handle all tunneled tool calls
    until the response completes.

    Polls for pending tool calls, executes them locally,
    PATCHes results back, and repeats until the response
    reaches a terminal state.

    :param client: HTTP client.
    :param model: Agent name.
    :param user_input: User message.
    :returns: The terminal response body.
    """
    body = _create_response(client, model, user_input)
    response_id = body["id"]

    while True:
        # Check for pending tunneled tool calls.
        pending = poll_for_pending_tool_calls(client, response_id, timeout=120)
        if pending:
            _handle_tunneled_calls(client, response_id, pending)
            continue
        # No pending calls — check if terminal.
        resp = client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        # Still in progress but no pending calls yet —
        # keep polling.


def test_coder_spawns_reviewer_and_collects(
    http_client: httpx.Client,
    coder_agent: str,
    sample_code_dir: Path,
) -> None:
    """
    Coder agent spawns the reviewer sub-agent, the reviewer
    uses client-side tools (Read, Glob, etc.) to inspect files,
    and the parent auto-collects and produces a final response
    incorporating the review.

    This is the full end-to-end flow that caught:
    - Empty sub-agent output (client tools not tunneled)
    - "Unknown tool" errors (client re-executing server tools)
    - Deadlock (time.sleep polling exhausting DBOS threads)
    - Turn completing before sub-agent finishes (no auto-collect)
    """
    result = _run_with_tunneling(
        http_client,
        coder_agent,
        f"Use spawn_sub_agents to spawn the reviewer sub-agent. "
        f"Tell it to review the Python code in {sample_code_dir}. "
        f"Do NOT read the files yourself — delegate to the reviewer. "
        f"After the reviewer finishes, show me its findings.",
    )

    assert result["status"] == "completed", (
        f"Expected completed, got {result['status']}. Error: {result.get('error')}"
    )

    output = result["output"]

    # The response must contain spawn_sub_agents tool call,
    # proving the LLM actually spawned instead of acting
    # directly.
    spawn_calls = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == "spawn_sub_agents"
    ]
    assert len(spawn_calls) >= 1, (
        "LLM didn't call spawn_sub_agents — it may have used "
        "client tools directly instead of delegating. Output: "
        + str([i.get("name") for i in output if i.get("type") == "function_call"])
    )

    # The output must contain text from the assistant with
    # substantial review content.
    text_items = [item for item in output if item.get("type") == "message"]
    assert len(text_items) >= 1, f"Expected at least one message, got: {output}"
    all_text = " ".join(
        c.get("text", "") for item in text_items for c in item.get("content", [])
    ).lower()
    assert len(all_text) > 100, (
        f"Expected substantial review output, got {len(all_text)} chars: {all_text[:200]!r}"
    )


def test_coder_spawns_parallel_subagents(
    http_client: httpx.Client,
    coder_agent: str,
    sample_code_dir: Path,
) -> None:
    """
    Coder agent spawns BOTH reviewer and researcher sub-agents
    in parallel in a single ``spawn_sub_agents`` call. Both
    sub-agents run concurrently, the reviewer uses client-side
    tools (tunneled), the researcher uses web search (server-side),
    and the parent auto-collects both results.

    This exercises:
    - Parallel sub-agent spawning (multiple entries in agents[])
    - Concurrent tunneling (client tools from one sub-agent while
      another runs independently)
    - Auto-collect waiting for ALL sub-agents before completing
    - Merging results from multiple sub-agents in final output
    """
    result = _run_with_tunneling(
        http_client,
        coder_agent,
        f"Use spawn_sub_agents to spawn BOTH agents in a single "
        f"call (one spawn_sub_agents with two entries in the "
        f"agents array):\n"
        f"1. reviewer — review the Python code in {sample_code_dir}\n"
        f"2. researcher — find what's new in Python 3.14\n"
        f"Do NOT read files or search yourself — delegate to the "
        f"sub-agents. After they finish, show me both results.",
    )

    assert result["status"] == "completed", (
        f"Expected completed, got {result['status']}. Error: {result.get('error')}"
    )

    output = result["output"]

    # Must have at least one spawn_sub_agents call.
    spawn_calls = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == "spawn_sub_agents"
    ]
    assert len(spawn_calls) >= 1, "LLM didn't call spawn_sub_agents"

    # Check the spawn call had both agents. The arguments
    # JSON should contain both "reviewer" and "researcher".
    spawn_args = spawn_calls[0].get("arguments", "")
    assert "reviewer" in spawn_args, f"spawn_sub_agents didn't include reviewer: {spawn_args}"
    assert "researcher" in spawn_args, f"spawn_sub_agents didn't include researcher: {spawn_args}"

    # The final output must contain text from the assistant
    # with substantial content from both sub-agents.
    text_items = [item for item in output if item.get("type") == "message"]
    assert len(text_items) >= 1, f"Expected at least one message, got: {output}"
    all_text = " ".join(
        c.get("text", "") for item in text_items for c in item.get("content", [])
    ).lower()
    # Loose check — LLM output is non-deterministic, but
    # it should mention something from both sub-agents.
    assert len(all_text) > 200, (
        f"Expected substantial output from 2 sub-agents, "
        f"got {len(all_text)} chars: {all_text[:300]!r}"
    )
