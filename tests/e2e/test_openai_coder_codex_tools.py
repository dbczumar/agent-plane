"""E2E test: openai-coder agent uses codex MCP tools (Shell, ApplyPatch).

The openai-coder agent has ``codex:Shell`` and ``codex:ApplyPatch`` as
server-side builtins routed through the Codex MCP server — a subprocess
running ``codex mcp-server``.  These are NOT ``code_sandbox``; they are
independent MCP tools that manage their own workspace.

This test verifies that the agent can use the ``codex`` MCP tool to run
shell commands and create files, proving the Codex MCP integration works
end-to-end without any client-side tools or code_sandbox.

Usage::

    pytest tests/e2e/test_openai_coder_codex_tools.py \\
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _collect_function_calls(body: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract all function_call items from a response body.

    :param body: The terminal response body.
    :returns: List of function_call output items.
    """
    return [item for item in body.get("output", []) if item.get("type") == "function_call"]


def _assert_codex_tool_called(body: dict[str, Any]) -> None:
    """
    Assert that at least one ``codex`` MCP tool call appears in the output.

    The Codex MCP server exposes a tool named ``codex`` (the session
    entry point). Shell and ApplyPatch are modes within that tool,
    not separate tool names.

    :param body: The terminal response body.
    """
    function_calls = _collect_function_calls(body)
    codex_calls = [fc for fc in function_calls if fc.get("name") == "codex"]
    assert codex_calls, (
        f"Expected at least one 'codex' tool call but found: "
        f"{[fc.get('name') for fc in function_calls]}. "
        f"The codex MCP server may not have started (is the "
        f"'codex' binary on PATH?)."
    )


def _get_codex_tool_outputs(body: dict[str, Any]) -> list[str]:
    """
    Return the output strings from all codex MCP tool calls.

    Matches ``function_call_output`` items by ``call_id`` against
    codex ``function_call`` items.

    :param body: The terminal response body.
    :returns: List of non-empty output strings from codex calls.
    """
    function_calls = _collect_function_calls(body)
    codex_call_ids = {fc["call_id"] for fc in function_calls if fc.get("name") == "codex"}
    return [
        item["output"]
        for item in body.get("output", [])
        if item.get("type") == "function_call_output"
        and item.get("call_id") in codex_call_ids
        and item.get("output", "").strip()
    ]


def _assert_codex_tool_output_contains(
    body: dict[str, Any],
    needle: str,
) -> None:
    """
    Assert that at least one codex tool output contains ``needle``.

    This checks the TOOL OUTPUT, not the agent's response text —
    preventing false-positives from LLM hallucination.

    :param body: The terminal response body.
    :param needle: String that must appear in a codex tool output.
    """
    outputs = _get_codex_tool_outputs(body)
    assert outputs, (
        "No codex tool outputs found. The MCP server may "
        "have failed silently or returned empty results."
    )
    assert any(needle in out for out in outputs), (
        f"Expected '{needle}' in codex tool output (not just "
        f"agent text). Tool outputs: {[o[:200] for o in outputs]}"
    )


def test_openai_coder_uses_codex_shell(
    http_client: httpx.Client,
    openai_coder_agent: str,
) -> None:
    """
    The openai-coder agent uses the codex MCP tool to run a shell
    command, proving codex:Shell works as a server-side MCP tool
    independent of code_sandbox.

    No client-side tools are passed — the agent can ONLY use its
    built-in codex MCP tools and web search.

    **What breaks if wrong:** If the ``codex`` binary is missing
    from PATH, ``_build_codex_mcp`` returns None and the MCP
    server is never created — the agent has no tools to run
    commands. If the MCP session rewriter fails, subsequent calls
    lose context and return errors.

    :param http_client: HTTP client pointed at the live server.
    :param openai_coder_agent: The uploaded openai-coder agent name.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": openai_coder_agent,
            "input": (
                "Use your shell tool to run: echo 'CODEX_MCP_WORKS' Then show me the output."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=180)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )
    _assert_codex_tool_called(body)

    # Assert on the TOOL OUTPUT, not agent text — prevents
    # false-positives from LLM hallucination.
    _assert_codex_tool_output_contains(body, "CODEX_MCP_WORKS")


def _create_codex_response(
    client: httpx.Client,
    model: str,
    user_input: str,
) -> str:
    """
    Create a background response (no client-side tools) and return its ID.

    :param client: HTTP client pointed at the live server.
    :param model: Agent name.
    :param user_input: The user message.
    :returns: The response ID.
    """
    resp = client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": user_input,
            "background": True,
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


def test_openai_coder_codex_creates_and_lists_file(
    http_client: httpx.Client,
    openai_coder_agent: str,
) -> None:
    """
    The openai-coder agent uses the codex MCP tool to create a
    file and list it, proving multi-step filesystem operations
    work through the Codex MCP session.

    **What breaks if wrong:** If session rewriting fails
    (threadId not captured), the second call starts a fresh
    session and the created file is invisible.

    :param http_client: HTTP client pointed at the live server.
    :param openai_coder_agent: The uploaded openai-coder agent name.
    """
    response_id = _create_codex_response(
        http_client,
        openai_coder_agent,
        "Create a file called 'canary.txt' containing the text "
        "'CODEX_CANARY_2026', then list the directory to confirm "
        "it exists and show me the file contents. Use your "
        "built-in tools only.",
    )
    body = poll_until_terminal(http_client, response_id, timeout=180)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )
    _assert_codex_tool_called(body)

    # Assert on TOOL OUTPUT: the sentinel or filename must appear
    # in what codex actually returned, not just the agent's prose.
    _assert_codex_tool_output_contains(body, "canary")


def test_openai_coder_codex_lists_directory(
    http_client: httpx.Client,
    openai_coder_agent: str,
) -> None:
    """
    The openai-coder agent uses the codex MCP tool to list files
    in a directory, proving the tool output is captured and
    returned to the response.

    This reproduces a real failure: MCP tool calls returned
    empty output because the ``tool_output`` event was not
    handled. The agent appeared to freeze and the tool result
    was blank.

    **What breaks if wrong:** If ``_map_run_item_event`` does
    not handle ``tool_output`` events, the codex tool result
    is always empty — the response shows a blank tool output
    and the agent either hallucinates the listing or says the
    tool failed.

    :param http_client: HTTP client pointed at the live server.
    :param openai_coder_agent: The uploaded openai-coder agent name.
    """
    response_id = _create_codex_response(
        http_client,
        openai_coder_agent,
        "List the files in /tmp. Use your built-in shell tool.",
    )
    body = poll_until_terminal(http_client, response_id, timeout=180)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )
    _assert_codex_tool_called(body)

    # The tool output must contain actual directory content —
    # not be empty. An empty output means the tool_output
    # event was not captured by the executor.
    outputs = _get_codex_tool_outputs(body)
    assert outputs, (
        "Codex tool output is empty. The executor may not be "
        "handling 'tool_output' events from the Agents SDK."
    )
