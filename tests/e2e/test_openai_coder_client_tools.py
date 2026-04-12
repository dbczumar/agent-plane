"""E2E test: openai-coder agent uses client-side tools to list and manipulate files.

The openai-coder agent has ``codex:Shell`` and ``codex:ApplyPatch`` as
server-side MCP builtins, but should ALSO be able to use client-side
tools (Read, Write, Edit, Glob, Grep, Bash) when the frontend passes
them.  This test verifies that the agent actually invokes client-side
tools — proving they are not masked by the codex MCP builtins.

See ``test_openai_coder_codex_tools.py`` for tests covering the
server-side codex MCP tools (Shell, ApplyPatch) directly.

Usage::

    pytest tests/e2e/test_openai_coder_client_tools.py \\
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx

from tests.e2e.conftest import poll_for_pending_tool_calls

# Load the coder tool set for client-side tool execution.
_TOOL_SET_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "frontends" / "tool_sets" / "coder.py"
)
_spec = importlib.util.spec_from_file_location("coder_tools", _TOOL_SET_PATH)
assert _spec is not None and _spec.loader is not None
_tool_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool_mod)
TOOLS: list[dict[str, Any]] = _tool_mod.TOOLS
execute_tool = _tool_mod.execute_tool


def _create_response(
    client: httpx.Client,
    model: str,
    user_input: str,
) -> dict[str, Any]:
    """
    Create a background response with client-side tools registered.

    :param client: HTTP client pointed at the live server.
    :param model: Agent name, e.g. ``"openai-coder"``.
    :param user_input: The user message to send.
    :returns: The initial response body dict.
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
    Execute pending client-side tool calls locally and PATCH results back.

    :param client: HTTP client pointed at the live server.
    :param response_id: Root response ID for the PATCH endpoint.
    :param pending: List of ``action_required`` function_call items.
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
    Create a response and handle all tunneled tool calls until terminal.

    Polls for pending client-side tool calls, executes them locally,
    PATCHes results back, and repeats until the response reaches a
    terminal state (completed or failed).

    :param client: HTTP client pointed at the live server.
    :param model: Agent name.
    :param user_input: The user message.
    :returns: The terminal response body.
    """
    body = _create_response(client, model, user_input)
    response_id = body["id"]

    while True:
        pending = poll_for_pending_tool_calls(client, response_id, timeout=120)
        if pending:
            _handle_tunneled_calls(client, response_id, pending)
            continue
        resp = client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all assistant text from a terminal response body.

    :param body: The terminal response body from GET /v1/responses/{id}.
    :returns: All assistant text blocks joined by newlines, lowercased.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _collect_function_calls(body: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract all function_call items from a response body.

    :param body: The terminal response body.
    :returns: List of function_call output items.
    """
    return [item for item in body.get("output", []) if item.get("type") == "function_call"]


def _assert_client_tool_output_contains(
    result: dict[str, Any],
    needle: str,
) -> None:
    """
    Assert that at least one client-side tool output contains ``needle``.

    Checks the actual ``function_call_output`` items for client-side
    tools (Glob, Read, Bash, Grep) — not the agent's prose. This
    prevents false-positives from LLM hallucination.

    :param result: The terminal response body.
    :param needle: String that must appear in a tool output.
    """
    function_calls = _collect_function_calls(result)
    client_tools = {"Glob", "Read", "Bash", "Grep"}
    client_call_ids = {fc["call_id"] for fc in function_calls if fc.get("name") in client_tools}
    assert client_call_ids, (
        f"No client-side tool calls found. Called: {[fc.get('name') for fc in function_calls]}"
    )
    outputs = [
        item["output"]
        for item in result.get("output", [])
        if item.get("type") == "function_call_output"
        and item.get("call_id") in client_call_ids
        and item.get("output")
    ]
    assert any(needle in out for out in outputs), (
        f"Expected '{needle}' in client tool output (not just "
        f"agent text). Tool outputs: {[o[:200] for o in outputs]}"
    )


def test_openai_coder_lists_files_with_client_tools(
    http_client: httpx.Client,
    openai_coder_agent: str,
    sample_code_dir: Path,
) -> None:
    """
    The openai-coder agent uses client-side Glob/Read to list and
    read files, proving client-side tools work alongside codex
    sandbox builtins.

    **What breaks if wrong:** If client-side tools are not
    registered, the agent only sees codex:Shell (sandbox) and
    cannot access the host temp directory. If tunneling fails,
    tool calls never park and the poll times out.

    :param http_client: HTTP client pointed at the live server.
    :param openai_coder_agent: The uploaded openai-coder agent name.
    :param sample_code_dir: Temp dir with calculator.py, utils.py.
    """
    result = _run_with_tunneling(
        http_client,
        openai_coder_agent,
        f"List all Python files in {sample_code_dir} and tell me "
        f"their names. Use the Glob tool with pattern '**/*.py' and "
        f"path '{sample_code_dir}'. Then use the Read tool to read "
        f"the contents of calculator.py from that directory. "
        f"Do NOT use Shell or ApplyPatch — use the Glob and Read "
        f"tools only.",
    )

    assert result["status"] == "completed", (
        f"Expected completed, got {result['status']}. Error: {result.get('error')}"
    )
    # Assert on TOOL OUTPUT: "calculator" must appear in what the
    # client-side tool returned, not just the agent's summary.
    _assert_client_tool_output_contains(result, "calculator")


def _assert_write_and_read_called(result: dict[str, Any]) -> None:
    """
    Assert that both Write and Read appear in the response's tool calls.

    :param result: The terminal response body.
    """
    called_names = [fc["name"] for fc in _collect_function_calls(result)]
    assert "Write" in called_names, (
        f"Expected Write tool call but got: {called_names}. "
        f"The agent may have used codex:ApplyPatch instead."
    )
    assert "Read" in called_names, (
        f"Expected Read tool call but got: {called_names}. "
        f"The agent may have used codex:Shell 'cat' instead."
    )


def _assert_file_written_locally(
    target_file: Path,
    sentinel: str,
) -> None:
    """
    Assert that a file exists on the local filesystem with expected content.

    Proves client-side Write executed locally, not in the sandbox.

    :param target_file: Path to the file that should exist.
    :param sentinel: String that must appear in the file contents.
    """
    assert target_file.exists(), (
        f"File {target_file} was not created on the local "
        f"filesystem. The Write tool may have executed in the "
        f"codex sandbox instead of locally."
    )
    actual_content = target_file.read_text()
    assert sentinel in actual_content, (
        f"Expected '{sentinel}' in file contents, got: {actual_content[:200]!r}."
    )


def test_openai_coder_writes_and_reads_file(
    http_client: httpx.Client,
    openai_coder_agent: str,
    tmp_path: Path,
) -> None:
    """
    The openai-coder agent uses client-side Write and Read to
    create a file and verify its contents locally.

    **What breaks if wrong:** If Write/Read fall back to
    codex:ApplyPatch/Shell, the file lands in the sandbox — not
    on the host filesystem. The local file assertion fails.

    :param http_client: HTTP client pointed at the live server.
    :param openai_coder_agent: The uploaded openai-coder agent name.
    :param tmp_path: Pytest-provided temporary directory.
    """
    target_file = tmp_path / "agent_test_output.txt"
    sentinel = "AGENT_PLANE_E2E_CANARY_2026"

    result = _run_with_tunneling(
        http_client,
        openai_coder_agent,
        f"Do exactly these two steps:\n"
        f"1. Use the Write tool to create a file at "
        f"'{target_file}' with the content '{sentinel}'\n"
        f"2. Use the Read tool to read back the file at "
        f"'{target_file}' and show me its contents.\n"
        f"Use ONLY the Write and Read tools. Do NOT use Shell "
        f"or ApplyPatch.",
    )

    assert result["status"] == "completed", (
        f"Expected completed, got {result['status']}. Error: {result.get('error')}"
    )
    _assert_write_and_read_called(result)
    _assert_file_written_locally(target_file, sentinel)

    all_text = _extract_all_text(result)
    assert sentinel in all_text, (
        f"Expected '{sentinel}' in agent response (from Read tool output), got: {all_text[:500]}"
    )
