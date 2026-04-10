"""E2E test: archer agent reads bundled skill reference files.

Verifies that the archer agent can load a skill with bundled
reference files and read their contents via ``read_skill_file``.

Uses the ``deep-research`` skill which has a
``references/research-checklist.md`` file bundled with it.

Usage::

    pytest tests/e2e/test_archer_skill_files.py \
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


def _extract_tool_results(
    body: dict[str, Any],
) -> list[str]:
    """
    Extract all function_call_output strings from a response.

    :param body: The terminal response body.
    :returns: List of tool output strings.
    """
    return [
        item.get("output", "")
        for item in body.get("output", [])
        if item.get("type") == "function_call_output"
    ]


def test_archer_loads_skill_and_reads_reference_file(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    The archer agent loads the ``deep-research`` skill, sees the
    bundled reference file listing, reads the research checklist,
    and references its contents in the response.

    This exercises the full skill file pipeline end-to-end with
    a real LLM:
    1. Agent calls ``load_skill("deep-research")``
    2. Tool result includes ``references/research-checklist.md``
    3. Agent calls ``read_skill_file`` to read it
    4. Agent response references the checklist content

    :param http_client: HTTP client pointed at the live e2e server.
    :param archer_agent: The uploaded archer agent name.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Call load_skill with name=deep-research. "
                "Then call read_skill_file with "
                "skill_name=deep-research and "
                "path=references/research-checklist.md. "
                "Tell me what it says."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=300)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    # The agent must have called load_skill.
    tool_names = [
        item.get("name") for item in body.get("output", []) if item.get("type") == "function_call"
    ]
    assert "load_skill" in tool_names, (
        f"Expected load_skill tool call. "
        f"Tool calls: {tool_names}. "
        f"The agent may not have loaded the skill."
    )

    # The agent should have called read_skill_file to read
    # the checklist. If not, the file listing wasn't shown
    # or the agent ignored it.
    assert "read_skill_file" in tool_names, (
        f"Expected read_skill_file tool call. "
        f"Tool calls: {tool_names}. "
        f"The agent may not have seen the file listing in "
        f"load_skill output, or read_skill_file was not registered."
    )

    # The tool result from read_skill_file must contain the
    # checklist content — proves the file was actually read
    # from the extracted bundle.
    tool_results = _extract_tool_results(body)
    checklist_found = any("3 independent sources" in r for r in tool_results)
    assert checklist_found, (
        f"Expected '3 independent sources' in read_skill_file "
        f"result (from research-checklist.md). "
        f"Tool results: {[r[:100] for r in tool_results]}. "
        f"The bundled file may not have been extracted correctly."
    )

    # The final response should mention the quality threshold
    # from the checklist — proves the LLM used the file content.
    text = _extract_all_text(body)
    assert "3" in text and "source" in text.lower(), (
        f"Expected the agent to mention '3 sources' from the checklist. Got: {text[:300]}"
    )
