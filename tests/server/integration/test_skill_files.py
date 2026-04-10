"""Server integration tests for skill bundled reference files.

Verifies the full pipeline: agent with a skill containing bundled
reference files → LLM calls ``load_skill`` → sees file listing →
LLM calls ``read_skill_file`` → gets file contents → includes them
in the final response.

Uses the ControllableMockClient with scripted tool calls to
deterministically exercise both tools in sequence.
"""

from __future__ import annotations

import io
import tarfile
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient

pytestmark = pytest.mark.asyncio

# ── Agent bundle builder with skill + reference files ─────


def _build_agent_with_skill_files() -> bytes:
    """
    Build an agent bundle containing a skill with bundled
    reference files in ``references/`` and ``assets/``.

    The skill ``code-review`` has:
    - ``SKILL.md`` with frontmatter and content
    - ``references/style-guide.md`` with a style guide
    - ``assets/patterns.json`` with a JSON config

    :returns: tar.gz bytes for the agent bundle.
    """
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": "skill-test-agent",
        "llm": {"model": "skill-test-agent"},
    }
    config_bytes = yaml.dump(config).encode()

    skill_md = (
        "---\n"
        "name: code-review\n"
        "description: Reviews code for bugs and style.\n"
        "---\n"
        "When reviewing code, check for bugs, style issues, "
        "and potential improvements.\n"
    )
    style_guide = (
        "# Style Guide\n\n"
        "- Use snake_case for functions\n"
        "- Use PascalCase for classes\n"
        "- Max line length: 100 characters\n"
    )
    patterns_json = '{"forbidden": ["eval", "exec"], "max_lines": 500}'

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # config.yaml
        _add_file(tf, "config.yaml", config_bytes)
        # SKILL.md
        _add_file(
            tf,
            "skills/code-review/SKILL.md",
            skill_md.encode(),
        )
        # references/style-guide.md
        _add_file(
            tf,
            "skills/code-review/references/style-guide.md",
            style_guide.encode(),
        )
        # assets/patterns.json
        _add_file(
            tf,
            "skills/code-review/assets/patterns.json",
            patterns_json.encode(),
        )
    return buf.getvalue()


def _add_file(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    """
    Add a file to a tarball.

    :param tf: The open TarFile.
    :param name: Archive path, e.g. ``"skills/code-review/SKILL.md"``.
    :param data: File content bytes.
    """
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


async def _create_skill_agent(
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Upload the skill-test-agent and return the response JSON.

    :param client: Async HTTP client.
    :returns: The agent creation response body.
    """
    bundle = _build_agent_with_skill_files()
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent upload failed: {resp.status_code} {resp.text}"
    return resp.json()


# ── Tests ─────────────────────────────────────────────────


async def test_load_skill_lists_bundled_files(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When the LLM calls ``load_skill``, the result includes the
    skill content AND a listing of bundled reference files.

    The mock LLM's first call returns a ``load_skill`` tool call.
    The second call (after receiving the tool result) returns a
    text response that should reference the file listing.

    Breakage this catches:
    - ``list_skill_resources`` not finding files in the bundle
    - ``skill_dir`` being ``None`` after extraction (path lost)
    - ``ReadSkillFileTool`` not registered when resources exist
    """
    await _create_skill_agent(client)

    # First LLM call: call load_skill
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_load",
                "name": "load_skill",
                "arguments": '{"name": "code-review"}',
            },
        ],
    )
    # Second LLM call: respond with text after seeing skill content
    mock_llm.add_call(text="I loaded the code-review skill.")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "skill-test-agent",
            "input": "Load the code-review skill.",
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    # Find the load_skill tool result in the output.
    tool_outputs = [
        item
        for item in body.get("output", [])
        if item.get("type") == "function_call_output" and item.get("call_id") == "call_load"
    ]
    assert len(tool_outputs) == 1, (
        f"Expected 1 function_call_output for load_skill, "
        f"got {len(tool_outputs)}. "
        f"Output types: {[i.get('type') for i in body.get('output', [])]}"
    )

    tool_result = tool_outputs[0]["output"]

    # The skill content must be present.
    assert "check for bugs" in tool_result, (
        f"Skill content missing from load_skill result. Got: {tool_result[:200]}"
    )

    # The file listing must include both bundled files.
    # If missing, list_skill_resources didn't find the files
    # or skill_dir was lost during bundle extraction.
    assert "references/style-guide.md" in tool_result, (
        f"references/style-guide.md not listed in load_skill result. Got: {tool_result[:300]}"
    )
    assert "assets/patterns.json" in tool_result, (
        f"assets/patterns.json not listed in load_skill result. Got: {tool_result[:300]}"
    )

    # The listing must mention read_skill_file so the LLM
    # knows which tool to use.
    assert "read_skill_file" in tool_result, (
        "read_skill_file not mentioned in load_skill result. "
        "The LLM won't know how to read the files."
    )


async def test_read_skill_file_returns_file_contents(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    When the LLM calls ``read_skill_file``, the result contains
    the exact contents of the bundled reference file.

    The mock LLM calls ``read_skill_file`` with the path from
    the listing. The tool result must contain the file's content.

    Breakage this catches:
    - Path resolution failing after bundle extraction
    - ``skill_dir`` pointing to wrong directory
    - Traversal protection rejecting valid relative paths
    """
    await _create_skill_agent(client)

    # First LLM call: call read_skill_file directly
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_read",
                "name": "read_skill_file",
                "arguments": (
                    '{"skill_name": "code-review", "path": "references/style-guide.md"}'
                ),
            },
        ],
    )
    # Second LLM call: respond with text
    mock_llm.add_call(text="The style guide says use snake_case.")

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "skill-test-agent",
            "input": "Read the style guide.",
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    # Find the read_skill_file tool result.
    tool_outputs = [
        item
        for item in body.get("output", [])
        if item.get("type") == "function_call_output" and item.get("call_id") == "call_read"
    ]
    assert len(tool_outputs) == 1, (
        f"Expected 1 function_call_output for read_skill_file, got {len(tool_outputs)}."
    )

    tool_result = tool_outputs[0]["output"]

    # The exact file content must be present — proves the file
    # was read from the extracted bundle, not fabricated.
    assert "snake_case" in tool_result, (
        f"Expected 'snake_case' from style-guide.md. Got: {tool_result[:200]}"
    )
    assert "PascalCase" in tool_result, (
        f"Expected 'PascalCase' from style-guide.md. Got: {tool_result[:200]}"
    )
    assert "100 characters" in tool_result, (
        f"Expected '100 characters' from style-guide.md. Got: {tool_result[:200]}"
    )


async def test_load_then_read_skill_file_full_flow(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Full flow: LLM loads a skill, sees the file listing, reads a
    file, and produces a response referencing the file contents.

    Three LLM calls:
    1. ``load_skill("code-review")`` → skill content + file listing
    2. ``read_skill_file("code-review", "assets/patterns.json")`` → JSON
    3. Final text response referencing the JSON content

    Breakage this catches:
    - Multi-step tool call flow broken (load → read → respond)
    - Conversation history not including tool results correctly
    - JSON reference file not readable
    """
    await _create_skill_agent(client)

    # Step 1: load the skill
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_load",
                "name": "load_skill",
                "arguments": '{"name": "code-review"}',
            },
        ],
    )
    # Step 2: read the JSON patterns file
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_read",
                "name": "read_skill_file",
                "arguments": ('{"skill_name": "code-review", "path": "assets/patterns.json"}'),
            },
        ],
    )
    # Step 3: final response
    mock_llm.add_call(
        text="The forbidden patterns are eval and exec.",
    )

    resp = await client.post(
        "/v1/responses",
        json={
            "model": "skill-test-agent",
            "input": "Load code-review and read its patterns.",
            "background": False,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    # Verify the read_skill_file result contains the JSON.
    read_outputs = [
        item
        for item in body.get("output", [])
        if item.get("type") == "function_call_output" and item.get("call_id") == "call_read"
    ]
    assert len(read_outputs) == 1
    read_result = read_outputs[0]["output"]

    # The JSON content must be present — proves the assets/
    # directory is readable, not just references/.
    assert "eval" in read_result, f"Expected 'eval' from patterns.json. Got: {read_result[:200]}"
    assert "exec" in read_result, f"Expected 'exec' from patterns.json. Got: {read_result[:200]}"

    # The final assistant message must reference the content.
    # 3 LLM calls = load_skill + read_skill_file + final text.
    assert mock_llm.call_count == 3, (
        f"Expected 3 LLM calls (load + read + respond), "
        f"got {mock_llm.call_count}. If 2, read_skill_file "
        f"was not dispatched as a separate tool call."
    )

    # The final text must be in the output.
    assistant_texts = [
        block.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message" and item.get("role") == "assistant"
        for block in item.get("content", [])
    ]
    assert any("eval" in t and "exec" in t for t in assistant_texts), (
        f"Expected final response to reference 'eval' and 'exec'. Got: {assistant_texts}"
    )
