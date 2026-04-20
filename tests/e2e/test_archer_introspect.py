"""E2E test: archer agent introspect tool.

Verifies that archer can use ``introspect`` to accurately describe
its own configuration — tools, skills, sub-agents. Uses LLM judges.

Usage::

    pytest tests/e2e/test_archer_introspect.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
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


def _start_archer(api_key: str) -> tuple[subprocess.Popen[bytes], str]:
    """
    Start a standalone server with the archer agent.

    :param api_key: OpenAI API key.
    :returns: (server process, base URL).
    """
    port = find_free_port()
    tmpdir = tempfile.mkdtemp(prefix="ap-e2e-introspect-")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-uri",
            f"sqlite:///{tmpdir}/test.db",
            "--artifact-location",
            f"{tmpdir}/artifacts",
            "--agent",
            str(_ARCHER_DIR),
        ],
        env={**os.environ, "OPENAI_API_KEY": api_key},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"
    wait_for_server(base_url)
    return proc, base_url


def test_archer_introspect_tools_and_skills(
    llm_api_key: str,
) -> None:
    """
    Archer uses introspect to accurately list its own tools and skills.

    **What breaks if this fails:**
    - IntrospectTool not registered → agent can't call it.
    - ToolManager didn't pass spec → summary is empty.
    - Tool/skill names missing from summary.
    """
    proc, base_url = _start_archer(llm_api_key)
    try:
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": "archer",
                "input": (
                    "Use the introspect tool to check what tools and "
                    "skills you have. List them for me. Be brief."
                ),
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        body = resp.json()

        assert body["status"] == "completed", (
            f"Status: {body['status']!r}. Output: {body.get('output', [])}"
        )

        full_text = _extract_all_text(body)

        os.environ["OPENAI_API_KEY"] = llm_api_key
        from mlflow.genai.judges import make_judge

        judge = make_judge(
            name="introspect_tools_skills",
            instructions=(
                "You are evaluating whether an AI assistant accurately "
                "described its own tools and skills after introspecting.\n\n"
                "The assistant's ACTUAL configuration is:\n"
                "  Tools: web_search, web_fetch, terminal_run, "
                "terminal_list, terminal_close, upload_file, "
                "search_conversations, introspect\n"
                "  Skills: deep-research, explain\n"
                "  Sub-agents: fact_checker, summarizer\n\n"
                "The assistant's response is:\n"
                "{{ outputs }}\n\n"
                "Does the response mention at least 3 of these tools "
                "and both skills? It doesn't need every single one, "
                "but should demonstrate successful introspection.\n\n"
                "Return True if accurate, False if hallucinated or "
                "failed to introspect."
            ),
            feedback_value_type=bool,
        )

        feedback = judge(outputs=full_text)
        assert feedback.value is True, (
            f"Judge: archer did NOT accurately describe its tools/skills.\n"
            f"Rationale: {feedback.rationale}\n"
            f"Response: {full_text[:500]}"
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_archer_introspect_sub_agent_details(
    llm_api_key: str,
) -> None:
    """
    Archer uses introspect to describe its sub-agents.

    **What breaks if this fails:**
    - Sub-agent section drilling broken.
    - Sub-agents not in parent spec.
    """
    proc, base_url = _start_archer(llm_api_key)
    try:
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": "archer",
                "input": (
                    "Use introspect to look at your sub-agents. "
                    "What sub-agents do you have and what does each "
                    "one do? Be brief."
                ),
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        body = resp.json()

        assert body["status"] == "completed", (
            f"Status: {body['status']!r}. Output: {body.get('output', [])}"
        )

        full_text = _extract_all_text(body)

        os.environ["OPENAI_API_KEY"] = llm_api_key
        from mlflow.genai.judges import make_judge

        judge = make_judge(
            name="introspect_sub_agents",
            instructions=(
                "You are evaluating whether an AI assistant accurately "
                "described its sub-agents after introspecting.\n\n"
                "The assistant's ACTUAL sub-agents are:\n"
                "  - fact_checker: verifies claims with evidence\n"
                "  - summarizer: summarizes content\n"
                "  - __web_researcher: internal helper for web_fetch (OK to mention)\n\n"
                "The assistant's response is:\n"
                "{{ outputs }}\n\n"
                "Does the response mention both fact_checker and "
                "summarizer with roughly accurate descriptions? "
                "Ignore __web_researcher — it's an internal sub-agent "
                "and mentioning it is fine.\n\n"
                "Return True if both fact_checker and summarizer are "
                "mentioned accurately, False otherwise."
            ),
            feedback_value_type=bool,
        )

        feedback = judge(outputs=full_text)
        assert feedback.value is True, (
            f"Judge: archer did NOT accurately describe sub-agents.\n"
            f"Rationale: {feedback.rationale}\n"
            f"Response: {full_text[:500]}"
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
