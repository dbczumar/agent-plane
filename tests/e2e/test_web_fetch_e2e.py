"""E2E test for ``web_fetch`` built-in tool.

Verifies that an agent with ``web_fetch`` can actually spawn the
``__web_researcher`` sub-agent, which uses ``terminal_run`` to
fetch web content and return results. Uses an LLM judge to evaluate
whether the fetched content is relevant.

Usage::

    pytest tests/e2e/test_web_fetch_e2e.py \
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
import yaml

from tests.e2e.conftest import find_free_port, wait_for_server


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


def test_web_fetch_returns_live_content(
    llm_api_key: str,
    tmp_path: Path,
) -> None:
    """
    An agent with ``web_fetch`` can fetch live web data and return it.

    Asks for the current GitHub star count of mlflow/mlflow — a live
    number that changes, so the LLM cannot hallucinate the correct
    answer. An LLM judge verifies the response contains a plausible
    star count.

    **What breaks if this fails:**
    - __web_researcher sub-agent spec is malformed → spawn fails
    - Model inheritance broken → sub-agent has no LLM → workflow error
    - terminal_run not available to sub-agent → can't run scripts
    - Network blocked in sandbox → DNS failure (known srt issue)
    - Polling loop broken → timeout instead of results
    - Output extraction broken → empty response
    """
    agent_dir = tmp_path / "web-fetch-agent"
    agent_dir.mkdir()
    config = {
        "spec_version": 1,
        "name": "web-fetch-tester",
        "description": "Test agent for web_fetch e2e.",
        "llm": {
            "model": "openai/gpt-5.4",
            "connection": {"api_key": llm_api_key},
        },
        "tools": {
            "builtins": ["web_fetch"],
        },
        "instructions": (
            "You have a web_fetch tool. When asked to look something "
            "up, use web_fetch with the query provided. Report the "
            "findings clearly."
        ),
    }
    (agent_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    port = find_free_port()
    tmpdir = tempfile.mkdtemp(prefix="ap-e2e-webfetch-")
    server_proc = subprocess.Popen(
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
            str(agent_dir),
        ],
        env={**os.environ, "OPENAI_API_KEY": llm_api_key},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(base_url)

        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": "web-fetch-tester",
                "input": (
                    "Use web_fetch to find how many GitHub stars the "
                    "mlflow/mlflow repository currently has. Report the "
                    "exact number."
                ),
                "stream": False,
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        body = resp.json()

        # "completed" means the full chain worked: agent → web_fetch
        # → spawn __web_researcher → terminal_run → fetch → return.
        assert body["status"] == "completed", (
            f"Response status is {body['status']!r}, expected 'completed'. "
            f"Output: {body.get('output', [])}"
        )

        full_text = _extract_all_text(body)
        assert len(full_text) > 20, f"Response too short ({len(full_text)} chars)."

        # ── LLM judge: did the agent report a plausible star count? ──
        os.environ["OPENAI_API_KEY"] = llm_api_key

        from mlflow.genai.judges import make_judge

        judge = make_judge(
            name="web_fetch_stars",
            instructions=(
                "You are evaluating whether an AI assistant successfully "
                "fetched live data from the web.\n\n"
                "The assistant was asked to find the current GitHub star "
                "count for mlflow/mlflow. The repo has had over 19,000 "
                "stars since 2024.\n\n"
                "The assistant's response is:\n"
                "{{ outputs }}\n\n"
                "Evaluate:\n"
                "1. Does the response contain a specific number of stars "
                "(not just 'many' or 'popular')?\n"
                "2. Is the number plausible (at least 19,000)?\n\n"
                "Return True if the assistant reported a specific, "
                "plausible star count. Return False if it gave a vague "
                "answer, failed to fetch, or hallucinated a number."
            ),
            feedback_value_type=bool,
        )

        feedback = judge(outputs=full_text)
        assert feedback.value is True, (
            f"LLM judge ruled the agent did NOT successfully attempt "
            f"a live web fetch.\n"
            f"Judge rationale: {feedback.rationale}\n"
            f"Agent response: {full_text[:500]}"
        )
    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
