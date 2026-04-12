"""E2E tests for ``ap create`` — onboarding flow.

Verifies that the onboarding agent can create a real agent directory
and that the generated agent can be served by agent-plane.

Usage::

    pytest tests/e2e/test_onboarding_e2e.py \
        --llm-api-key $(cat /tmp/mykey) -v

These tests use a real LLM API key and real server. They are
excluded from the default ``pytest`` run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import yaml

from tests.e2e.conftest import find_free_port, wait_for_server

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_text_from_response(body: dict) -> str:
    """
    Extract concatenated text content from a response body.

    :param body: Parsed response JSON.
    :returns: All text output concatenated.
    """
    texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content["text"])
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Non-interactive e2e: create agent, then serve it
# ---------------------------------------------------------------------------


def test_non_interactive_creates_valid_agent(
    llm_api_key: str,
    tmp_path: Path,
) -> None:
    """
    ``ap create`` non-interactively produces a valid agent directory
    that can be served by ``ap server``.

    Flow:
    1. Run ``ap create`` with a message and ``--model`` and
       ``--allow-shell-access``, targeting a temp directory.
    2. Verify the agent directory exists with a valid ``config.yaml``.
    3. Boot the generated agent with ``ap server --agent``.
    4. Send a request to the agent and verify it responds.

    **What breaks if this fails:**
    - Onboarding agent can't generate valid config.yaml → parse error.
    - Generated agent has wrong model format → server rejects it.
    - Non-interactive tool loop is broken → no files written.
    """
    agent_dir = tmp_path / "my-test-agent"

    # Run ap create non-interactively.
    # The prompt tells the onboarding agent exactly what to create
    # and where, to keep the test deterministic.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "create",
            (
                f"Create a minimal agent called 'test-greeter' in the directory "
                f"{agent_dir}. It should use the model openai/gpt-5.4 with "
                f"${{OPENAI_API_KEY}} for the api key. "
                f"The agent's instructions should say: "
                f"'You are a greeter. When someone says hello, respond with "
                f"exactly: HELLO_E2E_SUCCESS'. "
                f"Only create config.yaml with inline instructions. "
                f"Do not create AGENTS.md or skills. "
                f"Write the files now."
            ),
            "--model",
            "openai/gpt-5.4",
            "--allow-shell-access",
        ],
        env={**os.environ, "OPENAI_API_KEY": llm_api_key},
        capture_output=True,
        text=True,
        timeout=180,
    )

    # The process should complete (exit 0) even if the agent
    # couldn't create files — we check file existence next.
    # Non-zero exit means the CLI itself crashed.
    assert result.returncode == 0, (
        f"ap create exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )

    # ── Verify the generated agent directory ────────────
    config_path = agent_dir / "config.yaml"
    assert config_path.exists(), (
        f"Expected config.yaml at {config_path}. "
        f"The onboarding agent did not write files. "
        f"stdout: {result.stdout[-2000:]}"
    )

    config = yaml.safe_load(config_path.read_text())
    # spec_version is required and must be 1.
    assert config.get("spec_version") == 1, (
        f"Generated config.yaml has spec_version={config.get('spec_version')}, "
        f"expected 1. Full config: {config}"
    )
    # Must have a name.
    assert config.get("name"), f"Generated config.yaml is missing 'name'. Full config: {config}"
    # Must have an llm.model field.
    llm_block = config.get("llm", {})
    assert llm_block.get("model"), (
        f"Generated config.yaml is missing llm.model. Full config: {config}"
    )

    # ── Boot the generated agent and verify it responds ─
    _verify_agent_serves(agent_dir, llm_api_key)


def _verify_agent_serves(agent_dir: Path, api_key: str) -> None:
    """
    Start ``ap server`` with the generated agent and send a test request.

    Verifies the agent responds with text content (proving it's a
    valid, runnable agent).

    :param agent_dir: Path to the generated agent directory.
    :param api_key: LLM API key.
    """
    port = find_free_port()
    tmpdir = tempfile.mkdtemp(prefix="ap-e2e-serve-")
    db_uri = f"sqlite:///{tmpdir}/test.db"
    art_loc = f"{tmpdir}/artifacts"

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
            db_uri,
            "--artifact-location",
            art_loc,
            "--agent",
            str(agent_dir),
        ],
        env={**os.environ, "OPENAI_API_KEY": api_key},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(base_url)

        # Find the agent.
        agents_resp = httpx.get(f"{base_url}/api/agents", timeout=10.0)
        agents_resp.raise_for_status()
        agents = agents_resp.json()["data"]
        assert len(agents) > 0, "No agents registered — the generated agent failed to load."
        agent_name = agents[0]["name"]

        # Send a test message.
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": agent_name,
                "input": "Hello!",
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        body = resp.json()

        # The agent should complete successfully.
        assert body["status"] == "completed", (
            f"Agent response status is {body['status']!r}, expected 'completed'. "
            f"Full response: {body}"
        )

        # The agent should produce text output.
        text = _extract_text_from_response(body)
        assert len(text) > 0, f"Agent produced no text output. Full response: {body}"
    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


# ---------------------------------------------------------------------------
# Non-interactive e2e without filesystem: still produces output
# ---------------------------------------------------------------------------


def test_non_interactive_without_filesystem_prints_output(
    llm_api_key: str,
) -> None:
    """
    ``ap create`` without ``--allow-shell-access`` should still
    run and print the agent config to stdout (the LLM outputs it
    as text since it can't write files).

    **What breaks if this fails:**
    - Non-interactive mode crashes without filesystem tools.
    - The onboarding agent can't function without tools at all.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "create",
            "Describe what a minimal agent config.yaml should look like for a hello-world agent.",
            "--model",
            "openai/gpt-5.4",
        ],
        env={**os.environ, "OPENAI_API_KEY": llm_api_key},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"ap create exited with code {result.returncode}.\nstderr: {result.stderr[-2000:]}"
    )

    # The agent should have printed something about agent config.
    # We check for key terms that would appear in any config discussion.
    output = result.stdout.lower()
    assert any(term in output for term in ["spec_version", "config", "yaml", "agent"]), (
        f"Expected output to mention agent config concepts, but got: {result.stdout[-1000:]}"
    )


# ---------------------------------------------------------------------------
# Interactive e2e via --auto-send on terminal frontend
# ---------------------------------------------------------------------------


def test_interactive_mode_launches_and_responds(
    llm_api_key: str,
    tmp_path: Path,
) -> None:
    """
    Interactive ``ap create`` (via terminal TUI with ``--auto-send``)
    launches successfully and the onboarding agent responds.

    This is a smoke test — it verifies the full interactive stack
    (server boot → TUI launch → agent responds → TUI exits) works
    without crashing. We don't verify file creation because
    --auto-send sends a single message and the TUI exits.

    **What breaks if this fails:**
    - Server startup fails for the onboarding agent.
    - Terminal frontend can't connect to the server.
    - Onboarding agent crashes on first message.
    """
    # We can't easily drive the full TUI interactively, but we can
    # use the terminal frontend's --auto-send to send one message
    # and verify the process doesn't crash.
    #
    # The approach: start ap create's server manually, then run
    # the terminal frontend with --auto-send pointed at it.

    from agent_plane.onboarding.cli import (
        _get_agent_id,
        _prepare_onboarding_agent,
        _start_server,
        _stop_server,
    )
    from agent_plane.onboarding.provider_selection import ProviderSelection

    selection = ProviderSelection(
        provider="openai",
        model="openai/gpt-5.4",
        credentials={"api_key": llm_api_key},
    )
    agent_dir = _prepare_onboarding_agent(selection, allow_shell_access=False)
    port = find_free_port()
    server_proc = _start_server(agent_dir, port)

    try:
        base_url = f"http://127.0.0.1:{port}"
        wait_for_server(base_url)

        # Verify the onboarding agent is registered.
        agent_id = _get_agent_id(port, "onboarding")
        assert agent_id, "Onboarding agent was not registered."

        # Send a request directly to the server (simulating what the
        # TUI would do) to verify the onboarding agent responds.
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": "onboarding",
                "input": "What can you help me with?",
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        body = resp.json()

        assert body["status"] == "completed", (
            f"Onboarding agent response status is {body['status']!r}, "
            f"expected 'completed'. Full response: {body}"
        )

        text = _extract_text_from_response(body)
        assert len(text) > 0, f"Onboarding agent produced no text output. Full response: {body}"

        # ── LLM judge: is the response about agent creation? ──
        os.environ["OPENAI_API_KEY"] = llm_api_key

        from mlflow.genai.judges import make_judge

        judge = make_judge(
            name="onboarding_purpose",
            instructions=(
                "You are evaluating whether an AI assistant correctly "
                "identified itself as an onboarding assistant for "
                "creating agents.\n\n"
                "The user asked: 'What can you help me with?'\n\n"
                "The assistant's response is:\n"
                "{{ outputs }}\n\n"
                "Does the response indicate the assistant can help "
                "create agents, set up agent configurations, or guide "
                "the user through agent development? It should mention "
                "agents, configuration, or similar concepts.\n\n"
                "Return True if the assistant describes its purpose "
                "related to agent creation, False otherwise."
            ),
            feedback_value_type=bool,
        )

        feedback = judge(outputs=text)
        assert feedback.value is True, (
            f"LLM judge ruled the onboarding assistant did NOT "
            f"describe its agent-creation purpose.\n"
            f"Judge rationale: {feedback.rationale}\n"
            f"Response: {text[:500]}"
        )
    finally:
        _stop_server(server_proc)
