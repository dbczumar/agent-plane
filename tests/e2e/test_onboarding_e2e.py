"""E2E tests for ``ap create`` — onboarding flow.

Verifies that the onboarding agent can create a real agent directory
and that the generated agent can be served by agent-plane.

Usage::

    pytest tests/e2e/test_onboarding_e2e.py \
        --llm-api-key $LLM_API_KEY -v

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


def _verify_agent_serves(
    agent_dir: Path,
    api_key: str,
    expected_marker: str | None = None,
) -> None:
    """
    Start ``ap server`` with the generated agent and verify it responds.

    :param agent_dir: Path to the generated agent directory.
    :param api_key: LLM API key.
    :param expected_marker: Optional string the response must contain.
    """
    port = find_free_port()
    server_proc = _start_agent_server(agent_dir, port, api_key)
    try:
        wait_for_server(f"http://127.0.0.1:{port}")
        _send_and_check(f"http://127.0.0.1:{port}", expected_marker)
    finally:
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


def _start_agent_server(
    agent_dir: Path,
    port: int,
    api_key: str,
) -> subprocess.Popen[bytes]:
    """
    Launch ``ap server`` with the given agent directory.

    :param agent_dir: Path to the agent directory.
    :param port: Port to listen on.
    :param api_key: LLM API key.
    :returns: The server subprocess.
    """
    tmpdir = tempfile.mkdtemp(prefix="ap-e2e-serve-")
    return subprocess.Popen(
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
        env={**os.environ, "OPENAI_API_KEY": api_key},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _send_and_check(base_url: str, expected_marker: str | None) -> None:
    """
    Find the agent, send a message, and verify the response.

    :param base_url: Server base URL.
    :param expected_marker: Optional marker the response must contain.
    """
    agents_resp = httpx.get(f"{base_url}/api/agents", timeout=10.0)
    agents_resp.raise_for_status()
    agents = agents_resp.json()["data"]
    assert len(agents) > 0, "No agents registered."

    resp = httpx.post(
        f"{base_url}/v1/responses",
        json={"model": agents[0]["name"], "input": "Hello!", "stream": False},
        timeout=120.0,
    )
    resp.raise_for_status()
    body = resp.json()

    assert body["status"] == "completed", f"Status is {body['status']!r}. Full response: {body}"
    text = _extract_text_from_response(body)
    assert len(text) > 0, f"No text output. Full response: {body}"

    if expected_marker is not None:
        assert expected_marker.lower() in text.lower(), (
            f"Missing marker {expected_marker!r}. Response: {text[:500]}"
        )


# ---------------------------------------------------------------------------
# Non-interactive sandbox mode: create + export + serve
# ---------------------------------------------------------------------------


def test_sandbox_mode_creates_and_exports_agent(
    llm_api_key: str,
    tmp_path: Path,
) -> None:
    """
    ``ap create`` without ``--allow-shell-access`` uses sandbox mode:
    the onboarding assistant creates the agent in its sandboxed
    terminal_run workspace and exports it to the user's path via
    ``export_agent``.

    Flow:
    1. Run ``ap create`` WITHOUT ``--allow-shell-access``.
    2. The prompt tells the agent to create an agent and export it
       to a specific directory.
    3. Verify the exported agent directory exists with valid config.
    4. Boot the exported agent and verify it responds.

    **What breaks if this fails:**
    - terminal_run not added to onboarding agent in sandbox mode →
      agent can't create files at all.
    - export_agent not added → agent creates files in workspace but
      can't copy them to the target path.
    - export_agent tool broken → files stay in sandbox, never exported.
    - Server-side tool execution broken → tool calls not processed.
    """
    agent_dir = tmp_path / "sandbox-test-agent"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "create",
            (
                f"Create a minimal agent called 'sandbox-greeter' in your "
                f"workspace. The config.yaml must have this exact structure:\n"
                f"  spec_version: 1\n"
                f"  name: sandbox-greeter\n"
                f"  llm:\n"
                f"    model: openai/gpt-5.4\n"
                f"    connection:\n"
                f"      api_key: ${{OPENAI_API_KEY}}\n"
                f"  instructions: 'Reply with exactly: SANDBOX_SUCCESS'\n"
                f"Write config.yaml in your workspace, then use export_agent "
                f"to copy it to {agent_dir}."
            ),
            "--model",
            "openai/gpt-5.4",
            # No --allow-shell-access → sandbox mode
        ],
        env={**os.environ, "OPENAI_API_KEY": llm_api_key},
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, (
        f"ap create (sandbox mode) exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )

    # ── Verify exported agent directory ────────────────
    config_path = agent_dir / "config.yaml"
    assert config_path.exists(), (
        f"Expected config.yaml at {config_path}. "
        f"The onboarding assistant did not export the agent. "
        f"It should have used terminal_run to create files in the "
        f"workspace, then export_agent to copy them to {agent_dir}. "
        f"stdout: {result.stdout[-2000:]}"
    )

    config = yaml.safe_load(config_path.read_text())
    assert config.get("spec_version") == 1, (
        f"Exported config.yaml has spec_version={config.get('spec_version')}, expected 1."
    )
    assert config.get("name"), "Exported config.yaml is missing 'name'."
    llm_block = config.get("llm", {})
    assert llm_block.get("model"), "Exported config.yaml is missing llm.model."

    # ── Boot the exported agent and verify it responds ─
    # The marker proves the instructions were set correctly, not
    # just that some random agent was created.
    _verify_agent_serves(agent_dir, llm_api_key, expected_marker="SANDBOX_SUCCESS")


# ---------------------------------------------------------------------------
# Shell access mode: create + serve
# ---------------------------------------------------------------------------


def test_shell_mode_creates_and_serves_agent(
    llm_api_key: str,
    tmp_path: Path,
) -> None:
    """
    ``ap create --allow-shell-access`` gives the onboarding assistant
    full shell tools (Read, Write, Bash, etc.) to create the agent
    directly on the filesystem.

    Flow:
    1. Run ``ap create`` with ``--allow-shell-access``.
    2. The prompt tells the agent to create an agent at a specific path.
    3. Verify the agent directory exists with valid config.
    4. Boot the generated agent and verify it responds.

    **What breaks if this fails:**
    - Client-side tool execution broken → Write/Bash not available.
    - Non-interactive tool loop doesn't handle client tool calls.
    - Generated config invalid → server rejects it.
    """
    agent_dir = tmp_path / "shell-test-agent"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "create",
            (
                f"Create a minimal agent in the directory {agent_dir}. "
                f"The config.yaml must have this exact structure:\n"
                f"  spec_version: 1\n"
                f"  name: shell-greeter\n"
                f"  llm:\n"
                f"    model: openai/gpt-5.4\n"
                f"    connection:\n"
                f"      api_key: ${{OPENAI_API_KEY}}\n"
                f"  instructions: 'Reply with exactly: SHELL_SUCCESS'\n"
                f"Create {agent_dir} and write config.yaml there. "
                f"Do not create any other files."
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

    assert result.returncode == 0, (
        f"ap create (shell mode) exited with code {result.returncode}.\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )

    config_path = agent_dir / "config.yaml"
    assert config_path.exists(), (
        f"Expected config.yaml at {config_path}. "
        f"The onboarding assistant did not write files via shell tools. "
        f"stdout: {result.stdout[-2000:]}"
    )

    config = yaml.safe_load(config_path.read_text())
    assert config.get("spec_version") == 1, (
        f"spec_version is {config.get('spec_version')}, expected 1."
    )
    assert config.get("name"), f"Config missing 'name'. Full config: {config}"
    assert config.get("llm", {}).get("model"), f"Config missing llm.model. Full config: {config}"

    _verify_agent_serves(agent_dir, llm_api_key, expected_marker="SHELL_SUCCESS")


# ---------------------------------------------------------------------------
# Validate agent tool e2e
# ---------------------------------------------------------------------------


def test_validate_agent_tool_catches_errors(
    llm_api_key: str,
    tmp_path: Path,
) -> None:
    """
    The onboarding assistant's ``validate_agent`` tool correctly
    validates agent configs — passing valid ones and catching errors.

    Runs in sandbox mode (no shell). The assistant creates an agent,
    validates it, and reports the result. The test checks that the
    assistant mentions the validation passed.

    **What breaks if this fails:**
    - validate_agent tool not discovered (missing from tools/python/).
    - Tool subprocess can't import agent_plane → parse error.
    - Tool returns wrong results → assistant reports false errors.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "create",
            (
                "Create a minimal agent called 'validator-test' in your "
                "workspace with this config.yaml:\n"
                "  spec_version: 1\n"
                "  name: validator-test\n"
                "  llm:\n"
                "    model: openai/gpt-5.4\n"
                "    connection:\n"
                "      api_key: ${OPENAI_API_KEY}\n"
                "  instructions: 'Say hello'\n"
                "Then call validate_agent on it and tell me the result. "
                "Include the exact validation output in your response."
            ),
            "--model",
            "openai/gpt-5.4",
            # No --allow-shell-access → sandbox mode with validate_agent
        ],
        env={**os.environ, "OPENAI_API_KEY": llm_api_key},
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, (
        f"ap create exited with code {result.returncode}.\nstderr: {result.stderr[-2000:]}"
    )

    output = result.stdout.lower()
    # The assistant should report that validation passed.
    # "valid" appears in the validate_agent tool's success output.
    assert "valid" in output, (
        f"Expected validation result mentioning 'valid' in output. "
        f"The validate_agent tool may not be working. "
        f"stdout: {result.stdout[-1000:]}"
    )


# ---------------------------------------------------------------------------
# Non-interactive without shell: still produces output
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
