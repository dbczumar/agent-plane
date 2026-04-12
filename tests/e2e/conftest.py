"""Fixtures for end-to-end tests with real LLM and real server.

Usage::

    pytest tests/e2e/ --llm-api-key $(cat /tmp/mykey) -v

These tests start a real ``ap server`` subprocess, upload real
agent bundles, and call real LLM APIs. They are excluded from
the default ``pytest`` run via ``--ignore=tests/e2e`` in
``pyproject.toml``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

# Agent bundle directories relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODER_DIR = _REPO_ROOT / "examples" / "agents" / "coder"
_ARCHER_DIR = _REPO_ROOT / "examples" / "agents" / "archer"
_CLAUDE_CODER_DIR = _REPO_ROOT / "examples" / "agents" / "claude-coder"
_OPENAI_CODER_DIR = _REPO_ROOT / "examples" / "agents" / "openai-coder"


def find_free_port() -> int:
    """
    Find a free TCP port by binding to port 0.

    :returns: An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(base_url: str, timeout: float = 20.0) -> None:
    """
    Poll until the server responds on its conversations endpoint.

    :param base_url: Server base URL, e.g. ``"http://127.0.0.1:8000"``.
    :param timeout: Max seconds to wait.
    :raises RuntimeError: If the server doesn't respond.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/v1/conversations", timeout=2.0)
            if resp.status_code in (200, 404):
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not respond within {timeout}s")


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register ``--llm-api-key`` CLI option.

    :param parser: The pytest argument parser.
    """
    parser.addoption(
        "--llm-api-key",
        action="store",
        required=True,
        help="OpenAI API key for real LLM calls.",
    )


@pytest.fixture(scope="session")
def llm_api_key(request: pytest.FixtureRequest) -> str:
    """
    The LLM API key from ``--llm-api-key``.

    :param request: Pytest request object.
    :returns: The API key string.
    """
    key: str = request.config.getoption("--llm-api-key")
    return key


@pytest.fixture(scope="session")
def live_server(
    llm_api_key: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """
    Start a real ``ap server`` subprocess and yield its base URL.

    The server runs on a random high port. The fixture waits
    for the health endpoint before yielding, and kills the
    process on teardown.

    :param llm_api_key: The API key for the LLM.
    :param tmp_path_factory: Pytest temp path factory for the DB.
    :returns: The server's base URL, e.g. ``"http://localhost:18501"``.
    """
    port = 18501
    db_path = tmp_path_factory.mktemp("e2e") / "e2e.db"
    env = {
        **os.environ,
        "OPENAI_API_KEY": llm_api_key,
        "AP_DB_URI": f"sqlite:///{db_path}",
    }
    proc = subprocess.Popen(
        ["ap", "server", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://localhost:{port}"

    # Wait for health.
    for _ in range(60):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        raise RuntimeError(f"Server didn't start within 30s. Output:\n{stdout}")

    yield base_url

    # Teardown: kill server.
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def http_client(live_server: str) -> Iterator[httpx.Client]:
    """
    HTTP client pointed at the live server.

    :param live_server: The server base URL.
    :returns: An ``httpx.Client`` with long timeout.
    """
    with httpx.Client(base_url=live_server, timeout=300) as client:
        yield client


def _upload_agent(
    client: httpx.Client,
    agent_dir: Path,
) -> str:
    """
    Upload an agent bundle from a directory.

    :param client: HTTP client pointed at the server.
    :param agent_dir: Path to the agent directory.
    :returns: The agent ID or name.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(agent_dir), arcname=".")
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = client.post(
                "/api/agents",
                files={
                    "bundle": (
                        "agent.tar.gz",
                        f,
                        "application/gzip",
                    ),
                },
            )
        if resp.status_code == 409:
            return agent_dir.name
        resp.raise_for_status()
        return resp.json()["name"]
    finally:
        os.unlink(tmp_path)


@pytest.fixture(scope="session")
def coder_agent(http_client: httpx.Client) -> str:
    """
    Upload the coder agent (with reviewer sub-agent) and
    return its name.

    :param http_client: HTTP client pointed at the server.
    :returns: The agent name, e.g. ``"coder"``.
    """
    return _upload_agent(http_client, _CODER_DIR)


@pytest.fixture(scope="session")
def archer_agent(http_client: httpx.Client) -> str:
    """
    Upload the archer agent (with fact_checker and summarizer
    sub-agents) and return its name.

    :param http_client: HTTP client pointed at the server.
    :returns: The agent name, e.g. ``"archer"``.
    """
    return _upload_agent(http_client, _ARCHER_DIR)


@pytest.fixture(scope="session")
def claude_coder_agent(http_client: httpx.Client) -> str:
    """
    Upload the claude-coder agent and return its name.

    The Claude Agent SDK authenticates via the ``claude`` CLI's
    own session (OAuth), so no explicit API key env var is required.

    :param http_client: HTTP client pointed at the server.
    :returns: The agent name, ``"claude-coder"``.
    """
    return _upload_agent(http_client, _CLAUDE_CODER_DIR)


@pytest.fixture(scope="session")
def openai_coder_agent(http_client: httpx.Client) -> str:
    """
    Upload the openai-coder agent (with reviewer sub-agent
    and skills) and return its name.

    :param http_client: HTTP client pointed at the server.
    :returns: The agent name, ``"openai-coder"``.
    """
    return _upload_agent(http_client, _OPENAI_CODER_DIR)


@pytest.fixture(scope="session")
def sample_code_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """
    Create a temp directory with sample Python files for the
    reviewer sub-agent to inspect.

    :param tmp_path_factory: Pytest temp path factory.
    :returns: Path to the directory containing sample files.
    """
    d = tmp_path_factory.mktemp("sample_code")

    # A module with a deliberate bug (division by zero risk).
    (d / "calculator.py").write_text(
        "def divide(a: float, b: float) -> float:\n"
        '    """Divide a by b."""\n'
        "    return a / b\n"
        "\n"
        "\n"
        "def average(numbers: list[float]) -> float:\n"
        '    """Return the mean of a list of numbers."""\n'
        "    total = sum(numbers)\n"
        "    return divide(total, len(numbers))\n"
    )

    # A test file with incomplete coverage.
    (d / "test_calculator.py").write_text(
        "from calculator import divide, average\n"
        "\n"
        "\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5.0\n"
        "\n"
        "\n"
        "def test_average():\n"
        "    assert average([1, 2, 3]) == 2.0\n"
        "    # Missing: test for empty list (ZeroDivisionError)\n"
    )

    # A utility with a bare except and hardcoded path.
    (d / "utils.py").write_text(
        "import json\n"
        "import os\n"
        "\n"
        "\n"
        "def load_config():\n"
        "    try:\n"
        '        with open("/etc/myapp/config.json") as f:\n'
        "            return json.load(f)\n"
        "    except:\n"
        "        return {}\n"
        "\n"
        "\n"
        "def get_temp_dir():\n"
        '    return os.environ.get("TEMP_DIR", "/tmp/myapp")\n'
    )

    return d


def poll_until_terminal(
    client: httpx.Client,
    response_id: str,
    timeout: float = 300,
) -> dict[str, Any]:
    """
    Poll GET /v1/responses/{id} until terminal state.

    :param client: HTTP client.
    :param response_id: The response ID to poll.
    :param timeout: Max seconds to wait.
    :returns: The terminal response body.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/responses/{response_id}")
        resp.raise_for_status()
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.5)
    raise AssertionError(f"Response {response_id} didn't complete within {timeout}s")


def poll_for_pending_tool_calls(
    client: httpx.Client,
    response_id: str,
    timeout: float = 120,
) -> list[dict[str, Any]]:
    """
    Poll until ``action_required`` function_calls appear.

    :param client: HTTP client.
    :param response_id: The root response ID.
    :param timeout: Max seconds to wait.
    :returns: List of action_required function_call items.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        pending = [
            item
            for item in body.get("output", [])
            if item.get("type") == "function_call" and item.get("status") == "action_required"
        ]
        if pending:
            return pending
        if body["status"] in ("completed", "failed"):
            return []
        time.sleep(0.5)
    return []
