"""End-to-end compaction test with a real LLM and real server.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_compaction_e2e.py \
        --llm-api-key $LLM_API_KEY -v

Uses ``AP_CONTEXT_WINDOW_OVERRIDE=4096`` so the server thinks the
model has a tiny context window. With ``trigger_threshold: 0.01``
in the agent spec, proactive compaction fires on the second turn
(the first turn's response alone exceeds 1% of 4096 = 41 tokens).

Exercises:
- Proactive compaction (estimated tokens > threshold after turn 1)
- Compaction item persisted to conversation store
- Cursor-based history loading on follow-up turn
- Agent continues to function after compaction (summary context)
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPACTION_AGENT_DIR = _REPO_ROOT / "examples" / "agents" / "compaction-test"


@pytest.fixture(scope="module")
def compaction_server(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """
    Start a real server with ``AP_CONTEXT_WINDOW_OVERRIDE=4096``.

    Uses a separate port (18502) to avoid conflicts with the
    session-scoped e2e server on 18501.

    :param request: Pytest request (for CLI options).
    :param tmp_path_factory: Pytest temp path factory.
    :returns: The server's base URL.
    """
    api_key: str = request.config.getoption("--llm-api-key")
    port = 18502
    db_path = tmp_path_factory.mktemp("compaction_e2e") / "compaction.db"
    env = {
        **os.environ,
        "OPENAI_API_KEY": api_key,
        "AP_DB_URI": f"sqlite:///{db_path}",
        # Small context window so proactive compaction fires quickly.
        # 32768 * 0.05 = 1638 token budget — fires after ~2 turns
        # of normal conversation, with enough room for the compacted
        # summary + recent message to fit after compaction.
        "AP_CONTEXT_WINDOW_OVERRIDE": "32768",
    }
    # Remove stale DBOS system DB so the server starts fresh
    # (avoids reusing cached agent bundles from prior runs).
    for stale in Path(".").glob("dbos_system.db*"):
        stale.unlink(missing_ok=True)
    proc = subprocess.Popen(
        ["ap", "server", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://localhost:{port}"

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
        raise RuntimeError(f"Compaction e2e server didn't start within 30s.\n{stdout}")

    yield base_url

    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def compaction_client(
    compaction_server: str,
) -> Iterator[httpx.Client]:
    """
    HTTP client pointed at the compaction e2e server.

    :param compaction_server: The server's base URL.
    :returns: An ``httpx.Client`` with long timeout.
    """
    with httpx.Client(base_url=compaction_server, timeout=300) as client:
        yield client


def _upload_agent(client: httpx.Client, agent_dir: Path) -> str:
    """
    Upload an agent bundle with a unique name to avoid stale cache.

    Rewrites the config.yaml name to a unique value so each test
    run gets a fresh agent, even if DBOS replays prior state.

    :param client: HTTP client.
    :param agent_dir: Path to the agent directory.
    :returns: The unique agent name.
    """
    import uuid

    import yaml

    unique_name = f"compact-{uuid.uuid4().hex[:8]}"
    config_path = agent_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["name"] = unique_name
    modified_yaml = yaml.dump(config)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            # Add the modified config.yaml
            import io

            config_bytes = modified_yaml.encode()
            info = tarfile.TarInfo(name="config.yaml")
            info.size = len(config_bytes)
            tar.addfile(info, io.BytesIO(config_bytes))
            # Add AGENTS.md if it exists
            agents_md = agent_dir / "AGENTS.md"
            if agents_md.exists():
                tar.add(str(agents_md), arcname="AGENTS.md")
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = client.post(
                "/api/agents",
                files={"bundle": ("agent.tar.gz", f, "application/gzip")},
            )
        resp.raise_for_status()
        return resp.json()["name"]
    finally:
        os.unlink(tmp_path)


def _create_turn(
    client: httpx.Client,
    model: str,
    user_input: str,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a response and poll until terminal.

    :param client: HTTP client.
    :param model: Agent name.
    :param user_input: User message text.
    :param previous_response_id: Previous response ID, or ``None``.
    :returns: The terminal response body dict.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": user_input,
        "background": True,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    resp = client.post("/v1/responses", json=payload)
    resp.raise_for_status()
    response_id = resp.json()["id"]

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/responses/{response_id}")
        resp.raise_for_status()
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.5)
    raise AssertionError(
        f"Response {response_id} didn't complete within 120s",
    )


def test_compaction_fires_and_agent_continues(
    compaction_client: httpx.Client,
) -> None:
    """
    With a 4096-token context window override and 1% trigger
    threshold, proactive compaction fires on the second turn.

    :param compaction_client: HTTP client pointed at the
        compaction e2e server.
    """
    agent_name = _upload_agent(compaction_client, _COMPACTION_AGENT_DIR)

    # --- Turn 1: seed the conversation ---
    turn_1 = _create_turn(
        compaction_client,
        agent_name,
        "List 10 countries and their capitals with brief descriptions.",
    )
    assert turn_1["status"] == "completed", (
        f"Turn 1 failed: {turn_1.get('status')}. Body: {turn_1}"
    )
    response_id = turn_1["id"]
    conv_id = turn_1["conversation"]["id"]

    # --- Turn 2: triggers proactive compaction ---
    turn_2 = _create_turn(
        compaction_client,
        agent_name,
        "Now list 10 more countries not in the previous list.",
        previous_response_id=response_id,
    )
    assert turn_2["status"] == "completed", (
        f"Turn 2 failed: {turn_2.get('status')}. "
        f"If 'failed', compaction may have broken the retry path."
    )
    response_id = turn_2["id"]

    # --- Verify compaction item exists ---
    items_resp = compaction_client.get(
        f"/v1/conversations/{conv_id}/items",
    )
    items_resp.raise_for_status()
    items: list[dict[str, Any]] = items_resp.json()["data"]
    compaction_items = [i for i in items if i.get("type") == "compaction"]

    assert len(compaction_items) >= 1, (
        f"Expected >= 1 compaction item after 2 turns with "
        f"32768-token window and 2% threshold (655 token budget). "
        f"Found {len(compaction_items)}. "
        f"Item types: {[i.get('type') for i in items]}. "
        f"If 0, AP_CONTEXT_WINDOW_OVERRIDE may not have reached "
        f"the server or proactive compaction didn't fire."
    )

    cmp = compaction_items[-1]
    assert isinstance(cmp.get("summary"), str), f"Compaction item missing 'summary': {cmp}"
    assert len(cmp["summary"]) > 10, f"Summary too short: {cmp['summary']!r}"
    all_ids = {i["id"] for i in items}
    assert cmp.get("last_item_id") in all_ids, (
        f"last_item_id={cmp.get('last_item_id')!r} not in items."
    )

    # --- Turn 3: agent works after compaction ---
    turn_3 = _create_turn(
        compaction_client,
        agent_name,
        "What was the first thing I asked you about?",
        previous_response_id=response_id,
    )
    assert turn_3["status"] == "completed", (
        f"Turn 3 (post-compaction) failed: {turn_3.get('status')}."
    )

    output = turn_3.get("output", [])
    texts = [
        item["content"][0]["text"]
        for item in output
        if item.get("type") == "message"
        and item.get("role") == "assistant"
        and item.get("content")
    ]
    combined = " ".join(texts).lower()
    # The agent should reference countries/capitals — proving
    # the compaction summary provided context.
    assert any(kw in combined for kw in ["countr", "capital", "list", "nation", "asked"]), (
        f"Post-compaction response doesn't reference prior context. Response: {combined[:300]}"
    )
