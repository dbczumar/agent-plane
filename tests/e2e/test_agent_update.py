"""E2E test: zero-downtime agent update.

Verifies that an in-flight request on the old agent version completes
successfully, and a new request after the update uses the new version
(observable via changed instructions that affect the response content).

Usage::

    pytest tests/e2e/test_agent_update.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from tests.e2e.conftest import poll_until_terminal

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHER_DIR = _REPO_ROOT / "examples" / "agents" / "archer"

# Marker phrase injected into v2 instructions so we can verify
# the v2 response was produced by the updated spec.
_V2_MARKER = "ZEBRAFINCH"


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


def _upload_agent_with_id(
    client: httpx.Client,
    agent_dir: Path,
) -> dict[str, Any]:
    """
    Upload an agent bundle and return the full response body
    (including ``id``).

    :param client: HTTP client pointed at the server.
    :param agent_dir: Path to the agent directory.
    :returns: The agent response JSON with ``id``, ``name``,
        ``version``, etc.
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
        resp.raise_for_status()
        return resp.json()
    finally:
        os.unlink(tmp_path)


def _build_updated_bundle(
    agent_dir: Path,
    config_overrides: dict[str, Any],
) -> bytes:
    """
    Build a tarball from an agent directory with config.yaml
    fields overridden.

    Reads the original config.yaml, merges the overrides, and
    writes the modified config into the tarball. All other files
    are included as-is.

    :param agent_dir: Path to the original agent directory.
    :param config_overrides: Dict of fields to merge into
        config.yaml, e.g. ``{"description": "v2"}``.
    :returns: Raw bytes of the ``.tar.gz`` bundle.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in Path(agent_dir).rglob("*"):
            if not item.is_file():
                continue
            arcname = str(item.relative_to(agent_dir))
            if item.name == "config.yaml" and item.parent == agent_dir:
                # Override the root config.yaml
                config = yaml.safe_load(item.read_text())
                config.update(config_overrides)
                data = yaml.dump(config).encode()
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(str(item), arcname=arcname)
    return buf.getvalue()


def _update_agent(
    client: httpx.Client,
    agent_id: str,
    bundle_bytes: bytes,
) -> dict[str, Any]:
    """
    PUT a new bundle to update an existing agent.

    :param client: HTTP client pointed at the server.
    :param agent_id: The agent's ID.
    :param bundle_bytes: Raw bytes of the new ``.tar.gz`` bundle.
    :returns: The updated agent response JSON.
    """
    resp = client.put(
        f"/api/agents/{agent_id}",
        files={
            "bundle": (
                "agent.tar.gz",
                bundle_bytes,
                "application/gzip",
            ),
        },
    )
    resp.raise_for_status()
    return resp.json()


def test_update_agent_zero_downtime(
    http_client: httpx.Client,
    llm_api_key: str,
) -> None:
    """
    Verifies that the update endpoint doesn't disrupt in-flight
    requests and that new requests use the updated spec.

    **What this test proves:**
    - The PUT endpoint succeeds while a background request is
      running (the server doesn't crash or deadlock).
    - A request created after the update uses the new spec
      (verified via a marker phrase in instructions).
    - The in-flight request completes without error.

    **What this test does NOT prove (inherent E2E limitation):**
    - It does not guarantee the v1 request was mid-LLM-call when
      the update happened. The time.sleep + status check is
      best-effort, not a deterministic synchronization gate.
      True mid-execution concurrent update testing requires the
      mock LLM's blocking mechanism, which is not available in
      E2E tests with a real LLM.
    - The marker assertion depends on the LLM following the
      injected instruction. If the LLM ignores it, the test
      gives a false negative, not a false positive.

    Steps:
    1. Upload archer agent (version 1).
    2. Send a long-running background request (v1 instructions).
    3. While in_progress, PUT a new bundle with modified
       instructions containing a marker phrase (version 2).
    4. Send a second request — the marker must appear.
    5. Both requests complete successfully.
    6. V1 response does NOT contain the marker.
    7. V2 response DOES contain the marker.
    8. Agent metadata shows version=2 and updated_at is set.
    """
    # Step 1: Upload archer (v1)
    created = _upload_agent_with_id(http_client, _ARCHER_DIR)
    agent_id = created["id"]
    assert created["version"] == 1

    # Step 2: Long-running request on v1
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": "archer",
            "input": (
                "Research the history of quantum computing. "
                "Cover at least 5 major milestones. Be thorough."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    response_id_1 = resp1.json()["id"]

    # Best-effort wait for the workflow to start processing.
    # This is NOT a deterministic synchronization gate — the
    # request may finish before or after this sleep. The status
    # check below catches the "finished too fast" case and fails
    # the test explicitly rather than silently passing a
    # sequential scenario. A real LLM E2E test cannot use the
    # mock LLM's blocking mechanism.
    time.sleep(2)

    # Confirm request is still running — if it already completed,
    # the test can't prove the update happened mid-flight.
    check = http_client.get(f"/v1/responses/{response_id_1}")
    status = check.json()["status"]
    assert status in ("queued", "in_progress"), (
        f"Expected v1 request to still be running but got "
        f"{status!r}. The request completed too fast to test "
        f"concurrent update behavior."
    )

    # Step 3: Update agent to v2 with marker in instructions.
    # The inline instructions override the file-based AGENTS.md.
    v2_bundle = _build_updated_bundle(
        _ARCHER_DIR,
        {
            "description": "Updated archer v2 for e2e test",
            "instructions": (
                f"You MUST include the word '{_V2_MARKER}' "
                f"somewhere in every response you give. This is "
                f"a mandatory requirement."
            ),
        },
    )
    updated = _update_agent(http_client, agent_id, v2_bundle)
    assert updated["version"] == 2

    # Step 4: New request on v2 — ask something simple so
    # the marker instruction is the main driver of content.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": "archer",
            "input": "What is 2+2? Answer briefly.",
            "background": True,
        },
    )
    resp2.raise_for_status()
    response_id_2 = resp2.json()["id"]

    # Step 5: Poll both to terminal state
    body1 = poll_until_terminal(http_client, response_id_1, timeout=300)
    body2 = poll_until_terminal(http_client, response_id_2, timeout=300)

    # Both requests completed — neither was disrupted
    assert body1["status"] == "completed", (
        f"V1 request failed with status {body1['status']!r}. "
        f"The update should not disrupt in-flight requests. "
        f"Output: {body1.get('output', [])}"
    )
    assert body2["status"] == "completed", (
        f"V2 request failed with status {body2['status']!r}. Output: {body2.get('output', [])}"
    )

    # Step 6: V1 response should NOT contain the marker —
    # it was served by the old spec before the update.
    v1_text = _extract_all_text(body1)
    assert _V2_MARKER not in v1_text, (
        f"V1 response unexpectedly contains the v2 marker "
        f"'{_V2_MARKER}'. This means the in-flight request "
        f"picked up the new spec instead of using the one it "
        f"loaded at start. First 500 chars: {v1_text[:500]}"
    )

    # Step 7: V2 response MUST contain the marker — it was
    # served by the updated spec with the injected instruction.
    # NOTE: This assertion depends on the LLM following the
    # mandatory instruction. A false negative (test fails but
    # spec was loaded correctly) is possible if the LLM ignores
    # the instruction. A false positive is NOT possible — the
    # marker only exists in v2's instructions.
    v2_text = _extract_all_text(body2)
    assert _V2_MARKER in v2_text, (
        f"V2 response does NOT contain the marker "
        f"'{_V2_MARKER}'. This means the new request did not "
        f"use the updated spec's instructions. The cache swap "
        f"may have failed. First 500 chars: {v2_text[:500]}"
    )

    # Step 8: Agent metadata reflects the update
    agent_resp = http_client.get(f"/api/agents/{agent_id}")
    agent_resp.raise_for_status()
    agent = agent_resp.json()
    assert agent["version"] == 2
    assert agent["updated_at"] is not None
