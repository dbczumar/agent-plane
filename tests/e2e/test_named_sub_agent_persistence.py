"""End-to-end tests for the Phase 4 named-sub-agent pipeline.

Real-LLM coverage of:

* ``test_spawn_named_sub_agent_e2e`` — the LLM picks up the
  ``spawn_sub_agent(type, name, input)`` signature and a child
  conversation persists with title="<type>:<name>".
* ``test_send_to_named_sub_agent_continuation_e2e`` — turn 2
  uses ``send_to_sub_agent`` to continue the existing child;
  the child's history accumulates across turns.
* ``test_ambient_hint_steers_followup_to_send_e2e`` — turn 2's
  user prompt is neutral; the LLM uses the ambient hint
  ("Open sub-agents:") to choose ``send_to_sub_agent`` over a
  duplicate spawn (the critical D6 test — if it fails, named
  persistence is useless because the LLM forgets across turns).
* ``test_parallel_named_sub_agents_e2e`` — both researcher
  ("first") and summarizer ("second") in one turn; both
  markers reach the final reply.
* ``test_cross_parent_named_isolation_e2e`` — same name in two
  separate top-level conversations doesn't leak.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_named_sub_agent_persistence.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path

import httpx
import pytest

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_NAMED_FIXTURE = _FIXTURES_DIR / "named-sub-agent-test"


def _upload(http_client: httpx.Client, agent_dir: Path) -> str:
    """
    Upload an agent bundle from a directory tree.

    :param http_client: HTTP client pointed at the live server.
    :param agent_dir: Directory containing config.yaml.
    :returns: The agent's name.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(agent_dir), arcname=".")
        bundle_path = tmp.name
    try:
        with open(bundle_path, "rb") as f:
            resp = http_client.post(
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
        Path(bundle_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def named_sub_agent_test_agent(http_client: httpx.Client) -> str:
    """Upload the named-sub-agent-test fixture (parent + 2 sub-agents)."""
    return _upload(http_client, _NAMED_FIXTURE)


def _create_response_blocking(
    http_client: httpx.Client,
    *,
    model: str,
    user_text: str,
    timeout_s: float = 240.0,
    previous_response_id: str | None = None,
) -> dict:
    """
    POST a response, poll until terminal, return the final body.

    :param http_client: HTTP client.
    :param model: Agent name to invoke.
    :param user_text: Plain-text input message.
    :param timeout_s: Max seconds to wait. Higher than Phase 3
        because some tests issue 2-turn exchanges.
    :param previous_response_id: Optional previous response id
        for multi-turn flows.
    :returns: The terminal response JSON.
    """
    payload: dict = {
        "model": model,
        "input": user_text,
        "background": True,
        "store": True,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    resp = http_client.post("/v1/responses", json=payload)
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        get_resp = http_client.get(f"/v1/responses/{response_id}")
        get_resp.raise_for_status()
        body = get_resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(1.0)
    raise AssertionError(
        f"Response {response_id} did not complete within {timeout_s}s; "
        f"final status was {body.get('status')!r}."
    )


def _final_text(response_body: dict) -> str:
    """Concatenate assistant message text from a response body."""
    parts: list[str] = []
    for item in response_body.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n\n".join(parts)


def _conversation_items(http_client: httpx.Client, conversation_id: str) -> list[dict]:
    """Fetch conversation items in store order."""
    resp = http_client.get(
        f"/v1/conversations/{conversation_id}/items",
        params={"limit": 100},
    )
    resp.raise_for_status()
    data: list[dict] = resp.json()["data"]
    return data


def _function_call_names(items: list[dict]) -> list[str]:
    """Return the names of all function_call items in order."""
    return [item.get("name", "") for item in items if item.get("type") == "function_call"]


# ─── Tests ───────────────────────────────────────────────────


def test_spawn_named_sub_agent_e2e(
    http_client: httpx.Client,
    named_sub_agent_test_agent: str,
) -> None:
    """
    Real LLM dispatches ``spawn_sub_agent(type, name, input)``
    and the child conversation persists with the documented
    title shape ``"<type>:<name>"``. Without this the
    follow-up ``send_to_sub_agent`` lookup wouldn't find the
    child.
    """
    body = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Spawn the researcher sub-agent named 'auth' with "
            "input 'What are common auth patterns?'. Quote the "
            "literal marker the sub-agent returns."
        ),
    )
    assert body["status"] == "completed", (
        f"Turn did not complete: status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    assert "RESEARCHER_PHASE4_OK" in final, (
        f"Expected the researcher marker in the final response. Got: {final!r}"
    )

    # Verify the child conversation was titled "researcher:auth".
    conv_id = body["conversation"]["id"]
    items = _conversation_items(http_client, conv_id)
    spawn_calls = [
        item
        for item in items
        if item.get("type") == "function_call" and item.get("name") == "spawn_sub_agent"
    ]
    assert len(spawn_calls) >= 1, (
        f"Expected at least 1 spawn_sub_agent call; got {_function_call_names(items)}"
    )
    # The spawn arguments must include name="auth" — proves
    # the LLM picked up the new required field from the schema.
    first_spawn_args = spawn_calls[0]["arguments"]
    assert '"name"' in first_spawn_args and "auth" in first_spawn_args, (
        f"Spawn call arguments missing name='auth'; got {first_spawn_args!r}"
    )


def test_send_to_named_sub_agent_continuation_e2e(
    http_client: httpx.Client,
    named_sub_agent_test_agent: str,
) -> None:
    """
    Two-turn flow: turn 1 spawns ``researcher:focus``; turn 2
    uses ``send_to_sub_agent`` to continue the same conversation.
    The sub-agent's child conversation accumulates items from
    BOTH turns — proves the continuation reuses the existing
    conversation rather than creating a new one.
    """
    # Turn 1: spawn.
    r1 = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Spawn the researcher sub-agent named 'focus' with "
            "input 'Initial research on quantum computing'. "
            "Quote its literal marker."
        ),
    )
    assert r1["status"] == "completed"
    assert "RESEARCHER_PHASE4_OK" in _final_text(r1), (
        f"Turn 1 final missing marker: {_final_text(r1)!r}"
    )

    # Turn 2: continue via send_to_sub_agent.
    r2 = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Now use send_to_sub_agent on the SAME researcher "
            "named 'focus' with input 'Follow-up: applications "
            "in cryptography'. Quote its literal marker."
        ),
        previous_response_id=r1["id"],
    )
    assert r2["status"] == "completed"
    assert "RESEARCHER_PHASE4_OK" in _final_text(r2)

    # Inspect the parent conversation's function_calls — turn 2
    # must have called send_to_sub_agent (NOT spawn_sub_agent
    # again, which would create a new child and lose context).
    items = _conversation_items(http_client, r1["conversation"]["id"])
    fc_names = _function_call_names(items)
    assert "send_to_sub_agent" in fc_names, (
        f"Expected send_to_sub_agent in the call list — turn 2 "
        f"may have re-spawned instead. Got: {fc_names}"
    )


def test_ambient_hint_steers_followup_to_send_e2e(
    http_client: httpx.Client,
    named_sub_agent_test_agent: str,
) -> None:
    """
    The ambient hint must let the LLM REMEMBER previously-spawned
    sub-agents across turns. Turn 1 spawns ``researcher:topic``.
    Turn 2's user prompt is deliberately neutral — it doesn't
    name the sub-agent or use the words "researcher" or
    "topic" — so the only way the LLM can correctly continue
    is by reading the ambient hint and choosing
    ``send_to_sub_agent``.

    If the ambient hint isn't injected, the LLM will either
    re-spawn (create a duplicate) or fail to invoke any
    sub-agent tool. Either failure mode breaks named
    persistence.
    """
    # Turn 1: spawn with explicit name + topic.
    r1 = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Spawn the researcher sub-agent named 'topic' with "
            "input 'Investigate the foundations'. Quote the "
            "literal marker it returns."
        ),
    )
    assert r1["status"] == "completed"
    items_t1 = _conversation_items(http_client, r1["conversation"]["id"])
    assert "spawn_sub_agent" in _function_call_names(items_t1), (
        "Turn 1 didn't spawn — test premise broken."
    )

    # Turn 2: NEUTRAL prompt that doesn't say "researcher",
    # "topic", or any other identifier. The LLM has to pick up
    # the existing sub-agent from the ambient hint.
    r2 = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "I want to keep working on what we started. Continue "
            "the prior investigation. Quote whatever marker comes "
            "back."
        ),
        previous_response_id=r1["id"],
    )
    assert r2["status"] == "completed"

    # The critical assertion — turn 2 used send_to_sub_agent
    # (NOT a fresh spawn). If the ambient hint is broken the
    # LLM has no way to know the existing sub-agent's name.
    items_t2 = _conversation_items(http_client, r1["conversation"]["id"])
    fc_names = _function_call_names(items_t2)
    # Find function calls that came AFTER the turn-1 ones —
    # turn 1 had a spawn, turn 2's calls are the new ones.
    new_calls = fc_names[len(_function_call_names(items_t1)) :]
    assert "send_to_sub_agent" in new_calls, (
        f"Turn 2's neutral prompt didn't trigger send_to_sub_agent "
        f"— ambient hint failed to surface the existing sub-agent. "
        f"New calls: {new_calls}. Without the hint working, named "
        f"persistence is useless because the LLM forgets across "
        f"turns."
    )


def test_parallel_named_sub_agents_e2e(
    http_client: httpx.Client,
    named_sub_agent_test_agent: str,
) -> None:
    """
    Real LLM dispatches researcher and summarizer in parallel
    with distinct names. Both markers reach the final reply,
    proving each child ran independently.
    """
    body = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Spawn TWO sub-agents in parallel — emit both "
            "spawn_sub_agent tool calls in the same response: "
            "researcher named 'r1' with input 'topic A', and "
            "summarizer named 's1' with input 'topic B'. Once "
            "both finish, quote both literal markers in your "
            "final reply."
        ),
    )
    assert body["status"] == "completed"
    final = _final_text(body)
    assert "RESEARCHER_PHASE4_OK" in final, f"Researcher marker missing from final: {final!r}"
    assert "SUMMARIZER_PHASE4_OK" in final, f"Summarizer marker missing from final: {final!r}"


def test_cross_parent_named_isolation_e2e(
    http_client: httpx.Client,
    named_sub_agent_test_agent: str,
) -> None:
    """
    Same name (``researcher:auth``) in two independent top-level
    conversations: both spawns succeed, neither sees the other's
    history. The partial unique index is per-parent.
    """
    # Conversation A.
    r_a = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Spawn the researcher sub-agent named 'auth' with "
            "input 'Authentication strategies for project A'. "
            "Quote its marker."
        ),
    )
    assert r_a["status"] == "completed"
    assert "RESEARCHER_PHASE4_OK" in _final_text(r_a)

    # Conversation B (NEW, no previous_response_id).
    r_b = _create_response_blocking(
        http_client,
        model=named_sub_agent_test_agent,
        user_text=(
            "Spawn the researcher sub-agent named 'auth' with "
            "input 'Authentication strategies for project B'. "
            "Quote its marker."
        ),
    )
    assert r_b["status"] == "completed", (
        f"Conversation B's spawn must succeed even though "
        f"conversation A already has a researcher:auth — the "
        f"unique index is per-parent. Got: {r_b!r}"
    )
    assert "RESEARCHER_PHASE4_OK" in _final_text(r_b)

    # The two parent conversations must be distinct.
    conv_a = r_a["conversation"]["id"]
    conv_b = r_b["conversation"]["id"]
    assert conv_a != conv_b, "Conversations A and B should be distinct"
