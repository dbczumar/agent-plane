"""
E2E tests for the policy system through the real workflow.

Uploads the ``e2e-policy-gate`` fixture agent (FunctionPolicy
at INPUT that DENYs messages containing a sentinel token),
posts responses with real LLM calls through the server, and
verifies:

- Clean messages pass through → real LLM response.
- Sentinel-containing messages hit the policy DENY path →
  assistant sentinel text, no LLM call.
- The DENY sentinel is persisted to conversation_items so a
  follow-up turn sees it.
- The DENY path terminates the turn in ``completed`` status
  (the agent didn't crash, it just replied with the
  sentinel).
- Agents without any guardrails block run unchanged (the
  archer agent is the regression test for this — if the
  no-op engine path broke, every non-policy agent would
  too).

Usage::

    pytest tests/e2e/test_policies_e2e.py \\
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import _upload_agent, poll_until_terminal

_E2E_POLICY_GATE_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "e2e-policy-gate"
)
_E2E_LABEL_GATE_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "e2e-label-gate"
)


@pytest.fixture(scope="session")
def policy_gate_agent(http_client: httpx.Client) -> str:
    """Upload the e2e-policy-gate fixture and return its name."""
    return _upload_agent(http_client, _E2E_POLICY_GATE_DIR)


@pytest.fixture(scope="session")
def label_gate_agent(http_client: httpx.Client) -> str:
    """Upload the e2e-label-gate fixture and return its name."""
    return _upload_agent(http_client, _E2E_LABEL_GATE_DIR)


def _extract_all_assistant_text(body: dict) -> str:
    """Concatenate assistant-message text from a response body."""
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            if isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


# ── Clean-path: no policy trigger ─────────────────────


def test_policy_gate_allows_clean_message(
    http_client: httpx.Client,
    policy_gate_agent: str,
) -> None:
    """A normal message (no sentinel) passes through the
    policy → reaches the LLM → gets a real response. If
    this regresses, the policy is over-firing and blocking
    legitimate traffic."""
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": policy_gate_agent,
            "input": "Say hi in exactly three words.",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    body = poll_until_terminal(http_client, rid, timeout=120)
    # Terminal status must be completed — policy ALLOW should
    # not turn the turn into a failure.
    assert body["status"] == "completed", f"Unexpected status: {body.get('error')}"
    text = _extract_all_assistant_text(body)
    # Real LLM response — just verify something came back
    # (content varies; checking for non-empty is the right
    # granularity since we're testing policy pass-through,
    # not LLM output quality).
    assert len(text.strip()) > 0, (
        "Expected real LLM output after policy ALLOW; got empty response."
    )
    # Sentinel must NOT appear — the clean path doesn't
    # invoke the DENY branch.
    assert "[Denied by policy" not in text


# ── DENY path: sentinel-containing message ────────────


def test_policy_gate_denies_sentinel_message(
    http_client: httpx.Client,
    policy_gate_agent: str,
) -> None:
    """A message containing the sentinel token hits the
    FunctionPolicy DENY → sentinel text persisted as the
    assistant reply, no LLM call. If this regresses, the
    policy system is not wired into the workflow and
    policies are effectively no-ops in production."""
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": policy_gate_agent,
            "input": "Please process this: BLOCK_THIS_TOKEN now.",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    body = poll_until_terminal(http_client, rid, timeout=60)
    # Turn completes (not failed) — policy DENY is a
    # well-defined terminal state, not a runtime error.
    assert body["status"] == "completed", (
        f"DENY path must complete cleanly; got {body['status']}: {body.get('error')}"
    )
    text = _extract_all_assistant_text(body)
    # The sentinel-shaped text appears in the assistant
    # output, proving the DENY path fired.
    assert "[Denied by policy" in text, f"Expected DENY sentinel in output; got: {text[:400]!r}"
    # The policy's reason is included in the sentinel — drives
    # the UI's "why was this blocked?" surface.
    assert "BLOCK_THIS_TOKEN" in text, (
        f"Expected reason mentioning BLOCK_THIS_TOKEN; got: {text[:400]!r}"
    )


# ── DENY persisted for follow-up turns ────────────────


def test_policy_gate_deny_persists_to_history(
    http_client: httpx.Client,
    policy_gate_agent: str,
) -> None:
    """After a DENY, a follow-up turn on the same
    conversation sees the sentinel in history. Proves the
    sentinel was written to conversation_items (not just
    surfaced on the stream)."""
    # Turn 1: DENY.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": policy_gate_agent,
            "input": "Trigger BLOCK_THIS_TOKEN please.",
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    body1 = poll_until_terminal(http_client, rid1, timeout=60)
    assert body1["status"] == "completed"
    assert "[Denied by policy" in _extract_all_assistant_text(body1)

    # Turn 2: clean follow-up on the same conversation.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": policy_gate_agent,
            "input": "Reply with a single word: OK",
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    body2 = poll_until_terminal(http_client, rid2, timeout=120)
    # Turn 2 completed without crashing — the engine rebuilt
    # cleanly on a conversation that already had a DENYed
    # turn-1 sentinel in history.
    assert body2["status"] == "completed", f"Turn 2 failed: {body2.get('error')}"
    # The LLM ran on turn 2 (no sentinel in its input) and
    # produced a non-empty response. We do NOT assert that
    # the LLM didn't echo the sentinel from history — the
    # LLM sees the prior turn's assistant message (the
    # sentinel text) and may repeat part of it when asked
    # a follow-up, which is LLM behavior, not a policy bug.
    text2 = _extract_all_assistant_text(body2)
    assert len(text2.strip()) > 0
    # Fetch conversation items — the turn-1 sentinel MUST
    # be persisted so replay sees it.
    conv_id = body2["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items_resp.raise_for_status()
    items = items_resp.json().get("data", [])
    assistant_texts = [
        block.get("text") or block.get("output_text") or ""
        for item in items
        if item.get("type") == "message" and item.get("role") == "assistant"
        for block in item.get("content", [])
        if isinstance(block, dict)
    ]
    # Turn-1 sentinel is in the persisted history.
    assert any("[Denied by policy" in t for t in assistant_texts), (
        f"DENY sentinel not persisted to conversation_items. Assistant texts: {assistant_texts!r}"
    )


# ── Regression: no-guardrails agents still work ──────


# ── Multi-policy composition via labels across turns ─


def test_label_gate_taint_persists_across_turns(
    http_client: httpx.Client,
    label_gate_agent: str,
) -> None:
    """Turn 1: user triggers FunctionPolicy that writes
    ``tainted: "1"``. Turn 2: clean input, but
    LabelPolicy's condition ``tainted: "1"`` now matches →
    DENY.

    End-to-end proof that FunctionPolicy set_labels reach
    the store, persist across workflow restarts, and drive
    condition gates on the next turn — the core IFC-through-
    labels pattern."""
    # Turn 1: trigger the taint.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": label_gate_agent,
            "input": "BANANA_TRIGGER — say hi briefly.",
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    body1 = poll_until_terminal(http_client, rid1, timeout=120)
    # Turn 1 completes with a real LLM reply (taint_on_banana
    # is ALLOW-with-set_labels; deny_when_tainted hasn't
    # fired yet because the condition was evaluated against
    # the pre-turn-1 label snapshot).
    assert body1["status"] == "completed", f"Turn 1 failed: {body1.get('error')}"
    text1 = _extract_all_assistant_text(body1)
    assert "[Denied by policy" not in text1
    assert len(text1.strip()) > 0

    # Turn 2: clean input — no trigger. But the label is
    # already persisted from turn 1.
    conv_id = body1["conversation"]["id"]
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": label_gate_agent,
            "input": "A clean follow-up message.",
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    body2 = poll_until_terminal(http_client, rid2, timeout=60)
    assert body2["status"] == "completed"
    text2 = _extract_all_assistant_text(body2)
    # Turn 2 MUST hit the DENY path because tainted=1
    # survived to this turn.
    assert "[Denied by policy" in text2, (
        f"Turn 2 should DENY on tainted conversation; got: {text2[:400]!r}"
    )
    # Reason matches the LabelPolicy declaration.
    assert "tainted" in text2.lower()


def test_label_gate_untainted_conversation_passes(
    http_client: httpx.Client,
    label_gate_agent: str,
) -> None:
    """A conversation that never triggers taint_on_banana
    should pass every turn — the condition
    ``tainted: "1"`` never matches against the default
    ``tainted: "0"`` seed."""
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": label_gate_agent,
            "input": "Hello. Reply briefly.",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    body = poll_until_terminal(http_client, rid, timeout=120)
    assert body["status"] == "completed", f"Clean conversation failed: {body.get('error')}"
    text = _extract_all_assistant_text(body)
    assert "[Denied by policy" not in text
    assert len(text.strip()) > 0


def test_label_gate_persisted_labels_in_store(
    http_client: httpx.Client,
    label_gate_agent: str,
) -> None:
    """After the taint turn, the ``tainted`` label is
    persisted to ``conversation_labels`` — verifiable via
    a follow-up request that creates a fresh workflow
    (engine rebuilt from persisted state).

    Not just an in-memory snapshot — the labels survive
    workflow restarts, which is what Phase 1's store API
    guarantees."""
    # Turn 1: taint.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": label_gate_agent,
            "input": "BANANA_TRIGGER, please acknowledge.",
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    body1 = poll_until_terminal(http_client, rid1, timeout=120)
    assert body1["status"] == "completed"

    # Turn 2 on same conversation. The workflow rebuilds
    # the engine from persisted state — if the label didn't
    # persist, the condition wouldn't match and turn 2 would
    # pass through. Behavior asserts persistence.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": label_gate_agent,
            "input": "ok.",
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    body2 = poll_until_terminal(http_client, rid2, timeout=60)
    text2 = _extract_all_assistant_text(body2)
    # Persisted → condition matches → DENY.
    assert "[Denied by policy" in text2


def test_no_guardrails_agent_unaffected(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """Archer has no guardrails block — the engine is a
    no-op, every INPUT ALLOWs, workflow runs normally.

    Regression test for the Phase 6 wiring: if
    `build_policy_engine` misbehaves on the no-guardrails
    path, OR `_enforce_input_policies` over-fires, EVERY
    production agent without policies would start failing.
    Detecting this at the e2e level catches bugs the unit
    tests' `noop_engine` doesn't cover (real workflow,
    real message flow, real LLM round-trip)."""
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "What is 2 + 2? Answer with one number only.",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    body = poll_until_terminal(http_client, rid, timeout=120)
    assert body["status"] == "completed", f"Archer (no guardrails) failed: {body.get('error')}"
    text = _extract_all_assistant_text(body)
    # Real LLM output — not a policy sentinel.
    assert len(text.strip()) > 0
    assert "[Denied by policy" not in text
