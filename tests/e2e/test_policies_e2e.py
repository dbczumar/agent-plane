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

import json
import time
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
_ASK_DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "agents" / "ask-demo"
_E2E_PROMPT_POLICY_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "e2e-prompt-policy"
)


@pytest.fixture(scope="session")
def policy_gate_agent(http_client: httpx.Client) -> str:
    """Upload the e2e-policy-gate fixture and return its name."""
    return _upload_agent(http_client, _E2E_POLICY_GATE_DIR)


@pytest.fixture(scope="session")
def label_gate_agent(http_client: httpx.Client) -> str:
    """Upload the e2e-label-gate fixture and return its name."""
    return _upload_agent(http_client, _E2E_LABEL_GATE_DIR)


@pytest.fixture(scope="session")
def ask_demo_agent(http_client: httpx.Client) -> str:
    """Upload the ``ask-demo`` example agent — always-ASK on INPUT."""
    return _upload_agent(http_client, _ASK_DEMO_DIR)


@pytest.fixture(scope="session")
def prompt_policy_agent(http_client: httpx.Client) -> str:
    """Upload the e2e-prompt-policy fixture and return its name."""
    return _upload_agent(http_client, _E2E_PROMPT_POLICY_DIR)


def _find_pending_approval(body: dict) -> dict | None:
    """
    Locate the synthetic ``request_approval`` function_call
    in a polled response body.

    Returns the item dict (carries ``call_id`` and
    ``arguments``) or ``None`` when no approval is pending.

    :param body: The response body from GET /v1/responses/{id}.
    :returns: The item dict, or ``None``.
    """
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if item.get("name") != "request_approval":
            continue
        if item.get("status") != "action_required":
            continue
        return item
    return None


def _wait_for_pending_approval(
    client: httpx.Client,
    response_id: str,
    timeout: float = 30.0,
) -> dict:
    """
    Poll until a ``request_approval`` appears in the output.

    :param client: HTTP client.
    :param response_id: In-progress response id.
    :param timeout: Max seconds to wait.
    :returns: The approval function_call item.
    :raises AssertionError: If no approval appears in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/responses/{response_id}")
        resp.raise_for_status()
        body = resp.json()
        item = _find_pending_approval(body)
        if item is not None:
            return item
        if body.get("status") in ("completed", "failed"):
            raise AssertionError(
                f"Response finished with status {body['status']} but no "
                "approval was ever requested.",
            )
        time.sleep(0.5)
    raise AssertionError(f"No approval surfaced within {timeout}s.")


def _patch_approval_verdict(
    client: httpx.Client,
    response_id: str,
    call_id: str,
    approved: bool,
) -> None:
    """
    PATCH the approval verdict through the existing
    ``tool_results`` contract.

    The server's ``_parse_verdict`` does a strict ``is True``
    on ``parsed["approved"]`` — anything else (missing field,
    wrong type, explicit false) counts as refuse.

    :param client: HTTP client.
    :param response_id: The parked response id.
    :param call_id: The synthetic call_id from the approval.
    :param approved: ``True`` to approve, ``False`` to
        refuse.
    """
    resp = client.patch(
        f"/v1/responses/{response_id}",
        json={
            "tool_results": [
                {
                    "call_id": call_id,
                    "output": json.dumps({"approved": approved}),
                },
            ],
        },
    )
    resp.raise_for_status()


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


# ── Polling-API ASK coverage ──────────────────────────────
#
# The REPL tests in ``test_repl_approval_e2e.py`` exercise
# the streaming-SSE path. These tests cover the polling
# path that headless / scripted clients use:
# POST background=true → poll for pending approval →
# PATCH a verdict → poll to terminal. Same server-side
# enforcement, different client-side protocol.


def test_polling_api_explicit_approval_allows_llm(
    http_client: httpx.Client,
    ask_demo_agent: str,
) -> None:
    """
    Polling client approves an ASK → server unparks → LLM
    runs → response terminal status = completed with real
    text.

    Proves scripted / headless clients (CI runners,
    dashboards, background automation) can participate in
    the approval flow via the existing PATCH contract —
    no streaming SDK required.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": ask_demo_agent,
            "input": "hello polling",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]

    approval = _wait_for_pending_approval(http_client, rid, timeout=60)
    call_id = approval["call_id"]
    # Sanity: the arguments dict carries the policy identity
    # so clients can render a real approval UI.
    args = json.loads(approval["arguments"])
    assert args["policy_name"] == "always_ask_on_input"
    assert args["phase"] == "input"

    _patch_approval_verdict(http_client, rid, call_id, approved=True)

    body = poll_until_terminal(http_client, rid, timeout=60)
    assert body["status"] == "completed", f"Response did not complete: {body.get('error')}"
    text = _extract_all_assistant_text(body)
    assert len(text.strip()) > 0, "Approve path produced no assistant text"
    # No DENY sentinel — approve must NOT substitute the
    # blocked text.
    assert "[Denied by policy" not in text, f"Approve leaked a DENY sentinel: {text!r}"


def test_polling_api_explicit_refusal_denies(
    http_client: httpx.Client,
    ask_demo_agent: str,
) -> None:
    """
    Polling client refuses an ASK → server returns DENY
    sentinel as the assistant reply.

    Same contract as the REPL refuse path, different
    transport. The terminal response carries ``status:
    completed`` (the agent turn finished cleanly — the
    policy just replaced the reply).
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": ask_demo_agent,
            "input": "hello refusal",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]

    approval = _wait_for_pending_approval(http_client, rid, timeout=60)
    _patch_approval_verdict(
        http_client,
        rid,
        approval["call_id"],
        approved=False,
    )

    body = poll_until_terminal(http_client, rid, timeout=60)
    assert body["status"] == "completed"
    text = _extract_all_assistant_text(body)
    assert "[Denied by policy" in text, (
        f"Refuse path did not produce a DENY sentinel.\nGot: {text!r}"
    )


def test_polling_api_malformed_verdict_treated_as_refuse(
    http_client: httpx.Client,
    ask_demo_agent: str,
) -> None:
    """
    A client that PATCHes a malformed verdict (missing
    ``approved`` key, wrong type, non-JSON output) → server's
    ``_parse_verdict`` strict-checks → refuse fail-closed.
    POLICIES.md §13 invariant: only exact
    ``{"approved": true}`` approves; everything else denies.

    This is the critical safety rail. A buggy client that
    accidentally sends garbage must NOT accidentally approve.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": ask_demo_agent,
            "input": "malformed verdict test",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]

    approval = _wait_for_pending_approval(http_client, rid, timeout=60)
    # Garbage verdict — not JSON, not "approved: true".
    patch_resp = http_client.patch(
        f"/v1/responses/{rid}",
        json={
            "tool_results": [
                {
                    "call_id": approval["call_id"],
                    "output": "not even json, definitely not approved",
                },
            ],
        },
    )
    patch_resp.raise_for_status()

    body = poll_until_terminal(http_client, rid, timeout=60)
    text = _extract_all_assistant_text(body)
    assert "[Denied by policy" in text, (
        "Malformed verdict did not fail-closed refuse — major safety "
        "regression. The server must treat anything other than exact "
        f'{{"approved": true}} as a refusal.\nGot: {text!r}'
    )


# ── PromptPolicy (Phase 9): real LLM classifier end-to-end ─
#
# These tests exercise the production path of
# :func:`make_default_classifier` — the real LLM gets called
# with the framework-generated envelope + author prompt, and
# the parsed JSON verdict drives the ALLOW / DENY branch.


def test_prompt_policy_allow_path_reaches_llm(
    http_client: httpx.Client,
    prompt_policy_agent: str,
) -> None:
    """
    Non-Canadian input → classifier ALLOWs → agent LLM runs →
    assistant text comes back. Proves the real classifier
    works end-to-end through the real LLM, the policy engine
    composes ALLOW, and the full turn completes normally.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": prompt_policy_agent,
            "input": "What's 2+2? Answer with the number only.",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    body = poll_until_terminal(http_client, rid, timeout=120)
    assert body["status"] == "completed", f"Unexpected status: {body.get('error')}"
    text = _extract_all_assistant_text(body)
    # Real LLM answered the question — "4" must appear.
    # Stronger than a non-empty check: proves the request
    # actually reached the LLM and the LLM's output
    # propagated through the ALLOW path.
    assert "4" in text, f"Expected the LLM's answer to 2+2 ('4') in the reply.\nGot: {text!r}"
    # Policy did NOT deny — the DENY sentinel must not appear.
    assert "[Denied by policy" not in text, (
        f"ALLOW path accidentally emitted a DENY sentinel: {text!r}"
    )


def test_prompt_policy_deny_path_short_circuits(
    http_client: httpx.Client,
    prompt_policy_agent: str,
) -> None:
    """
    Canadian-topic input → classifier DENYs → sentinel replaces
    the assistant reply → LLM never produces its normal
    output.

    This is the canonical reason PromptPolicy exists: a
    topic-level content filter an author describes in prose
    rather than a Python predicate. If the real classifier
    isn't wired, the policy falls back to
    :class:`NotImplementedError` and the turn would fail —
    so this test is simultaneously a Phase 9 wiring proof
    AND a regression guard against someone accidentally
    reverting to the raising stub.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": prompt_policy_agent,
            "input": "What's the capital of Canada?",
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    body = poll_until_terminal(http_client, rid, timeout=120)
    assert body["status"] == "completed", f"Unexpected status: {body.get('error')}"
    text = _extract_all_assistant_text(body)
    assert "[Denied by policy" in text, (
        f"PromptPolicy DENY did not short-circuit the turn.\nGot: {text!r}"
    )
    # The reason carried in the sentinel should mention
    # Canada — the author's prompt instructs the classifier
    # to emit exactly ``"mentions Canada"`` as the reason,
    # and the server interpolates it into ``[Denied by
    # policy: <reason>]``. Casefold-compare so model
    # capitalization variance doesn't break the test.
    assert "canada" in text.lower(), (
        f"DENY sentinel didn't carry the expected reason ('Canada').\nGot: {text!r}"
    )
