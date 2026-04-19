"""End-to-end tests for ``@tool(synchronous=False)`` against a real LLM.

Verifies the full Phase 2 pipeline against the live ``ap server`` +
real OpenAI calls:

* Async tool dispatch returns a JSON handle to the LLM (not the
  inline result).
* The ``background_tool_workflow`` runs the function in a
  subprocess, signals ``async_work_complete``.
* The parent's drain auto-delivers the result as a system message.
* The LLM sees the system message in its prompt on the next
  iteration and references the literal marker.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_async_tools_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v

**TUI verification** (mandatory per CLAUDE.md before merge):
``python examples/frontends/terminal.py
tests/_fixtures/agents/async-tools-test/`` then ask "run delayed_echo
with label='alpha'". The auto-delivered result must render as a
dim ``⤵ [System: task ...]`` line — proves the frontend wiring
landed in part 11.
"""

from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path

import httpx
import pytest

_ASYNC_TOOLS_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "async-tools-test"
)


@pytest.fixture(scope="session")
def async_tools_agent(http_client: httpx.Client) -> str:
    """
    Upload the async-tools-test fixture agent.

    :param http_client: HTTP client pointed at the live server.
    :returns: The agent's name (matches its config.yaml ``name``).
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(_ASYNC_TOOLS_FIXTURE_DIR), arcname=".")
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
            # Already registered from a prior test in the same session.
            return _ASYNC_TOOLS_FIXTURE_DIR.name
        resp.raise_for_status()
        return resp.json()["name"]
    finally:
        Path(bundle_path).unlink(missing_ok=True)


def _create_response_blocking(
    http_client: httpx.Client,
    *,
    model: str,
    user_text: str,
    timeout_s: float = 180.0,
) -> dict:
    """
    POST a response, poll until terminal, return the final body.

    :param http_client: HTTP client pointed at the live server.
    :param model: Agent name to invoke.
    :param user_text: Plain-text input message for the agent.
    :param timeout_s: Max seconds to wait for the response to
        complete. Default 180 s — async tools sleep 2 s and the
        LLM may take a couple of turns to converge.
    :returns: The final response JSON.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": user_text,
            "background": True,
            "store": True,
        },
    )
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
    """
    Extract the assistant's final text from a response.

    :param response_body: The response JSON returned from
        ``GET /v1/responses/{id}``.
    :returns: Concatenated assistant text. Empty string if no
        assistant message exists.
    """
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
    """
    Fetch the full ordered list of conversation items.

    :param http_client: HTTP client pointed at the live server.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc..."``.
    :returns: Conversation items in store order.
    """
    resp = http_client.get(
        f"/v1/conversations/{conversation_id}/items",
        params={"limit": 100},
    )
    resp.raise_for_status()
    data: list[dict] = resp.json()["data"]
    return data


# ─── Tests ───────────────────────────────────────────────────


def test_async_tool_real_llm_e2e(
    http_client: httpx.Client,
    async_tools_agent: str,
) -> None:
    """
    Real LLM dispatches an async tool, sees the auto-delivered
    result, and surfaces the literal marker in its final answer.

    What this catches end-to-end:
    * Schema derivation handed the LLM a usable tool spec.
    * Dispatch produced a handle (no inline result).
    * Background workflow ran in a subprocess.
    * Drain delivered the system message.
    * The LLM read the system message and quoted the marker.
    """
    body = _create_response_blocking(
        http_client,
        model=async_tools_agent,
        user_text=(
            "Call delayed_echo with label='alpha'. After it completes, "
            "tell me the literal string the tool returned."
        ),
    )
    assert body["status"] == "completed", (
        f"async-tools turn did not complete: status={body.get('status')!r}, "
        f"error={body.get('error')!r}"
    )
    final = _final_text(body)
    # The marker is distinctive enough that the LLM can't have
    # invented it. If absent, either the auto-delivered system
    # message was missing or the LLM ignored it.
    assert "ECHO_FROM_ASYNC[alpha]" in final, (
        f"Expected the tool's literal marker 'ECHO_FROM_ASYNC[alpha]' "
        f"in the final response. Got: {final!r}"
    )

    # Cross-check the conversation store: the auto-delivered
    # [System: task ... completed] message must be persisted.
    conv_id = body["conversation"]["id"]
    items = _conversation_items(http_client, conv_id)
    user_texts = [
        item["content"][0]["text"]
        for item in items
        if item.get("type") == "message"
        and item.get("role") == "user"
        and item.get("content")
        and item["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    assert len(completion_messages) == 1, (
        f"Expected exactly one auto-delivered completion message; "
        f"got {len(completion_messages)}. user_texts={user_texts}"
    )
    assert "ECHO_FROM_ASYNC[alpha]" in completion_messages[0], (
        f"The auto-delivered system message must carry the tool's "
        f"actual return value. Got: {completion_messages[0]!r}"
    )


def test_mixed_sync_and_async_tools_e2e(
    http_client: httpx.Client,
    async_tools_agent: str,
) -> None:
    """
    The same turn dispatches both an async tool and a sync tool.

    Proves the runtime handles mixed-kind tool batches in
    ``_execute_tools``: the async dispatch returns immediately
    with a handle while the sync tool runs to completion inline,
    then the async result auto-delivers and the LLM references
    both.
    """
    body = _create_response_blocking(
        http_client,
        model=async_tools_agent,
        user_text=(
            "Run TWO tools in this turn: count_chars on the text "
            "'hello' (which is 5 characters), AND delayed_echo with "
            "label='beta'. After both finish, tell me both results "
            "verbatim — the count_chars number and the delayed_echo "
            "literal string."
        ),
    )
    assert body["status"] == "completed", (
        f"mixed-tools turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    # Sync tool result — straightforward integer assert.
    assert "5" in final, (
        f"Expected the count_chars result '5' in the final response. "
        f"Got: {final!r}"
    )
    # Async tool result — distinctive marker.
    assert "ECHO_FROM_ASYNC[beta]" in final, (
        f"Expected the delayed_echo marker 'ECHO_FROM_ASYNC[beta]' "
        f"in the final response. Got: {final!r}"
    )


def test_async_tool_failure_surfaces_e2e(
    http_client: httpx.Client,
    async_tools_agent: str,
) -> None:
    """
    Real LLM invokes the failing async tool, sees the failure
    system message, and acknowledges the error in its response.

    Without G86 the parent's drain would never wake — this test
    would time out at the polling loop instead of asserting on
    the LLM's text.
    """
    body = _create_response_blocking(
        http_client,
        model=async_tools_agent,
        user_text=(
            "Call boom_async. Then tell me what happened — include "
            "the literal error marker string from the system message "
            "in your reply so I can verify it."
        ),
    )
    # The agent's response itself must complete (only the tool
    # task fails). If status="failed" here, the failure was
    # incorrectly propagated as an agent-level error.
    assert body["status"] == "completed", (
        f"async failure must not fail the agent turn: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    # The exception message marker proves the failure traceback
    # survived format_failure_payload + truncate_traceback +
    # drain → system message → next LLM prompt.
    assert "ASYNC_TOOL_BOOM_MARKER" in final, (
        f"Expected the failure marker 'ASYNC_TOOL_BOOM_MARKER' in "
        f"the final response — failure path likely dropped the "
        f"exception detail somewhere between the tool body and "
        f"the LLM's view. Got: {final!r}"
    )
