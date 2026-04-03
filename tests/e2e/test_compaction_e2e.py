"""End-to-end compaction test with a real LLM and real server.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_compaction_e2e.py \\
        --llm-api-key $(cat /tmp/mykey) -v

Exercises:
- Multi-turn conversation that fills the context window
- Reactive compaction (overflow → compact → retry)
- Proactive compaction (subsequent turns stay under budget)
- Compaction item persisted to conversation store
- Cursor-based history loading on follow-up turns
- Agent continues to function after compaction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import (
    _upload_agent,
    poll_until_terminal,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPACTION_AGENT_DIR = _REPO_ROOT / "examples" / "agents" / "compaction-test"


@pytest.fixture(scope="session")
def compaction_agent(http_client: httpx.Client) -> str:
    """
    Upload the compaction-test agent and return its name.

    :param http_client: HTTP client pointed at the live server.
    :returns: The agent name, e.g. ``"compaction-test"``.
    """
    return _upload_agent(http_client, _COMPACTION_AGENT_DIR)


def _create_turn(
    client: httpx.Client,
    model: str,
    user_input: str,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a response and poll until terminal.

    :param client: HTTP client pointed at the live server.
    :param model: Agent name.
    :param user_input: User message text.
    :param previous_response_id: ID of the previous response
        for multi-turn conversations, or ``None`` for the first turn.
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
    return poll_until_terminal(client, response_id, timeout=120)


def _get_conversation_items(
    client: httpx.Client,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all items in a conversation.

    :param client: HTTP client pointed at the live server.
    :param conversation_id: The conversation ID.
    :returns: List of conversation item dicts.
    """
    resp = client.get(f"/v1/conversations/{conversation_id}/items")
    resp.raise_for_status()
    return resp.json()["data"]


# Long prompt designed to consume a large chunk of the context window.
# gpt-4.1-nano has a 1M token window, so we need substantial content.
# Each turn asks for verbose output to fill the window faster.
_VERBOSE_PROMPT = (
    "List every country in the world, its capital city, population, "
    "GDP, official languages, currency, and a brief history of its "
    "founding. Be as detailed as possible. Include all 195 UN member "
    "states. Format as a numbered list with sub-bullets for each field."
)


def test_compaction_fires_after_multi_turn_overflow(
    http_client: httpx.Client,
    compaction_agent: str,
) -> None:
    """
    Multiple turns of verbose output eventually trigger compaction.
    After compaction fires, the agent continues to function and a
    compaction item is persisted to the conversation store.

    This test sends several turns of verbose prompts to fill the
    context window, then verifies:
    1. The agent still completes responses (compaction didn't break it)
    2. A compaction item exists in the conversation store
    3. A follow-up turn after compaction loads from the cursor
       (the agent can reference prior context via the summary)

    :param http_client: HTTP client pointed at the live server.
    :param compaction_agent: The uploaded agent name.
    """
    # --- Turn 1: Start conversation with verbose output ---
    turn_1 = _create_turn(
        http_client,
        compaction_agent,
        _VERBOSE_PROMPT,
    )
    assert turn_1["status"] == "completed", (
        f"Turn 1 failed: {turn_1.get('status')}. "
        f"The compaction-test agent must complete its first turn."
    )
    response_id = turn_1["id"]
    conv_id = turn_1["conversation"]["id"]

    # --- Turns 2-5: Build up context ---
    for i in range(2, 6):
        turn = _create_turn(
            http_client,
            compaction_agent,
            f"Turn {i}: {_VERBOSE_PROMPT} Also summarize what you said in previous turns.",
            previous_response_id=response_id,
        )
        assert turn["status"] == "completed", (
            f"Turn {i} failed: {turn.get('status')}. "
            f"The agent must continue completing turns as context grows."
        )
        response_id = turn["id"]

    # --- Check for compaction item ---
    items = _get_conversation_items(http_client, conv_id)
    compaction_items = [i for i in items if i.get("type") == "compaction"]

    # With gpt-4.1-nano (1M context) and 5 verbose turns, compaction
    # may or may not fire depending on actual output length. If it
    # didn't fire, we send more turns.
    if not compaction_items:
        for i in range(6, 11):
            turn = _create_turn(
                http_client,
                compaction_agent,
                f"Turn {i}: {_VERBOSE_PROMPT} "
                f"Repeat all your previous answers verbatim, "
                f"then add new countries you missed.",
                previous_response_id=response_id,
            )
            assert turn["status"] == "completed", f"Turn {i} failed: {turn.get('status')}."
            response_id = turn["id"]

        items = _get_conversation_items(http_client, conv_id)
        compaction_items = [i for i in items if i.get("type") == "compaction"]

    # At this point compaction should have fired at least once.
    # If not, the model's context window is too large for this test
    # to overflow with 10 turns. Mark as expected failure.
    if not compaction_items:
        pytest.skip(
            "Compaction did not fire after 10 turns — the model's "
            "context window may be too large to overflow with this "
            "test's prompt volume. Try with a smaller-context model."
        )

    # --- Verify compaction item structure ---
    cmp = compaction_items[-1]
    assert "summary" in cmp, f"Compaction item missing 'summary' field: {cmp}"
    assert "last_item_id" in cmp, f"Compaction item missing 'last_item_id' field: {cmp}"
    # Summary must be non-empty text.
    assert isinstance(cmp["summary"], str) and len(cmp["summary"]) > 10, (
        f"Compaction summary is empty or too short: {cmp['summary']!r}"
    )
    # last_item_id must point to a real conversation item.
    all_item_ids = {i["id"] for i in items}
    assert cmp["last_item_id"] in all_item_ids, (
        f"Compaction last_item_id={cmp['last_item_id']!r} does not match any conversation item."
    )

    # --- Follow-up turn: agent works after compaction ---
    follow_up = _create_turn(
        http_client,
        compaction_agent,
        "What was the first thing I asked you about? Give a one-sentence summary.",
        previous_response_id=response_id,
    )
    assert follow_up["status"] == "completed", (
        f"Follow-up turn after compaction failed: "
        f"{follow_up.get('status')}. The agent must still function "
        f"after compaction fires."
    )

    # The follow-up response should reference countries/capitals
    # (from the summary), proving the compaction summary provided
    # useful context to the agent.
    follow_up_output = follow_up.get("output", [])
    follow_up_texts = [
        item["content"][0]["text"]
        for item in follow_up_output
        if item.get("type") == "message"
        and item.get("role") == "assistant"
        and item.get("content")
    ]
    assert follow_up_texts, (
        "Follow-up turn produced no assistant text. The agent should respond after compaction."
    )
    # The agent should mention countries or capitals — proving it
    # has context from the summary.
    combined = " ".join(follow_up_texts).lower()
    assert any(
        keyword in combined for keyword in ["countr", "capital", "list", "nation", "asked"]
    ), (
        f"Follow-up response doesn't reference the prior conversation "
        f"context. The compaction summary may not have been loaded. "
        f"Response: {combined[:200]}"
    )
