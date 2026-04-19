"""
Server-integration tests for the Phase 4 named-sub-agent pipeline.

Mirrors ``test_sub_agent_integration.py`` but for the new
``send_to_sub_agent`` continuation idiom and the
``name_already_exists`` / ``sub_agent_not_found`` /
``sub_agent_busy`` error paths. The full chain under test:

1. Parent LLM emits ``spawn_sub_agent(type, name, input)``.
2. Child conversation persists with title ``"<type>:<name>"``
   and ``parent_conversation_id`` pointing at the parent's
   conversation.
3. On a later turn, the parent emits
   ``send_to_sub_agent(type, name, input)``.
4. ``SendToSubAgentTool`` resolves the existing child and
   creates a NEW task on it; the sub-agent's
   ``_load_initial_history`` returns the full prior history.
5. The continuation's LLM sees both the original input AND the
   new input, so it "remembers" earlier turns.

Plus the error-path tests:
* duplicate ``(type, name)`` rejection at the partial unique
  index, surfaced as a ``name_already_exists`` tool result.
* lookup miss in ``send_to_sub_agent`` → ``sub_agent_not_found``.
* in-flight task on the same child → ``sub_agent_busy``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import build_agent_bundle, create_test_response

pytestmark = [pytest.mark.asyncio]


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
    timeout_iters: int = 200,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status.

    :param client: HTTP client.
    :param response_id: The response/task ID to poll.
    :param timeout_iters: Max number of 0.1s polls.
    :returns: The terminal response body.
    """
    for _ in range(timeout_iters):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"Response {response_id} did not reach terminal status",
    )


async def _get_items(
    client: httpx.AsyncClient,
    conv_id: str,
) -> list[dict[str, Any]]:
    """Fetch all conversation items in store order."""
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


async def _create_parent_with_sub_agent(
    client: httpx.AsyncClient,
    *,
    parent_name: str,
    sub_agent_name: str,
) -> None:
    """Upload a parent agent declaring one named sub-agent."""
    bundle = build_agent_bundle(
        name=parent_name,
        sub_agents=[
            {"name": sub_agent_name, "description": f"{sub_agent_name} sub-agent"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent upload failed: {resp.status_code} {resp.text}"


# ─── Tests ───────────────────────────────────────────────────


async def test_spawn_persists_named_child_with_correct_title(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    A named spawn creates a child conversation with
    ``title="<type>:<name>"`` and ``parent_conversation_id``
    pointing at the parent's conversation. Without these the
    Phase 4 lookup machinery (`send_to_sub_agent`,
    `list_sub_agents`, ambient hint) cannot find the child.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-persist-1",
        sub_agent_name="coder",
    )

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_spawn_p4_1",
                "name": "spawn_sub_agent",
                "arguments": json.dumps(
                    {"type": "coder", "name": "auth", "input": "refactor X"},
                ),
            },
        ],
    )
    for _ in range(5):
        mock_llm.add_call(text="OK")

    result = await create_test_response(
        client,
        model="parent-persist-1",
        input_text="Refactor auth via the coder sub-agent named 'auth'",
    )
    response_id = result.body["id"]
    parent_conv_id = result.body["conversation"]["id"]
    await _wait_for_completion(client, response_id)

    # Use the conversation_store directly to verify the row's
    # title + parent — the LLM-facing handle JSON also carries
    # type+name but the persistent state is what matters for
    # later continuation.
    from agent_plane.runtime import get_conversation_store

    conv_store = get_conversation_store()
    children = conv_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=parent_conv_id,
        limit=100,
    )
    assert len(children.data) == 1, (
        f"Expected exactly 1 child conversation; got {len(children.data)}"
    )
    child = children.data[0]
    # If title were just "auth" or just "coder" the
    # send_to_sub_agent lookup (which builds "coder:auth")
    # would never match. If parent_conversation_id were null
    # the partial unique index wouldn't fire and ambient hint
    # wouldn't surface this child.
    assert child.title == "coder:auth", (
        f"Child title must be '<type>:<name>'; got {child.title!r}"
    )
    assert child.parent_conversation_id == parent_conv_id


async def test_send_to_sub_agent_appends_to_existing_conversation(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Turn 1: spawn ``coder:auth``. Turn 2: ``send_to_sub_agent``
    with the same ``(type, name)``. The sub-agent's child
    conversation must accumulate items from BOTH turns — proves
    the continuation reuses the existing conversation rather
    than creating a new one. If a new conversation were created
    the sub-agent would lose all prior context.

    Uses ``tool_calls_fn`` predicates so each call inspects its
    own input and emits the right tool_call (or None to fall
    back to text). The shared FIFO mock queue makes simple
    pre-queued tool_calls non-deterministic when parent and
    sub-agent race; predicates make the routing explicit.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-persist-2",
        sub_agent_name="coder",
    )

    # The parent's first user message contains "spawn-marker".
    # The follow-up turn's first user message contains "send-marker".
    # ``tool_calls_fn`` for each call inspects the input and emits
    # the spawn / send tool_call only when its marker is present
    # AND no function_call_output exists yet (avoids re-emitting
    # the tool_call after the runtime returned the result).
    # Single predicate that emits the right tool_call based on
    # which marker is in the input AND whether that call_id has
    # already been issued. Using one fn for every queued call
    # means routing is independent of which queue position pops.
    def _route(kwargs: dict[str, Any]) -> list[dict[str, str]] | None:
        input_str = json.dumps(kwargs.get("input", []))
        # Send takes priority — if both markers are present
        # (turn 2 input has both because previous_response_id
        # threads the history), the new "send-marker" turn is
        # the one we want to act on.
        if "send-marker" in input_str and "call_send_t2" not in input_str:
            return [
                {
                    "call_id": "call_send_t2",
                    "name": "send_to_sub_agent",
                    "arguments": json.dumps(
                        {"type": "coder", "name": "auth", "input": "now add tests"},
                    ),
                }
            ]
        if "spawn-marker" in input_str and "call_spawn_t1" not in input_str:
            return [
                {
                    "call_id": "call_spawn_t1",
                    "name": "spawn_sub_agent",
                    "arguments": json.dumps(
                        {"type": "coder", "name": "auth", "input": "refactor X"},
                    ),
                }
            ]
        return None

    for _ in range(40):
        mock_llm.add_call(text="ok", tool_calls_fn=_route)

    r1 = await create_test_response(
        client,
        model="parent-persist-2",
        input_text="spawn-marker: refactor auth via the coder",
    )
    parent_conv_id = r1.body["conversation"]["id"]
    await _wait_for_completion(client, r1.body["id"])

    from agent_plane.runtime import get_conversation_store

    conv_store = get_conversation_store()
    children = conv_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=parent_conv_id,
        limit=100,
    )
    assert len(children.data) == 1, (
        f"Turn 1 should have produced exactly 1 child; got {len(children.data)}"
    )
    child_conv_id = children.data[0].id

    r2 = await create_test_response(
        client,
        model="parent-persist-2",
        input_text="send-marker: continue and add tests",
        previous_response_id=r1.body["id"],
    )
    await _wait_for_completion(client, r2.body["id"])

    # After turn 2, the same child conversation must contain
    # BOTH turn-1's input AND turn-2's input — proves the
    # continuation appended to the existing child.
    items_after_t2 = await _get_items(client, child_conv_id)
    user_texts = [
        i["content"][0]["text"]
        for i in items_after_t2
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    assert any("refactor X" in t for t in user_texts), (
        f"Turn 1's input missing from child conversation after turn 2 — "
        f"send_to_sub_agent may have created a new conversation. "
        f"user_texts={user_texts}"
    )
    assert any("now add tests" in t for t in user_texts), (
        f"Turn 2's input missing from child conversation. "
        f"user_texts={user_texts}"
    )


async def test_spawn_duplicate_name_returns_name_already_exists(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    G36: a second ``spawn_sub_agent`` with the same
    ``(type, name)`` under the same parent must surface a
    clean ``name_already_exists`` tool error to the LLM —
    NOT a raw IntegrityError or a silent duplicate row.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-dup-name",
        sub_agent_name="coder",
    )

    # Two parallel spawns with identical (type, name) in one
    # response. The partial unique index lets at most one win;
    # the other must surface the error.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_dup_a",
                "name": "spawn_sub_agent",
                "arguments": json.dumps(
                    {"type": "coder", "name": "auth", "input": "first"},
                ),
            },
            {
                "call_id": "call_dup_b",
                "name": "spawn_sub_agent",
                "arguments": json.dumps(
                    {"type": "coder", "name": "auth", "input": "second"},
                ),
            },
        ],
    )
    for _ in range(5):
        mock_llm.add_call(text="ok")

    result = await create_test_response(
        client,
        model="parent-dup-name",
        input_text="Spawn two coders with the same name (test)",
    )
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]
    await _wait_for_completion(client, response_id)

    items = await _get_items(client, conv_id)
    fco_outputs = [
        json.loads(i["output"])
        for i in items
        if i.get("type") == "function_call_output"
        and i.get("call_id") in {"call_dup_a", "call_dup_b"}
    ]
    # Exactly one success + exactly one name_already_exists.
    # If both succeeded, the partial unique index isn't
    # working. If both failed, _spawn_one is too aggressive.
    successes = [o for o in fco_outputs if "task_id" in o]
    failures = [
        o
        for o in fco_outputs
        if isinstance(o, dict) and o.get("error") == "name_already_exists"
    ]
    assert len(successes) == 1, (
        f"Expected exactly 1 successful spawn; got {len(successes)}. "
        f"Both successful = unique index not enforced."
    )
    assert len(failures) == 1, (
        f"Expected exactly 1 name_already_exists; got {len(failures)}. "
        f"Outputs: {fco_outputs}"
    )
    # The failure payload must include the offending (type, name)
    # so the LLM can recover deterministically.
    assert failures[0]["type"] == "coder"
    assert failures[0]["name"] == "auth"


async def test_send_to_unknown_name_returns_sub_agent_not_found(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    ``send_to_sub_agent`` with a name the parent never spawned
    must return the documented ``sub_agent_not_found`` error
    (NOT silently spawn a new one — Phase 4 strict
    continue-only semantics).
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-unknown-name",
        sub_agent_name="coder",
    )

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_send_unknown",
                "name": "send_to_sub_agent",
                "arguments": json.dumps(
                    {"type": "coder", "name": "ghost", "input": "x"},
                ),
            },
        ],
    )
    for _ in range(3):
        mock_llm.add_call(text="ok")

    result = await create_test_response(
        client,
        model="parent-unknown-name",
        input_text="Send to a name that doesn't exist",
    )
    await _wait_for_completion(client, result.body["id"])

    items = await _get_items(client, result.body["conversation"]["id"])
    fco = next(
        (
            i
            for i in items
            if i.get("type") == "function_call_output"
            and i.get("call_id") == "call_send_unknown"
        ),
        None,
    )
    assert fco is not None, (
        "Expected a function_call_output for the unknown-name send; "
        "if missing, the tool didn't even run."
    )
    payload = json.loads(fco["output"])
    assert payload.get("error") == "sub_agent_not_found", (
        f"Expected error=sub_agent_not_found; got {payload}"
    )
    # The (type, name) round-trip on the error payload lets the
    # LLM craft a recovery (e.g. spawn first, then resend).
    assert payload.get("type") == "coder"
    assert payload.get("name") == "ghost"


async def test_list_sub_agents_returns_named_children(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    ``list_sub_agents`` must surface every named child created
    by spawn under the caller's conversation. Output is the
    documented minimal shape ``{"sub_agents": [{type, name},
    ...]}``.
    """
    bundle = build_agent_bundle(
        name="parent-list-1",
        sub_agents=[
            {"name": "researcher", "description": "researcher"},
            {"name": "summarizer", "description": "summarizer"},
        ],
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201

    # One predicate handles both turns: spawn (when "spawn-marker"
    # in input AND no spawns yet) and list (when "list-marker"
    # in input AND no list call yet).
    def _route(kwargs: dict[str, Any]) -> list[dict[str, str]] | None:
        input_str = json.dumps(kwargs.get("input", []))
        if "list-marker" in input_str and "call_list_q" not in input_str:
            return [
                {
                    "call_id": "call_list_q",
                    "name": "list_sub_agents",
                    "arguments": "{}",
                }
            ]
        if "spawn-marker" in input_str and "call_spawn_r" not in input_str:
            return [
                {
                    "call_id": "call_spawn_r",
                    "name": "spawn_sub_agent",
                    "arguments": json.dumps(
                        {"type": "researcher", "name": "first", "input": "x"},
                    ),
                },
                {
                    "call_id": "call_spawn_s",
                    "name": "spawn_sub_agent",
                    "arguments": json.dumps(
                        {"type": "summarizer", "name": "second", "input": "y"},
                    ),
                },
            ]
        return None

    for _ in range(40):
        mock_llm.add_call(text="ok", tool_calls_fn=_route)

    r1 = await create_test_response(
        client,
        model="parent-list-1",
        input_text="spawn-marker: spawn both",
    )
    await _wait_for_completion(client, r1.body["id"])

    r2 = await create_test_response(
        client,
        model="parent-list-1",
        input_text="list-marker: enumerate sub-agents",
        previous_response_id=r1.body["id"],
    )
    await _wait_for_completion(client, r2.body["id"])

    items = await _get_items(client, r1.body["conversation"]["id"])
    fco = next(
        (
            i
            for i in items
            if i.get("type") == "function_call_output" and i.get("call_id") == "call_list_q"
        ),
        None,
    )
    assert fco is not None, (
        f"list_sub_agents was never invoked; items={[i.get('type') for i in items]}"
    )
    payload = json.loads(fco["output"])
    assert "sub_agents" in payload
    sub_agents = payload["sub_agents"]
    # Both children must be in the list — order isn't asserted
    # because list_conversations sorts by created_at ASC and
    # parallel tool dispatches don't guarantee ordering.
    pairs = sorted((sa["type"], sa["name"]) for sa in sub_agents)
    assert pairs == [("researcher", "first"), ("summarizer", "second")], (
        f"Expected both named sub-agents; got {sub_agents}"
    )


async def test_cross_parent_name_isolation(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Two top-level conversations can both spawn ``coder:auth``
    independently — the partial unique index is per-parent,
    not global.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-iso",
        sub_agent_name="coder",
    )

    # Both parents share a single router predicate: emit spawn
    # for whichever marker is present (A or B). The call_id
    # disambiguates so the same predicate doesn't re-fire
    # spawn after the tool has run in this turn.
    def _route(kwargs: dict[str, Any]) -> list[dict[str, str]] | None:
        input_str = json.dumps(kwargs.get("input", []))
        if "iso-A-marker" in input_str and "call_iso_a" not in input_str:
            return [
                {
                    "call_id": "call_iso_a",
                    "name": "spawn_sub_agent",
                    "arguments": json.dumps(
                        {"type": "coder", "name": "auth", "input": "for project A"},
                    ),
                }
            ]
        if "iso-B-marker" in input_str and "call_iso_b" not in input_str:
            return [
                {
                    "call_id": "call_iso_b",
                    "name": "spawn_sub_agent",
                    "arguments": json.dumps(
                        {"type": "coder", "name": "auth", "input": "for project B"},
                    ),
                }
            ]
        return None

    for _ in range(40):
        mock_llm.add_call(text="ok", tool_calls_fn=_route)

    r_a = await create_test_response(
        client,
        model="parent-iso",
        input_text="iso-A-marker: spawn for A",
    )
    parent_a_conv_id = r_a.body["conversation"]["id"]
    await _wait_for_completion(client, r_a.body["id"])

    r_b = await create_test_response(
        client,
        model="parent-iso",
        input_text="iso-B-marker: spawn for B",
    )
    parent_b_conv_id = r_b.body["conversation"]["id"]
    await _wait_for_completion(client, r_b.body["id"])

    # Both spawns must have created distinct child rows.
    from agent_plane.runtime import get_conversation_store

    conv_store = get_conversation_store()
    children_a = conv_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=parent_a_conv_id,
        limit=100,
    )
    children_b = conv_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=parent_b_conv_id,
        limit=100,
    )
    assert len(children_a.data) == 1
    assert len(children_b.data) == 1
    assert children_a.data[0].title == "coder:auth"
    assert children_b.data[0].title == "coder:auth"
    # IDs must be distinct — proves they're different rows.
    assert children_a.data[0].id != children_b.data[0].id


async def test_ambient_hint_appears_in_followup_prompt(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    Phase 4 D6: after a named sub-agent is spawned, every
    subsequent LLM call's system instructions include an
    ``Open sub-agents:`` block listing it. Without the hint
    the LLM forgets the name across turns and re-spawns
    instead of continuing.
    """
    await _create_parent_with_sub_agent(
        client,
        parent_name="parent-hint",
        sub_agent_name="coder",
    )

    # Single router for the whole flow: spawn on turn 1, no
    # tools on turn 2 (so turn 2 finalizes after one iter and
    # we get a clean prompt to inspect).
    def _route(kwargs: dict[str, Any]) -> list[dict[str, str]] | None:
        input_str = json.dumps(kwargs.get("input", []))
        if "spawn-hint-marker" in input_str and "call_spawn_hint" not in input_str:
            return [
                {
                    "call_id": "call_spawn_hint",
                    "name": "spawn_sub_agent",
                    "arguments": json.dumps(
                        {"type": "coder", "name": "auth", "input": "x"},
                    ),
                }
            ]
        return None

    for _ in range(40):
        mock_llm.add_call(text="ok", tool_calls_fn=_route)

    r1 = await create_test_response(
        client,
        model="parent-hint",
        input_text="spawn-hint-marker: start it",
    )
    await _wait_for_completion(client, r1.body["id"])

    r2 = await create_test_response(
        client,
        model="parent-hint",
        input_text="What sub-agents do I have?",
        previous_response_id=r1.body["id"],
    )
    await _wait_for_completion(client, r2.body["id"])

    # Walk every queued mock call's received_kwargs and find
    # one whose ``instructions`` carries the ambient hint with
    # the spawned name. By the time turn 2 finishes, at least
    # one of its LLM calls must have seen the hint.
    matched = [
        c
        for c in mock_llm._calls
        if c.received_kwargs is not None
        and "coder:auth" in str(c.received_kwargs.get("instructions") or "")
        and "Open sub-agents:" in str(c.received_kwargs.get("instructions") or "")
    ]
    assert matched, (
        f"No LLM call across the whole test had the ambient hint "
        f"with 'Open sub-agents:' + 'coder:auth' in its "
        f"instructions field. D6 wiring broken or no turn-2 call "
        f"reached the LLM with the hint applied."
    )
