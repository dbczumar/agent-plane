"""End-to-end test for stateful ``@tool`` functions via ``ToolState``.

Exercises the task queue tools bundled with the archer agent:
``add_task``, ``list_tasks``, ``update_task_status``. The queue is
held in per-agent ``ToolState`` (see ``designs/TOOL_STATE.md``), so
state must survive across multiple user turns in the same
conversation.

The assertions are made against the raw tool outputs stored on the
conversation — NOT against the LLM's natural-language summary of
what it did — so LLM wording flakiness can't poison the test. As
long as the LLM actually calls the tools (which we assert by
inspecting ``function_call`` items), the state machinery itself is
what we're measuring.

What breaks if wrong:

- Schema builder doesn't skip ``tool_state``: the LLM tries to fill
  the parameter and the call returns a validation error. No
  ``function_call`` items for the tool name.
- Runner doesn't inject ``tool_state``: subprocess exits non-zero
  with TypeError; the output is an "Error:" string, not valid JSON.
- srt sandbox blocks writes to ``.tool_state``: output is the
  "Read-only file system" error string.
- ``transaction()`` broken or state not per-agent-per-conversation:
  turn 2's ``list_tasks`` output is an empty list instead of the
  tasks added in turn 1.

Runs against a real LLM via the archer agent. Requires
``--llm-api-key``.
"""

from __future__ import annotations

import json

import pytest
from agent_plane_client import AgentPlaneClient


async def _get_tool_outputs(
    client: AgentPlaneClient,
    conversation_id: str,
    tool_name: str,
) -> list[str]:
    """Return the raw string outputs of every ``tool_name`` call in order.

    Walks the conversation's function_call / function_call_output
    items and returns the outputs matching ``tool_name``. This is
    the assertion surface for state-persistence tests — the LLM's
    natural-language summary is not — because tool outputs are
    deterministic where the LLM's prose isn't.

    :param client: The agent-plane SDK client.
    :param conversation_id: The conversation to inspect.
    :param tool_name: Only outputs of calls to this tool are returned.
    :returns: Ordered list of raw output strings.
    """
    items = await client.conversations.list_items(conversation_id, limit=100)
    calls_by_id: dict[str, dict] = {}
    for item in items:
        if item.get("type") == "function_call" and item.get("name") == tool_name:
            calls_by_id[item["call_id"]] = item
    outputs: list[str] = []
    for item in items:
        if item.get("type") == "function_call_output":
            cid = item.get("call_id")
            if cid in calls_by_id:
                outputs.append(str(item.get("output", "")))
    return outputs


@pytest.mark.asyncio()
async def test_archer_task_queue_persists_state_across_turns(
    live_server: str,
    archer_agent: str,
    llm_api_key: str,
) -> None:
    """Verify ToolState round-trips across four agent turns.

    Turn 1: agent adds two tasks (alpha, beta).
    Turn 2: agent lists tasks — output contains both.
    Turn 3: agent marks task 1 as done.
    Turn 4: agent lists pending — output contains beta, not alpha.

    Assertions are on the raw tool outputs stored on the
    conversation, NOT the LLM's summary, so wording variance
    can't break the test.
    """
    async with AgentPlaneClient(base_url=live_server) as client:
        session = client.session(model=archer_agent)

        # ── Turn 1: add alpha then beta ──────────────────────
        await session.query(
            "Call add_task twice. First call: description='alpha'. "
            "Second call: description='beta'. Then reply 'ok'."
        )
        # session.current_response_id reflects turn 1's response.
        # The conversation id is needed for item listing; we have
        # to fetch the response to discover it (the SDK doesn't
        # expose conversation_id on the session directly).
        resp = await client.responses.get(session.current_response_id)
        conversation_id = resp.conversation.id if resp.conversation else None
        assert conversation_id is not None, (
            "Turn 1 response has no conversation_id — the server "
            "didn't attach one, or the SDK parse is broken."
        )

        add_outputs = await _get_tool_outputs(client, conversation_id, "add_task")
        assert len(add_outputs) == 2, (
            f"Turn 1 should have called add_task exactly twice (LLM "
            f"compliance check), got {len(add_outputs)}. If 0, the "
            f"tool never ran; if 1, the LLM stopped short."
        )
        added = [json.loads(o) for o in add_outputs]
        descs = sorted(t["description"] for t in added)
        assert descs == ["alpha", "beta"], (
            f"Expected tasks ['alpha', 'beta'], got {descs}. LLM likely ignored the prompt."
        )
        ids = sorted(t["id"] for t in added)
        assert ids == [1, 2], (
            f"Expected IDs [1, 2] (monotonic from _empty_state), got {ids}. "
            f"A regression in the next_id bump would show up as duplicates."
        )

        # ── Turn 2: list all ─────────────────────────────────
        await session.query("Call list_tasks with no arguments (status null). Reply 'listed'.")
        list_outputs = await _get_tool_outputs(client, conversation_id, "list_tasks")
        assert len(list_outputs) == 1, (
            f"Turn 2 should have called list_tasks once, got {len(list_outputs)}."
        )
        listed = json.loads(list_outputs[0])
        listed_descs = sorted(t["description"] for t in listed)
        # If ToolState isn't persisting across turns, this list
        # is empty. The canary for the whole feature.
        assert listed_descs == ["alpha", "beta"], (
            f"Turn 2 list_tasks should see both tasks from turn 1, "
            f"got {listed_descs}. If [], ToolState isn't persisting; "
            f"if one entry, add_task's transaction dropped a write."
        )

        # ── Turn 3: mark task 1 done ─────────────────────────
        await session.query(
            "Call update_task_status with task_id=1 and new_status='done'. Reply 'done'."
        )
        upd_outputs = await _get_tool_outputs(client, conversation_id, "update_task_status")
        assert len(upd_outputs) == 1, (
            f"Turn 3 should have called update_task_status once, got {len(upd_outputs)}."
        )
        updated = json.loads(upd_outputs[0])
        assert updated["id"] == 1 and updated["status"] == "done", (
            f"update_task_status should have returned the updated task "
            f"with id=1 status='done', got {updated!r}."
        )

        # ── Turn 4: list pending ─────────────────────────────
        await session.query("Call list_tasks with status='pending'. Reply 'listed'.")
        list_outputs_2 = await _get_tool_outputs(client, conversation_id, "list_tasks")
        assert len(list_outputs_2) == 2, (
            f"Cumulative list_tasks calls should be 2, got {len(list_outputs_2)}."
        )
        pending = json.loads(list_outputs_2[-1])
        pending_descs = sorted(t["description"] for t in pending)
        assert pending_descs == ["beta"], (
            f"Turn 4 list_tasks(status='pending') should see only "
            f"'beta' (alpha was marked done), got {pending_descs}. "
            f"If ['alpha', 'beta'], update_task_status didn't "
            f"persist; if [], the filter is broken or state was lost."
        )
