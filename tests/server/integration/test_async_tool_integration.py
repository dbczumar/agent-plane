"""
Server-integration tests for ``@tool(synchronous=False)`` end-to-end.

These tests exercise the full Phase 2 async-tool pipeline:

1. The LLM invokes an async tool;
2. ``_call_tool`` dispatches it via ``_dispatch_async_tool`` and
   returns a JSON handle (NOT the inline result);
3. The ``background_tool_workflow`` runs the function in a
   subprocess via the fd-3 protocol;
4. On completion it sends an ``async_work_complete`` payload to
   the parent on the documented topic;
5. The parent's between-iteration drain (D4) picks it up and
   persists a ``[System: task ... completed]`` user message;
6. The LLM sees the system message on the next iteration and
   produces a final response.

This is the only place all six steps run together. Unit tests
cover the helpers individually; this is the integration glue.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response

pytestmark = [pytest.mark.asyncio]

_AGENT_NAME = "async-tool-test-agent"

# A deliberately-trivial async tool — the production breakage we
# care about is in the dispatch + drain plumbing, not in the tool
# body. Returning a fixed string makes the assertion at the end
# unambiguous.
_ASYNC_TOOL_SOURCE = '''\
"""Test async tool — returns a fixed marker so assertions are deterministic."""
from agent_plane.tools import tool


@tool(synchronous=False)
def slow_compute() -> str:
    """Pretend to do background work and return a marker.

    Args:
        (no arguments)
    """
    return "ASYNC_TOOL_DONE_MARKER"
'''


def _build_async_tool_agent_bundle() -> bytes:
    """
    Build an agent bundle exporting one ``synchronous=False`` tool.

    :returns: tar.gz bytes containing config.yaml plus
        ``tools/python/slow_compute.py``.
    """
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {
            "model": _AGENT_NAME,
            "connection": {"api_key": "test-key"},
        },
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        cfg_bytes = yaml.dump(config).encode()
        cfg_info = tarfile.TarInfo(name="config.yaml")
        cfg_info.size = len(cfg_bytes)
        tf.addfile(cfg_info, io.BytesIO(cfg_bytes))

        src_bytes = _ASYNC_TOOL_SOURCE.encode()
        src_info = tarfile.TarInfo(name="tools/python/slow_compute.py")
        src_info.size = len(src_bytes)
        tf.addfile(src_info, io.BytesIO(src_bytes))
    return buf.getvalue()


async def _create_async_tool_agent(client: httpx.AsyncClient) -> dict[str, Any]:
    """
    Upload the async-tool test bundle.

    :param client: HTTP client for the test server.
    :returns: Agent creation response JSON.
    """
    bundle = _build_async_tool_agent_bundle()
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent upload failed: {resp.status_code} {resp.text}"
    return resp.json()


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status or fail loud.

    :param client: HTTP client.
    :param response_id: The response/task ID to poll.
    :returns: Terminal response body.
    :raises AssertionError: If the response never reaches terminal
        within ~20 seconds (200 polls × 0.1s).
    """
    for _ in range(200):
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
    """
    Fetch all conversation items.

    :param client: HTTP client.
    :param conv_id: The conversation ID.
    :returns: List of item dicts in store order.
    """
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


# ─── Tests ───────────────────────────────────────────────────


async def test_async_tool_dispatch_returns_handle_and_auto_delivers_result(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """
    End-to-end: async tool dispatch → handle → background work →
    drain → system message → LLM final response.

    Sequence:
    * LLM call 1 returns a tool_call to ``slow_compute``.
    * ``_call_tool`` dispatches via ``_dispatch_async_tool``;
      returns a JSON handle dict to the LLM. The
      ``function_call_output`` for that turn carries the JSON
      handle string (not the eventual ``ASYNC_TOOL_DONE_MARKER``).
    * ``background_tool_workflow`` runs in the background and
      signals ``async_work_complete``.
    * The parent's between-iteration drain wakes on the next
      iteration, persists a ``[System: ...]`` user message
      containing the marker.
    * LLM call 2 receives the system message in its history and
      produces a final response.

    What breaks if wrong (each anchors a specific assertion):
    * Dispatch missed: ``function_call_output`` would carry the
      tool's actual return value (the marker) — not a JSON handle.
    * Drain skipped: the ``[System: ... completed]`` message would
      be absent from the conversation; LLM call 2 wouldn't see it
      in its input.
    * Wrong handle shape: the parsed JSON would be missing fields
      (``task_id``, ``status="in_progress"``, ``message``).
    """
    await _create_async_tool_agent(client)

    # LLM call 1: invoke the async tool. The runtime intercepts
    # this in _call_tool because the tool's metadata.synchronous
    # is False — the LLM never gets the marker inline.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_async_1",
                "name": "slow_compute",
                "arguments": "{}",
            },
        ],
    )
    # LLM call 2: the LLM has the handle in history but the
    # background tool may not have completed yet. The LLM
    # responds with placeholder text — since pending_tool_tasks
    # is non-empty, the runtime will block on the drain instead
    # of treating this as the final response.
    mock_llm.add_call(text="Working on it…")
    # LLM call 3: by now the auto-delivered system message is in
    # history. We capture received_kwargs on this call to assert
    # the marker reached the prompt.
    call_3 = mock_llm.add_call(text="Got the async result.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Run the async compute, please.",
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    completed = await _wait_for_completion(client, response_id)
    # Completion path proves the workflow finished — without the
    # drain, the end-of-turn auto-collect would block forever
    # waiting on a topic signal that never resulted in a
    # persisted system message.
    assert completed["status"] == "completed", (
        f"Expected completed; got {completed['status']}. "
        f"Error: {completed.get('error')}"
    )

    items = await _get_items(client, conv_id)
    types_in_order = [i["type"] for i in items]

    # ── 1. dispatch — function_call_output carries the JSON handle
    fco_items = [
        i
        for i in items
        if i.get("type") == "function_call_output"
        and i.get("call_id") == "call_async_1"
    ]
    assert len(fco_items) == 1, (
        f"Expected 1 function_call_output for the async tool; "
        f"got {len(fco_items)}. items={types_in_order}"
    )
    handle_payload = json.loads(fco_items[0]["output"])
    # If dispatch fell back to the sync path, this string would
    # be the marker ASYNC_TOOL_DONE_MARKER, not a JSON dict.
    assert handle_payload["tool_name"] == "slow_compute"
    assert handle_payload["status"] == "in_progress"
    # The handle's task_id must be a string and non-empty — the
    # LLM relies on it for check_task / cancel_task lookups.
    assert isinstance(handle_payload["task_id"], str)
    assert handle_payload["task_id"], "Empty task_id in handle"
    # The message field must name the literal task_id so the LLM
    # can see what to pass to check_task. If absent, the LLM has
    # to guess.
    assert handle_payload["task_id"] in handle_payload["message"]

    # ── 2. drain — a [System: task ... completed] user message
    # must appear in the conversation, and it must contain the
    # tool's actual return value (the marker).
    user_texts = [
        i["content"][0]["text"]
        for i in items
        if i.get("role") == "user" and i["content"][0].get("type") == "input_text"
    ]
    completion_messages = [
        t for t in user_texts if t.startswith("[System: task ")
    ]
    assert len(completion_messages) == 1, (
        f"Expected exactly 1 auto-delivered completion message; "
        f"got {len(completion_messages)}. user_texts={user_texts}"
    )
    completion_text = completion_messages[0]
    assert "completed" in completion_text, (
        f"Auto-delivered message must mark status; got {completion_text!r}"
    )
    assert "ASYNC_TOOL_DONE_MARKER" in completion_text, (
        f"Tool's actual return value missing from auto-delivered "
        f"message — drain may have dropped the payload body. "
        f"Got {completion_text!r}"
    )

    # ── 3. LLM saw the system message
    # Without this, the test would pass even if the system message
    # were persisted but never reached the prompt (the half-broken
    # case where drain runs but _sync_history doesn't pick it up).
    assert call_3.received_kwargs is not None, (
        "Third LLM call was never made — workflow ended before "
        "the LLM could respond to the auto-delivered result."
    )
    llm_input_text = json.dumps(call_3.received_kwargs["input"])
    assert "ASYNC_TOOL_DONE_MARKER" in llm_input_text, (
        "The marker must appear in LLM call 2's input. If missing, "
        "the auto-delivered system message was persisted but not "
        "synced into the in-memory history."
    )
