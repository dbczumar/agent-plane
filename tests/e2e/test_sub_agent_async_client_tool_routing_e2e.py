"""
E2E reproduction of audit-fix-#1: sub-agent's async client-tool
result is misrouted to the root parent's drain instead of the
sub-agent's drain.

Setup:
- Parent agent ``sub-agent-async-client-tool-test`` spawns one
  sub-agent ``worker:alpha``.
- Worker calls the request-supplied client tool ``async_compute``
  with ``synchronous: false`` — server dispatches as a
  ``kind="client_tool"`` task, returns a
  ``{task_id, kind: "client_tool", ...}`` handle to the worker.
- Test PATCHes ``async_tool_results`` with a marker payload.

Expected (after fix):
- The async_work_complete signal is delivered to the WORKER's
  drain (worker is the immediate calling agent).
- Worker's conversation gains
  ``[System: task <task_id> (client_tool) completed]\\n<output>``.
- Worker's final assistant message is ``WORKER_FINAL:<output>``.
- Parent reads the worker's reply and emits ``WORKER_REPLY:<output>``.

Today (bug):
- The PATCH handler signals ``task.root_task_id`` (the parent),
  not the immediate caller (the worker). The worker's drain never
  wakes; the worker times out / hits max_iterations / never sees
  the result. Test fails at the worker-completion assertion.

This test is intentionally written to PASS only after the fix
lands. Run as::

    pytest tests/e2e/test_sub_agent_async_client_tool_routing_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v

Excluded from the default ``pytest`` run via
``--ignore=tests/e2e``.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_FIXTURE = _FIXTURES_DIR / "sub-agent-async-client-tool-test"

# Marker the test PATCHes back. The worker's AGENTS.md tells it
# to echo the system-message body verbatim with WORKER_FINAL:
# prefix, and the parent forwards with WORKER_REPLY: prefix —
# so finding both prefixes containing this marker proves the
# end-to-end signal landed on the worker's drain (not the
# parent's, which would skip the worker entirely).
_MARKER = "ASYNC_CLIENT_TOOL_DRAIN_ROUTING_OK_42"

# Async client tool the worker is told to call with
# ``synchronous: false``. The schema declares ``synchronous`` in
# properties so the server's ``_wants_async_dispatch`` honors
# the per-call arg (audit fix #3).
_ASYNC_CLIENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "async_compute",
        "description": (
            "Long-running client-side computation. "
            "Always call with synchronous=false. The result is "
            "delivered later as a system message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "Echo this string back as the result.",
                },
                "synchronous": {
                    "type": "boolean",
                    "description": (
                        "MUST be set to false. Dispatches as a "
                        "background task and returns a "
                        "{task_id, kind: 'client_tool'} handle "
                        "immediately. The actual result arrives "
                        "later as a system message."
                    ),
                },
            },
            # `synchronous` is required so the LLM cannot omit it
            # — without that, the server's _wants_async_dispatch
            # check returns False (sync default) and the test
            # exercises the wrong code path.
            "required": ["value", "synchronous"],
        },
    },
}


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
def routing_test_agent(http_client: httpx.Client) -> str:
    """Upload the sub-agent-async-client-tool-test fixture."""
    return _upload(http_client, _FIXTURE)


def _items(http_client: httpx.Client, conv_id: str) -> list[dict[str, Any]]:
    """
    Fetch all items in a conversation in store order.

    :param http_client: HTTP client.
    :param conv_id: Conversation id.
    :returns: List of conversation item dicts.
    """
    resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    resp.raise_for_status()
    data: list[dict[str, Any]] = resp.json()["data"]
    return data


def _wait_for_worker_conversation(
    http_client: httpx.Client,
    parent_conv_id: str,
    timeout_s: float = 60.0,
) -> str:
    """
    Discover the worker sub-agent's conversation id.

    The route layer doesn't currently expose a
    ``parent_conversation_id`` filter on
    ``GET /v1/conversations``, so this helper uses the
    spawn-side handle FCO to learn the sub-agent's task_id,
    then fetches that task to read its ``conversation_id``.

    :param http_client: HTTP client.
    :param parent_conv_id: The parent agent's conversation id.
    :param timeout_s: Max seconds to wait.
    :returns: The worker conversation's id.
    :raises AssertionError: If no spawn FCO appears within
        the timeout — the parent never spawned the worker
        (LLM didn't follow instructions, or spawn failed).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        items = _items(http_client, parent_conv_id)
        last_call_name: str | None = None
        for item in items:
            if item.get("type") == "function_call":
                last_call_name = item.get("name")
            elif item.get("type") == "function_call_output":
                if last_call_name != "spawn_sub_agent":
                    continue
                try:
                    handle = json.loads(item.get("output") or "")
                except json.JSONDecodeError:
                    continue
                if not isinstance(handle, dict):
                    continue
                if handle.get("kind") != "sub_agent":
                    continue
                sub_task_id = handle.get("task_id")
                if not sub_task_id:
                    continue
                # Fetch the sub-agent task to read its conv id.
                task_resp = http_client.get(f"/v1/responses/{sub_task_id}")
                if task_resp.status_code == 200:
                    body = task_resp.json()
                    conv = body.get("conversation") or {}
                    if conv.get("id"):
                        return str(conv["id"])
        time.sleep(0.5)
    raise AssertionError(
        f"No spawn_sub_agent handle appeared in parent conv "
        f"{parent_conv_id} within {timeout_s}s — parent agent never "
        f"spawned the worker. Check parent AGENTS.md / model behavior."
    )


def _wait_for_handle_in_conv(
    http_client: httpx.Client,
    conv_id: str,
    tool_name: str,
    timeout_s: float = 90.0,
) -> str:
    """
    Poll a conversation for a function_call_output whose output
    JSON is an async client-tool handle for ``tool_name``.

    :param http_client: HTTP client.
    :param conv_id: Conversation id to scan.
    :param tool_name: The tool name to match against the
        preceding function_call (so we don't pick up a stray
        handle from another tool).
    :param timeout_s: Max seconds to wait.
    :returns: The handle's ``task_id``.
    :raises AssertionError: If no matching handle appears.
    """
    deadline = time.monotonic() + timeout_s
    last_items: list[dict[str, Any]] = []
    # Once we see the function_call (proof the worker LLM ran the
    # call) we only wait briefly more for the FCO — if dispatch
    # was going to persist it, it would do so within seconds. A
    # long stall after the function_call is the real bug signature.
    saw_call = False
    saw_call_at = 0.0
    while time.monotonic() < deadline:
        items = _items(http_client, conv_id)
        last_items = items
        last_call_name: str | None = None
        for item in items:
            if item.get("type") == "function_call":
                last_call_name = item.get("name")
                if last_call_name == tool_name and not saw_call:
                    saw_call = True
                    saw_call_at = time.monotonic()
            elif item.get("type") == "function_call_output":
                if last_call_name != tool_name:
                    continue
                try:
                    handle = json.loads(item.get("output") or "")
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(handle, dict)
                    and handle.get("kind") == "client_tool"
                    and handle.get("task_id")
                ):
                    return str(handle["task_id"])
        # Cap the wait at +20s after seeing the function_call —
        # FCO dispatch is server-side and synchronous within the
        # workflow, so a 20s gap means the dispatch silently
        # didn't fire (likely audit-fix-#3 schema-validation gate).
        if saw_call and time.monotonic() - saw_call_at > 20.0:
            break
        time.sleep(0.5)
    # Diagnostic dump — show what the conversation actually
    # contains so failures pinpoint LLM-behavior issues vs.
    # test-helper bugs.
    summary = []
    for item in last_items:
        t = item.get("type")
        if t == "function_call":
            summary.append(
                f"function_call name={item.get('name')!r} args={item.get('arguments')!r}"
            )
        elif t == "function_call_output":
            summary.append(f"function_call_output output={(item.get('output') or '')[:200]!r}")
        elif t == "message":
            content = item.get("content") or []
            text = (content[0].get("text") or "") if content else ""
            summary.append(f"message role={item.get('role')!r} text={text[:200]!r}")
        else:
            summary.append(f"{t}=...")
    raise AssertionError(
        f"No async client-tool handle for {tool_name!r} appeared in "
        f"conversation {conv_id} within {timeout_s}s — the worker "
        f"either didn't call the tool or didn't request async dispatch. "
        f"Conversation items ({len(last_items)} total):\n  " + "\n  ".join(summary)
    )


def _patch_async_tool_result(
    http_client: httpx.Client,
    response_id: str,
    task_id: str,
    output: str,
) -> None:
    """
    PATCH the async-tool result back to the server.

    :param http_client: HTTP client.
    :param response_id: The ROOT response id (parent's task id) —
        the PATCH endpoint is on the root response, not the
        client_tool task itself, per server/API.md.
    :param task_id: The client_tool task id from the handle.
    :param output: Result string to deliver.
    """
    resp = http_client.patch(
        f"/v1/responses/{response_id}",
        json={
            "async_tool_results": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "output": output,
                },
            ],
        },
    )
    assert resp.status_code == 200, f"PATCH failed: {resp.status_code} {resp.text!r}"


def _wait_for_response_terminal(
    http_client: httpx.Client,
    response_id: str,
    timeout_s: float = 240.0,
) -> dict[str, Any]:
    """
    Poll a response until it reaches a terminal status.

    :param http_client: HTTP client.
    :param response_id: Response id to poll.
    :param timeout_s: Max seconds to wait.
    :returns: The terminal response body.
    :raises AssertionError: If the response never reaches
        terminal — likely the bug (worker is blocked on its
        drain forever because the signal went to the root
        parent's drain instead).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = http_client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled", "incomplete"):
            return body
        time.sleep(1.0)
    raise AssertionError(
        f"Response {response_id} did not reach terminal within "
        f"{timeout_s}s — likely audit-fix-#1 bug: the worker's drain "
        f"never woke because the async_work_complete signal was "
        f"delivered to the ROOT parent's drain instead of the "
        f"immediate caller (worker)."
    )


def test_sub_agent_async_client_tool_signal_routes_to_worker_drain(
    http_client: httpx.Client,
    routing_test_agent: str,
) -> None:
    """
    The async-client-tool ``async_work_complete`` signal must
    land on the **immediate calling agent**'s drain (the
    worker), NOT on the top-level parent. Reproduces audit
    fix #1.

    Failure modes this test catches:

    - Bug present (today): server signals ``task.root_task_id``
      so the parent (root) gets the message, not the worker.
      The worker's drain never wakes → the worker stalls on
      its next iteration → the response either times out or
      completes without delivering the result. Either the
      poll-for-terminal assertion fires, or the
      worker-conv-contains-marker assertion fires.

    - Fix correct: worker's conversation contains exactly one
      ``[System: task <task_id> (client_tool) completed]``
      message carrying the marker; worker's final assistant
      message is ``WORKER_FINAL:...<marker>...``; parent's
      final assistant message is ``WORKER_REPLY:...<marker>...``.
    """
    # Step 1 — POST: parent will spawn worker which will call
    # async_compute(synchronous: false).
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": routing_test_agent,
            "input": (
                f"Have your worker compute on the value {_MARKER!r}. Forward the worker's reply."
            ),
            "background": True,
            "stream": False,
            "tools": [_ASYNC_CLIENT_TOOL],
        },
    )
    assert resp.status_code == 200, f"POST failed: {resp.status_code} {resp.text!r}"
    parent_response_id = resp.json()["id"]
    parent_conv_id = resp.json()["conversation"]["id"]

    # Step 2 — find the worker's conversation (parent spawned it).
    worker_conv_id = _wait_for_worker_conversation(http_client, parent_conv_id)

    # Step 3 — find the async client-tool handle in the WORKER's
    # conversation. The handle's task_id is the client_tool task
    # the test must PATCH.
    try:
        task_id = _wait_for_handle_in_conv(http_client, worker_conv_id, "async_compute")
    except AssertionError as exc:
        # Augment with parent-conv state so we can tell if
        # the failure is "parent never spawned correctly"
        # vs "worker started but didn't call the tool."
        parent_items = _items(http_client, parent_conv_id)
        parent_summary = []
        for item in parent_items:
            t = item.get("type")
            if t == "function_call":
                parent_summary.append(
                    f"function_call name={item.get('name')!r} args={item.get('arguments')!r}"
                )
            elif t == "function_call_output":
                parent_summary.append(
                    f"function_call_output output={(item.get('output') or '')[:300]!r}"
                )
            elif t == "message":
                content = item.get("content") or []
                text = (content[0].get("text") or "") if content else ""
                parent_summary.append(f"message role={item.get('role')!r} text={text[:300]!r}")
            else:
                parent_summary.append(f"{t}=...")
        raise AssertionError(
            f"{exc}\n\nParent conv items ({len(parent_items)} total):\n  "
            + "\n  ".join(parent_summary)
        ) from exc

    # Step 4 — PATCH the result. Server should signal the
    # worker's drain (audit fix #1). Today it signals the parent's
    # drain instead.
    _patch_async_tool_result(http_client, parent_response_id, task_id, _MARKER)

    # Step 5 — wait for the parent response to terminate. With
    # the audit-fix-#1 bug, the parent often fails (the
    # mis-routed system message confuses the conversation) or
    # times out — either way, terminal is the gate. We don't
    # require ``completed`` here because the misrouted system
    # message can crash the parent's next LLM call (e.g.
    # OpenAI rejects null assistant text after spawn-then-wait).
    # The load-bearing assertions are steps 6-9 below: where
    # did the system message land?
    _wait_for_response_terminal(http_client, parent_response_id, timeout_s=240.0)

    # Step 6 — assert the worker's conversation contains the
    # auto-delivered system message. This is the load-bearing
    # assertion: it proves the signal reached the worker's drain.
    worker_items = _items(http_client, worker_conv_id)
    system_messages = [
        i["content"][0]["text"]
        for i in worker_items
        if i.get("role") == "user"
        and isinstance(i.get("content"), list)
        and i["content"]
        and i["content"][0].get("type") == "input_text"
        and (i["content"][0].get("text") or "").startswith("[System: task ")
    ]
    matching = [m for m in system_messages if _MARKER in m]
    # Pre-check: gather the parent's system messages too so the
    # diagnostic explicitly proves the signal landed on the
    # WRONG drain (the parent's), not just "the worker missed it."
    parent_items_for_diag = _items(http_client, parent_conv_id)
    parent_system_messages = [
        (i["content"][0].get("text") or "")
        for i in parent_items_for_diag
        if i.get("role") == "user"
        and isinstance(i.get("content"), list)
        and i["content"]
        and i["content"][0].get("type") == "input_text"
        and (i["content"][0].get("text") or "").startswith("[System: task ")
    ]
    parent_misrouted = [m for m in parent_system_messages if _MARKER in m]
    assert len(matching) == 1, (
        f"AUDIT BUG #1 CONFIRMED: the worker's conversation has "
        f"{len(matching)} system messages carrying the PATCH'd marker "
        f"(expected 1). The parent's conversation has "
        f"{len(parent_misrouted)} (expected 0). The async_work_complete "
        f"signal was delivered to {'the parent (root)' if parent_misrouted else 'NEITHER agent'} "
        f"instead of the immediate calling agent (the worker). "
        f"Fix: route to task.parent_task_id, not task.root_task_id, "
        f"in agent_plane/server/routes/responses.py:_finalize_and_signal. "
        f"\n  worker_system={system_messages!r}"
        f"\n  parent_system={parent_system_messages!r}"
    )

    # Step 7 — assert the parent's conversation does NOT contain
    # the same system message. Catches the inverse failure: if
    # the signal goes to BOTH (a future fan-out fix gone wrong),
    # the parent shouldn't see the worker's tool result.
    parent_items = _items(http_client, parent_conv_id)
    parent_system_with_marker = [
        i
        for i in parent_items
        if i.get("role") == "user"
        and isinstance(i.get("content"), list)
        and i["content"]
        and i["content"][0].get("type") == "input_text"
        and _MARKER in i["content"][0].get("text", "")
        and i["content"][0].get("text", "").startswith("[System: task ")
    ]
    assert parent_system_with_marker == [], (
        f"Parent's conversation must NOT contain the worker's "
        f"client-tool [System: ...] message — it belongs on the "
        f"worker's drain. If non-empty, the routing has been "
        f"flipped from 'always root' to 'fan out to both' "
        f"(also a bug). Got {parent_system_with_marker!r}"
    )

    # Step 8 — assert the worker's final assistant message
    # contains the marker. Proves the worker's LLM saw the
    # system message AND incorporated it into its reply.
    worker_assistant_texts = [
        i["content"][0]["text"]
        for i in worker_items
        if i.get("role") == "assistant"
        and isinstance(i.get("content"), list)
        and i["content"]
        and i["content"][0].get("type") == "output_text"
    ]
    worker_final = [t for t in worker_assistant_texts if _MARKER in t]
    assert worker_final, (
        f"Worker's final assistant message must contain the "
        f"marker — proves the worker's LLM not only received "
        f"the system message but acted on it. "
        f"worker_assistant_texts={worker_assistant_texts!r}"
    )

    # Step 9 — assert the parent's final assistant message
    # contains the marker (parent forwards worker's reply).
    # End-to-end seal.
    parent_assistant_texts = [
        i["content"][0]["text"]
        for i in parent_items
        if i.get("role") == "assistant"
        and isinstance(i.get("content"), list)
        and i["content"]
        and i["content"][0].get("type") == "output_text"
    ]
    parent_final = [t for t in parent_assistant_texts if _MARKER in t]
    assert parent_final, (
        f"Parent's final assistant message must include the "
        f"marker (parent's job is to forward the worker's reply). "
        f"parent_assistant_texts={parent_assistant_texts!r}"
    )
