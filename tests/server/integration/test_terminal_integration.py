"""Integration tests for the terminal_run tool through the full server stack.

Mocks the LLM to issue specific ``terminal_run`` calls, then drives
the real FastAPI → ToolManager → TerminalManager → sandboxed bash
pipeline. Covers invariants that are too timing-dependent to provoke
via a real LLM (shell_busy races, 10-shell cap), plus crash recovery
which the real LLM can't reliably trigger.

Unit tests cover the primitives (``tests/terminals/``); these tests
prove the primitives work through the actual HTTP dispatch + DBOS
workflow path.
"""

from __future__ import annotations

import io
import json
import tarfile
from typing import Any

import httpx
import pytest
import yaml

from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response
from tests.server.integration.test_local_tool_integration import (
    _wait_for_completion,
)

pytestmark = [pytest.mark.asyncio]

_AGENT_NAME = "terminal-integration-agent"


def _build_terminal_agent_bundle() -> bytes:
    """Build an agent bundle whose only tools are the three terminal builtins.

    Matches the ``terminal_test`` fixture agent shape but inline so
    this file doesn't couple to the example agent directory. Sandbox
    is disabled for this agent because these integration tests run
    without srt guaranteed on PATH — the shell_busy, cap, and crash
    assertions are about the manager / shell layer, not the sandbox.

    :returns: Raw tar.gz bytes with a config.yaml declaring
        ``terminal_run``, ``terminal_list``, ``terminal_close`` as
        builtins.
    """
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {
            "model": _AGENT_NAME,
            "connection": {"api_key": "test-key"},
        },
        "tools": {
            "builtins": ["terminal_run", "terminal_list", "terminal_close"],
        },
    }
    config_bytes = yaml.dump(config).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


async def _create_terminal_agent(client: httpx.AsyncClient) -> None:
    """Upload the terminal-integration agent bundle.

    :param client: HTTP client pointed at the test server.
    """
    bundle = _build_terminal_agent_bundle()
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, (
        f"Agent creation failed: {resp.status_code} {resp.text}"
    )


def _tool_outputs(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and JSON-parse every function_call_output in a response.

    :param body: The response body from ``GET /v1/responses/{id}``.
    :returns: Parsed tool outputs. Unparseable entries are skipped.
    """
    parsed: list[dict[str, Any]] = []
    for item in body.get("output", []):
        if item.get("type") != "function_call_output":
            continue
        raw = item.get("output") or ""
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            obj["_call_id"] = item.get("call_id")
            parsed.append(obj)
    return parsed


@pytest.fixture(autouse=True)
def _force_unsandboxed_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the terminal registry to spawn shells without srt.

    These integration tests assert behavior of the manager/shell
    layer (concurrency, cap, crash recovery) which is independent
    of whether the sandbox is active. Running without srt makes the
    tests portable to any environment without a node/srt install,
    and avoids accidentally exercising sandbox code paths whose
    own tests are elsewhere.

    :param monkeypatch: Pytest's monkeypatch fixture.
    """
    from agent_plane.runtime import _globals
    from agent_plane.terminals import TerminalManagerRegistry

    # Replace the global registry with an unsandboxed one. Tiny
    # reaper interval doesn't matter — reaper isn't started in
    # these tests (no lifespan hook on the test client).
    monkeypatch.setattr(
        _globals,
        "_terminal_registry",
        TerminalManagerRegistry(sandbox_enabled=False),
    )


# ── shell_busy via real parallel tool calls ────────────────────


async def test_shell_busy_on_parallel_same_shell_dispatch(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Two parallel terminal_run calls on one shell → exactly one shell_busy.

    The mock LLM returns a single response containing two
    ``terminal_run`` function_calls targeting ``shell="default"``.
    The workflow dispatches them via ``asyncio.ensure_future`` +
    ``asyncio.to_thread``, so both land in the shell's ``run_sync``
    nearly simultaneously. The non-blocking cmd lock lets exactly
    one command through; the other gets ``status="shell_busy"``
    without blocking.

    Failure modes this catches:
    - Blocking acquire: the second call queues instead of bailing
      out → both complete, no shell_busy ever appears.
    - Both bailing: a race on the lock's state where both threads
      see it free → two shells or corrupted PTY state.
    - Missing propagation: the tool handler catches ShellBusy and
      converts to an unrelated error string instead of preserving
      the ``status`` field.
    """
    await _create_terminal_agent(client)

    # Call 1: one assistant message with two parallel terminal_run
    # tool calls. Call 1 runs a ~0.5s sleep to hold the cmd lock;
    # call 2 is a quick echo — it must land while the sleep is
    # in flight, get shell_busy back, and return without blocking.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_sleep",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {"command": "sleep 0.5 && echo slow", "shell": "default"}
                ),
            },
            {
                "call_id": "call_fast",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {"command": "echo fast", "shell": "default"}
                ),
            },
        ],
    )
    # Call 2: final assistant text after tool outputs arrive.
    mock_llm.add_call(text="Both tool calls completed.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Fire two parallel terminal_run calls.",
    )
    assert result.status_code == 200
    body = await _wait_for_completion(client, result.body["id"])
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    outputs = _tool_outputs(body)
    # Both tool calls should produce outputs (workflow dispatches
    # all declared function_calls, even losers on the shell_busy
    # race).
    assert len(outputs) == 2, (
        f"Expected 2 terminal_run outputs (one per parallel call), "
        f"got {len(outputs)}. If 1, the workflow lost a tool output "
        f"instead of propagating shell_busy."
    )
    statuses = [o.get("status") for o in outputs]
    # Exactly one of the two got the lock and ran; the other got
    # fail-fast shell_busy. Which one wins is timing-dependent and
    # not worth asserting — only the count matters.
    assert statuses.count("shell_busy") == 1, (
        f"Expected exactly one shell_busy status across the two "
        f"parallel calls, got {statuses}. If 0, both serialized "
        f"behind the cmd lock (blocking acquire regression). If 2, "
        f"both calls raced past the lock (missing lock regression)."
    )
    assert statuses.count("completed") == 1, (
        f"Expected exactly one 'completed' status (the winner), "
        f"got {statuses}."
    )


# ── shell_cap_exceeded after 10 shells ─────────────────────────


async def test_shell_cap_exceeded_on_eleventh_shell(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """Creating an 11th distinct shell in one conversation returns
    ``shell_cap_exceeded``.

    The mock LLM issues 11 terminal_run calls in a single response,
    each targeting a unique shell name (``s0`` … ``s10``). The
    manager's 10-shell cap must fire on the 11th, producing one
    ``shell_cap_exceeded`` output alongside 10 ``completed`` outputs.

    Failure modes this catches:
    - Cap never enforced: all 11 succeed (unbounded shell growth).
    - Cap wrong value: failure at 9 (cap=8) or at 12 (cap=11) etc.
    - Error wrapped as generic error: shell_cap_exceeded not
      surfaced so agents can't recover cleanly.
    """
    await _create_terminal_agent(client)

    tool_calls = [
        {
            "call_id": f"call_{i}",
            "name": "terminal_run",
            "arguments": json.dumps(
                {"command": "echo hi", "shell": f"s{i}"}
            ),
        }
        for i in range(11)
    ]
    mock_llm.add_call(tool_calls=tool_calls)
    mock_llm.add_call(text="Done.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Spawn 11 shells.",
    )
    assert result.status_code == 200
    body = await _wait_for_completion(client, result.body["id"])
    assert body["status"] == "completed"

    outputs = _tool_outputs(body)
    assert len(outputs) == 11, (
        f"Expected 11 terminal_run outputs (one per declared call), "
        f"got {len(outputs)}."
    )
    statuses = [o.get("status") for o in outputs]
    # Exactly one over-cap call: the 11th attempt raises
    # ShellCapExceeded which the tool translates to
    # ``status="shell_cap_exceeded"``.
    assert statuses.count("shell_cap_exceeded") == 1, (
        f"Expected exactly one shell_cap_exceeded among 11 parallel "
        f"calls, got statuses={statuses}. If 0, the cap isn't firing "
        f"— the manager accepted all 11 shells. If >1, more than one "
        f"call hit the cap-check race simultaneously (unexpected — "
        f"the manager's lock should serialize cap checks)."
    )
    # The other 10 completed successfully.
    assert statuses.count("completed") == 10, (
        f"Expected 10 'completed' under the cap, got "
        f"{statuses.count('completed')} (total statuses: {statuses})."
    )


# ── Crash recovery: dead shell replaced on next run ───────────


async def test_crashed_shell_replaced_on_next_run(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
) -> None:
    """A shell that died (via ``exit``) is swept and replaced on next use.

    Turn 1: ``exit 0`` — terminates the bash subprocess itself.
    Turn 2: ``echo alive`` — the manager must detect the dead
    shell, sweep it from its map, spawn a fresh replacement, and
    run the command successfully.

    Without the sweep, turn 2 would return ``shell_crashed`` and
    the agent would be permanently wedged on this shell name.
    Covers ``TerminalManager._get_or_create_shell``'s dead-shell
    branch through the full HTTP path.
    """
    await _create_terminal_agent(client)

    # Turn 1: the exit command. Whether bash emits the D marker
    # before dying is timing-dependent, so this turn's tool output
    # status could be either ``completed`` or ``shell_crashed`` —
    # that's fine. What matters is the shell is DEAD afterwards.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_exit",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {"command": "exit 0", "shell": "default"}
                ),
            },
        ],
    )
    mock_llm.add_call(text="Shell is gone.")

    r1 = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Kill the shell.",
    )
    assert r1.status_code == 200
    body1 = await _wait_for_completion(client, r1.body["id"])
    assert body1["status"] == "completed"

    # Turn 2: same shell name, fresh bash should appear via
    # auto-sweep.
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_alive",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {"command": "echo alive-in-fresh-shell", "shell": "default"}
                ),
            },
        ],
    )
    mock_llm.add_call(text="Shell recovered.")

    r2 = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Use the shell again.",
        previous_response_id=r1.body["id"],
    )
    assert r2.status_code == 200
    body2 = await _wait_for_completion(client, r2.body["id"])
    assert body2["status"] == "completed"

    # Turn 2's tool output must be ``completed`` (fresh shell ran
    # echo). If it's ``shell_crashed``, the manager didn't sweep
    # and the agent is permanently stuck.
    outputs2 = _tool_outputs(body2)
    assert len(outputs2) == 1
    assert outputs2[0].get("status") == "completed", (
        f"Expected 'completed' on fresh shell, got "
        f"{outputs2[0].get('status')!r}. The dead-shell sweep in "
        f"TerminalManager._get_or_create_shell is not running — "
        f"agents who ``exit`` a shell would be permanently wedged."
    )
    # And the echo output actually appeared in stdout, proving the
    # fresh shell actually executed the command (not just returned
    # a canned 'completed' from some other path).
    stdout = outputs2[0].get("stdout") or ""
    assert "alive-in-fresh-shell" in stdout, (
        f"Fresh shell ran but produced no output — likely a spawn "
        f"failure. stdout: {stdout!r}"
    )


# ── Idle reaper collects abandoned managers through the registry ──


async def test_idle_reaper_collects_managers_through_registry(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle manager is reaped when its threshold elapses.

    Swaps in a registry with ``idle_timeout_s=0.05`` so the reaper's
    single-sweep method treats the manager as idle immediately.
    After issuing a terminal_run to create the manager, we drive
    one sweep manually (avoids racing the real asyncio reaper loop)
    and assert the conversation's manager is gone.

    Complements the unit test ``test_reaper_collects_idle_managers``
    by proving the reaper operates on the same registry the HTTP
    dispatch path uses — not a standalone instance. Catches a class
    of bug where the reaper is wired to the wrong registry object.
    """
    from agent_plane.runtime import _globals
    from agent_plane.terminals import TerminalManagerRegistry

    # Replace the autouse-fixture's registry with one whose idle
    # threshold is 50 ms. Directly assigning to the module global
    # overrides the earlier monkeypatch.
    reg = TerminalManagerRegistry(
        sandbox_enabled=False,
        idle_timeout_s=0.05,
    )
    monkeypatch.setattr(_globals, "_terminal_registry", reg)

    await _create_terminal_agent(client)

    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_touch",
                "name": "terminal_run",
                "arguments": json.dumps(
                    {"command": "echo hi", "shell": "default"}
                ),
            },
        ],
    )
    mock_llm.add_call(text="done")

    r = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Create a shell.",
    )
    body = await _wait_for_completion(client, r.body["id"])
    assert body["status"] == "completed"

    # After the task, the registry has one manager (whichever
    # conversation the workflow created). Its ``last_activity`` is
    # from when the tool ran, i.e. already > 50 ms ago by the time
    # the HTTP round-trip + DBOS workflow + polling all finish.
    before = reg.active_conversation_ids()
    assert len(before) == 1, (
        f"Expected exactly one registered manager before reaping, "
        f"got {before}. If 0, the terminal_run call didn't create "
        f"a manager — likely a dispatch failure masked by the task "
        f"status. If >1, prior tests leaked state."
    )

    # Drive one sync sweep through the public-ish reaper entry.
    # We use the module-level ``_reap_idle_once`` helper so we don't
    # have to spin up an asyncio reaper loop just for a single pass.
    reg._reap_idle_once()  # noqa: SLF001

    after = reg.active_conversation_ids()
    assert after == [], (
        f"Expected the reaper to collect the idle manager, but "
        f"{after} remain. Either the reaper isn't operating on this "
        f"registry or the idle threshold check is broken. If this "
        f"fails with `before == after`, check that _reap_idle_once "
        f"actually mutates the registry's map."
    )
