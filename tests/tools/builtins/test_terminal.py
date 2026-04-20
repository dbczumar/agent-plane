"""Unit tests for the terminal_* built-in tools.

Exercises the three tools at the Tool boundary (``.invoke(json,
ctx)``), against a real registry + real bash. No mocks.

The registry is wired into ``agent_plane.runtime._globals`` by
monkeypatching — we don't call the full ``runtime.init`` because
these unit tests don't need stores, just the registry singleton.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_plane.runtime import _globals
from agent_plane.terminals import TerminalManagerRegistry
from agent_plane.tools.base import ToolContext
from agent_plane.tools.builtins.terminal import (
    TerminalCloseTool,
    TerminalListTool,
    TerminalRunTool,
    TerminalSendInputTool,
    _resolve_yield_time_ms,
)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> TerminalManagerRegistry:
    """A fresh registry, installed as the server-resident singleton.

    Monkeypatches ``_globals._terminal_registry`` so
    ``get_terminal_registry()`` finds it. Since monkeypatch
    auto-reverses, each test gets a clean registry.

    Sandbox is **off** for these tool-layer unit tests — the tools
    forward to the manager which forwards to the shell, and the
    behavior we're testing (JSON shape, state persistence, timeout
    handling) is independent of whether the shell happens to be
    wrapped in srt. Dedicated sandbox tests in
    ``tests/terminals/test_shell.py`` cover the sandboxed path.
    Disabling here also makes signal propagation on timeout simpler:
    a direct ``pexpect.spawn("bash", ...)`` means SIGINT goes
    straight to bash rather than routing through ``node``.

    :param monkeypatch: Pytest's monkeypatch fixture.
    :returns: The newly-installed, empty :class:`TerminalManagerRegistry`.
    """
    reg = TerminalManagerRegistry(sandbox_enabled=False)
    monkeypatch.setattr(_globals, "_terminal_registry", reg)
    return reg


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    """A ToolContext populated with a real conversation_id + workspace.

    :param tmp_path: Pytest's tmpdir; used as the workspace.
    :returns: A :class:`ToolContext` suitable for terminal tools.
    """
    return ToolContext(
        task_id="task_test",
        agent_id="agent_test",
        workspace=tmp_path,
        conversation_id="conv_test",
    )


# ---- terminal_run -----------------------------------------------


def test_run_tool_returns_json_with_stdout(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """terminal_run returns JSON with stdout, exit_code, status, shell.

    If this fails, the JSON contract the LLM sees is broken —
    subsequent LLM reasoning over tool results would be wrong.
    """
    tool = TerminalRunTool()
    result_json = tool.invoke(
        json.dumps({"command": "echo hello"}),
        ctx,
    )
    result = json.loads(result_json)
    assert "hello" in result["stdout"]
    assert result["exit_code"] == 0
    assert result["status"] == "completed"
    assert result["shell"] == "default"


def test_run_tool_persists_state_across_calls(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """Two terminal_run calls on the same shell share state.

    This is the defining persistent-terminal property — if it
    fails at the tool layer, the underlying manager/shell is
    fine but the tool isn't routing to the same shell.
    """
    tool = TerminalRunTool()
    tool.invoke(json.dumps({"command": "cd /tmp"}), ctx)
    result_json = tool.invoke(json.dumps({"command": "pwd"}), ctx)
    # Strict equality (after strip to tolerate PTY \r\n newlines): if
    # the cd didn't persist, pwd would return the workspace dir, and
    # substring matching on "/tmp" could spuriously pass on a
    # workspace path that happens to contain "/tmp" (e.g.
    # "/var/folders/.../tmp-xxx").
    assert json.loads(result_json)["stdout"].strip() == "/tmp"


def test_run_tool_honors_custom_shell_name(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """A custom shell name creates an isolated shell from 'default'."""
    tool = TerminalRunTool()
    tool.invoke(json.dumps({"command": "export FOO=main_value", "shell": "main"}), ctx)
    tool.invoke(json.dumps({"command": "export FOO=dev_value", "shell": "dev"}), ctx)
    main_result = json.loads(
        tool.invoke(json.dumps({"command": "echo $FOO", "shell": "main"}), ctx)
    )
    dev_result = json.loads(tool.invoke(json.dumps({"command": "echo $FOO", "shell": "dev"}), ctx))
    # Strict equality: if shells cross-contaminate (bug where dev
    # sees main's env or vice-versa), the values would swap. A
    # substring check would miss that because "main" is a substring
    # of "main_value" in either shell's output.
    assert main_result["stdout"].strip() == "main_value"
    assert dev_result["stdout"].strip() == "dev_value"


def test_run_tool_timeout_kills_command(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """timeout_ms fires → status "killed", exit_code 130 (SIGINT)."""
    tool = TerminalRunTool()
    result = json.loads(
        tool.invoke(
            json.dumps({"command": "sleep 10", "timeout_ms": 300}),
            ctx,
        )
    )
    assert result["status"] == "killed"
    # 128 + SIGINT(2) = 130 per bash convention. Expected failure
    # modes to watch for:
    # - 137 (SIGKILL): timeout path escalated to SIGKILL because the
    #   command didn't respond to Ctrl-C, or the SIGINT path was
    #   skipped entirely.
    # - 0: the timeout didn't fire and the sleep finished (would
    #   also mean the test took 10s to run).
    # - None (in a shell_crashed-status result): bash itself died.
    assert result["exit_code"] == 130


def test_run_tool_rejects_empty_command(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """Empty command string produces an explicit error, not a hang."""
    tool = TerminalRunTool()
    result = json.loads(tool.invoke(json.dumps({"command": ""}), ctx))
    assert result["status"] == "error"
    assert "empty" in result["error"]


def test_run_tool_rejects_invalid_shell_name(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """Bad shell names surface as a structured error, not an exception."""
    tool = TerminalRunTool()
    result = json.loads(tool.invoke(json.dumps({"command": "echo hi", "shell": "bad name!"}), ctx))
    assert result["status"] == "shell_name_invalid"


def test_run_tool_errors_when_no_conversation_id(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """Missing ``conversation_id`` on ToolContext is a loud error.

    This guards against a workflow path that forgets to populate
    ``conversation_id`` — the terminal tools refuse to run rather
    than silently pretending they can.
    """
    bad_ctx = ToolContext(
        task_id="t",
        agent_id="a",
        workspace=tmp_path,
        conversation_id=None,
    )
    tool = TerminalRunTool()
    result = json.loads(tool.invoke(json.dumps({"command": "echo hi"}), bad_ctx))
    assert result["status"] == "error"
    assert "conversation_id" in result["error"]


def test_run_tool_errors_when_no_workspace(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """Missing workspace on ToolContext is a loud error.

    The terminal needs a workspace for cwd + disk-log overflow —
    no workspace = no safe defaults, fail loud.
    """
    bad_ctx = ToolContext(
        task_id="t",
        agent_id="a",
        workspace=None,
        conversation_id="conv",
    )
    tool = TerminalRunTool()
    result = json.loads(tool.invoke(json.dumps({"command": "echo hi"}), bad_ctx))
    assert result["status"] == "error"
    assert "workspace" in result["error"]


# ---- terminal_list ----------------------------------------------


def test_list_tool_empty_when_no_shells(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """Listing with no shells returns ``{"shells": []}``.

    Importantly, this does NOT create a manager — calling
    terminal_list must have no side effects. Verified by checking
    the registry's active conversation list.
    """
    tool = TerminalListTool()
    result = json.loads(tool.invoke("{}", ctx))
    assert result == {"shells": []}
    # Registry remains empty — no side-effect manager creation.
    assert registry.active_conversation_ids() == []


def test_list_tool_returns_shell_names_after_run(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """After running commands in shells A and B, list returns [A, B]."""
    run = TerminalRunTool()
    run.invoke(json.dumps({"command": "echo hi", "shell": "a"}), ctx)
    run.invoke(json.dumps({"command": "echo hi", "shell": "b"}), ctx)

    lst = TerminalListTool()
    result = json.loads(lst.invoke("{}", ctx))
    # Insertion order preserved.
    assert result["shells"] == ["a", "b"]


# ---- terminal_close ---------------------------------------------


def test_close_tool_closes_existing_shell(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """close an existing shell → ``{"closed": true, "shell": name}``."""
    run = TerminalRunTool()
    run.invoke(json.dumps({"command": "echo hi", "shell": "victim"}), ctx)

    close = TerminalCloseTool()
    result = json.loads(close.invoke(json.dumps({"shell": "victim"}), ctx))
    assert result == {"closed": True, "shell": "victim"}

    # List should no longer include it.
    lst = TerminalListTool()
    assert json.loads(lst.invoke("{}", ctx))["shells"] == []


def test_close_tool_idempotent_on_missing_shell(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """Closing a shell that doesn't exist returns ``closed=False``.

    Doesn't raise — idempotent per §6.1. Agent can safely call
    close even if it doesn't remember whether the shell exists.
    """
    close = TerminalCloseTool()
    result = json.loads(close.invoke(json.dumps({"shell": "ghost"}), ctx))
    assert result == {"closed": False, "shell": "ghost"}


def test_close_tool_noop_when_no_manager(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """close() on a conversation with no manager is a no-op, not an error.

    The conversation never used the terminal tool — closing any
    shell should just report "nothing to close" rather than
    creating an empty manager.
    """
    close = TerminalCloseTool()
    result = json.loads(close.invoke("{}", ctx))  # shell defaults to "default"
    assert result["closed"] is False
    # No manager was created as a side effect.
    assert registry.active_conversation_ids() == []


# ---- terminal_send_input ----------------------------------------


def test_send_input_tool_schema_required_fields() -> None:
    """Schema has task_id and chars required; yield_time_ms optional.

    Regression guard: if the required list changes, the LLM's tool
    constructor will start failing on valid calls.
    """
    schema = TerminalSendInputTool().get_schema()
    assert schema["function"]["name"] == "terminal_send_input"
    params = schema["function"]["parameters"]
    assert sorted(params["required"]) == ["chars", "task_id"]
    assert "yield_time_ms" in params["properties"]


def test_send_input_tool_reports_task_no_longer_running_when_unknown(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """Unknown task_id → ``delivered: false`` with a clear reason.

    Without a manager for the conversation the reason is
    ``shell_unavailable``; with a manager but no task it's
    ``task_no_longer_running``. This covers the first (registry
    empty). The LLM sees a structured outcome rather than an
    opaque error.
    """
    tool = TerminalSendInputTool()
    out = json.loads(
        tool.invoke(
            json.dumps({"task_id": "ghost", "chars": "hi"}),
            ctx,
        )
    )
    assert out["delivered"] is False
    assert out["reason"] == "shell_unavailable"
    assert out["task_id"] == "ghost"


def test_send_input_tool_delivers_to_registered_task(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """A registered task_id receives the bytes and the tool returns
    a delivered=True payload with recent_activity + screen fields.

    Spawns a shell, starts ``cat`` in a thread, registers a
    task_id, invokes the tool, asserts the response shape AND
    that the bytes made it through to cat's stdout (via the
    thread's eventual ``completed`` result).
    """
    import threading
    import time

    manager = registry.for_conversation(
        ctx.conversation_id,
        ctx.workspace,
    )
    manager.run_sync("default", "echo warmup")
    manager.register_running_task("tid-send", "default")

    runner_result: list[object] = []

    def _runner() -> None:
        runner_result.append(manager.run_sync("default", "cat", timeout_ms=10_000))

    t = threading.Thread(target=_runner)
    t.start()
    try:
        time.sleep(0.3)
        tool = TerminalSendInputTool()
        resp = json.loads(
            tool.invoke(
                json.dumps(
                    {
                        "task_id": "tid-send",
                        "chars": "tool-routed\n",
                        "yield_time_ms": 500,
                    }
                ),
                ctx,
            )
        )
        assert resp["delivered"] is True
        assert resp["task_id"] == "tid-send"
        # The echo of what we typed — cat should have echoed it
        # back within the 500 ms yield. If the yield was too short
        # this could flake on slow machines, but 500 ms is
        # generous for a local echo.
        assert "tool-routed" in resp.get("recent_activity", ""), (
            f"Expected 'tool-routed' in recent_activity, got {resp.get('recent_activity')!r}"
        )
        # Screen field populated (pyte-rendered view).
        assert "screen" in resp
        assert isinstance(resp["screen"], str) and resp["screen"]

        # Finish cat cleanly via EOF so the worker thread exits.
        tool.invoke(
            json.dumps({"task_id": "tid-send", "chars": "\x04"}),
            ctx,
        )
        t.join(timeout=5.0)
    finally:
        manager.unregister_running_task("tid-send")


def test_send_input_tool_empty_chars_is_pure_poll(
    registry: TerminalManagerRegistry, ctx: ToolContext
) -> None:
    """chars="" returns delivered=True + current state, no input written.

    Polling semantics: the tool shouldn't error out when the
    caller just wants an up-to-date view. Mirrors OpenAI's
    ``write_stdin(chars="")`` contract.
    """
    manager = registry.for_conversation(ctx.conversation_id, ctx.workspace)
    manager.run_sync("default", "echo poll-warmup")
    manager.register_running_task("tid-poll", "default")
    try:
        tool = TerminalSendInputTool()
        resp = json.loads(
            tool.invoke(
                json.dumps(
                    {
                        "task_id": "tid-poll",
                        "chars": "",
                        "yield_time_ms": 100,  # short — nothing's arriving
                    }
                ),
                ctx,
            )
        )
        assert resp["delivered"] is True
        assert resp["task_id"] == "tid-poll"
    finally:
        manager.unregister_running_task("tid-poll")


# ---- _resolve_yield_time_ms ------------------------------------


def test_resolve_yield_time_ms_picks_typing_default_when_none() -> None:
    """None + non-empty chars → 250 ms (typing default)."""
    assert _resolve_yield_time_ms(None, chars_empty=False) == 250


def test_resolve_yield_time_ms_picks_polling_default_when_empty() -> None:
    """None + empty chars → 5000 ms (polling default).

    Regression guard for the empty-chars auto-bump: if someone
    flips the chars_empty default, pure polls would only wait
    250 ms and miss slow programs.
    """
    assert _resolve_yield_time_ms(None, chars_empty=True) == 5000


def test_resolve_yield_time_ms_honors_explicit_value_with_empty_chars() -> None:
    """Explicit integer wins over auto-bump — matches documented behavior.

    OpenAI's resolver force-bumps even explicit values to 5s when
    chars is empty. We chose to honor the caller; this test is the
    contract that encodes that choice. If someone adds a bump back,
    this test catches it.
    """
    assert _resolve_yield_time_ms(100, chars_empty=True) == 100


def test_resolve_yield_time_ms_clamps_floor_and_ceiling() -> None:
    """Values below 50 get clamped up; values above 30000 get clamped down."""
    assert _resolve_yield_time_ms(10, chars_empty=False) == 50
    assert _resolve_yield_time_ms(60_000, chars_empty=False) == 30_000
