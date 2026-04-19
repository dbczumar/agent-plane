"""Tests for the task-lifecycle builtins (check_task, cancel_task, list_tasks).

Pure-helper coverage is here. The full ``invoke()`` paths require
a running task_store + DBOS runtime and are covered in the
server-integration suite.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_plane.entities.task import Task
from agent_plane.terminals.registry import TerminalManagerRegistry
from agent_plane.tools.builtins.task_lifecycle import (
    _ACTIVITY_MAX_CHARS,
    CancelTaskTool,
    CheckTaskTool,
    ListTasksTool,
    _build_check_payload,
    _get_recent_terminal_activity,
    _truncate_content_field,
)

# ─── truncation helper ───────────────────────────────────────


def test_truncate_string_under_cap_returns_unchanged() -> None:
    """Strings under the cap pass through verbatim."""
    text = "short"
    assert _truncate_content_field(text) == text


def test_truncate_string_over_cap_appends_marker() -> None:
    """Over-cap strings get truncated with a ``[truncated]`` suffix."""
    text = "x" * (_ACTIVITY_MAX_CHARS + 100)
    result = _truncate_content_field(text)
    # Length must be cap + suffix length.
    assert result.startswith("x" * _ACTIVITY_MAX_CHARS)
    assert result.endswith(" [truncated]")
    # If the suffix is missing, the LLM has no signal that data
    # was dropped.


def test_truncate_walks_dict_values() -> None:
    """Recursion into dict values truncates nested strings."""
    item = {
        "type": "message",
        "content": [{"type": "input_text", "text": "y" * (_ACTIVITY_MAX_CHARS + 50)}],
        "role": "assistant",
    }
    result = _truncate_content_field(item)
    inner_text = result["content"][0]["text"]
    assert inner_text.endswith(" [truncated]")
    # Other fields must remain unchanged — a bug here would mutate
    # the role/type strings (both well under the cap, but the
    # invariant matters).
    assert result["role"] == "assistant"
    assert result["type"] == "message"


def test_truncate_passes_through_non_strings() -> None:
    """Numeric and bool fields aren't touched."""
    item = {"position": 42, "active": True, "ratio": 1.5}
    assert _truncate_content_field(item) == item


# ─── _build_check_payload ────────────────────────────────────


def _make_task(
    *,
    kind: str = "tool",
    status: str = "completed",
    output: list | None = None,
    error: dict[str, str] | None = None,
) -> Task:
    """Build a Task with the fields :func:`_build_check_payload` reads."""
    return Task(
        id="tsk_test",
        conversation_id="conv_test",
        agent_id="ag_test",
        agent_name="test-agent",
        created_at=1000,
        completed_at=2000 if status in {"completed", "failed", "cancelled"} else None,
        kind=kind,
        status=status,
        output=output if output is not None else [],
        error=error,
    )


def test_check_payload_for_completed_tool_includes_result_and_no_activity() -> None:
    """A completed tool task surfaces its output as ``result`` (no activity)."""
    task = _make_task(
        kind="tool",
        status="completed",
        output=[{"text": "hello"}],
    )
    payload = _build_check_payload(task)
    assert payload["task_id"] == "tsk_test"
    assert payload["kind"] == "tool"
    assert payload["status"] == "completed"
    assert payload["result"] == [{"text": "hello"}]
    # tools have no per-step recent_activity by design (G50).
    assert "recent_activity" not in payload


def test_check_payload_for_failed_task_includes_error() -> None:
    """A failed task surfaces both result and error."""
    task = _make_task(
        kind="tool",
        status="failed",
        output=[],
        error={"message": "ValueError: oops", "traceback": "stack..."},
    )
    payload = _build_check_payload(task)
    assert payload["status"] == "failed"
    assert payload["error"] == {"message": "ValueError: oops", "traceback": "stack..."}


def test_check_payload_uses_completed_at_as_updated_when_terminal() -> None:
    """Terminal tasks expose the completed timestamp via ``updated_at``."""
    task = _make_task(status="completed")
    payload = _build_check_payload(task)
    assert payload["created_at"] == 1000
    # If the test task were missing completed_at, this would be the
    # created_at instead — so the assertion proves the conditional
    # in _build_check_payload picked the right field.
    assert payload["updated_at"] == 2000


def test_check_payload_for_running_tool_omits_recent_activity() -> None:
    """A running TOOL task has no recent_activity (only sub-agents do)."""
    task = _make_task(kind="tool", status="in_progress")
    payload = _build_check_payload(task)
    assert payload["status"] == "in_progress"
    assert "recent_activity" not in payload


# ─── tool schemas ────────────────────────────────────────────


def test_check_task_schema_shape() -> None:
    """Schema is OpenAI function-format with required task_id."""
    schema = CheckTaskTool().get_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "check_task"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "task_id" in params["properties"]
    assert params["required"] == ["task_id"]


def test_cancel_task_schema_shape() -> None:
    """cancel_task requires task_id."""
    schema = CancelTaskTool().get_schema()
    assert schema["function"]["name"] == "cancel_task"
    assert schema["function"]["parameters"]["required"] == ["task_id"]


def test_list_tasks_schema_default_filter() -> None:
    """list_tasks has an optional ``filter`` enum with the documented values."""
    schema = ListTasksTool().get_schema()
    assert schema["function"]["name"] == "list_tasks"
    params = schema["function"]["parameters"]
    # filter is optional and defaults to "running".
    assert params["required"] == []
    filter_prop = params["properties"]["filter"]
    assert sorted(filter_prop["enum"]) == sorted(["running", "completed", "all"])
    assert filter_prop["default"] == "running"


def test_check_task_description_mentions_handle_origin() -> None:
    """The LLM-facing description explains where task_id comes from."""
    desc = CheckTaskTool.description().lower()
    # Must mention the handle / spawn origin so the LLM knows to
    # pass the task_id from a previous async tool call.
    assert "task_id" in desc
    assert "spawn" in desc or "asynchronous" in desc


def test_cancel_task_description_mentions_non_blocking() -> None:
    """The LLM-facing description sets expectations about timing."""
    desc = CancelTaskTool.description().lower()
    assert "non-blocking" in desc or "background" in desc


# ─── Terminal-kind helpers ───────────────────────────────────
#
# These tests exercise the terminal-specific branches of
# :func:`_build_check_payload` and :func:`_get_recent_terminal_activity`
# using a real :class:`TerminalManager` + real bash subprocess. The
# alternative (mocking the manager) would silently mask breakage
# in the manager ↔ helper wiring.


_TERMINAL_CONV_ID = "conv_term_test"


def _make_terminal_task(
    task_id: str,
    *,
    status: str = "in_progress",
    conversation_id: str = _TERMINAL_CONV_ID,
) -> Task:
    """Build a terminal-kind Task with the fields the check helpers read.

    :param task_id: The task id, e.g. ``"tsk_term1"``.
    :param status: The DB status. Defaults to ``"in_progress"``
        which is the only status where ``recent_activity`` is
        populated for terminal tasks.
    :param conversation_id: Conversation the task belongs to — must
        match the registry's active conversation in tests so the
        registry lookup in :func:`_get_recent_terminal_activity`
        succeeds.
    :returns: A :class:`Task` the helpers can consume.
    """
    return Task(
        id=task_id,
        conversation_id=conversation_id,
        agent_id="ag_term",
        agent_name="terminal-test-agent",
        created_at=1000,
        completed_at=None if status == "in_progress" else 2000,
        kind="terminal",
        status=status,
        output=[],
        error=None,
    )


@pytest.fixture
def terminal_registry_with_running_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[tuple[TerminalManagerRegistry, str]]:
    """Registry + manager with a task registered + actively emitting stdout.

    Spawns a real bash via ``manager.run_sync`` (warms the shell),
    then kicks off a second command in a background thread that
    emits a known line and sleeps, giving the test time to call
    peek / check_task while the task is still running.
    Monkeypatches ``get_terminal_registry`` so helpers that import
    it lazily see the test registry.

    :param monkeypatch: Pytest's monkeypatch fixture.
    :param tmp_path: Pytest's per-test tmpdir (the shell's
        workspace).
    :yields: ``(registry, task_id)`` — the test uses task_id to
        call helpers; the registry is yielded only so the test can
        assert on it if needed.
    """
    registry = TerminalManagerRegistry(sandbox_enabled=False)
    manager = registry.for_conversation(_TERMINAL_CONV_ID, tmp_path)

    # Warm the shell — `_get_or_create_shell` inside run_sync needs
    # a command, so we do a harmless echo first.
    manager.run_sync("default", "echo warmup")

    task_id = "tsk_term_peek"
    manager.register_running_task(task_id, "default")

    # Fire a command that emits an expected marker, then sleeps
    # long enough for the test to do its peek. Run it in a daemon
    # thread so we don't block the test; the `finally` join handles
    # cleanup deterministically.
    def _runner() -> None:
        manager.run_sync("default", "echo peek-marker; sleep 0.8")

    runner_thread = threading.Thread(target=_runner, daemon=True)
    runner_thread.start()

    # Point the helper at our registry instead of the global one.
    from agent_plane import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "get_terminal_registry", lambda: registry)

    try:
        yield registry, task_id
    finally:
        runner_thread.join(timeout=5.0)
        manager.unregister_running_task(task_id)
        manager.close_all()


def test_get_recent_terminal_activity_returns_none_for_non_terminal_kind() -> None:
    """Non-terminal tasks short-circuit — no registry lookup at all.

    If this returned something non-None for a tool-kind task, the
    check_task payload would gain a ``recent_activity`` field that
    the LLM isn't expecting for that kind, and every terminal helper
    would be invoked on every non-terminal check.
    """
    tool_task = Task(
        id="tsk_tool",
        conversation_id="conv_any",
        agent_id="ag_any",
        agent_name="any",
        created_at=1000,
        completed_at=None,
        kind="tool",
        status="in_progress",
        output=[],
        error=None,
    )
    assert _get_recent_terminal_activity(tool_task) is None


def test_get_recent_terminal_activity_returns_none_when_conversation_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task whose conversation has no registered manager → None.

    This is the "manager was reaped" case from the helper docstring.
    A None return keeps check_task responses well-formed (the field
    is just omitted) instead of crashing on registry.for_conversation
    with a made-up workspace.
    """
    empty_registry = TerminalManagerRegistry(sandbox_enabled=False)
    from agent_plane import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "get_terminal_registry", lambda: empty_registry)
    task = _make_terminal_task("tsk_missing")
    assert _get_recent_terminal_activity(task) is None


def test_get_recent_terminal_activity_returns_stdout_for_running_task(
    terminal_registry_with_running_task: tuple[TerminalManagerRegistry, str],
) -> None:
    """A running terminal task with buffered stdout surfaces that text.

    Fails if the registry→manager→peek chain is broken anywhere
    (wrong conversation lookup, shell lookup mismatch, cursor
    initialization).
    """
    _registry, task_id = terminal_registry_with_running_task

    # Poll up to ~3s for the "peek-marker" line to land in the ring
    # buffer. The test's runner thread emits it and sleeps 0.8s, so
    # on any reasonable machine the marker is present well inside
    # this window. Timing out here points to the echo not being
    # readable through the helper, not slow hardware.
    task = _make_terminal_task(task_id)
    deadline = time.monotonic() + 3.0
    activity: str | None = None
    while time.monotonic() < deadline:
        activity = _get_recent_terminal_activity(task)
        if activity is not None and "peek-marker" in activity:
            break
    assert activity is not None, (
        "Helper returned None for a registered running task — the "
        "registry/manager/shell lookup chain is broken."
    )
    assert "peek-marker" in activity, (
        f"Helper returned text but it doesn't contain 'peek-marker'. "
        f"Got {activity!r}. Likely means the peek cursor advanced "
        f"past the line or the shell's stdout wasn't captured."
    )


def test_build_check_payload_for_running_terminal_task_includes_recent_activity(
    terminal_registry_with_running_task: tuple[TerminalManagerRegistry, str],
) -> None:
    """End-to-end: _build_check_payload populates ``recent_activity``
    for a running terminal task.

    This is the behavior check_task exposes to the LLM. Without it,
    the LLM sees a status=in_progress response with no stdout — the
    "tail -f" story in §6.11 is broken.
    """
    _registry, task_id = terminal_registry_with_running_task

    task = _make_terminal_task(task_id, status="in_progress")

    # Poll because the runner thread emits asynchronously; a single
    # call immediately after start might see an empty buffer.
    deadline = time.monotonic() + 3.0
    payload: dict[str, object] = {}
    while time.monotonic() < deadline:
        payload = _build_check_payload(task)
        activity = payload.get("recent_activity")
        if isinstance(activity, str) and "peek-marker" in activity:
            break
    assert payload.get("status") == "in_progress"
    assert payload.get("kind") == "terminal"
    # `result` is for terminal-status tasks only; a running task
    # must not carry a result field (see _build_check_payload).
    assert "result" not in payload, (
        f"Running task should not have 'result' populated yet; payload={payload!r}"
    )
    activity = payload.get("recent_activity")
    assert isinstance(activity, str) and "peek-marker" in activity, (
        f"Expected 'peek-marker' in recent_activity, got "
        f"{activity!r}. If recent_activity is missing entirely, the "
        f"terminal-kind branch of _build_check_payload didn't fire."
    )


def test_build_check_payload_for_completed_terminal_task_has_result_no_activity(
    terminal_registry_with_running_task: tuple[TerminalManagerRegistry, str],
) -> None:
    """A completed terminal task carries its output as ``result`` and
    does NOT attempt to peek stdout (the shell may have moved on).

    Regression guard: the ``is_terminal`` branch must gate before the
    kind==\"terminal\" recent_activity branch, otherwise completed
    tasks would trigger an unnecessary peek_task_stdout call — which
    is harmless but a waste, and masks the intent of the code.
    """
    _registry, _ = terminal_registry_with_running_task

    completed = Task(
        id="tsk_term_done",
        conversation_id=_TERMINAL_CONV_ID,
        agent_id="ag_term",
        agent_name="terminal-test-agent",
        created_at=1000,
        completed_at=2000,
        kind="terminal",
        status="completed",
        output=[{"kind": "stdout", "text": "final stdout"}],
        error=None,
    )

    payload = _build_check_payload(completed)
    assert payload["status"] == "completed"
    assert payload["kind"] == "terminal"
    assert payload["result"] == [{"kind": "stdout", "text": "final stdout"}]
    # Completed terminal tasks must NOT populate recent_activity —
    # the result field carries the final output.
    assert "recent_activity" not in payload, (
        f"Completed task should not include recent_activity; payload={payload!r}"
    )
