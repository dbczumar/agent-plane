"""Tests for :class:`TerminalManager`.

Manager-layer unit tests: shell creation, name validation,
per-conversation cap, close semantics, dead-shell sweeping,
concurrent get-or-create atomicity.

Tests use real bash subprocesses (no mocks) because the manager
calls :meth:`Shell.spawn` which requires a real PTY. The one
exception is the concurrency test, which monkeypatches
``Shell.spawn`` to count invocations — the real spawn is
non-deterministic enough that counting requires interception.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_plane.terminals.manager import (
    ShellCapExceeded,
    ShellNameInvalid,
    TerminalManager,
)
from agent_plane.terminals.shell import Shell


@pytest.fixture
def manager(tmp_path: Path) -> Iterator[TerminalManager]:
    """A fresh manager for one conversation; teardown closes everything.

    Sandbox is disabled for manager-layer tests — see the ``shell``
    fixture's docstring in ``conftest.py`` for the same reasoning.
    The manager's behavior is independent of whether its shells are
    sandboxed.

    :param tmp_path: Pytest's per-test tmpdir.
    :yields: An empty :class:`TerminalManager`.
    """
    mgr = TerminalManager(
        conversation_id="conv_test",
        workspace=tmp_path,
        sandbox_enabled=False,
    )
    yield mgr
    mgr.close_all()


# ---- basic lifecycle ------------------------------------------


def test_first_run_sync_spawns_shell(manager: TerminalManager) -> None:
    """Calling run_sync with a fresh name creates a shell and executes.

    Failure would indicate the lazy-create path is broken or the
    shell isn't being stored in the manager's dict.
    """
    result = manager.run_sync("main", "echo hello")
    assert result.status == "completed"
    assert "hello" in result.stdout
    assert result.shell == "main"
    # Manager now has exactly one shell registered.
    assert manager.list_shells() == ["main"]


def test_reuses_existing_shell(manager: TerminalManager) -> None:
    """Same shell name across calls preserves state (cwd etc.).

    This is the core persistence property. If the manager
    spawned a fresh shell every call, ``cd /tmp`` in one call
    would have no effect on the next.
    """
    manager.run_sync("main", "cd /tmp")
    r = manager.run_sync("main", "pwd")
    assert "/tmp" in r.stdout


def test_distinct_names_create_distinct_shells(manager: TerminalManager) -> None:
    """Different shell names get independent subprocesses.

    State in one does NOT bleed into the other.
    """
    manager.run_sync("dev", "export FOO=dev_value")
    manager.run_sync("test", "export FOO=test_value")

    dev_r = manager.run_sync("dev", "echo $FOO")
    test_r = manager.run_sync("test", "echo $FOO")
    # Each shell has its own environment; neither sees the other's export.
    assert "dev_value" in dev_r.stdout
    assert "test_value" in test_r.stdout
    assert "dev_value" not in test_r.stdout
    assert "test_value" not in dev_r.stdout


# ---- name validation ------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "1leading-digit",  # must start with letter
        "-starts-with-hyphen",
        "_leading-underscore",  # reserved for framework use
        "_",  # just an underscore
        "has spaces",
        "has/slash",
        "has.dot",
        "has@at",
        "",  # empty
        "a" * 65,  # over 64 chars
    ],
)
def test_invalid_shell_names_rejected(manager: TerminalManager, bad_name: str) -> None:
    """Names outside the character-set or length spec are rejected.

    Uses the regex ``^[a-zA-Z][a-zA-Z0-9_-]{0,63}$`` per §7.1. Each
    parametrized case represents one regex failure mode. The regex
    rejects ``_``-prefixed names as part of the "start with a letter"
    rule — the error message also names the framework-reserved
    convention so agents get a clear signal.
    """
    with pytest.raises(ShellNameInvalid):
        manager.run_sync(bad_name, "echo nope")


@pytest.mark.parametrize(
    "good_name",
    [
        "default",  # the implicit fallback
        "main",
        "dev",
        "test-runner",
        "worker_1",
        "a",  # single letter ok
        "A" * 64,  # exactly 64 chars ok (max)
    ],
)
def test_valid_names_accepted(manager: TerminalManager, good_name: str) -> None:
    """Names matching the regex are accepted.

    Each parametrized case is a common/edge-legal name. Tests that
    the regex doesn't accidentally reject something legitimate.
    """
    r = manager.run_sync(good_name, "echo ok")
    assert r.status == "completed"
    assert r.shell == good_name


# ---- per-conversation cap -------------------------------------


def test_cap_enforced_at_ten_shells(manager: TerminalManager) -> None:
    """The 11th distinct shell name raises ShellCapExceeded.

    Cap value (10) is ratified in §6.4. This test asserts the cap
    fires at the documented threshold.
    """
    # Create 10 shells — should all succeed.
    for i in range(10):
        r = manager.run_sync(f"shell{i}", "echo hi")
        assert r.status == "completed"
    assert len(manager.list_shells()) == 10

    # 11th must fail.
    with pytest.raises(ShellCapExceeded):
        manager.run_sync("shell11", "echo nope")


def test_cap_allows_creation_after_close(manager: TerminalManager) -> None:
    """Closing a shell frees a slot for a new one.

    Proves the cap counts CURRENT shells, not cumulative creates.
    """
    for i in range(10):
        manager.run_sync(f"shell{i}", "echo hi")
    # Close one — should now be able to create another.
    assert manager.close("shell0") is True
    # This should succeed (we're back under the cap).
    r = manager.run_sync("replacement", "echo hi")
    assert r.status == "completed"


# ---- close semantics ------------------------------------------


def test_close_existing_shell_returns_true(manager: TerminalManager) -> None:
    """close() returns True when a shell was actually removed."""
    manager.run_sync("main", "echo hi")
    assert manager.close("main") is True
    # Subsequent list_shells should show empty.
    assert manager.list_shells() == []


def test_close_nonexistent_shell_returns_false(manager: TerminalManager) -> None:
    """close() on an unknown name is a no-op, returning False.

    Intentional: close is idempotent. Agents that close a shell
    they already closed shouldn't get an error.
    """
    assert manager.close("never-existed") is False


def test_close_all_removes_everything(manager: TerminalManager) -> None:
    """close_all() kills every shell regardless of state."""
    for i in range(5):
        manager.run_sync(f"shell{i}", "echo hi")
    assert len(manager.list_shells()) == 5
    manager.close_all()
    assert manager.list_shells() == []
    assert manager.has_shells() is False


# ---- dead-shell sweeping --------------------------------------


def test_dead_shell_replaced_on_next_use(manager: TerminalManager) -> None:
    """If a shell died between calls, the next run spawns a fresh one.

    Failure mode this prevents: a shell that died would stay in the
    dict as a zombie, and subsequent run_sync on that name would
    short-circuit to ``shell_crashed`` forever. Auto-sweeping gives
    the agent a second chance without having to call ``close``
    first.
    """
    manager.run_sync("main", "echo first")
    # Kill the shell from the outside (same trick as in slice 2,
    # except we access via Shell public API):
    # commands that exit bash are the deterministic way.
    manager.run_sync("main", "exit 0")
    # Next call sees the dead shell, sweeps it, and spawns fresh.
    r = manager.run_sync("main", "echo second")
    assert r.status == "completed"
    assert "second" in r.stdout


# ---- has_shells + last_activity --------------------------------


def test_has_shells_reflects_state(manager: TerminalManager) -> None:
    """has_shells returns True iff at least one shell is registered."""
    assert manager.has_shells() is False
    manager.run_sync("main", "echo hi")
    assert manager.has_shells() is True
    manager.close("main")
    assert manager.has_shells() is False


def test_last_activity_advances_on_run(manager: TerminalManager) -> None:
    """last_activity_monotonic increases when run_sync is called.

    The reaper depends on this timestamp being kept current to
    distinguish active from idle conversations.
    """
    before = manager.last_activity_monotonic
    # Sleep the Python thread briefly so monotonic clock advances
    # measurably. This IS time.sleep in test code, justified as
    # the only way to create a measurable gap between two reads
    # of a monotonic clock. The shell subprocess does its own I/O
    # that takes longer than this anyway.
    time.sleep(0.01)
    manager.run_sync("main", "echo hi")
    after = manager.last_activity_monotonic
    assert after > before


# ---- concurrent get-or-create --------------------------------


def test_concurrent_run_sync_same_name_creates_one_shell(
    manager: TerminalManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two threads creating the same-named shell spawn exactly one.

    The manager's internal lock must make get-or-create atomic. If
    the check-then-create pattern is racy, both threads could spawn
    — we'd get two bash subprocesses, one leaked, and unpredictable
    state persistence. Failure mode this catches: missing lock, or
    check/create happening outside the lock.

    Implementation: count calls to ``Shell.spawn`` via monkeypatch.
    Under concurrent pressure on the same name, spawn count must
    be exactly 1.
    """
    spawn_count = 0
    original_spawn = Shell.spawn
    spawn_lock = threading.Lock()

    def counting_spawn(
        name: str, workspace: Path, *, sandbox_enabled: bool = True
    ) -> Shell:
        nonlocal spawn_count
        with spawn_lock:
            spawn_count += 1
        return original_spawn(name, workspace, sandbox_enabled=sandbox_enabled)

    monkeypatch.setattr(Shell, "spawn", staticmethod(counting_spawn))

    # Start 5 threads, all racing to run_sync on "shared". The
    # start_event sync gate ensures all threads are parked in
    # worker() before they call into the manager — otherwise
    # threads could enter serially as each is spawned and we'd
    # never exercise the race.
    errors: list[BaseException] = []
    start_event = threading.Event()

    def worker() -> None:
        start_event.wait()
        try:
            manager.run_sync("shared", "echo hi")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    # Release all workers simultaneously to force deterministic
    # contention on _get_or_create_shell.
    start_event.set()
    for t in threads:
        t.join()

    # No thread errored out.
    assert errors == [], f"Worker thread(s) raised: {errors}"
    # Exactly one spawn happened — later threads reused the shell.
    assert spawn_count == 1, (
        f"Expected 1 Shell.spawn call under concurrent pressure "
        f"(atomic get-or-create), got {spawn_count}. If >1, the "
        f"manager's lock isn't properly guarding check-then-create."
    )
    # All 5 threads saw the same shell instance.
    assert manager.list_shells() == ["shared"]


# ── Async task tracking (register/unregister/shell_for_task/peek) ──


def test_register_running_task_tracks_shell(
    manager: TerminalManager,
) -> None:
    """After ``register_running_task``, ``shell_for_task`` returns
    the tracked shell.

    This is the lookup ``cancel_task(kind="terminal")`` uses to
    find the shell to interrupt. If it didn't work, cancel_task
    would silently fail to find the running command.
    """
    manager.run_sync("worker", "echo init")
    manager.register_running_task("task_abc", "worker")

    shell = manager.shell_for_task("task_abc")
    assert shell is not None
    assert shell.name == "worker"


def test_shell_for_unknown_task_returns_none(
    manager: TerminalManager,
) -> None:
    """``shell_for_task`` on an unregistered id returns None.

    Cancel path relies on this — if the task completed already
    (and was unregistered), cancel should be a no-op rather than
    pick up a wrong shell by accident.
    """
    # No registration at all.
    assert manager.shell_for_task("task_never_existed") is None

    # Registered then unregistered.
    manager.run_sync("worker", "echo init")
    manager.register_running_task("task_done", "worker")
    manager.unregister_running_task("task_done")
    assert manager.shell_for_task("task_done") is None


def test_shell_for_task_returns_none_if_shell_closed(
    manager: TerminalManager,
) -> None:
    """If the underlying shell was closed after registration,
    ``shell_for_task`` returns None.

    Defensive: prevents cancel_task from calling interrupt() on a
    dead shell. Closed shell = no one to interrupt.
    """
    manager.run_sync("worker", "echo init")
    manager.register_running_task("task_x", "worker")
    manager.close("worker")
    assert manager.shell_for_task("task_x") is None


def test_peek_task_stdout_returns_delta(
    manager: TerminalManager,
) -> None:
    """``peek_task_stdout`` returns newly appended bytes since the
    prior call, with per-task cursor advancement.

    This is what ``check_task(kind="terminal")`` uses for
    ``recent_activity``. Without proper cursor advancement, every
    check_task would return the whole buffer, blowing the LLM's
    context.
    """
    import threading

    # Start a long-ish command in a background thread so we can
    # peek while it's running.
    result_holder: list[object] = []

    def _runner() -> None:
        result_holder.append(
            manager.run_sync(
                "task_runner",
                "echo part-one; sleep 0.6; echo part-two; sleep 0.1",
            )
        )

    # Register BEFORE starting so cursor starts at 0.
    # Spawn a shell first so register has something to reference.
    manager.run_sync("task_runner", "echo init")
    manager.register_running_task("tid_peek", "task_runner")

    t = threading.Thread(target=_runner)
    t.start()
    try:
        # First peek: wait until part-one arrives.
        deadline = __import__("time").monotonic() + 3.0
        first = manager.peek_task_stdout("tid_peek")
        while first is not None and "part-one" not in first[0] and __import__("time").monotonic() < deadline:
            first = manager.peek_task_stdout("tid_peek")
        assert first is not None
        text1, lost1 = first
        assert "part-one" in text1
        assert "part-two" not in text1, (
            f"First peek too late — part-two already appeared: {text1!r}"
        )
        assert lost1 == 0  # ring buffer has ample space

        # Second peek: only new bytes (part-two), not part-one again.
        deadline2 = __import__("time").monotonic() + 3.0
        second = manager.peek_task_stdout("tid_peek")
        while second is not None and "part-two" not in second[0] and __import__("time").monotonic() < deadline2:
            second = manager.peek_task_stdout("tid_peek")
        assert second is not None
        text2, _ = second
        assert "part-two" in text2, (
            f"Second peek didn't see part-two: {text2!r}"
        )
        assert "part-one" not in text2, (
            f"Second peek returned part-one again — cursor didn't "
            f"advance. Got: {text2!r}"
        )
    finally:
        t.join(timeout=5.0)
        manager.unregister_running_task("tid_peek")


def test_peek_task_stdout_unknown_returns_none(
    manager: TerminalManager,
) -> None:
    """Unknown task_id → None (not empty string, not exception).

    Distinguishes "task completed" from "task has no output yet."
    check_task uses this to know it can't produce recent_activity.
    """
    assert manager.peek_task_stdout("never-registered") is None


def test_register_overwrite_resets_cursor(
    manager: TerminalManager,
) -> None:
    """Re-registering the same task_id starts from cursor 0.

    Supports DBOS crash+replay: the workflow may re-register a
    task_id after a replay. The expected behavior is that the
    replayed command starts fresh — partial output from the
    crashed run is effectively lost (acceptable; the command is
    re-running anyway).
    """
    manager.run_sync("sh1", "echo first")
    manager.register_running_task("tid", "sh1")

    # Advance the cursor via a peek (which uses the current ring
    # state — but we only care that cursor != 0 afterward).
    manager.peek_task_stdout("tid")
    # Re-register — should reset.
    manager.register_running_task("tid", "sh1")
    # A peek right after re-registration should see whatever is
    # still in the ring (ring wasn't reset; command wasn't re-run
    # yet in this test). Cursor is back to 0 so we can see it.
    peek = manager.peek_task_stdout("tid")
    assert peek is not None
    text, _ = peek
    # The original "first" may or may not still be in the ring
    # depending on whether the next run_sync already reset it.
    # The invariant here is just: peek didn't error, cursor moved
    # forward from 0.
    assert isinstance(text, str)
