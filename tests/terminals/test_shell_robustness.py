"""Robustness tests for Shell — slice 2 additions.

Covers the new failure modes and output-management features added
in slice 2: timeout_ms, shell crashes, ring-buffer eviction,
ANSI stripping on output, 30 KB inline cap with head+tail
truncation, disk persistence on overflow.

Happy-path tests live in test_shell.py (slice 1 coverage). This
file tests the edges.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_plane.terminals import Shell

# The ``shell`` fixture lives in tests/terminals/conftest.py — shared
# with test_shell.py.


# ---- timeout_ms --------------------------------------------------


def test_timeout_kills_running_command(shell: Shell) -> None:
    """A command exceeding timeout_ms is Ctrl-C'd; shell stays usable.

    The Ctrl-C path delivers a real bash exit code (130 = 128+SIGINT)
    rather than losing the exit status, and status is "killed" so
    the agent can distinguish abort-by-timeout from normal exit.
    """
    # Sleep for 10s; timeout at 500ms. Grace period is 2s so total
    # worst-case runtime ~2.5s.
    result = shell.run_sync("sleep 10", timeout_ms=500)
    assert result.status == "killed"
    # SIGINT killed the sleep → bash reports 128 + 2 = 130.
    assert result.exit_code == 130
    assert result.shell == "default"


def test_shell_survives_timeout_kill(shell: Shell) -> None:
    """After a timeout-kill, the shell is still alive and usable.

    This is THE key property distinguishing "kill the command" (SIGINT,
    shell survives) from "kill the shell" (SIGKILL, shell is gone).
    If this fails, a single timeout effectively destroys the shell.
    """
    shell.run_sync("sleep 10", timeout_ms=300)
    # Immediately follow with a simple command to verify the shell
    # still responds.
    result = shell.run_sync("echo alive")
    assert result.status == "completed"
    assert "alive" in result.stdout
    assert result.exit_code == 0


def test_no_timeout_allows_long_command(shell: Shell) -> None:
    """timeout_ms=None means no bound; short sleep still completes.

    We don't actually test "runs forever" (that's impractical) — we
    confirm that without a timeout, a slow-ish command completes
    rather than being cut off prematurely.
    """
    result = shell.run_sync("sleep 0.5; echo done", timeout_ms=None)
    assert result.status == "completed"
    assert "done" in result.stdout


def test_timeout_not_triggered_when_command_finishes_early(shell: Shell) -> None:
    """A fast command under the timeout returns "completed" normally.

    Guards against false-positive timeouts (firing when not needed).
    """
    result = shell.run_sync("echo fast", timeout_ms=60_000)
    assert result.status == "completed"
    assert result.exit_code == 0
    assert "fast" in result.stdout


# ---- ANSI stripping ---------------------------------------------


def test_ansi_colors_stripped_from_stdout(shell: Shell) -> None:
    """Colored output surfaces as plain text.

    ``echo -e`` interprets escape codes. We verify the escape is
    PRODUCED by bash (not preserved literally) and then STRIPPED
    on our read path.
    """
    result = shell.run_sync("echo -e '\\033[31mRED\\033[0m'")
    assert result.status == "completed"
    # Visible text preserved.
    assert "RED" in result.stdout
    # Escape bytes stripped.
    assert "\x1b" not in result.stdout


# ---- shell_crashed ----------------------------------------------


def test_exit_kills_the_shell(shell: Shell) -> None:
    """If the command exits the shell, subsequent calls see shell_crashed.

    ``exit 7`` from within the shell terminates bash itself. The next
    run_sync should detect the dead shell and return shell_crashed
    immediately without trying to send a command.
    """
    # First command exits the shell. We don't assert its status
    # because it's racy: whether bash emits the D marker before the
    # PTY tears down is timing-dependent (we may see either
    # "completed" with exit_code=7, or "shell_crashed" with no exit
    # code). The test's real assertion is on the SECOND call, which
    # is deterministic: by then is_alive() is definitely false.
    shell.run_sync("exit 7")

    # Subsequent call MUST see shell_crashed (shell is dead).
    result = shell.run_sync("echo nope")
    assert result.status == "shell_crashed"
    assert result.exit_code is None


# ---- large output: eviction + truncation + disk persistence ----


def test_small_output_stays_in_buffer_no_disk(shell: Shell, tmp_path: Path) -> None:
    """Normal-sized output doesn't spill to disk.

    Explicit coverage that small commands never create log files —
    disk persistence is "overflow only" per §6.7 Layer 3.
    """
    result = shell.run_sync("echo small")
    assert result.status == "completed"
    terminal_dir = tmp_path / ".agent_plane" / "terminal"
    # Directory shouldn't even be created for small commands.
    assert not terminal_dir.exists()


def test_large_output_truncates_and_persists(shell: Shell, tmp_path: Path) -> None:
    """Over-30KB output is head+tail truncated and full output is on disk.

    Produces ~100 KB of predictable text via a yes-like loop. We
    verify:
    1. Returned stdout is at most ~30 KB (head + marker + tail fits).
    2. The truncation marker appears in the middle.
    3. A disk file exists with the full output.
    """
    # Produce 100 KB of "x\n" lines — ~50k lines.
    # Using a single printf keeps this under a second.
    result = shell.run_sync(
        'python3 -c \'print("x" * 50 + "\\n" * 1, end="")\' '
        "| python3 -c 'import sys; sys.stdout.write(sys.stdin.read() * 1000)'"
    )
    assert result.status == "completed"
    assert result.exit_code == 0
    # Inline output is bounded: head(10k) + marker(~100) + tail(10k) = ~20k.
    # Give some slack for eviction markers and the specific marker string.
    assert len(result.stdout) < 25_000
    assert "truncated" in result.stdout
    assert ".agent_plane/terminal/" in result.stdout

    # Disk file should exist.
    terminal_dir = tmp_path / ".agent_plane" / "terminal"
    assert terminal_dir.is_dir()
    log_files = list(terminal_dir.glob("default-*.log"))
    # Exactly one file for this one overflow run.
    assert len(log_files) == 1
    # Full output is meaningfully larger than the inline cap.
    content = log_files[0].read_text()
    assert len(content) > 30_000


def test_truncation_marker_includes_disk_path(shell: Shell, tmp_path: Path) -> None:
    """The truncation marker references the disk log file by relative path.

    Agent can follow the path with Read or ``cat`` to recover the
    full output.
    """
    result = shell.run_sync("python3 -c 'import sys; sys.stdout.write(\"a\" * 50000)'")
    assert result.status == "completed"
    # Marker names the file so the agent can open it.
    assert "default-1.log" in result.stdout


def test_run_index_bumps_on_collision(shell: Shell, tmp_path: Path) -> None:
    """If a log file already exists at the chosen run_index, we bump past it.

    Protects against overwriting stale logs from before a server
    restart (where the workspace survives but the in-memory counter
    resets to 0).
    """
    # Pre-populate a "stale" log file at index 1 — simulating a
    # counter-reset scenario where default-1.log already exists on
    # disk from a previous shell lifetime.
    terminal_dir = tmp_path / ".agent_plane" / "terminal"
    terminal_dir.mkdir(parents=True, exist_ok=True)
    stale = terminal_dir / "default-1.log"
    stale.write_text("stale content from before")

    # First overflow run — counter was 0, bumps to 1, collides with
    # stale, bumps to 2, succeeds.
    shell.run_sync("python3 -c 'import sys; sys.stdout.write(\"a\" * 50000)'")

    # Stale file untouched.
    assert stale.read_text() == "stale content from before"
    # New file is at index 2 (skipped past the stale 1).
    new_file = terminal_dir / "default-2.log"
    assert new_file.is_file()
    assert len(new_file.read_text()) > 30_000


def test_run_index_increments_across_overflows(shell: Shell, tmp_path: Path) -> None:
    """Multiple overflow runs in one shell produce distinct files.

    Covers the expected common case (not the collision case above):
    no pre-existing files, counter increments 1, 2, 3...
    """
    shell.run_sync("python3 -c 'import sys; sys.stdout.write(\"a\" * 50000)'")
    shell.run_sync("python3 -c 'import sys; sys.stdout.write(\"b\" * 50000)'")
    shell.run_sync("python3 -c 'import sys; sys.stdout.write(\"c\" * 50000)'")

    terminal_dir = tmp_path / ".agent_plane" / "terminal"
    log_files = sorted(terminal_dir.glob("default-*.log"))
    # Expected: three files at indexes 1, 2, 3.
    assert [p.name for p in log_files] == [
        "default-1.log",
        "default-2.log",
        "default-3.log",
    ]


# ---- ring-buffer eviction marker --------------------------------


def test_eviction_marker_appears_when_ring_overflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output bigger than the ring buffer capacity shows an eviction marker.

    Shrinks ``_RING_CAPACITY_BYTES`` via monkeypatch so we can trigger
    eviction with a few kilobytes of output rather than a full
    megabyte. Doesn't use the shared ``shell`` fixture because the
    fixture spawns with the default 1 MB capacity; we need to spawn
    after patching the module constant.
    """
    # Patch the module constant BEFORE spawning so the new Shell
    # gets a tiny ring. Uses only public API from the test's POV
    # (no reaching into instance internals).
    monkeypatch.setattr("agent_plane.terminals.shell._RING_CAPACITY_BYTES", 1000)
    s = Shell.spawn("default", tmp_path, sandbox_enabled=False)
    try:
        result = s.run_sync("python3 -c 'import sys; sys.stdout.write(\"a\" * 5000)'")
    finally:
        s.close()

    assert result.status == "completed"
    # Eviction marker is surfaced so the agent knows data was lost.
    assert "evicted" in result.stdout
    # And because eviction happened, a (partial) disk log was written
    # and its path appears in the eviction marker so the agent can
    # retrieve whatever survived.
    assert "partial log at" in result.stdout
    assert ".agent_plane/terminal/" in result.stdout


# ---- shell_busy (fail-fast on concurrent same-shell run) --------


def _wait_for_flag(flag: Path, timeout_s: float) -> bool:
    """Busy-wait until a flag file appears or the timeout elapses.

    Used by ``shell_busy`` tests to deterministically synchronize
    with bash-side command execution. When the agent touches a flag
    file in the first line of a chained command, the flag's
    existence proves bash has already consumed the ``sendline`` and
    is executing — which means the cmd lock is held. Subsequent
    ``sleep`` on the same chained line gives the test window to
    fire a second ``run_sync`` against the busy shell.

    :param flag: The path to poll for existence.
    :param timeout_s: Max seconds to wait before returning False.
    :returns: True if the flag appeared, False on timeout.
    """
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        if flag.exists():
            return True
    return False


def test_concurrent_run_on_same_shell_returns_shell_busy(
    shell: Shell, tmp_path: Path
) -> None:
    """A second thread's run_sync returns shell_busy while the first is running.

    The §6.1 Option D semantic: a terminal_run on a shell that's
    currently executing another command must fail-fast with
    ``shell_busy`` rather than queueing behind the per-shell cmd lock.
    Without fail-fast, an agent that accidentally double-ran a
    long-running build would silently wedge — we want the agent to
    see the collision immediately so it can choose: retry later,
    spawn a parallel shell, abandon.

    This is a true concurrency test. Synchronization is done
    entirely through the public shell API + a filesystem flag the
    bash subprocess creates. Once the flag exists, bash has already
    consumed the first ``sendline`` and is executing the chained
    ``sleep`` — which means the cmd lock is held. The second
    ``run_sync`` call then lands mid-execution, exercising the
    fail-fast path. No access to private shell state.
    """
    import threading

    first_result: list[object] = []
    flag = tmp_path / "first-started.flag"

    def first_runner() -> None:
        # touch → flag becomes visible; sleep → holds the cmd lock.
        # The second run_sync call fires *after* the flag appears,
        # so it's guaranteed to collide with the sleep.
        first_result.append(
            shell.run_sync(f"touch {flag} && sleep 2 && echo first")
        )

    t1 = threading.Thread(target=first_runner)
    t1.start()
    try:
        assert _wait_for_flag(flag, timeout_s=3.0), (
            "first_runner never created the flag file — bash may "
            "have failed to start or the shell is broken before "
            "the concurrency test even starts."
        )

        # Second call fires while first still holds the lock.
        # MUST return shell_busy immediately — not block.
        second = shell.run_sync("echo second")
    finally:
        t1.join()

    # Assert the collision surfaced as expected. If "completed",
    # the run_sync non-blocking acquire isn't working and calls
    # are serializing behind the cmd lock.
    assert second.status == "shell_busy", (
        f"Expected shell_busy when second run arrived mid-execution, "
        f"got status={second.status!r}. If 'completed', the run_sync "
        f"non-blocking acquire isn't working and calls are serializing."
    )
    assert second.stdout == ""
    assert second.exit_code is None
    assert second.shell == shell.name

    # The first call still completed normally — shell_busy on the
    # second doesn't interfere with the first's execution.
    assert len(first_result) == 1
    r1 = first_result[0]
    assert r1.status == "completed"  # type: ignore[attr-defined]
    assert "first" in r1.stdout  # type: ignore[attr-defined]


def test_shell_busy_does_not_leave_shell_wedged(
    shell: Shell, tmp_path: Path
) -> None:
    """After a shell_busy collision resolves, the shell is usable again.

    Regression guard: if the non-blocking acquire path mistakenly
    ran command-setup side-effects (ring.reset, proc.sendline) before
    returning shell_busy, the shell's state would be corrupted and
    subsequent commands would see garbage. Assert that a clean
    echo command after the collision resolves returns clean output.
    """
    import threading

    flag = tmp_path / "runner-2.flag"

    def first_runner() -> None:
        shell.run_sync(f"touch {flag} && sleep 1 && echo first-done")

    t1 = threading.Thread(target=first_runner)
    t1.start()
    try:
        assert _wait_for_flag(flag, timeout_s=3.0)
        # Trigger a shell_busy while the first runner holds the lock.
        busy = shell.run_sync("echo busy-attempt")
        # Sanity check: the setup really did produce a busy result.
        # If this is "completed", the fail-fast path isn't being
        # exercised and the state-recovery assertion below is
        # meaningless (the test would pass for the wrong reason).
        assert busy.status == "shell_busy"
    finally:
        t1.join()

    # Shell should be clean and usable.
    r = shell.run_sync("echo post")
    assert r.status == "completed", (
        f"Shell appeared wedged after shell_busy path — status={r.status!r}. "
        f"If 'shell_crashed' or unexpected stdout, run_sync's busy path "
        f"left the shell in a bad state."
    )
    assert "post" in r.stdout
    assert r.exit_code == 0


# ── interrupt() — sends SIGINT to the running command ──────────


def test_interrupt_kills_running_command_shell_survives(
    shell: Shell, tmp_path: Path
) -> None:
    """``shell.interrupt()`` unblocks a sleeping command and returns
    ``status="killed"``; the shell itself is still usable.

    This is what ``cancel_task`` calls on a ``kind="terminal"`` task.
    If interrupt failed to reach bash (wrong fd, wrong signal), the
    sleep would time out instead of getting SIGINT, and exit_code
    would be 0, not 130.
    """
    import threading
    import time as _time

    # Start a long sleep in a background thread so we can interrupt
    # it from this thread.
    result_holder: list[object] = []
    started = threading.Event()

    def _run() -> None:
        started.set()
        result_holder.append(shell.run_sync("sleep 10 && echo late"))

    t = threading.Thread(target=_run)
    t.start()
    try:
        assert started.wait(timeout=2.0)
        # Give the sleep a moment to actually start. We need bash
        # to be IN the sleep (holding the cmd lock) before interrupt
        # arrives. 300 ms is >> sendline + read-loop entry time.
        # This isn't time.sleep in test assertion territory — it's
        # a deliberate small wait to cross a known setup barrier
        # (no better primitive available without reaching into
        # private Shell state).
        _time.sleep(0.3)
        shell.interrupt()
    finally:
        t.join(timeout=10.0)

    assert len(result_holder) == 1
    r = result_holder[0]
    assert r.status == "killed", (  # type: ignore[attr-defined]
        f"Expected 'killed' after SIGINT, got {r.status!r}. "  # type: ignore[attr-defined]
        f"If 'completed', the SIGINT didn't reach bash — interrupt "
        f"may be writing to the wrong fd or the PTY isn't wired up."
    )
    # SIGINT → exit 130 (128 + 2) per bash convention.
    assert r.exit_code == 130  # type: ignore[attr-defined]

    # Shell still works.
    r2 = shell.run_sync("echo alive")
    assert r2.status == "completed"
    assert "alive" in r2.stdout


def test_interrupt_on_dead_shell_is_noop(tmp_path: Path) -> None:
    """Calling ``interrupt()`` after the shell has died doesn't raise.

    Regression guard for a naive implementation that would call
    ``proc.sendintr()`` unconditionally and blow up on a closed fd.
    """
    s = Shell.spawn("dead-test", tmp_path, sandbox_enabled=False)
    s.close()
    # Should be a silent no-op, not an exception.
    s.interrupt()


# ── peek_partial_stdout — delta reads during a running command ─


def test_peek_partial_stdout_empty_before_command() -> None:
    """Peeking before any command produces empty delta + cursor 0.

    Establishes the baseline: a shell with no history returns
    no bytes. If this produced junk, ``check_task`` on a brand-new
    terminal task would show phantom output.
    """
    from pathlib import Path as _Path

    from agent_plane.terminals import PartialReadResult, Shell as _Shell

    with tempfile_dir() as d:
        s = _Shell.spawn("peek-empty", _Path(d), sandbox_enabled=False)
        try:
            p = s.peek_partial_stdout(0)
            assert isinstance(p, PartialReadResult)
            assert p.text == ""
            assert p.new_cursor == 0
            assert p.lost_bytes == 0
        finally:
            s.close()


def test_peek_partial_stdout_returns_delta_during_command(
    shell: Shell,
) -> None:
    """During a running command, successive peeks return only new bytes.

    Simulates check_task's polling. Uses a command that prints
    two chunks ~0.5s apart, peeks between them, asserts:
    - first peek sees chunk 1 but not chunk 2 (yet)
    - second peek sees chunk 2 but not chunk 1 again
    """
    import threading

    result_holder: list[object] = []
    done = threading.Event()

    def _run() -> None:
        result_holder.append(
            shell.run_sync(
                "echo chunk-one; sleep 0.6; echo chunk-two; sleep 0.1"
            )
        )
        done.set()

    t = threading.Thread(target=_run)
    t.start()
    try:
        # First peek: wait for chunk-one to appear in the buffer.
        deadline = __import__("time").monotonic() + 3.0
        first = shell.peek_partial_stdout(0)
        while "chunk-one" not in first.text and __import__("time").monotonic() < deadline:
            first = shell.peek_partial_stdout(0)
        assert "chunk-one" in first.text, (
            f"First peek didn't see chunk-one; got {first.text!r}."
        )
        assert "chunk-two" not in first.text, (
            f"First peek already saw chunk-two — the sleep in the "
            f"bash command didn't hold: {first.text!r}."
        )

        # Second peek: use the new cursor from first.
        deadline2 = __import__("time").monotonic() + 3.0
        second = shell.peek_partial_stdout(first.new_cursor)
        while "chunk-two" not in second.text and __import__("time").monotonic() < deadline2:
            second = shell.peek_partial_stdout(first.new_cursor)
        assert "chunk-two" in second.text, (
            f"Second peek didn't see chunk-two within 3s; got "
            f"{second.text!r}."
        )
        # Crucially: chunk-one should NOT appear again — it was
        # before the cursor.
        assert "chunk-one" not in second.text, (
            f"Second peek returned chunk-one again — cursor wasn't "
            f"advancing: {second.text!r}."
        )
    finally:
        done.wait(timeout=5.0)
        t.join(timeout=5.0)

    assert len(result_holder) == 1
    r = result_holder[0]
    assert r.status == "completed"  # type: ignore[attr-defined]


# Helper for test_peek_partial_stdout_empty_before_command — builds
# a tempdir context manager so we don't need a fixture. Kept local
# to this file (not fixture-promotion territory) because it's used
# in one test here.


def tempfile_dir():
    """Context manager for a throwaway tmp dir. Used by the empty-peek
    test which needs its own shell independent of the shared fixture.

    :returns: A tempfile.TemporaryDirectory instance.
    """
    import tempfile as _t

    return _t.TemporaryDirectory()
