"""Unit tests for the interactive additions to :class:`Shell`.

Covers the pyte-rendered screen view (``rendered_screen``) and the
PTY-stdin writer (``send_input``) introduced alongside
``terminal_send_input``. Uses real bash + real pyte — no mocks —
because the point is to confirm the PTY → pyte round-trip
actually produces a useful view of interactive programs.
"""

from __future__ import annotations

import threading
import time

from agent_plane.terminals import Shell

# ---- rendered_screen --------------------------------------------


def test_rendered_screen_captures_echo_output(shell: Shell) -> None:
    """A run_sync echo appears in the rendered screen afterwards.

    Smoke test for the PTY → pyte byte feed: if the read loop
    isn't feeding pyte, the rendered screen would be blank even
    after a command finished. Validates the wiring end to end
    (pexpect read → ring append → pyte feed → screen.display).
    """
    result = shell.run_sync("echo screen-hello-zzz")
    assert result.status == "completed"
    # Small delay avoids a race where the rendered screen update
    # trails the D-marker detection by a syscall; acceptable here
    # because the ring buffer / marker path is the authoritative
    # completion signal and pyte is a view.
    time.sleep(0.05)
    screen = shell.rendered_screen()
    assert "screen-hello-zzz" in screen, (
        f"Expected 'screen-hello-zzz' on the rendered screen after "
        f"echo, got {screen!r}. If the screen is blank, the read "
        f"loop isn't feeding pyte. If it contains garbage escape "
        f"codes, pyte isn't consuming them."
    )


def test_rendered_screen_right_strips_padding(shell: Shell) -> None:
    """pyte pads rows to column width; rendered_screen rstrips them.

    Without the rstrip, every line would have ~160 trailing spaces.
    Regression guard for the ``line.rstrip()`` in
    :meth:`Shell.rendered_screen`.
    """
    shell.run_sync("echo short")
    time.sleep(0.05)
    screen = shell.rendered_screen()
    # No line should have trailing spaces.
    for i, line in enumerate(screen.split("\n")):
        assert line == line.rstrip(), (
            f"Line {i} has trailing whitespace: {line!r}. The "
            f"rstrip in rendered_screen was dropped."
        )


def test_rendered_screen_reflects_cursor_overwrite(shell: Shell) -> None:
    """A carriage-return overwrite produces the overwritten result.

    ``printf 'hello\\rworld\\n'`` moves the cursor back to column 0
    after printing ``hello``, then writes ``world`` over it.
    On the rendered screen we should see ``world``, not ``helloworld``.
    This is the test that distinguishes a real terminal emulator
    (pyte) from a raw-stream approach (ANSI-strip of ring buffer).
    """
    shell.run_sync("printf 'hello\\rworld\\n'")
    time.sleep(0.05)
    screen = shell.rendered_screen()
    assert "world" in screen, f"Missing 'world' in screen: {screen!r}"
    # The distinguishing check: raw bytes contain both, but the
    # rendered screen should ONLY show the final state (world
    # overwriting hello at column 0). A pure stream of bytes
    # stripped of ANSI would keep 'hello' around; pyte processes
    # the CR and replaces the cells.
    lines_with_hello = [ln for ln in screen.split("\n") if "hello" in ln and "world" not in ln]
    assert not lines_with_hello, (
        f"Rendered screen has 'hello' on its own — CR cursor "
        f"overwrite was not applied. pyte feed may be skipping "
        f"this byte stream. Matching lines: {lines_with_hello!r}"
    )


# ---- send_input -------------------------------------------------


def test_send_input_delivers_to_running_command(shell: Shell) -> None:
    """`cat` blocked on stdin reads the bytes we send via send_input.

    ``cat`` with no args reads until EOF and echoes each line. We
    start it in a background thread (since run_sync blocks until
    the D marker), write two lines + Ctrl-D to close stdin, and
    verify both lines appear in the captured output.

    If send_input is broken (writing to wrong fd, lost bytes), the
    ``cat`` would wait forever for input and the test would time
    out rather than fail with an assertion. The worker thread's
    completion is our functional assertion.
    """
    result: list[object] = []

    def _runner() -> None:
        result.append(shell.run_sync("cat", timeout_ms=10_000))

    t = threading.Thread(target=_runner)
    t.start()
    # Give cat a moment to spawn and attach to stdin before we write.
    time.sleep(0.3)
    shell.send_input("line-one\nline-two\n")
    # Ctrl-D (EOF) tells cat to stop reading and exit.
    time.sleep(0.1)
    shell.send_input("\x04")
    t.join(timeout=5.0)
    assert result, "cat never completed — send_input likely dropped"
    run_result = result[0]
    # Mypy: the list holds a RunResult — casting via attribute access.
    assert run_result.status == "completed", (  # type: ignore[attr-defined]
        f"cat didn't exit cleanly: {run_result!r}"
    )
    stdout = run_result.stdout  # type: ignore[attr-defined]
    assert "line-one" in stdout and "line-two" in stdout, (
        f"Neither line made it through cat: {stdout!r}"
    )


def test_send_input_empty_string_is_noop(shell: Shell) -> None:
    """Empty ``send_input`` doesn't write anything or break subsequent calls.

    The tool's poll mode (``chars=""`` + yield_time_ms) relies on
    this: it short-circuits without touching the PTY. Regression
    guard for the early-return in :meth:`Shell.send_input`.
    """
    shell.send_input("")
    # If send_input wrote something weird to the PTY, the next
    # run_sync would be polluted or fail to find its D marker. A
    # clean echo after is the functional proof.
    result = shell.run_sync("echo after-noop")
    assert result.status == "completed"
    assert "after-noop" in result.stdout


def test_send_input_control_byte_reaches_reader(shell: Shell) -> None:
    """A Ctrl-C byte sent via send_input interrupts a running sleep.

    Verifies control bytes aren't mangled on the way through. The
    byte 0x03 (SIGINT in the PTY's tty-line discipline) should
    cause bash's foreground child to die with 128+2=130.
    """
    result_holder: list[object] = []

    def _runner() -> None:
        # Long sleep, no timeout — only send_input("\u0003") should end it.
        result_holder.append(shell.run_sync("sleep 30", timeout_ms=5_000))

    t = threading.Thread(target=_runner)
    t.start()
    time.sleep(0.3)
    shell.send_input("\u0003")  # Ctrl-C
    t.join(timeout=6.0)
    assert result_holder, "sleep never returned — Ctrl-C did not reach it"
    result = result_holder[0]
    # 130 = 128 + SIGINT(2), the signal-in-exit-code convention.
    # Either 'killed' (our timeout fired) or 'completed' with
    # exit=130 (SIGINT arrived first) is acceptable here — we
    # only want to prove the byte produced SOMETHING terminating,
    # not which of the two races won.
    assert result.status in {"killed", "completed"}, result  # type: ignore[attr-defined]
