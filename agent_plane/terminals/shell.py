"""A persistent bash shell with OSC 633 completion detection.

A ``Shell`` wraps a single long-lived ``pexpect.spawn``-managed bash
subprocess. The shell is started with our shell-integration snippet
as its rcfile, which installs an OSC 633 D marker after every
command's exit. ``run_sync`` sends a command, drives the read loop
until the D marker arrives (or ``timeout_ms`` fires, or bash dies),
and returns a structured :class:`RunResult`.

See ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.1 (tool shape),
§6.7 (output buffering + size limits), §6.8 (shell launch contract),
and §6.9 (threading model). This module is slices 1 + 2; slices 3–5
add the manager, registry, tool wiring, and migration on top.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pexpect

from agent_plane.terminals._sandbox import (
    build_srt_config,
    is_srt_available,
    srt_shell_path,
)
from agent_plane.terminals.ansi import strip_ansi
from agent_plane.terminals.ring_buffer import RingBuffer

# Absolute path to the bundled shell-integration snippet. Sourced via
# ``bash --rcfile`` when spawning.
_INTEGRATION_SNIPPET = Path(__file__).parent / "_integration.sh"

# OSC 633 D marker: ``ESC ] 633 ; D ; <exit_code> BEL`` (or ST
# terminator ``ESC \``). Captures the exit code as a non-negative
# integer (bash's ``$?`` convention: 128 + signal for signal
# terminations, e.g. SIGINT → 130, SIGKILL → 137). See
# ``_integration.sh`` for the emitter side.
_D_MARKER = re.compile(rb"\x1b\]633;D;(-?\d+)(?:\x07|\x1b\\)")

# Seconds to wait for the initial D marker after spawning bash.
# The snippet's PROMPT_COMMAND should fire before the first prompt
# with exit_code=0. 5s is generous; a healthy shell takes <100ms.
_SPAWN_READY_TIMEOUT_S = 5.0

# Per-read-call pexpect timeout used in the manual read loop. Must be
# short enough that we react to user-requested timeouts promptly.
# 0.25s means the loop wakes at least 4x/sec to check the deadline.
_READ_POLL_INTERVAL_S = 0.25

# Bytes to read per pexpect.read_nonblocking call. Bigger = fewer
# iterations of the read loop; smaller = faster reaction to timeout.
# 4 KiB balances both.
_READ_CHUNK_SIZE = 4096

# Grace window after sending Ctrl-C, before we escalate to SIGKILL on
# the whole bash process. Most commands respond to SIGINT within a
# few hundred ms.
_TIMEOUT_GRACE_S = 2.0

# Per-shell ring-buffer capacity (see §6.7 Layer 1). 1 MB = ~30k
# typical output lines; eviction markers surface any loss.
_RING_CAPACITY_BYTES = 1_000_000

# Inline return cap (see §6.7 Layer 2). Above this size, the stdout
# string is head+tail truncated and the full output is persisted to
# disk. Matches Claude Code's 30 KB default.
_INLINE_CAP_CHARS = 30_000

# Head and tail slice sizes for truncation. Sum (20 KB) plus the
# truncation marker must comfortably fit under the inline cap.
_HEAD_CHARS = 10_000
_TAIL_CHARS = 10_000

# Subdirectory of the workspace where overflow logs live. Stored
# relative so the path shown in stdout markers is workspace-relative
# and copy-pasteable into ``cat`` / ``Read``-style tools.
_DISK_LOG_SUBDIR = Path(".agent_plane") / "terminal"


@dataclasses.dataclass(frozen=True)
class PartialReadResult:
    """One delta read of a shell's stdout since a caller-supplied cursor.

    Returned by :meth:`Shell.peek_partial_stdout`. The caller stores
    ``new_cursor`` and passes it on the next call to resume cleanly.

    :param text: ANSI-stripped UTF-8 text of the delta. Empty if no
        new bytes since the cursor.
    :param new_cursor: Byte offset the caller should pass next time.
        Monotonically non-decreasing.
    :param lost_bytes: Count of bytes that were written between the
        caller's cursor and the retained bytes — i.e. evicted before
        the caller could read them. Non-zero when the ring buffer
        overflows between observations. Callers can surface this as
        a ``[... N bytes lost ...]`` marker.
    """

    text: str
    new_cursor: int
    lost_bytes: int


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Outcome of running a command in a shell.

    :param stdout: Command output, ANSI-stripped and possibly
        truncated. If truncated (over :data:`_INLINE_CAP_CHARS`),
        contains a head-marker-tail layout with a reference to the
        full-output disk file. If the ring buffer evicted bytes
        mid-command, an ``[... N bytes evicted ...]`` marker is
        prepended.
    :param exit_code: Command exit code per bash's ``$?`` convention
        (128 + signal for signal termination: SIGINT → 130, SIGKILL
        → 137). ``None`` only when ``status == "shell_crashed"``, in
        which case we have no exit code because bash itself died
        before emitting the D marker.
    :param status: One of ``"completed"`` (ran, D marker seen),
        ``"killed"`` (timeout fired, command was Ctrl-C'd
        successfully, bash still alive), ``"shell_crashed"`` (bash
        died underneath the command; shell is no longer usable), or
        ``"shell_busy"`` (another thread is running a command in
        this shell; nothing was executed and the shell is still
        alive — see §6.1 Option D fail-fast semantic).
    :param shell: Logical name of the shell that ran the command;
        echoes the name passed to :meth:`Shell.spawn`.
    """

    stdout: str
    exit_code: int | None
    status: str
    shell: str


class Shell:
    """One persistent bash subprocess with OSC 633 command-boundary detection.

    Instances are created via :meth:`Shell.spawn` and torn down via
    :meth:`Shell.close`. Commands are run via :meth:`Shell.run_sync`.

    Cross-thread safety: the per-shell mutex (``self._cmd_lock``)
    serializes command execution so concurrent ``run_sync`` calls on
    the same shell can't corrupt the PTY stream. Slice 3's
    ``TerminalManager`` will convert a collision attempt into an
    immediate ``shell_busy`` response; here we just block callers
    behind the lock.

    A shell becomes "dead" after the bash subprocess exits
    unexpectedly (SIGKILL escalation, crash, OOM). Subsequent
    ``run_sync`` calls on a dead shell return status
    ``"shell_crashed"`` immediately. The owning ``TerminalManager``
    (slice 3) is responsible for removing dead shells from its map.
    """

    def __init__(self, name: str, workspace: Path, proc: pexpect.spawn) -> None:
        """Direct construction is internal; use :meth:`Shell.spawn`.

        :param name: Logical shell name (e.g. ``"default"``).
        :param workspace: Workspace directory. Used as disk-log
            location for output-overflow files.
        :param proc: An already-spawned, already-ready pexpect handle
            — one that has consumed its initial OSC 633 D marker
            and is parked at an empty prompt, ready for
            :meth:`run_sync`.
        """
        self._name = name
        self._workspace = workspace
        self._proc = proc
        self._cmd_lock = threading.Lock()
        self._ring = RingBuffer(_RING_CAPACITY_BYTES)
        # Monotonic per-shell index for disk log filenames. Reset
        # when the shell restarts (because the Shell instance does
        # too). File-existence check on write bumps past any
        # stale-log collisions from before a server restart.
        self._run_index = 0
        # Flag: did ``interrupt()`` get called during the current
        # command? Reset at the top of each ``_run_locked``. When
        # set, the result builder returns status ``"killed"`` even
        # though the command's D marker arrived normally — matches
        # the ``timeout_ms`` path's status semantics.
        #
        # ``threading.Event`` rather than a bare bool so the
        # interrupt-arrival signal is visible across threads without
        # needing a separate lock.
        self._interrupt_signal = threading.Event()

    @property
    def name(self) -> str:
        """The shell's logical name.

        :returns: The name passed to :meth:`spawn`.
        """
        return self._name

    def is_alive(self) -> bool:
        """Whether the underlying bash subprocess is still running.

        Used by :meth:`run_sync` to short-circuit on dead shells
        and (in slice 3) by ``TerminalManager`` to prune dead
        entries from its map.

        :returns: True if bash is running; False if it has exited
            (normally or via crash/kill).
        """
        return bool(self._proc.isalive())

    @classmethod
    def spawn(cls, name: str, workspace: Path, *, sandbox_enabled: bool = True) -> Shell:
        """Spawn a new bash subprocess configured for OSC 633 detection.

        Launches ``bash --rcfile <integration-snippet>`` with the
        workspace as cwd and workspace-relative package-install env
        vars set. Consumes the initial D marker emitted by the
        snippet's ``PROMPT_COMMAND`` on the first prompt — once
        that's seen, the shell is ready for commands.

        When ``sandbox_enabled`` is True (default), the bash subprocess
        is wrapped in an ``srt`` filesystem sandbox via the PTY-compatible
        :file:`_srt_shell.mjs` Node wrapper. The sandbox allows read+write
        inside ``workspace``, denies reads of the user's home and
        ``/tmp``, and lets network traffic pass through. See
        ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.5.

        :param name: Logical name for the shell (stored and returned
            in every :class:`RunResult`).
        :param workspace: Working directory for the spawned bash.
            Package installs (pip, npm) will be directed into this
            tree via ``PIP_TARGET`` etc.
        :param sandbox_enabled: If True (default), wrap the bash in an
            srt filesystem sandbox. If srt isn't available, spawn
            fails loud rather than silently running unsandboxed. If
            False, runs plain bash with no filesystem isolation.
        :returns: A :class:`Shell` parked at an empty prompt.
        :raises RuntimeError: If ``bash`` is not on ``PATH``; the
            integration snippet is missing; ``sandbox_enabled`` is
            True but ``srt`` or ``node`` is not on ``PATH``; or the
            snippet fails to emit the initial D marker within
            :data:`_SPAWN_READY_TIMEOUT_S` seconds.
        """
        if shutil.which("bash") is None:
            raise RuntimeError("bash is not installed or not on PATH")

        # Fail loud if our bundled snippet has gone missing (packaging
        # glitch, accidental deletion during refactor, etc.) — better
        # than a vague "snippet may be broken" timeout error from the
        # pexpect path below.
        if not _INTEGRATION_SNIPPET.is_file():
            raise RuntimeError(
                f"Terminal integration snippet missing at {_INTEGRATION_SNIPPET}. "
                "This is a packaging bug — the snippet should ship alongside shell.py."
            )

        # Sandbox pre-flight: if sandboxing is requested but srt/node
        # aren't on PATH, refuse to launch. Silently falling back to
        # unsandboxed bash would be a security-surprise behavior (the
        # operator configured sandboxing but got none). Per Principle
        # #3 (fail loud), raise instead.
        if sandbox_enabled and not is_srt_available():
            raise RuntimeError(
                "sandbox_enabled=True but srt or node is not on PATH. "
                "Install srt + node, or pass sandbox_enabled=False to "
                "run bash without filesystem isolation."
            )

        env = cls._build_env(workspace)
        bin_name, bin_args = cls._build_spawn_argv(workspace, sandbox_enabled)
        proc = pexpect.spawn(
            bin_name,
            bin_args,
            cwd=str(workspace),
            env=env,
            # Bytes mode: OSC 633 markers are byte sequences; decoding
            # in pexpect itself can trip over partially-received
            # multi-byte chars mid-marker.
            encoding=None,
        )
        # Disable local echo so command text doesn't appear in captured
        # stdout between marker boundaries.
        proc.setecho(False)

        try:
            proc.expect(_D_MARKER, timeout=_SPAWN_READY_TIMEOUT_S)
        except pexpect.exceptions.TIMEOUT as exc:
            proc.close(force=True)
            raise RuntimeError(
                "Shell did not emit initial OSC 633 D marker within "
                f"{_SPAWN_READY_TIMEOUT_S:.1f}s. The integration "
                f"snippet at {_INTEGRATION_SNIPPET} may be broken."
            ) from exc
        except pexpect.exceptions.EOF as exc:
            proc.close(force=True)
            raise RuntimeError(
                "Bash exited before emitting the initial OSC 633 D "
                "marker. Likely a problem sourcing the integration "
                f"snippet at {_INTEGRATION_SNIPPET}."
            ) from exc

        return cls(name=name, workspace=workspace, proc=proc)

    @staticmethod
    def _build_spawn_argv(
        workspace: Path, sandbox_enabled: bool
    ) -> tuple[str, list[str]]:
        """Build the (executable, argv) pair passed to pexpect.spawn.

        When ``sandbox_enabled`` is True, returns an invocation of
        :file:`_srt_shell.mjs` with the filesystem config + the bash
        argv as JSON. The Node wrapper execs bash inside the srt
        sandbox with stdio inherited, so the PTY pexpect attached to
        node transparently reaches bash.

        When ``sandbox_enabled`` is False, returns a direct bash
        invocation. This code path exists for:

        - Unit test environments where srt isn't installed.
        - Operators who explicitly disabled sandboxing.

        This is the ONLY sanctioned dual path in the terminal tool
        (Principle #2). The two branches produce the same on-PTY
        behavior; only filesystem isolation differs.

        :param workspace: Per-conversation workspace used as bash
            cwd and as the writable root inside the sandbox.
        :param sandbox_enabled: Whether to wrap in srt.
        :returns: ``(executable, argv)`` suitable for
            :func:`pexpect.spawn`. Executable is a string; argv is
            a list of strings (not including executable at index 0).
        """
        bash_argv = ["bash", "--rcfile", str(_INTEGRATION_SNIPPET)]
        if not sandbox_enabled:
            # Direct bash spawn — no wrapping.
            return bash_argv[0], bash_argv[1:]
        # Sandboxed path: node _srt_shell.mjs <config> <bash-argv-json>.
        # The wrapper srt-initializes, then execs bash inside the
        # sandbox with stdio inherited, chaining back to pexpect's PTY.
        import json as _json

        config_json = build_srt_config(workspace)
        argv_json = _json.dumps(bash_argv)
        return "node", [srt_shell_path(), config_json, argv_json]

    @staticmethod
    def _build_env(workspace: Path) -> dict[str, str]:
        """Build the bash subprocess environment.

        Inherits the server's environment, then overlays the
        workspace-relative package-install paths so ``pip install``
        and ``npm install`` stay within the conversation's workspace.
        Applied once at shell launch so the agent's first command
        already sees these variables.

        :param workspace: The workspace directory path.
        :returns: A new environment mapping suitable for
            :func:`pexpect.spawn`'s ``env=`` kwarg.
        """
        ws = str(workspace)
        return {
            **os.environ,
            "PIP_TARGET": f"{ws}/.pip",
            "PIP_CACHE_DIR": f"{ws}/.cache/pip",
            "PYTHONPATH": f"{ws}/.pip",
            "NODE_PATH": f"{ws}/node_modules",
            "npm_config_prefix": ws,
            "npm_config_cache": f"{ws}/.cache/npm",
        }

    def run_sync(
        self,
        command: str,
        timeout_ms: int | None = None,
        cancel_predicate: Callable[[], bool] | None = None,
    ) -> RunResult:
        """Run a command synchronously and return its captured output.

        Sends ``command`` to the bash subprocess and drives a read
        loop until one of four outcomes:

        - **Completed**: the OSC 633 D marker appears; exit code is
          extracted, output is ANSI-stripped + possibly truncated,
          and :class:`RunResult` with ``status="completed"`` is
          returned.
        - **Killed**: ``timeout_ms`` elapses; a Ctrl-C is sent to
          interrupt the command. If bash responds within
          :data:`_TIMEOUT_GRACE_S` with a D marker, we return
          ``status="killed"`` with the real signal-exit code (typ.
          130 for SIGINT). Otherwise bash itself is stuck and we
          SIGKILL the whole shell, returning
          ``status="shell_crashed"``.
        - **Shell crashed**: bash exits during the command (segfault,
          OOM, SIGKILL). Returns ``status="shell_crashed"`` with
          ``exit_code=None``; the shell is no longer usable.
        - **Shell busy**: another thread is already running a command
          in this shell. Returns ``status="shell_busy"`` immediately
          — the caller never blocks behind the cmd lock. This is
          the §6.1 Option D semantic: concurrent calls fail-fast
          rather than queue, so the agent can decide how to react
          (retry, spawn a new shell, abandon) rather than getting
          wedged waiting for an unknown-duration command.

        :param command: Bash command to execute. May contain chained
            simple commands — exit code reflects the last one per
            ``$?`` semantics.
        :param timeout_ms: Maximum milliseconds the command may run.
            ``None`` means no bound (command may run indefinitely).
            When it fires, the command is Ctrl-C'd, not the shell.
        :param cancel_predicate: Optional callable returning True
            when the caller wants the command cancelled mid-flight.
            Checked from the shell's own thread (the read loop) so
            there's no cross-thread pexpect race. When it flips
            True, the read loop sends SIGINT to bash and keeps
            reading until the D marker arrives with exit 130.
            Result status will be ``"killed"``. Called at most once
            per read-loop iteration (~4 Hz); side-effect-free
            predicates are recommended.
        :returns: A :class:`RunResult` describing the outcome.
        """
        if not self.is_alive():
            return RunResult(
                stdout="",
                exit_code=None,
                status="shell_crashed",
                shell=self._name,
            )
        # Non-blocking acquire: if another caller holds the cmd lock,
        # bail out immediately with ``shell_busy``. Blocking would
        # stall the agent for an unbounded time and hide genuine
        # concurrency from the agent's view.
        acquired = self._cmd_lock.acquire(blocking=False)
        if not acquired:
            return RunResult(
                stdout="",
                exit_code=None,
                status="shell_busy",
                shell=self._name,
            )
        try:
            return self._run_locked(command, timeout_ms, cancel_predicate)
        finally:
            self._cmd_lock.release()

    def _run_locked(
        self,
        command: str,
        timeout_ms: int | None,
        cancel_predicate: Callable[[], bool] | None = None,
    ) -> RunResult:
        """Execute one command while holding ``self._cmd_lock``.

        Drives :meth:`_read_until_done`, then dispatches to the
        appropriate result-builder based on the outcome.

        :param command: See :meth:`run_sync`.
        :param timeout_ms: See :meth:`run_sync`.
        :returns: See :meth:`run_sync`.
        """
        self._ring.reset()
        # Clear any pending interrupt signal from a prior command.
        # A new command starts cleanly: if interrupt() is called
        # before the command's D marker arrives, the flag will be
        # set again and we'll report "killed".
        self._interrupt_signal.clear()
        self._proc.sendline(command)

        deadline = None
        if timeout_ms is not None:
            deadline = time.monotonic() + timeout_ms / 1000.0

        outcome = self._read_until_done(
            deadline=deadline,
            cancel_predicate=cancel_predicate,
        )
        if outcome == "completed":
            # If ``interrupt()`` was called during this command's
            # execution, the D marker arrived because of the SIGINT
            # (exit 130) — surface that as ``status="killed"`` so
            # the caller can distinguish agent-requested cancel from
            # clean completion. Without this, a cancel that races
            # the command's own completion would show up as
            # "completed" and hide the cancel semantics from
            # check_task.
            if self._interrupt_signal.is_set():
                return self._build_completed_or_killed_result("killed")
            return self._build_completed_or_killed_result("completed")
        if outcome == "timeout":
            return self._handle_timeout()
        # outcome == "crashed"
        return self._build_crashed_result()

    def _read_until_done(
        self,
        deadline: float | None,
        cancel_predicate: Callable[[], bool] | None = None,
    ) -> str:
        """Read chunks into the ring buffer until D marker, deadline, or EOF.

        :param deadline: Absolute monotonic-time deadline after which
            the command is considered timed out. ``None`` means no
            deadline.
        :param cancel_predicate: Optional callable polled between
            read iterations. When it returns True we send SIGINT
            (once — subsequent polls re-check but don't double-send)
            and keep reading until D arrives (exit 130) so we can
            classify the result as ``"killed"``. Called from this
            thread, so it's safe to inspect shared state the caller
            also mutates from another thread.
        :returns: One of ``"completed"`` (D marker seen — caller
            inspects :attr:`_interrupt_signal` to tell killed from
            clean completion), ``"timeout"`` (deadline reached), or
            ``"crashed"`` (bash EOF).
        """
        interrupted_here = False
        # Short grace so the cancel predicate is only polled after
        # bash has had a moment to set up the child. 100 ms is
        # sufficient in practice — the SIGINT path uses direct
        # ``os.kill`` on bash's children (see
        # :meth:`_interrupt_children`), not the PTY's VINTR, so
        # there's no risk of killing bash itself.
        cancel_grace_until = time.monotonic() + 0.1
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return "timeout"
            if (
                cancel_predicate is not None
                and not interrupted_here
                and time.monotonic() >= cancel_grace_until
                and cancel_predicate()
            ):
                # Caller requested cancel. Send SIGINT directly to
                # bash's foreground child via ``_interrupt_children``
                # (same reasoning as :meth:`interrupt`: PTY-level
                # VINTR is flaky under non-interactive bash +
                # pexpect, direct kill is reliable). Setting
                # ``_interrupt_signal`` makes _run_locked classify
                # the resulting D marker as ``"killed"``.
                self._interrupt_signal.set()
                self._interrupt_children()
                interrupted_here = True
            try:
                chunk = self._proc.read_nonblocking(
                    size=_READ_CHUNK_SIZE, timeout=_READ_POLL_INTERVAL_S
                )
            except pexpect.exceptions.TIMEOUT:
                # No data in this poll interval; loop and recheck deadline.
                continue
            except pexpect.exceptions.EOF:
                return "crashed"
            self._ring.append(chunk)
            if _D_MARKER.search(self._ring.bytes()):
                return "completed"

    def _handle_timeout(self) -> RunResult:
        """React to a user timeout: Ctrl-C, wait for D, SIGKILL if stuck.

        Standard two-stage kill: SIGINT first (the command should
        respond by emitting its own D marker with exit 128+2=130),
        then SIGKILL the whole shell if bash doesn't recover within
        :data:`_TIMEOUT_GRACE_S`. A SIGKILL is shell-ending — the
        shell is subsequently dead and future calls return
        ``shell_crashed``.

        :returns: ``RunResult`` with status ``"killed"`` if Ctrl-C
            worked, or ``"shell_crashed"`` if we had to SIGKILL.
        """
        # SIGINT bash's foreground child via os.kill. See
        # :meth:`_interrupt_children` for why we don't use the
        # PTY's VINTR path here. Child dies, bash emits D;130.
        self._interrupt_children()
        grace_deadline = time.monotonic() + _TIMEOUT_GRACE_S
        outcome = self._read_until_done(grace_deadline)
        if outcome == "completed":
            return self._build_completed_or_killed_result("killed")
        # Bash didn't recover: SIGKILL the whole shell. This ends the
        # shell; next run_sync gets shell_crashed from the is_alive
        # short-circuit.
        try:
            self._proc.kill(9)
        except ProcessLookupError:
            pass
        return self._build_crashed_result()

    def _build_completed_or_killed_result(self, status: str) -> RunResult:
        """Build a completed/killed result from the buffered output.

        Requires the D marker to already be present somewhere in the
        ring buffer. Slices output at the marker boundary, extracts
        the exit code, strips ANSI, prepends an eviction marker if
        the ring overflowed, and truncates to the inline cap.

        :param status: Either ``"completed"`` (normal finish) or
            ``"killed"`` (timeout_ms fired and Ctrl-C worked).
        :returns: The constructed :class:`RunResult`.
        """
        raw = self._ring.bytes()
        match = _D_MARKER.search(raw)
        assert match is not None, "caller must verify D marker present"
        exit_code = int(match.group(1))
        output_bytes = raw[: match.start()]
        evicted = self._ring.evicted_bytes
        stdout = strip_ansi(output_bytes)

        # Determine disk persistence up-front so both markers (eviction
        # and truncation) can reference the log path.
        disk_path = None
        if len(stdout) > _INLINE_CAP_CHARS or evicted > 0:
            # When eviction occurred, the disk file is itself partial
            # (we can only persist what's left in the ring) — the
            # eviction marker's phrasing makes that explicit.
            disk_path = self._persist_to_disk(stdout)

        if evicted > 0:
            suffix = f", partial log at {disk_path}" if disk_path else ""
            stdout = f"[... {evicted} bytes evicted (ring buffer overflow){suffix} ...]\n{stdout}"

        if len(stdout) > _INLINE_CAP_CHARS:
            stdout = self._truncate_head_tail(stdout, disk_path)

        return RunResult(
            stdout=stdout,
            exit_code=exit_code,
            status=status,
            shell=self._name,
        )

    def _build_crashed_result(self) -> RunResult:
        """Build a shell_crashed result from whatever the ring buffer has.

        Called when bash exited (EOF) or was SIGKILL'd during the
        command. No exit code is available (the D marker was never
        emitted), and the shell is no longer usable.

        :returns: The constructed :class:`RunResult`.
        """
        raw = self._ring.bytes()
        evicted = self._ring.evicted_bytes
        stdout = strip_ansi(raw)
        if evicted > 0:
            stdout = f"[... {evicted} bytes evicted (ring buffer overflow) ...]\n{stdout}"
        if len(stdout) > _INLINE_CAP_CHARS:
            disk_path = self._persist_to_disk(stdout)
            stdout = self._truncate_head_tail(stdout, disk_path)
        return RunResult(
            stdout=stdout,
            exit_code=None,
            status="shell_crashed",
            shell=self._name,
        )

    def _persist_to_disk(self, full_output: str) -> str:
        """Write the full output to a per-run log file.

        File named ``<shell>-<run_index>.log`` in the workspace's
        ``.agent_plane/terminal/`` subdir. ``run_index`` increments
        each call; on collision (e.g. after a server restart where
        the workspace survives but the counter was reset), bumps
        until a free name is found.

        :param full_output: The stdout to persist (already ANSI-stripped).
        :returns: Workspace-relative path string suitable for
            inclusion in the truncation marker.
        """
        terminal_dir = self._workspace / _DISK_LOG_SUBDIR
        terminal_dir.mkdir(parents=True, exist_ok=True)
        # Loop is bounded in practice: each iteration bumps run_index,
        # and unique filenames exist above any stale-index ceiling.
        # The only way this loops unboundedly is if ALL writes are
        # failing (permissions, full disk) — at which point the
        # subsequent write_text raises and the loop dies with a clear
        # error rather than spinning.
        while True:
            self._run_index += 1
            path = terminal_dir / f"{self._name}-{self._run_index}.log"
            if not path.exists():
                break
        path.write_text(full_output, encoding="utf-8")
        return str(_DISK_LOG_SUBDIR / path.name)

    @staticmethod
    def _truncate_head_tail(text: str, disk_path: str | None) -> str:
        """Reduce ``text`` to ``HEAD + marker + TAIL`` if over the inline cap.

        Head and tail are :data:`_HEAD_CHARS` / :data:`_TAIL_CHARS`
        characters respectively. Marker line names the number of
        truncated bytes and the disk path (if present) where the
        full output lives.

        :param text: Full (post-ANSI-strip) output text.
        :param disk_path: Workspace-relative path to the disk log,
            if one was written. ``None`` if truncation happened
            without a disk spill (shouldn't occur in practice — the
            caller always persists before truncating — but defended
            against for safety).
        :returns: The possibly-truncated output. If ``text`` was
            already within the cap, returns it unchanged.
        """
        if len(text) <= _INLINE_CAP_CHARS:
            return text
        head = text[:_HEAD_CHARS]
        tail = text[-_TAIL_CHARS:]
        truncated = len(text) - len(head) - len(tail)
        path_ref = f", full output at {disk_path}" if disk_path else ""
        marker = f"\n[... {truncated} bytes truncated{path_ref} ...]\n"
        return head + marker + tail

    def interrupt(self) -> None:
        """Send SIGINT (Ctrl-C) to the bash subprocess.

        Used by async-terminal ``cancel_task`` to stop a running
        command without killing the shell itself. If a command is
        currently executing inside ``run_sync``, bash delivers SIGINT
        to the foreground process; the command dies, the OSC 633 ``D``
        marker fires with exit code 130, and the in-flight ``run_sync``
        returns with ``status="killed"``. Bash itself survives and
        the shell can accept subsequent commands.

        Thread-safe: callable from any thread. Internally writes a
        single control byte to the PTY via ``pexpect.sendintr``,
        which is atomic at the kernel level. The caller does not
        need to hold the cmd lock — that's what makes this useful
        for cancel, where the command-runner thread already owns
        the lock.

        No-op if the shell is not currently alive (already exited
        or never spawned). Does not raise.
        """
        if self._proc.isalive():
            # Mark the interrupt BEFORE sending the signal so the
            # result builder can't race us: if the SIGINT reaches
            # bash's child and the D marker is read before we set
            # the flag, _run_locked would wrongly classify the
            # result as "completed".
            self._interrupt_signal.set()
            # Send SIGINT directly to bash's foreground child via
            # os.kill rather than via the PTY's VINTR. The PTY path
            # (``pexpect.sendintr``) is theoretically equivalent
            # (terminal driver sees VINTR → generates SIGINT for
            # the foreground PGID), but in practice —
            # specifically in non-interactive bash spawned under
            # pexpect — SIGINT sometimes kills bash itself rather
            # than the foreground child. Direct kill bypasses that
            # quirk: we enumerate bash's direct children via
            # ``pgrep -P`` and SIGINT each one. Bash sees the child
            # exit with signal, emits its D marker with exit 130.
            self._interrupt_children()

    def _interrupt_children(self) -> None:
        """SIGINT every direct child of the bash subprocess.

        Helper for :meth:`interrupt`. Lists children via ``pgrep -P``
        (portable across Linux and macOS), then sends SIGINT to each.
        No-op if bash has no children (idle at prompt). Exceptions
        from ``pgrep`` (e.g. not installed) or ``os.kill``
        (e.g. child exited between listing and kill) are swallowed —
        interrupt is best-effort; if the child is already gone,
        there's nothing to interrupt.

        Why not ``killpg``: bash spawned by pexpect doesn't always
        create a new process group for each foreground command, so
        killpg would go to bash's own group (including bash) and
        kill the shell.
        """
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(self._proc.pid)],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return
        for token in result.stdout.split():
            try:
                child_pid = int(token)
            except ValueError:
                continue
            try:
                os.kill(child_pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                # Child exited between pgrep and kill, or we don't
                # own it (shouldn't happen for our own bash). Either
                # way, nothing to interrupt; ignore and continue.
                continue

    def peek_partial_stdout(self, cursor: int) -> PartialReadResult:
        """Return stdout bytes produced since ``cursor``, ANSI-stripped.

        Used by ``check_task`` on ``kind="terminal"`` tasks to show
        the agent a "tail -f"-style delta of what the command has
        produced since the last poll. Does NOT advance any persistent
        cursor — the caller is responsible for storing the returned
        ``new_cursor`` and passing it on the next call. This keeps
        the Shell stateless with respect to observers and lets
        multiple callers poll independently if that's ever needed.

        Thread-safe: reads the ring buffer under its internal lock,
        so concurrent reads and the writer (the read loop in
        ``_read_until_done``) coexist safely. ANSI stripping happens
        on the delta bytes only — consistent with the finished-output
        path in ``_build_completed_or_killed_result``.

        :param cursor: Byte offset returned by a prior
            ``peek_partial_stdout`` call, or ``0`` on the first call.
            Negative values are clamped to 0.
        :returns: A :class:`PartialReadResult` with the ANSI-stripped
            text, the new cursor to pass next time, and any
            lost-byte count from eviction.
        """
        partial = self._ring.slice_since(cursor)
        # strip_ansi takes bytes and returns a decoded str, matching
        # the finished-output path in _build_completed_or_killed_result.
        # Using the same helper keeps the decoding semantics (UTF-8
        # with errors="replace" per the implementation) consistent
        # between sync and async terminal reads.
        text = strip_ansi(partial.data)
        return PartialReadResult(
            text=text,
            new_cursor=partial.new_cursor,
            lost_bytes=partial.lost_bytes,
        )

    def close(self) -> None:
        """Terminate the bash subprocess.

        Sends SIGHUP via pexpect's ``close(force=True)``, which
        escalates to SIGKILL if bash doesn't exit promptly. Safe to
        call multiple times; subsequent calls are no-ops because
        pexpect tracks the subprocess lifecycle.
        """
        if self._proc.isalive():
            self._proc.close(force=True)
