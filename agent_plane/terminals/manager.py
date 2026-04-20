"""Per-conversation shell manager.

A :class:`TerminalManager` owns the shells for one conversation. It
is the middle layer of the three-layer terminal stack described in
``designs/PERSISTENT_TERMINAL_RESEARCH.md``:

- Bottom: :class:`Shell` (one bash subprocess) — slices 1 + 2
- **Middle: TerminalManager** (per-conversation shell registry) — this slice
- Top: :class:`TerminalManagerRegistry` (server-wide, keyed by
  conversation_id) — also this slice

The manager enforces the per-conversation shell cap and validates
agent-provided shell names. It does NOT know about conversations or
the registry — it's just a named-shell collection. The registry (one
layer up) keys managers by conversation_id and wires cleanup.

Thread-safety: encapsulated. A `threading.Lock` on the instance
guards the `name → Shell` dict for atomic get-or-create / close. The
Shell's own `_cmd_lock` backs the per-shell ``shell_busy`` semantic.
Callers never see either lock.
"""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from agent_plane.terminals.shell import RunResult, Shell


@dataclasses.dataclass(frozen=True)
class TaskStdoutDelta:
    """Delta stdout read for an async terminal task.

    Returned by :meth:`TerminalManager.peek_task_stdout`. Named
    fields (rather than a tuple) so callers can access ``.text`` and
    ``.lost_bytes`` without positional fragility and so the shape
    is extensible if we add e.g. timestamps later.

    :param text: ANSI-stripped stdout bytes emitted by the command
        since the caller's last :meth:`peek_task_stdout` call.
        Empty string if no new output.
    :param lost_bytes: Count of bytes that were evicted from the
        shell's ring buffer before the caller could read them.
        Non-zero when the command produced bursty output exceeding
        the 1 MB ring's remaining capacity between peeks. Callers
        can surface this as a ``[... N bytes lost ...]`` marker.
    """

    text: str
    lost_bytes: int


# Agent-provided shell names. Ratified in §7.1 sub-decisions:
# - must start with a letter,
# - alphanumeric / underscore / hyphen only,
# - up to 64 characters total,
# - leading underscores are reserved for framework use.
#
# ``"default"`` is the implicit fallback when the agent omits a name
# and is always allowed (it matches the regex and is not
# framework-reserved).
_SHELL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

# Maximum shells per conversation — §6.4 ratified. Caps the blast
# radius of an agent bug that loops spawning new shells.
_MAX_SHELLS_PER_CONVERSATION = 10


class ShellNameInvalid(ValueError):
    """Raised when an agent-provided shell name fails the regex check.

    Kept as a distinct exception so tool-level code can translate it
    to a user-visible error rather than confusing it with the generic
    ``ValueError``.
    """


class ShellCapExceeded(RuntimeError):
    """Raised when a new shell would exceed the per-conversation cap.

    The agent must close an existing shell before creating another.
    """


class TerminalManager:
    """The set of shells belonging to one conversation.

    Created lazily by :class:`TerminalManagerRegistry` on first use
    of a conversation. Lifetime matches that of the conversation
    (modulo idle-timeout reaping and server-crash loss; see §6.4).

    All mutating methods are internally serialized by
    ``self._lock``. Callers don't interact with the lock.
    """

    def __init__(
        self,
        conversation_id: str,
        workspace: Path,
        *,
        sandbox_enabled: bool = True,
        on_empty: Callable[[str, TerminalManager], None] | None = None,
    ) -> None:
        """Construct an empty manager for one conversation.

        :param conversation_id: The owning conversation's id. Stored
            for diagnostics / logging only — the registry uses its
            own dict key, so the manager doesn't authoritatively own
            this identifier.
        :param workspace: Per-conversation workspace directory. All
            shells this manager creates will spawn with this as cwd
            and use it for disk-log overflow.
        :param sandbox_enabled: Whether to wrap newly-spawned shells
            in the srt sandbox. Propagated to :meth:`Shell.spawn` on
            every new shell this manager creates. Defaults to ``True``
            — disabling is a deployment decision surfaced through
            :class:`RuntimeCaps`, not an agent-visible switch.
        :param on_empty: Optional callback invoked with
            ``(conversation_id, self)`` once the manager drops to
            zero shells via :meth:`close`. The
            :class:`TerminalManagerRegistry` uses this to evict
            empty managers from its map — §6.4's "Empty manager"
            rule: a conversation that closes its last shell should
            not keep a zero-shell manager object pinned in the
            registry. The ``self`` reference lets the registry
            verify it's evicting the right object (guarding against
            a racing ``cleanup_conversation`` + recreate that would
            otherwise evict a fresh manager by mistake). Fired
            outside the manager's internal lock so the registry can
            safely acquire its own lock inside the callback without
            risking deadlock.
        """
        self._conversation_id = conversation_id
        self._workspace = workspace
        self._sandbox_enabled = sandbox_enabled
        self._on_empty = on_empty
        self._shells: dict[str, Shell] = {}
        self._lock = threading.Lock()
        # Monotonic wall-clock (perf_counter equivalent) for the idle
        # reaper — set to "now" whenever any activity happens.
        self._last_activity = time.monotonic()
        # Map of task_id → shell_name for async (``synchronous=False``)
        # commands currently running. ``cancel_task`` uses this to
        # find the shell to ``interrupt()``; ``check_task`` uses it
        # plus ``_task_cursors`` to produce "tail -f" deltas. Cleared
        # on command completion in ``unregister_running_task``.
        #
        # In-memory — lost on server crash. The task itself is durable
        # via DBOS, but live command state is not. A crashed workflow
        # re-executing the step will re-register (possibly on a new
        # shell if the old shell was also lost).
        self._task_shells: dict[str, str] = {}
        self._task_cursors: dict[str, int] = {}
        # Tasks for which ``cancel_task`` has been requested but the
        # step hasn't yet started the bash command. Set by
        # :meth:`request_cancel`; checked by
        # ``background_terminal_workflow`` before invoking
        # :meth:`run_sync`. Eliminates the race where a cancel
        # arrives after ``register_running_task`` but before the
        # child workflow's thread has actually sent the command
        # into bash.
        self._cancel_requested: set[str] = set()

    @property
    def conversation_id(self) -> str:
        """The owning conversation id.

        :returns: The id passed to :meth:`__init__`.
        """
        return self._conversation_id

    @property
    def last_activity_monotonic(self) -> float:
        """Monotonic-clock timestamp of the most recent activity.

        Used by the registry's idle reaper to identify abandoned
        conversations. "Activity" means any call into this manager
        that touches shells (:meth:`run_sync`, :meth:`close`,
        :meth:`close_all`, :meth:`list_shells`). Construction does
        NOT count as activity; the reaper treats a never-used manager
        as idle from the moment it's created.

        :returns: Seconds from an arbitrary monotonic epoch.
        """
        return self._last_activity

    def run_sync(
        self,
        shell_name: str,
        command: str,
        timeout_ms: int | None = None,
        cancel_predicate: Callable[[], bool] | None = None,
    ) -> RunResult:
        """Run a command in the named shell, creating the shell if needed.

        Validates the shell name, acquires the manager lock long
        enough to get-or-create the :class:`Shell`, releases the lock,
        then runs the command (which acquires the shell's own
        ``_cmd_lock``). Releasing the manager lock before running
        means other shells in the same conversation can run
        concurrently — only the shell-map mutation is serialized, not
        command execution.

        :param shell_name: Agent-chosen shell name. Must match the
            ``_SHELL_NAME_RE`` pattern (starts with letter, up to
            64 alphanumeric + ``_-`` chars). The implicit default
            (``"default"``) is just a name; same rules.
        :param command: Bash command text. See :meth:`Shell.run_sync`.
        :param timeout_ms: Task lifetime bound; see
            :meth:`Shell.run_sync`.
        :returns: A :class:`RunResult` from the underlying shell.
        :raises ShellNameInvalid: If ``shell_name`` doesn't match the
            allowed pattern.
        :raises ShellCapExceeded: If creating this shell would
            exceed the per-conversation cap and the shell doesn't
            already exist.
        """
        shell = self._get_or_create_shell(shell_name)
        self._last_activity = time.monotonic()
        return shell.run_sync(
            command,
            timeout_ms=timeout_ms,
            cancel_predicate=cancel_predicate,
        )

    def _get_or_create_shell(self, shell_name: str) -> Shell:
        """Look up or create the named shell. Thread-safe.

        **Contention note**: ``Shell.spawn`` (bash fork + OSC 633
        handshake, typically 100–500ms) runs while the manager lock
        is held. This blocks other threads in the same conversation
        that want a shell — even a different one — for the duration
        of the spawn. Acceptable in v1 because agent-plane serializes
        tasks per conversation, so concurrent manager access is rare.
        Phase 2 (async tool tasks) may introduce real concurrency;
        at that point, consider a "reserve-slot-and-spawn-outside-lock"
        refactor using a per-name event for coordination. For now,
        simpler is better.

        :param shell_name: See :meth:`run_sync`.
        :returns: The existing or newly-spawned :class:`Shell`.
        :raises ShellNameInvalid: Name regex fails.
        :raises ShellCapExceeded: At cap and shell doesn't exist.
        """
        if not _SHELL_NAME_RE.match(shell_name):
            # Regex enforces three properties at once: starts with a
            # letter (so ``_``-prefixed names — reserved for framework
            # use — are rejected), uses only the allowed character set,
            # and fits within 64 chars.
            raise ShellNameInvalid(
                f"Shell name {shell_name!r} is invalid. Names must start "
                "with a letter and contain only letters, digits, "
                "underscores, or hyphens (max 64 chars). "
                "Leading underscores are reserved for framework use."
            )

        with self._lock:
            existing = self._shells.get(shell_name)
            if existing is not None:
                # Quick reaping of dead shells: if bash crashed, the
                # agent would have seen ``shell_crashed`` on its
                # previous call. Subsequent calls get a fresh shell
                # under the same name.
                if not existing.is_alive():
                    self._shells.pop(shell_name, None)
                else:
                    return existing
            if len(self._shells) >= _MAX_SHELLS_PER_CONVERSATION:
                raise ShellCapExceeded(
                    f"Conversation has {len(self._shells)} shells already "
                    f"(cap: {_MAX_SHELLS_PER_CONVERSATION}). Close an "
                    "existing shell before creating another."
                )
            new_shell = Shell.spawn(
                shell_name,
                self._workspace,
                sandbox_enabled=self._sandbox_enabled,
            )
            self._shells[shell_name] = new_shell
            return new_shell

    def close(self, shell_name: str) -> bool:
        """Kill and remove a named shell, if present.

        If this close empties the manager, the ``on_empty`` callback
        (if any) is invoked with the manager's ``conversation_id``
        *outside* the manager lock, after the shell process is
        killed. The registry uses this to evict the now-empty
        manager from its map — see §6.4's "Empty manager" rule.

        :param shell_name: The shell to close.
        :returns: True if a shell was closed, False if no shell with
            that name existed. No name-validation error — closing
            an invalid/nonexistent name is idempotent.
        """
        self._last_activity = time.monotonic()
        with self._lock:
            shell = self._shells.pop(shell_name, None)
            became_empty = shell is not None and not self._shells
        if shell is None:
            return False
        shell.close()
        if became_empty and self._on_empty is not None:
            # Outside the manager lock so the callback can safely
            # grab the registry's lock without risking lock-order
            # inversion if some future code path calls back into
            # the manager.
            self._on_empty(self._conversation_id, self)
        return True

    def close_all(self) -> None:
        """Kill and remove every shell in the conversation.

        Called by the registry during ``cleanup_conversation`` and
        the idle reaper. Best-effort: individual shell-close failures
        don't prevent closing the rest.
        """
        self._last_activity = time.monotonic()
        with self._lock:
            shells = list(self._shells.values())
            self._shells.clear()
        for shell in shells:
            shell.close()

    def list_shells(self) -> list[str]:
        """Return the names of every shell currently in the manager.

        Order is insertion order (Python 3.7+ dict semantics). Dead
        shells are included if they haven't been swept yet — the
        sweep happens lazily on next :meth:`run_sync` for that name.
        Slice 4 extends this to return richer status info for
        ``terminal_list`` tool consumption.

        :returns: List of shell names.
        """
        with self._lock:
            return list(self._shells.keys())

    def has_shells(self) -> bool:
        """Whether the manager owns any shells right now.

        Used by the registry to decide whether to remove an empty
        manager from its map after a close — see §6.4 "Empty manager"
        rule.

        :returns: True if at least one shell is registered.
        """
        with self._lock:
            return bool(self._shells)

    def ensure_shell(self, shell_name: str) -> None:
        """Spawn ``shell_name`` if not already alive, without running
        a command.

        Used by ``TerminalRunTool.dispatch_async`` to pre-create the
        shell before returning a handle, so a racing ``cancel_task``
        can locate the shell via ``shell_for_task`` even if the
        child workflow's first ``run_sync`` hasn't started yet.

        Name validation and 10-shell cap enforcement run normally;
        exceptions propagate exactly as for ``run_sync`` so async
        dispatch fails loud if the agent tried to use a bad name
        or exceeded the cap.

        :param shell_name: The shell to spawn. Must pass the name
            regex (see :data:`_SHELL_NAME_RE`).
        :raises ShellNameInvalid: If the name fails the regex.
        :raises ShellCapExceeded: If spawning would exceed the
            per-conversation cap.
        """
        self._last_activity = time.monotonic()
        # _get_or_create_shell is the same code run_sync uses; all
        # validation and cap checks happen there.
        self._get_or_create_shell(shell_name)

    # ── Async task tracking (for synchronous=False commands) ─────

    def register_running_task(self, task_id: str, shell_name: str) -> None:
        """Record that ``task_id`` is running a command on ``shell_name``.

        Used by ``background_terminal_workflow`` before it invokes
        :meth:`run_sync` so subsequent ``cancel_task(task_id)`` calls
        can find the right shell to ``interrupt()``, and ``check_task``
        can read stdout deltas via :meth:`peek_task_stdout`.

        Idempotent: overwriting an existing registration resets the
        read cursor. The cursor position starts at 0 — the first
        ``peek_task_stdout`` call returns everything produced since
        the registration. Does NOT clear any prior cancel request —
        a cancel racing in before register_running_task should still
        be honored by the subsequent step.

        :param task_id: The background task's id, e.g.
            ``"resp_abc123"``.
        :param shell_name: The name of the shell the command is
            running on, e.g. ``"default"``.
        """
        with self._lock:
            self._task_shells[task_id] = shell_name
            self._task_cursors[task_id] = 0

    def request_cancel(self, task_id: str) -> bool:
        """Mark ``task_id`` as cancelled.

        Used by ``cancel_task(kind="terminal")`` to signal the
        background workflow even when the shell isn't running a
        command yet (pre-registration cancel race). The background
        workflow's step checks :meth:`is_cancel_requested` before
        calling ``run_sync`` and short-circuits to a ``killed`` result
        if set.

        Independent of ``shell.interrupt()``: the interrupt sends
        SIGINT to a currently-running command, this marks an
        intent to cancel that's durable across the sync-vs-async
        race.

        :param task_id: The background task's id.
        :returns: True iff the task was known (registered) at the
            time of the call. False when the task id isn't
            registered — the caller can use this to distinguish
            "marked for cancel" from "nothing to cancel."
        """
        with self._lock:
            if task_id not in self._task_shells:
                return False
            self._cancel_requested.add(task_id)
            return True

    def is_cancel_requested(self, task_id: str) -> bool:
        """Check whether :meth:`request_cancel` has been called for
        ``task_id``.

        :param task_id: The background task's id.
        :returns: True iff a cancel has been requested and not yet
            unregistered.
        """
        with self._lock:
            return task_id in self._cancel_requested

    def unregister_running_task(self, task_id: str) -> None:
        """Drop the task→shell mapping, cursor, and cancel flag for
        ``task_id``.

        Called by ``background_terminal_workflow`` in a ``finally``
        after ``run_sync`` returns (success, timeout, or crash).
        After this, ``cancel_task`` on the same id becomes a no-op —
        the command is already gone.

        :param task_id: The background task's id.
        """
        with self._lock:
            self._task_shells.pop(task_id, None)
            self._task_cursors.pop(task_id, None)
            self._cancel_requested.discard(task_id)

    def shell_for_task(self, task_id: str) -> Shell | None:
        """Look up the :class:`Shell` currently running ``task_id``.

        Used by ``cancel_task(kind="terminal")`` to find the shell
        whose PTY should receive SIGINT. Returns ``None`` if the
        task has already completed, was never registered, or targets
        a shell that has since been closed.

        :param task_id: The background task's id.
        :returns: The :class:`Shell` instance, or ``None``.
        """
        with self._lock:
            shell_name = self._task_shells.get(task_id)
            if shell_name is None:
                return None
            return self._shells.get(shell_name)

    def send_input_to_task(self, task_id: str, chars: str) -> Shell | None:
        """Write ``chars`` to the PTY of ``task_id``'s shell and return it.

        Used by ``terminal_send_input`` to drive interactive
        programs running under an async ``terminal_run``. Looks up
        the shell via the task-registration map
        (:meth:`register_running_task`) and forwards to
        :meth:`Shell.send_input`.

        Returning the :class:`Shell` rather than a boolean lets the
        caller hold a reference past the task's lifecycle — if the
        program exits during the caller's yield loop and
        ``_task_shells`` unregisters before the caller builds its
        response, the shell object itself is still valid for a
        final ``rendered_screen`` / ring-buffer read. The
        alternative (re-looking-up by task_id at response time)
        fails for fast interactions.

        :param task_id: The background task's id.
        :param chars: Bytes (as a string) to write. Empty strings
            are allowed — they short-circuit to a non-None return
            without touching the PTY so callers can use
            ``terminal_send_input(task_id, chars="")`` as a
            pure-poll while still requiring a registered shell.
        :returns: The :class:`Shell` if the task is registered and
            its shell exists; ``None`` if the task is not
            registered or its shell has been closed.
        """
        with self._lock:
            shell_name = self._task_shells.get(task_id)
            if shell_name is None:
                return None
            shell = self._shells.get(shell_name)
            if shell is None:
                return None
        # Send outside the manager lock — the PTY write is a syscall
        # and we don't want to block concurrent shell operations on
        # the same conversation while it blocks on kernel buffers.
        shell.send_input(chars)
        return shell

    def rendered_screen_for_task(self, task_id: str) -> str | None:
        """Return the pyte-rendered screen of ``task_id``'s shell.

        Used by ``check_task`` and ``terminal_send_input`` on
        terminal-kind tasks to give the LLM a view of what a full-
        screen program is currently displaying. Complementary to
        :meth:`peek_task_stdout`, which returns the streaming text
        delta.

        :param task_id: The background task's id.
        :returns: The rendered screen as ``\\n``-separated lines,
            or ``None`` if the task is unknown (completed,
            unregistered, or its shell was closed).
        """
        with self._lock:
            shell_name = self._task_shells.get(task_id)
            if shell_name is None:
                return None
            shell = self._shells.get(shell_name)
            if shell is None:
                return None
        # Render outside the manager lock — Shell has its own pyte
        # lock and manager-lock contention on conversation-level
        # ops should not block screen reads.
        return shell.rendered_screen()

    def total_bytes_written_for_task(self, task_id: str) -> int | None:
        """Return the monotonic total-bytes-written counter for the task's shell.

        Used by :class:`TerminalSendInputTool` for quiescence
        detection: the tool writes input, then polls this counter
        until it stops growing for a short window. Unlike
        :meth:`peek_task_stdout`, this does NOT advance the per-
        task cursor — that way the final ``peek_task_stdout`` call
        in the tool's response-build phase still sees every byte
        written during the wait.

        :param task_id: The background task's id.
        :returns: The ring buffer's cumulative
            ``total_appended`` counter for the task's shell, or
            ``None`` if the task / shell isn't resolvable.
        """
        with self._lock:
            shell_name = self._task_shells.get(task_id)
            if shell_name is None:
                return None
            shell = self._shells.get(shell_name)
            if shell is None:
                return None
        return shell.total_bytes_written()

    def peek_task_stdout(self, task_id: str) -> TaskStdoutDelta | None:
        """Return the stdout delta for ``task_id`` since the last peek.

        Used by ``check_task(kind="terminal")`` to produce
        ``recent_activity``. Reads from the underlying shell's ring
        buffer, ANSI-stripped, and advances the per-task cursor so
        a subsequent call returns only newer bytes.

        :param task_id: The background task's id.
        :returns: A :class:`TaskStdoutDelta` with the ANSI-stripped
            delta text and the count of evicted-before-peek bytes
            (non-zero when the ring buffer overflowed between peeks).
            Returns ``None`` if the task is unknown (completed,
            never started, or its shell was closed).
        """
        with self._lock:
            shell_name = self._task_shells.get(task_id)
            if shell_name is None:
                return None
            shell = self._shells.get(shell_name)
            if shell is None:
                return None
            cursor = self._task_cursors.get(task_id, 0)
        # Peek outside the manager lock — the Shell's ring buffer has
        # its own lock. Holding the manager lock during a potentially
        # non-trivial read would block concurrent shell creation /
        # close operations on the same conversation unnecessarily.
        partial = shell.peek_partial_stdout(cursor)
        with self._lock:
            # Only advance if the task is still registered; otherwise
            # we've raced with unregister_running_task and our read
            # is stale.
            if task_id in self._task_cursors:
                self._task_cursors[task_id] = partial.new_cursor
        return TaskStdoutDelta(
            text=partial.text,
            lost_bytes=partial.lost_bytes,
        )
