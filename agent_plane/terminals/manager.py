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

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from agent_plane.terminals.shell import RunResult, Shell

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
        on_empty: Callable[[str, "TerminalManager"], None] | None = None,
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
        return shell.run_sync(command, timeout_ms=timeout_ms)

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
