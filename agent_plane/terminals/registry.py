"""Server-resident registry of per-conversation terminal managers.

Top of the three-layer terminal stack (see
``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.4 and §6.9):

- Bottom: :class:`Shell` — slices 1 + 2
- Middle: :class:`TerminalManager` — this slice
- **Top: TerminalManagerRegistry** — this module

One registry exists per server process. It maps ``conversation_id``
to :class:`TerminalManager`, enforces the encapsulation contract
(only the accessor and method APIs are public; all locks, state,
and coordination live inside), runs the idle reaper, and wires
conversation-delete cleanup.

**This introduces a new pattern for agent-plane.** Prior server-level
state is either per-workflow (``ContextVar``, e.g. ``ToolManager``)
or initialized-once-then-read-only (stores, ``AgentCache``). A
conversation-scoped, actively-mutated server-resident registry did
not exist before. Future contributors: DO NOT "fix" this to use the
ContextVar pattern — conversation-scoped persistence across turns
is the whole point. See §6.4.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

from agent_plane.terminals.manager import TerminalManager

logger = logging.getLogger(__name__)

# Default idle-timeout — §7.1 ratified at 24 hours. Conversations
# with no terminal-activity within this window get their shells
# killed and their managers removed from the registry. Next terminal
# activity on that conversation spawns fresh shells.
_DEFAULT_IDLE_TIMEOUT_S = 24 * 60 * 60

# How often the reaper wakes to scan for idle managers. 10 minutes
# balances responsiveness against overhead. A manager that becomes
# idle just after a reaper sweep survives another ~10 minutes before
# it's collected, which is acceptable.
_DEFAULT_REAPER_INTERVAL_S = 10 * 60


class TerminalManagerRegistry:
    """The single registry of per-conversation shell managers.

    Instances are constructed once at server startup (from
    ``agent_plane.runtime._globals.init``) and accessed via the
    ``get_terminal_registry()`` accessor in ``agent_plane.runtime``.

    Callers interact through three methods:

    - :meth:`for_conversation` — get-or-create a
      :class:`TerminalManager` for a given conversation id.
    - :meth:`cleanup_conversation` — kill all shells for a
      conversation and remove its manager. Called synchronously from
      the conversation-delete route.
    - :meth:`shutdown` — kill everything, called on server shutdown.

    Idle reaping (:meth:`start_reaper` / :meth:`stop_reaper`) is
    separate because it requires an asyncio event loop. In tests, the
    reaper is typically not started. In production, the FastAPI
    lifespan hook calls ``start_reaper(asyncio.get_running_loop())``
    after the server is up.

    All concurrency primitives are instance members. No module-level
    locks, no exported dicts, no caller-held locks. See §6.9.
    """

    def __init__(
        self,
        *,
        sandbox_enabled: bool = True,
        idle_timeout_s: float = _DEFAULT_IDLE_TIMEOUT_S,
        reaper_interval_s: float = _DEFAULT_REAPER_INTERVAL_S,
    ) -> None:
        """Construct an empty registry.

        :param sandbox_enabled: Whether shells spawned through this
            registry get wrapped in the srt sandbox. Propagated to
            each :class:`TerminalManager` at creation time, which
            propagates it to every :meth:`Shell.spawn`. Default
            ``True`` — an operator-configured policy from
            :class:`RuntimeCaps`.
        :param idle_timeout_s: Seconds a manager can be inactive
            before the reaper (if running) collects it. Default
            24 hours per §7.1.
        :param reaper_interval_s: Seconds between reaper sweeps.
            Default 10 minutes — a compromise between responsiveness
            and wake overhead.
        """
        self._managers: dict[str, TerminalManager] = {}
        self._lock = threading.Lock()
        self._sandbox_enabled = sandbox_enabled
        self._idle_timeout_s = idle_timeout_s
        self._reaper_interval_s = reaper_interval_s
        self._reaper_task: asyncio.Task[None] | None = None
        self._reaper_stop_event: asyncio.Event | None = None

    def for_conversation(
        self,
        conversation_id: str,
        workspace: Path,
    ) -> TerminalManager:
        """Return the manager for ``conversation_id``, creating if absent.

        Atomic get-or-create under the registry lock. Callers get
        back a manager and never need to coordinate with other
        concurrent callers — the lock is not exposed.

        The ``workspace`` parameter is used only when creating a new
        manager. On cache hits (existing manager), it is ignored —
        the previously-stored workspace is kept. In practice the
        workspace for a given conversation is stable in agent-plane
        (derived deterministically from conversation id + agent
        name), so mismatches are a caller bug; we silently prefer
        the stored workspace rather than failing loud, because the
        cache-hit case is the hot path and we don't want to pay for
        a check on every call.

        :param conversation_id: The conversation's id.
        :param workspace: The workspace directory for shells in this
            conversation. Must be supplied on every call because
            the caller (the terminal tool) has the workspace handy
            in its :class:`ToolContext` — requiring it avoids
            injecting a store-shaped dependency into the registry.
        :returns: The existing manager if one exists, otherwise a
            fresh manager bound to ``workspace``.
        """
        with self._lock:
            existing = self._managers.get(conversation_id)
            if existing is not None:
                return existing
            new_manager = TerminalManager(
                conversation_id,
                workspace,
                sandbox_enabled=self._sandbox_enabled,
                on_empty=self._on_manager_empty,
            )
            self._managers[conversation_id] = new_manager
            return new_manager

    def _on_manager_empty(
        self, conversation_id: str, mgr: TerminalManager
    ) -> None:
        """Evict a manager that has dropped to zero shells.

        Called (from the manager, outside its own lock) after the
        manager's last shell was closed via :meth:`TerminalManager
        .close`. §6.4's "Empty manager" rule keeps the registry
        from growing unboundedly as agents create and close shells
        across conversations. Idempotent — if the manager was
        already evicted by another path (cleanup, reaper, shutdown),
        this is a no-op.

        Two race guards:
        - ``mgr is not self._managers.get(conversation_id)``: the
          slot has already been replaced by a fresh manager (e.g.
          ``cleanup_conversation`` + recreate). Evicting the fresh
          one by mistake would orphan its shells. Skip.
        - ``mgr.has_shells()``: after the zero-shell callback fired,
          someone spawned a new shell in the same manager before
          the registry lock was acquired. The manager is still
          useful; leave it in place.

        :param conversation_id: The emptied manager's conversation id.
        :param mgr: The specific manager instance that emptied.
            Identity-compared against the registry's current entry
            to rule out the "slot replaced" race above.
        """
        with self._lock:
            current = self._managers.get(conversation_id)
            if current is None or current is not mgr:
                return
            if mgr.has_shells():
                return
            self._managers.pop(conversation_id, None)

    def cleanup_conversation(self, conversation_id: str) -> None:
        """Kill all shells for ``conversation_id`` and drop its manager.

        Called synchronously from ``DELETE /conversations/{id}``
        between task deletion and conversation-row deletion (see
        §6.9). No-op if no manager exists for the id (common case:
        conversations that never used the terminal tool).

        The manager's ``close_all`` happens outside the registry
        lock so slow shell teardown doesn't block other conversations.

        :param conversation_id: The conversation being deleted.
        """
        with self._lock:
            mgr = self._managers.pop(conversation_id, None)
        if mgr is not None:
            mgr.close_all()

    def shutdown(self) -> None:
        """Tear down every manager; used on server shutdown.

        Synchronous — any asyncio reaper must be stopped separately
        via :meth:`stop_reaper` from the event loop. Typical FastAPI
        lifespan shutdown pattern: await ``stop_reaper``, then call
        ``shutdown``.
        """
        with self._lock:
            managers = list(self._managers.values())
            self._managers.clear()
        for mgr in managers:
            mgr.close_all()

    def active_conversation_ids(self) -> list[str]:
        """Return ids of conversations with a live manager.

        Used by tests and the reaper. Snapshot semantics — the list
        reflects the state at call time; it doesn't track changes.

        :returns: List of conversation ids currently in the registry.
        """
        with self._lock:
            return list(self._managers.keys())

    # ── Idle reaper ───────────────────────────────────────────────

    def start_reaper(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the periodic idle-manager sweep on ``loop``.

        The reaper wakes every ``reaper_interval_s``, checks each
        manager's ``last_activity_monotonic``, and cleans up any
        that have been idle longer than ``idle_timeout_s``.

        Must be called from the target event loop's thread, or use
        ``loop.call_soon_threadsafe`` if calling from elsewhere.

        Idempotent: calling twice is a no-op (second call is ignored).

        :param loop: The asyncio event loop to run the reaper on.
            In production this is the FastAPI server's loop.
        """
        if self._reaper_task is not None:
            return
        self._reaper_stop_event = asyncio.Event()
        self._reaper_task = loop.create_task(self._reaper_loop())

    async def stop_reaper(self) -> None:
        """Signal the reaper to stop and await its exit.

        Safe to call when the reaper isn't running (no-op).
        Called from the server's lifespan shutdown before
        :meth:`shutdown`.
        """
        if self._reaper_task is None:
            return
        assert self._reaper_stop_event is not None
        self._reaper_stop_event.set()
        try:
            await self._reaper_task
        except asyncio.CancelledError:
            pass
        self._reaper_task = None
        self._reaper_stop_event = None

    async def _reaper_loop(self) -> None:
        """The reaper's periodic loop.

        Wakes every ``reaper_interval_s`` (or earlier if
        ``_reaper_stop_event`` is set), collects idle managers,
        and calls their ``close_all``. Runs until stop_event fires.
        """
        assert self._reaper_stop_event is not None
        while not self._reaper_stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._reaper_stop_event.wait(),
                    timeout=self._reaper_interval_s,
                )
                # Stop event fired — exit cleanly.
                return
            except asyncio.TimeoutError:
                # Normal wake — do a sweep.
                try:
                    self._reap_idle_once()
                except Exception:
                    # A reaper crash must not kill the loop; log and
                    # continue. The next sweep will retry.
                    logger.exception("Idle reaper sweep failed")

    def _reap_idle_once(self) -> None:
        """Perform one idle-manager sweep.

        Acquires the registry lock briefly to identify and pop idle
        managers, then closes them outside the lock (same pattern as
        :meth:`cleanup_conversation` — slow shell teardown must not
        block other registry operations).
        """
        now = time.monotonic()
        deadline = now - self._idle_timeout_s
        victim_ids: list[str] = []
        victim_managers: list[TerminalManager] = []
        with self._lock:
            for conv_id, mgr in list(self._managers.items()):
                if mgr.last_activity_monotonic < deadline:
                    victim_ids.append(conv_id)
                    victim_managers.append(mgr)
                    self._managers.pop(conv_id, None)
        if victim_ids:
            logger.info(
                "Idle-reaped %d terminal manager(s): %s",
                len(victim_ids),
                victim_ids,
            )
            for mgr in victim_managers:
                mgr.close_all()
