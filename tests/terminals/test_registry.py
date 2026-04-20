"""Tests for :class:`TerminalManagerRegistry`.

Registry-layer unit tests: per-conversation manager lifecycle,
cleanup, shutdown, concurrent for_conversation atomicity, idle
reaper behavior.

Tests construct the registry with a test-controlled
``workspace_resolver`` that returns tmp paths. No real server or
store is needed.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from agent_plane.terminals.registry import TerminalManagerRegistry


@pytest.fixture
def registry() -> TerminalManagerRegistry:
    """A fresh, empty registry with sandboxing disabled.

    Sandbox is off for registry-layer tests — the registry itself
    doesn't care about the sandbox; it only forwards the flag to
    newly-created managers. Default-off avoids slowing these tests
    down with srt/node invocations.

    :returns: An empty :class:`TerminalManagerRegistry` with
        ``sandbox_enabled=False``.
    """
    return TerminalManagerRegistry(sandbox_enabled=False)


def _ws(tmp_path: Path, conv_id: str) -> Path:
    """Create and return the workspace dir for ``conv_id`` under ``tmp_path``."""
    p = tmp_path / conv_id
    p.mkdir(exist_ok=True)
    return p


# ---- basic lifecycle ------------------------------------------


def test_for_conversation_creates_on_first_call(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """First lookup creates a new manager bound to the passed workspace."""
    mgr = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    assert mgr.conversation_id == "conv_a"
    # Registry records the conversation as active.
    assert registry.active_conversation_ids() == ["conv_a"]


def test_for_conversation_is_idempotent(registry: TerminalManagerRegistry, tmp_path: Path) -> None:
    """Second call with the same id returns the same manager instance.

    This is the core property — conversation-scoped state only
    persists across calls if the same manager is handed out each
    time. ``is`` identity check, not equality.
    """
    ws = _ws(tmp_path, "conv_a")
    first = registry.for_conversation("conv_a", ws)
    second = registry.for_conversation("conv_a", ws)
    assert first is second


def test_distinct_ids_get_distinct_managers(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """Different conversation ids get different managers (isolation)."""
    a = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    b = registry.for_conversation("conv_b", _ws(tmp_path, "conv_b"))
    assert a is not b
    # And both appear in the active list.
    assert set(registry.active_conversation_ids()) == {"conv_a", "conv_b"}


# ---- cleanup_conversation -------------------------------------


def test_cleanup_conversation_removes_manager(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """cleanup removes the manager from the registry.

    After cleanup, a subsequent for_conversation lookup creates a
    fresh manager rather than returning the cleaned-up one.
    """
    ws = _ws(tmp_path, "conv_a")
    first = registry.for_conversation("conv_a", ws)
    registry.cleanup_conversation("conv_a")
    assert registry.active_conversation_ids() == []

    # Next lookup creates a new instance, not the cleaned-up one.
    second = registry.for_conversation("conv_a", ws)
    assert second is not first


def test_cleanup_conversation_closes_shells(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """cleanup also kills any shells the manager was holding.

    If cleanup only removed the registry entry without closing
    shells, bash processes would be orphaned.
    """
    mgr = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    mgr.run_sync("main", "echo hi")
    assert mgr.has_shells() is True

    registry.cleanup_conversation("conv_a")
    # Manager is gone from the registry.
    assert registry.active_conversation_ids() == []
    # The manager object we still hold a reference to has no shells.
    assert mgr.has_shells() is False


def test_cleanup_unknown_conversation_is_noop(
    registry: TerminalManagerRegistry,
) -> None:
    """Cleaning up a conversation with no manager silently succeeds.

    Common case: conversations that never used the terminal tool
    get DELETE'd just like any other — the cleanup hook must not
    error on them.
    """
    registry.cleanup_conversation("never-existed")  # no exception


# ---- empty-manager auto-eviction (§6.4 "Empty manager" rule) ---


def test_closing_last_shell_evicts_manager_from_registry(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """When ``close`` drops a manager to zero shells, registry drops it.

    §6.4's "Empty manager" rule: agents that create, use, then close
    a shell shouldn't leave a zero-shell manager object pinned in
    the registry forever — that's a memory-growth bug on busy
    servers that cycle through many one-off conversations.

    Failure mode this catches: manager.close() doesn't notify the
    registry, or the registry callback isn't wired up. A leftover
    manager would show up in active_conversation_ids even after
    its only shell was closed.
    """
    mgr = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    mgr.run_sync("main", "echo hi")
    assert registry.active_conversation_ids() == ["conv_a"]

    closed = mgr.close("main")
    assert closed is True
    # Registry should have evicted the now-empty manager.
    assert registry.active_conversation_ids() == [], (
        "Manager still in registry after its last shell was closed. "
        "The on_empty callback is missing or not wired."
    )


def test_closing_non_last_shell_does_not_evict(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """Closing one of several shells keeps the manager registered.

    Regression guard: the eviction must fire ONLY on the zero-shell
    transition, not on every close. If eviction fires on any close,
    closing shell 'a' while 'b' is still live would orphan the
    manager holding 'b' — a severe bug.
    """
    mgr = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    mgr.run_sync("alpha", "echo a")
    mgr.run_sync("beta", "echo b")
    assert registry.active_conversation_ids() == ["conv_a"]

    # Closing one of two shells — manager still has another one.
    mgr.close("alpha")
    assert registry.active_conversation_ids() == ["conv_a"], (
        "Manager was evicted after closing one of two shells. "
        "Eviction should fire only when the manager drops to ZERO."
    )
    # Now close the second — manager evicts.
    mgr.close("beta")
    assert registry.active_conversation_ids() == []


def test_next_lookup_after_eviction_creates_fresh_manager(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """After auto-eviction, ``for_conversation`` creates a new manager.

    Confirms the eviction doesn't block re-use of the conversation
    id — a subsequent terminal tool invocation must still work. The
    new manager is a distinct object (identity check) from the
    evicted one.
    """
    first = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    first.run_sync("main", "echo hi")
    first.close("main")
    # Manager is gone.
    assert registry.active_conversation_ids() == []

    # Lookup brings up a fresh manager (not the old one).
    second = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    assert second is not first, (
        "Expected a fresh manager after eviction, got the evicted one. "
        "The registry is handing out stale entries."
    )


def test_close_no_op_does_not_trigger_eviction(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """Closing a shell that doesn't exist doesn't evict the manager.

    A shell that was never created (or was already closed) shouldn't
    bump eviction logic — close() returns False in that case, and
    the manager's shell count didn't change. Regression guard for
    a bug where eviction fires on every close() call regardless of
    whether a shell was actually removed.
    """
    mgr = registry.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    mgr.run_sync("main", "echo hi")
    # Attempt to close a non-existent name — no state change.
    assert mgr.close("never-existed") is False
    # Manager still registered and still holds "main".
    assert registry.active_conversation_ids() == ["conv_a"]
    assert mgr.list_shells() == ["main"]


# ---- shutdown ------------------------------------------------


def test_shutdown_closes_every_manager(registry: TerminalManagerRegistry, tmp_path: Path) -> None:
    """shutdown() tears down every manager and their shells."""
    for conv_id in ["a", "b", "c"]:
        mgr = registry.for_conversation(conv_id, _ws(tmp_path, conv_id))
        mgr.run_sync("main", "echo hi")

    registry.shutdown()
    # Nothing left in the registry.
    assert registry.active_conversation_ids() == []


# ---- concurrent for_conversation -----------------------------


def test_concurrent_for_conversation_returns_same_instance(
    registry: TerminalManagerRegistry, tmp_path: Path
) -> None:
    """Many threads racing to for_conversation(same id) get one manager.

    If the check-then-create pattern isn't atomic, two threads
    could both create a manager and one would be leaked (with its
    workspace pointing at the same dir, leading to confused shell
    lookups). The registry's lock must make this atomic.
    """
    ws = _ws(tmp_path, "conv_shared")
    collected: list[object] = []
    errors: list[BaseException] = []
    start_event = threading.Event()

    def worker() -> None:
        # Wait for the release signal so all threads race into
        # for_conversation at roughly the same instant.
        start_event.wait()
        try:
            mgr = registry.for_conversation("conv_shared", ws)
            collected.append(mgr)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    start_event.set()
    for t in threads:
        t.join()

    assert errors == [], f"Worker threads errored: {errors}"
    # All 10 worker threads returned a manager; none exited via the
    # errors path. If this is <10, some workers either crashed or
    # got stuck — both would indicate a bug in for_conversation's
    # locking.
    assert len(collected) == 10
    # All 10 got the SAME manager instance. If any `is not first`,
    # the check-then-create pattern isn't atomic — two threads
    # both reached the "missing, create" branch and allocated
    # separate managers, leaking one.
    first = collected[0]
    for m in collected[1:]:
        assert m is first


# ---- idle reaper (sync path — _reap_idle_once) ----------------


def test_reaper_skips_active_managers(tmp_path: Path) -> None:
    """A manager with recent activity is NOT reaped.

    Directly call _reap_idle_once with a short-threshold registry
    to test the sweep logic without asyncio.
    """
    reg = TerminalManagerRegistry(sandbox_enabled=False, idle_timeout_s=60.0)
    mgr = reg.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    # Manager was just created; last_activity is fresh.
    reg._reap_idle_once()
    # Still there.
    assert reg.active_conversation_ids() == ["conv_a"]
    assert mgr.has_shells() is False  # (never spawned any)


def test_reaper_collects_idle_managers(tmp_path: Path) -> None:
    """A manager older than the threshold is reaped and its shells closed.

    Controls the test by using a zero-threshold registry — every
    manager is immediately "idle." Confirms the sweep actually
    removes entries and calls close_all.
    """
    reg = TerminalManagerRegistry(sandbox_enabled=False, idle_timeout_s=0.0)
    mgr = reg.for_conversation("conv_a", _ws(tmp_path, "conv_a"))
    mgr.run_sync("main", "echo hi")
    assert mgr.has_shells() is True

    reg._reap_idle_once()
    # Registry entry removed.
    assert reg.active_conversation_ids() == []
    # Its shells were closed as part of close_all.
    assert mgr.has_shells() is False


# ---- idle reaper (async path — start/stop_reaper) ------------


@pytest.mark.asyncio
async def test_reaper_task_starts_and_stops_cleanly() -> None:
    """start_reaper schedules the task; stop_reaper awaits its exit.

    Failure mode this catches: the reaper task is never scheduled,
    or stop_reaper doesn't actually stop it (leaking a background
    task into the event loop).
    """
    reg = TerminalManagerRegistry(reaper_interval_s=30.0)
    loop = asyncio.get_running_loop()
    reg.start_reaper(loop)
    # Give the loop a chance to actually start the task.
    await asyncio.sleep(0)
    assert reg._reaper_task is not None
    assert not reg._reaper_task.done()

    await reg.stop_reaper()
    # After stop_reaper returns, the task is gone.
    assert reg._reaper_task is None


@pytest.mark.asyncio
async def test_start_reaper_is_idempotent() -> None:
    """Calling start_reaper twice doesn't spawn two reaper tasks."""
    reg = TerminalManagerRegistry(reaper_interval_s=30.0)
    loop = asyncio.get_running_loop()
    reg.start_reaper(loop)
    first_task = reg._reaper_task
    reg.start_reaper(loop)  # second call — no-op
    assert reg._reaper_task is first_task

    await reg.stop_reaper()


@pytest.mark.asyncio
async def test_stop_reaper_without_start_is_noop() -> None:
    """Calling stop_reaper when the reaper isn't running doesn't raise."""
    reg = TerminalManagerRegistry()
    await reg.stop_reaper()  # no exception
