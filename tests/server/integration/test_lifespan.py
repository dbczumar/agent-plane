"""Tests for the FastAPI app lifespan hook.

Exercises the ``_lifespan`` context manager in
``agent_plane.server.app`` to verify that terminal-registry
startup + shutdown wiring is correct. See
``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.4 + §6.9 for the
contract: on server shutdown, all shells must be killed and the
idle reaper must stop cleanly.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.asyncio


async def test_lifespan_shutdown_clears_every_manager(
    app: FastAPI,
    client: httpx.AsyncClient,  # noqa: ARG001 — ensures real DBOS is up
    tmp_path: Path,
) -> None:
    """After lifespan exit, the terminal registry has no managers left.

    This verifies the observable contract: every
    :class:`TerminalManager` that existed during the server's life
    is gone once the lifespan shutdown path runs. Failure mode this
    catches: the lifespan doesn't call ``registry.shutdown()``
    (managers stay → bash subprocesses leak across restarts in
    a real deploy).

    We exercise two conversations so both managers must be cleared,
    not just one (a partial-shutdown bug like "break on first
    exception" would pass with a single-manager test).
    """
    from agent_plane.runtime import get_terminal_registry

    registry = get_terminal_registry()

    async with app.router.lifespan_context(app):
        ws_a = tmp_path / "conv_a"
        ws_a.mkdir()
        ws_b = tmp_path / "conv_b"
        ws_b.mkdir()
        # Registering via ``for_conversation`` is the real API path
        # that tool invocations hit — so this exercises the full
        # lifecycle of a manager from creation to lifespan-driven
        # shutdown.
        registry.for_conversation("conv_a", ws_a)
        registry.for_conversation("conv_b", ws_b)
        assert set(registry.active_conversation_ids()) == {
            "conv_a",
            "conv_b",
        }

    # Out of the lifespan — shutdown hook must have cleared both.
    # If a manager remains, either shutdown wasn't called or it
    # stopped after processing the first entry (a partial-shutdown
    # bug). Either way, real servers would leak bash processes.
    assert registry.active_conversation_ids() == [], (
        f"Terminal registry still holds "
        f"{len(registry.active_conversation_ids())} manager(s) after "
        f"server shutdown: {registry.active_conversation_ids()}. "
        f"The lifespan's registry.shutdown() call is missing or was "
        f"short-circuited."
    )


async def test_lifespan_runs_reaper_start_and_stop(
    app: FastAPI,
    client: httpx.AsyncClient,  # noqa: ARG001 — ensures real DBOS is up
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan context calls ``start_reaper`` and ``stop_reaper``.

    Spies on the registry's reaper methods to prove the lifespan
    context actually invokes them. A missing ``start_reaper`` would
    mean idle managers never get GC'd on a long-running server;
    a missing ``stop_reaper`` would mean the reaper task leaks
    across server restarts.

    The spy wraps the real method so the reaper actually starts +
    stops — otherwise the lifespan exit would deadlock waiting for
    a task that was never created.
    """
    from agent_plane.runtime import get_terminal_registry

    registry = get_terminal_registry()

    start_calls = 0
    stop_calls = 0

    real_start = registry.start_reaper
    real_stop = registry.stop_reaper

    def spy_start(loop: object) -> None:
        nonlocal start_calls
        start_calls += 1
        real_start(loop)  # type: ignore[arg-type]

    async def spy_stop() -> None:
        nonlocal stop_calls
        stop_calls += 1
        await real_stop()

    monkeypatch.setattr(registry, "start_reaper", spy_start)
    monkeypatch.setattr(registry, "stop_reaper", spy_stop)

    async with app.router.lifespan_context(app):
        # Inside lifespan: start_reaper was called exactly once by
        # the startup hook. If 0, the hook doesn't wire start_reaper.
        # If >1, something is double-invoking the hook (not normal).
        assert start_calls == 1, (
            f"Expected start_reaper to be called exactly once during "
            f"lifespan startup, got {start_calls}. If 0, the "
            f"lifespan context doesn't call start_reaper."
        )
        # stop_reaper hasn't been called yet — only fires on shutdown.
        assert stop_calls == 0

    # After lifespan exit: stop_reaper was called exactly once.
    assert stop_calls == 1, (
        f"Expected stop_reaper to be called exactly once during "
        f"lifespan shutdown, got {stop_calls}. If 0, the reaper "
        f"task leaks across server lifetimes."
    )
