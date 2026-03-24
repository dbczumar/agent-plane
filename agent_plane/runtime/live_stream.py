"""In-process live stream for real-time SSE token delivery.

Bypasses DBOS's ~1s database polling by bridging the sync workflow
thread and the async FastAPI event loop via asyncio.Queue.

Producer (workflow thread, sync):
    publish(task_id, event)   — thread-safe push onto the queue
    close(task_id)            — signals end-of-stream via sentinel

Consumer (SSE endpoint, async):
    subscribe(task_id) -> AsyncIterator  — yields events in real-time
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

# Sentinel object that signals end-of-stream.
_DONE = object()

# Global registry: task_id -> (queue, event_loop).
# The event loop reference is needed so the sync producer thread can
# safely enqueue items via call_soon_threadsafe.
_streams: dict[str, tuple[asyncio.Queue[dict[str, Any] | object], asyncio.AbstractEventLoop]] = {}
_lock = threading.Lock()


def register(task_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """
    Create a live stream for a task. Called by the SSE endpoint
    before the workflow starts producing events.
    """
    with _lock:
        if task_id not in _streams:
            # maxsize=0 means unbounded — tokens are tiny and
            # the consumer drains faster than the LLM produces
            _streams[task_id] = (asyncio.Queue(), loop)


def unregister(task_id: str) -> None:
    """Remove the live stream for a task. Called on SSE cleanup."""
    with _lock:
        _streams.pop(task_id, None)


def is_registered(task_id: str) -> bool:
    """Check if a live stream exists for this task."""
    with _lock:
        return task_id in _streams


def publish(task_id: str, event: dict[str, Any]) -> None:
    """
    Push an event to the live stream (called from sync workflow thread).
    No-op if no subscriber is registered — events still go to DBOS
    for durability regardless.
    """
    with _lock:
        entry = _streams.get(task_id)
    if entry is None:
        return
    queue, loop = entry
    loop.call_soon_threadsafe(queue.put_nowait, event)


def close(task_id: str) -> None:
    """
    Signal end-of-stream (called from sync workflow thread).
    The consumer will stop iterating after receiving the sentinel.
    """
    with _lock:
        entry = _streams.get(task_id)
    if entry is None:
        return
    queue, loop = entry
    loop.call_soon_threadsafe(queue.put_nowait, _DONE)


async def subscribe(task_id: str) -> AsyncIterator[dict[str, Any]]:
    """
    Yield events in real-time as the workflow produces them.
    Ends when the workflow calls close(). Must be called from
    the same event loop passed to register().
    """
    with _lock:
        entry = _streams.get(task_id)
    if entry is None:
        return
    queue = entry[0]
    while True:
        item = await queue.get()
        if item is _DONE:
            return
        # Type is narrowed: _DONE is the only non-dict sentinel
        assert isinstance(item, dict)
        yield item
