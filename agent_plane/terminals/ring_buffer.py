"""Byte-bounded FIFO buffer with eviction tracking.

Used by ``Shell`` to cap in-memory output capture per command at a
fixed byte budget (default 1 MB; see
``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.7). When a command
produces more bytes than the buffer can hold, older bytes are
evicted FIFO-style and the eviction count is tracked so callers can
surface a ``[... N bytes evicted ...]`` marker.

Kept as its own module (not an inner class of ``Shell``) so the
buffer is independently testable and reusable — the disk-persistence
path in slice 2 and any future render-screen path will want the same
primitive.

**Why not ``collections.deque(maxlen=N)``**: ``deque`` stores one
Python ``int`` per byte (each ~28 bytes in CPython), so a 1 MB cap
would cost ~28 MB. We want byte-bounded memory, not item-bounded.
``bytearray`` with slice-based eviction keeps 1 MB of output in
exactly 1 MB of memory plus a negligible overhead, and the slice
del is O(overflow_size) which is the minimum possible. No stdlib
or popular package provides this specific combination (byte-level
FIFO eviction + eviction counter), so a thin hand-rolled class is
justified.
"""

from __future__ import annotations

import dataclasses
import threading


@dataclasses.dataclass(frozen=True)
class PartialRead:
    """One delta read of a ring buffer since a caller-supplied cursor.

    :param data: Bytes appended between the caller's cursor and the
        buffer's current write position — truncated to what's still
        in the buffer after any eviction. Empty if no new bytes.
    :param new_cursor: Cursor value the caller should pass on the
        next read to resume where this one left off. Monotonically
        non-decreasing across successive reads.
    :param lost_bytes: Bytes that were written to the buffer between
        the caller's supplied cursor and ``data``'s start — i.e.
        bytes that were evicted before the caller could see them.
        Zero in the common case.
    """

    data: bytes
    new_cursor: int
    lost_bytes: int


class RingBuffer:
    """A byte-bounded FIFO buffer with monotonic append cursor.

    Thread-safe for concurrent ``append`` and ``slice_since``
    calls via an internal ``threading.Lock``. The cost is one
    uncontended lock acquire per read-loop tick (~4 Hz for Shell),
    far below the threshold where it would matter.

    Methods are deliberately minimal: ``append``, ``bytes``, ``reset``,
    ``evicted_bytes``, and ``slice_since`` for delta reads.
    """

    def __init__(self, capacity: int) -> None:
        """Construct an empty buffer with the given capacity.

        :param capacity: Maximum bytes to retain. Must be positive.
        :raises ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._data = bytearray()
        self._evicted = 0
        # Monotonically increasing count of bytes ever appended since
        # the last ``reset``. Used by ``slice_since`` to compute
        # deltas; not reset by eviction (only by ``reset``).
        self._total_appended = 0
        # Guards all reads and writes to ``_data``, ``_evicted``, and
        # ``_total_appended`` so the Shell's read-loop thread and an
        # observer (check_task) can interleave safely.
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        """Append bytes, evicting from the front if we exceed capacity.

        Eviction is FIFO: the oldest bytes fall off when the buffer
        overflows. Evicted byte count is tracked in
        :attr:`evicted_bytes` for the caller's eviction marker.

        :param chunk: Bytes to append. Empty chunks are a no-op.
        """
        if not chunk:
            return
        with self._lock:
            self._data.extend(chunk)
            self._total_appended += len(chunk)
            overflow = len(self._data) - self._capacity
            if overflow > 0:
                del self._data[:overflow]
                self._evicted += overflow

    def bytes(self) -> bytes:
        """Current buffered bytes (up to capacity).

        :returns: An immutable bytes copy of the current buffer
            contents. Does not clear the buffer.
        """
        with self._lock:
            return bytes(self._data)

    @property
    def evicted_bytes(self) -> int:
        """Total bytes evicted since the last :meth:`reset`.

        :returns: Count of bytes dropped due to overflow. Zero if
            the buffer never overflowed.
        """
        with self._lock:
            return self._evicted

    @property
    def total_appended(self) -> int:
        """Monotonic count of bytes appended since the last :meth:`reset`.

        Does NOT decrement on eviction — evicted bytes still count.
        Used as the ``new_cursor`` returned by :meth:`slice_since` so
        callers can resume reading from exactly where they left off.

        :returns: Total bytes appended since the last reset.
        """
        with self._lock:
            return self._total_appended

    def reset(self) -> None:
        """Clear buffer contents, eviction counter, and append cursor.

        Called by ``Shell`` at the start of each command so
        per-command captures are independent.
        """
        with self._lock:
            self._data.clear()
            self._evicted = 0
            self._total_appended = 0

    def slice_since(self, cursor: int) -> PartialRead:
        """Return bytes appended since ``cursor``, atomically.

        Use case: a caller (e.g. ``check_task``) polls the buffer
        while a command is running and wants to see only what's new
        since the previous poll. The returned ``new_cursor`` is the
        value to pass on the next call.

        Handles three cases:

        - ``cursor >= total_appended``: nothing new; returns empty
          ``data`` and unchanged ``new_cursor``.
        - Cursor's bytes are still in the buffer: returns bytes from
          ``cursor`` to ``total_appended`` exactly.
        - Cursor is behind the evicted range (some bytes were lost
          before the caller could read them): returns whatever is
          still in the buffer and reports ``lost_bytes > 0``.

        :param cursor: Byte offset returned by a prior
            :meth:`slice_since` call, or ``0`` on the first call.
            Negative values are clamped to 0.
        :returns: A :class:`PartialRead` with the delta bytes, new
            cursor, and any lost-byte count.
        """
        if cursor < 0:
            cursor = 0
        with self._lock:
            total = self._total_appended
            if cursor >= total:
                return PartialRead(data=b"", new_cursor=total, lost_bytes=0)
            # Oldest retained byte in the buffer has position
            # ``total - len(data)`` in the absolute stream.
            oldest_retained_pos = total - len(self._data)
            if cursor < oldest_retained_pos:
                lost = oldest_retained_pos - cursor
                return PartialRead(
                    data=bytes(self._data),
                    new_cursor=total,
                    lost_bytes=lost,
                )
            start = cursor - oldest_retained_pos
            return PartialRead(
                data=bytes(self._data[start:]),
                new_cursor=total,
                lost_bytes=0,
            )
