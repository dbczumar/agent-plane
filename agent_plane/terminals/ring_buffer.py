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


class RingBuffer:
    """A byte-bounded FIFO buffer.

    Not thread-safe on its own — the owning ``Shell`` serializes
    access via its per-shell command mutex (see §6.9). Exposed
    methods are deliberately minimal: ``append``, ``bytes``,
    ``reset``, plus ``evicted_bytes``.
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

    def append(self, chunk: bytes) -> None:
        """Append bytes, evicting from the front if we exceed capacity.

        Eviction is FIFO: the oldest bytes fall off when the buffer
        overflows. Evicted byte count is tracked in
        :attr:`evicted_bytes` for the caller's eviction marker.

        :param chunk: Bytes to append. Empty chunks are a no-op.
        """
        if not chunk:
            return
        self._data.extend(chunk)
        overflow = len(self._data) - self._capacity
        if overflow > 0:
            del self._data[:overflow]
            self._evicted += overflow

    def bytes(self) -> bytes:
        """Current buffered bytes (up to capacity).

        :returns: An immutable bytes copy of the current buffer
            contents. Does not clear the buffer.
        """
        return bytes(self._data)

    @property
    def evicted_bytes(self) -> int:
        """Total bytes evicted since the last :meth:`reset`.

        :returns: Count of bytes dropped due to overflow. Zero if
            the buffer never overflowed.
        """
        return self._evicted

    def reset(self) -> None:
        """Clear buffer contents and the eviction counter.

        Called by ``Shell`` at the start of each command so
        per-command captures are independent.
        """
        self._data.clear()
        self._evicted = 0
