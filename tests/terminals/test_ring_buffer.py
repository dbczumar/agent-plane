"""Tests for the byte-bounded ring buffer.

Covers append, eviction, reset, and the evicted-byte counter.
"""

from __future__ import annotations

import pytest

from agent_plane.terminals.ring_buffer import RingBuffer


def test_empty_buffer_reads_as_empty_bytes() -> None:
    """A fresh buffer has no bytes and zero evictions."""
    buf = RingBuffer(capacity=10)
    assert buf.bytes() == b""
    assert buf.evicted_bytes == 0


def test_append_under_capacity_retains_all_bytes() -> None:
    """Appending less than capacity keeps everything; no eviction."""
    buf = RingBuffer(capacity=10)
    buf.append(b"hello")
    assert buf.bytes() == b"hello"
    assert buf.evicted_bytes == 0


def test_append_exactly_at_capacity_retains_all_bytes() -> None:
    """Exactly-capacity bytes fill without eviction."""
    buf = RingBuffer(capacity=5)
    buf.append(b"abcde")
    assert buf.bytes() == b"abcde"
    assert buf.evicted_bytes == 0


def test_append_over_capacity_evicts_fifo() -> None:
    """Over-capacity appends drop oldest bytes first.

    Capacity 5, append 8 → retains the last 5, evicts the first 3.
    """
    buf = RingBuffer(capacity=5)
    buf.append(b"12345678")
    assert buf.bytes() == b"45678"
    # 3 bytes (1,2,3) were evicted.
    assert buf.evicted_bytes == 3


def test_repeated_appends_cumulative_eviction() -> None:
    """Multiple appends that each push out old bytes accumulate the count."""
    buf = RingBuffer(capacity=3)
    buf.append(b"ab")  # buf=ab, no eviction yet
    buf.append(b"cde")  # buf=cde, 2 evicted (a,b)
    buf.append(b"fgh")  # buf=fgh, 3 more evicted (c,d,e) → 5 total
    assert buf.bytes() == b"fgh"
    # 2 bytes from first spillover + 3 from second.
    assert buf.evicted_bytes == 5


def test_empty_chunk_append_is_noop() -> None:
    """Appending empty bytes doesn't change the buffer."""
    buf = RingBuffer(capacity=5)
    buf.append(b"abc")
    buf.append(b"")
    assert buf.bytes() == b"abc"
    assert buf.evicted_bytes == 0


def test_reset_clears_data_and_counter() -> None:
    """reset() returns the buffer to its fresh state."""
    buf = RingBuffer(capacity=3)
    buf.append(b"abcdef")  # evicts some
    assert buf.evicted_bytes > 0
    buf.reset()
    assert buf.bytes() == b""
    # Eviction counter explicitly zeroed — verifies reset() doesn't
    # leak a growing counter across command boundaries.
    assert buf.evicted_bytes == 0


def test_bytes_returns_snapshot_not_live_view() -> None:
    """bytes() returns a snapshot; later appends don't mutate it.

    Protects callers against accidental sharing that could corrupt
    a cached snapshot. ``bytes`` is already an immutable type, but
    verifies we're not returning the underlying bytearray or a view.
    """
    buf = RingBuffer(capacity=20)
    buf.append(b"hello")
    snapshot = buf.bytes()
    buf.append(b" world")
    # Snapshot taken before the second append stays "hello".
    assert snapshot == b"hello"
    # Current buffer has the combined content (fits in capacity).
    assert buf.bytes() == b"hello world"


def test_zero_capacity_rejected() -> None:
    """Zero capacity is a logical error, not a degenerate valid buffer."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        RingBuffer(capacity=0)


def test_negative_capacity_rejected() -> None:
    """Negative capacity is a logical error."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        RingBuffer(capacity=-1)


def test_single_giant_append_fits_in_capacity() -> None:
    """A single append larger than capacity is handled correctly.

    The buffer retains exactly the last `capacity` bytes and reports
    the evicted count. This path can happen for a single command
    that produces 2 MB of output in one read chunk.
    """
    buf = RingBuffer(capacity=100)
    # 1000 distinct bytes so we can verify which ones remain.
    data = bytes(i % 256 for i in range(1000))
    buf.append(data)
    assert buf.bytes() == data[-100:]
    # 900 bytes were evicted: everything before the last 100.
    assert buf.evicted_bytes == 900
