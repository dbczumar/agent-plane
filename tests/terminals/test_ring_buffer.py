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


# ── slice_since / PartialRead (delta reads) ────────────────────


def test_slice_since_empty_buffer_returns_empty_delta() -> None:
    """Fresh buffer: any cursor returns empty data with cursor unchanged.

    Proves the no-new-bytes fast path. If ``slice_since`` returned
    non-empty data here, ``check_task`` would echo stale / garbage
    bytes on the first poll.
    """
    from agent_plane.terminals.ring_buffer import PartialRead, RingBuffer

    buf = RingBuffer(capacity=10)
    p = buf.slice_since(0)
    assert isinstance(p, PartialRead)
    assert p.data == b""
    assert p.new_cursor == 0
    assert p.lost_bytes == 0


def test_slice_since_cursor_zero_returns_all() -> None:
    """Cursor=0 returns everything appended so far.

    First poll of a running command: caller has no prior cursor,
    passes 0, expects all bytes produced so far.
    """
    from agent_plane.terminals.ring_buffer import RingBuffer

    buf = RingBuffer(capacity=100)
    buf.append(b"hello ")
    buf.append(b"world")
    p = buf.slice_since(0)
    assert p.data == b"hello world"
    assert p.new_cursor == 11
    assert p.lost_bytes == 0


def test_slice_since_advances_cursor_across_calls() -> None:
    """Successive reads return only new bytes since the prior cursor.

    Core "tail -f" property: caller stores new_cursor, passes it
    next time, sees only the delta. If this broke (e.g. cursor
    never advanced), consecutive check_tasks would return the
    same bytes repeatedly.
    """
    from agent_plane.terminals.ring_buffer import RingBuffer

    buf = RingBuffer(capacity=100)
    buf.append(b"first ")
    p1 = buf.slice_since(0)
    assert p1.data == b"first "

    buf.append(b"second ")
    p2 = buf.slice_since(p1.new_cursor)
    assert p2.data == b"second "
    assert p2.new_cursor == p1.new_cursor + len(b"second ")

    # Third call, no new data: empty delta, cursor unchanged.
    p3 = buf.slice_since(p2.new_cursor)
    assert p3.data == b""
    assert p3.new_cursor == p2.new_cursor


def test_slice_since_cursor_past_total_is_safe() -> None:
    """A cursor beyond ``total_appended`` returns empty, not a slice bug.

    Defensive: if the caller somehow passes a cursor larger than
    total (shouldn't happen, but could on a crash + replay with
    stale state), we must not return garbage or raise IndexError.
    """
    from agent_plane.terminals.ring_buffer import RingBuffer

    buf = RingBuffer(capacity=100)
    buf.append(b"hi")
    p = buf.slice_since(1000)
    assert p.data == b""
    # new_cursor clamps to total — doesn't invent a value past it.
    assert p.new_cursor == buf.total_appended


def test_slice_since_reports_lost_bytes_on_eviction() -> None:
    """When the buffer evicted bytes the caller hadn't read yet,
    ``lost_bytes`` surfaces the count.

    Scenario: small buffer, bursty writer. If we didn't track this,
    the agent's recent_activity would look like a clean continuation
    when in fact some bytes were dropped. lost_bytes is how check_task
    tells the LLM "your stream has a gap."
    """
    from agent_plane.terminals.ring_buffer import RingBuffer

    buf = RingBuffer(capacity=5)
    buf.append(b"first_")  # 6 bytes → 1 evicted
    # Caller is at cursor=0 (before any data appended).
    # Buffer now has only last 5 bytes, and 1 was lost before cursor.
    p = buf.slice_since(0)
    assert p.lost_bytes == 1, (
        f"Expected 1 lost byte (buffer capacity 5, appended 6, cursor at 0), got {p.lost_bytes}."
    )
    assert p.data == b"irst_"
    assert p.new_cursor == 6


def test_slice_since_reset_resets_total_appended() -> None:
    """After ``reset``, ``total_appended`` drops to 0 and reads
    behave as on a fresh buffer.

    Critical for the sync terminal_run path which resets before
    every command. Without this, a cursor retained from a prior
    command would read wrong offsets into the next command's data.
    """
    from agent_plane.terminals.ring_buffer import RingBuffer

    buf = RingBuffer(capacity=100)
    buf.append(b"first command output")
    assert buf.total_appended == 20
    buf.reset()
    assert buf.total_appended == 0
    assert buf.bytes() == b""

    buf.append(b"new")
    p = buf.slice_since(0)
    assert p.data == b"new"
    assert p.new_cursor == 3


def test_slice_since_negative_cursor_clamps_to_zero() -> None:
    """Negative cursors are treated as 0 rather than crashing.

    Guards against a caller bug (e.g. subtracting before they
    have a prior cursor). Returning empty or raising IndexError
    would surface the bug elsewhere; clamping to 0 is the
    friendly, survivable choice.
    """
    from agent_plane.terminals.ring_buffer import RingBuffer

    buf = RingBuffer(capacity=100)
    buf.append(b"abc")
    p = buf.slice_since(-5)
    assert p.data == b"abc"
    assert p.new_cursor == 3
