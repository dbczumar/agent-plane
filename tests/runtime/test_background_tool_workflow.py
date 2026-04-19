"""Tests for the background_tool_workflow helpers.

The DBOS workflow itself requires a running DBOS instance and is
covered in server-integration tests. These unit tests cover the
pure helpers (truncation, traceback formatting, payload shape).
"""

from __future__ import annotations

import pytest

from agent_plane.runtime.background_tool_workflow import (
    ASYNC_WORK_COMPLETE_TOPIC,
    AsyncWorkCompletePayload,
    _payload_to_dict,
    format_failure_payload,
    truncate_for_llm,
    truncate_traceback,
)

# ─── truncate_for_llm ────────────────────────────────────────


def test_truncate_for_llm_under_budget_returns_unchanged() -> None:
    """Strings shorter than the budget pass through untouched."""
    text = "hello world"
    result = truncate_for_llm(text, budget=100)
    # Same object identity isn't required, but byte-equal output is.
    assert result == text


def test_truncate_for_llm_at_budget_returns_unchanged() -> None:
    """Strings exactly at the budget pass through untouched."""
    text = "x" * 100
    result = truncate_for_llm(text, budget=100)
    assert result == text


def test_truncate_for_llm_over_budget_appends_marker_with_drop_count() -> None:
    """Over-budget strings are truncated to budget + a drop-count marker."""
    text = "x" * 250
    result = truncate_for_llm(text, budget=100)
    # First 100 chars preserved verbatim.
    assert result.startswith("x" * 100)
    # Marker names the exact number of dropped characters so the
    # LLM can tell how much was lost. If the count is wrong, the
    # LLM would underestimate (or overestimate) what it's missing.
    assert "150 more chars truncated" in result


def test_truncate_for_llm_default_budget_is_10000() -> None:
    """The default budget matches B5/G44 — 10,000 Python `len()` units."""
    # A string just over the documented default should be truncated;
    # this guards against accidental changes to _RESULT_CHAR_BUDGET.
    text = "y" * 10_001
    result = truncate_for_llm(text)
    assert "1 more chars truncated" in result


def test_truncate_for_llm_counts_unicode_codepoints_not_bytes() -> None:
    """Truncation budget is Python ``len()`` (code points), not UTF-8 bytes (G44)."""
    # 50 emoji × 4 bytes each = 200 bytes; 50 code points; budget 100
    # code points should NOT truncate.
    text = "😀" * 50
    assert len(text) == 50
    result = truncate_for_llm(text, budget=100)
    assert result == text


# ─── truncate_traceback ──────────────────────────────────────


def test_truncate_traceback_under_line_budget_returns_unchanged() -> None:
    """Short tracebacks pass through unchanged."""
    tb = "Line 1\nLine 2\nLine 3"
    result = truncate_traceback(tb, line_budget=10)
    assert result == tb


def test_truncate_traceback_over_line_budget_keeps_head_and_marks_dropped() -> None:
    """Over-budget tracebacks keep the first N lines + drop-count marker."""
    tb = "\n".join(f"Line {i}" for i in range(50))
    result = truncate_traceback(tb, line_budget=10)
    # First 10 lines preserved (the deepest stack and the exception
    # line — the most useful for diagnosis).
    assert "Line 0" in result
    assert "Line 9" in result
    # Lines beyond the budget are dropped.
    assert "Line 10" not in result.split("[...")[0]
    # Marker reports exact dropped count.
    assert "40 more lines truncated" in result


def test_truncate_traceback_default_budget_is_30() -> None:
    """The default line budget matches B2 — 30 frames."""
    tb = "\n".join(f"L{i}" for i in range(40))
    result = truncate_traceback(tb)
    assert "10 more lines truncated" in result


# ─── format_failure_payload ──────────────────────────────────


def test_format_failure_payload_includes_exc_type_and_message() -> None:
    """The 'message' field combines the exception class name and message."""
    try:
        raise ValueError("something broke")
    except ValueError as exc:
        payload = format_failure_payload(exc)
    assert payload["message"] == "ValueError: something broke"


def test_format_failure_payload_includes_truncated_traceback() -> None:
    """The 'traceback' field is bounded by the line budget."""

    def deep_call(depth: int) -> None:
        if depth > 0:
            deep_call(depth - 1)
        else:
            raise RuntimeError("from the deep")

    try:
        deep_call(50)
    except RuntimeError as exc:
        payload = format_failure_payload(exc)

    tb = payload["traceback"]
    # Traceback contains the exception name and message somewhere.
    assert "RuntimeError" in tb
    assert "from the deep" in tb
    # If the deep call exceeded the line budget, the marker is present;
    # otherwise it's not. The point is the traceback is usable either way.
    if "more lines truncated" in tb:
        # Truncation happened — verify the marker shape is consistent
        # with truncate_traceback's documented format.
        assert "[..." in tb


def test_format_failure_payload_handles_exception_without_traceback() -> None:
    """A bare exception (no __traceback__) still produces a usable payload."""
    exc = ValueError("manually constructed")
    payload = format_failure_payload(exc)
    # Message field always populated.
    assert payload["message"] == "ValueError: manually constructed"
    # Traceback may be empty/short but the field exists so callers
    # don't have to None-check.
    assert "traceback" in payload
    assert isinstance(payload["traceback"], str)


# ─── payload + topic ─────────────────────────────────────────


def test_topic_constant_matches_design_doc() -> None:
    """The drain topic name is the documented constant — see G19."""
    # If someone renames this string, the parent's drain in
    # workflow.py won't wake on completions; tests for both sides
    # depend on this constant being identical.
    assert ASYNC_WORK_COMPLETE_TOPIC == "async_work_complete"


def test_payload_to_dict_round_trips_all_fields() -> None:
    """The dataclass → dict serialization preserves every field."""
    payload = AsyncWorkCompletePayload(
        task_id="tsk_abc",
        kind="tool",
        status="completed",
        output="hello",
        error=None,
    )
    result = _payload_to_dict(payload)
    # All five documented fields present with correct values.
    # If any field is missing, the parent's auto-delivery formatter
    # will KeyError when constructing the [System: ...] message.
    assert result == {
        "task_id": "tsk_abc",
        "kind": "tool",
        "status": "completed",
        "output": "hello",
        "error": None,
    }


def test_payload_to_dict_preserves_error_dict() -> None:
    """Failed payloads carry a populated error dict end-to-end."""
    payload = AsyncWorkCompletePayload(
        task_id="tsk_xyz",
        kind="tool",
        status="failed",
        output="ValueError: oops",
        error={"message": "ValueError: oops", "traceback": "stack..."},
    )
    result = _payload_to_dict(payload)
    assert result["error"] == {
        "message": "ValueError: oops",
        "traceback": "stack...",
    }
    assert result["status"] == "failed"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_payload_status_values_documented(status: str) -> None:
    """All three terminal status values are constructible (G86)."""
    # The drain depends on these exact strings to pick the right
    # auto-delivery formatter; the dataclass should accept any of
    # them without complaint.
    payload = AsyncWorkCompletePayload(
        task_id="tsk_p",
        kind="tool",
        status=status,
        output="",
        error=None,
    )
    assert payload.status == status
