"""
Tests for the async-work drain helpers in
:mod:`agent_plane.runtime.workflow`.

These cover the pure formatting + persistence layer added by Phase 2
(D4/G18). The actual ``_drain_async_completions`` ``recv_async`` loop
needs a live DBOS workflow context and is exercised by the server
integration suite (``test_async_tool_integration.py`` once it lands).
"""

from __future__ import annotations

from agent_plane.runtime.workflow import (
    _build_async_completion_item,
    _format_async_completion_text,
)

# ─── _format_async_completion_text ───────────────────────────


def test_completed_payload_includes_task_id_kind_and_output() -> None:
    """A completed payload renders as ``[System: task ... completed]\\n<output>``."""
    payload = {
        "task_id": "tsk_abc",
        "kind": "tool",
        "status": "completed",
        "output": "the result body",
        "error": None,
    }
    text = _format_async_completion_text(payload)
    # Header carries both the literal task_id (so the LLM can
    # cross-reference its handle) and the kind (so the LLM can
    # tell a tool from a sub-agent).
    assert text.startswith("[System: task tsk_abc (tool) completed]")
    # Output body follows on the next line — if missing, the LLM
    # would see only the header and have no result to act on.
    assert "the result body" in text


def test_failed_payload_includes_message_and_traceback() -> None:
    """A failed payload renders the error message AND traceback."""
    payload = {
        "task_id": "tsk_xyz",
        "kind": "tool",
        "status": "failed",
        "output": "",
        "error": {
            "message": "ValueError: oops",
            "traceback": "Traceback...\n  line\nValueError: oops",
        },
    }
    text = _format_async_completion_text(payload)
    assert "[System: task tsk_xyz (tool) failed]" in text
    # Both fields must surface — message alone is not enough for
    # the LLM to suggest a fix; traceback alone is unreadable.
    assert "ValueError: oops" in text
    assert "Traceback..." in text


def test_failed_payload_handles_missing_traceback_gracefully() -> None:
    """Missing traceback falls back to message-only — never crashes."""
    payload = {
        "task_id": "tsk_q",
        "kind": "tool",
        "status": "failed",
        "output": "",
        "error": {"message": "RuntimeError: no tb"},
    }
    text = _format_async_completion_text(payload)
    # Header + message present; no spurious "None" or empty
    # traceback line at the end.
    assert text.endswith("RuntimeError: no tb")


def test_cancelled_payload_uses_status_string(  # G86
) -> None:
    """A cancelled payload surfaces ``cancelled`` in the header."""
    payload = {
        "task_id": "tsk_c",
        "kind": "tool",
        "status": "cancelled",
        "output": "",
        "error": None,
    }
    text = _format_async_completion_text(payload)
    # Without "cancelled" in the text, the LLM might assume the
    # task is still running and re-call check_task. The status
    # string is the LLM's only signal that the work stopped.
    assert text == "[System: task tsk_c (tool) cancelled]"


def test_completed_payload_with_empty_output_still_renders_header() -> None:
    """Empty output collapses to header + newline only."""
    payload = {
        "task_id": "tsk_empty",
        "kind": "tool",
        "status": "completed",
        "output": "",
        "error": None,
    }
    text = _format_async_completion_text(payload)
    # Header always present; trailing newline is fine — the LLM
    # parses by line and an empty body just means no result text.
    assert text.startswith("[System: task tsk_empty (tool) completed]")


# ─── _build_async_completion_item ────────────────────────────


def test_completion_item_is_user_role_input_text_block() -> None:
    """Built item matches the ``[System: ...]`` convention used elsewhere.

    The auto-collect/steering convention is a ``role=user`` message
    with an ``input_text`` content block. If we used ``role=system``
    here, the LLM would see it as a separate system instruction (and
    some providers reject mid-conversation system messages); if we
    used ``role=assistant``, the LLM would treat it as its own prior
    output and not respond to it.
    """
    payload = {
        "task_id": "tsk_z",
        "kind": "tool",
        "status": "completed",
        "output": "done",
        "error": None,
    }
    item = _build_async_completion_item("parent_tsk", payload)
    assert item.type == "message"
    # Persisted under the parent's response_id so subsequent GETs
    # group it under the right turn.
    assert item.response_id == "parent_tsk"
    assert item.data.role == "user"
    assert item.data.content[0]["type"] == "input_text"
    # The content text is the formatted body — proves the helpers
    # compose correctly end to end.
    assert "tsk_z" in item.data.content[0]["text"]
    assert "done" in item.data.content[0]["text"]
