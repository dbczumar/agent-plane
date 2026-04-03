"""Unit tests for spawn/collect tool helpers."""

from __future__ import annotations

from typing import Any

import pytest

from agent_plane.entities.task import Task, TaskStatus
from agent_plane.tools.builtins.spawn import (
    _extract_output_text,
    _task_to_result,
)

# ── Helpers ──────────────────────────────────────────────


def _make_message_item(text: str) -> dict[str, Any]:
    """
    Build an output item dict matching the structure produced
    by ``_item_to_output()`` for an assistant message.

    :param text: The assistant's text content.
    :returns: A dict with ``type == "message"`` and nested
        ``output_text`` content block.
    """
    return {
        "type": "message",
        "id": "msg_1",
        "response_id": "task_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _make_function_call_item() -> dict[str, Any]:
    """
    Build an output item dict for a function call.

    :returns: A dict with ``type == "function_call"``.
    """
    return {
        "type": "function_call",
        "id": "call_1",
        "response_id": "task_1",
        "status": "completed",
        "name": "search",
        "arguments": '{"q": "test"}',
    }


def _make_task(
    *,
    status: str = TaskStatus.COMPLETED,
    output: list[dict[str, Any]] | None = None,
    agent_name: str = "researcher",
) -> Task:
    """
    Build a minimal Task for testing.

    :param status: Task status string.
    :param output: Output items list, or ``None`` for default
        empty list.
    :param agent_name: Agent name for the task.
    :returns: A :class:`Task` instance.
    """
    return Task(
        id="task_1",
        conversation_id="conv_1",
        status=status,
        agent_id="agent_1",
        agent_name=agent_name,
        created_at=1000,
        output=output if output is not None else [],
    )


# ── _extract_output_text tests ───────────────────────────


def test_extract_output_text_single_message() -> None:
    """Single assistant message — extracts the text content."""
    output = [_make_message_item("Hello world")]
    result = _extract_output_text(output)
    assert result == "Hello world"


def test_extract_output_text_multiple_messages() -> None:
    """Multiple messages — concatenated with double newline."""
    output = [
        _make_message_item("First"),
        _make_message_item("Second"),
    ]
    result = _extract_output_text(output)
    assert result == "First\n\nSecond"


def test_extract_output_text_skips_non_message_items() -> None:
    """Function call items are ignored — only messages extracted."""
    output = [
        _make_function_call_item(),
        _make_message_item("After tool"),
    ]
    result = _extract_output_text(output)
    assert result == "After tool"


def test_extract_output_text_empty_output() -> None:
    """Empty output list — returns empty string."""
    assert _extract_output_text([]) == ""


def test_extract_output_text_no_text_blocks() -> None:
    """Message with no output_text blocks — returns empty string."""
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_image", "url": "http://x"}],
        }
    ]
    assert _extract_output_text(output) == ""


def test_extract_output_text_skips_empty_text() -> None:
    """Empty and None text values are filtered out."""
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": ""},
                {"type": "output_text", "text": None},
                {"type": "output_text", "text": "Valid"},
            ],
        }
    ]
    assert _extract_output_text(output) == "Valid"


def test_extract_output_text_multiple_blocks_in_one_message() -> None:
    """Multiple output_text blocks in a single message content array."""
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Part A"},
                {"type": "output_text", "text": "Part B"},
            ],
        }
    ]
    assert _extract_output_text(output) == "Part A\n\nPart B"


def test_extract_output_text_message_with_no_content_key() -> None:
    """Message item missing 'content' key — gracefully skipped."""
    output = [{"type": "message", "role": "assistant"}]
    assert _extract_output_text(output) == ""


# ── _task_to_result tests ────────────────────────────────


def test_task_to_result_completed_with_output() -> None:
    """Completed task with output — extracts text, status 'completed'."""
    task = _make_task(output=[_make_message_item("Done!")])
    result = _task_to_result(task)
    assert result["response_id"] == "task_1"
    assert result["agent_name"] == "researcher"
    assert result["status"] == "completed"
    assert result["output"] == "Done!"


def test_task_to_result_completed_empty_output() -> None:
    """Completed but empty output — falls back to 'finished with status' message."""
    task = _make_task(output=[])
    result = _task_to_result(task)
    # Status is preserved as-is from the task.
    assert result["status"] == "completed"
    # Empty output list is falsy, so the terminal fallback message
    # is used instead of extracting text from output items.
    assert "finished with status: completed" in result["output"]


def test_task_to_result_failed() -> None:
    """Failed task — status 'failed', terminal fallback message."""
    task = _make_task(status=TaskStatus.FAILED)
    result = _task_to_result(task)
    assert result["status"] == "failed"
    # Terminal status produces a "finished with status" message.
    assert "finished with status: failed" in result["output"]


def test_task_to_result_cancelled() -> None:
    """Cancelled task — status 'cancelled', terminal fallback message."""
    task = _make_task(status=TaskStatus.CANCELLED)
    result = _task_to_result(task)
    assert result["status"] == "cancelled"
    assert "finished with status: cancelled" in result["output"]


@pytest.mark.parametrize(
    "status",
    [TaskStatus.QUEUED, TaskStatus.IN_PROGRESS],
    ids=["queued", "in_progress"],
)
def test_task_to_result_non_terminal_is_still_running(status: str) -> None:
    """Non-terminal status — preserves real status, 'still running' message."""
    task = _make_task(status=status)
    result = _task_to_result(task)
    # Status is preserved as-is (not remapped to "incomplete").
    assert result["status"] == status
    # Non-terminal tasks get a "still running" message.
    assert "is still running" in result["output"]


def test_task_to_result_includes_agent_name_in_fallback() -> None:
    """Fallback message includes the agent name for debuggability."""
    task = _make_task(status=TaskStatus.FAILED, agent_name="my-agent")
    result = _task_to_result(task)
    # Agent name appears in the fallback message (repr-quoted).
    assert "my-agent" in result["output"]
