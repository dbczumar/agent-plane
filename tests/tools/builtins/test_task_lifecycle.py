"""Tests for the task-lifecycle builtins (check_task, cancel_task, list_tasks).

Pure-helper coverage is here. The full ``invoke()`` paths require
a running task_store + DBOS runtime and are covered in the
server-integration suite.
"""

from __future__ import annotations

from agent_plane.entities.task import Task
from agent_plane.tools.builtins.task_lifecycle import (
    _ACTIVITY_MAX_CHARS,
    CancelTaskTool,
    CheckTaskTool,
    ListTasksTool,
    _build_check_payload,
    _truncate_content_field,
)

# ─── truncation helper ───────────────────────────────────────


def test_truncate_string_under_cap_returns_unchanged() -> None:
    """Strings under the cap pass through verbatim."""
    text = "short"
    assert _truncate_content_field(text) == text


def test_truncate_string_over_cap_appends_marker() -> None:
    """Over-cap strings get truncated with a ``[truncated]`` suffix."""
    text = "x" * (_ACTIVITY_MAX_CHARS + 100)
    result = _truncate_content_field(text)
    # Length must be cap + suffix length.
    assert result.startswith("x" * _ACTIVITY_MAX_CHARS)
    assert result.endswith(" [truncated]")
    # If the suffix is missing, the LLM has no signal that data
    # was dropped.


def test_truncate_walks_dict_values() -> None:
    """Recursion into dict values truncates nested strings."""
    item = {
        "type": "message",
        "content": [{"type": "input_text", "text": "y" * (_ACTIVITY_MAX_CHARS + 50)}],
        "role": "assistant",
    }
    result = _truncate_content_field(item)
    inner_text = result["content"][0]["text"]
    assert inner_text.endswith(" [truncated]")
    # Other fields must remain unchanged — a bug here would mutate
    # the role/type strings (both well under the cap, but the
    # invariant matters).
    assert result["role"] == "assistant"
    assert result["type"] == "message"


def test_truncate_passes_through_non_strings() -> None:
    """Numeric and bool fields aren't touched."""
    item = {"position": 42, "active": True, "ratio": 1.5}
    assert _truncate_content_field(item) == item


# ─── _build_check_payload ────────────────────────────────────


def _make_task(
    *,
    kind: str = "tool",
    status: str = "completed",
    output: list | None = None,
    error: dict[str, str] | None = None,
) -> Task:
    """Build a Task with the fields :func:`_build_check_payload` reads."""
    return Task(
        id="tsk_test",
        conversation_id="conv_test",
        agent_id="ag_test",
        agent_name="test-agent",
        created_at=1000,
        completed_at=2000 if status in {"completed", "failed", "cancelled"} else None,
        kind=kind,
        status=status,
        output=output if output is not None else [],
        error=error,
    )


def test_check_payload_for_completed_tool_includes_result_and_no_activity() -> None:
    """A completed tool task surfaces its output as ``result`` (no activity)."""
    task = _make_task(
        kind="tool",
        status="completed",
        output=[{"text": "hello"}],
    )
    payload = _build_check_payload(task)
    assert payload["task_id"] == "tsk_test"
    assert payload["kind"] == "tool"
    assert payload["status"] == "completed"
    assert payload["result"] == [{"text": "hello"}]
    # tools have no per-step recent_activity by design (G50).
    assert "recent_activity" not in payload


def test_check_payload_for_failed_task_includes_error() -> None:
    """A failed task surfaces both result and error."""
    task = _make_task(
        kind="tool",
        status="failed",
        output=[],
        error={"message": "ValueError: oops", "traceback": "stack..."},
    )
    payload = _build_check_payload(task)
    assert payload["status"] == "failed"
    assert payload["error"] == {"message": "ValueError: oops", "traceback": "stack..."}


def test_check_payload_uses_completed_at_as_updated_when_terminal() -> None:
    """Terminal tasks expose the completed timestamp via ``updated_at``."""
    task = _make_task(status="completed")
    payload = _build_check_payload(task)
    assert payload["created_at"] == 1000
    # If the test task were missing completed_at, this would be the
    # created_at instead — so the assertion proves the conditional
    # in _build_check_payload picked the right field.
    assert payload["updated_at"] == 2000


def test_check_payload_for_running_tool_omits_recent_activity() -> None:
    """A running TOOL task has no recent_activity (only sub-agents do)."""
    task = _make_task(kind="tool", status="in_progress")
    payload = _build_check_payload(task)
    assert payload["status"] == "in_progress"
    assert "recent_activity" not in payload


# ─── tool schemas ────────────────────────────────────────────


def test_check_task_schema_shape() -> None:
    """Schema is OpenAI function-format with required task_id."""
    schema = CheckTaskTool().get_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "check_task"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "task_id" in params["properties"]
    assert params["required"] == ["task_id"]


def test_cancel_task_schema_shape() -> None:
    """cancel_task requires task_id."""
    schema = CancelTaskTool().get_schema()
    assert schema["function"]["name"] == "cancel_task"
    assert schema["function"]["parameters"]["required"] == ["task_id"]


def test_list_tasks_schema_default_filter() -> None:
    """list_tasks has an optional ``filter`` enum with the documented values."""
    schema = ListTasksTool().get_schema()
    assert schema["function"]["name"] == "list_tasks"
    params = schema["function"]["parameters"]
    # filter is optional and defaults to "running".
    assert params["required"] == []
    filter_prop = params["properties"]["filter"]
    assert sorted(filter_prop["enum"]) == sorted(["running", "completed", "all"])
    assert filter_prop["default"] == "running"


def test_check_task_description_mentions_handle_origin() -> None:
    """The LLM-facing description explains where task_id comes from."""
    desc = CheckTaskTool.description().lower()
    # Must mention the handle / spawn origin so the LLM knows to
    # pass the task_id from a previous async tool call.
    assert "task_id" in desc
    assert "spawn" in desc or "asynchronous" in desc


def test_cancel_task_description_mentions_non_blocking() -> None:
    """The LLM-facing description sets expectations about timing."""
    desc = CancelTaskTool.description().lower()
    assert "non-blocking" in desc or "background" in desc
