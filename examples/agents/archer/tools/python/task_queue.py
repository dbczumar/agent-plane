"""A persistent task queue exposed as a set of ``@tool`` functions.

State is held per-conversation, per-agent via the framework's
:class:`ToolState` primitive — the agent can add, list, update,
and remove tasks across multiple turns and the queue survives
until the conversation ends. No MCP server, no database; just a
JSON blob keyed by ``"tasks"`` in the agent's state namespace.

Design choice: both ``tasks`` (id → task dict) and ``next_id``
(the monotonic counter) live under the **same** transaction key
so two parallel ``add_task`` calls can't produce colliding IDs.
See ``designs/TOOL_STATE.md`` for the underlying state model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from agent_plane_client import ToolState, tool

# Status values the LLM is allowed to use. Keeping the set closed
# (rather than a free-form string) makes ``list_tasks(status=...)``
# predictable and surfaces typos as schema errors at call time.
Status = Literal["pending", "in_progress", "done", "cancelled"]

# Sole state key. Value shape:
#   {"next_id": int, "tasks": {"1": {Task}, "2": {Task}, ...}}
# Bundling both sub-fields under one key means ``transaction``'s
# flock covers both reads and writes — no race between reading
# ``next_id`` and writing back.
_STATE_KEY = "queue"


def _empty_state() -> dict:
    """Initial state for an empty queue."""
    return {"next_id": 1, "tasks": {}}


def _now() -> str:
    """UTC timestamp in ISO-8601 — used for created_at / note timestamps."""
    return datetime.now(timezone.utc).isoformat()


@tool
def add_task(description: str, tool_state: ToolState) -> dict:
    """Add a task to the queue.

    Args:
        description: What the task is, e.g. "write tests for foo".
    """
    with tool_state.transaction(_STATE_KEY, default=_empty_state()) as q:
        task_id = q["next_id"]
        task = {
            "id": task_id,
            "description": description,
            "status": "pending",
            "created_at": _now(),
            "notes": [],
        }
        q["tasks"][str(task_id)] = task
        q["next_id"] = task_id + 1
    return task


@tool
def list_tasks(
    tool_state: ToolState,
    status: Status | None = None,
) -> list[dict]:
    """List all tasks, optionally filtered by status.

    Args:
        status: If given, only tasks with this status are returned.
    """
    q = tool_state.get(_STATE_KEY, default=_empty_state())
    tasks = list(q["tasks"].values())
    if status is not None:
        tasks = [t for t in tasks if t["status"] == status]
    return tasks


@tool
def get_task(task_id: int, tool_state: ToolState) -> dict | None:
    """Get one task by ID, or None if no such task exists.

    Args:
        task_id: The task ID returned by add_task.
    """
    q = tool_state.get(_STATE_KEY, default=_empty_state())
    return q["tasks"].get(str(task_id))


@tool
def update_task_status(
    task_id: int,
    new_status: Status,
    tool_state: ToolState,
) -> dict:
    """Change a task's status.

    Args:
        task_id: The task ID.
        new_status: One of pending, in_progress, done, cancelled.
    """
    with tool_state.transaction(_STATE_KEY, default=_empty_state()) as q:
        key = str(task_id)
        if key not in q["tasks"]:
            raise ValueError(f"no task with id {task_id}")
        q["tasks"][key]["status"] = new_status
        return q["tasks"][key]


@tool
def add_task_note(task_id: int, note: str, tool_state: ToolState) -> dict:
    """Append a progress note to a task.

    Useful for recording intermediate results or status updates
    during a long-running task.

    Args:
        task_id: The task ID.
        note: The note text.
    """
    with tool_state.transaction(_STATE_KEY, default=_empty_state()) as q:
        key = str(task_id)
        if key not in q["tasks"]:
            raise ValueError(f"no task with id {task_id}")
        q["tasks"][key]["notes"].append({"at": _now(), "text": note})
        return q["tasks"][key]


@tool
def remove_task(task_id: int, tool_state: ToolState) -> str:
    """Remove a task from the queue entirely.

    Args:
        task_id: The task ID.
    """
    with tool_state.transaction(_STATE_KEY, default=_empty_state()) as q:
        q["tasks"].pop(str(task_id), None)
    return "ok"
