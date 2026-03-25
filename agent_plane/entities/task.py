"""Task entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """
    Task lifecycle states.

    :cvar QUEUED: Task created but not yet started.
    :cvar IN_PROGRESS: Task is actively running.
    :cvar COMPLETED: Task finished successfully.
    :cvar FAILED: Task terminated with an error.
    :cvar INCOMPLETE: Task stopped early (e.g. max iterations).
    :cvar CANCELLED: Task was cancelled by the client.
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"


# Statuses where the task is no longer running.
TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.INCOMPLETE, TaskStatus.CANCELLED}
)

# Statuses where the task is still active (steering eligible).
ACTIVE_STATUSES = frozenset({TaskStatus.QUEUED, TaskStatus.IN_PROGRESS})


@dataclass
class Task:
    """
    A task representing a single response execution.

    :param id: Unique task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: ID of the conversation this task belongs
        to, e.g. ``"conv_xyz789"``.
    :param status: Current lifecycle status, e.g. ``"completed"``.
    :param agent_id: ID of the agent executing this task.
    :param agent_name: Denormalized agent name, persisted at creation
        so the API can return a stable model name even if the agent
        is renamed or deleted.
    :param created_at: Unix epoch timestamp of creation.
    :param completed_at: Unix epoch timestamp of completion, or
        ``None`` if not yet completed.
    :param output: Heterogeneous output items (messages, reasoning,
        function_calls) serialized as dicts; shape varies by item
        type. Values are heterogeneous so ``Any`` is the narrowest
        safe type.
    :param inbox_closed: Whether the steering inbox is closed.
    :param instructions: Per-request system instructions override.
    :param reasoning: Reasoning config,
        e.g. ``{"effort": "medium"}``.
    :param background: Whether this is a background task.
    :param previous_response_id: ID of the prior response in the
        conversation thread, or ``None`` for the first turn.
    :param usage: Serialized usage stats — contains nested
        output_tokens_details dict, so ``dict[str, Any]`` is the
        narrowest safe type.
    :param error: Error details, e.g.
        ``{"code": "server_error", "message": "..."}``.
    :param incomplete_details: Details on why the task is incomplete,
        e.g. ``{"reason": "max_output_tokens"}``.
    """

    id: str
    conversation_id: str
    status: str
    agent_id: str
    agent_name: str
    created_at: int
    completed_at: int | None = None
    # Heterogeneous output items (messages, reasoning, function_calls)
    # serialized as dicts; shape varies by item type.
    output: list[dict[str, Any]] = field(default_factory=list)
    inbox_closed: bool = False
    instructions: str | None = None
    # Reasoning config, e.g. {"effort": "low"|"medium"|"high"}
    reasoning: dict[str, str] | None = None
    background: bool = False
    previous_response_id: str | None = None
    # Serialized Usage — contains nested output_tokens_details dict,
    # so dict[str, Any] is the narrowest safe type.
    usage: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    incomplete_details: dict[str, str] | None = None
