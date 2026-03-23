"""Task entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Task lifecycle states."""

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
    """A task representing a single response execution."""

    task_id: str
    conversation_id: str
    status: str
    agent_id: str
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
