"""Task entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A task representing a single response execution."""

    task_id: str
    conversation_id: str
    status: str  # "queued", "in_progress", "completed", "failed", "incomplete", "cancelled"
    agent_id: str
    created_at: int
    completed_at: int | None = None
    output: list[Any] = field(default_factory=list)
    inbox_closed: bool = False
    instructions: str | None = None
    background: bool = False
    previous_response_id: str | None = None
    usage: dict[str, Any] | None = None  # serialized Usage model
    error: dict[str, str] | None = None
    incomplete_details: dict[str, str] | None = None
