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
    :param root_task_id: ID of the top-level task that initiated
        this sub-agent's spawn tree, or ``None`` for top-level
        tasks, e.g. ``"task_abc123"``.
    :param parent_task_id: ID of the IMMEDIATE parent task that
        caused this task to be created (distinct from
        :attr:`root_task_id`, which walks to the top-level).
        For top-level user turns this is ``None``. For async
        child tasks (``kind="tool"``, ``"sub_agent"``,
        ``"client_tool"``) created from inside a sub-agent, this
        is the sub-agent's own task id — *not* the root.
        Drives the ``async_work_complete`` drain's signal target
        so the immediate caller's drain wakes when the child
        completes.
    :param tools: Client-specified tool dicts (OpenAI format with
        ``agent_plane`` extension) supplied at request time. Restored
        from DBOS workflow inputs on recovery. ``None`` means no
        client tools were supplied.
    :param kind: Task kind discriminator. ``"agent_task"`` for
        user-initiated parent turns; ``"tool"`` for background
        custom-tool invocations spawned via ``@tool(synchronous=False)``;
        ``"sub_agent"`` for sub-agent workflows (Phase 3);
        ``"client_tool"`` for async client-side tools (Phase 5).
        Drives the unified task lifecycle (`check_task` /
        `cancel_task` / `list_tasks`) — the LLM only sees background
        work it spawned, not its own parent turn (G57).
    """

    id: str
    conversation_id: str
    status: str
    agent_id: str
    agent_name: str
    created_at: int
    completed_at: int | None = None
    root_task_id: str | None = None
    parent_task_id: str | None = None
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
    # Client-specified tool dicts (OpenAI format with agent_plane extension).
    # Restored from DBOS workflow inputs on recovery; None = no client tools.
    tools: list[dict[str, Any]] | None = None
    # G74: kind defaults to "agent_task" for backward compatibility
    # with existing call sites; new spawn paths set it explicitly.
    kind: str = "agent_task"
