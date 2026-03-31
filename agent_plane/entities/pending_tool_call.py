"""Pending tool call entity for tunneled client-side tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompletePendingToolCallResult(str, Enum):
    """
    Outcome of attempting to complete a pending tool call.

    Used by the PATCH handler to map to HTTP status codes.

    :cvar COMPLETED: Row updated from action_required to completed.
    :cvar NOT_FOUND: call_id does not exist in the table.
    :cvar ALREADY_COMPLETED: Row already has status=completed
        (idempotent re-PATCH). First writer wins -- the stored
        result is not overwritten.
    :cvar SUB_AGENT_DONE: Row exists but the sub-agent's task has
        reached a terminal status (completed, failed, cancelled).
        The tool result cannot be delivered because no one is
        waiting for it.
    """

    COMPLETED = "completed"
    NOT_FOUND = "not_found"
    ALREADY_COMPLETED = "already_completed"
    SUB_AGENT_DONE = "sub_agent_done"


@dataclass
class PendingToolCall:
    """
    A tunneled client-side tool call awaiting client execution.

    Represents one row in the ``pending_tool_calls`` table. Created
    when a sub-agent parks for a client-side tool, completed when
    the client PATCHes the result.

    :param call_id: Tool call ID (PK), matches the LLM-generated
        call ID, e.g. ``"call_abc123"``.
    :param root_task_id: The top-level task whose response output
        contains the ``function_call`` item, e.g. ``"task_root1"``.
    :param task_id: The parked sub-agent's task ID,
        e.g. ``"task_sub2"``.
    :param status: ``"action_required"`` or ``"completed"``.
    :param result: The tool's string output from the client.
        ``None`` until the client PATCHes.
    :param created_at: Unix epoch when the sub-agent parked.
    :param completed_at: Unix epoch when the client PATCHed.
        ``None`` until completed.
    """

    call_id: str
    root_task_id: str
    task_id: str
    status: str
    result: str | None
    created_at: int
    completed_at: int | None
