"""
Unified task lifecycle builtins: ``check_task``, ``cancel_task``,
``list_tasks``.

These tools operate on the task_id handles returned by background
work (sub-agents, ``@tool(synchronous=False)`` invocations,
``client_tool`` async calls). They are auto-enabled per §3.5 of
the design — any agent that can produce background work gets
them in its tool set.

Scoping rules:
- ``check_task`` / ``cancel_task`` reject ``kind="agent_task"``
  task_ids with ``task_not_found`` (G57). The LLM should never
  cancel its own parent turn.
- All three tools restrict results to the caller's conversation
  tree (G23) — task_ids belonging to other conversations look
  identical to nonexistent task_ids.
- ``list_tasks(filter="running")`` is the default — surfaces only
  what's still in flight.

Result shapes match §4.3 of the design doc (the unified handle).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_plane.entities.task import TERMINAL_STATUSES, Task
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

# G50 — match the existing _ACTIVITY_TAIL / _ACTIVITY_MAX_CHARS
# constants in agent_plane.tools.builtins.spawn so check_task on
# a sub-agent returns the same recent_activity shape that the
# old check_sub_agents path returned.
_ACTIVITY_TAIL = 5
_ACTIVITY_MAX_CHARS = 2000

# G57 — kinds the LLM is allowed to inspect/cancel via the
# unified lifecycle. agent_task (the parent turn) is excluded
# so the LLM can't accidentally cancel itself.
_LLM_VISIBLE_KINDS = frozenset({"tool", "sub_agent", "client_tool"})


# ── Helpers shared by all three tools ───────────────────────


def _truncate_content_field(value: Any, *, max_chars: int = _ACTIVITY_MAX_CHARS) -> Any:
    """
    Truncate string fields inside a conversation item to a length cap.

    Walks the item dict recursively and shortens any ``str`` value
    longer than ``max_chars``. Lists and nested dicts are traversed.
    Non-string scalars pass through unchanged. The cap protects the
    LLM context from a single huge tool result blowing out
    ``recent_activity``.

    :param value: The item, list, dict, or scalar to walk.
    :param max_chars: Per-string maximum, e.g. ``2000``.
    :returns: The same shape with strings truncated where needed.
    """
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + " [truncated]"
    if isinstance(value, dict):
        return {k: _truncate_content_field(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_content_field(v, max_chars=max_chars) for v in value]
    return value


def _get_recent_activity_for_task(task: Task) -> list[dict[str, Any]] | None:
    """
    Fetch the last :data:`_ACTIVITY_TAIL` items from a sub-agent's conversation.

    Returns ``None`` for non-sub-agent tasks. For sub-agents, fetches
    the trailing items from the child conversation and truncates each
    item's content fields to :data:`_ACTIVITY_MAX_CHARS`.

    :param task: The task whose conversation to inspect.
    :returns: A list of item dicts (most recent last), or ``None``
        if the task isn't a sub-agent (other kinds have no
        per-step activity to surface).
    """
    if task.kind != "sub_agent":
        return None

    # Lazy import — avoids pulling the runtime globals into modules
    # that import this file at agent-image load time.
    from agent_plane.runtime import get_conversation_store

    conv_store = get_conversation_store()
    page = conv_store.list_items(task.conversation_id, limit=_ACTIVITY_TAIL)
    # list_items returns a PagedList wrapper; the items live on .data.
    return [_truncate_content_field(item) for item in page.data]


def _build_check_payload(task: Task) -> dict[str, Any]:
    """
    Build the ``check_task`` response payload for one task.

    :param task: The resolved task.
    :returns: A dict with the unified handle shape per §4.3:
        ``{task_id, kind, status, result?, recent_activity?,
        created_at, updated_at, sub_agent?: {type, name},
        client_tool?: {name}}``.
    """
    payload: dict[str, Any] = {
        "task_id": task.id,
        "kind": task.kind,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.completed_at if task.completed_at is not None else task.created_at,
    }

    is_terminal = task.status in {s.value for s in TERMINAL_STATUSES}

    if is_terminal:
        # Surface the task's output. For sub-agents, output is the
        # accumulated assistant items; for tools/client_tools it's
        # the truncated result. Either way, the LLM gets it via
        # the `result` field.
        payload["result"] = task.output
        if task.error is not None:
            payload["error"] = task.error
    else:
        # Still running — recent_activity is the per-step view for
        # sub-agents (G50). Tool / client_tool kinds have no
        # per-step visibility; drop the field.
        activity = _get_recent_activity_for_task(task)
        if activity is not None:
            payload["recent_activity"] = activity

    return payload


def _scope_check(task: Task, ctx: ToolContext) -> bool:
    """
    Decide whether ``ctx`` is allowed to inspect/cancel this task.

    Per G23, ``check_task`` / ``cancel_task`` only operate on tasks
    within the caller's conversation tree. The simplest correct
    implementation: a task is in scope iff its conversation is the
    caller's conversation OR a child of it. Sub-agent children
    have ``parent_conversation_id`` set, so the relationship is
    one DB hop away.

    :param task: The task being inspected/cancelled.
    :param ctx: The calling tool's execution context.
    :returns: ``True`` if the task is in the caller's tree.
    """
    # The ToolContext doesn't carry conversation_id directly today,
    # but task_id ↔ workflow_id ↔ task row ↔ conversation_id is the
    # walk. For Phase 2 we scope by the caller's task being the
    # immediate parent task; sub-agent recursion (Phase 3+) extends
    # this to include the conversation tree.
    from agent_plane.runtime import get_task_store

    caller_task = get_task_store().get_sync(ctx.task_id)
    if caller_task is None:
        return False

    # Direct match: caller is inspecting a task in its own conversation.
    if task.conversation_id == caller_task.conversation_id:
        return True

    # Subtree match: caller spawned this task tree as the root.
    if task.root_task_id is not None and task.root_task_id == caller_task.id:
        return True

    # Caller is a sub-agent inspecting one of its own children;
    # the caller's task_id appears as the child's root_task_id when
    # caller IS a root, OR same root_task_id otherwise.
    if caller_task.root_task_id is not None and task.root_task_id == caller_task.root_task_id:
        return True

    return False


# ── check_task ──────────────────────────────────────────────


class CheckTaskTool(Tool):
    """
    Inspect the current state of a background task by ``task_id``.

    Returns the unified handle shape per §4.3 of the design doc.
    Hidden kinds (``"agent_task"``) and out-of-scope task_ids
    return ``task_not_found`` so the LLM cannot use this to probe
    other workflows.
    """

    @classmethod
    def name(cls) -> str:
        """:returns: ``"check_task"``."""
        return "check_task"

    @classmethod
    def description(cls) -> str:
        """:returns: Human-readable description for the LLM."""
        return (
            "Inspect the current state of a background task by task_id. "
            "Returns kind, status, and either the result (if terminal) "
            "or recent activity (if still running). The task_id comes from "
            "the handle returned by spawn_sub_agent / send_to_sub_agent / "
            "an asynchronous tool call."
        )

    def get_schema(self) -> dict[str, Any]:
        """:returns: OpenAI function-format tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": (
                                "The task identifier returned by a previous "
                                "spawn or async tool call."
                            ),
                        },
                    },
                    "required": ["task_id"],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Resolve the task and return its check payload as JSON.

        :param arguments: JSON-encoded ``{"task_id": "..."}``.
        :param ctx: Server-side execution context.
        :returns: A JSON string with the check payload, or an
            error envelope (``{"error": "task_not_found"}`` for
            missing/out-of-scope/agent_task kinds).
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid arguments: {exc}"})

        task_id = args.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return json.dumps({"error": "task_id is required"})

        from agent_plane.runtime import get_task_store

        task = get_task_store().get_sync(task_id)
        if task is None or task.kind not in _LLM_VISIBLE_KINDS or not _scope_check(task, ctx):
            # Same response for nonexistent / wrong-kind / out-of-scope
            # so the LLM can't probe for other tasks (G23).
            return json.dumps({"error": "task_not_found", "task_id": task_id})

        return json.dumps(_build_check_payload(task))


# ── cancel_task ─────────────────────────────────────────────


class CancelTaskTool(Tool):
    """
    Cancel a background task by ``task_id``.

    Non-blocking: marks the task cancelled in ``task_store`` and
    returns immediately. The child workflow observes the cancel
    at its next DBOS checkpoint and emits the
    ``async_work_complete`` signal so the parent's drain wakes
    and removes it from ``pending_tasks`` (G86).
    """

    @classmethod
    def name(cls) -> str:
        """:returns: ``"cancel_task"``."""
        return "cancel_task"

    @classmethod
    def description(cls) -> str:
        """:returns: Human-readable description."""
        return (
            "Cancel a running background task. Non-blocking — the task "
            "will transition to cancelled status; you'll see a "
            "[System: task ... cancelled] message before your next "
            "iteration. Already-terminal tasks are unchanged (no error)."
        )

    def get_schema(self) -> dict[str, Any]:
        """:returns: OpenAI function-format tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": (
                                "The task identifier to cancel — must "
                                "be a background task you spawned."
                            ),
                        },
                    },
                    "required": ["task_id"],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Cancel the task if found and in scope.

        :param arguments: JSON-encoded ``{"task_id": "..."}``.
        :param ctx: Server-side execution context.
        :returns: JSON ``{"cancelled": bool, "prior_status": "..."}``,
            or ``{"error": "task_not_found"}`` for missing /
            out-of-scope / agent_task kinds.
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid arguments: {exc}"})

        task_id = args.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return json.dumps({"error": "task_id is required"})

        from agent_plane.runtime import get_task_store

        task_store = get_task_store()
        task = task_store.get_sync(task_id)
        if task is None or task.kind not in _LLM_VISIBLE_KINDS or not _scope_check(task, ctx):
            return json.dumps({"error": "task_not_found", "task_id": task_id})

        prior_status = task.status
        if prior_status in {s.value for s in TERMINAL_STATUSES}:
            # Already done; return without calling cancel — preserves the
            # original terminal state (first-write-wins, G3).
            return json.dumps(
                {
                    "cancelled": False,
                    "prior_status": prior_status,
                    "task_id": task_id,
                }
            )

        # For client_tool kind, emit the response.client_task.cancel
        # SSE event on the parent's stream BEFORE cancelling the
        # holder workflow. The SDK's D6 lifecycle uses this event
        # to cancel the local asyncio.Task running the tool body —
        # without it, the SDK keeps running the body even though
        # the agent has moved on. The body's eventual PATCH will
        # be a no-op (G3) but compute / I/O / etc. is wasted.
        # Mirrors what cancel_pending_child_tools does for the
        # parent-cancel propagation path; uses _write_output so the
        # cancel lands on both the live stream (for connected SDKs)
        # and the durable stream (for crash-recovery / inspection).
        if task.kind == "client_tool":
            # Use _live_publish (real-time only) instead of
            # _write_output (which also writes to the durable
            # stream via DBOS write_stream). The durable
            # write requires a DBOS step context, but
            # CancelTaskTool.invoke runs inside the parent
            # workflow's thread executor (via asyncio.to_thread)
            # — calling sync DBOS APIs from there can deadlock
            # against the workflow's own recv loop. Live-only
            # publish goes through an in-process channel and
            # is what the SDK actually reads via SSE.
            from agent_plane.runtime.live_stream import publish as _live_publish

            cancel_target_task_id = task.parent_task_id or task.root_task_id
            if cancel_target_task_id is not None:
                _live_publish(
                    cancel_target_task_id,
                    {
                        "type": "response.client_task.cancel",
                        "task_id": task_id,
                    },
                )

        task_store.cancel(task_id)
        return json.dumps(
            {
                "cancelled": True,
                "prior_status": prior_status,
                "task_id": task_id,
            }
        )


# ── list_tasks ──────────────────────────────────────────────


class ListTasksTool(Tool):
    """
    Enumerate background tasks the LLM has spawned in this conversation.

    Default ``filter="running"`` returns only in-flight tasks. The
    user-initiated parent turn (``kind="agent_task"``) is always
    excluded (G57) so the LLM cannot list / interfere with its own
    workflow.
    """

    @classmethod
    def name(cls) -> str:
        """:returns: ``"list_tasks"``."""
        return "list_tasks"

    @classmethod
    def description(cls) -> str:
        """:returns: Human-readable description."""
        return (
            "List background tasks you've spawned in this conversation. "
            "Default returns running tasks only. Pass filter='completed' "
            "or filter='all' to see terminal tasks. Excludes your own "
            "parent turn — only background work you spawned shows up."
        )

    def get_schema(self) -> dict[str, Any]:
        """:returns: OpenAI function-format tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "enum": ["running", "completed", "all"],
                            "default": "running",
                            "description": (
                                "Which tasks to include. 'running' shows "
                                "only in-flight tasks; 'completed' shows "
                                "terminal tasks; 'all' shows both."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Return matching tasks as a JSON list.

        :param arguments: JSON-encoded ``{"filter": "..."}`` (optional).
        :param ctx: Server-side execution context.
        :returns: JSON ``{"tasks": [{task_id, kind, status,
            sub_agent?: {type, name}, created_at}, ...]}``.
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid arguments: {exc}"})

        filter_value = args.get("filter", "running")
        if filter_value not in {"running", "completed", "all"}:
            return json.dumps(
                {"error": f"invalid filter {filter_value!r}; expected running/completed/all"}
            )

        from agent_plane.runtime import get_task_store

        task_store = get_task_store()
        caller_task = task_store.get_sync(ctx.task_id)
        if caller_task is None:
            # Caller's own task missing — treat as no tasks visible
            # rather than raising (defensive for tests / edge cases).
            return json.dumps({"tasks": []})

        # Pull all tasks under the caller's root and filter in Python.
        # Smaller blast radius than adding a kind-aware list_tasks
        # method to the abstract store right now; the result set is
        # bounded by the conversation tree's task count.
        root_id = caller_task.root_task_id or caller_task.id
        # list_tasks is async; we're in a sync tool invocation. Fall
        # back to the sync helper that the spawn tool uses for the
        # same purpose.
        candidates = _list_tasks_sync(task_store, root_task_id=root_id)

        results: list[dict[str, Any]] = []
        terminal_strs = {s.value for s in TERMINAL_STATUSES}
        for task in candidates:
            if task.kind not in _LLM_VISIBLE_KINDS:
                continue
            is_terminal = task.status in terminal_strs
            if filter_value == "running" and is_terminal:
                continue
            if filter_value == "completed" and not is_terminal:
                continue
            entry: dict[str, Any] = {
                "task_id": task.id,
                "kind": task.kind,
                "status": task.status,
                "created_at": task.created_at,
            }
            if task.kind == "sub_agent":
                # Display the sub-agent's name so the LLM can match
                # the task back to a (type, name) handle. Sub-agent
                # type is encoded into agent_name; full sub_agent
                # block lands in Phase 4.
                entry["sub_agent"] = {"name": task.agent_name}
            results.append(entry)

        return json.dumps({"tasks": results})


def _list_tasks_sync(task_store: Any, *, root_task_id: str) -> list[Task]:
    """
    Synchronous task enumeration by root_task_id.

    The abstract store's ``list_tasks`` is async; tools run in a
    sync context, so we use the store's sync helper. Falls back to
    iterating asyncio if no sync method exists, matching the
    spawn.py pattern.

    :param task_store: The runtime task store.
    :param root_task_id: Root task for the spawn tree.
    :returns: Tasks under that root, including the root itself.
    """
    # Most stores expose list_tasks as both sync and async; prefer
    # the sync attr when present to avoid the asyncio.run hop.
    list_sync = getattr(task_store, "list_tasks_sync", None)
    if list_sync is not None:
        return list(list_sync(root_task_id=root_task_id))

    # Async fallback — wrap a coroutine with asyncio.run. Fine for
    # tool dispatch (already sync; small one-off).
    import asyncio

    return list(asyncio.run(task_store.list_tasks(root_task_id=root_task_id)))
