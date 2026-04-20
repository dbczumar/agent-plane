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
from agent_plane.stores.task_store import TaskStore
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
# so the LLM can't accidentally cancel itself. ``terminal`` is
# included so ``terminal_run(synchronous=False)`` tasks work
# through the same surface as @tool and sub-agent tasks.
_LLM_VISIBLE_KINDS = frozenset({"tool", "sub_agent", "client_tool", "terminal"})

# Clamp bounds for ``check_task``'s ``wait_ms`` parameter. The
# ceiling matches terminal_send_input's — 30 s is the longest a
# single tool call should hold an open LLM response before the
# parent turn starts looking broken to the user. The floor of 0
# preserves "don't wait" semantics when the caller passes 0.
_CHECK_TASK_WAIT_FLOOR_MS = 0
_CHECK_TASK_WAIT_CEILING_MS = 30_000
# Cadence of the poll loop inside ``check_task(wait_ms>0)``. 100ms
# is short enough that the LLM-facing latency is dominated by
# ``wait_ms`` itself (not by how fast we notice completion), but
# long enough that a 30s wait is only 300 task-store reads —
# cheap.
_CHECK_TASK_WAIT_POLL_INTERVAL_S = 0.1

# Cache the set of terminal-status string values so every poll
# iteration doesn't rebuild it. TERMINAL_STATUSES is an IntEnum;
# check_task compares against the ``.value`` string form that
# lives in the DB.
_TERMINAL_STATUS_STRS: frozenset[str] = frozenset(s.value for s in TERMINAL_STATUSES)


# ── Helpers shared by all three tools ───────────────────────


def _cancel_unavailable_json(task: Task, prior_status: str, *, reason: str) -> str:
    """Build the ``cancel_task`` JSON response when cancel can't run.

    Shared across the two non-cancellable branches of
    :meth:`CancelTaskTool._cancel_terminal_task` (shell manager gone,
    task already unregistered) so the response shape stays
    consistent.

    :param task: The terminal task that couldn't be cancelled.
    :param prior_status: The task's DB status at inspection time —
        echoed back so the LLM can tell "already completed" from
        "running but unreachable."
    :param reason: One of the documented reason codes:
        ``"shell_unavailable"`` (no manager for the conversation)
        or ``"task_no_longer_running"`` (manager exists but task
        not registered — completed or never started).
    :returns: JSON string with
        ``{cancelled: False, prior_status, task_id, reason}``.
    """
    return json.dumps(
        {
            "cancelled": False,
            "prior_status": prior_status,
            "task_id": task.id,
            "reason": reason,
        }
    )


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


def _get_recent_terminal_activity(task: Task) -> str | None:
    """
    Fetch the stdout delta for a running ``kind="terminal"`` task.

    Reads from the :class:`TerminalManager`'s per-task cursor,
    advancing it so successive ``check_task`` calls see only new
    bytes. The returned string is ANSI-stripped and bounded by the
    ring buffer size (1 MB) minus whatever bytes remain unread.

    Returns ``None`` when:
    - The task isn't a terminal task.
    - The task is registered nowhere (completed and unregistered, or
      never registered — the latter shouldn't happen with the
      current dispatch path).
    - The task's shell was closed between the registration and now.

    :param task: The task whose terminal shell to peek.
    :returns: ANSI-stripped stdout delta since the last call, or
        ``None`` when no live cursor exists.
    """
    if task.kind != "terminal":
        return None

    from agent_plane.runtime import get_terminal_registry

    registry = get_terminal_registry()
    # active_conversation_ids is the only public "does this
    # conversation have a manager?" check — if the conversation
    # was reaped or the server restarted, we have no shell to
    # poll and the delta is unavailable.
    if task.conversation_id not in registry.active_conversation_ids():
        return None
    # The workspace arg is only used on cache miss; the manager
    # already exists since active_conversation_ids saw it. Pass a
    # sentinel PurePath to satisfy the type. Reaching into the
    # registry's internal map would be cleaner but the public API
    # requires the arg — accept the minor awkwardness.
    from pathlib import Path as _Path

    manager = registry.for_conversation(task.conversation_id, _Path("."))
    peek = manager.peek_task_stdout(task.id)
    if peek is None:
        return None
    text = peek.text
    if peek.lost_bytes > 0:
        # Surface the gap so the agent knows stdout was dropped.
        text = f"[... {peek.lost_bytes} bytes evicted before this poll ...]\n{text}"
    return text


def _get_rendered_terminal_screen(task: Task) -> str | None:
    """Fetch the pyte-rendered screen for a running terminal task.

    Complementary to :func:`_get_recent_terminal_activity`: the
    stream view is best for commands that scroll (``make``, tests),
    the screen view is best for full-screen programs (``vim``,
    ``less``, ``htop``) where raw stdout is mostly cursor codes
    and ANSI-stripping yields nothing useful.

    Returns ``None`` under the same conditions that make the stream
    unavailable — wrong kind, no manager, closed shell — so both
    fields are omitted together from the check_task response when
    the shell is gone.

    :param task: The task whose shell to render.
    :returns: The rendered 50x160 screen as ``\\n``-separated
        lines, or ``None`` when no live shell.
    """
    if task.kind != "terminal":
        return None

    from agent_plane.runtime import get_terminal_registry

    registry = get_terminal_registry()
    if task.conversation_id not in registry.active_conversation_ids():
        return None
    from pathlib import Path as _Path

    # The workspace arg is only consulted on cache MISS; the
    # ``active_conversation_ids`` check above guarantees the manager
    # already exists, so this sentinel path is never used as a
    # workspace. Same pattern as ``_get_recent_terminal_activity``.
    manager = registry.for_conversation(task.conversation_id, _Path("."))
    return manager.rendered_screen_for_task(task.id)


def _resolve_check_task_wait_ms(raw: Any) -> int:
    """Coerce the caller's ``wait_ms`` input to a clamped integer.

    Handles three cases: ``None`` / missing → 0 (no wait), integer
    → clamp to the configured floor/ceiling, anything else → 0
    (fail-safe to non-blocking behavior rather than raising, since
    the caller is the LLM and the worst outcome should just be
    "it didn't wait" not "the tool errored").

    :param raw: Whatever the LLM passed as ``wait_ms``.
    :returns: Clamped int in ``[_CHECK_TASK_WAIT_FLOOR_MS,
        _CHECK_TASK_WAIT_CEILING_MS]``.
    """
    if not isinstance(raw, int) or isinstance(raw, bool):
        return _CHECK_TASK_WAIT_FLOOR_MS
    return max(
        _CHECK_TASK_WAIT_FLOOR_MS,
        min(_CHECK_TASK_WAIT_CEILING_MS, raw),
    )


def _poll_until_terminal_or_timeout(
    task_store: TaskStore,
    task_id: str,
    wait_ms: int,
) -> Task | None:
    """Poll ``task_store`` until ``task_id`` reaches a terminal status
    or ``wait_ms`` elapses.

    Used by ``check_task`` to implement server-side waiting without
    forcing the LLM to loop-poll. The poll reads DBOS workflow
    status via the store's sync path — which is cheap since DBOS
    memoizes the state — so a 30-second wait is hundreds of O(1)
    reads, not hundreds of heavyweight queries.

    :param task_store: The global :class:`TaskStore` (passed in so
        callers don't re-import).
    :param task_id: The task to watch.
    :param wait_ms: Maximum wait in milliseconds. Caller should
        have clamped this already via
        :func:`_resolve_check_task_wait_ms`.
    :returns: The latest :class:`Task` observation (terminal or not),
        or ``None`` if the task disappeared mid-wait (shouldn't
        happen but defensive — caller falls back to its earlier
        observation).
    """
    import time as _time

    deadline = _time.monotonic() + wait_ms / 1000
    while True:
        fresh = task_store.get_sync(task_id)
        if fresh is None:
            return None
        if fresh.status in _TERMINAL_STATUS_STRS:
            return fresh
        if _time.monotonic() >= deadline:
            return fresh
        _time.sleep(_CHECK_TASK_WAIT_POLL_INTERVAL_S)


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
        # Still running — recent_activity is kind-specific:
        # - sub_agent: the last few items from the child's conversation.
        # - terminal: stdout delta since the last check_task call
        #   (tail -f semantics; see §6.11 of the terminal research doc).
        # - tool / client_tool: no per-step visibility; field dropped.
        if task.kind == "terminal":
            terminal_activity = _get_recent_terminal_activity(task)
            if terminal_activity is not None:
                payload["recent_activity"] = terminal_activity
            # Add the pyte-rendered screen alongside ``recent_activity``
            # so the LLM can see what a full-screen program currently
            # displays (vim, less, htop) in addition to the streaming
            # byte delta. Both views are populated from the same
            # shell; either may be the more informative one depending
            # on the program.
            screen = _get_rendered_terminal_screen(task)
            if screen is not None:
                payload["screen"] = screen
        else:
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
            "an asynchronous tool call. Pass wait_ms to block up to N "
            "milliseconds waiting for the task to reach a terminal status "
            "— useful when you fired an async job and want to wait for it "
            "to finish without polling in a tight loop."
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
                        "wait_ms": {
                            "type": ["integer", "null"],
                            "description": (
                                "If set, block up to this many milliseconds "
                                "waiting for the task to reach a terminal "
                                "status (completed, failed, or cancelled) "
                                "before returning. Null or 0 means return "
                                "immediately with the current state. "
                                "Clamped to [0, 30000]. Useful for async "
                                "commands where you want to wait for "
                                "completion without polling in a loop — "
                                "the tool does the waiting server-side."
                            ),
                            "default": None,
                        },
                    },
                    "required": ["task_id"],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Resolve the task and return its check payload as JSON.

        :param arguments: JSON-encoded
            ``{"task_id": "...", "wait_ms": ...}``.
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

        wait_ms_raw = args.get("wait_ms")
        wait_ms = _resolve_check_task_wait_ms(wait_ms_raw)

        from agent_plane.runtime import get_task_store

        task_store = get_task_store()
        task = task_store.get_sync(task_id)
        if task is None or task.kind not in _LLM_VISIBLE_KINDS or not _scope_check(task, ctx):
            # Same response for nonexistent / wrong-kind / out-of-scope
            # so the LLM can't probe for other tasks (G23).
            return json.dumps({"error": "task_not_found", "task_id": task_id})

        if wait_ms > 0 and task.status not in _TERMINAL_STATUS_STRS:
            task = _poll_until_terminal_or_timeout(task_store, task_id, wait_ms) or task

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
        # parent-cancel propagation path; uses _live_publish so
        # the cancel lands on the live stream without the durable
        # write that would require a DBOS step context.
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

        # Terminal-kind tasks have their own cancel primitive: send
        # SIGINT to the running shell via the TerminalManager. This
        # is NOT the same as DBOS cancel_workflow — the background
        # workflow is parked on a blocking ``run_sync`` in a thread
        # pool and doesn't respond to asyncio cancellation. SIGINT
        # unblocks bash (D marker fires with exit 130), run_sync
        # returns with status="killed", and the workflow completes
        # normally with payload status="cancelled" (auto-delivered
        # to the parent via the async_work_complete drain).
        if task.kind == "terminal":
            return self._cancel_terminal_task(task, prior_status)

        task_store.cancel(task_id)
        return json.dumps(
            {
                "cancelled": True,
                "prior_status": prior_status,
                "task_id": task_id,
            }
        )

    def _cancel_terminal_task(self, task: Task, prior_status: str) -> str:
        """Mark a terminal task for cancellation and return the JSON result.

        Flags the task via :meth:`TerminalManager.request_cancel`.
        The actual SIGINT happens in the background workflow's own
        thread — the ``run_sync`` read loop polls a cancel predicate
        that reads this flag, then calls ``_interrupt_children`` to
        send SIGINT to bash's foreground child. Keeping the
        interrupt thread-local avoids cross-thread pexpect races.

        Two no-cancel branches return early with a ``reason`` field
        so the LLM can distinguish "no shell state" from "already
        done" without parsing status codes.

        :param task: The terminal task being cancelled.
        :param prior_status: The task's status at inspection time
            (echoed back in the response for LLM diagnostics).
        :returns: JSON string with ``{cancelled, task_id,
            prior_status, reason?}``.
        """
        from agent_plane.runtime import get_terminal_registry

        registry = get_terminal_registry()
        if task.conversation_id not in registry.active_conversation_ids():
            # No manager for this conversation — either server
            # restarted after the task was started (shell state
            # lost) or the manager was reaped.
            return _cancel_unavailable_json(task, prior_status, reason="shell_unavailable")
        from pathlib import Path as _Path

        # workspace arg is unused on cache hit; manager already exists.
        manager = registry.for_conversation(task.conversation_id, _Path("."))
        registered = manager.request_cancel(task.id)
        if not registered:
            return _cancel_unavailable_json(task, prior_status, reason="task_no_longer_running")
        return json.dumps(
            {
                "cancelled": True,
                "prior_status": prior_status,
                "task_id": task.id,
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
