"""Task store — manages task lifecycle and durable execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from agent_plane.entities import (
    CompletePendingToolCallResult,
    ConversationItem,
    NewConversationItem,
    PendingToolCall,
    Task,
)


class TaskStore(ABC):
    """
    Abstract base for task persistence and durable execution.

    Manages the full task lifecycle: creation, durable execution
    (start, stream, wait), the steering handshake (try_deliver,
    close_inbox), cancellation, and deletion.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the task store.

        The ``storage_location`` is a database URI. Concrete
        implementations also initialize the durable execution
        engine.

        :param storage_location: Database URI,
            e.g. ``"sqlite:///tasks.db"`` or
            ``"postgresql://user:pass@host/db"``.
        """
        self.storage_location = storage_location

    @abstractmethod
    def create(
        self,
        conversation_id: str,
        agent_id: str,
        agent_name: str,
        previous_response_id: str | None = None,
        background: bool = False,
        root_task_id: str | None = None,
        kind: str = "agent_task",
    ) -> Task:
        """
        Create a new task for executing an agent in the given
        conversation. Generates a unique task_id (which doubles as
        the response_id), stores the task record with
        status="queued", and returns the Task. Does not start
        execution -- call :meth:`start` to begin.

        ``agent_name`` is persisted so the API can return the
        original model name even if the agent is later renamed
        or deleted.

        ``instructions`` and ``reasoning`` are NOT stored in the
        task row -- they are pure workflow inputs passed to
        :meth:`start`.

        :param conversation_id: ID of the conversation this task
            belongs to, e.g. ``"conv_abc123"``.
        :param agent_id: ID of the agent to execute,
            e.g. ``"agent_xyz789"``.
        :param agent_name: Human-readable agent name at creation
            time, e.g. ``"code-assistant"``.
        :param previous_response_id: ID of the prior response in
            the thread, or ``None`` for the first turn,
            e.g. ``"resp_def456"``.
        :param background: Whether this is a background task
            (``True``) or blocking (``False``).
        :param root_task_id: ID of the top-level task that
            initiated this sub-agent's spawn tree. ``None``
            for top-level tasks, e.g. ``"task_abc123"``.
        :param kind: Task kind discriminator, one of
            ``"agent_task"`` (default — user-initiated turn),
            ``"tool"`` (background ``@tool(synchronous=False)``),
            ``"sub_agent"`` (sub-agent workflow, Phase 3), or
            ``"client_tool"`` (async client-side tool, Phase 5).
            Set explicitly per G74 by every spawn site so
            ``list_tasks`` / ``check_task`` can classify rows
            correctly without relying on the default backfill.
        :returns: The newly created :class:`Task` with status
            ``"queued"``.
        """
        ...

    @abstractmethod
    def start(
        self,
        task_id: str,
        instructions: str | None = None,
        reasoning: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Begin execution of a previously created task.

        Launch the workflow asynchronously and return immediately.

        The task remains ``"queued"`` until the workflow actually
        begins running, at which point it transitions to
        ``"in_progress"``. The task must exist and be in
        ``"queued"`` status.

        ``instructions``, ``reasoning``, and ``tools`` are passed
        as workflow inputs (persisted for crash recovery, not in
        the tasks table).

        Enforces the task/workflow invariant via a compensating
        transaction: if the workflow fails to start, the task row
        is deleted so neither artifact exists.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param instructions: Optional per-request system
            instructions override.
        :param reasoning: Optional reasoning configuration,
            e.g. ``{"effort": "medium"}``.
        :param tools: Optional list of client-specified tool dicts
            in OpenAI function format extended with ``agent_plane``
            callback metadata. Each entry must include
            ``agent_plane.callback.url``. ``None`` and ``[]`` are
            equivalent (no client tools), e.g.
            ``[{"type": "function", "function": {...},
            "agent_plane": {"callback": {"url": "..."}}}]``.
        """
        ...

    @abstractmethod
    def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """
        Yield streaming events as they are produced by the runtime.

        Awaits until the next event is available. The iterator ends
        when the task completes or is cancelled. Each event is a
        dict with a ``"type"`` field (e.g. ``"text_delta"``,
        ``"tool_call"``). Async because it long-polls for events.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: An async iterator of event dicts.
        """
        ...

    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """
        Return a fully enriched snapshot of the task's current state.

        Combines persisted DB state with runtime execution state
        (status, output) for a complete picture. Output is populated
        only when status is ``"completed"``; for other terminal
        states the output list is empty. Returns ``None`` if the
        task does not exist.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The :class:`Task` snapshot, or ``None``.
        """
        ...

    @abstractmethod
    def get_sync(self, task_id: str) -> Task | None:
        """
        Synchronous equivalent of :meth:`get`.

        Returns the fully enriched task (DB state + runtime
        execution state). Safe to call from synchronous contexts
        (workflow code, tool implementations) where ``await`` is
        not available.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The enriched :class:`Task` snapshot, or ``None``.
        """
        ...

    @abstractmethod
    async def wait(self, task_id: str, timeout: float | None = None) -> Task:
        """
        Await until the task reaches a terminal state and return
        the final Task.

        Terminal states: completed, failed, incomplete, or
        cancelled. Used by the server for blocking mode
        (``background=False``). Async because it blocks until
        completion.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param timeout: Maximum seconds to wait. ``None``
            blocks indefinitely (current behavior). If the
            deadline expires, returns the task in its current
            (non-terminal) state instead of raising.
        :returns: The final :class:`Task` in a terminal state,
            or the current state if timeout expired.
        """
        ...

    @abstractmethod
    def wait_sync(self, task_id: str, timeout: float | None = None) -> Task:
        """
        Synchronous equivalent of :meth:`wait`.

        Blocks until the task reaches a terminal state and returns
        the final enriched :class:`Task`. Safe to call from
        synchronous contexts (workflow code, tool implementations)
        where ``await`` is not available.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param timeout: Maximum seconds to wait. ``None`` blocks
            indefinitely. If the deadline expires, returns the
            task in its current (non-terminal) state.
        :returns: The final :class:`Task` in a terminal state,
            or the current state if timeout expired.
        :raises LookupError: If the task does not exist.
        """
        ...

    @abstractmethod
    def try_deliver(
        self,
        task_id: str,
        conversation_id: str,
        item: NewConversationItem,
    ) -> bool:
        """
        Atomically deliver a steering message to a running task,
        or report that the inbox is already closed.

        Single transaction: if the agent's inbox is still open,
        appends the item to the conversation and returns ``True``.
        If the agent has already closed its inbox (finishing up),
        returns ``False`` -- the caller should create a new
        response instead.

        Server-side half of the steering handshake.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param conversation_id: ID of the conversation to append
            the item to, e.g. ``"conv_abc123"``.
        :param item: The :class:`NewConversationItem` to deliver
            to the running agent.
        :returns: ``True`` if the item was delivered, ``False``
            if the inbox was already closed.
        """
        ...

    @abstractmethod
    def close_inbox(
        self,
        task_id: str,
        conversation_id: str,
        last_seen_item_id: str | None,
    ) -> list[ConversationItem]:
        """
        Atomically attempt to close the inbox for a finishing task.

        Within a single transaction: queries for items in the
        conversation newer than ``last_seen_item_id`` (or all items
        if ``None``). If found, returns them (inbox stays open --
        agent must continue). If none found, sets
        ``inbox_closed=True`` and returns empty list (agent may
        complete). This is the agent-side half of the steering
        handshake.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param conversation_id: ID of the conversation to check
            for new items, e.g. ``"conv_abc123"``.
        :param last_seen_item_id: ID of the last item the agent
            has processed, or ``None`` to check all items,
            e.g. ``"msg_xyz789"``.
        :returns: List of unseen :class:`ConversationItem` objects
            (empty if inbox was successfully closed).
        """
        ...

    @abstractmethod
    def cancel(self, task_id: str) -> Task:
        """
        Stop execution and mark the task as cancelled.

        Sets the DBOS workflow status to ``CANCELLED`` via a
        single DB UPDATE — non-blocking. The workflow observes
        the cancellation on its next DBOS checkpoint and winds
        down (close_stream, inbox drain happen in the workflow's
        finally block). The task record is preserved —
        :meth:`get` still works, and the response can be
        referenced as ``previous_response_id`` to continue or
        redirect the conversation.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The cancelled :class:`Task`.
        """
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """
        Remove a task record entirely.

        If in progress, stops the workflow first and waits for
        the finally block to complete. Then deletes the record
        -- subsequent :meth:`get` returns ``None``. Async because
        stopping an in-progress workflow may block while the
        finally block runs.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        """
        ...

    async def delete_all(
        self,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """
        Cancel and delete all tasks matching the filter.

        Delegates to :meth:`list_tasks` + :meth:`delete` so
        workflow cleanup is honoured.

        :param agent_id: Optional agent ID filter,
            e.g. ``"agent_xyz789"``.
        :param conversation_id: Optional conversation ID filter,
            e.g. ``"conv_abc123"``.
        """
        for task in await self.list_tasks(agent_id=agent_id, conversation_id=conversation_id):
            await self.delete(task.id)

    @abstractmethod
    async def list_tasks(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
        root_task_id: str | None = None,
    ) -> list[Task]:
        """
        Return tasks matching the given filters.

        All filters are optional and combined with AND. Used by
        the route layer to find in-flight tasks for cancellation
        (e.g. before deleting an agent or conversation).

        :param conversation_id: Optional conversation ID filter,
            e.g. ``"conv_abc123"``.
        :param agent_id: Optional agent ID filter,
            e.g. ``"agent_xyz789"``.
        :param root_task_id: Optional root task ID filter. When
            set, returns only sub-agent tasks spawned under this
            root, e.g. ``"task_abc123"``.
        :returns: A list of matching :class:`Task` objects,
            ordered by ``created_at`` descending.
        """
        ...

    @abstractmethod
    def list_tasks_sync(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
        root_task_id: str | None = None,
    ) -> list[Task]:
        """
        Synchronous variant of :meth:`list_tasks`.

        Required by sync builtin tools (e.g. Phase 4
        ``send_to_sub_agent``) that run inside the workflow's
        thread executor and cannot await the async variant.

        Same filter semantics as :meth:`list_tasks`.

        :param conversation_id: Optional conversation ID filter.
        :param agent_id: Optional agent ID filter.
        :param root_task_id: Optional root task ID filter.
        :returns: A list of matching :class:`Task` objects,
            ordered by ``created_at`` descending.
        """
        ...

    @abstractmethod
    async def finalize_async_task(
        self,
        *,
        task_id: str,
        status: str,
        output: str = "",
        error: dict[str, str] | None = None,
    ) -> None:
        """
        Phase 5: mark a non-DBOS task terminal with a result.

        Used by the PATCH ``async_tool_results`` handler to
        record the client's reported outcome on a
        ``kind="client_tool"`` task. Tasks WITH a DBOS
        workflow should NOT use this — DBOS itself is the
        source of truth for their status.

        :param task_id: The task's id.
        :param status: Terminal status — one of ``"completed"``,
            ``"failed"``, ``"cancelled"``.
        :param output: The string output for ``"completed"``.
        :param error: For ``"failed"`` only — dict with
            ``message`` and ``traceback`` keys.
        :raises LookupError: If the task does not exist.
        """
        ...

    # ── Pending tool call methods ─────────────────────────

    @abstractmethod
    def create_pending_tool_call(
        self,
        call_id: str,
        root_task_id: str,
        task_id: str,
        tool_name: str,
        arguments: str,
    ) -> None:
        """
        Insert a routing entry for a tunneled client-side tool call.

        Uses INSERT ON CONFLICT DO NOTHING for replay safety.

        :param call_id: The tool call ID (PK),
            e.g. ``"call_abc123"``.
        :param root_task_id: The root task whose response output
            contains the function_call item,
            e.g. ``"task_root1"``.
        :param task_id: The parked sub-agent's task ID,
            e.g. ``"task_sub2"``.
        :param tool_name: The tool function name,
            e.g. ``"Read"``.
        :param arguments: JSON-encoded arguments from the LLM,
            e.g. ``'{"file_path": "/tmp/foo.py"}'``.
        """
        ...

    @abstractmethod
    def complete_pending_tool_call(
        self,
        call_id: str,
        result: str,
    ) -> CompletePendingToolCallResult:
        """
        Attempt to mark a pending tool call as completed.

        Checks three conditions in order:

        1. Row exists? If not, returns ``NOT_FOUND``.
        2. Row already completed? Returns ``ALREADY_COMPLETED``
           (no-op, first writer wins).
        3. Sub-agent task still running? If terminal, returns
           ``SUB_AGENT_DONE`` (row is NOT updated).
        4. Otherwise, UPDATEs to completed and returns
           ``COMPLETED``.

        :param call_id: The tool call ID,
            e.g. ``"call_abc123"``.
        :param result: The tool's string output from the
            client.
        :returns: The outcome -- caller maps to HTTP status
            codes.
        """
        ...

    @abstractmethod
    def list_pending_tool_calls(
        self,
        *,
        task_id: str | None = None,
        root_task_id: str | None = None,
        call_id: str | None = None,
        status: str | None = None,
    ) -> list[PendingToolCall]:
        """
        Query pending tool calls with optional filters.

        All filters are AND-ed. At least one filter should be
        provided to avoid scanning the entire table.

        :param task_id: Filter by sub-agent task ID,
            e.g. ``"task_sub2"``.
        :param root_task_id: Filter by root task ID,
            e.g. ``"task_root1"``.
        :param call_id: Filter by tool call ID,
            e.g. ``"call_abc123"``.
        :param status: Filter by status.
            ``"completed"`` returns only delivered results.
            ``"action_required"`` returns only waiting calls.
            ``None`` skips status filtering.
        :returns: Matching pending tool call rows.
        """
        ...
