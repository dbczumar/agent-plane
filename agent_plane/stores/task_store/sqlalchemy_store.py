"""SQLAlchemy-backed task store with DBOS durable execution.

DB-only methods (create, try_deliver, close_inbox, list_tasks) use
SQLAlchemy directly. Execution-related methods (start, get, stream,
wait, cancel, delete) delegate to the DBOS runtime.

instructions and reasoning are pure workflow inputs — stored by DBOS
(in workflow_status.input), not in the tasks table. The task row holds
only relationship/identity columns (agent_id, conversation_id, etc.)
and the steering handshake flag (inbox_closed). TaskStore.get()
assembles the full Task entity from both the DB row and DBOS state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from agent_plane.db.db_models import (
    SqlConversation,
    SqlConversationItem,
    SqlPendingToolCall,
    SqlTask,
)
from agent_plane.db.utils import (
    ensure_fts_table,
    extract_search_text,
    generate_item_id,
    generate_task_id,
    get_or_create_engine,
    insert_fts,
    make_managed_session_maker,
    now_epoch,
)
from agent_plane.entities import (
    CompletePendingToolCallResult,
    ConversationItem,
    NewConversationItem,
    PendingToolCall,
    Task,
    TaskStatus,
    parse_item_data,
)
from agent_plane.entities.task import TERMINAL_STATUSES
from agent_plane.runtime.durability import (
    SetWorkflowID,
    WorkflowHandleAsync,
    WorkflowStatus,
    WorkflowStatusString,
    cancel_workflow_async,
    ensure_dbos,
    get_workflow_status,
    get_workflow_status_async,
    read_stream_async,
    retrieve_workflow_async,
    start_workflow,
)
from agent_plane.stores.task_store import TaskStore

_logger = logging.getLogger(__name__)

# ── DBOS → Task status mapping ───────────────────────────

_DBOS_TO_TASK_STATUS: dict[str, str] = {
    WorkflowStatusString.ENQUEUED.value: TaskStatus.QUEUED,
    WorkflowStatusString.PENDING.value: TaskStatus.IN_PROGRESS,
    WorkflowStatusString.SUCCESS.value: TaskStatus.COMPLETED,
    WorkflowStatusString.ERROR.value: TaskStatus.FAILED,
    WorkflowStatusString.CANCELLED.value: TaskStatus.CANCELLED,
    WorkflowStatusString.MAX_RECOVERY_ATTEMPTS_EXCEEDED.value: TaskStatus.FAILED,
}

# DBOS statuses that mean the workflow is still running.
_DBOS_ACTIVE = frozenset({WorkflowStatusString.PENDING.value, WorkflowStatusString.ENQUEUED.value})


def _map_dbos_status(dbos_status_value: str) -> str:
    """
    Map a DBOS ``WorkflowStatusString`` value to a
    :class:`TaskStatus` string.

    Falls back to :attr:`TaskStatus.FAILED` for unknown statuses
    so that unmapped DBOS states surface as failures rather than
    silently misclassifying the task.

    :param dbos_status_value: The raw DBOS status string,
        e.g. ``"SUCCESS"``, ``"PENDING"``.
    :returns: The corresponding :class:`TaskStatus` value,
        e.g. ``"completed"``, ``"in_progress"``.
    """
    # Fallback to FAILED: if DBOS introduces a status we haven't mapped,
    # treating it as failed is the safest option — it surfaces the gap
    # via the API rather than silently misclassifying the task.
    return _DBOS_TO_TASK_STATUS.get(dbos_status_value, TaskStatus.FAILED)


# ── Row → entity helpers ─────────────────────────────────


def _to_entity(row: SqlTask) -> Task:
    """
    Build a :class:`Task` from a DB row with ``status="queued"``
    as the default.

    Call :func:`_enrich_from_dbos` afterwards to merge DBOS
    workflow state (status, output, error, instructions, reasoning).

    :param row: The :class:`SqlTask` ORM row to convert.
    :returns: A :class:`Task` dataclass instance with status
        defaulting to ``"queued"``.
    """
    return Task(
        id=row.id,
        conversation_id=row.conversation_id,
        agent_id=row.agent_id,
        agent_name=row.agent_name,
        created_at=row.created_at,
        inbox_closed=row.inbox_closed,
        previous_response_id=row.previous_response_id,
        background=row.background,
        root_task_id=row.root_task_id,
        status=TaskStatus.QUEUED,
        # instructions and reasoning are populated by _apply_workflow_status
        # from DBOS workflow inputs, not from the DB row.
    )


def _apply_workflow_status(task: Task, wf_status: WorkflowStatus) -> Task:
    """
    Apply a DBOS :class:`WorkflowStatus` to a :class:`Task`,
    populating status, output, error, and restoring instructions
    and reasoning from the workflow inputs.

    :param task: The :class:`Task` to mutate.
    :param wf_status: The DBOS workflow status containing the
        execution state.
    :returns: The same :class:`Task` instance, mutated in place.
    """
    # wf_status.status is a plain string (e.g. "SUCCESS", "PENDING")
    task.status = _map_dbos_status(str(wf_status.status))

    # Restore instructions, reasoning, and tools from DBOS workflow inputs.
    # These are passed as kwargs to start_workflow() and stored by DBOS.
    if wf_status.input is not None:
        kwargs: dict[str, Any] = wf_status.input.get("kwargs", {})
        task.instructions = kwargs.get("instructions")
        task.reasoning = kwargs.get("reasoning")
        task.tools = kwargs.get("tools")

    if task.status == TaskStatus.COMPLETED and wf_status.output is not None:
        # The workflow returns {"task_id": ..., "output": [...], ...}
        result: dict[str, Any] = wf_status.output
        task.output = result["output"]
        task.usage = result.get("usage")
        task.completed_at = result.get("completed_at")

    if task.status == TaskStatus.FAILED and wf_status.error is not None:
        task.error = {
            "code": "runtime_error",
            "message": str(wf_status.error),
        }

    return task


async def _enrich_from_dbos(task: Task) -> Task:
    """
    Merge DBOS workflow state (status, output, error) into a
    :class:`Task`.

    Fetches the workflow status from DBOS and applies it. If no
    workflow status exists (e.g. workflow not yet registered),
    returns the task unchanged.

    :param task: The :class:`Task` to enrich.
    :returns: The enriched :class:`Task`.
    """
    wf_status: WorkflowStatus | None = await get_workflow_status_async(task.id)
    if wf_status is None:
        return task
    return _apply_workflow_status(task, wf_status)


def _row_to_item(row: SqlConversationItem) -> ConversationItem:
    """
    Convert a :class:`SqlConversationItem` ORM row to a
    :class:`ConversationItem` entity.

    Deserializes the JSON ``data`` column and parses it into
    the appropriate typed data model.

    :param row: The SQLAlchemy ORM row to convert.
    :returns: A :class:`ConversationItem` Pydantic model.
    """
    return ConversationItem(
        id=row.id,
        type=row.type,
        status=row.status,
        response_id=row.response_id,
        created_at=row.created_at,
        data=parse_item_data(row.type, json.loads(row.data)),
    )


class SqlAlchemyTaskStore(TaskStore):
    """
    SQLAlchemy-backed implementation of :class:`TaskStore` with DBOS
    durable execution.

    DB-only methods (create, try_deliver, close_inbox, list_tasks)
    use SQLAlchemy directly. Execution-related methods (start, get,
    stream, wait, cancel, delete) delegate to the DBOS runtime.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the SQLAlchemy task store.

        Creates or reuses a SQLAlchemy engine and session factory,
        ensures the FTS virtual table exists, and initializes
        the DBOS durable execution engine.

        :param storage_location: SQLAlchemy database URI,
            e.g. ``"sqlite:///tasks.db"`` or
            ``"postgresql://user:pass@host/db"``.
        """
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)
        self._supports_for_update = self._engine.dialect.name != "sqlite"
        ensure_fts_table(self._engine)
        ensure_dbos(storage_location)

    # ── Create ────────────────────────────────────────────

    def create(
        self,
        conversation_id: str,
        agent_id: str,
        agent_name: str,
        previous_response_id: str | None = None,
        background: bool = False,
        root_task_id: str | None = None,
    ) -> Task:
        """
        Create a new task in the database.

        :param conversation_id: ID of the conversation,
            e.g. ``"conv_abc123"``.
        :param agent_id: ID of the agent to execute,
            e.g. ``"agent_xyz789"``.
        :param agent_name: Human-readable agent name,
            e.g. ``"code-assistant"``.
        :param previous_response_id: ID of the prior response
            in the thread, or ``None`` for the first turn.
        :param background: Whether this is a background task.
        :param root_task_id: ID of the top-level task for
            sub-agent spawns. ``None`` for top-level tasks.
        :returns: The newly created :class:`Task` with status
            ``"queued"``.
        """
        row = SqlTask(
            id=generate_task_id(),
            agent_id=agent_id,
            agent_name=agent_name,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            created_at=now_epoch(),
            inbox_closed=False,
            background=background,
            root_task_id=root_task_id,
        )
        with self._session() as session:
            session.add(row)
            return _to_entity(row)

    # ── DBOS-backed execution methods ─────────────────────

    def start(
        self,
        task_id: str,
        instructions: str | None = None,
        reasoning: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Begin execution of a previously created task.

        Launches the DBOS workflow asynchronously pinned to the
        task_id. If the workflow fails to start, deletes the
        orphaned task row (compensating transaction).

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param instructions: Optional per-request system
            instructions override.
        :param reasoning: Optional reasoning configuration,
            e.g. ``{"effort": "medium"}``.
        :param tools: Optional list of client-specified tool dicts
            (OpenAI format with ``agent_plane`` extension). Passed
            through to the workflow as a DBOS input kwarg so they
            are checkpointed and restored on recovery.
        :raises LookupError: If the task does not exist.
        """
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                raise LookupError(f"task {task_id!r} not found")
            agent_id = row.agent_id
            conversation_id = row.conversation_id
            previous_response_id = row.previous_response_id

        # Lazy import: workflow module uses DBOS decorators that
        # require DBOS to be initialized (which happens in __init__).
        from agent_plane.runtime.workflow import agent_execution_workflow

        try:
            # Pin the DBOS workflow_uuid to our task_id so we can
            # retrieve workflow state by task_id later.
            # Pass optional params as kwargs so DBOS stores them
            # with named keys in workflow_status.input.kwargs.
            with SetWorkflowID(task_id):
                start_workflow(
                    agent_execution_workflow,
                    agent_id,
                    conversation_id,
                    previous_response_id=previous_response_id,
                    instructions=instructions,
                    reasoning=reasoning,
                    tools=tools,
                )
        except Exception:
            # Compensating transaction: delete the orphaned task row
            # so the invariant holds (task row exists ↔ DBOS workflow
            # exists).
            _logger.warning(
                "DBOS workflow failed to start for task %s; deleting orphaned row",
                task_id,
            )
            with self._session() as session:
                orphan = session.get(SqlTask, task_id)
                if orphan is not None:
                    session.delete(orphan)
            raise

    async def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """
        Yield streaming events from the DBOS output stream.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: An async iterator of event dicts, each with a
            ``"type"`` field (e.g. ``"text_delta"``).
        """
        # read_stream_async yields events as they arrive from the
        # DBOS stream. Layer 2 will add richer event types.
        async for event in read_stream_async(task_id, "output"):
            yield event

    async def get(self, task_id: str) -> Task | None:
        """
        Return a snapshot of the task's current state, enriched
        with DBOS workflow state.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The :class:`Task` snapshot, or ``None`` if the
            task does not exist.
        """
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                return None
            task = _to_entity(row)
        return await _enrich_from_dbos(task)

    def get_sync(self, task_id: str) -> Task | None:
        """
        Synchronous DB-only read of a task row.

        Returns the task entity **without** DBOS workflow enrichment.
        Used by DBOS workflow code and tool implementations that
        run inside an event loop and cannot call async methods.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The :class:`Task` snapshot, or ``None`` if the
            task does not exist.
        """
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                return None
            return _to_entity(row)

    async def wait(self, task_id: str, timeout: float | None = None) -> Task:
        """
        Await until the task reaches a terminal state and return
        the final :class:`Task`.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param timeout: Maximum seconds to wait. ``None`` blocks
            indefinitely. If the deadline expires, returns the
            task in its current (non-terminal) state.
        :returns: The final :class:`Task` in a terminal state,
            or the current state if timeout expired.
        :raises LookupError: If the task does not exist after
            completion.
        """
        import asyncio

        handle: WorkflowHandleAsync[dict[str, Any]] = await retrieve_workflow_async(task_id)
        try:
            if timeout is not None:
                await asyncio.wait_for(handle.get_result(), timeout=timeout)
            else:
                await handle.get_result()
        except asyncio.TimeoutError:
            # Deadline expired — return current (non-terminal) state
            pass
        task = await self.get(task_id)
        if task is None:
            raise LookupError(f"task {task_id!r} not found")
        return task

    # ── Steering handshake (DB-only, unchanged) ──────────

    def _lock_conversation(self, session: Session, conversation_id: str) -> None:
        """
        Acquire a row-level lock on the conversation to serialize
        position writes.

        On PostgreSQL, issues ``SELECT ... FOR UPDATE`` on the
        conversation row. On SQLite, this is a no-op because
        database-level locking already serializes transactions.

        :param session: The active SQLAlchemy session.
        :param conversation_id: The conversation to lock,
            e.g. ``"conv_abc123"``.
        """
        if self._supports_for_update:
            stmt = (
                select(SqlConversation.id)
                .where(SqlConversation.id == conversation_id)
                .with_for_update()
            )
            session.execute(stmt)

    def _get_task_for_update(self, session: Session, task_id: str) -> SqlTask | None:
        """
        Fetch a task row with ``FOR UPDATE`` locking on PostgreSQL.
        SQLite relies on database-level locking instead.

        :param session: The active SQLAlchemy session.
        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The :class:`SqlTask` row, or ``None`` if not
            found.
        """
        stmt = select(SqlTask).where(SqlTask.id == task_id)
        if self._supports_for_update:
            stmt = stmt.with_for_update()
        return session.execute(stmt).scalar_one_or_none()

    def try_deliver(
        self,
        task_id: str,
        conversation_id: str,
        item: NewConversationItem,
    ) -> bool:
        """
        Server-side steering handshake.

        Within a single transaction: check ``inbox_closed``; if
        open, append the item and return ``True``; if closed,
        return ``False``.

        Inserts into ``conversation_items`` directly (rather than
        delegating to ``ConversationStore.append``) because the
        inbox check and the item insert MUST be atomic.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param conversation_id: ID of the conversation to append
            the item to, e.g. ``"conv_abc123"``.
        :param item: The :class:`NewConversationItem` to deliver.
        :returns: ``True`` if the item was delivered, ``False``
            if the inbox was already closed.
        """
        with self._session() as session:
            row = self._get_task_for_update(session, task_id)
            if row is None or row.inbox_closed:
                return False

            # Lock the conversation row to serialize position writes
            # with concurrent append() calls. On PostgreSQL this is a
            # row-level FOR UPDATE lock; on SQLite the database-level
            # lock already serializes.
            self._lock_conversation(session, conversation_id)

            max_pos: int = session.execute(
                select(func.coalesce(func.max(SqlConversationItem.position), -1)).where(
                    SqlConversationItem.conversation_id == conversation_id
                )
            ).scalar_one()

            data_dict = item.data.model_dump(exclude_none=True)
            search = extract_search_text(item)
            item_id = generate_item_id(item.type)
            session.add(
                SqlConversationItem(
                    id=item_id,
                    conversation_id=conversation_id,
                    response_id=item.response_id,
                    created_at=now_epoch(),
                    status="completed",
                    position=max_pos + 1,
                    type=item.type,
                    data=json.dumps(data_dict),
                    search_text=search,
                )
            )
            # Dual-write to FTS so steered messages are searchable.
            insert_fts(session, item_id, conversation_id, search)
            return True

    def close_inbox(
        self,
        task_id: str,
        conversation_id: str,
        last_seen_item_id: str | None,
    ) -> list[ConversationItem]:
        """
        Agent-side steering handshake.

        Within a single transaction: check for items newer than
        ``last_seen_item_id``. If found, return them (inbox stays
        open). If none, set ``inbox_closed=True`` and return
        an empty list.

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
        with self._session() as session:
            stmt = select(SqlConversationItem).where(
                SqlConversationItem.conversation_id == conversation_id
            )
            if last_seen_item_id is not None:
                sub = (
                    select(SqlConversationItem.position)
                    .where(SqlConversationItem.id == last_seen_item_id)
                    .scalar_subquery()
                )
                stmt = stmt.where(SqlConversationItem.position > sub)

            stmt = stmt.order_by(SqlConversationItem.position.asc())
            new_rows = list(session.execute(stmt).scalars().all())

            if new_rows:
                return [_row_to_item(r) for r in new_rows]

            row = self._get_task_for_update(session, task_id)
            if row is not None:
                row.inbox_closed = True
            return []

    # ── Cancel / Delete ───────────────────────────────────

    async def cancel(self, task_id: str) -> Task:
        """
        Cancel a task by stopping its DBOS workflow.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The cancelled :class:`Task`.
        :raises LookupError: If the task does not exist after
            cancellation.
        """
        await cancel_workflow_async(task_id)
        task = await self.get(task_id)
        if task is None:
            raise LookupError(f"task {task_id!r} not found")
        return task

    async def delete(self, task_id: str) -> None:
        """
        Remove a task record entirely.

        Cancels any active DBOS workflow first, then deletes the
        database row.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        """
        # Cancel the DBOS workflow if it's still running
        wf_status = await get_workflow_status_async(task_id)
        if wf_status is not None and str(wf_status.status) in _DBOS_ACTIVE:
            await cancel_workflow_async(task_id)

        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row:
                session.delete(row)

    # ── List ──────────────────────────────────────────────

    async def list_tasks(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[Task]:
        """
        Return tasks matching the given filters, enriched with
        DBOS workflow state.

        :param conversation_id: Optional conversation ID filter,
            e.g. ``"conv_abc123"``.
        :param agent_id: Optional agent ID filter,
            e.g. ``"agent_xyz789"``.
        :returns: A list of matching :class:`Task` objects,
            ordered by ``created_at`` descending.
        """
        with self._session() as session:
            stmt = select(SqlTask)
            if conversation_id:
                stmt = stmt.where(SqlTask.conversation_id == conversation_id)
            if agent_id:
                stmt = stmt.where(SqlTask.agent_id == agent_id)
            # Not paginated (internal use only), but ordered for determinism.
            stmt = stmt.order_by(SqlTask.created_at.desc())
            rows = list(session.execute(stmt).scalars().all())
            tasks = [_to_entity(r) for r in rows]

        enriched: list[Task] = []
        for t in tasks:
            enriched.append(await _enrich_from_dbos(t))
        return enriched

    # ── Pending tool call helpers ─────────────────────────

    def _is_sub_agent_terminal(self, task_id: str) -> bool:
        """
        Check whether a sub-agent's DBOS workflow has reached a
        terminal state.

        Uses the sync DBOS status API. Returns ``True`` if the
        mapped task status is in :data:`TERMINAL_STATUSES`, or if
        no workflow status exists (workflow never started).

        :param task_id: The sub-agent's task ID,
            e.g. ``"task_sub2"``.
        :returns: ``True`` if terminal, ``False`` if still active.
        """
        wf_status: WorkflowStatus | None = get_workflow_status(task_id)
        if wf_status is None:
            # No workflow registered — treat as terminal (never ran).
            return True
        mapped = _map_dbos_status(str(wf_status.status))
        return mapped in TERMINAL_STATUSES

    # ── Pending tool call methods ─────────────────────────

    def create_pending_tool_call(
        self,
        call_id: str,
        root_task_id: str,
        task_id: str,
    ) -> None:
        """
        Insert a routing entry for a tunneled client-side tool
        call. Uses ON CONFLICT DO NOTHING for DBOS replay safety.

        :param call_id: The tool call ID (PK),
            e.g. ``"call_abc123"``.
        :param root_task_id: The root task whose response output
            contains the function_call item.
        :param task_id: The parked sub-agent's task ID.
        """
        with self._session() as session:
            # ON CONFLICT DO NOTHING: safe for DBOS replay — if the
            # row already exists from a previous attempt, skip it.
            stmt = (
                sqlite_insert(SqlPendingToolCall)
                .values(
                    call_id=call_id,
                    root_task_id=root_task_id,
                    task_id=task_id,
                    status="action_required",
                    result=None,
                    created_at=now_epoch(),
                    completed_at=None,
                )
                .on_conflict_do_nothing()
            )
            session.execute(stmt)

    def check_pending_tool_call(
        self,
        call_id: str,
    ) -> CompletePendingToolCallResult:
        """
        Check whether a pending tool call can be completed
        without mutating it.

        :param call_id: The tool call ID,
            e.g. ``"call_abc123"``.
        :returns: The outcome that ``complete_pending_tool_call``
            would return.
        """
        with self._session() as session:
            row = session.get(SqlPendingToolCall, call_id)
            if row is None:
                return CompletePendingToolCallResult.NOT_FOUND
            if row.status == "completed":
                return CompletePendingToolCallResult.ALREADY_COMPLETED
            if self._is_sub_agent_terminal(row.task_id):
                return CompletePendingToolCallResult.SUB_AGENT_DONE
            return CompletePendingToolCallResult.COMPLETED

    def complete_pending_tool_call(
        self,
        call_id: str,
        result: str,
    ) -> CompletePendingToolCallResult:
        """
        Attempt to mark a pending tool call as completed.

        :param call_id: The tool call ID,
            e.g. ``"call_abc123"``.
        :param result: The tool's string output from the client.
        :returns: The outcome enum value.
        """
        with self._session() as session:
            row = session.get(SqlPendingToolCall, call_id)
            if row is None:
                return CompletePendingToolCallResult.NOT_FOUND
            if row.status == "completed":
                return CompletePendingToolCallResult.ALREADY_COMPLETED
            # Check if sub-agent's DBOS workflow has reached a
            # terminal state. Uses the sync DBOS API because this
            # method is called from sync HTTP handlers.
            if self._is_sub_agent_terminal(row.task_id):
                return CompletePendingToolCallResult.SUB_AGENT_DONE
            row.status = "completed"
            row.result = result
            row.completed_at = now_epoch()
            return CompletePendingToolCallResult.COMPLETED

    def get_pending_tool_calls(
        self,
        task_id: str,
        status: str | None = None,
    ) -> list[PendingToolCall]:
        """
        Query pending tool calls for a task.

        :param task_id: The sub-agent's task ID.
        :param status: Optional status filter.
        :returns: Matching pending tool call rows.
        """
        with self._session() as session:
            stmt = select(SqlPendingToolCall).where(SqlPendingToolCall.task_id == task_id)
            if status is not None:
                stmt = stmt.where(SqlPendingToolCall.status == status)
            rows = list(session.execute(stmt).scalars().all())
            return [
                PendingToolCall(
                    call_id=r.call_id,
                    root_task_id=r.root_task_id,
                    task_id=r.task_id,
                    status=r.status,
                    result=r.result,
                    created_at=r.created_at,
                    completed_at=r.completed_at,
                )
                for r in rows
            ]
