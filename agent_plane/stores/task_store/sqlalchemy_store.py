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
from sqlalchemy.orm import Session

from agent_plane.db.db_models import SqlConversationItem, SqlTask
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
    ConversationItem,
    NewConversationItem,
    Task,
    TaskStatus,
    parse_item_data,
)
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
    """Map a DBOS WorkflowStatusString value to our TaskStatus."""
    # Fallback to FAILED: if DBOS introduces a status we haven't mapped,
    # treating it as failed is the safest option — it surfaces the gap
    # via the API rather than silently misclassifying the task.
    return _DBOS_TO_TASK_STATUS.get(dbos_status_value, TaskStatus.FAILED)


# ── Row → entity helpers ─────────────────────────────────


def _to_entity(row: SqlTask) -> Task:
    """
    Build a Task from a DB row with status="queued" as default.
    Call _enrich_from_dbos() afterwards to merge DBOS workflow state
    (including instructions and reasoning from workflow inputs).
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
        status=TaskStatus.QUEUED,
        # instructions and reasoning are populated by _apply_workflow_status
        # from DBOS workflow inputs, not from the DB row.
    )


def _apply_workflow_status(task: Task, wf_status: WorkflowStatus) -> Task:
    """
    Apply a DBOS WorkflowStatus to a Task, populating status/output/error
    and restoring instructions/reasoning from the workflow inputs.
    """
    # wf_status.status is a plain string (e.g. "SUCCESS", "PENDING")
    task.status = _map_dbos_status(str(wf_status.status))

    # Restore instructions and reasoning from DBOS workflow inputs.
    # These are passed as kwargs to start_workflow() and stored by DBOS.
    if wf_status.input is not None:
        kwargs: dict[str, Any] = wf_status.input.get("kwargs", {})
        task.instructions = kwargs.get("instructions")
        task.reasoning = kwargs.get("reasoning")

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


def _enrich_from_dbos(task: Task) -> Task:
    """
    Sync enrichment — merge DBOS workflow state into a Task.
    Must NOT be called from an async context (DBOS will raise).
    """
    wf_status: WorkflowStatus | None = get_workflow_status(task.id)
    if wf_status is None:
        return task
    return _apply_workflow_status(task, wf_status)


async def _enrich_from_dbos_async(task: Task) -> Task:
    """
    Async enrichment — for use in async methods where an event
    loop is already running.
    """
    wf_status: WorkflowStatus | None = await get_workflow_status_async(task.id)
    if wf_status is None:
        return task
    return _apply_workflow_status(task, wf_status)


def _row_to_item(row: SqlConversationItem) -> ConversationItem:
    return ConversationItem(
        id=row.id,
        type=row.type,
        status=row.status,
        response_id=row.response_id,
        created_at=row.created_at,
        data=parse_item_data(row.type, json.loads(row.data)),
    )


class SqlAlchemyTaskStore(TaskStore):
    def __init__(self, storage_location: str) -> None:
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
    ) -> Task:
        row = SqlTask(
            id=generate_task_id(),
            agent_id=agent_id,
            agent_name=agent_name,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            created_at=now_epoch(),
            inbox_closed=False,
            background=background,
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
    ) -> None:
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
        # read_stream_async yields events as they arrive from the
        # DBOS stream. Layer 2 will add richer event types.
        async for event in read_stream_async(task_id, "output"):
            yield event

    def get(self, task_id: str) -> Task | None:
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                return None
            task = _to_entity(row)
        return _enrich_from_dbos(task)

    async def _get_async(self, task_id: str) -> Task | None:
        """Async variant of get() for use in async methods."""
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                return None
            task = _to_entity(row)
        return await _enrich_from_dbos_async(task)

    async def wait(self, task_id: str) -> Task:
        handle: WorkflowHandleAsync[dict[str, Any]] = await retrieve_workflow_async(task_id)
        await handle.get_result()
        task = await self._get_async(task_id)
        if task is None:
            raise LookupError(f"task {task_id!r} not found")
        return task

    # ── Steering handshake (DB-only, unchanged) ──────────

    def _get_task_for_update(self, session: Session, task_id: str) -> SqlTask | None:
        """
        Fetch a task row with FOR UPDATE locking on PostgreSQL.
        SQLite relies on database-level locking instead.
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
        Server-side steering handshake. Within a single transaction:
        check inbox_closed; if open, append the item and return True;
        if closed, return False.

        This inserts into conversation_items directly (rather than
        delegating to ConversationStore.append) because the inbox check
        and the item insert MUST be atomic — if they were separate
        operations, close_inbox could run between them, and the
        delivered message would never be seen by the agent.
        """
        with self._session() as session:
            row = self._get_task_for_update(session, task_id)
            if row is None or row.inbox_closed:
                return False

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
        Agent-side steering handshake. Within a single transaction:
        check for items newer than last_seen_item_id. If found, return
        them (inbox stays open). If none, set inbox_closed=1 and return [].
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
        await cancel_workflow_async(task_id)
        task = await self._get_async(task_id)
        if task is None:
            raise LookupError(f"task {task_id!r} not found")
        return task

    async def delete(self, task_id: str) -> None:
        # Cancel the DBOS workflow if it's still running
        wf_status = await get_workflow_status_async(task_id)
        if wf_status is not None and str(wf_status.status) in _DBOS_ACTIVE:
            await cancel_workflow_async(task_id)

        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row:
                session.delete(row)

    # ── List ──────────────────────────────────────────────

    def list_tasks(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[Task]:
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

        return [_enrich_from_dbos(t) for t in tasks]
