"""SQLAlchemy-backed task store.

Implements the DB-only methods of TaskStore. DBOS-dependent methods
(start, stream, wait, cancel) raise NotImplementedError until the
DBOS integration is wired in.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent_plane.db.db_models import SqlConversationItem, SqlTask
from agent_plane.db.utils import (
    extract_search_text,
    generate_item_id,
    generate_task_id,
    get_or_create_engine,
    make_managed_session_maker,
    now_epoch,
)
from agent_plane.entities import (
    ConversationItem,
    NewConversationItem,
    Task,
    parse_item_data,
)
from agent_plane.stores.task_store import TaskStore


def _to_entity(row: SqlTask) -> Task:
    """
    Build a Task from a DB row. DBOS-managed fields (status, output,
    error, usage, etc.) use defaults until DBOS integration populates them.
    """
    return Task(
        task_id=row.id,
        conversation_id=row.conversation_id,
        agent_id=row.agent_id,
        created_at=row.created_at,
        inbox_closed=bool(row.inbox_closed),
        previous_response_id=row.previous_response_id,
        status="queued",
    )


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

    def create(
        self,
        conversation_id: str,
        agent_id: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        background: bool = False,
    ) -> Task:
        row = SqlTask(
            id=generate_task_id(),
            agent_id=agent_id,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            created_at=now_epoch(),
            inbox_closed=0,
        )
        with self._session() as session:
            session.add(row)
            return Task(
                task_id=row.id,
                conversation_id=row.conversation_id,
                agent_id=row.agent_id,
                created_at=row.created_at,
                status="queued",
                instructions=instructions,
                background=background,
                previous_response_id=previous_response_id,
            )

    def start(self, task_id: str) -> None:
        raise NotImplementedError("start() requires DBOS integration")

    def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("stream() requires DBOS integration")

    def get(self, task_id: str) -> Task | None:
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            return _to_entity(row) if row else None

    async def wait(self, task_id: str) -> Task:
        raise NotImplementedError("wait() requires DBOS integration")

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
            session.add(
                SqlConversationItem(
                    id=generate_item_id(item.type),
                    conversation_id=conversation_id,
                    response_id=item.response_id,
                    created_at=now_epoch(),
                    status="completed",
                    position=max_pos + 1,
                    type=item.type,
                    data=json.dumps(data_dict),
                    search_text=extract_search_text(item),
                )
            )
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
                row.inbox_closed = 1
            return []

    async def cancel(self, task_id: str) -> Task:
        raise NotImplementedError("cancel() requires DBOS integration")

    async def delete(self, task_id: str) -> None:
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row:
                session.delete(row)

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
            return [_to_entity(r) for r in rows]
