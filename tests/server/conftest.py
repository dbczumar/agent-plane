"""Shared fixtures for server integration tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import select

from agent_plane.db.db_models import SqlTask
from agent_plane.db.utils import (
    ensure_fts_table,
    get_or_create_engine,
    make_managed_session_maker,
    now_epoch,
)
from agent_plane.entities import Task, TaskStatus
from agent_plane.server.app import create_app
from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from agent_plane.stores.task_store import TaskStore

# Canned output for completed tasks — a single assistant message.
_CANNED_OUTPUT: list[dict[str, Any]] = [
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hello from the test agent."}],
    }
]


class IntegrationTaskStore(TaskStore):
    """
    Task store for integration tests — no DBOS runtime dependency.

    Uses real SQLAlchemy for persistence (create, try_deliver, close_inbox)
    by replicating the essential DB logic from SqlAlchemyTaskStore.
    Replaces the DBOS runtime with deterministic in-memory behavior:

    - start() immediately marks the task as completed with canned output
    - stream() yields a single text delta event
    - wait() returns immediately (task is already completed)
    - get() reads from DB and applies in-memory completion/cancellation
    - cancel() marks the task as cancelled
    - delete() removes the DB row
    """

    def __init__(self, db_uri: str) -> None:
        super().__init__(db_uri)
        self._engine = get_or_create_engine(db_uri)
        self._session = make_managed_session_maker(self._engine)
        ensure_fts_table(self._engine)
        # In-memory runtime state (no DBOS)
        self._completed: set[str] = set()
        self._cancelled: set[str] = set()
        self._deleted: set[str] = set()

    # ── Internal helpers ──────────────────────────────────

    def _row_to_task(self, row: SqlTask) -> Task:
        """Build a Task from a DB row and apply in-memory runtime state."""
        task = Task(
            id=row.id,
            conversation_id=row.conversation_id,
            agent_id=row.agent_id,
            agent_name=row.agent_name,
            created_at=row.created_at,
            inbox_closed=row.inbox_closed,
            previous_response_id=row.previous_response_id,
            instructions=row.instructions,
            reasoning=json.loads(row.reasoning) if row.reasoning else None,
            background=row.background,
            status=TaskStatus.QUEUED,
        )
        if row.id in self._cancelled:
            task.status = TaskStatus.CANCELLED
        elif row.id in self._completed:
            task.status = TaskStatus.COMPLETED
            task.output = list(_CANNED_OUTPUT)
            task.completed_at = now_epoch()
        return task

    # ── DB-only methods (real SQL, no DBOS) ───────────────

    def create(
        self,
        conversation_id: str,
        agent_id: str,
        agent_name: str,
        instructions: str | None = None,
        reasoning: dict[str, str] | None = None,
        previous_response_id: str | None = None,
        background: bool = False,
    ) -> Task:
        from agent_plane.db.utils import generate_task_id

        row = SqlTask(
            id=generate_task_id(),
            agent_id=agent_id,
            agent_name=agent_name,
            conversation_id=conversation_id,
            previous_response_id=previous_response_id,
            created_at=now_epoch(),
            inbox_closed=False,
            instructions=instructions,
            reasoning=json.dumps(reasoning) if reasoning else None,
            background=background,
        )
        with self._session() as session:
            session.add(row)
            return self._row_to_task(row)

    def try_deliver(
        self,
        task_id: str,
        conversation_id: str,
        item: Any,
    ) -> bool:
        from agent_plane.db.db_models import SqlConversationItem
        from agent_plane.db.utils import (
            extract_search_text,
            generate_item_id,
            insert_fts,
        )
        from sqlalchemy import func

        with self._session() as session:
            stmt = select(SqlTask).where(SqlTask.id == task_id)
            row = session.execute(stmt).scalar_one_or_none()
            if row is None or row.inbox_closed:
                return False

            max_pos: int = session.execute(
                select(
                    func.coalesce(func.max(SqlConversationItem.position), -1)
                ).where(
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
            insert_fts(session, item_id, conversation_id, search)
            return True

    def close_inbox(
        self,
        task_id: str,
        conversation_id: str,
        last_seen_item_id: str | None,
    ) -> list[Any]:
        from agent_plane.db.db_models import SqlConversationItem
        from agent_plane.entities import ConversationItem, parse_item_data

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
                return [
                    ConversationItem(
                        id=r.id,
                        type=r.type,
                        status=r.status,
                        response_id=r.response_id,
                        created_at=r.created_at,
                        data=parse_item_data(r.type, json.loads(r.data)),
                    )
                    for r in new_rows
                ]

            row_stmt = select(SqlTask).where(SqlTask.id == task_id)
            row = session.execute(row_stmt).scalar_one_or_none()
            if row is not None:
                row.inbox_closed = True
            return []

    # ── Runtime-replacement methods (in-memory, no DBOS) ──

    def start(self, task_id: str) -> None:
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                raise LookupError(f"task {task_id!r} not found")
        self._completed.add(task_id)

    async def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "response.output_text.delta",
            "delta": "Hello from the test agent.",
        }

    def get(self, task_id: str) -> Task | None:
        if task_id in self._deleted:
            return None
        with self._session() as session:
            row = session.get(SqlTask, task_id)
            if row is None:
                return None
            return self._row_to_task(row)

    async def wait(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            raise LookupError(f"task {task_id!r} not found")
        return task

    async def cancel(self, task_id: str) -> Task:
        self._cancelled.add(task_id)
        self._completed.discard(task_id)
        task = self.get(task_id)
        if task is None:
            raise LookupError(f"task {task_id!r} not found")
        return task

    async def delete(self, task_id: str) -> None:
        self._deleted.add(task_id)
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
            stmt = stmt.order_by(SqlTask.created_at.desc())
            rows = list(session.execute(stmt).scalars().all())
            # Build tasks inside the session so row attributes are accessible
            tasks = [
                self._row_to_task(r)
                for r in rows
                if r.id not in self._deleted
            ]
        return tasks


# ── Fixtures ──────────────────────────────────────────


@pytest.fixture()
def app(db_uri: str, tmp_path: Path) -> FastAPI:
    """Build the FastAPI app with real stores and a stubbed task runtime."""
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        task_store=IntegrationTaskStore(db_uri),
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=LocalArtifactStore(str(tmp_path / "artifacts")),
    )


@pytest_asyncio.fixture()
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP client wired to the FastAPI app (no real server)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c
