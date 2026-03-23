"""SQLAlchemy-backed agent store."""

from __future__ import annotations

from sqlalchemy import and_, or_, select

from agent_plane.db.db_models import SqlAgent
from agent_plane.db.utils import (
    generate_agent_id,
    get_or_create_engine,
    make_managed_session_maker,
    now_epoch,
)
from agent_plane.entities import Agent, PagedList
from agent_plane.stores.agent_store import AgentStore


def _to_entity(row: SqlAgent) -> Agent:
    return Agent(
        id=row.id,
        created_at=row.created_at,
        name=row.name,
        description=row.description,
    )


class SqlAlchemyAgentStore(AgentStore):
    def __init__(self, storage_location: str) -> None:
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)

    def create(
        self,
        name: str,
        description: str | None = None,
    ) -> Agent:
        row = SqlAgent(
            id=generate_agent_id(),
            created_at=now_epoch(),
            name=name,
            description=description,
        )
        with self._session() as session:
            session.add(row)
            return _to_entity(row)

    def get(self, agent_id: str) -> Agent | None:
        with self._session() as session:
            row = session.get(SqlAgent, agent_id)
            return _to_entity(row) if row else None

    def get_by_name(self, name: str) -> Agent | None:
        with self._session() as session:
            row = session.execute(
                select(SqlAgent).where(SqlAgent.name == name)
            ).scalar_one_or_none()
            return _to_entity(row) if row else None

    def list(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[Agent]:
        with self._session() as session:
            stmt = select(SqlAgent)
            if after:
                sub = select(SqlAgent.created_at).where(SqlAgent.id == after).scalar_subquery()
                stmt = stmt.where(
                    or_(
                        SqlAgent.created_at < sub,
                        and_(SqlAgent.created_at == sub, SqlAgent.id < after),
                    )
                )
            if before:
                sub = select(SqlAgent.created_at).where(SqlAgent.id == before).scalar_subquery()
                stmt = stmt.where(
                    or_(
                        SqlAgent.created_at > sub,
                        and_(SqlAgent.created_at == sub, SqlAgent.id > before),
                    )
                )
            stmt = stmt.order_by(SqlAgent.created_at.desc(), SqlAgent.id.desc()).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            entities = [_to_entity(r) for r in rows]
            return PagedList(
                data=entities,
                first_id=entities[0].id if entities else None,
                last_id=entities[-1].id if entities else None,
                has_more=has_more,
            )

    def delete(self, agent_id: str) -> bool:
        with self._session() as session:
            row = session.get(SqlAgent, agent_id)
            if not row:
                return False
            session.delete(row)
            return True
