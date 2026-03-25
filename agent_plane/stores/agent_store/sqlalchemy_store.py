"""SQLAlchemy-backed agent store."""

from __future__ import annotations

from sqlalchemy import and_, asc, desc, or_, select

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
    """
    Convert a :class:`SqlAgent` ORM row to an :class:`Agent` entity.

    :param row: The SQLAlchemy ORM row to convert.
    :returns: An :class:`Agent` dataclass instance.
    """
    return Agent(
        id=row.id,
        created_at=row.created_at,
        name=row.name,
        description=row.description,
    )


class SqlAlchemyAgentStore(AgentStore):
    """
    SQLAlchemy-backed implementation of :class:`AgentStore`.

    Persists agents in a relational database via SQLAlchemy ORM.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the SQLAlchemy agent store.

        Creates or reuses a SQLAlchemy engine and session factory
        for the given database URI.

        :param storage_location: SQLAlchemy database URI,
            e.g. ``"sqlite:///agents.db"`` or
            ``"postgresql://user:pass@host/db"``.
        """
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)

    def create(
        self,
        name: str,
        description: str | None = None,
    ) -> Agent:
        """
        Register a new agent in the database.

        :param name: Human-readable agent name. Must be unique,
            e.g. ``"code-assistant"``.
        :param description: Optional free-text description.
        :returns: The newly created :class:`Agent`.
        """
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
        """
        Fetch an agent by its unique ID.

        :param agent_id: Unique agent identifier,
            e.g. ``"agent_abc123"``.
        :returns: The :class:`Agent` if found, otherwise ``None``.
        """
        with self._session() as session:
            row = session.get(SqlAgent, agent_id)
            return _to_entity(row) if row else None

    def get_by_name(self, name: str) -> Agent | None:
        """
        Look up an agent by its unique name.

        :param name: The agent's unique name,
            e.g. ``"code-assistant"``.
        :returns: The :class:`Agent` if found, otherwise ``None``.
        """
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
        order: str = "desc",
    ) -> PagedList[Agent]:
        """
        List agents with cursor-based pagination.

        :param limit: Maximum number of agents to return.
        :param after: Cursor agent ID; return agents appearing
            after this agent in sort order,
            e.g. ``"agent_abc123"``.
        :param before: Cursor agent ID; return agents appearing
            before this agent in sort order.
        :param order: Sort direction, ``"desc"`` or ``"asc"``.
        :returns: A :class:`PagedList` of :class:`Agent` objects.
        """
        with self._session() as session:
            is_desc = order == "desc"
            sort_fn = desc if is_desc else asc
            stmt = select(SqlAgent)
            if after:
                sub = select(SqlAgent.created_at).where(SqlAgent.id == after).scalar_subquery()
                # "after" = further in sort direction
                ts_cmp = SqlAgent.created_at < sub if is_desc else SqlAgent.created_at > sub
                id_cmp = SqlAgent.id < after if is_desc else SqlAgent.id > after
                stmt = stmt.where(or_(ts_cmp, and_(SqlAgent.created_at == sub, id_cmp)))
            if before:
                sub = select(SqlAgent.created_at).where(SqlAgent.id == before).scalar_subquery()
                # "before" = opposite of sort direction
                ts_cmp = SqlAgent.created_at > sub if is_desc else SqlAgent.created_at < sub
                id_cmp = SqlAgent.id > before if is_desc else SqlAgent.id < before
                stmt = stmt.where(or_(ts_cmp, and_(SqlAgent.created_at == sub, id_cmp)))
            stmt = stmt.order_by(sort_fn(SqlAgent.created_at), sort_fn(SqlAgent.id)).limit(
                limit + 1
            )
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
        """
        Delete an agent by ID.

        :param agent_id: Unique agent identifier,
            e.g. ``"agent_abc123"``.
        :returns: ``True`` if the agent was deleted, ``False`` if
            it did not exist.
        """
        with self._session() as session:
            row = session.get(SqlAgent, agent_id)
            if not row:
                return False
            session.delete(row)
            return True
