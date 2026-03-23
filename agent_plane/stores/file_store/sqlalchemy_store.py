"""SQLAlchemy-backed file store."""

from __future__ import annotations

from sqlalchemy import and_, asc, desc, or_, select

from agent_plane.db.db_models import SqlFile
from agent_plane.db.utils import (
    generate_file_id,
    get_or_create_engine,
    make_managed_session_maker,
    now_epoch,
)
from agent_plane.entities import PagedList, StoredFile
from agent_plane.stores.file_store import FileStore


def _to_entity(row: SqlFile) -> StoredFile:
    return StoredFile(
        id=row.id,
        created_at=row.created_at,
        filename=row.filename,
        bytes=row.bytes,
        content_type=row.content_type,
    )


class SqlAlchemyFileStore(FileStore):
    def __init__(self, storage_location: str) -> None:
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)

    def create(
        self,
        filename: str,
        bytes: int,
        content_type: str | None = None,
    ) -> StoredFile:
        row = SqlFile(
            id=generate_file_id(),
            created_at=now_epoch(),
            filename=filename,
            bytes=bytes,
            content_type=content_type,
        )
        with self._session() as session:
            session.add(row)
            return _to_entity(row)

    def get(self, file_id: str) -> StoredFile | None:
        with self._session() as session:
            row = session.get(SqlFile, file_id)
            return _to_entity(row) if row else None

    def list(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ) -> PagedList[StoredFile]:
        with self._session() as session:
            is_desc = order == "desc"
            sort_fn = desc if is_desc else asc
            stmt = select(SqlFile)
            if after:
                sub = select(SqlFile.created_at).where(SqlFile.id == after).scalar_subquery()
                # "after" = further in sort direction
                ts_cmp = SqlFile.created_at < sub if is_desc else SqlFile.created_at > sub
                id_cmp = SqlFile.id < after if is_desc else SqlFile.id > after
                stmt = stmt.where(or_(ts_cmp, and_(SqlFile.created_at == sub, id_cmp)))
            if before:
                sub = select(SqlFile.created_at).where(SqlFile.id == before).scalar_subquery()
                # "before" = opposite of sort direction
                ts_cmp = SqlFile.created_at > sub if is_desc else SqlFile.created_at < sub
                id_cmp = SqlFile.id > before if is_desc else SqlFile.id < before
                stmt = stmt.where(or_(ts_cmp, and_(SqlFile.created_at == sub, id_cmp)))
            stmt = stmt.order_by(sort_fn(SqlFile.created_at), sort_fn(SqlFile.id)).limit(limit + 1)
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

    def delete(self, file_id: str) -> bool:
        with self._session() as session:
            row = session.get(SqlFile, file_id)
            if not row:
                return False
            session.delete(row)
            return True
