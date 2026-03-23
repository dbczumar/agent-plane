"""SQLAlchemy-backed file store."""

from __future__ import annotations

from sqlalchemy import and_, or_, select

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
        content_location=row.content_location,
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
        content_location: str,
        content_type: str | None = None,
    ) -> StoredFile:
        row = SqlFile(
            id=generate_file_id(),
            created_at=now_epoch(),
            filename=filename,
            bytes=bytes,
            content_location=content_location,
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
    ) -> PagedList[StoredFile]:
        with self._session() as session:
            stmt = select(SqlFile)
            if after:
                sub = (
                    select(SqlFile.created_at)
                    .where(SqlFile.id == after)
                    .scalar_subquery()
                )
                stmt = stmt.where(
                    or_(
                        SqlFile.created_at < sub,
                        and_(SqlFile.created_at == sub, SqlFile.id < after),
                    )
                )
            if before:
                sub = (
                    select(SqlFile.created_at)
                    .where(SqlFile.id == before)
                    .scalar_subquery()
                )
                stmt = stmt.where(
                    or_(
                        SqlFile.created_at > sub,
                        and_(SqlFile.created_at == sub, SqlFile.id > before),
                    )
                )
            stmt = stmt.order_by(
                SqlFile.created_at.desc(), SqlFile.id.desc()
            ).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            return PagedList(
                data=[_to_entity(r) for r in rows],
                next_page_token=rows[-1].id if has_more else None,
            )

    def delete(self, file_id: str) -> bool:
        with self._session() as session:
            row = session.get(SqlFile, file_id)
            if not row:
                return False
            session.delete(row)
            return True
