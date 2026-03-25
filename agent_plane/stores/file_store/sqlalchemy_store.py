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
    """
    Convert a :class:`SqlFile` ORM row to a :class:`StoredFile` entity.

    :param row: The SQLAlchemy ORM row to convert.
    :returns: A :class:`StoredFile` dataclass instance.
    """
    return StoredFile(
        id=row.id,
        created_at=row.created_at,
        filename=row.filename,
        bytes=row.bytes,
        content_type=row.content_type,
    )


class SqlAlchemyFileStore(FileStore):
    """
    SQLAlchemy-backed implementation of :class:`FileStore`.

    Persists file metadata in a relational database via
    SQLAlchemy ORM.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the SQLAlchemy file store.

        Creates or reuses a SQLAlchemy engine and session factory
        for the given database URI.

        :param storage_location: SQLAlchemy database URI,
            e.g. ``"sqlite:///files.db"`` or
            ``"postgresql://user:pass@host/db"``.
        """
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)

    def create(
        self,
        filename: str,
        bytes: int,
        content_type: str | None = None,
    ) -> StoredFile:
        """
        Record a new file in the database.

        :param filename: Original filename,
            e.g. ``"report.pdf"``.
        :param bytes: File size in bytes.
        :param content_type: MIME type, e.g.
            ``"application/pdf"``.
        :returns: The newly created :class:`StoredFile`.
        """
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
        """
        Fetch file metadata by its unique ID.

        :param file_id: Unique file identifier,
            e.g. ``"file_abc123"``.
        :returns: The :class:`StoredFile` if found, otherwise
            ``None``.
        """
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
        """
        List files with cursor-based pagination.

        :param limit: Maximum number of files to return.
        :param after: Cursor file ID; return files appearing
            after this file in sort order,
            e.g. ``"file_abc123"``.
        :param before: Cursor file ID; return files appearing
            before this file in sort order.
        :param order: Sort direction, ``"desc"`` or ``"asc"``.
        :returns: A :class:`PagedList` of :class:`StoredFile`
            objects.
        """
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
        """
        Delete file metadata by ID.

        :param file_id: Unique file identifier,
            e.g. ``"file_abc123"``.
        :returns: ``True`` if the file was deleted, ``False`` if
            it did not exist.
        """
        with self._session() as session:
            row = session.get(SqlFile, file_id)
            if not row:
                return False
            session.delete(row)
            return True
