"""SQLAlchemy-backed conversation store."""

from __future__ import annotations

import json

from sqlalchemy import and_, asc, delete, desc, func, or_, select, text
from sqlalchemy.orm import Session

from agent_plane.db.db_models import SqlConversation, SqlConversationItem, SqlTask
from agent_plane.db.utils import (
    delete_fts_by_conversation,
    ensure_fts_table,
    extract_search_text,
    generate_conversation_id,
    generate_item_id,
    get_or_create_engine,
    insert_fts,
    make_managed_session_maker,
    now_epoch,
)
from agent_plane.entities import (
    Conversation,
    ConversationItem,
    NewConversationItem,
    PagedList,
    parse_item_data,
)
from agent_plane.stores.conversation_store import ConversationStore


def _to_conversation(row: SqlConversation) -> Conversation:
    """
    Convert a :class:`SqlConversation` ORM row to a
    :class:`Conversation` entity.

    :param row: The SQLAlchemy ORM row to convert.
    :returns: A :class:`Conversation` dataclass instance.
    """
    return Conversation(
        id=row.id,
        created_at=row.created_at,
        title=row.title,
        kind=row.kind,
    )


def _to_item(row: SqlConversationItem) -> ConversationItem:
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


class SqlAlchemyConversationStore(ConversationStore):
    """
    SQLAlchemy-backed implementation of :class:`ConversationStore`.

    Persists conversations and their items in a relational database
    via SQLAlchemy ORM. Also manages a full-text search (FTS) table
    for item content.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the SQLAlchemy conversation store.

        Creates or reuses a SQLAlchemy engine and session factory,
        and ensures the FTS virtual table exists.

        :param storage_location: SQLAlchemy database URI,
            e.g. ``"sqlite:///conversations.db"`` or
            ``"postgresql://user:pass@host/db"``.
        """
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)
        self._supports_for_update = self._engine.dialect.name != "sqlite"
        ensure_fts_table(self._engine)

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

    def create_conversation(self, kind: str = "default") -> Conversation:
        """
        Create a new conversation in the database.

        :param kind: Conversation type. ``"default"`` for
            user-initiated, ``"sub_agent"`` for sub-agent
            execution conversations.
        :returns: The newly created :class:`Conversation`.
        """
        row = SqlConversation(
            id=generate_conversation_id(),
            created_at=now_epoch(),
            kind=kind,
        )
        with self._session() as session:
            session.add(row)
            return _to_conversation(row)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Fetch a conversation by its unique ID.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: The :class:`Conversation` if found, otherwise
            ``None``.
        """
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            return _to_conversation(row) if row else None

    def get_conversation_id(self, response_id: str) -> str:
        """
        Resolve a response_id to the conversation it belongs to.

        :param response_id: The task/response ID to resolve,
            e.g. ``"resp_abc123"``.
        :returns: The conversation ID string.
        :raises LookupError: If no item with the given
            response_id exists.
        """
        with self._session() as session:
            conv_id = session.execute(
                select(SqlConversationItem.conversation_id)
                .where(SqlConversationItem.response_id == response_id)
                .limit(1)
            ).scalar_one_or_none()
            if conv_id is None:
                raise LookupError(f"no items found for response_id={response_id!r}")
            return conv_id

    def get_latest_response_id(self, conversation_id: str) -> str | None:
        """
        Return the response_id of the most recent item in the
        conversation.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: The response_id string, or ``None`` if the
            conversation has no items.
        """
        with self._session() as session:
            return session.execute(
                select(SqlConversationItem.response_id)
                .where(SqlConversationItem.conversation_id == conversation_id)
                .order_by(SqlConversationItem.position.desc())
                .limit(1)
            ).scalar_one_or_none()

    def search(
        self,
        query: str,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[ConversationItem]:
        """
        Full-text search over conversation items.

        Uses the FTS virtual table to match items by
        ``search_text``, ranked by relevance.

        :param query: The FTS search query string,
            e.g. ``"deployment error"``.
        :param conversation_id: Optional conversation to scope
            the search to, e.g. ``"conv_abc123"``.
        :param limit: Maximum number of results to return.
        :returns: A list of matching :class:`ConversationItem`
            objects in relevance order.
        """
        with self._session() as session:
            is_sqlite = self._engine.dialect.name == "sqlite"
            if is_sqlite:
                # SQLite FTS5: MATCH syntax with rank ordering.
                if conversation_id is not None:
                    stmt = text(
                        "SELECT item_id FROM conversation_items_fts "
                        "WHERE conversation_id = :cid "
                        "AND search_text MATCH :query "
                        "ORDER BY rank LIMIT :limit"
                    )
                else:
                    stmt = text(
                        "SELECT item_id FROM conversation_items_fts "
                        "WHERE search_text MATCH :query "
                        "ORDER BY rank LIMIT :limit"
                    )
            else:
                # PostgreSQL: ILIKE fallback (no FTS5 virtual table).
                # Full tsvector/tsquery indexing can be added later.
                like_pattern = f"%{query}%"
                if conversation_id is not None:
                    stmt = text(
                        "SELECT ci.id FROM conversation_items ci "
                        "WHERE ci.conversation_id = :cid "
                        "AND ci.data::text ILIKE :query "
                        "ORDER BY ci.created_at DESC LIMIT :limit"
                    )
                else:
                    stmt = text(
                        "SELECT ci.id FROM conversation_items ci "
                        "WHERE ci.data::text ILIKE :query "
                        "ORDER BY ci.created_at DESC LIMIT :limit"
                    )
                query = like_pattern
            params: dict[str, str | int] = {"query": query, "limit": limit}
            if conversation_id is not None:
                params["cid"] = conversation_id
            item_ids = [row[0] for row in session.execute(stmt, params).fetchall()]
            if not item_ids:
                return []
            rows = (
                session.execute(
                    select(SqlConversationItem).where(SqlConversationItem.id.in_(item_ids))
                )
                .scalars()
                .all()
            )
            # Preserve FTS rank order
            order = {iid: i for i, iid in enumerate(item_ids)}
            return [_to_item(r) for r in sorted(rows, key=lambda r: order[r.id])]

    def list_items(
        self,
        conversation_id: str,
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
        order: str = "asc",
        type: str | None = None,
    ) -> PagedList[ConversationItem]:
        """
        List items in a conversation with cursor-based pagination.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param limit: Maximum number of items to return.
        :param after: Cursor item ID; return items appearing
            after this item in sort order,
            e.g. ``"msg_xyz789"``.
        :param before: Cursor item ID; return items appearing
            before this item in sort order.
        :param order: Sort direction on position,
            ``"asc"`` or ``"desc"``.
        :param type: Optional item type filter. When provided, only items
            with this type are returned, e.g. ``"compaction"``. ``None``
            means return all types.
        :returns: A :class:`PagedList` of
            :class:`ConversationItem` objects.
        """
        with self._session() as session:
            is_asc = order == "asc"
            sort_fn = asc if is_asc else desc
            stmt = select(SqlConversationItem).where(
                SqlConversationItem.conversation_id == conversation_id
            )
            if type is not None:
                stmt = stmt.where(SqlConversationItem.type == type)
            if after:
                sub = (
                    select(SqlConversationItem.position)
                    .where(SqlConversationItem.id == after)
                    .scalar_subquery()
                )
                # "after" = further in sort direction
                stmt = stmt.where(
                    SqlConversationItem.position > sub
                    if is_asc
                    else SqlConversationItem.position < sub
                )
            if before:
                sub = (
                    select(SqlConversationItem.position)
                    .where(SqlConversationItem.id == before)
                    .scalar_subquery()
                )
                # "before" = opposite of sort direction
                stmt = stmt.where(
                    SqlConversationItem.position < sub
                    if is_asc
                    else SqlConversationItem.position > sub
                )
            stmt = stmt.order_by(sort_fn(SqlConversationItem.position)).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            items = [_to_item(r) for r in rows]
            return PagedList(
                data=items,
                first_id=items[0].id if items else None,
                last_id=items[-1].id if items else None,
                has_more=has_more,
            )

    def append(
        self,
        conversation_id: str,
        items: list[NewConversationItem],
    ) -> list[ConversationItem]:
        """
        Append items to a conversation.

        Assigns a globally unique ID, timestamp, and incrementing
        position to each item. Also inserts FTS records for
        searchability.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param items: List of :class:`NewConversationItem` objects
            to persist.
        :returns: The persisted :class:`ConversationItem` list
            with store-assigned IDs and timestamps.
        """
        now = now_epoch()
        persisted: list[ConversationItem] = []

        with self._session() as session:
            # Lock the conversation row to serialize position writes.
            # On PostgreSQL this is a row-level FOR UPDATE lock; on
            # SQLite the database-level lock already serializes.
            self._lock_conversation(session, conversation_id)

            # coalesce to -1 so the first appended item gets position 0.
            max_pos = session.execute(
                select(func.coalesce(func.max(SqlConversationItem.position), -1)).where(
                    SqlConversationItem.conversation_id == conversation_id
                )
            ).scalar_one()

            for item in items:
                max_pos += 1
                data_dict = item.data.model_dump(exclude_none=True)
                search = extract_search_text(item)
                item_id = generate_item_id(item.type)
                row = SqlConversationItem(
                    id=item_id,
                    conversation_id=conversation_id,
                    response_id=item.response_id,
                    created_at=now,
                    status="completed",  # items are final on append
                    position=max_pos,
                    type=item.type,
                    data=json.dumps(data_dict),
                    search_text=search,
                )
                session.add(row)
                insert_fts(session, item_id, conversation_id, search)
                persisted.append(
                    ConversationItem(
                        id=row.id,
                        type=row.type,
                        status=row.status,
                        response_id=row.response_id,
                        created_at=row.created_at,
                        data=item.data,
                    )
                )

        return persisted

    def list_conversations(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
        kind: str | None = "default",
    ) -> PagedList[Conversation]:
        """
        List conversations with cursor-based pagination.

        :param limit: Maximum number of conversations to return.
        :param after: Cursor conversation ID; return
            conversations appearing after this one in sort
            order, e.g. ``"conv_abc123"``.
        :param before: Cursor conversation ID; return
            conversations appearing before this one in sort
            order.
        :param order: Sort direction on ``created_at``,
            ``"desc"`` or ``"asc"``.
        :returns: A :class:`PagedList` of :class:`Conversation`
            objects.
        """
        with self._session() as session:
            is_desc = order == "desc"
            sort_fn = desc if is_desc else asc
            stmt = select(SqlConversation)
            # Filter by kind when specified (None = no filter).
            if kind is not None:
                stmt = stmt.where(SqlConversation.kind == kind)
            if after:
                sub = (
                    select(SqlConversation.created_at)
                    .where(SqlConversation.id == after)
                    .scalar_subquery()
                )
                # "after" = further in the sort direction:
                # desc → smaller created_at, asc → larger created_at
                ts_cmp = (
                    SqlConversation.created_at < sub
                    if is_desc
                    else SqlConversation.created_at > sub
                )
                id_cmp = SqlConversation.id < after if is_desc else SqlConversation.id > after
                stmt = stmt.where(or_(ts_cmp, and_(SqlConversation.created_at == sub, id_cmp)))
            if before:
                sub = (
                    select(SqlConversation.created_at)
                    .where(SqlConversation.id == before)
                    .scalar_subquery()
                )
                # "before" = opposite of sort direction
                ts_cmp = (
                    SqlConversation.created_at > sub
                    if is_desc
                    else SqlConversation.created_at < sub
                )
                id_cmp = SqlConversation.id > before if is_desc else SqlConversation.id < before
                stmt = stmt.where(or_(ts_cmp, and_(SqlConversation.created_at == sub, id_cmp)))
            stmt = stmt.order_by(
                sort_fn(SqlConversation.created_at),
                sort_fn(SqlConversation.id),
            ).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            convs = [_to_conversation(r) for r in rows]
            return PagedList(
                data=convs,
                first_id=convs[0].id if convs else None,
                last_id=convs[-1].id if convs else None,
                has_more=has_more,
            )

    def update_conversation(
        self, conversation_id: str, title: str | None = None
    ) -> Conversation | None:
        """
        Update mutable fields on a conversation.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param title: New title, or ``None`` to leave unchanged.
        :returns: The updated :class:`Conversation`, or ``None``
            if the conversation does not exist.
        """
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            if not row:
                return None
            if title is not None:
                row.title = title
            return _to_conversation(row)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation, its items, related tasks, and FTS
        records.

        Deletes in FK-safe order: tasks, FTS records, items,
        then the conversation itself.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: ``True`` if the conversation existed,
            ``False`` otherwise.
        """
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            if not row:
                return False
            # Order matters for FK constraints:
            # tasks and items reference the conversation.
            session.execute(delete(SqlTask).where(SqlTask.conversation_id == conversation_id))
            delete_fts_by_conversation(session, conversation_id)
            session.execute(
                delete(SqlConversationItem).where(
                    SqlConversationItem.conversation_id == conversation_id
                )
            )
            session.delete(row)
            return True
