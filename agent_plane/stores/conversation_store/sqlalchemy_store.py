"""SQLAlchemy-backed conversation store."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Select, and_, asc, delete, desc, func, or_, select, text
from sqlalchemy.orm import QueryableAttribute, Session

from agent_plane.db.db_models import (
    SqlConversation,
    SqlConversationItem,
    SqlConversationLabel,
    SqlTask,
)
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


def _to_conversation(
    row: SqlConversation,
    labels: dict[str, str] | None = None,
) -> Conversation:
    """
    Convert a :class:`SqlConversation` ORM row to a
    :class:`Conversation` entity.

    :param row: The SQLAlchemy ORM row to convert.
    :param labels: Pre-fetched guardrails labels for this
        conversation. ``None`` means "no label fetch was
        performed" (callers that don't need labels pass
        ``None`` rather than forcing a second query); this
        maps to an empty dict on the entity. Populated
        callers pass the JOINed ``{key: value}`` map.
    :returns: A :class:`Conversation` dataclass instance.
    """
    return Conversation(
        id=row.id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        title=row.title,
        kind=row.kind,
        parent_conversation_id=row.parent_conversation_id,
        labels=labels if labels is not None else {},
    )


def _upsert_labels(
    session: Session,
    conversation_id: str,
    updates: dict[str, str],
    updated_at: int,
) -> None:
    """
    Atomically UPSERT multiple labels on one conversation.

    Dialect-aware: SQLite and PostgreSQL both support
    ``INSERT ... ON CONFLICT ... DO UPDATE``, so we use
    their dedicated INSERT builders. Other dialects fall
    back to a SELECT-then-INSERT/UPDATE path, which is
    race-safe inside one transaction under SERIALIZABLE or
    (for SQLite) its default single-writer semantics.

    :param session: Active SQLAlchemy session (the atomic
        unit of work).
    :param conversation_id: Owning conversation ID.
    :param updates: Non-empty dict of label key → value.
    :param updated_at: Timestamp to write on every row
        touched by this call.
    """
    dialect = session.bind.dialect.name if session.bind is not None else ""
    rows = [
        {
            "conversation_id": conversation_id,
            "key": key,
            "value": value,
            "updated_at": updated_at,
        }
        for key, value in updates.items()
    ]
    if dialect in ("sqlite", "postgresql"):
        _dialect_upsert_labels(session, dialect, rows)
        return
    # Generic dialect fallback — SELECT-then-INSERT/UPDATE in
    # one transaction. Safe for the v1 "one active workflow
    # per conversation" invariant (POLICIES.md §10); the
    # SQLite / Postgres dialect-specific paths above give
    # true atomic UPSERT for the supported production dbs.
    for row in rows:
        existing = session.get(
            SqlConversationLabel,
            (row["conversation_id"], row["key"]),
        )
        if existing is None:
            session.add(SqlConversationLabel(**row))
        else:
            # mypy sees existing.{value,updated_at} as the
            # Mapped[...] descriptor types; at runtime these
            # are plain attributes that accept the target
            # Python type directly. SQLAlchemy's ORM handles
            # the coercion.
            existing.value = row["value"]  # type: ignore[assignment]
            existing.updated_at = row["updated_at"]  # type: ignore[assignment]


def _dialect_upsert_labels(
    session: Session,
    dialect: str,
    rows: list[dict[str, Any]],
) -> None:
    """
    Dialect-specific UPSERT path for SQLite / PostgreSQL.

    Extracted from ``_upsert_labels`` so the two branches
    (which use different ``insert`` builders producing
    incompatible type variances at the mypy level) each live
    in their own narrow scope. The outer function selects the
    branch; this one executes it.

    :param session: Active SQLAlchemy session.
    :param dialect: ``"sqlite"`` or ``"postgresql"`` (the
        outer function gates all other dialects onto the
        generic fallback path).
    :param rows: Pre-built row dicts to upsert.
    """
    # Typed as Any to sidestep the mypy variance issue between
    # the two dialect-specific ``Insert`` classes; the runtime
    # shape of both classes is identical for our use.
    stmt: Any
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        stmt = sqlite_insert(SqlConversationLabel).values(rows)
    else:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(SqlConversationLabel).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["conversation_id", "key"],
        set_={
            "value": stmt.excluded.value,
            "updated_at": stmt.excluded.updated_at,
        },
    )
    session.execute(stmt)


def _fetch_labels(
    session: Session,
    conversation_id: str,
) -> dict[str, str]:
    """
    Load all guardrails labels for a conversation.

    Returns an empty dict when no labels have been written
    yet — a conversation that was created before its spec
    declared guardrails, or before any policy wrote a label.

    :param session: The active SQLAlchemy session.
    :param conversation_id: Unique conversation identifier,
        e.g. ``"conv_abc123"``.
    :returns: Mapping from label key to value (string-typed).
        Empty dict when no rows match.
    """
    rows = session.execute(
        select(SqlConversationLabel.key, SqlConversationLabel.value).where(
            SqlConversationLabel.conversation_id == conversation_id,
        )
    ).all()
    return {key: value for key, value in rows}


def _fetch_labels_bulk(
    session: Session,
    conversation_ids: list[str],
) -> dict[str, dict[str, str]]:
    """
    Load labels for many conversations in a single query.

    Used by ``list_conversations`` to avoid an N+1 fan-out.
    Empty input returns an empty map without touching the
    database.

    :param session: The active SQLAlchemy session.
    :param conversation_ids: Conversation IDs to fetch labels
        for, e.g. ``["conv_a", "conv_b"]``. Duplicates are
        tolerated but yield the same map entries.
    :returns: Mapping ``{conversation_id: {key: value}}``.
        Conversations with no label rows are absent from the
        outer map (callers should default to ``{}``).
    """
    if not conversation_ids:
        return {}
    rows = session.execute(
        select(
            SqlConversationLabel.conversation_id,
            SqlConversationLabel.key,
            SqlConversationLabel.value,
        ).where(SqlConversationLabel.conversation_id.in_(conversation_ids))
    ).all()
    out: dict[str, dict[str, str]] = {}
    for conv_id, key, value in rows:
        out.setdefault(conv_id, {})[key] = value
    return out


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

    def create_conversation(
        self,
        kind: str = "default",
        title: str | None = None,
        parent_conversation_id: str | None = None,
    ) -> Conversation:
        """
        Create a new conversation in the database.

        :param kind: Conversation type. ``"default"`` for
            user-initiated, ``"sub_agent"`` for sub-agent
            execution conversations.
        :param title: Optional title. Phase 4 named sub-agents
            store ``"<type>:<name>"`` so the partial unique
            index enforces ``(parent_conversation_id, title)``
            uniqueness within a parent.
        :param parent_conversation_id: Phase 4 — id of the
            owning parent conversation. ``None`` for top-level.
        :returns: The newly created :class:`Conversation`.
        :raises NameAlreadyExistsError: If
            ``parent_conversation_id`` is set and a sibling row
            with the same ``title`` already exists.
        """
        from sqlalchemy.exc import IntegrityError

        from agent_plane.stores.conversation_store import NameAlreadyExistsError

        now = now_epoch()
        row = SqlConversation(
            id=generate_conversation_id(),
            created_at=now,
            updated_at=now,
            title=title,
            kind=kind,
            parent_conversation_id=parent_conversation_id,
        )
        try:
            with self._session() as session:
                session.add(row)
                # Convert inside the session so the entity is
                # populated before SQLAlchemy detaches it on
                # session close.
                return _to_conversation(row)
        except IntegrityError as exc:
            # Translate the partial-unique-index violation into a
            # clean exception type the spawn/send tools can map
            # to a name_already_exists tool error. Other integrity
            # violations (FK, check constraints) re-raise.
            #
            # Detection prefers the specific index name (Postgres
            # surfaces it directly), and falls back to the
            # ``parent_conversation_id`` + ``title`` column
            # signature (SQLite tends to format the message that
            # way). This is narrower than a generic "unique"
            # check, which would misclassify any future unique
            # constraint added to the conversations table.
            msg = str(exc).lower()
            is_partial_index_violation = "ix_conversations_parent_title_unique" in msg or (
                "unique" in msg and "parent_conversation_id" in msg and "title" in msg
            )
            if is_partial_index_violation:
                raise NameAlreadyExistsError(
                    f"sub-agent name already exists under parent "
                    f"{parent_conversation_id!r}: title={title!r}"
                ) from exc
            raise

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Fetch a conversation by its unique ID.

        Populates ``Conversation.labels`` via a second query
        against ``conversation_labels`` — separate from the
        conversation row fetch because the label JOIN would
        otherwise multiply the row count by the label count
        and require post-processing. The two queries run in
        the same session so they see a consistent snapshot
        under serializable isolation.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: The :class:`Conversation` if found, otherwise
            ``None``.
        """
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            if row is None:
                return None
            return _to_conversation(row, _fetch_labels(session, conversation_id))

    def set_labels(
        self,
        conversation_id: str,
        updates: dict[str, str],
        updated_at: int | None = None,
    ) -> None:
        """
        Upsert guardrails labels on a conversation.

        Single-transaction batched UPSERT — either every key
        lands or none do (POLICIES.md §6.3). The dialect-aware
        path dispatches to ``INSERT ... ON CONFLICT`` on
        SQLite / PostgreSQL; other dialects fall back to
        SELECT-then-INSERT/UPDATE inside the same transaction.
        Empty updates is a no-op.

        :param conversation_id: The conversation to update,
            e.g. ``"conv_abc123"``.
        :param updates: Mapping from label key to new value.
            Example: ``{"integrity": "0"}``. Empty dict
            returns immediately without opening a transaction.
        :param updated_at: Caller-supplied timestamp
            (``None`` → current wall-clock). See the abstract
            method docstring for why callers may want to
            pass their own.
        """
        if not updates:
            return
        stamp = updated_at if updated_at is not None else now_epoch()
        with self._session() as session:
            _upsert_labels(session, conversation_id, updates, stamp)

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
            # Dialect-specific search: SQLite has FTS5 virtual tables
            # (MATCH + rank), PostgreSQL doesn't. ILIKE on the JSON
            # data column is a functional fallback. Proper tsvector
            # indexing is a future optimization (tracked in GAPS.md).
            is_sqlite = self._engine.dialect.name == "sqlite"
            if is_sqlite:
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

            # Bump updated_at on the conversation.
            conv_row = session.get(SqlConversation, conversation_id)
            if conv_row is not None:
                conv_row.updated_at = now

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
        sort_by: str = "created_at",
        parent_conversation_id: str | None = None,
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
        :param order: Sort direction, ``"desc"`` or ``"asc"``.
        :param kind: Filter to conversations of this kind.
        :param sort_by: Column to sort on, ``"created_at"``
            or ``"updated_at"``.
        :param parent_conversation_id: Phase 4 — when set, only
            return conversations whose parent matches. ``None``
            disables the filter.
        :returns: A :class:`PagedList` of :class:`Conversation`
            objects.
        """
        sort_col = self._resolve_sort_column(sort_by)
        with self._session() as session:
            is_desc = order == "desc"
            sort_fn = desc if is_desc else asc
            stmt = select(SqlConversation)
            # Filter by kind when specified (None = no filter).
            if kind is not None:
                stmt = stmt.where(SqlConversation.kind == kind)
            if parent_conversation_id is not None:
                stmt = stmt.where(
                    SqlConversation.parent_conversation_id == parent_conversation_id,
                )
            if after:
                stmt = self._apply_cursor(stmt, after, sort_col, is_desc, forward=True)
            if before:
                stmt = self._apply_cursor(stmt, before, sort_col, is_desc, forward=False)
            stmt = stmt.order_by(
                sort_fn(sort_col),
                sort_fn(SqlConversation.id),
            ).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            # Fetch labels for all returned conversations in a
            # single IN-clause query so the list-path is O(1)
            # queries regardless of page size. Dropping this
            # would either silently return empty-labels
            # conversations (silent data loss) or fan out to
            # N+1 per-row queries.
            labels_by_conv = _fetch_labels_bulk(
                session,
                [r.id for r in rows],
            )
            convs = [_to_conversation(r, labels_by_conv.get(r.id, {})) for r in rows]
            return PagedList(
                data=convs,
                first_id=convs[0].id if convs else None,
                last_id=convs[-1].id if convs else None,
                has_more=has_more,
            )

    @staticmethod
    def _resolve_sort_column(sort_by: str) -> QueryableAttribute[int]:
        """
        Map a ``sort_by`` string to the corresponding
        :class:`SqlConversation` column.

        :param sort_by: ``"created_at"`` or ``"updated_at"``.
        :returns: The mapped column attribute.
        :raises ValueError: If ``sort_by`` is not a valid column
            name.
        """
        allowed = {
            "created_at": SqlConversation.created_at,
            "updated_at": SqlConversation.updated_at,
        }
        col = allowed.get(sort_by)
        if col is None:
            raise ValueError(f"invalid sort_by: {sort_by!r}")
        return col

    @staticmethod
    def _apply_cursor(
        stmt: Select[tuple[SqlConversation]],
        cursor_id: str,
        sort_col: QueryableAttribute[int],
        is_desc: bool,
        forward: bool,
    ) -> Select[tuple[SqlConversation]]:
        """
        Add a cursor-based WHERE clause to the query.

        Uses a (sort_col, id) composite comparison to handle
        ties deterministically.

        :param stmt: The current SELECT statement.
        :param cursor_id: The conversation ID acting as cursor.
        :param sort_col: The column being sorted on.
        :param is_desc: Whether the sort direction is descending.
        :param forward: ``True`` for ``after`` cursors (further
            in sort direction), ``False`` for ``before`` cursors
            (opposite of sort direction).
        :returns: The statement with cursor filter applied.
        """
        sub = select(sort_col).where(SqlConversation.id == cursor_id).scalar_subquery()
        # "after" (forward=True) = further in sort direction;
        # "before" (forward=False) = opposite of sort direction.
        if forward:
            ts_cmp = sort_col < sub if is_desc else sort_col > sub
            id_cmp = SqlConversation.id < cursor_id if is_desc else SqlConversation.id > cursor_id
        else:
            ts_cmp = sort_col > sub if is_desc else sort_col < sub
            id_cmp = SqlConversation.id > cursor_id if is_desc else SqlConversation.id < cursor_id
        return stmt.where(or_(ts_cmp, and_(sort_col == sub, id_cmp)))

    def update_conversation(
        self, conversation_id: str, title: str | None = None
    ) -> Conversation | None:
        """
        Update mutable fields on a conversation.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param title: New title, or ``None`` to leave unchanged.
            When provided, ``updated_at`` is bumped to the
            current epoch time.
        :returns: The updated :class:`Conversation`, or ``None``
            if the conversation does not exist.
        """
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            if not row:
                return None
            if title is not None:
                row.title = title
                row.updated_at = now_epoch()
            # Populate labels for parity with get_conversation —
            # callers must not see an empty dict masquerading as
            # "no labels exist" when labels do exist.
            return _to_conversation(row, _fetch_labels(session, conversation_id))

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
