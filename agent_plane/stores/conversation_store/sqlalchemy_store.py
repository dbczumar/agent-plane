"""SQLAlchemy-backed conversation store."""

from __future__ import annotations

import json

from sqlalchemy import and_, asc, delete, desc, func, or_, select, text

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
    return Conversation(
        id=row.id,
        created_at=row.created_at,
        title=row.title,
    )


def _to_item(row: SqlConversationItem) -> ConversationItem:
    return ConversationItem(
        id=row.id,
        type=row.type,
        status=row.status,
        response_id=row.response_id,
        created_at=row.created_at,
        data=parse_item_data(row.type, json.loads(row.data)),
    )


class SqlAlchemyConversationStore(ConversationStore):
    def __init__(self, storage_location: str) -> None:
        super().__init__(storage_location)
        self._engine = get_or_create_engine(storage_location)
        self._session = make_managed_session_maker(self._engine)
        ensure_fts_table(self._engine)

    def create_conversation(self) -> Conversation:
        row = SqlConversation(
            id=generate_conversation_id(),
            created_at=now_epoch(),
        )
        with self._session() as session:
            session.add(row)
            return _to_conversation(row)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            return _to_conversation(row) if row else None

    def get_conversation_id(self, response_id: str) -> str:
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
        with self._session() as session:
            stmt = text(
                "SELECT item_id FROM conversation_items_fts "
                "WHERE search_text MATCH :query "
                "ORDER BY rank "
                "LIMIT :limit"
            )
            params: dict[str, str | int] = {"query": query, "limit": limit}
            if conversation_id is not None:
                stmt = text(
                    "SELECT item_id FROM conversation_items_fts "
                    "WHERE conversation_id = :cid AND search_text MATCH :query "
                    "ORDER BY rank "
                    "LIMIT :limit"
                )
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
    ) -> PagedList[ConversationItem]:
        with self._session() as session:
            sort_fn = asc if order == "asc" else desc
            # Always query in asc (position) order for consistent cursor
            # semantics, then reverse if the caller requested desc.
            stmt = select(SqlConversationItem).where(
                SqlConversationItem.conversation_id == conversation_id
            )
            if after:
                sub = (
                    select(SqlConversationItem.position)
                    .where(SqlConversationItem.id == after)
                    .scalar_subquery()
                )
                stmt = stmt.where(SqlConversationItem.position > sub)
            if before:
                sub = (
                    select(SqlConversationItem.position)
                    .where(SqlConversationItem.id == before)
                    .scalar_subquery()
                )
                stmt = stmt.where(SqlConversationItem.position < sub)
            stmt = stmt.order_by(SqlConversationItem.position.asc()).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            items = [_to_item(r) for r in rows]
            if sort_fn is desc:
                items.reverse()
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
        now = now_epoch()
        persisted: list[ConversationItem] = []

        with self._session() as session:
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
    ) -> PagedList[Conversation]:
        with self._session() as session:
            sort_fn = desc if order == "desc" else asc
            # Always query in desc order for consistent cursor semantics,
            # then reverse the results if the caller requested asc.
            stmt = select(SqlConversation)
            if after:
                sub = (
                    select(SqlConversation.created_at)
                    .where(SqlConversation.id == after)
                    .scalar_subquery()
                )
                stmt = stmt.where(
                    or_(
                        SqlConversation.created_at < sub,
                        and_(
                            SqlConversation.created_at == sub,
                            SqlConversation.id < after,
                        ),
                    )
                )
            if before:
                sub = (
                    select(SqlConversation.created_at)
                    .where(SqlConversation.id == before)
                    .scalar_subquery()
                )
                stmt = stmt.where(
                    or_(
                        SqlConversation.created_at > sub,
                        and_(
                            SqlConversation.created_at == sub,
                            SqlConversation.id > before,
                        ),
                    )
                )
            stmt = stmt.order_by(
                SqlConversation.created_at.desc(),
                SqlConversation.id.desc(),
            ).limit(limit + 1)
            rows = list(session.execute(stmt).scalars().all())
            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]
            convs = [_to_conversation(r) for r in rows]
            if sort_fn is asc:
                convs.reverse()
            return PagedList(
                data=convs,
                first_id=convs[0].id if convs else None,
                last_id=convs[-1].id if convs else None,
                has_more=has_more,
            )

    def update_conversation(
        self, conversation_id: str, title: str | None = None
    ) -> Conversation | None:
        with self._session() as session:
            row = session.get(SqlConversation, conversation_id)
            if not row:
                return None
            if title is not None:
                row.title = title
            return _to_conversation(row)

    async def delete_conversation(self, conversation_id: str) -> bool:
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
