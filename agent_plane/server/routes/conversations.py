"""Routes for the /v1/conversations endpoints."""

from fastapi import APIRouter, HTTPException, Query

from agent_plane.runtime.models import ConversationItem, Session
from agent_plane.stores import SessionStore, TaskStore
from agent_plane.server.models import (
    ConversationDeleted,
    ConversationObject,
    PaginatedList,
)


def _session_to_conversation(session: Session) -> ConversationObject:
    """Convert a runtime Session to an API-layer ConversationObject."""
    return ConversationObject(
        id=session.id,
        created_at=session.created_at,
    )


def _to_api_item(item: ConversationItem) -> dict:
    """
    Convert a runtime ConversationItem to the API shape defined
    in API.md. Common fields come from the item; type-specific
    fields (role, content, model, name, arguments, etc.) come
    from item.data. `exclude_none` ensures absent optional fields
    (e.g. `model` on user messages) don't appear in the output.
    """
    return {
        "id": item.id,
        "response_id": item.response_id,
        "type": item.type,
        "status": "completed",
        **item.data.model_dump(exclude_none=True, by_alias=True),
    }


def create_conversations_router(
    session_store: SessionStore,
    task_store: TaskStore,
) -> APIRouter:
    """
    Factory that builds the conversations router. Stores are closed
    over -- no dependency injection.
    """
    router = APIRouter()

    # ── GET /conversations ────────────────────────────────────────

    @router.get("/conversations")
    async def list_conversations(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ):
        page = session_store.list_sessions(
            limit=limit,
            after=after,
            before=before,
        )
        data = [_session_to_conversation(s) for s in page.data]
        if order == "asc":
            data = list(reversed(data))
        return PaginatedList(
            data=data,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
            has_more=page.next_page_token is not None,
        )

    # ── GET /conversations/{conversation_id} ──────────────────────

    @router.get("/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        session = session_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        return _session_to_conversation(session)

    # ── GET /conversations/{conversation_id}/items ────────────────

    @router.get("/conversations/{conversation_id}/items")
    async def list_conversation_items(
        conversation_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="asc", pattern="^(asc|desc)$"),
    ):
        session = session_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        page = session_store.search_items(
            session_id=conversation_id,
            limit=limit,
            after=after,
            before=before,
        )
        data = [_to_api_item(m) for m in page.data]
        if order == "desc":
            data = list(reversed(data))
        return PaginatedList(
            data=data,
            first_id=data[0]["id"] if data else None,
            last_id=data[-1]["id"] if data else None,
            has_more=page.next_page_token is not None,
        )

    # ── DELETE /conversations/{conversation_id} ───────────────────

    @router.delete("/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        session = session_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        for task in task_store.list_tasks(session_id=conversation_id):
            await task_store.cancel(task.task_id)
        deleted = await session_store.delete_session(conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        return ConversationDeleted(id=conversation_id)

    return router
