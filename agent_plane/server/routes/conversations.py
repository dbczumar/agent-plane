"""Routes for the /v1/conversations endpoints."""

from fastapi import APIRouter, HTTPException, Query

from agent_plane.runtime.models import ConversationItem
from agent_plane.stores import ConversationStore, TaskStore
from agent_plane.server.models import (
    ConversationDeleted,
    ConversationObject,
    PaginatedList,
)


def _to_conversation_object(conv):
    """Convert a runtime Conversation to an API-layer ConversationObject."""
    return ConversationObject(
        id=conv.id,
        created_at=conv.created_at,
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
    conversation_store: ConversationStore,
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
        page = conversation_store.list_conversations(
            limit=limit,
            after=after,
            before=before,
        )
        data = [_to_conversation_object(s) for s in page.data]
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
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        return _to_conversation_object(conv)

    # ── GET /conversations/{conversation_id}/items ────────────────

    @router.get("/conversations/{conversation_id}/items")
    async def list_conversation_items(
        conversation_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="asc", pattern="^(asc|desc)$"),
    ):
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        page = conversation_store.search_items(
            conversation_id=conversation_id,
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
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        for task in task_store.list_tasks(
            conversation_id=conversation_id
        ):
            await task_store.cancel(task.task_id)
        deleted = await conversation_store.delete_conversation(
            conversation_id
        )
        if not deleted:
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            )
        return ConversationDeleted(id=conversation_id)

    return router
