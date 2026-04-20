"""Routes for the /v1/conversations endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from agent_plane.entities import Conversation, ConversationItem
from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.server.schemas import (
    ConversationDeleted,
    ConversationObject,
    ItemDeleted,
    PaginatedList,
)
from agent_plane.stores import ConversationStore, TaskStore


class CreateConversationRequest(BaseModel):
    """
    Request body for ``POST /v1/conversations``.

    :param metadata: Optional key-value map of up to 16 pairs
        (keys ≤64 chars, values ≤512 chars).
    """

    metadata: dict[str, str] | None = None


class UpdateConversationRequest(BaseModel):
    """
    Request body for ``PATCH /v1/conversations/{conversation_id}``
    and ``POST /v1/conversations/{conversation_id}``.

    :param title: New title for the conversation, or ``None`` to
        leave unchanged.
    :param metadata: New metadata map to replace the existing one,
        or ``None`` to leave unchanged.
    """

    title: str | None = None
    metadata: dict[str, str] | None = None


def _to_conversation_object(conv: Conversation) -> ConversationObject:
    """
    Convert a runtime Conversation to an API-layer
    ConversationObject.

    :param conv: The runtime conversation entity.
    :returns: A :class:`ConversationObject` for the API response.
    """
    return ConversationObject(
        id=conv.id,
        title=conv.title,
        metadata=conv.metadata,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


def _to_api_item(item: ConversationItem) -> dict[str, Any]:
    """
    Convert a runtime ConversationItem to the API shape defined
    in API.md.

    Common fields come from the item; type-specific fields
    (role, content, model, name, arguments, etc.) come from
    ``item.data``. ``exclude_none`` ensures absent optional
    fields (e.g. ``model`` on user messages) don't appear in
    the output.

    Returns ``dict[str, Any]`` because value types vary across
    item types (str, int, list, etc.) due to the
    ``model_dump`` spread.

    :param item: The persisted conversation item to convert.
    :returns: A flat dict with all item fields merged.
    """
    return {
        "id": item.id,
        "response_id": item.response_id,
        "type": item.type,
        "status": item.status,
        **item.data.model_dump(exclude_none=True, by_alias=True),
    }


def create_conversations_router(
    conversation_store: ConversationStore,
    task_store: TaskStore,
) -> APIRouter:
    """
    Factory that builds the conversations router.

    Stores are closed over -- no dependency injection.

    :param conversation_store: Store for conversation and item
        persistence.
    :param task_store: Store for task deletion when a
        conversation is deleted.
    :returns: A configured :class:`APIRouter` with all
        ``/conversations`` endpoints.
    """
    router = APIRouter()

    # ── POST /conversations ───────────────────────────────────────

    @router.post("/conversations", status_code=201)
    async def create_conversation(
        body: CreateConversationRequest,
    ) -> ConversationObject:
        """
        Explicitly create a new conversation.

        The conversation is empty on creation; items are added
        by subsequent ``POST /v1/responses`` calls or via
        ``POST /v1/conversations/{id}/items``.

        :param body: Optional metadata to attach.
        :returns: The newly created :class:`ConversationObject`.
        """
        conv = conversation_store.create_conversation(metadata=body.metadata)
        return _to_conversation_object(conv)

    # ── GET /conversations ────────────────────────────────────────

    @router.get("/conversations")
    async def list_conversations(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
        sort_by: str = Query(default="created_at", pattern="^(created_at|updated_at)$"),
    ) -> PaginatedList:
        """
        List conversations with cursor-based pagination.

        :param limit: Maximum number of items to return (1-100).
        :param after: Cursor — return items after this
            conversation ID, e.g. ``"conv_abc123"``.
        :param before: Cursor — return items before this
            conversation ID.
        :param order: Sort order, ``"asc"`` or ``"desc"``.
        :param sort_by: Column to sort on,
            ``"created_at"`` or ``"updated_at"``.
        :returns: A :class:`PaginatedList` of conversations.
        """
        page = conversation_store.list_conversations(
            limit=limit,
            after=after,
            before=before,
            order=order,
            sort_by=sort_by,
        )
        data = [_to_conversation_object(s) for s in page.data]
        return PaginatedList(
            data=data,
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )

    # ── GET /conversations/{conversation_id} ──────────────────────

    @router.get("/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str,
    ) -> ConversationObject:
        """
        Retrieve a single conversation by ID.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: The matching :class:`ConversationObject`.
        :raises AgentPlaneError: If the conversation is not
            found.
        """
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        return _to_conversation_object(conv)

    # ── GET /conversations/{conversation_id}/items ────────────────

    @router.get("/conversations/{conversation_id}/items")
    async def list_conversation_items(
        conversation_id: str,
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="asc", pattern="^(asc|desc)$"),
    ) -> PaginatedList:
        """
        List items in a conversation with cursor-based
        pagination.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param limit: Maximum number of items to return
            (1-100).
        :param after: Cursor — return items after this item
            ID, e.g. ``"msg_abc123"``.
        :param before: Cursor — return items before this
            item ID.
        :param order: Sort order, ``"asc"`` or ``"desc"``.
        :returns: A :class:`PaginatedList` of conversation
            items.
        :raises AgentPlaneError: If the conversation is not
            found.
        """
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        page = conversation_store.list_items(
            conversation_id=conversation_id,
            limit=limit,
            after=after,
            before=before,
            order=order,
        )
        data = [_to_api_item(m) for m in page.data]
        return PaginatedList(
            data=data,
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )

    # ── GET /conversations/{conversation_id}/items/{item_id} ─────

    @router.get("/conversations/{conversation_id}/items/{item_id}")
    async def get_conversation_item(
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve a single item from a conversation.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param item_id: The item identifier,
            e.g. ``"msg_abc123"``.
        :returns: The conversation item as a flat dict.
        :raises AgentPlaneError: If the conversation or item is
            not found.
        """
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        item = conversation_store.get_item(conversation_id, item_id)
        if item is None:
            raise AgentPlaneError("Item not found", code=ErrorCode.NOT_FOUND)
        return _to_api_item(item)

    # ── DELETE /conversations/{conversation_id}/items/{item_id} ──

    @router.delete("/conversations/{conversation_id}/items/{item_id}")
    async def delete_conversation_item(
        conversation_id: str,
        item_id: str,
    ) -> ItemDeleted:
        """
        Delete a single item from a conversation.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param item_id: The item identifier to delete,
            e.g. ``"msg_abc123"``.
        :returns: An :class:`ItemDeleted` confirmation.
        :raises AgentPlaneError: If the conversation or item is
            not found.
        """
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        deleted = conversation_store.delete_item(conversation_id, item_id)
        if not deleted:
            raise AgentPlaneError("Item not found", code=ErrorCode.NOT_FOUND)
        return ItemDeleted(id=item_id, conversation_id=conversation_id)

    # ── PATCH /conversations/{conversation_id} ────────────────────
    # ── POST  /conversations/{conversation_id} (OpenAI compat) ───

    async def _do_update_conversation(
        conversation_id: str, body: UpdateConversationRequest
    ) -> ConversationObject:
        """
        Shared update logic for PATCH and POST update handlers.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param body: Request body with fields to update.
        :returns: The updated :class:`ConversationObject`.
        :raises AgentPlaneError: If the conversation is not found.
        """
        conv = conversation_store.update_conversation(
            conversation_id, title=body.title, metadata=body.metadata
        )
        if conv is None:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        return _to_conversation_object(conv)

    @router.patch("/conversations/{conversation_id}")
    async def patch_conversation(
        conversation_id: str, body: UpdateConversationRequest
    ) -> ConversationObject:
        """
        Update a conversation's mutable fields (REST-style PATCH).

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param body: Request body with fields to update.
        :returns: The updated :class:`ConversationObject`.
        :raises AgentPlaneError: If the conversation is not found.
        """
        return await _do_update_conversation(conversation_id, body)

    @router.post("/conversations/{conversation_id}")
    async def post_update_conversation(
        conversation_id: str, body: UpdateConversationRequest
    ) -> ConversationObject:
        """
        Update a conversation's mutable fields (OpenAI-compat POST).

        The OpenAI Conversations API uses ``POST /{id}`` rather
        than ``PATCH /{id}`` for updates. This endpoint is an
        alias so clients targeting the OpenAI spec work without
        modification.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param body: Request body with fields to update.
        :returns: The updated :class:`ConversationObject`.
        :raises AgentPlaneError: If the conversation is not found.
        """
        return await _do_update_conversation(conversation_id, body)

    # ── DELETE /conversations/{conversation_id} ───────────────────

    @router.delete("/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str,
    ) -> ConversationDeleted:
        """
        Delete a conversation and all associated tasks.

        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: A :class:`ConversationDeleted` confirmation.
        :raises AgentPlaneError: If the conversation is not
            found.
        """
        conv = conversation_store.get_conversation(conversation_id)
        if conv is None:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        await task_store.delete_all(conversation_id=conversation_id)
        deleted = await conversation_store.delete_conversation(conversation_id)
        if not deleted:
            raise AgentPlaneError("Conversation not found", code=ErrorCode.NOT_FOUND)
        return ConversationDeleted(id=conversation_id)

    return router
