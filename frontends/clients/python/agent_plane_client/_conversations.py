"""Conversations namespace — list, get, update, delete."""

from __future__ import annotations

import httpx

from ._errors import raise_for_status
from ._types import Conversation, PaginatedList


class ConversationsNamespace:
    """Methods for ``/v1/conversations`` endpoints."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        order: str = "desc",
    ) -> list[Conversation]:
        """List conversations.

        :param limit: Max conversations to return.
        :param after: Cursor for pagination.
        :param order: Sort order.
        :returns: List of conversations.
        """
        params: dict[str, object] = {"limit": limit, "order": order}
        if after is not None:
            params["after"] = after
        resp = await self._http.get(f"{self._base}/v1/conversations", params=params)
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        page = PaginatedList.from_dict(resp.json())
        return [Conversation.from_dict(d) for d in page.data]

    async def get(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID.

        :param conversation_id: The conversation ID.
        :returns: The conversation.
        """
        resp = await self._http.get(f"{self._base}/v1/conversations/{conversation_id}")
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Conversation.from_dict(resp.json())

    async def list_items(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        after: str | None = None,
        order: str = "asc",
    ) -> list[dict[str, object]]:
        """List conversation items (messages, tool calls, etc.).

        :param conversation_id: The conversation ID.
        :param limit: Max items to return.
        :param after: Cursor for pagination.
        :param order: Sort order (``"asc"`` = chronological).
        :returns: List of conversation item dicts.
        """
        params: dict[str, object] = {"limit": limit, "order": order}
        if after is not None:
            params["after"] = after
        resp = await self._http.get(
            f"{self._base}/v1/conversations/{conversation_id}/items",
            params=params,
        )
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        page = PaginatedList.from_dict(resp.json())
        return page.data

    async def update(self, conversation_id: str, *, title: str | None) -> Conversation:
        """Update a conversation (currently only title).

        :param conversation_id: The conversation ID.
        :param title: New title, or None to clear.
        :returns: The updated conversation.
        """
        resp = await self._http.patch(
            f"{self._base}/v1/conversations/{conversation_id}",
            json={"title": title},
        )
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Conversation.from_dict(resp.json())

    async def delete(self, conversation_id: str) -> None:
        """Delete a conversation and all its responses.

        :param conversation_id: The conversation ID.
        """
        resp = await self._http.delete(f"{self._base}/v1/conversations/{conversation_id}")
        if resp.status_code >= 400:
            data = resp.json() if resp.status_code < 500 else resp.text
            raise_for_status(resp.status_code, data)
