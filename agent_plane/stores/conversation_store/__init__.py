"""Conversation store — manages conversations and their items."""

from abc import ABC, abstractmethod

from agent_plane.entities import (
    Conversation,
    ConversationItem,
    NewConversationItem,
    PagedList,
)


class ConversationStore(ABC):
    """
    Abstract base for conversation persistence.

    Manages conversations and their items: creation, lookup,
    paginated listing, appending items, full-text search,
    updates, and deletion.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the conversation store.

        :param storage_location: Backend-specific storage URI,
            e.g. ``"sqlite:///conversations.db"``.
        """
        self.storage_location = storage_location

    @abstractmethod
    def create_conversation(
        self,
        kind: str = "default",
    ) -> Conversation:
        """
        Create a new conversation. Generates a unique
        conversation_id.

        :param kind: Conversation type. ``"default"`` for
            user-initiated, ``"sub_agent"`` for sub-agent
            execution conversations.
        :returns: The newly created :class:`Conversation`.
        """
        ...

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Return the conversation, or ``None`` if it does not exist.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: The :class:`Conversation` if found, otherwise
            ``None``.
        """
        ...

    @abstractmethod
    def get_conversation_id(self, response_id: str) -> str:
        """
        Resolve a response_id to the conversation it belongs to.

        Queries items by response_id (every item carries the
        response_id that produced it). This is the durable
        resolution path -- it works even after the task record
        has been cleaned up.

        :param response_id: The task/response ID to resolve,
            e.g. ``"resp_abc123"``.
        :returns: The conversation ID containing items with
            the given response_id.
        :raises LookupError: If no item with the given
            response_id exists.
        """
        ...

    @abstractmethod
    def get_latest_response_id(self, conversation_id: str) -> str | None:
        """
        Return the response_id of the most recent item in the
        conversation, or ``None`` if the conversation has no items.
        Used by the server to detect forks.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: The response_id string, or ``None``.
        """
        ...

    @abstractmethod
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
        Return items in a conversation with cursor-based pagination.

        ``order`` controls the sort direction on ``position``
        (``"asc"`` = chronological, ``"desc"`` = reverse).

        Both ``after`` and ``before`` can be used together to
        select a window. Used by the agent loop
        (``after=last_seen``) to poll for steering items.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param limit: Maximum number of items to return.
        :param after: Cursor item ID; only return items after
            this item in sort order, e.g. ``"msg_xyz789"``.
        :param before: Cursor item ID; only return items before
            this item in sort order.
        :param order: Sort direction, ``"asc"`` or ``"desc"``.
        :param type: Optional item type filter. When provided, only items
            with this type are returned, e.g. ``"compaction"``. ``None``
            means return all types.
        :returns: A :class:`PagedList` of
            :class:`ConversationItem` objects.
        """
        ...

    @abstractmethod
    def append(
        self,
        conversation_id: str,
        items: list[NewConversationItem],
    ) -> list[ConversationItem]:
        """
        Append items to a conversation. Assigns a globally unique
        ID and timestamp to each item.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param items: List of :class:`NewConversationItem` objects
            to persist.
        :returns: The persisted :class:`ConversationItem` list
            with store-assigned IDs and timestamps.
        """
        ...

    @abstractmethod
    def list_conversations(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
        kind: str | None = "default",
        sort_by: str = "created_at",
    ) -> PagedList[Conversation]:
        """
        List conversations with cursor-based pagination.

        ``order`` controls the sort direction on the column
        selected by ``sort_by`` (``"desc"`` = newest-first,
        ``"asc"`` = oldest-first).

        :param limit: Maximum number of conversations to return.
        :param after: Cursor conversation ID; return conversations
            appearing after this one in sort order,
            e.g. ``"conv_abc123"``.
        :param before: Cursor conversation ID; return conversations
            appearing before this one in sort order.
        :param order: Sort direction, ``"desc"`` or ``"asc"``.
        :param kind: Filter to conversations of this kind. Exact
            match. ``"default"`` returns only user-initiated.
            ``"sub_agent"`` returns only sub-agent conversations.
            ``None`` disables the filter and returns all.
        :param sort_by: Column to sort on, ``"created_at"`` or
            ``"updated_at"``.
        :returns: A :class:`PagedList` of :class:`Conversation`
            objects.
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[ConversationItem]:
        """
        Full-text search over conversation items.

        Returns items whose search_text matches the query,
        optionally scoped to a single conversation. Results are
        ranked by relevance.

        :param query: The search query string,
            e.g. ``"deployment error"``.
        :param conversation_id: Optional conversation to scope
            the search to, e.g. ``"conv_abc123"``.
        :param limit: Maximum number of results to return.
        :returns: A list of matching :class:`ConversationItem`
            objects ranked by relevance.
        """
        ...

    @abstractmethod
    def update_conversation(
        self, conversation_id: str, title: str | None = None
    ) -> Conversation | None:
        """
        Update mutable fields on a conversation.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :param title: New title for the conversation, or ``None``
            to leave unchanged.
        :returns: The updated :class:`Conversation`, or ``None``
            if the conversation does not exist.
        """
        ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its items.

        Async because it may need to cancel in-flight responses
        in the conversation first.

        :param conversation_id: Unique conversation identifier,
            e.g. ``"conv_abc123"``.
        :returns: ``True`` if the conversation existed,
            ``False`` otherwise.
        """
        ...
