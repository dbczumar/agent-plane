"""Conversation store — manages conversations and their items."""

from abc import ABC, abstractmethod

from agent_plane.runtime.models import (
    Conversation,
    ConversationItem,
    NewConversationItem,
    PagedList,
)


class ConversationStore(ABC):
    @abstractmethod
    def create_conversation(
        self, metadata: dict | None = None
    ) -> Conversation:
        """
        Create a new conversation. Generates a unique conversation_id.
        Metadata is optional caller-attached key-value pairs (e.g. user_id,
        title). Returns the Conversation.
        """
        ...

    @abstractmethod
    def get_conversation(
        self, conversation_id: str
    ) -> Conversation | None:
        """Return the conversation, or None if it does not exist."""
        ...

    @abstractmethod
    def get_conversation_id(self, response_id: str) -> str:
        """
        Resolve a response_id to the conversation it belongs to. Queries
        items by response_id (every item carries the response_id that
        produced it). This is the durable resolution path -- it works
        even after the task record has been cleaned up. Raises if no
        item with the given response_id exists.
        """
        ...

    @abstractmethod
    def get_latest_response_id(
        self, conversation_id: str
    ) -> str | None:
        """
        Return the response_id of the most recent item in the
        conversation, or None if the conversation has no items.
        Used by the server to detect forks.
        """
        ...

    @abstractmethod
    def search_items(
        self,
        conversation_id: str,
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[ConversationItem]:
        """
        Return items in a conversation with cursor-based pagination,
        ordered chronologically. Used by the runtime to load
        conversation history and by the UI to display conversations.

        `after`: only return items after this item ID.
        `before`: only return items before this item ID.
        Both can be used together to select a window. Used by the
        agent loop (after=last_seen) to poll for steering items.
        """
        ...

    @abstractmethod
    def append(
        self,
        conversation_id: str,
        items: list[NewConversationItem],
    ) -> list[ConversationItem]:
        """
        Append items to a conversation. Assigns a globally unique ID
        and timestamp to each item. Returns the persisted
        ConversationItems with their assigned IDs.
        """
        ...

    @abstractmethod
    def list_conversations(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[Conversation]:
        """List conversations with cursor-based pagination, newest first."""
        ...

    @abstractmethod
    async def delete_conversation(
        self, conversation_id: str
    ) -> bool:
        """
        Delete a conversation and all its items. Returns True if the
        conversation existed, False otherwise. Async because it may
        need to cancel in-flight responses in the conversation first.
        """
        ...
