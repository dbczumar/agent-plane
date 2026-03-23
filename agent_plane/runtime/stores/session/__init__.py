"""Session store — manages conversations and their items."""

from abc import ABC, abstractmethod

from agent_plane.runtime.models import (
    ConversationItem,
    NewConversationItem,
    PagedList,
    Session,
)


class SessionStore(ABC):
    @abstractmethod
    def create_session(self, metadata: dict | None = None) -> Session:
        """
        Create a new conversation session. Generates a unique session_id.
        Metadata is optional caller-attached key-value pairs (e.g. user_id,
        title). Returns the Session.
        """
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Session | None:
        """Return the session, or None if it does not exist."""
        ...

    @abstractmethod
    def get_session_id(self, response_id: str) -> str:
        """
        Resolve a response_id to the session it belongs to. Queries
        items by response_id (every item carries the response_id that
        produced it). This is the durable resolution path -- it works
        even after the task record has been cleaned up. Raises if no
        item with the given response_id exists.
        """
        ...

    @abstractmethod
    def get_latest_response_id(self, session_id: str) -> str | None:
        """
        Return the response_id of the most recent item in the session,
        or None if the session has no items. Used by the server to
        detect forks.
        """
        ...

    @abstractmethod
    def search_items(
        self,
        session_id: str,
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[ConversationItem]:
        """
        Return items in a session with cursor-based pagination,
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
        self, session_id: str, items: list[NewConversationItem]
    ) -> list[ConversationItem]:
        """
        Append items to a session. Assigns a globally unique ID and
        timestamp to each item. Returns the persisted ConversationItems
        with their assigned IDs.
        """
        ...

    @abstractmethod
    def list_sessions(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[Session]:
        """List sessions with cursor-based pagination, newest first."""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages. Returns True if the session
        existed, False otherwise. Async because it may need to cancel
        in-flight responses in the session first.
        """
        ...
