"""Agent store — manages registered agents."""

from abc import ABC, abstractmethod

from agent_plane.entities import Agent, PagedList


class AgentStore(ABC):
    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location

    @abstractmethod
    def create(
        self,
        name: str,
        bundle_location: str,
        description: str | None = None,
    ) -> Agent:
        """
        Register a new agent. Generates a unique agent_id. Name must
        be unique — raises if an agent with that name already exists.
        """
        ...

    @abstractmethod
    def get(self, agent_id: str) -> Agent | None:
        """Return the agent, or None if it does not exist."""
        ...

    @abstractmethod
    def get_by_name(self, name: str) -> Agent | None:
        """Look up an agent by its unique name. Returns None if not found."""
        ...

    @abstractmethod
    def list(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[Agent]:
        """List agents with cursor-based pagination."""
        ...

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """
        Delete an agent. Returns True if the agent existed, False
        otherwise. Caller is responsible for cancelling in-flight
        tasks before calling this.
        """
        ...
