"""Artifact store — blob storage for agent bundles and user files."""

from abc import ABC, abstractmethod


class ArtifactStore(ABC):
    """
    Blob storage for binary artifacts (agent bundles, user-uploaded
    files). Keyed by a unique string identifier. Metadata (filename,
    size, etc.) is managed separately by the route layer.
    """

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store a blob under the given key. Overwrites if exists."""
        ...

    @abstractmethod
    def get(self, key: str) -> bytes:
        """
        Retrieve a blob by key. Raises KeyError if not found.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Remove a blob. No-op if the key does not exist.
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a blob exists for the given key."""
        ...
