"""Artifact store — blob storage for agent bundles and user files."""

from abc import ABC, abstractmethod


class ArtifactStore(ABC):
    """
    Blob storage for binary artifacts (agent bundles, user-uploaded
    files). Keyed by a unique string identifier. Metadata (filename,
    size, etc.) is managed separately by the route layer.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the store with a backend-specific storage location.

        The interpretation of *storage_location* depends on the
        concrete implementation — e.g. a filesystem path for local
        storage, an S3 URI for cloud storage, etc.
        """
        self._storage_location = storage_location

    @property
    def storage_location(self) -> str:
        """The backend-specific storage location (path, URI, bucket, etc.)."""
        return self._storage_location

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
