"""File store — manages file metadata."""

from abc import ABC, abstractmethod

from agent_plane.entities import PagedList, StoredFile


class FileStore(ABC):
    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location

    @abstractmethod
    def create(
        self,
        filename: str,
        bytes: int,
        content_type: str | None = None,
    ) -> StoredFile:
        """
        Record a new file. Generates a unique file_id. Binary content
        is managed separately by ArtifactStore — this only tracks
        metadata.
        """
        ...

    @abstractmethod
    def get(self, file_id: str) -> StoredFile | None:
        """Return the file metadata, or None if it does not exist."""
        ...

    @abstractmethod
    def list(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ) -> PagedList[StoredFile]:
        """
        List files with cursor-based pagination.

        ``order`` controls the sort direction on ``created_at``
        (``"desc"`` = newest-first, ``"asc"`` = oldest-first).
        """
        ...

    @abstractmethod
    def delete(self, file_id: str) -> bool:
        """
        Delete file metadata. Returns True if the file existed,
        False otherwise. Caller is responsible for deleting the
        binary content from ArtifactStore.
        """
        ...
