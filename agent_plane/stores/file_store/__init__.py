"""File store — manages file metadata."""

from abc import ABC, abstractmethod

from agent_plane.entities import PagedList, StoredFile


class FileStore(ABC):
    """
    Abstract base for file metadata persistence.

    Tracks file metadata (filename, size, content type). Binary
    content is managed separately by :class:`ArtifactStore`.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the file store.

        :param storage_location: Backend-specific storage URI,
            e.g. ``"sqlite:///files.db"``.
        """
        self.storage_location = storage_location

    @abstractmethod
    def create(
        self,
        filename: str,
        bytes: int,
        content_type: str | None = None,
    ) -> StoredFile:
        """
        Record a new file. Generates a unique file_id. Binary
        content is managed separately by :class:`ArtifactStore`
        -- this only tracks metadata.

        :param filename: Original filename,
            e.g. ``"report.pdf"``.
        :param bytes: File size in bytes.
        :param content_type: MIME type of the file,
            e.g. ``"application/pdf"``.
        :returns: The newly created :class:`StoredFile`.
        """
        ...

    @abstractmethod
    def get(self, file_id: str) -> StoredFile | None:
        """
        Return the file metadata, or ``None`` if it does not
        exist.

        :param file_id: Unique file identifier,
            e.g. ``"file_abc123"``.
        :returns: The :class:`StoredFile` if found, otherwise
            ``None``.
        """
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

        :param limit: Maximum number of files to return.
        :param after: Cursor file ID; return files appearing
            after this file in sort order,
            e.g. ``"file_abc123"``.
        :param before: Cursor file ID; return files appearing
            before this file in sort order.
        :param order: Sort direction, ``"desc"`` or ``"asc"``.
        :returns: A :class:`PagedList` of :class:`StoredFile`
            objects.
        """
        ...

    @abstractmethod
    def delete(self, file_id: str) -> bool:
        """
        Delete file metadata. Returns ``True`` if the file existed,
        ``False`` otherwise. Caller is responsible for deleting the
        binary content from :class:`ArtifactStore`.

        :param file_id: Unique file identifier,
            e.g. ``"file_abc123"``.
        :returns: ``True`` if the file was deleted, ``False`` if
            it did not exist.
        """
        ...
