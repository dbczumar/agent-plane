"""File entity."""

from dataclasses import dataclass


@dataclass
class StoredFile:
    """
    A stored file with metadata.

    :param id: Unique file identifier, e.g. ``"file_abc123"``.
    :param created_at: Unix epoch timestamp of upload.
    :param filename: Original filename, e.g. ``"report.pdf"``.
    :param bytes: File size in bytes.
    :param content_type: MIME type, e.g. ``"application/pdf"``.
    """

    id: str
    created_at: int
    filename: str
    bytes: int
    content_type: str | None = None
