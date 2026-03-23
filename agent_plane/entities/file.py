"""File entity."""

from dataclasses import dataclass


@dataclass
class StoredFile:
    """A stored file with metadata."""

    id: str
    created_at: int
    filename: str
    bytes: int
    content_location: str
    content_type: str | None = None
