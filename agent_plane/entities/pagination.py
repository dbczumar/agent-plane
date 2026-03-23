"""Cursor-based pagination container."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PagedList(Generic[T]):
    """A page of results matching the OpenAI list pagination shape."""

    data: list[T] = field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
