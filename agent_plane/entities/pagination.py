"""Cursor-based pagination container."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PagedList(Generic[T]):
    """A page of results with an optional cursor for the next page."""

    data: list[T] = field(default_factory=list)
    next_page_token: str | None = None
