"""Cursor-based pagination container."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PagedList(Generic[T]):
    """
    A page of results matching the OpenAI list pagination shape.

    :param data: The items in this page.
    :param first_id: ID of the first item in the page, or ``None``
        if empty.
    :param last_id: ID of the last item in the page, or ``None``
        if empty.
    :param has_more: ``True`` if more pages exist after this one.
    """

    data: list[T] = field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
