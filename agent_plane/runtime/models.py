"""Data models for the runtime layer."""

from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")


# ── Task ───────────────────────────────────────────────


@dataclass
class Task:
    """A task representing a single response execution."""

    task_id: str
    session_id: str
    status: str  # "queued", "in_progress", "completed", "failed", "incomplete", "cancelled"
    agent_id: str
    agent_name: str
    created_at: int
    completed_at: int | None = None
    output: list = field(default_factory=list)
    inbox_closed: bool = False
    instructions: str | None = None
    metadata: dict = field(default_factory=dict)
    background: bool = False
    previous_response_id: str | None = None
    usage: dict | None = None
    error: dict | None = None
    incomplete_details: dict | None = None


# ── Conversation item data types ───────────────────────


class MessageData(BaseModel):
    """Data for a message item (user or assistant)."""

    role: Literal["user", "assistant"]
    content: list
    agent: str | None = Field(
        default=None, serialization_alias="model"
    )

    @model_validator(mode="after")
    def check_agent_for_assistant(self):
        """Assistant messages require agent; user messages must not."""
        if self.role == "assistant" and self.agent is None:
            raise ValueError(
                "assistant messages require 'agent'"
            )
        return self


class FunctionCallData(BaseModel):
    """Data for a function_call item."""

    agent: str = Field(serialization_alias="model")
    name: str
    arguments: str
    call_id: str


class FunctionCallOutputData(BaseModel):
    """Data for a function_call_output item."""

    call_id: str
    output: str


class ReasoningData(BaseModel):
    """Data for a reasoning item."""

    agent: str = Field(serialization_alias="model")
    summary: list
    content: list | None = None
    encrypted_content: str | None = None


ItemData = Union[
    MessageData,
    FunctionCallData,
    FunctionCallOutputData,
    ReasoningData,
]

ITEM_TYPE_TO_DATA_CLS: dict[str, type[BaseModel]] = {
    "message": MessageData,
    "function_call": FunctionCallData,
    "function_call_output": FunctionCallOutputData,
    "reasoning": ReasoningData,
}


def parse_item_data(item_type: str, raw: dict) -> ItemData:
    """
    Parse a raw dict into the appropriate ItemData model.
    Used by store implementations when deserializing from DB.
    """
    cls = ITEM_TYPE_TO_DATA_CLS.get(item_type)
    if cls is None:
        raise ValueError(f"unknown item type: {item_type!r}")
    return cls(**raw)


def _validate_type_matches_data(
    item_type: str, data: ItemData
) -> None:
    expected = ITEM_TYPE_TO_DATA_CLS.get(item_type)
    if expected is None:
        raise ValueError(f"unknown item type: {item_type!r}")
    if not isinstance(data, expected):
        raise ValueError(
            f"item type {item_type!r} requires "
            f"{expected.__name__}, got "
            f"{type(data).__name__}"
        )


# ── Conversation items ─────────────────────────────────


class NewConversationItem(BaseModel):
    """An item that has not yet been persisted. No ID or timestamp."""

    type: str
    response_id: str
    data: ItemData

    @model_validator(mode="after")
    def check_type_matches_data(self):
        """Ensure `type` field is consistent with `data` model."""
        _validate_type_matches_data(self.type, self.data)
        return self


class ConversationItem(BaseModel):
    """A persisted item with a store-assigned ID."""

    id: str
    type: str
    status: str
    response_id: str
    created_at: int
    data: ItemData

    @model_validator(mode="after")
    def check_type_matches_data(self):
        """Ensure `type` field is consistent with `data` model."""
        _validate_type_matches_data(self.type, self.data)
        return self


# ── Session ────────────────────────────────────────────


@dataclass
class Session:
    """A conversation session grouping related turns."""

    id: str
    metadata: dict = field(default_factory=dict)
    created_at: int = 0


# ── Pagination ─────────────────────────────────────────


@dataclass
class PagedList(Generic[T]):
    """A page of results with an optional cursor for the next page."""

    data: list[T] = field(default_factory=list)
    next_page_token: str | None = None
