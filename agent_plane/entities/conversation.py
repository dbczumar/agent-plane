"""Conversation entities — conversation, items, and item data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ── Conversation ──────────────────────────────────────


@dataclass
class Conversation:
    """A conversation grouping related turns."""

    id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    title: str | None = None


# ── Conversation item data types ───────────────────────


class MessageData(BaseModel):
    """Data for a message item (user or assistant)."""

    role: Literal["user", "assistant"]
    content: list[Any]
    agent: str | None = Field(
        default=None, serialization_alias="model"
    )

    @model_validator(mode="after")
    def check_agent_for_assistant(self) -> MessageData:
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
    summary: list[Any]
    content: list[Any] | None = None
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


def parse_item_data(item_type: str, raw: dict[str, Any]) -> ItemData:
    """
    Parse a raw dict into the appropriate ItemData model.
    Used by store implementations when deserializing from DB.
    """
    cls = ITEM_TYPE_TO_DATA_CLS.get(item_type)
    if cls is None:
        raise ValueError(f"unknown item type: {item_type!r}")
    return cls(**raw)  # type: ignore[return-value]


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
    def check_type_matches_data(self) -> NewConversationItem:
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
    def check_type_matches_data(self) -> ConversationItem:
        """Ensure `type` field is consistent with `data` model."""
        _validate_type_matches_data(self.type, self.data)
        return self
