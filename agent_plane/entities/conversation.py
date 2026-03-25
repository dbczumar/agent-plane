"""Conversation entities — conversation, items, and item data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ── Conversation ──────────────────────────────────────


@dataclass
class Conversation:
    """
    A conversation grouping related turns.

    :param id: Unique conversation identifier,
        e.g. ``"conv_abc123"``.
    :param created_at: Unix epoch timestamp of creation.
    :param title: Optional user-assigned title.
    """

    id: str
    created_at: int
    title: str | None = None


# ── Conversation item data types ───────────────────────


class MessageData(BaseModel):
    """
    Data for a message item (user or assistant).

    :param role: ``"user"`` or ``"assistant"``.
    :param content: Heterogeneous content blocks, e.g.
        ``[{"type": "input_text", "text": "Hello"}]``.
    :param agent: Agent name (required for assistant messages,
        absent for user). Serialized as ``"model"`` in JSON.
    """

    role: Literal["user", "assistant"]
    # Heterogeneous content blocks (input_text, output_text, input_image, etc.)
    content: list[dict[str, Any]]
    agent: str | None = Field(default=None, serialization_alias="model")

    @model_validator(mode="after")
    def check_agent_for_assistant(self) -> MessageData:
        """
        Validate that assistant messages have an agent and user
        messages do not.

        :returns: The validated instance.
        :raises ValueError: If an assistant message is missing
            ``agent``.
        """
        if self.role == "assistant" and self.agent is None:
            raise ValueError("assistant messages require 'agent'")
        return self


class FunctionCallData(BaseModel):
    """
    Data for a function_call item.

    :param agent: Agent name. Serialized as ``"model"`` in JSON.
    :param name: Tool function name, e.g. ``"search.web"``.
    :param arguments: JSON-encoded arguments string.
    :param call_id: Unique call identifier from the LLM,
        e.g. ``"call_abc123"``.
    """

    agent: str = Field(serialization_alias="model")
    name: str
    arguments: str
    call_id: str


class FunctionCallOutputData(BaseModel):
    """
    Data for a function_call_output item.

    :param call_id: The call_id this output corresponds to,
        e.g. ``"call_abc123"``.
    :param output: The tool's string result.
    """

    call_id: str
    output: str


class ReasoningData(BaseModel):
    """
    Data for a reasoning item.

    :param agent: Agent name. Serialized as ``"model"`` in JSON.
    :param summary: Summary text blocks,
        e.g. ``[{"type": "summary_text", "text": "..."}]``.
    :param content: Raw reasoning content blocks, or ``None`` if
        redacted.
    :param encrypted_content: Encrypted reasoning content, or
        ``None``.
    """

    agent: str = Field(serialization_alias="model")
    # Summary text blocks, e.g. [{"type": "summary_text", "text": "..."}]
    summary: list[dict[str, str]]
    # Raw reasoning content blocks; nullable (may be redacted).
    content: list[dict[str, str]] | None = None
    encrypted_content: str | None = None


ItemData = MessageData | FunctionCallData | FunctionCallOutputData | ReasoningData

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

    :param item_type: The item type string, e.g. ``"message"``,
        ``"function_call"``.
    :param raw: The raw dict from the DB ``data`` column.
    :returns: A validated ItemData instance.
    :raises ValueError: If ``item_type`` is unknown.
    """
    cls = ITEM_TYPE_TO_DATA_CLS.get(item_type)
    if cls is None:
        raise ValueError(f"unknown item type: {item_type!r}")
    return cls(**raw)  # type: ignore[return-value]


def _validate_type_matches_data(item_type: str, data: ItemData) -> None:
    """
    Validate that ``data`` is the correct model for ``item_type``.

    :param item_type: The declared type string, e.g. ``"message"``.
    :param data: The data model instance to validate.
    :raises ValueError: If ``item_type`` is unknown or ``data`` is
        the wrong model.
    """
    expected = ITEM_TYPE_TO_DATA_CLS.get(item_type)
    if expected is None:
        raise ValueError(f"unknown item type: {item_type!r}")
    if not isinstance(data, expected):
        raise ValueError(
            f"item type {item_type!r} requires {expected.__name__}, got {type(data).__name__}"
        )


# ── Conversation items ─────────────────────────────────


class NewConversationItem(BaseModel):
    """
    An item that has not yet been persisted. No ID or timestamp.

    :param type: Item type, e.g. ``"message"``,
        ``"function_call"``.
    :param response_id: The task/response ID this item belongs to.
    :param data: The typed data payload (MessageData, etc.).
    """

    type: str
    response_id: str
    data: ItemData

    @model_validator(mode="after")
    def check_type_matches_data(self) -> NewConversationItem:
        """
        Ensure ``type`` field is consistent with ``data`` model.

        :returns: The validated instance.
        :raises ValueError: If ``type`` does not match ``data``.
        """
        _validate_type_matches_data(self.type, self.data)
        return self


class ConversationItem(BaseModel):
    """
    A persisted item with a store-assigned ID.

    :param id: Store-assigned item ID, e.g. ``"msg_abc123"``.
    :param type: Item type, e.g. ``"message"``,
        ``"function_call"``.
    :param status: Item status, e.g. ``"completed"``.
    :param response_id: The task/response ID this item belongs to.
    :param created_at: Unix epoch timestamp of creation.
    :param data: The typed data payload (MessageData, etc.).
    """

    id: str
    type: str
    status: str
    response_id: str
    created_at: int
    data: ItemData

    @model_validator(mode="after")
    def check_type_matches_data(self) -> ConversationItem:
        """
        Ensure ``type`` field is consistent with ``data`` model.

        :returns: The validated instance.
        :raises ValueError: If ``type`` does not match ``data``.
        """
        _validate_type_matches_data(self.type, self.data)
        return self
