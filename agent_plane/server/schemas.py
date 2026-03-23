"""Pydantic models for the API layer — request and response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Shared ──────────────────────────────────────────────────────


class PaginatedList(BaseModel):
    object: str = "list"
    data: list[Any] = Field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


# ── Agents ──────────────────────────────────────────────────────


class AgentObject(BaseModel):
    id: str
    object: str = "agent"
    name: str
    description: str | None = None
    created_at: int


class AgentDeleted(BaseModel):
    id: str
    object: str = "agent.deleted"
    deleted: bool = True


# ── Files ───────────────────────────────────────────────────────


class FileObject(BaseModel):
    id: str
    object: str = "file"
    filename: str
    bytes: int
    created_at: int


class FileDeleted(BaseModel):
    id: str
    object: str = "file"
    deleted: bool = True


# ── Conversations ───────────────────────────────────────────────


class ConversationObject(BaseModel):
    id: str
    object: str = "conversation"
    title: str | None = None
    created_at: int


class ConversationDeleted(BaseModel):
    id: str
    object: str = "conversation.deleted"
    deleted: bool = True


class ConversationRef(BaseModel):
    id: str


# ── Responses ───────────────────────────────────────────────────


class UsageDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    output_tokens_details: UsageDetails = Field(default_factory=UsageDetails)
    total_tokens: int = 0


class ErrorDetail(BaseModel):
    code: str
    message: str


class IncompleteDetails(BaseModel):
    reason: str


class CreateResponseRequest(BaseModel):
    model: str
    input: str | list[Any]
    stream: bool = False
    background: bool = False
    store: bool = True
    instructions: str | None = None
    previous_response_id: str | None = None
    conversation: ConversationRef | None = None
    context_management: list[Any] | None = None
    # Ignored fields — agent controls these (silently dropped)
    temperature: float | None = None
    top_p: float | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    reasoning: Any | None = None
    max_output_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    parallel_tool_calls: bool | None = None
    max_tool_calls: int | None = None
    top_logprobs: int | None = None


class ResponseObject(BaseModel):
    id: str
    object: str = "response"
    status: str
    model: str
    created_at: int
    completed_at: int | None = None
    output: list[Any] = Field(default_factory=list)
    background: bool = False
    store: bool = True
    usage: Usage | None = None
    previous_response_id: str | None = None
    conversation: ConversationRef | None = None
    instructions: str | None = None
    error: ErrorDetail | None = None
    incomplete_details: IncompleteDetails | None = None


class ResponseDeleted(BaseModel):
    id: str
    object: str = "response.deleted"
    deleted: bool = True
