"""Pydantic models for the API layer — request and response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Shared ──────────────────────────────────────────────────────


class PaginatedList(BaseModel):
    """
    A paginated list response following cursor-based pagination.

    :param object: Fixed resource type, always ``"list"``.
    :param data: Page of results. Items are heterogeneous
        (``ResponseObject``, ``ConversationObject``, ``FileObject``,
        or dicts) and list is invariant, so no single concrete type
        satisfies all callers.
    :param first_id: ID of the first item in the page, or ``None``
        if the page is empty, e.g. ``"resp_abc123"``.
    :param last_id: ID of the last item in the page, or ``None``
        if the page is empty, e.g. ``"resp_xyz789"``.
    :param has_more: Whether more items exist beyond this page.
    """

    object: str = "list"
    # Any: items are heterogeneous (ResponseObject, ConversationObject,
    # FileObject, or dicts) and list is invariant, so no single concrete
    # type satisfies all callers.
    data: list[Any] = Field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


# ── Agents ──────────────────────────────────────────────────────


class AgentObject(BaseModel):
    """
    API representation of a registered agent.

    :param id: Unique agent identifier, e.g. ``"ag_abc123"``.
    :param object: Fixed resource type, always ``"agent"``.
    :param name: Human-readable agent name,
        e.g. ``"research-agent"``.
    :param description: Optional free-text description of the
        agent's purpose.
    :param created_at: Unix epoch timestamp of creation.
    """

    id: str
    object: str = "agent"
    name: str
    description: str | None = None
    created_at: int


class AgentDeleted(BaseModel):
    """
    Confirmation payload returned after deleting an agent.

    :param id: ID of the deleted agent, e.g. ``"ag_abc123"``.
    :param object: Fixed resource type, always
        ``"agent.deleted"``.
    :param deleted: Always ``True``.
    """

    id: str
    object: str = "agent.deleted"
    deleted: bool = True


# ── Files ───────────────────────────────────────────────────────


class FileObject(BaseModel):
    """
    API representation of an uploaded file.

    :param id: Unique file identifier, e.g. ``"file_abc123"``.
    :param object: Fixed resource type, always ``"file"``.
    :param filename: Original filename, e.g. ``"report.pdf"``.
    :param bytes: File size in bytes.
    :param created_at: Unix epoch timestamp of upload.
    """

    id: str
    object: str = "file"
    filename: str
    bytes: int
    created_at: int


class FileDeleted(BaseModel):
    """
    Confirmation payload returned after deleting a file.

    :param id: ID of the deleted file, e.g. ``"file_abc123"``.
    :param object: Fixed resource type, always ``"file"``.
    :param deleted: Always ``True``.
    """

    id: str
    object: str = "file"
    deleted: bool = True


# ── Conversations ───────────────────────────────────────────────


class ConversationObject(BaseModel):
    """
    API representation of a conversation.

    :param id: Unique conversation identifier,
        e.g. ``"conv_abc123"``.
    :param object: Fixed resource type, always
        ``"conversation"``.
    :param title: Optional user-assigned conversation title.
    :param created_at: Unix epoch timestamp of creation.
    :param updated_at: Unix epoch timestamp of the last
        update, e.g. ``1774118400``.
    """

    id: str
    object: str = "conversation"
    title: str | None = None
    created_at: int
    updated_at: int


class ConversationDeleted(BaseModel):
    """
    Confirmation payload returned after deleting a conversation.

    :param id: ID of the deleted conversation,
        e.g. ``"conv_abc123"``.
    :param object: Fixed resource type, always
        ``"conversation.deleted"``.
    :param deleted: Always ``True``.
    """

    id: str
    object: str = "conversation.deleted"
    deleted: bool = True


class ConversationRef(BaseModel):
    """
    Lightweight reference to a conversation, used in request and
    response bodies where only the conversation ID is needed.

    :param id: Conversation identifier, e.g. ``"conv_abc123"``.
    """

    id: str


# ── Responses ───────────────────────────────────────────────────


class UsageDetails(BaseModel):
    """
    Breakdown of output token usage.

    :param reasoning_tokens: Number of tokens consumed by
        chain-of-thought reasoning.
    """

    reasoning_tokens: int = 0


class Usage(BaseModel):
    """
    Token usage statistics for a response.

    :param input_tokens: Number of input (prompt) tokens consumed.
    :param output_tokens: Number of output (completion) tokens
        generated.
    :param output_tokens_details: Breakdown of output token usage
        (e.g. reasoning tokens).
    :param total_tokens: Sum of input and output tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    output_tokens_details: UsageDetails = Field(default_factory=UsageDetails)
    total_tokens: int = 0


class ErrorDetail(BaseModel):
    """
    Machine-readable error information attached to a failed response.

    :param code: Error code string, e.g. ``"server_error"``,
        ``"invalid_input"``.
    :param message: Human-readable error description.
    """

    code: str
    message: str


class IncompleteDetails(BaseModel):
    """
    Details explaining why a response is incomplete.

    :param reason: Reason the response stopped early, e.g.
        ``"max_output_tokens"``, ``"max_tool_calls"``.
    """

    reason: str


class CreateResponseRequest(BaseModel):
    """
    Request body for ``POST /v1/responses``.

    :param model: Agent name to invoke, e.g.
        ``"research-agent"``. Must match a registered agent.
    :param input: User input — either a plain string (converted
        to a single ``input_text`` block) or a list of content
        blocks, e.g.
        ``[{"type": "input_text", "text": "Hello"}]``.
    :param stream: If ``True``, return an SSE stream instead of
        blocking until completion.
    :param background: If ``True``, the task runs in the
        background and the caller may poll for results.
    :param store: Must be ``True`` (persisted responses). The
        server rejects ``False``.
    :param instructions: Per-request system instructions that
        override the agent's default instructions.
    :param previous_response_id: ID of the prior response in the
        conversation thread, e.g. ``"resp_abc123"``. Enables
        multi-turn continuation and steering.
    :param conversation: Explicit conversation reference for
        fork validation. Must match the conversation that owns
        ``previous_response_id``.
    :param reasoning: Reasoning configuration,
        e.g. ``{"effort": "medium"}``.
    :param context_management: Compaction strategy objects,
        e.g. ``[{"type": "compaction", ...}]``.
    :param temperature: Ignored — agent controls this. Silently
        dropped.
    :param top_p: Ignored — agent controls this. Silently
        dropped.
    :param tools: Optional list of client-specified tools in standard
        OpenAI function format. When the LLM invokes one, the
        ``function_call`` output items are returned to the caller (the
        response completes) rather than being executed server-side. The
        caller handles execution and continues via
        ``previous_response_id``. Returns 400 if any entry is malformed
        or missing ``function.name``, e.g.
        ``[{"type": "function", "function": {"name": "get_weather",
        "description": "...", "parameters": {...}}}]``.
    :param tool_choice: Ignored — agent controls this. Silently
        dropped.
    :param max_output_tokens: Ignored — agent controls this.
        Silently dropped.
    :param frequency_penalty: Ignored — agent controls this.
        Silently dropped.
    :param presence_penalty: Ignored — agent controls this.
        Silently dropped.
    :param parallel_tool_calls: Ignored — agent controls this.
        Silently dropped.
    :param max_tool_calls: Ignored — agent controls this.
        Silently dropped.
    :param top_logprobs: Ignored — agent controls this. Silently
        dropped.
    """

    model: str
    # Heterogeneous content blocks (input_text, input_image, input_file)
    # or a plain string shorthand; shape varies by block type.
    input: str | list[dict[str, Any]]
    stream: bool = False
    background: bool = False
    store: bool = True
    instructions: str | None = None
    previous_response_id: str | None = None
    conversation: ConversationRef | None = None
    # Reasoning config, e.g. {"effort": "low"|"medium"|"high"}
    reasoning: dict[str, str] | None = None
    # Compaction strategy objects, e.g. [{"type": "compaction", ...}]
    context_management: list[dict[str, Any]] | None = None
    # Ignored fields — agent controls these; silently dropped.
    # Typed loosely because we only need to accept and discard them.
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    max_output_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    parallel_tool_calls: bool | None = None
    max_tool_calls: int | None = None
    top_logprobs: int | None = None


class ResponseObject(BaseModel):
    """
    API representation of a response (task execution result).

    :param id: Unique response identifier, e.g.
        ``"resp_abc123"``.
    :param object: Fixed resource type, always ``"response"``.
    :param status: Lifecycle status, one of ``"queued"``,
        ``"in_progress"``, ``"completed"``, ``"failed"``,
        ``"incomplete"``, ``"cancelled"``.
    :param model: Agent name that produced this response,
        e.g. ``"research-agent"``.
    :param created_at: Unix epoch timestamp of creation.
    :param completed_at: Unix epoch timestamp of completion, or
        ``None`` if not yet complete.
    :param output: Heterogeneous output items (messages,
        reasoning, function_calls) serialized as dicts; shape
        varies by item type. Empty for non-completed responses.
    :param background: Whether this response was created as a
        background task.
    :param store: Whether this response is persisted. Always
        ``True``.
    :param usage: Token usage statistics, or ``None`` if not
        yet available.
    :param previous_response_id: ID of the prior response in
        the conversation thread, or ``None`` for the first turn.
    :param conversation: Reference to the owning conversation.
    :param instructions: Per-request system instructions
        override, or ``None``.
    :param reasoning: Reasoning configuration,
        e.g. ``{"effort": "medium"}``.
    :param error: Error details if the response failed.
    :param incomplete_details: Details if the response is
        incomplete (e.g. hit token limit).
    """

    id: str
    object: str = "response"
    status: str
    model: str
    created_at: int
    completed_at: int | None = None
    # Heterogeneous output items (messages, reasoning, function_calls);
    # shape varies by item type.
    output: list[dict[str, Any]] = Field(default_factory=list)
    background: bool = False
    store: bool = True
    usage: Usage | None = None
    previous_response_id: str | None = None
    conversation: ConversationRef | None = None
    instructions: str | None = None
    reasoning: dict[str, str] | None = None
    error: ErrorDetail | None = None
    incomplete_details: IncompleteDetails | None = None


class ResponseDeleted(BaseModel):
    """
    Confirmation payload returned after deleting a response.

    :param id: ID of the deleted response,
        e.g. ``"resp_abc123"``.
    :param object: Fixed resource type, always
        ``"response.deleted"``.
    :param deleted: Always ``True``.
    """

    id: str
    object: str = "response.deleted"
    deleted: bool = True


class ToolResult(BaseModel):
    """
    A single tool result submitted by the client via PATCH.

    :param call_id: The tool call ID that this result
        corresponds to, e.g. ``"call_abc123"``.
    :param output: The tool's string output,
        e.g. ``'["paper1.pdf", "paper2.pdf"]'``.
    """

    call_id: str
    output: str


class PatchResponseRequest(BaseModel):
    """
    Request body for ``PATCH /v1/responses/{id}``.

    Submits tool results for tunneled client-side tool calls
    that have ``status: "action_required"`` in the response
    output.

    :param tool_results: List of tool results to submit. Each
        entry maps a ``call_id`` to its output string.
    """

    tool_results: list[ToolResult]
