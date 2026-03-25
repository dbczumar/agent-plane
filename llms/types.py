"""
Response and streaming event types for the LLM client.

These dataclasses mirror the OpenAI Responses API types so that
``workflow.py``'s ``_response_to_dict()`` and ``_accumulate_stream()``
work unchanged — they access ``.type``, ``.output``, ``.delta``,
``.response``, ``.content``, ``.text``, ``.call_id``, ``.name``,
and ``.arguments`` attributes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputText:
    """
    A text content part within a message output.

    :param text: The text content, e.g. ``"Hello! How can I help?"``.
    :param type: Always ``"output_text"``.
    """

    text: str
    type: str = "output_text"


@dataclass
class MessageOutput:
    """
    An assistant message in the response output.

    :param content: List of content parts, e.g.
        ``[OutputText(text="Hello")]``.
    :param type: Always ``"message"``.
    """

    content: list[OutputText]
    type: str = "message"


@dataclass
class FunctionCallOutput:
    """
    A tool/function call in the response output.

    :param call_id: Unique identifier for the tool call, e.g.
        ``"call_abc123"``.
    :param name: The function name, e.g. ``"get_weather"``.
    :param arguments: JSON-encoded arguments string, e.g.
        ``'{"city": "London"}'``.
    :param type: Always ``"function_call"``.
    """

    call_id: str
    name: str
    arguments: str
    type: str = "function_call"


@dataclass
class Usage:
    """
    Token usage information.

    :param input_tokens: Number of input/prompt tokens.
    :param output_tokens: Number of output/completion tokens.
    :param total_tokens: Total tokens (input + output).
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class Response:
    """
    A completed LLM response.

    :param output: List of output items — ``MessageOutput`` and/or
        ``FunctionCallOutput`` instances.
    :param model: The model identifier that produced the response,
        e.g. ``"claude-sonnet-4-20250514"``.
    :param usage: Token usage information, or ``None`` if unavailable.
    """

    output: list[MessageOutput | FunctionCallOutput]
    model: str
    usage: Usage | None = None


# ── Streaming event types ─────────────────────────────────


@dataclass
class ResponseTextDeltaEvent:
    """
    Incremental text token from the assistant.

    :param delta: The text fragment, e.g. ``"Hello"``.
    :param type: Always ``"response.output_text.delta"``.
    """

    delta: str
    type: str = "response.output_text.delta"


@dataclass
class ResponseReasoningTextDeltaEvent:
    """
    Incremental reasoning token (full chain-of-thought).
    Only emitted by providers that support reasoning (e.g. OpenAI).

    :param delta: The reasoning text fragment.
    :param type: Always ``"response.reasoning_text.delta"``.
    """

    delta: str
    type: str = "response.reasoning_text.delta"


@dataclass
class ResponseReasoningSummaryTextDeltaEvent:
    """
    Incremental reasoning summary token.
    Only emitted when ``reasoning.summary`` is configured.

    :param delta: The summary text fragment.
    :param type: Always ``"response.reasoning_summary_text.delta"``.
    """

    delta: str
    type: str = "response.reasoning_summary_text.delta"


@dataclass
class ResponseReasoningStartedEvent:
    """
    Emitted once when a reasoning block begins.

    Fired when the model starts reasoning, even when reasoning content
    is encrypted and no delta events will follow. Allows clients to
    show a ``[thinking...]`` indicator regardless of org verification
    status.

    :param type: Always ``"response.reasoning.started"``.
    """

    type: str = "response.reasoning.started"


@dataclass
class ResponseCompletedEvent:
    """
    Emitted when the full response is complete.

    :param response: The assembled ``Response`` object.
    :param type: Always ``"response.completed"``.
    """

    response: Response
    type: str = "response.completed"


# Union type for all streaming events
ResponseStreamEvent = (
    ResponseTextDeltaEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningStartedEvent
    | ResponseCompletedEvent
)
