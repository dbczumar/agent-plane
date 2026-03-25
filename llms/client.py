"""
Main LLM client — presents the OpenAI Responses API interface and
routes to provider adapters.
"""

from __future__ import annotations

from typing import Any, Iterator

from llms._responses_to_chat import (
    chat_response_to_response,
    chat_stream_to_response_events,
    responses_input_to_chat_messages,
)
from llms.adapters import get_adapter
from llms.routing import parse_model_string
from llms.types import Response, ResponseStreamEvent


class _ResponsesNamespace:
    """
    Namespace providing ``client.responses.create()`` to mirror
    the OpenAI SDK interface.

    :param client: The parent :class:`Client` instance.
    """

    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        *,
        input: list[dict[str, Any]],
        instructions: str | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Response | Iterator[ResponseStreamEvent]:
        """
        Create a response from the LLM, routing to the appropriate
        provider based on the model string.

        :param input: Responses API input items, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        :param instructions: System instructions string.
        :param model: Provider-prefixed model string, e.g.
            ``"anthropic/claude-sonnet-4-20250514"`` or ``"gpt-5.4"``.
        :param tools: OpenAI-format tool schemas, or ``None``.
        :param reasoning: Reasoning configuration dict, e.g.
            ``{"effort": "high", "summary": "concise"}``.
        :param stream: If ``True``, return an iterator of streaming
            events. If ``False``, return a :class:`Response`.
        :param kwargs: Additional provider-specific kwargs (e.g.
            ``temperature``, ``max_tokens``).
        :returns: A :class:`Response` when ``stream=False``, or an
            iterator of :data:`ResponseStreamEvent` when
            ``stream=True``.
        """
        routed = parse_model_string(model)
        adapter = get_adapter(routed.provider)
        messages = responses_input_to_chat_messages(input, instructions)

        extra: dict[str, Any] = dict(kwargs)
        if reasoning:
            extra["reasoning_effort"] = reasoning.get("effort")

        if stream:
            chunks = adapter.chat_completions(
                messages, routed.model, tools, True, extra
            )
            assert not isinstance(chunks, dict)
            return chat_stream_to_response_events(chunks, model=routed.model)

        result = adapter.chat_completions(
            messages, routed.model, tools, False, extra
        )
        assert isinstance(result, dict)
        return chat_response_to_response(result)


class Client:
    """
    Multi-provider LLM client.

    Provides ``client.responses.create()`` matching the OpenAI SDK
    interface, routing to any supported provider based on the model
    string prefix.

    Usage::

        client = Client()
        resp = client.responses.create(
            input=[{"role": "user", "content": "Hello"}],
            instructions="You are helpful.",
            model="anthropic/claude-sonnet-4-20250514",
        )
    """

    def __init__(self) -> None:
        self.responses = _ResponsesNamespace(self)
