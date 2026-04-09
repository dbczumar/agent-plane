"""
Main LLM client — presents the OpenAI Responses API interface and
routes to provider adapters.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from agent_plane.llms._responses_to_chat import (
    chat_response_to_response,
    chat_stream_to_response_events,
    responses_input_to_chat_messages,
)
from agent_plane.llms.adapters import get_adapter
from agent_plane.llms.adapters.openai import OpenAIAdapter
from agent_plane.llms.errors import (
    PermanentLLMError,
    RetryableLLMError,
)
from agent_plane.llms.routing import parse_model_string
from agent_plane.llms.types import Response, ResponseStreamEvent, RetryConfig
from agent_plane.runtime.llm_retry import classify_llm_error

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")


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
        connection_params: dict[str, str] | None = None,
        timeout: int | None = None,
        retry: RetryConfig | None = None,
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
        :param connection_params: Per-call connection overrides.
            Keys are provider-specific, e.g.
            ``{"api_key": "...", "base_url": "..."}`` for
            OpenAI-compatible providers, or
            ``{"aws_region": "us-west-2"}`` for Bedrock.
            ``None`` uses the adapter's default credentials.
        :param timeout: Request timeout in seconds. ``None`` uses
            the adapter's default (120s non-streaming, 300s streaming).
        :param retry: Retry policy for transient failures (timeouts,
            rate limits). ``None`` disables client-level retries.
            Useful for standalone calls outside the workflow engine.
        :param kwargs: Additional provider-specific kwargs (e.g.
            ``temperature``, ``max_tokens``).
        :returns: A :class:`Response` when ``stream=False``, or an
            iterator of :data:`ResponseStreamEvent` when
            ``stream=True``.
        :raises PermanentLLMError: On non-retryable errors.
        :raises RetryableLLMError: When all retry attempts are
            exhausted.
        """

        def call_fn() -> Response | Iterator[ResponseStreamEvent]:
            return self._do_create(
                input=input,
                instructions=instructions,
                model=model,
                tools=tools,
                reasoning=reasoning,
                stream=stream,
                connection_params=connection_params,
                timeout=timeout,
                **kwargs,
            )

        if retry is None:
            return call_fn()
        return _execute_with_retry(call_fn, retry)

    def _do_create(
        self,
        *,
        input: list[dict[str, Any]],
        instructions: str | None,
        model: str,
        tools: list[dict[str, Any]] | None,
        reasoning: dict[str, str] | None,
        stream: bool,
        connection_params: dict[str, str] | None,
        timeout: int | None,
        **kwargs: Any,
    ) -> Response | Iterator[ResponseStreamEvent]:
        """
        Route the LLM call to the appropriate provider adapter.

        :param input: Responses API input items.
        :param instructions: System instructions string.
        :param model: Provider-prefixed model string.
        :param tools: Tool schemas or ``None``.
        :param reasoning: Reasoning config or ``None``.
        :param stream: Enable streaming.
        :param connection_params: Connection overrides or ``None``.
        :param timeout: Timeout in seconds or ``None``.
        :param kwargs: Additional provider-specific kwargs.
        :returns: Response or streaming event iterator.
        """
        routed = parse_model_string(model)
        adapter = get_adapter(routed.provider)

        # OpenAI supports the Responses API natively — use it directly
        # so reasoning token events flow through unmodified.
        if isinstance(adapter, OpenAIAdapter):
            return adapter.responses_create(
                input=input,
                instructions=instructions,
                model=routed.model,
                tools=tools,
                reasoning=reasoning,
                stream=stream,
                connection_params=connection_params,
                timeout=timeout,
                **kwargs,
            )

        messages = responses_input_to_chat_messages(input, instructions)

        extra: dict[str, Any] = dict(kwargs)
        if reasoning:
            extra["reasoning_effort"] = reasoning.get("effort")

        if stream:
            chunks = adapter.chat_completions(
                messages,
                routed.model,
                tools,
                True,
                extra,
                connection_params=connection_params,
                timeout=timeout,
            )
            assert not isinstance(chunks, dict)
            return chat_stream_to_response_events(chunks, model=routed.model)

        result = adapter.chat_completions(
            messages,
            routed.model,
            tools,
            False,
            extra,
            connection_params=connection_params,
            timeout=timeout,
        )
        assert isinstance(result, dict)
        return chat_response_to_response(result)


def _execute_with_retry(
    call_fn: Callable[[], _T],
    retry_config: RetryConfig,
) -> _T:
    """
    Execute ``call_fn`` with retry on transient failures.

    Standalone retry logic for the LLM client — no SSE events
    or workflow dependencies. For use in scripts, notebooks, and
    other contexts outside the agent workflow engine.

    :param call_fn: Zero-argument callable that performs the LLM
        call.
    :param retry_config: Retry policy (max_attempts, backoff, etc.).
    :returns: The successful result from ``call_fn``.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retry attempts are exhausted.
    """
    last_error: RetryableLLMError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return call_fn()
        except (PermanentLLMError, RetryableLLMError):
            raise
        except Exception as exc:
            classified = classify_llm_error(exc, retry_config.status_codes)
            if isinstance(classified, PermanentLLMError):
                raise classified from exc
            last_error = classified
            if attempt + 1 < retry_config.max_attempts:
                _backoff_sleep(attempt, retry_config)

    assert last_error is not None
    raise last_error


def _backoff_sleep(attempt: int, config: RetryConfig) -> None:
    """
    Sleep with exponential backoff and jitter.

    :param attempt: Zero-based attempt index (0 = first attempt).
    :param config: Retry policy with backoff parameters.
    """
    delay = min(config.backoff_base**attempt, config.backoff_max)
    delay *= random.uniform(0.5, 1.0)
    _logger.info(
        "LLM retry %d/%d after %.1fs",
        attempt + 2,
        config.max_attempts,
        delay,
    )
    time.sleep(delay)


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
