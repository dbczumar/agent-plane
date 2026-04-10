"""
Main LLM client — presents the OpenAI Responses API interface and
routes to provider adapters. All methods are async for non-blocking
I/O.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
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
from agent_plane.llms.types import (
    Response,
    ResponseStreamEvent,
    RetryConfig,
)
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

    async def create(
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
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """
        Create a response from the LLM, routing to the
        appropriate provider based on the model string.

        :param input: Responses API input items, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        :param instructions: System instructions string.
        :param model: Provider-prefixed model string, e.g.
            ``"anthropic/claude-sonnet-4-20250514"`` or
            ``"gpt-5.4"``.
        :param tools: OpenAI-format tool schemas, or ``None``.
        :param reasoning: Reasoning configuration dict, e.g.
            ``{"effort": "high", "summary": "concise"}``.
        :param stream: If ``True``, return an async iterator of
            streaming events. If ``False``, return a
            :class:`Response`.
        :param connection_params: Per-call connection overrides.
            Keys are provider-specific, e.g.
            ``{"api_key": "...", "base_url": "..."}`` for
            OpenAI-compatible providers, or
            ``{"aws_region": "us-west-2"}`` for Bedrock.
            ``None`` uses the adapter's default credentials.
        :param timeout: Request timeout in seconds. ``None``
            uses the adapter's default (120s non-streaming, 300s
            streaming).
        :param retry: Retry policy for transient failures
            (timeouts, rate limits). ``None`` disables
            client-level retries. Useful for standalone calls
            outside the workflow engine.
        :param kwargs: Additional provider-specific kwargs (e.g.
            ``temperature``, ``max_tokens``).
        :returns: A :class:`Response` when ``stream=False``, or
            an async iterator of :data:`ResponseStreamEvent`
            when ``stream=True``.
        :raises PermanentLLMError: On non-retryable errors.
        :raises RetryableLLMError: When all retry attempts are
            exhausted.
        """

        async def call_fn() -> Response | AsyncIterator[ResponseStreamEvent]:
            """
            Dispatch to the adapter.

            :returns: Response or streaming event iterator.
            """
            return await self._do_create(
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
            return await call_fn()
        return await _execute_with_retry(call_fn, retry)

    async def _do_create(
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
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """
        Route the LLM call to the appropriate provider adapter.

        :param input: Responses API input items.
        :param instructions: System instructions string.
        :param model: Provider-prefixed model string.
        :param tools: Tool schemas or ``None``.
        :param reasoning: Reasoning config or ``None``.
        :param stream: Enable streaming.
        :param connection_params: Connection overrides or
            ``None``.
        :param timeout: Timeout in seconds or ``None``.
        :param kwargs: Additional provider-specific kwargs.
        :returns: Response or async streaming event iterator.
        """
        routed = parse_model_string(model)
        adapter = get_adapter(routed.provider)

        # OpenAI supports the Responses API natively — use it
        # directly so reasoning token events flow through
        # unmodified.
        if isinstance(adapter, OpenAIAdapter):
            return await adapter.responses_create(
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

        messages = responses_input_to_chat_messages(
            input,
            instructions,
        )

        extra: dict[str, Any] = dict(kwargs)
        if reasoning:
            extra["reasoning_effort"] = reasoning.get("effort")

        if stream:
            chunks = await adapter.chat_completions(
                messages,
                routed.model,
                tools,
                True,
                extra,
                connection_params=connection_params,
                timeout=timeout,
            )
            assert not isinstance(chunks, dict)
            return chat_stream_to_response_events(
                chunks,
                model=routed.model,
            )

        result = await adapter.chat_completions(
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


async def _execute_with_retry(
    call_fn: Callable[[], Awaitable[_T]],
    retry_config: RetryConfig,
) -> _T:
    """
    Execute ``call_fn`` with retry on transient failures.

    Standalone retry logic for the LLM client. Uses
    ``asyncio.sleep`` for backoff so the event loop stays free.

    :param call_fn: Zero-argument async callable that performs
        the LLM call.
    :param retry_config: Retry policy (max_attempts, backoff,
        etc.).
    :returns: The successful result from ``call_fn``.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retry attempts are
        exhausted.
    """
    last_error: RetryableLLMError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return await call_fn()
        except (PermanentLLMError, RetryableLLMError):
            raise
        except Exception as exc:
            classified = classify_llm_error(
                exc,
                retry_config.status_codes,
            )
            if isinstance(classified, PermanentLLMError):
                raise classified from exc
            last_error = classified
            if attempt + 1 < retry_config.max_attempts:
                await _backoff_sleep(attempt, retry_config)

    assert last_error is not None
    raise last_error


async def _backoff_sleep(
    attempt: int,
    config: RetryConfig,
) -> None:
    """
    Sleep with exponential backoff and jitter.

    Uses ``asyncio.sleep`` for non-blocking backoff.

    :param attempt: Zero-based attempt index (0 = first
        attempt).
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
    await asyncio.sleep(delay)


class Client:
    """
    Multi-provider async LLM client.

    Provides ``await client.responses.create()`` matching the
    OpenAI SDK interface, routing to any supported provider based
    on the model string prefix.

    Usage::

        client = Client()
        resp = await client.responses.create(
            input=[{"role": "user", "content": "Hello"}],
            instructions="You are helpful.",
            model="anthropic/claude-sonnet-4-20250514",
        )
    """

    def __init__(self) -> None:
        """Initialize the client with a responses namespace."""
        self.responses = _ResponsesNamespace(self)
