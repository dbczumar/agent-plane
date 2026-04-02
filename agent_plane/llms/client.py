"""
Main LLM client — presents the OpenAI Responses API interface and
routes to provider adapters.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from agent_plane.llms._responses_to_chat import (
    chat_response_to_response,
    chat_stream_to_response_events,
    responses_input_to_chat_messages,
)
from agent_plane.llms.adapters import get_adapter
from agent_plane.llms.adapters.openai import OpenAIAdapter
from agent_plane.llms.errors import (
    ContextWindowExceededError,
    LLMErrorDetail,
    PermanentLLMError,
    RetryableLLMError,
)
from agent_plane.llms.routing import parse_model_string
from agent_plane.llms.types import Response, ResponseStreamEvent, RetryConfig

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
            classified = _classify_error(exc, retry_config.status_codes)
            if isinstance(classified, PermanentLLMError):
                raise classified from exc
            last_error = classified
            if attempt + 1 < retry_config.max_attempts:
                _backoff_sleep(attempt, retry_config)

    assert last_error is not None
    raise last_error


@dataclass
class _OverflowTokens:
    """
    Token counts parsed from a provider context-overflow error body.

    :param max_context_tokens: The model's context window size as
        reported by the provider, e.g. ``128000``.
    :param actual_tokens: The token count the provider measured for
        the rejected request, e.g. ``142000``.
    """

    max_context_tokens: int
    actual_tokens: int


def _detect_context_overflow(body: str) -> _OverflowTokens | None:
    """
    Parse provider-specific context-overflow error messages and
    extract token counts.

    Matches conservatively — only well-known error shapes produce a
    result. Unknown 400 errors return ``None`` so they propagate as
    :class:`PermanentLLMError` rather than entering a
    compact-retry loop.

    Supported providers:
    - **OpenAI**: status 400, ``error.code == "context_length_exceeded"``
    - **Anthropic**: status 400, message ``"{input} + {max_tokens} > {limit}"``
      or ``"prompt is too long: {actual} tokens > {limit} maximum"``
    - **Gemini**: status 400, message contains
      ``"input token count ({actual}) exceeds the maximum number
      of tokens allowed ({limit})"``

    :param body: The raw HTTP response body string from the provider.
    :returns: Parsed token counts, or ``None`` if the body does not
        match any known overflow pattern.
    """
    # OpenAI: {"error": {"code": "context_length_exceeded", "message": "..."}}
    # Message: "This model's maximum context length is 128000 tokens.
    #           However, you requested 142000 tokens"
    try:
        parsed = json.loads(body)
        error_obj = parsed.get("error", {})
        if error_obj.get("code") == "context_length_exceeded":
            msg = error_obj.get("message", "")
            max_match = re.search(r"maximum context length is (\d+) tokens", msg)
            actual_match = re.search(r"you requested (\d+) tokens", msg)
            if max_match and actual_match:
                return _OverflowTokens(
                    max_context_tokens=int(max_match.group(1)),
                    actual_tokens=int(actual_match.group(1)),
                )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Anthropic: "{input} + {max_tokens} > {limit}"
    # e.g. "197202 + 21333 > 200000"
    anthropic_sum = re.search(r"(\d+)\s*\+\s*\d+\s*>\s*(\d+)", body)
    if anthropic_sum:
        return _OverflowTokens(
            max_context_tokens=int(anthropic_sum.group(2)),
            actual_tokens=int(anthropic_sum.group(1)),
        )

    # Anthropic: "prompt is too long: {actual} tokens > {limit} maximum"
    anthropic_long = re.search(
        r"prompt is too long:\s*(\d+)\s*tokens\s*>\s*(\d+)\s*maximum",
        body,
    )
    if anthropic_long:
        return _OverflowTokens(
            max_context_tokens=int(anthropic_long.group(2)),
            actual_tokens=int(anthropic_long.group(1)),
        )

    # Gemini: "input token count ({actual}) exceeds the maximum number of tokens allowed ({limit})"
    gemini_match = re.search(
        r"input token count \((\d+)\) exceeds the maximum number of tokens allowed \((\d+)\)",
        body,
    )
    if gemini_match:
        return _OverflowTokens(
            max_context_tokens=int(gemini_match.group(2)),
            actual_tokens=int(gemini_match.group(1)),
        )

    return None


def _classify_error(
    exc: Exception,
    retryable_status_codes: list[int],
) -> RetryableLLMError | PermanentLLMError:
    """
    Classify an exception as retryable or permanent.

    :param exc: The exception raised by the adapter.
    :param retryable_status_codes: HTTP status codes configured
        as retryable, e.g. ``[429, 500, 502, 503]``.
    :returns: A classified LLM error.
    """
    if isinstance(exc, httpx.TimeoutException):
        return RetryableLLMError(
            f"LLM request timed out: {exc}",
            code="timeout",
            detail=LLMErrorDetail(),
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text
        detail = LLMErrorDetail(status_code=status, response_body=body)
        # HTTP 400 may be a context-window overflow — check before
        # the generic retryable/permanent split so the workflow can
        # catch ContextWindowExceededError and compact-retry.
        if status == 400:
            overflow = _detect_context_overflow(body)
            if overflow is not None:
                return ContextWindowExceededError(
                    f"Context window exceeded: {overflow.actual_tokens} tokens"
                    f" > {overflow.max_context_tokens} max",
                    code="context_length_exceeded",
                    detail=detail,
                    max_context_tokens=overflow.max_context_tokens,
                    actual_tokens=overflow.actual_tokens,
                )
        msg = f"LLM returned HTTP {status}"
        if status in retryable_status_codes:
            return RetryableLLMError(msg, code=str(status), detail=detail)
        return PermanentLLMError(msg, code=str(status), detail=detail)
    return PermanentLLMError(
        f"LLM call failed: {exc}",
        code="connection_error",
        detail=LLMErrorDetail(),
    )


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
