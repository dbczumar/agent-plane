"""LLM call retry logic with exponential backoff.

Classifies adapter exceptions as retryable or permanent, computes
backoff delays, and provides a retry loop that emits SSE events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from agent_plane.llms.errors import (
    ContextWindowExceededError,
    LLMErrorDetail,
    PermanentLLMError,
    RetryableLLMError,
)
from agent_plane.spec.types import RetryConfig

_logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    ``PermanentLLMError`` rather than entering a compact-retry loop.

    Supported providers:

    - **OpenAI**: ``error.code == "context_length_exceeded"``
    - **Anthropic**: ``"{input} + {max_tokens} > {limit}"`` or
      ``"prompt is too long: {actual} tokens > {limit} maximum"``
    - **Gemini**: ``"input token count ({actual}) exceeds the
      maximum number of tokens allowed ({limit})"``

    :param body: The raw HTTP response body string from the provider.
    :returns: Parsed token counts, or ``None`` if the body does not
        match any known overflow pattern.
    """
    # OpenAI: {"error": {"code": "context_length_exceeded", ...}}
    try:
        parsed = json.loads(body)
        error_obj = parsed.get("error", {})
        if error_obj.get("code") == "context_length_exceeded":
            msg = error_obj.get("message", "")
            max_m = re.search(r"maximum context length is (\d+) tokens", msg)
            act_m = re.search(r"you requested (\d+) tokens", msg)
            if max_m and act_m:
                return _OverflowTokens(
                    max_context_tokens=int(max_m.group(1)),
                    actual_tokens=int(act_m.group(1)),
                )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Anthropic: "{input} + {max_tokens} > {limit}"
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

    # Gemini: "input token count ({actual}) exceeds ... ({limit})"
    gemini_match = re.search(
        r"input token count \((\d+)\) exceeds the maximum number"
        r" of tokens allowed \((\d+)\)",
        body,
    )
    if gemini_match:
        return _OverflowTokens(
            max_context_tokens=int(gemini_match.group(2)),
            actual_tokens=int(gemini_match.group(1)),
        )

    return None


def classify_llm_error(
    exc: Exception,
    retryable_status_codes: list[int],
) -> RetryableLLMError | PermanentLLMError:
    """
    Classify an adapter exception as retryable or permanent.

    :param exc: The exception raised by the LLM adapter. Typically
        ``httpx.TimeoutException`` or ``httpx.HTTPStatusError``.
    :param retryable_status_codes: HTTP status codes configured as
        retryable, e.g. ``[429, 500, 502, 503]``.
    :returns: A :class:`RetryableLLMError` or
        :class:`PermanentLLMError`.
    """
    if isinstance(exc, httpx.TimeoutException):
        return RetryableLLMError(
            f"LLM request timed out: {exc}",
            code="timeout",
            detail=LLMErrorDetail(),
        )

    if isinstance(exc, httpx.HTTPStatusError):
        return _classify_http_error(exc, retryable_status_codes)

    # Connection errors, DNS failures, etc. — not retryable.
    return PermanentLLMError(
        f"LLM call failed: {exc}",
        code="connection_error",
        detail=LLMErrorDetail(),
    )


def _classify_http_error(
    exc: httpx.HTTPStatusError,
    retryable_status_codes: list[int],
) -> RetryableLLMError | PermanentLLMError:
    """
    Classify an HTTP status error as retryable or permanent.

    HTTP 400 is checked for context-window overflow before the
    generic retryable/permanent split. This allows the executor's
    retry logic to surface ``ContextWindowExceededError`` so the
    workflow can compact and retry.

    :param exc: The HTTP status error from httpx.
    :param retryable_status_codes: Status codes that trigger retry.
    :returns: A :class:`RetryableLLMError`,
        :class:`ContextWindowExceededError`, or
        :class:`PermanentLLMError`.
    """
    status = exc.response.status_code
    body = _safe_response_text(exc.response)
    detail = LLMErrorDetail(status_code=status, response_body=body)
    code = str(status)
    message = f"LLM returned HTTP {status}: {body}"

    # HTTP 400 may be a context-window overflow — check before
    # the generic split so the workflow can compact-retry.
    if status == 400:
        overflow = _detect_context_overflow(body)
        if overflow is not None:
            return ContextWindowExceededError(
                f"Context window exceeded: "
                f"{overflow.actual_tokens} tokens "
                f"> {overflow.max_context_tokens} max",
                code="context_length_exceeded",
                detail=detail,
                max_context_tokens=overflow.max_context_tokens,
                actual_tokens=overflow.actual_tokens,
            )

    if status in retryable_status_codes:
        return RetryableLLMError(message, code=code, detail=detail)
    return PermanentLLMError(message, code=code, detail=detail)


def compute_backoff_delay(
    attempt_index: int,
    backoff_base: float,
    backoff_max: float,
) -> float:
    """
    Compute the backoff delay with jitter for a retry attempt.

    :param attempt_index: Zero-based retry index (0 = first retry),
        e.g. ``0``.
    :param backoff_base: Exponential backoff base in seconds, e.g.
        ``2.0``.
    :param backoff_max: Maximum delay cap in seconds, e.g. ``30.0``.
    :returns: Delay in seconds with jitter applied, e.g. ``1.47``.
    """
    delay = min(backoff_base**attempt_index, backoff_max)
    # Jitter: multiply by uniform(0.5, 1.0) to avoid thundering herd.
    delay *= random.uniform(0.5, 1.0)
    return delay


def _safe_response_text(response: httpx.Response) -> str:
    """
    Safely extract response body text, truncating if very long.

    :param response: The httpx response object.
    :returns: Response body text, truncated to 1000 chars.
    """
    try:
        text = response.text
    except Exception:
        return "<unreadable response body>"
    if len(text) > 1000:
        return text[:1000] + "..."
    return text


def detail_to_dict(
    detail: LLMErrorDetail | None,
) -> dict[str, Any] | None:
    """
    Convert an :class:`LLMErrorDetail` to a JSON-serializable dict.

    :param detail: The error detail, or ``None``.
    :returns: Dict with non-None fields, or ``None``.
    """
    if detail is None:
        return None
    result: dict[str, Any] = {}
    if detail.provider is not None:
        result["provider"] = detail.provider
    if detail.status_code is not None:
        result["status_code"] = detail.status_code
    if detail.response_body is not None:
        result["response_body"] = detail.response_body
    # Empty dict → None to keep SSE JSON payload clean.
    return result or None


def execute_with_retry(
    call_fn: Callable[[], T],
    retry_config: RetryConfig,
    on_retry: Callable[[dict[str, Any]], None],
) -> T:
    """
    Execute ``call_fn`` with retry on transient failures.

    Called *inside* a ``@step`` boundary so retries don't cause
    duplicate DBOS checkpoints. Emits ``response.retry`` SSE events
    via ``on_retry`` before each backoff sleep.

    :param call_fn: Zero-argument callable that performs the LLM
        call. Raises httpx exceptions on failure.
    :param retry_config: Retry policy (max_attempts, backoff, etc.)
        from the agent's LLM config.
    :param on_retry: Callback to emit a ``response.retry`` SSE event.
        Called with the event dict before sleeping.
    :returns: The successful result from ``call_fn``.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retry attempts are exhausted.
    """
    last_error: RetryableLLMError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return call_fn()
        except Exception as exc:
            classified = classify_llm_error(exc, retry_config.status_codes)
            if isinstance(classified, PermanentLLMError):
                raise classified from exc

            last_error = classified
            if attempt + 1 < retry_config.max_attempts:
                _emit_retry_and_sleep(attempt, retry_config, classified, on_retry)

    # All retries exhausted.
    assert last_error is not None
    raise last_error


def _emit_retry_and_sleep(
    attempt: int,
    retry_config: RetryConfig,
    error: RetryableLLMError,
    on_retry: Callable[[dict[str, Any]], None],
) -> None:
    """
    Emit a retry SSE event and sleep for the backoff delay.

    :param attempt: Current zero-based attempt index, e.g. ``0``
        for the first attempt.
    :param retry_config: Retry policy with backoff parameters.
    :param error: The classified retryable error.
    :param on_retry: Callback to emit the ``response.retry``
        SSE event dict.
    """
    delay = compute_backoff_delay(
        attempt_index=attempt,
        backoff_base=retry_config.backoff_base,
        backoff_max=retry_config.backoff_max,
    )
    event: dict[str, Any] = {
        "type": "response.retry",
        "source": "llm",
        "attempt": attempt + 2,  # next attempt number (1-based)
        "max_attempts": retry_config.max_attempts,
        "delay_seconds": round(delay, 2),
        "error": {
            "code": error.code,
            "message": str(error),
            "detail": detail_to_dict(error.detail),
        },
    }
    on_retry(event)
    _logger.info(
        "LLM retry %d/%d after %.1fs: %s",
        attempt + 2,
        retry_config.max_attempts,
        delay,
        error.code,
    )
    time.sleep(delay)


async def execute_with_retry_async(
    call_fn: Callable[[], Awaitable[T]],
    retry_config: RetryConfig,
    on_retry: Callable[[dict[str, Any]], None],
) -> T:
    """
    Async variant of :func:`execute_with_retry`.

    Uses ``asyncio.sleep`` for backoff so the event loop stays
    free during retry waits.

    :param call_fn: Zero-argument async callable that performs
        the LLM call.
    :param retry_config: Retry policy (max_attempts, backoff,
        etc.) from the agent's LLM config.
    :param on_retry: Callback to emit a ``response.retry`` SSE
        event. Called with the event dict before sleeping.
    :returns: The successful result from ``call_fn``.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retry attempts are
        exhausted.
    """
    last_error: RetryableLLMError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return await call_fn()
        except Exception as exc:
            classified = classify_llm_error(
                exc,
                retry_config.status_codes,
            )
            if isinstance(classified, PermanentLLMError):
                raise classified from exc

            last_error = classified
            if attempt + 1 < retry_config.max_attempts:
                await _emit_retry_and_sleep_async(
                    attempt,
                    retry_config,
                    classified,
                    on_retry,
                )

    assert last_error is not None
    raise last_error


async def _emit_retry_and_sleep_async(
    attempt: int,
    retry_config: RetryConfig,
    error: RetryableLLMError,
    on_retry: Callable[[dict[str, Any]], None],
) -> None:
    """
    Async variant of :func:`_emit_retry_and_sleep`.

    :param attempt: Current zero-based attempt index.
    :param retry_config: Retry policy with backoff parameters.
    :param error: The classified retryable error.
    :param on_retry: Callback to emit the ``response.retry``
        SSE event dict.
    """
    delay = compute_backoff_delay(
        attempt_index=attempt,
        backoff_base=retry_config.backoff_base,
        backoff_max=retry_config.backoff_max,
    )
    event: dict[str, Any] = {
        "type": "response.retry",
        "source": "llm",
        # next attempt number (1-based)
        "attempt": attempt + 2,
        "max_attempts": retry_config.max_attempts,
        "delay_seconds": round(delay, 2),
        "error": {
            "code": error.code,
            "message": str(error),
            "detail": detail_to_dict(error.detail),
        },
    }
    on_retry(event)
    _logger.info(
        "LLM retry %d/%d after %.1fs: %s",
        attempt + 2,
        retry_config.max_attempts,
        delay,
        error.code,
    )
    await asyncio.sleep(delay)
