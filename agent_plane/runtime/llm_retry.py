"""LLM call retry logic with exponential backoff.

Classifies adapter exceptions as retryable or permanent, computes
backoff delays, and provides a retry loop that emits SSE events.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

from agent_plane.llms.errors import LLMErrorDetail, PermanentLLMError, RetryableLLMError
from agent_plane.spec.types import RetryConfig

_logger = logging.getLogger(__name__)

T = TypeVar("T")


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

    :param exc: The HTTP status error from httpx.
    :param retryable_status_codes: Status codes that trigger retry.
    :returns: A :class:`RetryableLLMError` or
        :class:`PermanentLLMError`.
    """
    status = exc.response.status_code
    body = _safe_response_text(exc.response)
    detail = LLMErrorDetail(status_code=status, response_body=body)
    code = str(status)
    message = f"LLM returned HTTP {status}: {body}"

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
