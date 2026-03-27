"""LLM client error types for retry classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMErrorDetail:
    """
    Structured detail about an LLM call failure.

    :param provider: Provider name, e.g. ``"openai"``, ``"anthropic"``.
        ``None`` when the provider cannot be determined.
    :param status_code: HTTP status code from the provider, e.g.
        ``429``. ``None`` for non-HTTP errors (timeouts, connection
        errors).
    :param response_body: Raw response body from the provider, e.g.
        ``'{"error": {"message": "Rate limit"}}'``. ``None`` when
        no response body is available.
    """

    provider: str | None = None
    status_code: int | None = None
    response_body: str | None = None


class RetryableLLMError(Exception):
    """
    An LLM call failure that may be retried.

    Raised by the retry loop when the adapter throws a retryable
    exception (timeout or configured HTTP status code). Carries
    a string ``code`` for SSE events and structured ``detail``
    for diagnostics.

    :param message: Human-readable error description, e.g.
        ``"OpenAI rate limit exceeded"``.
    :param code: Error code string for SSE events, e.g.
        ``"429"``, ``"timeout"``, ``"connection_error"``.
    :param detail: Structured provider-specific detail.
        ``None`` when no additional detail is available.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        detail: LLMErrorDetail | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class PermanentLLMError(Exception):
    """
    An LLM call failure that should NOT be retried.

    Raised when the adapter throws a non-retryable exception
    (auth failure, bad request, connection refused).

    :param message: Human-readable error description.
    :param code: Error code string for SSE events, e.g.
        ``"401"``, ``"connection_error"``.
    :param detail: Structured provider-specific detail.
        ``None`` when no additional detail is available.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        detail: LLMErrorDetail | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail
