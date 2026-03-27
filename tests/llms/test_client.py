"""
Tests for the standalone LLM client retry logic (llms/client.py).

Covers the public ``Client().responses.create(retry=...)`` interface,
verifying that transient failures are retried with backoff and permanent
failures surface immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from llms.client import Client
from llms.errors import LLMErrorDetail, PermanentLLMError, RetryableLLMError
from llms.types import (
    MessageOutput,
    OutputText,
    Response,
    RetryConfig,
)

# ── Helpers ──────────────────────────────────────────────────


@dataclass
class _SleepTracker:
    """
    Tracks calls to ``time.sleep`` during retry backoff.

    :param calls: List of sleep durations passed to each call.
    """

    calls: list[float]


def _make_response() -> Response:
    """
    Build a minimal ``Response`` for successful-call assertions.

    :returns: A ``Response`` with a single text output.
    """
    return Response(
        output=[MessageOutput(content=[OutputText(text="Hello")])],
        model="test-model",
    )


def _patch_client_deps(
    monkeypatch: pytest.MonkeyPatch,
    mock_adapter: MagicMock,
) -> _SleepTracker:
    """
    Patch all external dependencies of ``Client().responses.create()``
    so that calls route through ``mock_adapter.chat_completions``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param mock_adapter: A ``MagicMock`` whose ``.chat_completions``
        attribute controls call success/failure.
    :returns: A :class:`_SleepTracker` recording backoff sleep calls.
    """
    # Route model parsing to a fake routed model
    routed = MagicMock(provider="test", model="test-model")
    monkeypatch.setattr(
        "llms.client.parse_model_string",
        lambda model: routed,
    )

    # Return the mock adapter (not OpenAIAdapter, so we hit
    # the chat_completions path instead of responses_create)
    monkeypatch.setattr(
        "llms.client.get_adapter",
        lambda provider: mock_adapter,
    )

    # Stub the responses-to-chat conversion helpers
    monkeypatch.setattr(
        "llms.client.responses_input_to_chat_messages",
        lambda input, instructions: [{"role": "user", "content": "test"}],
    )
    monkeypatch.setattr(
        "llms.client.chat_response_to_response",
        lambda result: _make_response(),
    )

    # Capture sleep calls instead of actually sleeping
    tracker = _SleepTracker(calls=[])
    monkeypatch.setattr(
        "llms.client.time.sleep",
        lambda duration: tracker.calls.append(duration),
    )

    return tracker


def _default_create_kwargs() -> dict[str, Any]:
    """
    Minimal kwargs for ``Client().responses.create()``.

    :returns: Dict with required ``input`` and ``model`` keys.
    """
    return {
        "input": [{"role": "user", "content": "hi"}],
        "model": "test/test-model",
    }


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def retry_config() -> RetryConfig:
    """
    A retry config with 3 attempts and fast backoff for testing.
    """
    return RetryConfig(
        max_attempts=3,
        backoff_base=2.0,
        backoff_max=30.0,
    )


# ── Tests ────────────────────────────────────────────────────


def test_create_without_retry_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When ``retry=None``, the call succeeds normally without retry
    wrapping.
    """
    mock_adapter = MagicMock()
    mock_adapter.chat_completions.return_value = {"id": "test"}
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = Client().responses.create(**_default_create_kwargs())

    # The response should contain the expected text from the mock
    # conversion; failure means the non-retry path is broken.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # No backoff sleeps should occur when retry is disabled.
    assert tracker.calls == []

    # Adapter should be called exactly once.
    assert mock_adapter.chat_completions.call_count == 1


def test_create_with_retry_success_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    With retry config, first-attempt success works and no backoff
    sleep occurs.
    """
    mock_adapter = MagicMock()
    mock_adapter.chat_completions.return_value = {"id": "test"}
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = Client().responses.create(
        **_default_create_kwargs(),
        retry=retry_config,
    )

    # Successful first attempt returns the converted response.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # No backoff sleep when the first attempt succeeds.
    assert tracker.calls == []

    # Only one call to the adapter — no retries needed.
    assert mock_adapter.chat_completions.call_count == 1


def test_create_with_retry_timeout_then_success(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    Timeout on first attempt triggers a retry; second attempt succeeds.
    """
    mock_adapter = MagicMock()
    # First call times out, second call succeeds
    mock_adapter.chat_completions.side_effect = [
        httpx.TimeoutException("timeout"),
        {"id": "test"},
    ]
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = Client().responses.create(
        **_default_create_kwargs(),
        retry=retry_config,
    )

    # The retry should recover and return a valid response.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # Exactly one backoff sleep between the failed first attempt and
    # the successful second attempt.
    assert len(tracker.calls) == 1

    # Two adapter calls total: one timeout, one success.
    assert mock_adapter.chat_completions.call_count == 2


def test_create_with_retry_http_429_then_success(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    Rate limit (429) on first attempt triggers retry; second attempt
    succeeds.
    """
    mock_adapter = MagicMock()
    http_429 = httpx.HTTPStatusError(
        "rate limited",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(429),
    )
    mock_adapter.chat_completions.side_effect = [
        http_429,
        {"id": "test"},
    ]
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = Client().responses.create(
        **_default_create_kwargs(),
        retry=retry_config,
    )

    # Recovery after 429 should produce a valid response.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # One backoff sleep between the 429 and the successful retry.
    assert len(tracker.calls) == 1

    # Two adapter calls: one 429, one success.
    assert mock_adapter.chat_completions.call_count == 2


def test_create_with_retry_permanent_error_no_retry(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    HTTP 401 raises PermanentLLMError immediately with no retry.
    """
    mock_adapter = MagicMock()
    http_401 = httpx.HTTPStatusError(
        "unauthorized",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(401),
    )
    mock_adapter.chat_completions.side_effect = http_401
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(PermanentLLMError) as exc_info:
        Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Error code should reflect the HTTP status; failure means
    # _classify_error mapped to the wrong category.
    assert exc_info.value.code == "401"

    # Detail should carry the status code for diagnostics.
    assert exc_info.value.detail is not None
    assert exc_info.value.detail.status_code == 401

    # No backoff sleeps — permanent errors abort immediately.
    assert tracker.calls == []

    # Only one adapter call — no retry attempted.
    assert mock_adapter.chat_completions.call_count == 1


def test_create_with_retry_exhausted_raises(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    All attempts timeout, raising RetryableLLMError after exhaustion.
    """
    mock_adapter = MagicMock()
    # All 3 attempts time out
    mock_adapter.chat_completions.side_effect = httpx.TimeoutException("timeout")
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(RetryableLLMError) as exc_info:
        Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Code should be "timeout" since all failures were timeouts.
    assert exc_info.value.code == "timeout"

    # Two backoff sleeps (between attempt 1→2 and 2→3; no sleep
    # after the final failed attempt).
    assert len(tracker.calls) == 2

    # All 3 attempts should have been made before giving up.
    assert mock_adapter.chat_completions.call_count == 3


def test_create_with_retry_already_classified_reraise(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    If the adapter raises PermanentLLMError directly, it is re-raised
    without reclassification.
    """
    mock_adapter = MagicMock()
    # Adapter raises an already-classified error
    original_error = PermanentLLMError(
        "auth failed",
        code="auth_error",
        detail=LLMErrorDetail(provider="test"),
    )
    mock_adapter.chat_completions.side_effect = original_error
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(PermanentLLMError) as exc_info:
        Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # The exact same error object should be re-raised, not wrapped
    # in a new PermanentLLMError. Failure means _execute_with_retry
    # reclassified an already-classified error.
    assert exc_info.value is original_error
    assert exc_info.value.code == "auth_error"

    # No backoff sleeps — already-classified errors bypass retry.
    assert tracker.calls == []

    # Only one adapter call — no retry for pre-classified errors.
    assert mock_adapter.chat_completions.call_count == 1


def test_create_with_retry_already_classified_retryable_reraise(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    If the adapter raises RetryableLLMError directly, it is re-raised
    without reclassification or further retries.
    """
    mock_adapter = MagicMock()
    original_error = RetryableLLMError(
        "rate limited upstream",
        code="429",
        detail=LLMErrorDetail(status_code=429),
    )
    mock_adapter.chat_completions.side_effect = original_error
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(RetryableLLMError) as exc_info:
        Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Same error object re-raised, not reclassified or wrapped.
    assert exc_info.value is original_error

    # No backoff — pre-classified RetryableLLMError is immediately
    # re-raised by the `except (PermanentLLMError, RetryableLLMError)`
    # clause in _execute_with_retry.
    assert tracker.calls == []

    # Only one call — no further retries for pre-classified errors.
    assert mock_adapter.chat_completions.call_count == 1


def test_create_with_retry_connection_error_is_permanent(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
) -> None:
    """
    Generic ``Exception`` (e.g. connection error) is classified as
    PermanentLLMError with no retry.
    """
    mock_adapter = MagicMock()
    mock_adapter.chat_completions.side_effect = ConnectionError("connection refused")
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(PermanentLLMError) as exc_info:
        Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Generic exceptions map to "connection_error" code; failure
    # means _classify_error treated it as retryable.
    assert exc_info.value.code == "connection_error"
    assert "connection refused" in str(exc_info.value)

    # No backoff sleeps — permanent errors don't retry.
    assert tracker.calls == []

    # Only one adapter call.
    assert mock_adapter.chat_completions.call_count == 1


@pytest.mark.parametrize(
    ("status_code", "expected_error_type"),
    [
        # 429 is in default retryable status_codes — should retry
        (429, RetryableLLMError),
        # 500 is in default retryable status_codes — should retry
        (500, RetryableLLMError),
        # 502 is in default retryable status_codes — should retry
        (502, RetryableLLMError),
        # 503 is in default retryable status_codes — should retry
        (503, RetryableLLMError),
        # 400 is NOT retryable — should be permanent
        (400, PermanentLLMError),
        # 401 is NOT retryable — should be permanent
        (401, PermanentLLMError),
        # 403 is NOT retryable — should be permanent
        (403, PermanentLLMError),
        # 404 is NOT retryable — should be permanent
        (404, PermanentLLMError),
    ],
    ids=[
        "429-retryable",
        "500-retryable",
        "502-retryable",
        "503-retryable",
        "400-permanent",
        "401-permanent",
        "403-permanent",
        "404-permanent",
    ],
)
def test_create_with_retry_http_status_classification(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryConfig,
    status_code: int,
    expected_error_type: type,
) -> None:
    """
    HTTP status codes are classified correctly as retryable or permanent
    based on the retry config's ``status_codes`` list.
    """
    mock_adapter = MagicMock()
    http_error = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(status_code),
    )
    # Always fail so we can check classification
    mock_adapter.chat_completions.side_effect = http_error
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(expected_error_type) as exc_info:
        Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # The error code should match the HTTP status string.
    assert exc_info.value.code == str(status_code)

    # Detail must carry the status code for downstream diagnostics.
    assert exc_info.value.detail is not None
    assert exc_info.value.detail.status_code == status_code
