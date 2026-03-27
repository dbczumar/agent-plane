"""Tests for tool retry logic: timeout resolution, retry resolution, and execution."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.runtime.tool_retry import (
    call_tool_with_timeout,
    execute_tool_with_retry,
    resolve_tool_retry,
    resolve_tool_timeout,
)
from agent_plane.spec.types import RetryConfig, ToolsConfig


@pytest.fixture()
def global_tools_config() -> ToolsConfig:
    """
    A ToolsConfig with explicit global defaults for timeout and retry.

    :returns: ToolsConfig with timeout=60, retry max_attempts=2.
    """
    return ToolsConfig(
        timeout=60,
        retry=RetryConfig(max_attempts=2, backoff_base=1.0, backoff_max=10.0),
    )


@pytest.fixture()
def captured_events() -> list[dict[str, Any]]:
    """
    Mutable list that accumulates on_event dicts during test execution.

    :returns: Empty list that tests inspect after calling execute_tool_with_retry.
    """
    return []


@pytest.fixture()
def on_event(captured_events: list[dict[str, Any]]) -> MagicMock:
    """
    An on_event callback that records every dict it receives.

    :returns: A MagicMock whose side_effect appends to captured_events.
    """
    mock = MagicMock(side_effect=lambda evt: captured_events.append(evt))
    return mock


# -- resolve_tool_timeout --


def test_resolve_tool_timeout_uses_per_tool_override(
    global_tools_config: ToolsConfig,
) -> None:
    """Per-tool timeout takes precedence over the global default."""
    result = resolve_tool_timeout(
        tool_name="my_tool",
        tools_config=global_tools_config,
        per_tool_timeout=120,
    )
    # Per-tool override (120) must win over global (60).
    # Failure means the function ignores the per-tool value.
    assert result == 120


def test_resolve_tool_timeout_falls_back_to_global(
    global_tools_config: ToolsConfig,
) -> None:
    """When per-tool timeout is None, the global timeout is returned."""
    result = resolve_tool_timeout(
        tool_name="my_tool",
        tools_config=global_tools_config,
        per_tool_timeout=None,
    )
    # Should fall back to global_tools_config.timeout (60).
    # Failure means the function does not honour the global default.
    assert result == 60


# -- resolve_tool_retry --


def test_resolve_tool_retry_uses_per_tool_override(
    global_tools_config: ToolsConfig,
) -> None:
    """Per-tool retry config takes precedence over the global default."""
    per_tool = RetryConfig(max_attempts=5, backoff_base=3.0, backoff_max=60.0)
    result = resolve_tool_retry(
        tool_name="my_tool",
        tools_config=global_tools_config,
        per_tool_retry=per_tool,
    )
    # The returned config must be the per-tool override, not the global.
    # Failure means the function ignores the per-tool retry config.
    assert result is per_tool
    assert result.max_attempts == 5


def test_resolve_tool_retry_falls_back_to_global(
    global_tools_config: ToolsConfig,
) -> None:
    """When per-tool retry is None, the global retry config is returned."""
    result = resolve_tool_retry(
        tool_name="my_tool",
        tools_config=global_tools_config,
        per_tool_retry=None,
    )
    # Should fall back to global_tools_config.retry.
    # Failure means the function does not honour the global default.
    assert result is global_tools_config.retry
    assert result.max_attempts == 2


# -- call_tool_with_timeout --


def test_call_tool_with_timeout_succeeds() -> None:
    """A fast tool returns its result within the deadline."""
    result = call_tool_with_timeout(lambda: "ok", timeout=5)
    # The tool finished instantly so the result must be "ok".
    # Failure means the function lost the return value or raised unexpectedly.
    assert result == "ok"


def test_call_tool_with_timeout_raises_on_slow_tool() -> None:
    """A tool that exceeds its deadline raises TimeoutError."""

    def slow_tool() -> str:
        """Simulates a tool that takes longer than the allowed timeout."""
        time.sleep(2)
        return "late"

    # timeout=1 is shorter than the 2s sleep inside slow_tool.
    # Failure means the timeout enforcement is broken.
    with pytest.raises(TimeoutError, match="timed out"):
        call_tool_with_timeout(slow_tool, timeout=1)


# -- execute_tool_with_retry --


def test_execute_tool_with_retry_success_first_attempt(
    on_event: MagicMock,
    captured_events: list[dict[str, Any]],
) -> None:
    """When the tool succeeds on the first attempt, no retry events are emitted."""
    retry_config = RetryConfig(max_attempts=3, backoff_base=1.0, backoff_max=10.0)
    result = execute_tool_with_retry(
        tool_name="fast_tool",
        call_fn=lambda: "result",
        timeout=5,
        retry_config=retry_config,
        on_event=on_event,
    )
    # Tool succeeded immediately so we get the result back.
    # Failure means the function swallowed the result or retried unnecessarily.
    assert result == "result"
    # No retry or error events should have been emitted on a first-attempt success.
    # Failure means the function emits spurious events.
    retry_events = [e for e in captured_events if e["type"] == "response.retry"]
    assert len(retry_events) == 0


def test_execute_tool_with_retry_retries_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    on_event: MagicMock,
    captured_events: list[dict[str, Any]],
) -> None:
    """A timeout on the first attempt triggers a retry that then succeeds."""
    # Patch time.sleep inside the retry module to avoid real delays.
    monkeypatch.setattr("agent_plane.runtime.tool_retry.time.sleep", lambda _: None)

    call_count = 0

    def flaky_tool() -> str:
        """
        Fails with TimeoutError on the first call, succeeds on the second.
        """
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("Tool execution timed out after 5s")
        return "ok"

    retry_config = RetryConfig(max_attempts=3, backoff_base=1.0, backoff_max=10.0)

    # Patch call_tool_with_timeout so we control failures without real threads.
    monkeypatch.setattr(
        "agent_plane.runtime.tool_retry.call_tool_with_timeout",
        lambda call_fn, timeout: flaky_tool(),
    )

    result = execute_tool_with_retry(
        tool_name="flaky_tool",
        call_fn=lambda: "unused",
        timeout=5,
        retry_config=retry_config,
        on_event=on_event,
    )
    # Second attempt succeeded so the result must be "ok".
    # Failure means the retry loop did not re-invoke the tool.
    assert result == "ok"

    # Exactly one retry event should have been emitted (for the first failure).
    # Failure means the function either skipped the retry event or emitted too many.
    retry_events = [e for e in captured_events if e["type"] == "response.retry"]
    assert len(retry_events) == 1
    assert retry_events[0]["tool_name"] == "flaky_tool"


def test_execute_tool_with_retry_returns_error_string_on_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    on_event: MagicMock,
    captured_events: list[dict[str, Any]],
) -> None:
    """When all attempts time out, an error string is returned (not raised)."""
    # Patch time.sleep inside the retry module to avoid real delays.
    monkeypatch.setattr("agent_plane.runtime.tool_retry.time.sleep", lambda _: None)

    retry_config = RetryConfig(max_attempts=2, backoff_base=1.0, backoff_max=10.0)

    # Patch call_tool_with_timeout so every call raises TimeoutError.
    monkeypatch.setattr(
        "agent_plane.runtime.tool_retry.call_tool_with_timeout",
        lambda call_fn, timeout: (_ for _ in ()).throw(
            TimeoutError("Tool execution timed out after 5s")
        ),
    )

    result = execute_tool_with_retry(
        tool_name="stuck_tool",
        call_fn=lambda: "unused",
        timeout=5,
        retry_config=retry_config,
        on_event=on_event,
    )
    # The function must return an error string, not raise an exception.
    # Failure means the function lets TimeoutError propagate to the caller.
    assert isinstance(result, str)
    assert "Error" in result
    assert "2 attempts" in result

    # A response.error event must have been emitted for the terminal failure.
    # Failure means the caller would not be notified of the exhaustion.
    error_events = [e for e in captured_events if e["type"] == "response.error"]
    assert len(error_events) == 1
    assert error_events[0]["tool_name"] == "stuck_tool"
