"""Tool call retry logic with exponential backoff and timeout.

Wraps tool invocations with configurable timeout enforcement and
retry on transient failures. Errors are returned as strings to the
LLM (not raised) so the agent can decide how to proceed.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from agent_plane.spec.types import RetryConfig, ToolsConfig

_logger = logging.getLogger(__name__)


def resolve_tool_timeout(
    tool_name: str,
    tools_config: ToolsConfig,
    per_tool_timeout: int | None,
) -> int:
    """
    Resolve the effective timeout for a tool call.

    :param tool_name: Tool name for logging, e.g.
        ``"github.list_issues"``.
    :param tools_config: The agent's global tools config.
    :param per_tool_timeout: Per-tool timeout override. ``None``
        inherits ``tools_config.timeout``.
    :returns: Effective timeout in seconds.
    """
    if per_tool_timeout is not None:
        return per_tool_timeout
    return tools_config.timeout


def resolve_tool_retry(
    tool_name: str,
    tools_config: ToolsConfig,
    per_tool_retry: RetryConfig | None,
) -> RetryConfig:
    """
    Resolve the effective retry config for a tool call.

    :param tool_name: Tool name for logging, e.g.
        ``"github.list_issues"``.
    :param tools_config: The agent's global tools config.
    :param per_tool_retry: Per-tool retry override. ``None``
        inherits ``tools_config.retry``.
    :returns: Effective retry config.
    """
    if per_tool_retry is not None:
        return per_tool_retry
    return tools_config.retry


def call_tool_with_timeout(
    call_fn: Callable[[], str],
    timeout: int,
) -> str:
    """
    Execute a tool call with a wall-clock timeout.

    Uses a thread pool to enforce the timeout. If the tool
    exceeds the deadline, raises ``TimeoutError``.

    :param call_fn: Zero-argument callable that executes the
        tool and returns a string result.
    :param timeout: Timeout in seconds, e.g. ``60``.
    :returns: The tool's string result.
    :raises TimeoutError: If the tool does not complete within
        the timeout.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call_fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Tool execution timed out after {timeout}s"
            ) from None


def execute_tool_with_retry(
    tool_name: str,
    call_fn: Callable[[], str],
    timeout: int,
    retry_config: RetryConfig,
    on_event: Callable[[dict[str, Any]], None],
) -> str:
    """
    Execute a tool call with timeout and retry.

    On transient failure (timeout), retries with exponential
    backoff. When all retries are exhausted, returns an error
    string (not raised) so the LLM can decide how to proceed.

    :param tool_name: The tool name for error messages and SSE
        events, e.g. ``"github.list_issues"``.
    :param call_fn: Zero-argument callable that executes the
        tool and returns a string result.
    :param timeout: Per-call timeout in seconds, e.g. ``60``.
    :param retry_config: Retry policy for this tool.
    :param on_event: Callback to emit SSE events (retry and
        error). Called with the event dict.
    :returns: The tool's string result, or an error string if
        all retries are exhausted.
    """
    last_error: str | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            return call_tool_with_timeout(call_fn, timeout)
        except TimeoutError as exc:
            last_error = str(exc)
            if attempt + 1 < retry_config.max_attempts:
                _emit_tool_retry(
                    tool_name, attempt, retry_config,
                    "timeout", last_error, on_event,
                )
        except Exception as exc:
            last_error = f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}"
            # Non-timeout exceptions are not retried.
            break

    _emit_tool_error(
        tool_name, retry_config.max_attempts, last_error, on_event
    )
    return f"Error: {last_error} ({retry_config.max_attempts} attempts)"


def _emit_tool_retry(
    tool_name: str,
    attempt: int,
    retry_config: RetryConfig,
    error_code: str,
    error_message: str,
    on_event: Callable[[dict[str, Any]], None],
) -> None:
    """
    Emit a ``response.retry`` SSE event for a tool and sleep.

    :param tool_name: The tool name, e.g. ``"github.list_issues"``.
    :param attempt: Current zero-based attempt index.
    :param retry_config: Retry policy with backoff parameters.
    :param error_code: Error code string, e.g. ``"timeout"``.
    :param error_message: Human-readable error description.
    :param on_event: Callback to emit the SSE event.
    """
    delay = min(
        retry_config.backoff_base**attempt,
        retry_config.backoff_max,
    )
    delay *= random.uniform(0.5, 1.0)
    event: dict[str, Any] = {
        "type": "response.retry",
        "source": "tool",
        "tool_name": tool_name,
        "attempt": attempt + 2,
        "max_attempts": retry_config.max_attempts,
        "delay_seconds": round(delay, 2),
        "error": {
            "code": error_code,
            "message": error_message,
            "detail": None,
        },
    }
    on_event(event)
    _logger.info(
        "Tool %r retry %d/%d after %.1fs: %s",
        tool_name, attempt + 2, retry_config.max_attempts,
        delay, error_code,
    )
    time.sleep(delay)


def _emit_tool_error(
    tool_name: str,
    max_attempts: int,
    error_message: str | None,
    on_event: Callable[[dict[str, Any]], None],
) -> None:
    """
    Emit a ``response.error`` SSE event for a terminal tool failure.

    :param tool_name: The tool name, e.g. ``"github.list_issues"``.
    :param max_attempts: Total attempts configured.
    :param error_message: Human-readable error description.
    :param on_event: Callback to emit the SSE event.
    """
    # Defensive fallback — error_message should always be set by the
    # retry loop, but "Unknown error" is safe if somehow None.
    msg = error_message or "Unknown error"
    on_event({
        "type": "response.error",
        "source": "tool",
        "tool_name": tool_name,
        "error": {
            "code": "timeout",
            "message": f"{msg} ({max_attempts} attempts exhausted)",
            "detail": None,
        },
    })
