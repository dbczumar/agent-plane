"""
Rate-limit policy factories for the `rate-limited-search`
fixture agent.

Ports omniagents ``examples/search_rate_limit_policy.py`` and
``examples/rate_limit_policy.py``.
"""

from __future__ import annotations

from typing import Any

from agent_plane.spec.types import (
    EvaluationContext,
    PolicyAction,
    PolicyResult,
)


def rate_limit_search(limit: int = 3) -> Any:
    """
    Factory for a stateful web-search rate limiter.

    After ``limit`` free searches, additional calls ASK for
    user approval instead of blocking outright. The classic
    omniagents IFC-ergonomics example.

    :param limit: Free-call budget before ASK kicks in.
    :returns: Evaluator callable with closure state that
        counts invocations.
    """
    calls = 0

    def _eval(ctx: EvaluationContext) -> PolicyResult:
        """
        Evaluator: count invocations, ALLOW up to ``limit``,
        ASK thereafter.
        """
        nonlocal calls
        calls += 1
        if calls <= limit:
            return PolicyResult(action=PolicyAction.ALLOW)
        return PolicyResult(
            action=PolicyAction.ASK,
            reason=(f"Free search budget ({limit}) exhausted; this search is call #{calls}."),
        )

    return _eval


def max_tool_calls_per_turn(limit: int = 15) -> Any:
    """
    Factory for a per-workflow total-tool-call cap.

    Ports omniagents ``max_tool_calls_per_turn`` — used as a
    safety-net policy alongside more targeted guards.

    :param limit: Total calls before DENY.
    :returns: Evaluator callable with closure state.
    """
    calls = 0

    def _eval(ctx: EvaluationContext) -> PolicyResult:
        """Deny after ``limit`` total tool calls."""
        nonlocal calls
        calls += 1
        if calls > limit:
            return PolicyResult(
                action=PolicyAction.DENY,
                reason=f"Tool-call budget ({limit}) exceeded.",
            )
        return PolicyResult(action=PolicyAction.ALLOW)

    return _eval
