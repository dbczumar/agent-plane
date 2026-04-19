"""
Python policies for the `policies-demo` fixture agent.

Ports relevant callables from omniagents
`examples/tool_functions.py` into a location importable by the
agent_plane parser (dotted path `tests._fixtures.agents.*`).
"""

from __future__ import annotations

import json

from agent_plane.spec.types import (
    EvaluationContext,
    PolicyAction,
    PolicyResult,
)

# Long-sleep threshold. Sleep calls over this many seconds are
# blocked. Chosen small enough that trivial test args (like 8 s)
# trip the guard; large enough that the canonical "sleep 2" ALLOW
# path keeps working.
_MAX_SLEEP_SECONDS = 5


def block_long_sleep(ctx: EvaluationContext) -> PolicyResult:
    """
    Ported from omniagents ``block_long_sleep``.

    Inspects the tool_call arguments; blocks when the requested
    sleep duration exceeds :data:`_MAX_SLEEP_SECONDS`.

    :param ctx: The current evaluation context. On TOOL_CALL
        phase, ``ctx.content`` is a dict ``{"tool": name,
        "args": <args>}``.
    :returns: :class:`PolicyResult` — DENY when args ask for a
        long sleep, ALLOW otherwise.
    """
    args = _extract_args(ctx.content)
    seconds = args.get("seconds")
    try:
        secs_num = float(seconds) if seconds is not None else 0.0
    except (TypeError, ValueError):
        # Malformed args — let the tool handler produce its
        # own error. Policy does not gate on argument type.
        secs_num = 0.0
    if secs_num > _MAX_SLEEP_SECONDS:
        return PolicyResult(
            action=PolicyAction.DENY,
            reason=(
                f"Requested sleep {secs_num}s exceeds the {_MAX_SLEEP_SECONDS}s policy limit."
            ),
        )
    return PolicyResult(action=PolicyAction.ALLOW)


def _extract_args(content: object) -> dict[str, object]:
    """
    Pull the tool-call argument mapping out of ``ctx.content``.

    The workflow builds TOOL_CALL contexts as
    ``{"tool": name, "args": <args>}``. Args may be either
    already-parsed dicts or JSON-encoded strings (agent_plane's
    ToolManager passes strings before JSON-decode for some
    paths).

    :param content: Whatever was on ``ctx.content``.
    :returns: Argument dict. Empty dict when the content does
        not conform to the expected shape — safer than raising
        from a policy callable.
    """
    if not isinstance(content, dict):
        return {}
    args = content.get("args")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
