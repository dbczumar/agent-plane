"""
``_enforce_policy`` — thin call-site wrapper for the engine.

Called from the four enforcement sites (input, tool_call,
tool_result, output) in the workflow. Per POLICIES.md §5, this
helper does NOT apply label writes — the engine handles ALLOW
and DENY paths internally, and :func:`_await_policy_approval`
(Phase 8) applies ASK writes on approval. Wrapping the
evaluate call in a named helper keeps the enforcement-site
branching tight and gives observability a single symbol to
hook on later.
"""

from __future__ import annotations

from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.spec.types import EvaluationContext, PolicyResult


async def _enforce_policy(
    engine: PolicyEngine,
    ctx: EvaluationContext,
) -> PolicyResult:
    """
    Evaluate the engine for one phase and return the composed
    decision.

    :param engine: The per-workflow :class:`PolicyEngine`.
    :param ctx: The evaluation context the caller assembled
        (phase, content, resolved tool_name).
    :returns: The composed :class:`PolicyResult` — callers
        branch on :attr:`PolicyResult.action` to handle
        ALLOW / ASK / DENY at the enforcement site.
    """
    return await engine.evaluate(ctx)
