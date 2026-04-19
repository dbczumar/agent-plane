"""
Callables for the ``combined-policies`` fixture.
"""

from __future__ import annotations

from agent_plane.spec.types import (
    EvaluationContext,
    PolicyAction,
    PolicyResult,
)


def observe_all(ctx: EvaluationContext) -> PolicyResult:
    """
    Always-ALLOW observer.

    Pure classifier — records nothing, never blocks. Paired
    with a ``[allow]``-only action whitelist so a future
    exception-throwing bug (e.g. if someone swaps this for
    a logging stub that raises on disk full) would trigger
    the classifier-only carve-out and still ALLOW, honoring
    the author's declared intent.

    :param ctx: Evaluation context (unused — this function
        never blocks regardless of content).
    :returns: :class:`PolicyResult` with ``ALLOW``.
    """
    del ctx
    return PolicyResult(action=PolicyAction.ALLOW)
