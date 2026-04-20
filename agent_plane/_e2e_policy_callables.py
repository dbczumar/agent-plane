"""
E2E-test-only policy callables.

Lives under the ``agent_plane`` package so the server
subprocess (which imports from agent_plane, not tests/) can
resolve the dotted path. The module itself has no production
value — it exists solely so
``tests/_fixtures/agents/e2e-policy-gate/config.yaml`` can
reference a callable the live server process can import.

Would not exist in a deployment where agent authors ship
their own policy callables via pip-installed packages.
"""

from __future__ import annotations

from agent_plane.spec.types import (
    EvaluationContext,
    PolicyAction,
    PolicyResult,
)

# Deterministic sentinel — arbitrary string unlikely to
# appear in natural user messages, so the e2e test can
# reliably flip the DENY path on / off.
_SENTINEL = "BLOCK_THIS_TOKEN"


def block_on_sentinel(ctx: EvaluationContext) -> PolicyResult:
    """
    DENY any INPUT containing the sentinel token.

    :param ctx: Current evaluation context. On INPUT phase,
        ``ctx.content`` is the user message text (str).
    :returns: :class:`PolicyResult` — DENY if the sentinel
        appears in the text, ALLOW otherwise.
    """
    content = ctx.content
    if isinstance(content, str) and _SENTINEL in content:
        return PolicyResult(
            action=PolicyAction.DENY,
            reason=f"contains reserved token {_SENTINEL!r}",
        )
    return PolicyResult(action=PolicyAction.ALLOW)
