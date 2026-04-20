"""
Abstract base class for policy evaluator instances.

A :class:`Policy` is an instantiated, per-workflow evaluator
derived from a :class:`PolicySpec`. Subclasses implement one
:meth:`evaluate` method that returns a :class:`PolicyResult`;
the engine (in :mod:`agent_plane.runtime.policies.engine`) does
the filter-gate-dispatch-compose orchestration (see
POLICIES.md §4).

The three concrete subclasses live next to this module:

- :class:`agent_plane.policies.label.LabelPolicy`
- :class:`agent_plane.policies.function.FunctionPolicy`
- :class:`agent_plane.policies.prompt.PromptPolicy`

These classes are pure evaluators — they hold no mutable state
across calls, do no DB I/O, and don't know about conversations.
Mutable runtime state (label cache, conversation id,
write-through store) and the composition loop live in
:mod:`agent_plane.runtime.policies`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_plane.policies.types import EvaluationContext, PolicyResult
from agent_plane.spec.types import PolicySpec


class Policy(ABC):
    """
    Per-workflow policy instance.

    Subclasses declare their ``spec`` attribute (subclass of
    :class:`PolicySpec`) and implement :meth:`evaluate`. The
    engine calls :meth:`evaluate` only when the spec's
    selector and condition gates match the current context;
    implementations therefore don't need to re-check those.

    :param spec: The declarative :class:`PolicySpec` (or
        subclass) this policy was built from. Concrete
        subclasses narrow the type.
    """

    spec: PolicySpec

    @abstractmethod
    async def evaluate(
        self,
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> PolicyResult:
        """
        Return this policy's decision for one evaluation.

        :param ctx: Current evaluation context — phase,
            content, resolved tool_name. Immutable; the
            caller built it from whatever local state the
            enforcement site had.
        :param context: Read-only context bundle from the
            engine — labels snapshot, conversation_id, and
            other identity fields policy callables may want
            to inspect. Structured as a plain dict to keep
            the FunctionPolicy callable contract compatible
            with omniagents' signature (see POLICIES.md §9.1).
        :returns: The policy's single-policy
            :class:`PolicyResult` (``deciding_policy`` left
            ``None`` — engine fills it on composed results).
        """
