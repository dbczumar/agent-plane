"""
Policy building blocks — pure evaluators, no runtime state.

This package holds the three pieces an agent author (or the
parser) reaches for when declaring / implementing a policy:

- :class:`EvaluationContext`, :class:`PolicyResult`,
  :class:`ApprovalRequest` — the data shapes that cross the
  evaluate boundary (see :mod:`agent_plane.policies.types`).
- :class:`Policy` ABC (:mod:`agent_plane.policies.base`) and
  the three concrete subclasses: :class:`LabelPolicy`,
  :class:`FunctionPolicy`, :class:`PromptPolicy`.

The subclasses are pure in the important sense: they own no
mutable state across calls, do no DB I/O, and don't know about
conversations. State (label cache, conversation id,
write-through store) and orchestration (composition loop, ASK
parking, fail-closed) live in :mod:`agent_plane.runtime.policies`.

Agent-author callables should import :class:`EvaluationContext`
and :class:`PolicyResult` from here, not from
``agent_plane.spec.types`` — those are runtime evaluation
artifacts, not declarations that appear in a spec.
"""

from __future__ import annotations

from agent_plane.policies.base import Policy
from agent_plane.policies.function import (
    FunctionPolicy,
    resolve_function_policy,
)
from agent_plane.policies.label import LabelPolicy
from agent_plane.policies.prompt import (
    PromptPolicy,
    resolve_prompt_policy,
)
from agent_plane.policies.types import (
    ApprovalRequest,
    EvaluationContext,
    PolicyResult,
)

__all__ = [
    "ApprovalRequest",
    "EvaluationContext",
    "FunctionPolicy",
    "LabelPolicy",
    "Policy",
    "PolicyResult",
    "PromptPolicy",
    "resolve_function_policy",
    "resolve_prompt_policy",
]
