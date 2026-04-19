"""
Runtime policies — the PolicyEngine and its helpers.

Phase 2 scope: engine skeleton + label seeding + builder.
No concrete policy subclasses yet — those land per-type in
later phases:
- Phase 3: ``LabelPolicy`` (POLICIES.md §9.3)
- Phase 4: ``FunctionPolicy`` (POLICIES.md §9.1)
- Phase 7: ``PromptPolicy`` (POLICIES.md §9.2)

The public API for callers (workflow, executor hooks) is
``PolicyEngine`` + ``build_policy_engine``. Everything else is
internal and may change between phases.
"""

from __future__ import annotations

from agent_plane.runtime.policies.builder import build_policy_engine
from agent_plane.runtime.policies.engine import PolicyEngine

__all__ = [
    "PolicyEngine",
    "build_policy_engine",
]
