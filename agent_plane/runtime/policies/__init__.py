"""
Runtime policy orchestration — engine, builder, enforcement,
ASK approval.

Pure evaluators (:class:`Policy` ABC + the concrete
``LabelPolicy`` / ``FunctionPolicy`` / ``PromptPolicy``
subclasses) live in :mod:`agent_plane.policies`. This package
holds the code that actually runs during a workflow: the
composition loop, label write-through, approval parking.

The public API for callers (workflow, executor hooks) is
:class:`PolicyEngine` + :func:`build_policy_engine` +
:func:`_enforce_policy` + :func:`_await_policy_approval`.
"""

from __future__ import annotations

from agent_plane.runtime.policies.approval import _await_policy_approval
from agent_plane.runtime.policies.builder import build_policy_engine
from agent_plane.runtime.policies.enforcement import _enforce_policy
from agent_plane.runtime.policies.engine import PolicyEngine

__all__ = [
    "PolicyEngine",
    "_await_policy_approval",
    "_enforce_policy",
    "build_policy_engine",
]
