"""
Unit tests for :func:`_resolve_workdir_for_spec`.

This helper fixes a pre-existing bug where
:class:`ToolManager` resolved sub-agent local tool paths
against the ROOT agent's workdir. The parser stores
``LocalToolInfo.path`` relative to the agent that owns the
tool, so the runtime must join it with that agent's own
workdir — not the root's.

The bug was latent (no production agent shipped with
sub-agents that declare local Python tools), but the feature
is documented and the parser supports it, so the runtime
must too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_plane.runtime.workflow import _resolve_workdir_for_spec
from agent_plane.spec.types import AgentSpec


def _spec(name: str, sub_agents: list[AgentSpec] | None = None) -> AgentSpec:
    """Build a minimal AgentSpec stub for workdir-tree tests."""
    return AgentSpec(
        spec_version=1,
        name=name,
        sub_agents=sub_agents or [],
    )


def test_none_name_returns_root_workdir() -> None:
    """
    A ``None`` target (legacy spec without a ``name:`` field)
    resolves to the root workdir unchanged. The runtime uses
    this path for any top-level agent that didn't declare a
    name — the workflow IS the root, so its own workdir is
    correct.
    """
    root = _spec("my_root")
    root_dir = Path("/bundles/my_root")
    assert _resolve_workdir_for_spec(root, root_dir, None) == root_dir


def test_root_name_returns_root_workdir() -> None:
    """
    Passing the root's own name resolves to the root workdir.
    Sanity check: the resolver must not accidentally walk into
    ``agents/<root_name>`` when the target IS the root.
    """
    root = _spec("my_root")
    root_dir = Path("/bundles/my_root")
    assert _resolve_workdir_for_spec(root, root_dir, "my_root") == root_dir


def test_direct_subagent_resolves_to_agents_subdir() -> None:
    """
    A direct sub-agent resolves to ``<root>/agents/<sub>``.
    This is the bug's main case: before the fix, the runtime
    passed ``<root>`` here and local tool paths like
    ``tools/python/x.py`` joined to
    ``<root>/tools/python/x.py`` — which doesn't exist.
    """
    worker = _spec("worker")
    root = _spec("supervisor", sub_agents=[worker])
    root_dir = Path("/bundles/supervisor")
    assert _resolve_workdir_for_spec(root, root_dir, "worker") == root_dir / "agents" / "worker"


def test_nested_subagent_walks_the_tree() -> None:
    """
    A nested sub-agent (sub-agent of a sub-agent) resolves to
    ``<root>/agents/<level1>/agents/<level2>``. The bundle
    layout is recursive, so the resolver must walk the tree
    in parallel with the spec lookup — not just join one
    level.
    """
    deep = _spec("deep")
    mid = _spec("mid", sub_agents=[deep])
    root = _spec("root", sub_agents=[mid])
    root_dir = Path("/bundles/root")
    assert (
        _resolve_workdir_for_spec(root, root_dir, "deep")
        == root_dir / "agents" / "mid" / "agents" / "deep"
    )


def test_sibling_subagents_each_get_their_own_workdir() -> None:
    """
    Two sibling sub-agents under the same parent resolve to
    distinct directories. Regression guard for a bug where a
    naive resolver returned the first match regardless of
    name.
    """
    left = _spec("left")
    right = _spec("right")
    root = _spec("root", sub_agents=[left, right])
    root_dir = Path("/bundles/root")
    left_dir = _resolve_workdir_for_spec(root, root_dir, "left")
    right_dir = _resolve_workdir_for_spec(root, root_dir, "right")
    assert left_dir == root_dir / "agents" / "left"
    assert right_dir == root_dir / "agents" / "right"
    assert left_dir != right_dir


def test_unknown_name_raises_lookup_error() -> None:
    """
    An unknown sub-agent name must fail loud. Returning the
    root workdir silently would mean a typo in the task store
    sends the wrong tools to the wrong agent — catastrophic
    failure mode.
    """
    root = _spec("root", sub_agents=[_spec("worker")])
    with pytest.raises(LookupError, match="not found"):
        _resolve_workdir_for_spec(root, Path("/bundles/root"), "nonexistent")
