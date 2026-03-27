"""Tests for agent_plane.runtime.caps."""

from __future__ import annotations

from agent_plane.runtime.caps import RuntimeCaps


def test_runtime_caps_default_value() -> None:
    """RuntimeCaps with no args uses the 7200s default."""
    caps = RuntimeCaps()

    # Default execution_timeout is 7200s per the dataclass definition.
    # Failure means the default was changed without updating dependents.
    assert caps.execution_timeout == 7200


def test_runtime_caps_custom_value() -> None:
    """RuntimeCaps accepts a custom execution_timeout."""
    caps = RuntimeCaps(execution_timeout=3600)

    # Custom value should override the default.
    # Failure means the constructor ignores the argument.
    assert caps.execution_timeout == 3600
