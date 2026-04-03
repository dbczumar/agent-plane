"""Backwards-compatibility shim — moved to executors/claude.py."""

from agent_plane.runtime.executors.claude import (
    ClaudeAgentsExecutor,
)

__all__ = ["ClaudeAgentsExecutor"]
