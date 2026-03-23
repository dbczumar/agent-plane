"""Agent entity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_plane.spec.types import AgentSpec


@dataclass
class Agent:
    """A registered agent."""

    id: str
    created_at: int
    name: str
    description: str | None = None


@dataclass
class LoadedAgent:
    """
    A fully loaded agent — parsed spec plus the extracted working
    directory on disk. Returned by AgentCache.load().
    """

    spec: AgentSpec
    workdir: Path
