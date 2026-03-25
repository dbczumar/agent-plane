"""Agent entity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_plane.spec import AgentSpec


@dataclass
class Agent:
    """
    A registered agent.

    :param id: Unique agent identifier, e.g. ``"ag_abc123"``.
    :param created_at: Unix epoch timestamp of creation.
    :param name: Human-readable agent name, e.g. ``"research-agent"``.
    :param description: Optional free-text description of the agent.
    """

    id: str
    created_at: int
    name: str
    description: str | None = None


@dataclass
class LoadedAgent:
    """
    A fully loaded agent — parsed spec plus the extracted working
    directory on disk. Returned by ``AgentCache.load()``.

    :param spec: The parsed agent spec from config.yaml.
    :param workdir: Path to the extracted agent image directory on disk.
    """

    spec: AgentSpec
    workdir: Path
