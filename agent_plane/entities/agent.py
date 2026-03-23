"""Agent entity."""

from dataclasses import dataclass


@dataclass
class Agent:
    """A registered agent."""

    id: str
    created_at: int
    name: str
    description: str | None = None
    bundle_location: str = ""
