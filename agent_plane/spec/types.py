"""Typed dataclasses representing a parsed agent image spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """LLM configuration block from config.yaml."""

    model: str
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None  # "low" | "medium" | "high"


@dataclass
class ModalityConfig:
    """Declared input/output content types."""

    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class InteractionConfig:
    """Interaction contract: conversational mode and modalities."""

    conversational: bool = True
    modalities: ModalityConfig = field(default_factory=ModalityConfig)


@dataclass
class ToolsConfig:
    """Declared tool references from config.yaml."""

    agents: list[str] = field(default_factory=list)


@dataclass
class SkillSpec:
    """A parsed skill from skills/<name>/SKILL.md."""

    name: str
    description: str
    content: str
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """An MCP server declaration from tools/mcp/<name>.yaml."""

    name: str
    transport: str  # "stdio" | "http"
    description: str | None = None
    # stdio fields
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # http fields
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class LocalToolInfo:
    """A discovered local tool file (Python or TypeScript)."""

    name: str  # derived tool name, e.g. "arxiv.search"
    path: str  # relative path within the agent image, e.g. "tools/python/arxiv_search.py"
    language: str  # "python" | "typescript"


@dataclass
class AgentSpec:
    """
    A fully parsed agent image.

    Produced by the parser from a directory on disk; validated by the validator.
    """

    spec_version: int
    name: str | None = None
    description: str | None = None
    llm: LLMConfig | None = None
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    # Arbitrary key-value params readable by skills and tools.
    # Values are heterogeneous (str, int, bool, list, dict), so Any
    # is the narrowest safe type here.
    params: dict[str, Any] = field(default_factory=dict)
    instructions: str | None = None  # contents of AGENTS.md
    skills: list[SkillSpec] = field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    local_tools: list[LocalToolInfo] = field(default_factory=list)
    sub_agents: list[AgentSpec] = field(default_factory=list)
