"""Typed dataclasses representing a parsed agent image spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """
    LLM configuration block from config.yaml.

    Only ``model`` is a known field. All other keys from the YAML
    ``llm:`` block are collected into ``extra`` and passed through
    to the OpenAI SDK as-is, so any parameter the OpenAI API
    supports can be set in the agent spec without code changes here.

    :param model: The OpenAI model identifier,
        e.g. ``"gpt-4o"`` or ``"claude-sonnet-4-20250514"``.
    :param extra: Arbitrary OpenAI kwargs from the YAML ``llm:``
        block (everything except ``model``). Values are
        heterogeneous (str, int, dict, etc.) so ``Any`` is the
        narrowest safe type. Example:
        ``{"temperature": 0.7, "max_tokens": 4096}``.
    """

    model: str
    # Arbitrary OpenAI kwargs from the YAML llm block (everything
    # except ``model``). Values are heterogeneous (str, int, dict,
    # etc.) so Any is the narrowest safe type.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityConfig:
    """
    Declared input/output content types.

    :param input: Accepted input modalities. Valid values are
        ``"text"``, ``"image"``, ``"audio"``, ``"video"``, and
        ``"file"``. Defaults to ``["text"]``.
    :param output: Produced output modalities. Valid values are
        ``"text"``, ``"image"``, and ``"audio"``. Defaults to
        ``["text"]``.
    """

    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])


@dataclass
class InteractionConfig:
    """
    Interaction contract: conversational mode and modalities.

    :param conversational: Whether the agent supports multi-turn
        conversation. Defaults to ``True``.
    :param modalities: Input/output content type declarations.
        Defaults to text-only.
    """

    conversational: bool = True
    modalities: ModalityConfig = field(default_factory=ModalityConfig)


@dataclass
class ToolsConfig:
    """
    Declared tool references from config.yaml.

    :param agents: Names of sub-agents this agent can delegate to,
        e.g. ``["summarizer", "code-reviewer"]``. Each name must
        match a directory under ``agents/``.
    """

    agents: list[str] = field(default_factory=list)


@dataclass
class SkillSpec:
    """
    A parsed skill from ``skills/<name>/SKILL.md``.

    :param name: Lowercase kebab-case skill identifier, e.g.
        ``"code-review"``. Must match ``[a-z0-9-]+``.
    :param description: Human-readable summary of what the skill
        does (max 1024 characters).
    :param content: The body of the SKILL.md file after the YAML
        frontmatter, containing the skill's instructions.
    :param allowed_tools: Tool names this skill is permitted to
        invoke, e.g. ``["grep", "read-file"]``. Empty list means
        no tool restrictions.
    """

    name: str
    description: str
    content: str
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """
    An MCP server declaration from ``tools/mcp/<name>.yaml``.

    :param name: Unique server identifier, e.g. ``"github"``.
    :param transport: Connection protocol. Must be ``"stdio"`` or
        ``"http"``.
    :param description: Optional human-readable summary of the
        server's purpose.
    :param command: Executable to launch for ``stdio`` transport,
        e.g. ``"npx"``. Required when transport is ``"stdio"``.
    :param args: Command-line arguments for the ``stdio`` command,
        e.g. ``["-y", "@modelcontextprotocol/server-github"]``.
    :param env: Environment variables to set for the ``stdio``
        process, e.g. ``{"GITHUB_TOKEN": "ghp_abc123"}``.
    :param url: Endpoint URL for ``http`` transport, e.g.
        ``"https://mcp.example.com/sse"``. Required when transport
        is ``"http"``.
    :param headers: HTTP headers to include with ``http`` requests,
        e.g. ``{"Authorization": "Bearer tok_xyz"}``.
    """

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
    """
    A discovered local tool file (Python or TypeScript).

    :param name: Derived tool name with dots replacing underscores,
        e.g. ``"arxiv.search"`` (from ``arxiv_search.py``).
    :param path: Relative path within the agent image, e.g.
        ``"tools/python/arxiv_search.py"``.
    :param language: Source language. Either ``"python"`` or
        ``"typescript"``.
    """

    name: str  # derived tool name, e.g. "arxiv.search"
    path: str  # relative path within the agent image, e.g. "tools/python/arxiv_search.py"
    language: str  # "python" | "typescript"


@dataclass
class AgentSpec:
    """
    A fully parsed agent image.

    Produced by the parser from a directory on disk; validated by
    the validator.

    :param spec_version: Schema version of the agent spec. Currently
        must be ``1``.
    :param name: Human-readable agent name, e.g. ``"code-reviewer"``.
    :param description: Short summary of the agent's purpose.
    :param llm: LLM configuration. ``None`` means the agent does not
        declare an LLM preference.
    :param interaction: Conversational mode and modality settings.
    :param tools: Declared tool references (sub-agent names, etc.).
    :param params: Arbitrary key-value parameters readable by skills
        and tools. Values are heterogeneous (str, int, bool, list,
        dict), so ``Any`` is the narrowest safe type. Example:
        ``{"max_retries": 3, "style": "concise"}``.
    :param instructions: Agent system prompt, typically from
        ``AGENTS.md``. ``None`` if no instructions file is present.
    :param skills: Parsed skills from ``skills/<name>/SKILL.md``.
    :param mcp_servers: MCP server declarations from
        ``tools/mcp/<name>.yaml``.
    :param local_tools: Discovered local tool files from
        ``tools/python/`` and ``tools/typescript/``.
    :param sub_agents: Recursively parsed child agents from
        ``agents/<name>/``.
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
