"""Typed dataclasses representing a parsed agent image spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default HTTP status codes considered transient and retryable.
_DEFAULT_RETRYABLE_STATUS_CODES = [429, 500, 502, 503]


@dataclass
class RetryConfig:
    """
    Retry policy for LLM or tool calls.

    Reusable across LLM and tool configs. Timeouts always trigger
    a retry (if attempts remain). ``status_codes`` controls which
    HTTP error responses are retried on top of timeouts.

    :param max_attempts: Total attempts including the first call.
        ``3`` means up to 2 retries, e.g. ``3``.
    :param backoff_base: Base delay in seconds for exponential
        backoff, e.g. ``2.0``.
    :param backoff_max: Maximum delay between retries in seconds,
        e.g. ``30.0``.
    :param status_codes: HTTP status codes that trigger a retry.
        Timeouts always trigger a retry regardless of this list.
        Ignored for non-HTTP tools (local Python/TypeScript).
    """

    max_attempts: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 30.0
    status_codes: list[int] = field(default_factory=lambda: list(_DEFAULT_RETRYABLE_STATUS_CODES))


# LLM retry defaults — more aggressive than tool defaults because
# LLM providers have well-defined transient error semantics.
LLM_RETRY_DEFAULTS = RetryConfig(
    max_attempts=3,
    backoff_base=2.0,
    backoff_max=30.0,
)

# Tool retry defaults — more conservative because tool failures
# are often semantic (bad arguments, missing resource) not transient.
TOOL_RETRY_DEFAULTS = RetryConfig(
    max_attempts=2,
    backoff_base=1.0,
    backoff_max=10.0,
)


@dataclass
class ExecutionConfig:
    """
    Overall agent execution limits.

    :param timeout: Wall-clock deadline for the entire agent loop
        in seconds, e.g. ``3600``.
    :param max_iterations: Maximum agent loop iterations, e.g.
        ``1000``.
    """

    timeout: int = 3600
    max_iterations: int = 1000


@dataclass
class LLMConfig:
    """
    LLM configuration block from config.yaml.

    ``model`` is the only required field. ``timeout`` and ``retry``
    control call-level resilience. All other keys from the YAML
    ``llm:`` block are collected into ``extra`` and passed through
    to the OpenAI SDK as-is.

    :param model: The provider-prefixed model identifier, e.g.
        ``"openai/gpt-5.4"`` or ``"anthropic/claude-sonnet-4-20250514"``.
    :param extra: Arbitrary kwargs from the YAML ``llm:`` block
        (everything except ``model``, ``connection``, ``timeout``,
        and ``retry``). Values are heterogeneous (str, int, dict,
        etc.) so ``Any`` is the narrowest safe type. Example:
        ``{"temperature": 0.7, "max_tokens": 4096}``.
    :param connection: Per-provider connection overrides from the
        YAML ``connection:`` sub-block. Keys are provider-specific,
        e.g. ``{"api_key": "...", "base_url": "..."}`` for
        OpenAI-compatible providers or
        ``{"aws_region": "us-west-2"}`` for Bedrock.
        ``None`` means use environment variable defaults.
    :param timeout: LLM call timeout in seconds (both streaming
        and non-streaming), e.g. ``300``.
    :param retry: Retry policy for transient LLM failures.
    """

    model: str
    # Arbitrary kwargs from the YAML llm block (everything except
    # ``model``, ``connection``, ``timeout``, and ``retry``). Values
    # are heterogeneous (str, int, dict, etc.) so Any is the narrowest
    # safe type.
    extra: dict[str, Any] = field(default_factory=dict)
    # Per-provider connection overrides (api_key, base_url, etc.).
    # None means rely on environment variable defaults.
    connection: dict[str, str] | None = None
    timeout: int = 300
    retry: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_attempts=3,
            backoff_base=2.0,
            backoff_max=30.0,
        )
    )


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
    :param timeout: Default timeout in seconds for all tool calls,
        e.g. ``60``. Individual tools can override this.
    :param retry: Default retry policy for all tool calls.
        Individual tools can override this.
    """

    agents: list[str] = field(default_factory=list)
    timeout: int = 60
    retry: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_attempts=2,
            backoff_base=1.0,
            backoff_max=10.0,
        )
    )


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
    :param skill_dir: Absolute path to the skill's directory on
        disk, e.g. ``Path("/agents/code-review")``. Used by
        ``read_skill_file`` to resolve resource paths. ``None``
        when the skill was created in-memory (e.g. tests).
    """

    name: str
    description: str
    content: str
    skill_dir: Path | None = None


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
    :param timeout: Per-tool timeout in seconds. ``None`` inherits
        ``tools.timeout``.
    :param retry: Per-tool retry policy. ``None`` inherits
        ``tools.retry``.
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
    # Per-tool timeout/retry overrides. None = inherit from
    # tools.timeout / tools.retry.
    timeout: int | None = None
    retry: RetryConfig | None = None


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
    :param timeout: Per-tool timeout in seconds. ``None`` inherits
        ``tools.timeout``.
    :param retry: Per-tool retry policy. ``None`` inherits
        ``tools.retry``.
    """

    name: str  # derived tool name, e.g. "arxiv.search"
    path: str  # relative path within the agent image, e.g. "tools/python/arxiv_search.py"
    language: str  # "python" | "typescript"
    # Per-tool timeout/retry overrides. None = inherit from
    # tools.timeout / tools.retry.
    timeout: int | None = None
    retry: RetryConfig | None = None


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
    :param execution: Agent execution limits (wall-clock timeout
        and max iterations).
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
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
