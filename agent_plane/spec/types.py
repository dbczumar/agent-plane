"""Typed dataclasses representing a parsed agent image spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    # EvaluationContext is a runtime evaluation artifact (see
    # agent_plane.policies.types); PhaseSelector.matches takes
    # one by attribute access only, so the annotation lives
    # behind TYPE_CHECKING to avoid an import cycle.
    from agent_plane.policies.types import EvaluationContext

# Default HTTP status codes considered transient and retryable.
_DEFAULT_RETRYABLE_STATUS_CODES = [429, 500, 502, 503]

# Default classifier timeout for PromptPolicy evaluations. 30 s
# balances classifier latency against the cost of blocking the
# agent loop for every evaluated phase. The agent-level LLM
# default (300 s) is tuned for generation; using it for a
# blocking classifier stalls the loop for minutes on each tool
# call — see POLICIES.md §9.2. Overrideable via
# ``policy.llm.request_timeout`` on individual policies.
DEFAULT_POLICY_CLASSIFIER_TIMEOUT = 30

# Default timeout (seconds) for user approval on an ASK policy.
# 30 s matches omniagents parity. Overrideable per-policy via
# ``PolicySpec.ask_timeout`` or spec-wide via
# ``GuardrailsSpec.ask_timeout``. See POLICIES.md §7, §13.
DEFAULT_ASK_TIMEOUT = 30


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
class ExecutorSpec:
    """
    Top-level executor configuration.

    ``type`` is the discriminator for the entire spec's validity —
    it determines which other top-level sections and fields are
    valid. Invalid fields are rejected by the validator.

    :param type: Executor type. ``"llm"`` (default),
        ``"claude_sdk"``, ``"agents_sdk"``, or ``"remote"``.
    :param timeout: Task deadline in seconds (wall-clock limit for
        the entire agent loop), e.g. ``3600``.
    :param max_iterations: Maximum ``run_turn()`` calls before the
        loop terminates as incomplete, e.g. ``1000``.
    :param endpoint: URL for the remote executor's turn endpoint,
        e.g. ``"http://localhost:8000/v1/turns"``. Required when
        ``type`` is ``"remote"``, invalid otherwise.
    :param request_timeout: Per-HTTP-call timeout in seconds for the
        remote executor, e.g. ``300``. Only valid when ``type`` is
        ``"remote"``.
    """

    type: str = "llm"
    timeout: int = 3600
    max_iterations: int = 1000
    endpoint: str | None = None
    request_timeout: int | None = None


@dataclass
class CompactionConfig:
    """
    Context compaction configuration.

    Controls when the agent compacts its conversation history to
    stay within the LLM's context window. Compaction is layered:
    (1) clear tool result bodies, (2) LLM summarization, (3)
    truncation as emergency fallback.

    :param trigger_threshold: Fraction of the model's context window
        at which proactive compaction fires (after the first overflow
        has been observed and the window size is known), e.g. ``0.8``
        means fire at 80% of the window.
    :param recent_window: Number of recent LLM iterations to protect
        from compaction. Items within this window are never cleared or
        summarized — the agent always has verbatim access to its most
        recent work, e.g. ``5``.
    """

    trigger_threshold: float = 0.8
    recent_window: int = 5


@dataclass
class LLMConfig:
    """
    LLM configuration block from config.yaml.

    ``model`` is the only required field. ``request_timeout`` and
    ``retry`` control call-level resilience. All other keys from the
    YAML ``llm:`` block are collected into ``extra`` and passed
    through to the OpenAI SDK as-is.

    :param model: The provider-prefixed model identifier, e.g.
        ``"openai/gpt-5.4"`` or ``"anthropic/claude-sonnet-4-20250514"``.
    :param extra: Arbitrary kwargs from the YAML ``llm:`` block
        (everything except ``model``, ``connection``,
        ``request_timeout``, and ``retry``). Values are heterogeneous
        (str, int, dict, etc.) so ``Any`` is the narrowest safe type.
        Example: ``{"temperature": 0.7, "max_tokens": 4096}``.
    :param connection: Per-provider connection overrides from the
        YAML ``connection:`` sub-block. Keys are provider-specific,
        e.g. ``{"api_key": "...", "base_url": "..."}`` for
        OpenAI-compatible providers or
        ``{"aws_region": "us-west-2"}`` for Bedrock.
        ``None`` means use environment variable defaults.
    :param request_timeout: Per-LLM-call timeout in seconds (both
        streaming and non-streaming), e.g. ``300``. Named
        ``request_timeout`` to distinguish from the task-level
        ``executor.timeout``.
    :param retry: Retry policy for transient LLM failures.
    """

    model: str
    # Arbitrary kwargs from the YAML llm block (everything except
    # ``model``, ``connection``, ``request_timeout``, and ``retry``).
    # Values are heterogeneous (str, int, dict, etc.) so Any is the
    # narrowest safe type.
    extra: dict[str, Any] = field(default_factory=dict)
    # Per-provider connection overrides (api_key, base_url, etc.).
    # None means rely on environment variable defaults.
    connection: dict[str, str] | None = None
    request_timeout: int = 300
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
class BuiltinToolConfig:
    """
    Configuration for a single built-in tool declared in
    ``tools.builtins``.

    :param name: The registered tool name, e.g.
        ``"web_search"``.
    :param config: Tool-specific key-value pairs, e.g.
        ``{"api_key": "AIza...", "engine_id": "abc123"}``.
        Empty when the tool needs no configuration.
    """

    name: str
    config: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """
    Agent-level sandbox configuration for local tool execution.

    Only contains settings the agent author controls (what
    execution environment their tools need). Whether sandboxing
    is enabled/enforced is a runtime decision — see
    ``RuntimeCaps.sandbox_enabled``.

    :param docker_image: When set, tools run inside this Docker
        container instead of a local subprocess. Docker provides
        its own isolation, e.g. ``"python:3.12-slim"``.
    """

    docker_image: str | None = None


@dataclass
class ToolsConfig:
    """
    Declared tool references from config.yaml.

    :param agents: Names of sub-agents this agent can delegate to,
        e.g. ``["summarizer", "code-reviewer"]``. Each name must
        match a directory under ``agents/``.
    :param builtins: Built-in tools to enable, e.g.
        ``[BuiltinToolConfig(name="web_search")]``. Each
        entry carries the tool name and optional config fields
        (API keys, engine IDs, etc.).
    :param timeout: Default timeout in seconds for all tool calls,
        e.g. ``60``. Individual tools can override this.
    :param retry: Default retry policy for all tool calls.
        Individual tools can override this.
    """

    agents: list[str] = field(default_factory=list)
    builtins: list[BuiltinToolConfig] = field(default_factory=list)
    timeout: int = 60
    retry: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_attempts=2,
            backoff_base=1.0,
            backoff_max=10.0,
        )
    )
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


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

    Only the HTTP (SSE) transport is supported. Agents that need
    local subprocess tools should use client-side tools instead.

    :param name: Unique server identifier, e.g. ``"github"``.
    :param url: Endpoint URL for the HTTP (SSE) transport, e.g.
        ``"https://mcp.example.com/sse"``.
    :param description: Optional human-readable summary of the
        server's purpose.
    :param headers: HTTP headers to include with requests,
        e.g. ``{"Authorization": "Bearer tok_xyz"}``.
    :param timeout: Per-tool timeout in seconds. ``None`` inherits
        ``tools.timeout``. When ultimately ``None`` at runtime, the
        MCP SDK defaults apply: 5 s for the initial HTTP connection
        handshake and 300 s (5 min) for each SSE event read.
    :param retry: Per-tool retry policy. ``None`` inherits
        ``tools.retry``.
    """

    name: str
    url: str
    description: str | None = None
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    # Per-tool timeout/retry overrides. None = inherit from
    # tools.timeout / tools.retry.
    timeout: int | None = None
    retry: RetryConfig | None = None

    def __repr__(self) -> str:
        """
        String representation that redacts header values.

        Header keys are shown but values are replaced with
        ``"[REDACTED]"`` to prevent credential leakage in
        logs and exception tracebacks.
        """
        redacted = {k: "[REDACTED]" for k in self.headers} if self.headers else {}
        return (
            f"MCPServerConfig(name={self.name!r}, url={self.url!r}, "
            f"headers={redacted!r}, timeout={self.timeout!r}, "
            f"retry={self.retry!r})"
        )


@dataclass
class LocalToolInfo:
    """
    A discovered local tool file (Python or TypeScript).

    :param name: Derived tool name from filename stem,
        e.g. ``"arxiv_search"`` (from ``arxiv_search.py``).
    :param path: Relative path within the agent image, e.g.
        ``"tools/python/arxiv_search.py"``.
    :param language: Source language. Either ``"python"`` or
        ``"typescript"``.
    :param timeout: Per-tool timeout in seconds. ``None`` inherits
        ``tools.timeout``.
    :param retry: Per-tool retry policy. ``None`` inherits
        ``tools.retry``.
    :param has_inline_deps: ``True`` if the tool file contains
        PEP 723 inline script metadata with dependencies.
    :param inline_deps: PEP 508 dependency specifiers extracted
        from the ``# /// script`` block. ``None`` when no
        inline metadata is present.
    """

    name: str
    path: str
    language: str
    # Per-tool timeout/retry overrides. None = inherit from
    # tools.timeout / tools.retry.
    timeout: int | None = None
    retry: RetryConfig | None = None
    # PEP 723 inline dependency metadata. Populated at load time
    # by scanning the tool source file.
    has_inline_deps: bool = False
    inline_deps: list[str] | None = None


# ── Guardrails / Policies (POLICIES.md §3, §4) ──────────────
#
# Spec-level types for the policy system. These are pure data
# containers — no runtime behavior here. Runtime policy classes
# (LabelPolicy, FunctionPolicy, PromptPolicy) and the
# PolicyEngine live under agent_plane.runtime.policies in later
# phases.


class Phase(str, Enum):
    """
    The four points in the agent loop where policies fire.

    ``str`` mix-in keeps YAML parsing trivial
    (``Phase("tool_call")``) and preserves the string form in
    logs / JSON serialization.

    - ``INPUT``: after a new user message arrives, before the
      LLM turn.
    - ``TOOL_CALL``: before dispatching a tool call emitted by
      the LLM.
    - ``TOOL_RESULT``: after a tool returns, before the result
      is surfaced back to the LLM.
    - ``OUTPUT``: after the LLM's final assistant message,
      before persistence.
    """

    INPUT = "input"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OUTPUT = "output"


class PolicyAction(str, Enum):
    """
    The three decisions a policy can emit.

    - ``ALLOW``: the phase proceeds normally.
    - ``ASK``: park for user approval; on approve → ALLOW, on
      refuse/timeout → DENY.
    - ``DENY``: short-circuit the phase; replace content with a
      sentinel so downstream steps cannot act on it.
    """

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PhaseSelector:
    """
    One entry in a policy's ``on:`` list.

    YAML shapes:

    - ``"tool_call"`` →
      ``PhaseSelector(phase=Phase.TOOL_CALL, tool_name=None)``
      (wildcard — matches every tool call).
    - ``"tool_call:code_sandbox"`` →
      ``PhaseSelector(Phase.TOOL_CALL, "code_sandbox")``
      (narrows to one tool by name).

    Tool-name narrowing is only valid on ``TOOL_CALL`` and
    ``TOOL_RESULT`` phases — the parser rejects it on
    ``INPUT`` / ``OUTPUT``.

    :param phase: Which enforcement point this selector fires
        on.
    :param tool_name: When set, only matches that tool. When
        ``None``, matches any tool in the phase (wildcard).
    """

    phase: Phase
    tool_name: str | None = None

    def matches(self, ctx: EvaluationContext) -> bool:
        """
        Test whether this selector matches an evaluation
        context.

        :param ctx: The current evaluation context
            (phase + resolved tool name).
        :returns: ``True`` if the selector's phase matches and
            (when ``tool_name`` is set) the context's
            ``tool_name`` matches.
        """
        if ctx.phase != self.phase:
            return False
        if self.tool_name is None:
            return True
        return ctx.tool_name == self.tool_name


@dataclass(frozen=True)
class LabelDef:
    """
    Schema for one label key.

    Declared statically in
    ``spec.guardrails.labels[key]``. Controls what values the
    label can take and how it can change over the course of a
    conversation (see POLICIES.md §10).

    :param initial: Seed value written at conversation start.
        ``None`` means the label is unset until a policy
        writes it for the first time, e.g. ``"0"``.
    :param values: Ordered list of allowed values. Position
        defines ranking when ``monotonic`` is set. ``None``
        means schemaless — writes are unconstrained, e.g.
        ``["0", "1"]`` or
        ``["public", "internal", "confidential"]``.
    :param monotonic: Update constraint when ``values`` is
        declared. ``"increasing"`` means new index must be
        ``>=`` current; ``"decreasing"`` means ``<=``.
        ``None`` means free transitions between declared
        values.
    """

    initial: str | None = None
    values: list[str] | None = None
    monotonic: Literal["increasing", "decreasing"] | None = None


@dataclass(frozen=True)
class FunctionRef:
    """
    Reference to a policy callable, with optional factory
    kwargs.

    Two YAML shapes parse into this:

    - Bare string: ``function: myorg.policies.simple_check`` →
      ``FunctionRef(path="myorg.policies.simple_check",
      arguments=None)``. The resolved callable is the
      evaluator itself.
    - Dict: ``function: {path: myorg.policies.rate_limit,
      arguments: {limit: 10}}`` →
      ``FunctionRef(path="...", arguments={"limit": 10})``.
      The resolved callable is a factory called once at
      workflow start; its return value is the evaluator.

    :param path: Dotted import path to the callable or
        factory, e.g.
        ``"myorg.policies.search_rate_limit"``.
    :param arguments: Factory kwargs (present = factory form;
        absent = direct callable form). ``None`` when the
        YAML used the short string form.
    """

    path: str
    arguments: dict[str, Any] | None = None


@dataclass
class PolicySpec:
    """
    Base class for all policy specs.

    Concrete subtypes (``FunctionPolicySpec``,
    ``PromptPolicySpec``, ``LabelPolicySpec``) carry
    type-specific fields. The class itself *is* the
    discriminator — no separate ``type`` field at runtime.

    ``condition`` is a label-gate: if declared, the engine
    checks current label values against it BEFORE dispatching
    to the policy's ``evaluate()``. Non-matching policies are
    skipped entirely (no action emitted, no LLM call, no
    Python call) — cheap way to gate expensive policies on
    session state (POLICIES.md §4, §10).

    :param name: YAML key this policy was declared under,
        e.g. ``"block_canada_input"``.
    :param on: Phases this policy fires on, e.g.
        ``[PhaseSelector(Phase.TOOL_CALL, "web_search")]``.
    :param condition: Label-gate. Empty / absent =
        always-match; values coerced to strings at spec load,
        e.g. ``{"integrity": "0"}`` or
        ``{"role": ["admin", "ops"]}``.
    :param ask_timeout: Per-policy approval timeout override
        (seconds). ``None`` falls back to
        ``GuardrailsSpec.ask_timeout``. Useful when some ASKs
        are cheap (yes/no) and some expensive (review a 50 KB
        document).
    """

    name: str
    on: list[PhaseSelector]
    condition: dict[str, str | list[str]] | None = None
    ask_timeout: int | None = None


@dataclass
class FunctionPolicySpec(PolicySpec):
    """
    A policy backed by a Python callable (see POLICIES.md §9.1).

    :param function: Where the callable lives + optional
        factory kwargs.
    :param action: Allowed actions the callable may return.
        Returns outside this list → fail-closed DENY (or
        substituted ALLOW when the list contains no DENY, per
        the classifier-only carve-out in §13). ``None`` means
        accept any action.
    :param set_labels: Whitelist of label keys the callable
        may write. Keys outside dropped silently. ``None``
        means no writes declared (any key the callable emits
        that has a ``LabelDef`` will still validate against
        that LabelDef, but keys without a LabelDef are set
        freely per omniagents parity).
    """

    function: FunctionRef | None = None
    action: list[PolicyAction] | None = None
    set_labels: list[str] | None = None


@dataclass
class PromptPolicySpec(PolicySpec):
    """
    A policy backed by an LLM classifier (see POLICIES.md §9.2).

    :param prompt: Author-supplied domain logic, e.g. "Deny
        requests mentioning Canada." The framework generates
        the JSON-schema envelope; authors do NOT write JSON
        instructions.
    :param llm: Optional model override for this policy's
        classifier call. Omit to inherit the agent's
        top-level ``llm:`` **except for ``request_timeout``**,
        which defaults to
        :data:`DEFAULT_POLICY_CLASSIFIER_TIMEOUT` (30 s) for
        policies regardless of the agent's generation
        timeout. See POLICIES.md §9.2.
    :param action: Allowed actions the LLM may emit.
        Validation is strict — LLM output outside this list →
        fail-closed DENY (or substituted ALLOW when this list
        contains no DENY, per the classifier-only carve-out).
        Defaults to ``[ALLOW, DENY]``.
    :param set_labels: Whitelist of label keys the LLM may
        write. Keys outside dropped silently. ``None`` means
        the LLM cannot write labels at all.
    """

    prompt: str | None = None
    llm: LLMConfig | None = None
    action: list[PolicyAction] = field(
        default_factory=lambda: [PolicyAction.ALLOW, PolicyAction.DENY],
    )
    set_labels: list[str] | None = None


@dataclass
class LabelPolicySpec(PolicySpec):
    """
    A YAML-declared label policy (see POLICIES.md §9.3).

    Fires its declared action and emits its declared
    ``set_labels`` whenever selector + condition match. No
    Python, no LLM.

    :param action: The fixed action to emit — one of
        ``ALLOW``, ``ASK``, ``DENY``.
    :param reason: Human-readable reason — shown to users on
        ASK, included in logs on DENY. ``None`` on ALLOW.
    :param set_labels: Fixed label writes to emit. Applied
        through the standard validation path (``values`` +
        ``monotonic`` checks per ``LabelDef``).
    """

    action: PolicyAction | None = None
    reason: str | None = None
    set_labels: dict[str, str] | None = None


@dataclass
class GuardrailsSpec:
    """
    Top-level guardrails block from config.yaml.

    Bundles label definitions, policies, and the spec-wide
    ASK timeout default (POLICIES.md §3.2).

    :param labels: Per-key ``LabelDef`` schemas. ``None``
        means no labels declared — all label writes go
        through the schemaless-set-freely path.
    :param policies: Policies in YAML declaration order (the
        engine iterates in this order).
    :param ask_timeout: Spec-wide default approval timeout in
        seconds. Individual policies may override via
        ``PolicySpec.ask_timeout``. Defaults to
        :data:`DEFAULT_ASK_TIMEOUT` (30 s).
    """

    labels: dict[str, LabelDef] | None = None
    policies: list[PolicySpec] | None = None
    ask_timeout: int = DEFAULT_ASK_TIMEOUT


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
    :param executor: Executor configuration (type, task timeout,
        max iterations, remote endpoint). ``executor.type`` is the
        discriminator for the entire spec's validity.
    :param compaction: Compaction configuration for context management.
        ``None`` means use defaults (trigger at 80%, protect last 5
        iterations).
    :param guardrails: Guardrails configuration (labels + policies
        + ASK timeout). ``None`` means the agent declared no
        ``guardrails:`` block — runtime builds a no-op engine
        with zero policies and an empty label cache. See
        POLICIES.md §3, §4.
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
    executor: ExecutorSpec = field(default_factory=ExecutorSpec)
    compaction: CompactionConfig | None = None
    guardrails: GuardrailsSpec | None = None
