"""Parse an agent image directory into an AgentSpec."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.spec.types import (
    DEFAULT_ASK_TIMEOUT,
    AgentSpec,
    BuiltinToolConfig,
    CompactionConfig,
    ExecutorSpec,
    FunctionPolicySpec,
    FunctionRef,
    GuardrailsSpec,
    InteractionConfig,
    LabelDef,
    LabelPolicySpec,
    LLMConfig,
    LocalToolInfo,
    MCPServerConfig,
    ModalityConfig,
    Phase,
    PhaseSelector,
    PolicyAction,
    PolicySpec,
    PromptPolicySpec,
    RetryConfig,
    SandboxConfig,
    SkillSpec,
    ToolsConfig,
)

# Pattern for SKILL.md YAML frontmatter delimited by ---
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)", re.DOTALL)


class _ConfigYamlLoader(yaml.SafeLoader):
    """
    SafeLoader variant that does NOT treat ``on``/``off``/
    ``yes``/``no`` as booleans.

    Default PyYAML resolves these per the YAML 1.1 spec — a
    trap for our spec because the policy system uses
    ``on:`` as the selector field (see POLICIES.md §3.3
    implementation notes). Without this override, an author
    writing ``on: [input]`` would get a dict keyed by ``True``
    instead of ``"on"``. We scope the override to a dedicated
    loader class so the rest of the YAML 1.1 type inference
    stays intact.

    YAML 1.2 drops these bool aliases entirely; this override
    makes our loader YAML-1.2-aligned for the narrow set of
    aliases that matter here.
    """


# Replace the YAML 1.1 bool resolver pattern with a YAML 1.2
# pattern that accepts only ``true`` / ``false`` (and their
# title/upper-case variants). Strip the old bool resolvers
# first, then add back the narrowed one.
_BOOL_TAG = "tag:yaml.org,2002:bool"
_YAML_1_2_BOOL_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
for _ch in list(_ConfigYamlLoader.yaml_implicit_resolvers.keys()):
    _ConfigYamlLoader.yaml_implicit_resolvers[_ch] = [
        (tag, regexp)
        for tag, regexp in _ConfigYamlLoader.yaml_implicit_resolvers[_ch]
        if tag != _BOOL_TAG
    ]
# Re-register a narrowed bool resolver keyed on ``t`` / ``T`` /
# ``f`` / ``F`` only (the YAML 1.1 aliases keyed on o/O/y/Y/n/N
# are now gone, so those characters parse as plain strings).
_ConfigYamlLoader.add_implicit_resolver(
    _BOOL_TAG, _YAML_1_2_BOOL_RE, list("tTfF"),
)


def parse(root: Path, *, expand_env: bool = True) -> AgentSpec:
    """
    Parse an agent image directory into an :class:`AgentSpec`.

    :param root: Path to the agent image directory. Must contain
        ``config.yaml``.
    :param expand_env: Whether to expand ``${VAR}`` references in
        connection blocks and MCP headers. ``True`` (default) for
        deploy/runtime — raises on unresolved vars. ``False`` for
        scaffolding/validation where env vars may not yet be set.
    :returns: A fully populated :class:`AgentSpec` (not yet
        validated).
    :raises AgentPlaneError: If ``config.yaml`` is not valid YAML,
        has structural issues, or (when *expand_env* is ``True``)
        contains unresolved env vars.
    :raises FileNotFoundError: If ``config.yaml`` is missing.
    """
    config_path = root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {root}")

    raw = yaml.load(config_path.read_text(), Loader=_ConfigYamlLoader)
    if not isinstance(raw, dict):
        raise AgentPlaneError(
            f"config.yaml must be a YAML mapping, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )

    spec_version = raw.get("spec_version")
    if spec_version is None:
        raise AgentPlaneError(
            "config.yaml missing required field: spec_version",
            code=ErrorCode.INVALID_INPUT,
        )

    llm = _parse_llm(raw.get("llm"), expand_env=expand_env)
    interaction = _parse_interaction(raw.get("interaction"))
    tools_config = _parse_tools_config(raw.get("tools"))
    executor = _parse_executor(raw.get("executor"))
    compaction = _parse_compaction(raw.get("compaction"))
    guardrails = _parse_guardrails(raw.get("guardrails"), expand_env=expand_env)
    params = raw.get("params", {})

    instructions = _resolve_instructions(root, raw.get("instructions"))
    skills = _discover_skills(root / "skills")
    mcp_servers = _discover_mcp_servers(root / "tools" / "mcp", expand_env=expand_env)
    local_tools = _discover_local_tools(root / "tools")
    sub_agents = _discover_sub_agents(root / "agents", expand_env=expand_env)

    return AgentSpec(
        spec_version=spec_version,
        name=raw.get("name"),
        description=raw.get("description"),
        llm=llm,
        interaction=interaction,
        tools=tools_config,
        executor=executor,
        compaction=compaction,
        guardrails=guardrails,
        params=params,
        instructions=instructions,
        skills=skills,
        mcp_servers=mcp_servers,
        local_tools=local_tools,
        sub_agents=sub_agents,
    )


def _parse_llm(
    raw: dict[str, Any] | None,
    *,
    expand_env: bool = True,
) -> LLMConfig | None:
    """
    Parse the ``llm:`` block from config.yaml into an
    :class:`LLMConfig`.

    :param raw: The raw ``llm:`` mapping from config.yaml, or
        ``None`` if the block was absent. Example:
        ``{"model": "openai/gpt-4o", "temperature": 0.7}``.
    :param expand_env: Whether to expand ``${VAR}`` references in
        the connection block. ``False`` keeps literals as-is.
    :returns: A populated :class:`LLMConfig`, or ``None`` when
        the ``llm:`` block is absent.
    :raises AgentPlaneError: If the ``llm:`` block is present but
        missing the required ``model`` field.
    """
    if raw is None:
        return None
    model = raw.get("model")
    if model is None:
        raise AgentPlaneError(
            "llm block present but missing required field: model",
            code=ErrorCode.INVALID_INPUT,
        )
    # ``connection``, ``request_timeout``, and ``retry`` are separated
    # into their own typed fields; everything else is passed through
    # to the LLM SDK as extra kwargs.
    connection_raw = raw.get("connection")
    connection: dict[str, str] | None = None
    if isinstance(connection_raw, dict):
        raw_dict = {str(k): str(v) for k, v in connection_raw.items()}
        # Expand ${VAR} references so api_key: ${OPENAI_API_KEY} works.
        # Skipped when expand_env is False (scaffolding/validation).
        connection = expand_env_vars(raw_dict) if expand_env else raw_dict
    request_timeout = int(raw["request_timeout"]) if "request_timeout" in raw else 300
    retry = _parse_retry(raw.get("retry"))
    reserved = {"model", "connection", "request_timeout", "retry"}
    extra = {k: v for k, v in raw.items() if k not in reserved}
    return LLMConfig(
        model=str(model),
        extra=extra,
        connection=connection,
        request_timeout=request_timeout,
        retry=retry,
    )


def _parse_interaction(
    raw: dict[str, Any] | None,
) -> InteractionConfig:
    """
    Parse the ``interaction:`` block from config.yaml into an
    :class:`InteractionConfig`.

    :param raw: The raw ``interaction:`` mapping from config.yaml,
        or ``None`` if the block was absent. Example:
        ``{"conversational": false, "modalities": {"input":
        ["text", "image"]}}``.
    :returns: A populated :class:`InteractionConfig`. Returns
        defaults when *raw* is ``None``.
    """
    if raw is None:
        return InteractionConfig()
    modalities_raw = raw.get("modalities")
    if not isinstance(modalities_raw, dict):
        modalities = ModalityConfig()
    else:
        modalities = ModalityConfig(
            input=modalities_raw.get("input", ["text"]),
            output=modalities_raw.get("output", ["text"]),
        )
    conversational = raw.get("conversational", True)
    return InteractionConfig(
        conversational=bool(conversational),
        modalities=modalities,
    )


def _parse_tools_config(
    raw: dict[str, Any] | None,
) -> ToolsConfig:
    """
    Parse the ``tools:`` block from config.yaml into a
    :class:`ToolsConfig`.

    :param raw: The raw ``tools:`` mapping from config.yaml, or
        ``None`` if the block was absent. Example:
        ``{"agents": ["summarizer", "code-reviewer"],
        "timeout": 60}``.
    :returns: A populated :class:`ToolsConfig`. Returns defaults
        when *raw* is ``None``.
    """
    if raw is None:
        return ToolsConfig()
    timeout = int(raw["timeout"]) if "timeout" in raw else 60
    retry = _parse_retry(raw.get("retry"))
    builtins = _parse_builtin_tools(raw.get("builtins", []))
    sandbox = _parse_sandbox_config(raw.get("sandbox"))
    return ToolsConfig(
        agents=raw.get("agents", []),
        builtins=builtins,
        timeout=timeout,
        retry=retry,
        sandbox=sandbox,
    )


def _parse_sandbox_config(
    raw: dict[str, Any] | None,
) -> SandboxConfig:
    """
    Parse the ``tools.sandbox`` block from config.yaml.

    Only agent-level settings (``docker_image``) are parsed here.
    Whether sandboxing is enabled is a runtime decision, not an
    agent config decision::

        sandbox:
          docker_image: python:3.12-slim

    :param raw: The raw ``sandbox`` value from the ``tools``
        block. ``None`` means not specified (use defaults).
    :returns: A :class:`SandboxConfig`.
    """
    if raw is None or not isinstance(raw, dict):
        return SandboxConfig()
    return SandboxConfig(
        docker_image=raw.get("docker_image"),
    )


def _parse_builtin_tools(
    raw: list[str | dict[str, Any]],
) -> list[BuiltinToolConfig]:
    """
    Parse the ``tools.builtins`` list into
    :class:`BuiltinToolConfig` objects.

    Each entry is either a plain string (tool name with no config)
    or a dict with a ``name`` key and tool-specific config fields::

        builtins:
          - web_search
          - name: web_search
            api_key: ${GOOGLE_SEARCH_API_KEY}
            engine_id: ${GOOGLE_SEARCH_ENGINE_ID}

    :param raw: The raw ``builtins`` list from config.yaml.
    :returns: A list of :class:`BuiltinToolConfig` instances.
    :raises AgentPlaneError: If a dict entry is missing ``name``.
    """
    result: list[BuiltinToolConfig] = []
    for entry in raw:
        if isinstance(entry, str):
            result.append(BuiltinToolConfig(name=entry))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise AgentPlaneError(
                    "Each dict entry in tools.builtins must have a 'name' field.",
                    code=ErrorCode.INVALID_INPUT,
                )
            # Everything except 'name' is tool-specific config.
            config = {str(k): str(v) for k, v in entry.items() if k != "name"}
            result.append(
                BuiltinToolConfig(
                    name=str(name),
                    config=config,
                )
            )
        else:
            raise AgentPlaneError(
                f"tools.builtins entries must be strings or dicts, got {type(entry).__name__}.",
                code=ErrorCode.INVALID_INPUT,
            )
    return result


def _parse_retry(
    raw: dict[str, Any] | None,
) -> RetryConfig:
    """
    Parse a ``retry:`` block into a :class:`RetryConfig`.

    Returns defaults when *raw* is ``None`` or empty.

    :param raw: The raw ``retry:`` mapping, or ``None`` if absent.
        Example: ``{"max_attempts": 5, "status_codes": [429, 502]}``.
    :returns: A populated :class:`RetryConfig`.
    """
    if not raw:
        return RetryConfig()
    return RetryConfig(
        max_attempts=int(raw.get("max_attempts", 3)),
        backoff_base=float(raw.get("backoff_base", 2.0)),
        backoff_max=float(raw.get("backoff_max", 30.0)),
        status_codes=[int(c) for c in raw.get("status_codes", [429, 500, 502, 503])],
    )


def _parse_executor(
    raw: dict[str, Any] | None,
) -> ExecutorSpec:
    """
    Parse the ``executor:`` block into an :class:`ExecutorSpec`.

    Returns defaults (``type="llm"``) when *raw* is ``None``.

    :param raw: The raw ``executor:`` mapping, or ``None`` if
        absent. Example: ``{"type": "remote",
        "endpoint": "http://localhost:8000/v1/turns"}``.
    :returns: A populated :class:`ExecutorSpec`.
    """
    if raw is None:
        return ExecutorSpec()
    endpoint_raw = raw.get("endpoint")
    endpoint: str | None = None
    if endpoint_raw is not None:
        endpoint = str(endpoint_raw)
    request_timeout_raw = raw.get("request_timeout")
    request_timeout: int | None = None
    if request_timeout_raw is not None:
        request_timeout = int(request_timeout_raw)
    return ExecutorSpec(
        type=str(raw.get("type", "llm")),
        timeout=int(raw.get("timeout", 3600)),
        max_iterations=int(raw.get("max_iterations", 1000)),
        endpoint=endpoint,
        request_timeout=request_timeout,
    )


def _parse_compaction(
    raw: dict[str, Any] | None,
) -> CompactionConfig | None:
    """
    Parse the ``compaction:`` block from config.yaml into a
    :class:`CompactionConfig`.

    :param raw: The raw ``compaction:`` mapping from config.yaml, or
        ``None`` if the block was absent. Example:
        ``{"trigger_threshold": 0.8, "recent_window": 5}``.
    :returns: A populated :class:`CompactionConfig`, or ``None`` when
        the ``compaction:`` block is absent.
    """
    if raw is None:
        return None
    return CompactionConfig(
        trigger_threshold=float(raw.get("trigger_threshold", 0.8)),
        recent_window=int(raw.get("recent_window", 5)),
    )


def _resolve_instructions(root: Path, raw_value: object) -> str | None:
    """
    Resolve the instructions for an agent image.

    - If ``instructions`` is set in config.yaml and the value is
      a path to an existing file relative to *root*, read that
      file.
    - If ``instructions`` is set but is not a file path, treat
      the value as inline text.
    - If ``instructions`` is not set, fall back to reading
      ``AGENTS.md``.

    :param root: Path to the agent image directory.
    :param raw_value: The raw ``instructions`` value from
        config.yaml, or ``None`` if the key was absent. May be
        a relative file path (e.g. ``"prompts/system.md"``) or
        inline text.
    :returns: The resolved instruction text, or ``None`` if no
        instructions are available.
    """
    if raw_value is not None:
        text = str(raw_value)
        # Only attempt file lookup for short single-line values
        # that look like filenames (multiline text can't be a path).
        if "\n" not in text:
            candidate = root / text
            try:
                if candidate.is_file():
                    return candidate.read_text()
            except OSError:
                # Path too long or invalid characters — treat as inline text.
                pass
        return text
    # Default: read AGENTS.md if present
    agents_md = root / "AGENTS.md"
    if agents_md.exists():
        return agents_md.read_text()
    return None


def _discover_skills(skills_dir: Path) -> list[SkillSpec]:
    """
    Discover and parse all skills under the ``skills/`` directory.

    Each subdirectory containing a ``SKILL.md`` file is parsed via
    :func:`_parse_skill`.

    :param skills_dir: Path to the ``skills/`` directory, e.g.
        ``root / "skills"``.
    :returns: A sorted list of parsed :class:`SkillSpec` objects.
        Returns an empty list if *skills_dir* does not exist.
    """
    if not skills_dir.is_dir():
        return []
    skills: list[SkillSpec] = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        skill = _parse_skill(skill_md)
        skills.append(skill)
    return skills


def _parse_skill(skill_md: Path) -> SkillSpec:
    """
    Parse a single ``SKILL.md`` file into a :class:`SkillSpec`.

    The file must begin with YAML frontmatter delimited by ``---``
    lines, containing at least ``name`` and ``description`` keys.

    :param skill_md: Path to the ``SKILL.md`` file, e.g.
        ``skills/code-review/SKILL.md``.
    :returns: A populated :class:`SkillSpec`.
    :raises AgentPlaneError: If the frontmatter is missing,
        malformed, or lacks required fields.
    """
    text = skill_md.read_text()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise AgentPlaneError(
            f"SKILL.md missing YAML frontmatter: {skill_md}",
            code=ErrorCode.INVALID_INPUT,
        )
    frontmatter_str, content = match.groups()
    frontmatter = yaml.safe_load(frontmatter_str)
    if not isinstance(frontmatter, dict):
        raise AgentPlaneError(
            f"SKILL.md frontmatter must be a YAML mapping: {skill_md}",
            code=ErrorCode.INVALID_INPUT,
        )
    name = frontmatter.get("name")
    if name is None:
        raise AgentPlaneError(
            f"SKILL.md frontmatter missing required field 'name': {skill_md}",
            code=ErrorCode.INVALID_INPUT,
        )
    description = frontmatter.get("description")
    if description is None:
        raise AgentPlaneError(
            f"SKILL.md frontmatter missing required field 'description': {skill_md}",
            code=ErrorCode.INVALID_INPUT,
        )
    return SkillSpec(
        name=str(name),
        description=str(description),
        content=content.strip(),
        skill_dir=skill_md.parent,
    )


def expand_env_vars(
    mapping: dict[str, str],
) -> dict[str, str]:
    """
    Expand ``${VAR}`` and ``$VAR`` references in dict values
    against the current process environment.

    Raises :class:`AgentPlaneError` if any value still contains an
    unresolved ``$VAR`` or ``${VAR}`` reference after expansion.
    This catches typos and missing environment variables at parse
    time rather than silently passing literal ``${MISSING}`` to
    MCP servers or LLM clients.

    :param mapping: A string-to-string dict, e.g.
        ``{"TOKEN": "${GITHUB_TOKEN}"}``.
    :returns: A new dict with expanded values.
    :raises AgentPlaneError: If a value contains an unresolved
        environment variable reference after expansion.
    """
    result: dict[str, str] = {}
    for key, value in mapping.items():
        expanded = os.path.expandvars(value)
        check_unresolved_env_vars(key, expanded)
        result[key] = expanded
    return result


# Matches $VAR or ${VAR} patterns that survived expansion.
# Excludes $$ (escaped dollar sign).
_UNRESOLVED_VAR_RE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")


def check_unresolved_env_vars(key: str, value: str) -> None:
    """
    Raise if *value* contains unresolved environment variable
    references.

    Called after :func:`os.path.expandvars` to catch variables
    that were not set in the environment. Without this check,
    ``os.path.expandvars`` silently passes through the literal
    ``${VAR}`` string, which causes hard-to-debug failures
    downstream (e.g. an MCP server receiving ``$GITHUB_TOKEN``
    as a literal auth token).

    :param key: The dict key (for error messages), e.g.
        ``"GITHUB_TOKEN"``.
    :param value: The expanded value to check, e.g.
        ``"Bearer ${MISSING}"``.
    :raises AgentPlaneError: If *value* contains an unresolved
        ``$VAR`` or ``${VAR}`` reference.
    """
    match = _UNRESOLVED_VAR_RE.search(value)
    if match is not None:
        raise AgentPlaneError(
            f"Unresolved environment variable {match.group()!r} "
            f"in config key {key!r}. Set the variable in the "
            f"environment or remove the reference.",
            code=ErrorCode.INVALID_INPUT,
        )


def _discover_mcp_servers(
    mcp_dir: Path,
    *,
    expand_env: bool = True,
) -> list[MCPServerConfig]:
    """
    Discover and parse all MCP server configs under
    ``tools/mcp/``.

    Each ``.yaml`` file in the directory is parsed into an
    :class:`MCPServerConfig`.

    :param mcp_dir: Path to the ``tools/mcp/`` directory, e.g.
        ``root / "tools" / "mcp"``.
    :param expand_env: Whether to expand ``${VAR}`` references in
        headers. ``False`` keeps literals as-is.
    :returns: A sorted list of parsed :class:`MCPServerConfig`
        objects. Returns an empty list if *mcp_dir* does not
        exist.
    :raises AgentPlaneError: If any YAML file is malformed or
        missing required fields (``name``, ``transport``).
    """
    if not mcp_dir.is_dir():
        return []
    servers: list[MCPServerConfig] = []
    for yaml_file in sorted(mcp_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_file.read_text())
        if not isinstance(raw, dict):
            raise AgentPlaneError(
                f"MCP config must be a YAML mapping: {yaml_file}",
                code=ErrorCode.INVALID_INPUT,
            )
        name = raw.get("name")
        if name is None:
            raise AgentPlaneError(
                f"MCP config missing required field 'name': {yaml_file}",
                code=ErrorCode.INVALID_INPUT,
            )
        transport = raw.get("transport")
        if transport is None:
            raise AgentPlaneError(
                f"MCP config missing required field 'transport': {yaml_file}",
                code=ErrorCode.INVALID_INPUT,
            )
        if str(transport) != "http":
            raise AgentPlaneError(
                f"MCP server {name!r} uses unsupported transport "
                f"{transport!r} — only 'http' is supported: {yaml_file}",
                code=ErrorCode.INVALID_INPUT,
            )
        url = raw.get("url")
        if url is None:
            raise AgentPlaneError(
                f"MCP server {name!r} missing required field 'url': {yaml_file}",
                code=ErrorCode.INVALID_INPUT,
            )
        servers.append(
            MCPServerConfig(
                name=str(name),
                description=raw.get("description"),
                url=str(url),
                headers=(
                    expand_env_vars(raw.get("headers", {}))
                    if expand_env
                    else raw.get("headers", {})
                ),
                timeout=int(raw["timeout"]) if "timeout" in raw else None,
                retry=_parse_retry(raw["retry"]) if "retry" in raw else None,
            )
        )
    return servers


def _discover_local_tools(
    tools_dir: Path,
) -> list[LocalToolInfo]:
    """
    Discover local tool files under ``tools/python/`` and
    ``tools/typescript/``.

    Tool names are derived from the file stem directly (e.g.
    ``arxiv_search.py`` becomes ``"arxiv_search"``). Underscores
    are preserved — the tool name regex requires
    ``[a-zA-Z0-9_-]``.

    :param tools_dir: Path to the ``tools/`` directory, e.g.
        ``root / "tools"``.
    :returns: A sorted list of :class:`LocalToolInfo` objects
        covering both Python and TypeScript tools.
    """
    tools: list[LocalToolInfo] = []
    for language, subdir, ext in [
        ("python", "python", ".py"),
        ("typescript", "typescript", ".ts"),
    ]:
        lang_dir = tools_dir / subdir
        if not lang_dir.is_dir():
            continue
        for tool_file in sorted(lang_dir.glob(f"*{ext}")):
            tool_name = tool_file.stem
            rel_path = str(tool_file.relative_to(tools_dir.parent))
            tools.append(LocalToolInfo(name=tool_name, path=rel_path, language=language))
    return tools


def _discover_sub_agents(
    agents_dir: Path,
    *,
    expand_env: bool = True,
) -> list[AgentSpec]:
    """
    Recursively discover and parse sub-agents under ``agents/``.

    Each subdirectory containing a ``config.yaml`` is parsed via
    :func:`parse`, producing a nested :class:`AgentSpec`.

    :param agents_dir: Path to the ``agents/`` directory, e.g.
        ``root / "agents"``.
    :param expand_env: Whether to expand ``${VAR}`` references.
        Propagated to :func:`parse` for each sub-agent.
    :returns: A sorted list of recursively parsed
        :class:`AgentSpec` objects. Returns an empty list if
        *agents_dir* does not exist.
    """
    if not agents_dir.is_dir():
        return []
    sub_agents: list[AgentSpec] = []
    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        config_yaml = agent_dir / "config.yaml"
        if not config_yaml.exists():
            continue
        sub_agents.append(parse(agent_dir, expand_env=expand_env))
    return sub_agents


# ── Guardrails / policy parsers (POLICIES.md §3.3) ───────────
#
# Per POLICIES.md §13, most policy-spec errors fail LOUD at
# spec load — these helpers raise ``AgentPlaneError`` on
# malformed input rather than silently coercing to defaults.
# The exception is ``_parse_condition``, which permissively
# coerces scalar / list values to strings (matching omniagents
# parity for label values — see §14 of the audit).


def _parse_guardrails(
    raw: dict[str, Any] | None,
    *,
    expand_env: bool = True,
) -> GuardrailsSpec | None:
    """
    Parse the ``guardrails:`` block into a :class:`GuardrailsSpec`.

    Returns ``None`` when the block is absent entirely — the
    runtime builds a no-op policy engine in that case
    (POLICIES.md §10 zero-policy case).

    :param raw: The ``guardrails:`` mapping from config.yaml,
        or ``None`` when the block was absent. Example:
        ``{"labels": {"integrity": {"initial": "1",
        "values": ["0", "1"], "monotonic": "decreasing"}},
        "policies": {"block_canada_input": {"type": "prompt",
        ...}}, "ask_timeout": 30}``.
    :param expand_env: Whether to expand ``${VAR}`` references
        in any nested ``llm.connection`` blocks (PromptPolicy
        LLM overrides). Propagated to :func:`_parse_llm`.
    :returns: A populated :class:`GuardrailsSpec`, or ``None``
        when *raw* is ``None``.
    :raises AgentPlaneError: On any spec-load validation
        failure (unknown phases, empty ``on:`` lists, invalid
        label defs, bad policy types, etc.).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AgentPlaneError(
            "guardrails: must be a mapping, got "
            f"{type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    return GuardrailsSpec(
        labels=_parse_label_defs(raw.get("labels")),
        policies=_parse_policies(raw.get("policies"), expand_env=expand_env),
        ask_timeout=_parse_guardrails_ask_timeout(
            raw.get("ask_timeout", DEFAULT_ASK_TIMEOUT),
        ),
    )


def _parse_guardrails_ask_timeout(raw: Any) -> int:
    """
    Validate and coerce the spec-wide ``ask_timeout`` value.

    Accepts an integer (or string that parses as one);
    rejects ``<= 0`` at spec load per POLICIES.md §13. The
    ambiguity between "instant DENY" and "wait forever"
    drove the strict > 0 rule — both intents have explicit
    paths (omit ASK from action list; use a large finite
    number).

    :param raw: Raw ``guardrails.ask_timeout:`` value.
    :returns: Validated timeout in seconds.
    :raises AgentPlaneError: On non-integer or non-positive
        values.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise AgentPlaneError(
            f"guardrails.ask_timeout must be an integer, got {raw!r}",
            code=ErrorCode.INVALID_INPUT,
        )
    if value <= 0:
        raise AgentPlaneError(
            "guardrails.ask_timeout must be > 0 "
            "(omit ASK from policy action list for instant-DENY; "
            "use large finite values for long waits)",
            code=ErrorCode.INVALID_INPUT,
        )
    return value


def _parse_label_defs(
    raw: dict[str, Any] | None,
) -> dict[str, LabelDef] | None:
    """
    Parse the ``guardrails.labels:`` block into a dict of
    :class:`LabelDef` by key.

    Accepts three YAML shapes per POLICIES.md §3.1:

    - Bare string: ``integrity: "1"`` → schemaless with
      ``initial="1"``.
    - Dict (schema'd with initial):
      ``{initial: "1", values: [...], monotonic: ...}``.
    - Dict (schema'd without initial):
      ``{values: [...], monotonic: ...}``.

    :param raw: The ``labels:`` mapping, or ``None``.
    :returns: Dict mapping each label key to its
        :class:`LabelDef`. ``None`` when *raw* is ``None``.
    :raises AgentPlaneError: On malformed entries — empty
        dict, ``initial`` not in ``values``, unknown
        ``monotonic`` direction, etc.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AgentPlaneError(
            f"guardrails.labels: must be a mapping, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    defs: dict[str, LabelDef] = {}
    for key, entry in raw.items():
        defs[str(key)] = _parse_single_label_def(str(key), entry)
    return defs


def _parse_single_label_def(key: str, entry: Any) -> LabelDef:
    """
    Parse one label definition entry.

    :param key: The label key, used in error messages, e.g.
        ``"integrity"``.
    :param entry: Either a string (shorthand: value becomes
        ``initial``) or a dict with one or more of
        ``initial``, ``values``, ``monotonic``.
    :returns: A populated :class:`LabelDef`.
    :raises AgentPlaneError: On any malformed value.
    """
    # Bare-string shorthand: `integrity: "1"` → initial only.
    if isinstance(entry, str):
        return LabelDef(initial=entry)
    if isinstance(entry, bool) or entry is None or isinstance(entry, (int, float)):
        # Coerce scalar to string for shorthand form. YAML
        # authors often write `: 1` expecting "1"; coercing
        # matches the condition-value coercion policy elsewhere.
        return LabelDef(initial=str(entry) if entry is not None else None)
    if not isinstance(entry, dict):
        raise AgentPlaneError(
            f"label {key!r} must be a string or mapping, got "
            f"{type(entry).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    if not entry:
        # Empty-dict typo guard — matches POLICIES.md §13.
        raise AgentPlaneError(
            f"label {key!r} declares an empty dict — must contain at "
            f"least one of `initial`, `values`, or `monotonic`",
            code=ErrorCode.INVALID_INPUT,
        )
    initial = _coerce_label_initial(entry.get("initial"))
    values = _coerce_label_values(key, entry.get("values"))
    monotonic = _coerce_label_monotonic(key, entry.get("monotonic"))
    _validate_label_def_cross_fields(key, initial, values, monotonic)
    return LabelDef(initial=initial, values=values, monotonic=monotonic)


def _coerce_label_initial(raw: Any) -> str | None:
    """Coerce an ``initial:`` value to ``str | None``."""
    return None if raw is None else str(raw)


def _coerce_label_values(key: str, raw: Any) -> list[str] | None:
    """
    Coerce a ``values:`` list to ``list[str]`` or ``None``.

    :param key: Label key, for error messages.
    :param raw: Raw ``values:`` value from YAML.
    :returns: Every element str-coerced; ``None`` when
        *raw* is ``None``.
    :raises AgentPlaneError: When *raw* is a non-list
        non-None value.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise AgentPlaneError(
            f"label {key!r}: `values` must be a list, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    return [str(v) for v in raw]


def _coerce_label_monotonic(
    key: str,
    raw: Any,
) -> Literal["increasing", "decreasing"] | None:
    """
    Validate a ``monotonic:`` direction.

    :param key: Label key, for error messages.
    :param raw: Raw ``monotonic:`` value from YAML — must
        be ``"increasing"``, ``"decreasing"``, or absent.
    :returns: The validated direction, or ``None`` when
        *raw* is ``None``.
    :raises AgentPlaneError: On any other value.
    """
    if raw is None:
        return None
    if raw not in ("increasing", "decreasing"):
        raise AgentPlaneError(
            f"label {key!r}: `monotonic` must be 'increasing' or "
            f"'decreasing', got {raw!r}",
            code=ErrorCode.INVALID_INPUT,
        )
    return raw


def _validate_label_def_cross_fields(
    key: str,
    initial: str | None,
    values: list[str] | None,
    monotonic: Literal["increasing", "decreasing"] | None,
) -> None:
    """
    Enforce cross-field constraints on a :class:`LabelDef`.

    Per POLICIES.md §13:

    - ``monotonic`` requires ``values`` (no positions to
      order without them).
    - When both ``initial`` and ``values`` are declared,
      ``initial`` must be in ``values``.

    :param key: Label key, for error messages.
    :param initial: Pre-coerced initial value.
    :param values: Pre-coerced values list.
    :param monotonic: Pre-validated direction.
    :raises AgentPlaneError: On any cross-field violation.
    """
    if monotonic is not None and values is None:
        raise AgentPlaneError(
            f"label {key!r}: `monotonic` requires a `values` list "
            f"to order against",
            code=ErrorCode.INVALID_INPUT,
        )
    if initial is not None and values is not None and initial not in values:
        raise AgentPlaneError(
            f"label {key!r}: `initial` value {initial!r} is not in "
            f"declared `values` {values!r}",
            code=ErrorCode.INVALID_INPUT,
        )


def _parse_policies(
    raw: dict[str, Any] | list[Any] | None,
    *,
    expand_env: bool = True,
) -> list[PolicySpec] | None:
    """
    Parse the ``guardrails.policies:`` block.

    YAML uses a mapping keyed by policy name (preserving
    YAML declaration order, which the engine relies on per
    POLICIES.md §4). Returns a list of
    :class:`PolicySpec` instances in that order.

    :param raw: The ``policies:`` mapping, or ``None``.
    :param expand_env: Propagated to
        :func:`_parse_llm` for any PromptPolicy ``llm:``
        overrides.
    :returns: List of policy specs, or ``None`` when *raw*
        is ``None``.
    :raises AgentPlaneError: On any malformed policy entry.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AgentPlaneError(
            f"guardrails.policies: must be a mapping, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    policies: list[PolicySpec] = []
    for name, entry in raw.items():
        policies.append(
            _parse_policy_spec(str(name), entry, expand_env=expand_env),
        )
    return policies


def _parse_policy_spec(
    name: str,
    data: Any,
    *,
    expand_env: bool = True,
) -> PolicySpec:
    """
    Parse one policy's YAML block into the appropriate
    :class:`PolicySpec` subclass.

    Dispatches on the ``type:`` discriminator
    (``"function"``, ``"prompt"``, or ``"label"``).

    :param name: YAML key for this policy, used in error
        messages and recorded on the spec.
    :param data: Raw mapping from YAML (the value beneath
        ``policies.<name>:``).
    :param expand_env: Propagated for any nested ``llm:``
        connection overrides.
    :returns: A concrete ``PolicySpec`` subclass instance.
    :raises AgentPlaneError: On malformed data or unknown
        policy type.
    """
    if not isinstance(data, dict):
        raise AgentPlaneError(
            f"policy {name!r}: must be a mapping, got {type(data).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    policy_type = data.get("type")
    if policy_type is None:
        raise AgentPlaneError(
            f"policy {name!r}: missing required field `type` "
            f"(must be 'function', 'prompt', or 'label')",
            code=ErrorCode.INVALID_INPUT,
        )
    base_kwargs = _parse_policy_base_fields(name, data)
    if policy_type == "function":
        return _parse_function_policy(name, data, base_kwargs)
    if policy_type == "prompt":
        return _parse_prompt_policy(name, data, base_kwargs, expand_env=expand_env)
    if policy_type == "label":
        return _parse_label_policy(name, data, base_kwargs)
    raise AgentPlaneError(
        f"policy {name!r}: unknown type {policy_type!r} "
        f"(must be 'function', 'prompt', or 'label')",
        code=ErrorCode.INVALID_INPUT,
    )


def _parse_policy_base_fields(
    name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse the fields every policy type shares.

    Factored out of ``_parse_policy_spec`` so the dispatch
    function stays small. Fields: ``name``, ``on`` (with
    the ``[input, output]`` default per POLICIES.md §3.1),
    ``condition``, and per-policy ``ask_timeout`` override.

    :param name: Enclosing policy name.
    :param data: Raw YAML mapping for this policy.
    :returns: Kwargs dict ready to splat into any
        :class:`PolicySpec` subclass constructor.
    """
    return {
        "name": name,
        "on": _parse_on(data.get("on", ["input", "output"]), policy_name=name),
        "condition": _parse_condition(data.get("condition"), policy_name=name),
        "ask_timeout": _parse_policy_ask_timeout(
            data.get("ask_timeout"), policy_name=name,
        ),
    }


def _parse_function_policy(
    name: str,
    data: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> FunctionPolicySpec:
    """
    Parse a ``type: function`` policy block.

    :param name: Enclosing policy name (error messages +
        recorded on the spec).
    :param data: Raw YAML mapping for this policy.
    :param base_kwargs: Pre-parsed fields shared across
        policy types (``name``, ``on``, ``condition``,
        ``ask_timeout``).
    :returns: A populated :class:`FunctionPolicySpec`.
    :raises AgentPlaneError: On missing ``function:`` field
        or malformed ``action`` / ``set_labels`` values.
    """
    function_raw = data.get("function")
    if function_raw is None:
        raise AgentPlaneError(
            f"policy {name!r}: `function` policies require a `function:` field",
            code=ErrorCode.INVALID_INPUT,
        )
    action = (
        _parse_action_list(data["action"], policy_name=name)
        if "action" in data
        else None
    )
    set_labels = (
        _parse_writable_labels(data["set_labels"], policy_name=name)
        if "set_labels" in data
        else None
    )
    return FunctionPolicySpec(
        **base_kwargs,
        function=_parse_function_ref(function_raw, policy_name=name),
        action=action,
        set_labels=set_labels,
    )


def _parse_prompt_policy(
    name: str,
    data: dict[str, Any],
    base_kwargs: dict[str, Any],
    *,
    expand_env: bool,
) -> PromptPolicySpec:
    """
    Parse a ``type: prompt`` policy block.

    :param name: Enclosing policy name.
    :param data: Raw YAML mapping for this policy.
    :param base_kwargs: Pre-parsed shared fields.
    :param expand_env: Propagated to :func:`_parse_llm` for
        any nested ``llm.connection`` overrides.
    :returns: A populated :class:`PromptPolicySpec`.
    :raises AgentPlaneError: On missing / empty prompt or
        malformed ``action`` / ``set_labels`` / ``llm``.
    """
    prompt_raw = data.get("prompt")
    if not isinstance(prompt_raw, str) or not prompt_raw.strip():
        raise AgentPlaneError(
            f"policy {name!r}: `prompt` policies require a non-empty `prompt:` field",
            code=ErrorCode.INVALID_INPUT,
        )
    action = _parse_action_list(
        data.get("action", ["allow", "deny"]),
        policy_name=name,
    )
    set_labels = (
        _parse_writable_labels(data["set_labels"], policy_name=name)
        if "set_labels" in data
        else None
    )
    llm = _parse_llm(data.get("llm"), expand_env=expand_env)
    return PromptPolicySpec(
        **base_kwargs,
        prompt=prompt_raw,
        llm=llm,
        action=action,
        set_labels=set_labels,
    )


def _parse_label_policy(
    name: str,
    data: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> LabelPolicySpec:
    """
    Parse a ``type: label`` policy block.

    :param name: Enclosing policy name.
    :param data: Raw YAML mapping for this policy.
    :param base_kwargs: Pre-parsed shared fields.
    :returns: A populated :class:`LabelPolicySpec`.
    :raises AgentPlaneError: On missing / invalid ``action``
        or malformed ``set_labels`` (which on label policies
        is a ``dict[str, str]`` rather than a list).
    """
    action_raw = data.get("action")
    if action_raw is None:
        raise AgentPlaneError(
            f"policy {name!r}: `label` policies require an `action:` field "
            f"(one of allow, ask, deny)",
            code=ErrorCode.INVALID_INPUT,
        )
    try:
        action = PolicyAction(str(action_raw))
    except ValueError:
        raise AgentPlaneError(
            f"policy {name!r}: action must be one of "
            f"'allow', 'ask', 'deny', got {action_raw!r}",
            code=ErrorCode.INVALID_INPUT,
        )
    set_labels_raw = data.get("set_labels")
    label_writes: dict[str, str] | None = None
    if set_labels_raw is not None:
        if not isinstance(set_labels_raw, dict):
            raise AgentPlaneError(
                f"policy {name!r}: label-policy `set_labels` must be a "
                f"mapping of key → value, got {type(set_labels_raw).__name__}",
                code=ErrorCode.INVALID_INPUT,
            )
        label_writes = {str(k): str(v) for k, v in set_labels_raw.items()}
    return LabelPolicySpec(
        **base_kwargs,
        action=action,
        reason=data.get("reason"),
        set_labels=label_writes,
    )


def _parse_on(
    raw: Any,
    *,
    policy_name: str,
) -> list[PhaseSelector]:
    """
    Parse a policy's ``on:`` list into :class:`PhaseSelector`
    entries.

    YAML shapes:
    - ``"input"`` → wildcard selector for the INPUT phase.
    - ``"tool_call:web_search"`` → TOOL_CALL narrowed to
      one tool name.

    Tool-name narrowing is rejected on INPUT / OUTPUT phases
    (only meaningful for tool_call / tool_result).

    :param raw: The ``on:`` value from YAML. Must be a
        non-empty list of strings.
    :param policy_name: Enclosing policy name for error
        messages.
    :returns: List of :class:`PhaseSelector` entries, one
        per YAML list element.
    :raises AgentPlaneError: On empty list, unknown phase,
        or tool-narrowing on a non-tool phase.
    """
    if not isinstance(raw, list):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `on:` must be a list, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    if not raw:
        # POLICIES.md §13: empty `on:` creates a policy that
        # never fires — reject at spec load.
        raise AgentPlaneError(
            f"policy {policy_name!r}: `on:` must contain at least one "
            f"phase selector (empty list would create a policy that "
            f"never fires)",
            code=ErrorCode.INVALID_INPUT,
        )
    return [_parse_on_entry(entry, policy_name=policy_name) for entry in raw]


def _parse_on_entry(
    entry: Any,
    *,
    policy_name: str,
) -> PhaseSelector:
    """
    Parse one entry of a policy's ``on:`` list.

    Handles both forms: bare ``"<phase>"`` (wildcard) and
    ``"<phase>:<tool_name>"`` (tool-narrowed). Tool narrowing
    is rejected on phases other than TOOL_CALL / TOOL_RESULT.

    :param entry: One YAML list element — must be a string.
    :param policy_name: Enclosing policy name, used in error
        messages.
    :returns: A populated :class:`PhaseSelector`.
    :raises AgentPlaneError: On non-string entry, empty
        tool-name suffix, unknown phase, or tool narrowing
        on a non-tool phase.
    """
    if not isinstance(entry, str):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `on:` entries must be "
            f"strings, got {type(entry).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    if ":" not in entry:
        return PhaseSelector(phase=_resolve_phase(entry, entry, policy_name=policy_name))
    phase_str, tool_name = entry.split(":", 1)
    if not tool_name:
        raise AgentPlaneError(
            f"policy {policy_name!r}: empty tool name in "
            f"on-selector {entry!r}",
            code=ErrorCode.INVALID_INPUT,
        )
    phase = _resolve_phase(phase_str, entry, policy_name=policy_name)
    if phase not in (Phase.TOOL_CALL, Phase.TOOL_RESULT):
        raise AgentPlaneError(
            f"policy {policy_name!r}: phase {phase.value!r} "
            f"cannot be narrowed by tool name; tool filters "
            f"only apply to tool_call / tool_result",
            code=ErrorCode.INVALID_INPUT,
        )
    return PhaseSelector(phase=phase, tool_name=tool_name)


def _resolve_phase(
    phase_str: str,
    context: str,
    *,
    policy_name: str,
) -> Phase:
    """
    Resolve a phase-string into a :class:`Phase` enum.

    :param phase_str: The phase part of the selector
        (before any ``:``), e.g. ``"tool_call"``.
    :param context: Full on-selector value, used verbatim in
        the error message so the author can see which
        element failed, e.g. ``"tool_call:web_search"``.
    :param policy_name: Enclosing policy name, for error
        messages.
    :returns: The resolved :class:`Phase`.
    :raises AgentPlaneError: When *phase_str* is not a
        valid phase.
    """
    try:
        return Phase(phase_str)
    except ValueError:
        raise AgentPlaneError(
            f"policy {policy_name!r}: unknown phase "
            f"{phase_str!r} in {context!r}"
            if context != phase_str
            else f"policy {policy_name!r}: unknown phase {phase_str!r}",
            code=ErrorCode.INVALID_INPUT,
        )


def _parse_condition(
    raw: Any,
    *,
    policy_name: str,
) -> dict[str, str | list[str]] | None:
    """
    Parse a policy's ``condition:`` label-gate.

    Values are coerced to strings — label storage is always
    string-valued, and a YAML author writing
    ``condition: {integrity: 0}`` (unquoted int) would
    otherwise produce a silent runtime mismatch against the
    stored ``"0"``. The coercion matches omniagents parity
    for label values.

    :param raw: The ``condition:`` value from YAML, or
        ``None`` / absent.
    :param policy_name: Enclosing policy name for error
        messages.
    :returns: Dict mapping key → string value or list of
        string values. ``None`` when *raw* is absent.
    :raises AgentPlaneError: On empty dict (typo guard) or
        non-dict value.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `condition:` must be a "
            f"mapping, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    if not raw:
        # POLICIES.md §13: empty-dict typo guard. Authors who
        # want always-match should omit the field.
        raise AgentPlaneError(
            f"policy {policy_name!r}: `condition: {{}}` is rejected "
            f"(omit the field entirely for always-match; an empty "
            f"dict is almost always an unfinished edit)",
            code=ErrorCode.INVALID_INPUT,
        )
    coerced: dict[str, str | list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            coerced[str(key)] = [str(v) for v in value]
        else:
            coerced[str(key)] = str(value)
    return coerced


def _parse_action_list(
    raw: Any,
    *,
    policy_name: str,
) -> list[PolicyAction]:
    """
    Parse a policy's ``action:`` whitelist into a list of
    :class:`PolicyAction` enums.

    Accepts a bare string (single-element list sugar) or a
    list of strings. Validates each entry against the enum.

    :param raw: The ``action:`` value from YAML.
    :param policy_name: Enclosing policy name for error
        messages.
    :returns: List of :class:`PolicyAction` values.
    :raises AgentPlaneError: On empty list or unknown
        action value.
    """
    if isinstance(raw, str):
        strings = [raw]
    elif isinstance(raw, list):
        strings = [str(s) for s in raw]
    else:
        raise AgentPlaneError(
            f"policy {policy_name!r}: `action:` must be a string or "
            f"list of strings, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    if not strings:
        raise AgentPlaneError(
            f"policy {policy_name!r}: `action:` list must be non-empty",
            code=ErrorCode.INVALID_INPUT,
        )
    actions: list[PolicyAction] = []
    for s in strings:
        try:
            actions.append(PolicyAction(s))
        except ValueError:
            raise AgentPlaneError(
                f"policy {policy_name!r}: invalid action {s!r} "
                f"(must be one of 'allow', 'ask', 'deny')",
                code=ErrorCode.INVALID_INPUT,
            )
    return actions


def _parse_writable_labels(
    raw: Any,
    *,
    policy_name: str,
) -> list[str] | None:
    """
    Parse a policy's ``set_labels:`` whitelist (list form —
    used on PromptPolicy and FunctionPolicy).

    :param raw: The ``set_labels:`` list of allowed label
        keys (or ``None`` / absent).
    :param policy_name: Enclosing policy name for error
        messages.
    :returns: List of allowed label keys, or ``None`` when
        *raw* is absent.
    :raises AgentPlaneError: When *raw* is not a list.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `set_labels:` must be a list "
            f"of label keys, got {type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    return [str(k) for k in raw]


def _parse_function_ref(
    raw: Any,
    *,
    policy_name: str,
) -> FunctionRef:
    """
    Parse a ``function:`` YAML value into a :class:`FunctionRef`.

    Two accepted shapes:

    - Bare string: dotted import path of the evaluator
      callable.
    - Dict: ``{path: ..., arguments: {...}}`` — path resolves
      to a factory called with ``arguments`` kwargs at
      workflow start.

    :param raw: The raw ``function:`` value from YAML.
    :param policy_name: Enclosing policy name for error
        messages.
    :returns: A populated :class:`FunctionRef`.
    :raises AgentPlaneError: On malformed shape — non-string
        path, missing path in dict form, non-dict
        ``arguments``.
    """
    if isinstance(raw, str):
        if not raw:
            raise AgentPlaneError(
                f"policy {policy_name!r}: `function:` path must be non-empty",
                code=ErrorCode.INVALID_INPUT,
            )
        return FunctionRef(path=raw, arguments=None)
    if not isinstance(raw, dict):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `function:` must be a dotted-path "
            f"string or a dict with {{path, arguments}}, got "
            f"{type(raw).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise AgentPlaneError(
            f"policy {policy_name!r}: `function.path` must be a "
            f"non-empty dotted-path string",
            code=ErrorCode.INVALID_INPUT,
        )
    args = raw.get("arguments")
    if args is not None and not isinstance(args, dict):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `function.arguments` must be a "
            f"mapping (or omitted), got {type(args).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )
    return FunctionRef(path=path, arguments=args)


def _parse_policy_ask_timeout(
    raw: Any,
    *,
    policy_name: str,
) -> int | None:
    """
    Parse a per-policy ``ask_timeout:`` override.

    ``None`` / absent = fall back to the guardrails-level
    default. Values ``<= 0`` are rejected (POLICIES.md §13).

    :param raw: The ``ask_timeout:`` value from YAML.
    :param policy_name: Enclosing policy name for error
        messages.
    :returns: Integer override in seconds, or ``None`` when
        *raw* is absent.
    :raises AgentPlaneError: On non-integer or non-positive
        value.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise AgentPlaneError(
            f"policy {policy_name!r}: `ask_timeout` must be an integer, "
            f"got {raw!r}",
            code=ErrorCode.INVALID_INPUT,
        )
    if value <= 0:
        raise AgentPlaneError(
            f"policy {policy_name!r}: `ask_timeout` must be > 0 "
            f"(omit ASK from the policy's action list for instant-DENY)",
            code=ErrorCode.INVALID_INPUT,
        )
    return value
