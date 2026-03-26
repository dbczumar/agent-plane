"""Parse an agent image directory into an AgentSpec."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from agent_plane.spec.types import (
    AgentSpec,
    ExecutionConfig,
    InteractionConfig,
    LLMConfig,
    LocalToolInfo,
    MCPServerConfig,
    ModalityConfig,
    RetryConfig,
    SkillSpec,
    ToolsConfig,
)

# Pattern for SKILL.md YAML frontmatter delimited by ---
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)", re.DOTALL)


def parse(root: Path) -> AgentSpec:
    """
    Parse an agent image directory into an :class:`AgentSpec`.

    :param root: Path to the agent image directory. Must contain
        ``config.yaml``.
    :returns: A fully populated :class:`AgentSpec` (not yet
        validated).
    :raises FileNotFoundError: If ``config.yaml`` is missing.
    :raises ValueError: If ``config.yaml`` is not valid YAML or has
        structural issues.
    """
    config_path = root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {root}")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config.yaml must be a YAML mapping, got {type(raw).__name__}")

    spec_version = raw.get("spec_version")
    if spec_version is None:
        raise ValueError("config.yaml missing required field: spec_version")

    llm = _parse_llm(raw.get("llm"))
    interaction = _parse_interaction(raw.get("interaction"))
    tools_config = _parse_tools_config(raw.get("tools"))
    execution = _parse_execution(raw.get("execution"))
    params = raw.get("params", {})

    instructions = _resolve_instructions(root, raw.get("instructions"))
    skills = _discover_skills(root / "skills")
    mcp_servers = _discover_mcp_servers(root / "tools" / "mcp")
    local_tools = _discover_local_tools(root / "tools")
    sub_agents = _discover_sub_agents(root / "agents")

    return AgentSpec(
        spec_version=spec_version,
        name=raw.get("name"),
        description=raw.get("description"),
        llm=llm,
        interaction=interaction,
        tools=tools_config,
        execution=execution,
        params=params,
        instructions=instructions,
        skills=skills,
        mcp_servers=mcp_servers,
        local_tools=local_tools,
        sub_agents=sub_agents,
    )


def _parse_llm(raw: dict[str, object] | None) -> LLMConfig | None:
    """
    Parse the ``llm:`` block from config.yaml into an
    :class:`LLMConfig`.

    :param raw: The raw ``llm:`` mapping from config.yaml, or
        ``None`` if the block was absent. Example:
        ``{"model": "openai/gpt-4o", "temperature": 0.7}``.
    :returns: A populated :class:`LLMConfig`, or ``None`` when
        the ``llm:`` block is absent.
    :raises ValueError: If the ``llm:`` block is present but
        missing the required ``model`` field.
    """
    if raw is None:
        return None
    model = raw.get("model")
    if model is None:
        raise ValueError("llm block present but missing required field: model")
    # ``connection``, ``timeout``, and ``retry`` are separated into
    # their own typed fields; everything else is passed through to
    # the LLM SDK as extra kwargs.
    connection_raw = raw.get("connection")
    connection: dict[str, str] | None = None
    if isinstance(connection_raw, dict):
        # Expand ${VAR} references so api_key: ${OPENAI_API_KEY} works.
        connection = _expand_env_vars({str(k): str(v) for k, v in connection_raw.items()})
    timeout = int(raw["timeout"]) if "timeout" in raw else 300
    retry = _parse_retry(raw.get("retry"))
    reserved = {"model", "connection", "timeout", "retry"}
    extra = {k: v for k, v in raw.items() if k not in reserved}
    return LLMConfig(
        model=str(model),
        extra=extra,
        connection=connection,
        timeout=timeout,
        retry=retry,
    )


def _parse_interaction(
    raw: dict[str, object] | None,
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
    raw: dict[str, object] | None,
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
    return ToolsConfig(
        agents=raw.get("agents", []),  # type: ignore[arg-type]
        timeout=timeout,
        retry=retry,
    )


def _parse_retry(
    raw: dict[str, object] | None,
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


def _parse_execution(
    raw: dict[str, object] | None,
) -> ExecutionConfig:
    """
    Parse the ``execution:`` block into an :class:`ExecutionConfig`.

    Returns defaults when *raw* is ``None``.

    :param raw: The raw ``execution:`` mapping, or ``None`` if
        absent. Example: ``{"timeout": 3600, "max_iterations": 500}``.
    :returns: A populated :class:`ExecutionConfig`.
    """
    if raw is None:
        return ExecutionConfig()
    return ExecutionConfig(
        timeout=int(raw.get("timeout", 3600)),
        max_iterations=int(raw.get("max_iterations", 1000)),
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
    :raises ValueError: If the frontmatter is missing, malformed,
        or lacks required fields.
    """
    text = skill_md.read_text()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"SKILL.md missing YAML frontmatter: {skill_md}")
    frontmatter_str, content = match.groups()
    frontmatter = yaml.safe_load(frontmatter_str)
    if not isinstance(frontmatter, dict):
        raise ValueError(f"SKILL.md frontmatter must be a YAML mapping: {skill_md}")
    name = frontmatter.get("name")
    if name is None:
        raise ValueError(f"SKILL.md frontmatter missing required field 'name': {skill_md}")
    description = frontmatter.get("description")
    if description is None:
        raise ValueError(f"SKILL.md frontmatter missing required field 'description': {skill_md}")
    return SkillSpec(
        name=str(name),
        description=str(description),
        content=content.strip(),
        skill_dir=skill_md.parent,
    )


def _expand_env_vars(
    mapping: dict[str, str],
) -> dict[str, str]:
    """
    Expand ``${VAR}`` and ``$VAR`` references in dict values
    against the current process environment.

    Uses :func:`os.path.expandvars`, which leaves unresolved
    references as-is (e.g. ``${MISSING}`` stays literal if
    the variable is not set).

    :param mapping: A string-to-string dict, e.g.
        ``{"TOKEN": "${GITHUB_TOKEN}"}``.
    :returns: A new dict with expanded values.
    """
    return {key: os.path.expandvars(value) for key, value in mapping.items()}


def _discover_mcp_servers(
    mcp_dir: Path,
) -> list[MCPServerConfig]:
    """
    Discover and parse all MCP server configs under
    ``tools/mcp/``.

    Each ``.yaml`` file in the directory is parsed into an
    :class:`MCPServerConfig`.

    :param mcp_dir: Path to the ``tools/mcp/`` directory, e.g.
        ``root / "tools" / "mcp"``.
    :returns: A sorted list of parsed :class:`MCPServerConfig`
        objects. Returns an empty list if *mcp_dir* does not
        exist.
    :raises ValueError: If any YAML file is malformed or missing
        required fields (``name``, ``transport``).
    """
    if not mcp_dir.is_dir():
        return []
    servers: list[MCPServerConfig] = []
    for yaml_file in sorted(mcp_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_file.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"MCP config must be a YAML mapping: {yaml_file}")
        name = raw.get("name")
        if name is None:
            raise ValueError(f"MCP config missing required field 'name': {yaml_file}")
        transport = raw.get("transport")
        if transport is None:
            raise ValueError(f"MCP config missing required field 'transport': {yaml_file}")
        servers.append(
            MCPServerConfig(
                name=str(name),
                transport=str(transport),
                description=raw.get("description"),
                command=raw.get("command"),
                args=raw.get("args", []),
                env=_expand_env_vars(raw.get("env", {})),
                url=raw.get("url"),
                headers=_expand_env_vars(raw.get("headers", {})),
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

    Tool names are derived from the file stem by replacing
    underscores with dots (e.g. ``arxiv_search.py`` becomes
    ``"arxiv.search"``).

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
            # Derive tool name: arxiv_search.py -> arxiv.search
            stem = tool_file.stem
            tool_name = stem.replace("_", ".")
            rel_path = str(tool_file.relative_to(tools_dir.parent))
            tools.append(LocalToolInfo(name=tool_name, path=rel_path, language=language))
    return tools


def _discover_sub_agents(agents_dir: Path) -> list[AgentSpec]:
    """
    Recursively discover and parse sub-agents under ``agents/``.

    Each subdirectory containing a ``config.yaml`` is parsed via
    :func:`parse`, producing a nested :class:`AgentSpec`.

    :param agents_dir: Path to the ``agents/`` directory, e.g.
        ``root / "agents"``.
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
        # Recursive parse
        sub_agents.append(parse(agent_dir))
    return sub_agents
