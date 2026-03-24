"""Parse an agent image directory into an AgentSpec."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from agent_plane.spec.types import (
    AgentSpec,
    InteractionConfig,
    LLMConfig,
    LocalToolInfo,
    MCPServerConfig,
    ModalityConfig,
    SkillSpec,
    ToolsConfig,
)

# Pattern for SKILL.md YAML frontmatter delimited by ---
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)", re.DOTALL)


def parse(root: Path) -> AgentSpec:
    """
    Parse an agent image directory into an AgentSpec.

    Args:
        root: Path to the agent image directory. Must contain config.yaml.

    Returns:
        A fully populated AgentSpec (not yet validated).

    Raises:
        FileNotFoundError: If config.yaml is missing.
        ValueError: If config.yaml is not valid YAML or has structural issues.
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
        params=params,
        instructions=instructions,
        skills=skills,
        mcp_servers=mcp_servers,
        local_tools=local_tools,
        sub_agents=sub_agents,
    )


def _parse_llm(raw: dict[str, object] | None) -> LLMConfig | None:
    if raw is None:
        return None
    model = raw.get("model")
    if model is None:
        raise ValueError("llm block present but missing required field: model")
    # Everything except ``model`` is passed through to litellm as-is.
    extra = {k: v for k, v in raw.items() if k != "model"}
    return LLMConfig(model=str(model), extra=extra)


def _parse_interaction(raw: dict[str, object] | None) -> InteractionConfig:
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


def _parse_tools_config(raw: dict[str, object] | None) -> ToolsConfig:
    if raw is None:
        return ToolsConfig()
    return ToolsConfig(
        agents=raw.get("agents", []),  # type: ignore[arg-type]
    )


def _resolve_instructions(root: Path, raw_value: object) -> str | None:
    """
    Resolve the instructions for an agent image.

    - If ``instructions`` is set in config.yaml and the value is a path to an
      existing file relative to root, read that file.
    - If ``instructions`` is set but is not a file path, treat as inline text.
    - If ``instructions`` is not set, fall back to reading AGENTS.md.
    """
    if raw_value is not None:
        text = str(raw_value)
        candidate = root / text
        if candidate.is_file():
            return candidate.read_text()
        return text
    # Default: read AGENTS.md if present
    agents_md = root / "AGENTS.md"
    if agents_md.exists():
        return agents_md.read_text()
    return None


def _discover_skills(skills_dir: Path) -> list[SkillSpec]:
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
        allowed_tools=frontmatter.get("allowed_tools", []),
    )


def _discover_mcp_servers(mcp_dir: Path) -> list[MCPServerConfig]:
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
                env=raw.get("env", {}),
                url=raw.get("url"),
                headers=raw.get("headers", {}),
            )
        )
    return servers


def _discover_local_tools(tools_dir: Path) -> list[LocalToolInfo]:
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
