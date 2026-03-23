"""Validate an AgentSpec against the rules defined in AGENTSPEC.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_plane.spec.types import AgentSpec

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
_SKILL_NAME_MAX_LEN = 64
_SKILL_DESC_MAX_LEN = 1024
_VALID_TRANSPORTS = {"stdio", "http"}
_VALID_INPUT_MODALITIES = {"text", "image", "audio", "video", "file"}
_VALID_OUTPUT_MODALITIES = {"text", "image", "audio"}
_VALID_REASONING_EFFORTS = {"low", "medium", "high"}


@dataclass
class ValidationError:
    """A single validation issue."""

    path: str  # dot-separated location, e.g. "skills[0].name"
    message: str


@dataclass
class ValidationResult:
    """Aggregated validation outcome."""

    errors: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add(self, path: str, message: str) -> None:
        self.errors.append(ValidationError(path=path, message=message))


def validate(spec: AgentSpec) -> ValidationResult:
    """
    Validate an AgentSpec against AGENTSPEC.md rules.

    Returns a ValidationResult; check .valid to see if the spec passes.
    """
    result = ValidationResult()
    _validate_spec_version(spec, result)
    _validate_llm(spec, result)
    _validate_interaction(spec, result)
    _validate_skills(spec, result)
    _validate_mcp_servers(spec, result)
    _validate_local_tools(spec, result)
    _validate_sub_agents(spec, result)
    return result


def _validate_spec_version(spec: AgentSpec, result: ValidationResult) -> None:
    if spec.spec_version != 1:
        result.add("spec_version", f"must be 1, got {spec.spec_version}")


def _validate_llm(spec: AgentSpec, result: ValidationResult) -> None:
    if spec.llm is None:
        return
    if not spec.llm.model:
        result.add("llm.model", "must be present when llm block is present")
    if spec.llm.reasoning_effort is not None:
        if spec.llm.reasoning_effort not in _VALID_REASONING_EFFORTS:
            result.add(
                "llm.reasoning_effort",
                f"must be one of {sorted(_VALID_REASONING_EFFORTS)}, "
                f"got {spec.llm.reasoning_effort!r}",
            )


def _validate_interaction(spec: AgentSpec, result: ValidationResult) -> None:
    modalities = spec.interaction.modalities
    for m in modalities.input:
        if m not in _VALID_INPUT_MODALITIES:
            result.add(
                "interaction.modalities.input",
                f"unsupported input modality: {m!r}",
            )
    for m in modalities.output:
        if m not in _VALID_OUTPUT_MODALITIES:
            result.add(
                "interaction.modalities.output",
                f"unsupported output modality: {m!r}",
            )


def _validate_skills(spec: AgentSpec, result: ValidationResult) -> None:
    seen_names: set[str] = set()
    for i, skill in enumerate(spec.skills):
        prefix = f"skills[{i}]"
        # Name format
        if not _SKILL_NAME_PATTERN.match(skill.name):
            result.add(
                f"{prefix}.name",
                f"must match [a-z0-9-]+, got {skill.name!r}",
            )
        # Name length
        if len(skill.name) > _SKILL_NAME_MAX_LEN:
            result.add(
                f"{prefix}.name",
                f"must be at most {_SKILL_NAME_MAX_LEN} chars, got {len(skill.name)}",
            )
        # Description length
        if len(skill.description) > _SKILL_DESC_MAX_LEN:
            result.add(
                f"{prefix}.description",
                f"must be at most {_SKILL_DESC_MAX_LEN} chars, got {len(skill.description)}",
            )
        # Duplicate names
        if skill.name in seen_names:
            result.add(f"{prefix}.name", f"duplicate skill name: {skill.name!r}")
        seen_names.add(skill.name)


def _validate_mcp_servers(spec: AgentSpec, result: ValidationResult) -> None:
    seen_names: set[str] = set()
    for i, mcp in enumerate(spec.mcp_servers):
        prefix = f"mcp_servers[{i}]"
        # Transport
        if mcp.transport not in _VALID_TRANSPORTS:
            result.add(
                f"{prefix}.transport",
                f"must be 'stdio' or 'http', got {mcp.transport!r}",
            )
        # stdio requires command
        if mcp.transport == "stdio" and not mcp.command:
            result.add(f"{prefix}.command", "required for stdio transport")
        # http requires url
        if mcp.transport == "http" and not mcp.url:
            result.add(f"{prefix}.url", "required for http transport")
        # Duplicate names
        if mcp.name in seen_names:
            result.add(f"{prefix}.name", f"duplicate MCP server name: {mcp.name!r}")
        seen_names.add(mcp.name)


def _validate_local_tools(spec: AgentSpec, result: ValidationResult) -> None:
    # Check for duplicate tool names across all tool sources (MCP + local)
    all_tool_names: set[str] = set()
    for mcp in spec.mcp_servers:
        all_tool_names.add(mcp.name)
    for i, tool in enumerate(spec.local_tools):
        if tool.name in all_tool_names:
            result.add(
                f"local_tools[{i}].name",
                f"duplicate tool name: {tool.name!r}",
            )
        all_tool_names.add(tool.name)


def _validate_sub_agents(spec: AgentSpec, result: ValidationResult) -> None:
    # Every name in tools.agents must have a corresponding sub-agent
    sub_agent_names = {sa.name for sa in spec.sub_agents if sa.name is not None}
    for agent_ref in spec.tools.agents:
        if agent_ref not in sub_agent_names:
            result.add(
                "tools.agents",
                f"references sub-agent {agent_ref!r} but no matching "
                f"directory found under agents/",
            )
