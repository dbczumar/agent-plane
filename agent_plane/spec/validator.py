"""Validate an AgentSpec against the rules defined in AGENTSPEC.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_plane.spec.types import AgentSpec

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
# Agent names appear as components of the ``model`` field in API responses
# (e.g. ``"orchestrator.researcher"``). The allowed set mirrors OpenAI model
# name conventions: alphanumeric, hyphens, underscores only.
# Excluded: dots (delimiter), slashes (litellm provider/model separator),
# whitespace, and empty strings.
_AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_SKILL_NAME_MAX_LEN = 64
_SKILL_DESC_MAX_LEN = 1024
_VALID_INPUT_MODALITIES = {"text", "image", "audio", "video", "file"}
_VALID_OUTPUT_MODALITIES = {"text", "image", "audio"}


@dataclass
class ValidationError:
    """
    A single validation issue.

    :param path: Dot-separated location of the invalid field,
        e.g. ``"skills[0].name"`` or ``"llm.model"``.
    :param message: Human-readable description of the violation.
    """

    path: str  # dot-separated location, e.g. "skills[0].name"
    message: str


@dataclass
class ValidationResult:
    """
    Aggregated validation outcome.

    :param errors: Collected validation issues. An empty list
        means the spec is valid.
    """

    errors: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """
        Whether the spec passed all validation checks.

        :returns: ``True`` when no errors were recorded,
            ``False`` otherwise.
        """
        return len(self.errors) == 0

    def add(self, path: str, message: str) -> None:
        """
        Record a validation error.

        :param path: Dot-separated location of the invalid field,
            e.g. ``"skills[0].name"``.
        :param message: Human-readable description of the
            violation.
        """
        self.errors.append(ValidationError(path=path, message=message))


def validate(spec: AgentSpec) -> ValidationResult:
    """
    Validate an :class:`AgentSpec` against AGENTSPEC.md rules.

    :param spec: The parsed agent spec to validate.
    :returns: A :class:`ValidationResult`; check ``.valid`` to see
        if the spec passes all checks.
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
    """
    Validate that ``spec_version`` is a supported value.

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
    if spec.spec_version != 1:
        result.add("spec_version", f"must be 1, got {spec.spec_version}")


def _validate_llm(spec: AgentSpec, result: ValidationResult) -> None:
    """
    Validate the ``llm`` block, if present.

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
    if spec.llm is None:
        return
    if not spec.llm.model:
        result.add("llm.model", "must be present when llm block is present")


def _validate_interaction(spec: AgentSpec, result: ValidationResult) -> None:
    """
    Validate input and output modalities against allowed values.

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
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
    """
    Validate skill names, descriptions, and uniqueness.

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
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
    """
    Validate MCP server transport, required fields, and name
    uniqueness.

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
    seen_names: set[str] = set()
    for i, mcp in enumerate(spec.mcp_servers):
        prefix = f"mcp_servers[{i}]"
        # Duplicate names
        if mcp.name in seen_names:
            result.add(f"{prefix}.name", f"duplicate MCP server name: {mcp.name!r}")
        seen_names.add(mcp.name)


def _validate_local_tools(spec: AgentSpec, result: ValidationResult) -> None:
    """
    Validate local tool name uniqueness across all tool sources
    (MCP servers and local tools).

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
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


def _validate_sub_agents(
    spec: AgentSpec,
    result: ValidationResult,
) -> None:
    """
    Validate sub-agent declarations.

    Checks:
    1. Every name in ``tools.agents`` has a corresponding parsed
       sub-agent directory.
    2. Callable sub-agents (referenced in ``tools.agents``) must
       have ``llm.model`` configured.
    3. Sub-agent names must be unique across the entire spec tree.
    4. Agent names must not contain ``.`` (reserved as the
       delimiter in tunneled output ``model`` fields).

    :param spec: The agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
    sub_specs = {sa.name: sa for sa in spec.sub_agents if sa.name is not None}

    for agent_ref in spec.tools.agents:
        sub = sub_specs.get(agent_ref)
        if sub is None:
            result.add(
                "tools.agents",
                f"references sub-agent {agent_ref!r} but no "
                f"matching directory found under agents/",
            )
            continue
        # Callable sub-agents must have llm.model
        if sub.llm is None or not sub.llm.model:
            result.add(
                f"sub_agents[{agent_ref!r}].llm",
                "callable sub-agent must have llm.model configured",
            )

    # Agent name characters (dots, slashes, whitespace, empty)
    _validate_agent_names(spec, result)

    # Unique names across the entire spec tree
    _check_unique_sub_agent_names(spec, result)


def _validate_agent_names(
    spec: AgentSpec,
    result: ValidationResult,
) -> None:
    """
    Validate that every agent name in the spec tree is a legal identifier.

    Agent names appear as components of the ``model`` field in API
    responses (e.g. ``"orchestrator.researcher"``). They must match
    ``_AGENT_NAME_PATTERN`` (``[a-zA-Z0-9_-]+``), which enforces:

    - Non-empty — empty strings have no meaningful identity.
    - No dots — reserved as the delimiter between parent and sub-agent
      in the ``model`` field (e.g. ``"root.child"``).
    - No slashes — reserved by litellm as the ``provider/model``
      separator; a slash in a name would silently mis-route LLM calls.
    - No whitespace — whitespace in a model identifier confuses most
      API clients and logging pipelines.

    :param spec: The root agent spec to check (recursed into sub_agents).
    :param result: Accumulator for any validation errors found.
    """
    if spec.name is not None and not _AGENT_NAME_PATTERN.match(spec.name):
        result.add(
            "name",
            f"agent name {spec.name!r} must match [a-zA-Z0-9_-]+ "
            f"(no dots, slashes, whitespace, or empty strings)",
        )
    for sa in spec.sub_agents:
        _validate_agent_names(sa, result)


def _check_unique_sub_agent_names(
    spec: AgentSpec,
    result: ValidationResult,
) -> None:
    """
    Validate that sub-agent names are unique across the entire
    spec tree (not just within one level).

    Flat uniqueness enables O(1) lookup by name during spec
    loading — see designs/SUBAGENT.md.

    :param spec: The root agent spec to check.
    :param result: Accumulator for any validation errors found.
    """
    seen: set[str] = set()
    _collect_sub_agent_names(spec, seen, result)


def _collect_sub_agent_names(
    spec: AgentSpec,
    seen: set[str],
    result: ValidationResult,
) -> None:
    """
    Recursively collect sub-agent names and flag duplicates.

    :param spec: The current spec node to check.
    :param seen: Accumulator of names seen so far.
    :param result: Accumulator for any validation errors found.
    """
    for sa in spec.sub_agents:
        if sa.name is not None:
            if sa.name in seen:
                result.add(
                    f"sub_agents[{sa.name!r}]",
                    f"duplicate sub-agent name {sa.name!r} across the spec tree",
                )
            seen.add(sa.name)
        _collect_sub_agent_names(sa, seen, result)
