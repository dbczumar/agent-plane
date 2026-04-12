"""Built-in tool: introspect — browse the agent's own spec.

Lets an agent examine its own configuration progressively. With no
arguments, returns a high-level summary. With a ``section`` parameter,
drills into specific parts (skills, instructions, sub-agents, tools).

Usage in config.yaml::

    tools:
      builtins:
        - introspect
"""

from __future__ import annotations

import json

# Any: tool schemas are heterogeneous dicts.
from typing import Any

from agent_plane.spec.types import AgentSpec
from agent_plane.tools.base import Tool, ToolContext


class IntrospectTool(Tool):
    """
    Browse the agent's own spec for self-debugging and self-description.

    With no ``section`` argument, returns a high-level summary.
    With a ``section``, drills into a specific part of the spec.

    :param spec: The agent's parsed AgentSpec.
    """

    def __init__(self, spec: AgentSpec) -> None:
        """
        Create an introspect tool.

        :param spec: The agent's own AgentSpec.
        """
        self._spec = spec

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"introspect"``.
        """
        return "introspect"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema.

        :returns: The schema dict.
        """
        return {
            "type": "function",
            "function": {
                "name": "introspect",
                "description": (
                    "Examine your own agent configuration. With no "
                    "arguments, returns a summary of your name, model, "
                    "tools, skills, and sub-agents. With a section "
                    "parameter, drills into a specific part (e.g. "
                    "'skills/deep-research', 'sub_agents/researcher', "
                    "'instructions')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": (
                                "Path to a specific part of the spec. "
                                "Examples: 'instructions', 'skills', "
                                "'skills/<name>', 'tools', 'sub_agents', "
                                "'sub_agents/<name>/instructions', 'config'. "
                                "Omit for a high-level summary."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Return spec information for the requested section.

        :param arguments: JSON with optional ``section`` key.
        :param ctx: Tool execution context (unused).
        :returns: Formatted spec information.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        section = parsed.get("section", "").strip()

        if not section:
            return _format_summary(self._spec)
        return _resolve_section(self._spec, section)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _format_summary(spec: AgentSpec) -> str:
    """
    Build a high-level summary of the agent spec.

    :param spec: The agent's spec.
    :returns: Human-readable summary.
    """
    lines: list[str] = []
    lines.append(f"Agent: {spec.name or '(unnamed)'}")
    if spec.description:
        lines.append(f"Description: {spec.description}")
    if spec.llm:
        lines.append(f"Model: {spec.llm.model}")
    lines.append(f"Executor: {spec.executor.type or 'llm'}")
    lines.append("")

    _append_tools(lines, spec)
    _append_skills(lines, spec)
    _append_interaction(lines, spec)

    lines.append('Use introspect(section="...") to drill into any section.')
    return "\n".join(lines)


def _append_tools(lines: list[str], spec: AgentSpec) -> None:
    """
    Append the tools section to the summary.

    :param lines: Mutable list of output lines.
    :param spec: The agent's spec.
    """
    lines.append("Tools:")
    if spec.tools.builtins:
        names = [b.name for b in spec.tools.builtins]
        lines.append(f"  builtins: {', '.join(names)}")
    if spec.tools.agents:
        lines.append(f"  sub-agents: {', '.join(spec.tools.agents)}")
    if spec.mcp_servers:
        for mcp in spec.mcp_servers:
            lines.append(f"  mcp: {mcp.name}")
    if spec.local_tools:
        names = [t.name for t in spec.local_tools]
        lines.append(f"  local: {', '.join(names)}")
    if not (spec.tools.builtins or spec.tools.agents or spec.mcp_servers or spec.local_tools):
        lines.append("  (none)")
    lines.append("")


def _append_skills(lines: list[str], spec: AgentSpec) -> None:
    """
    Append the skills section to the summary.

    :param lines: Mutable list of output lines.
    :param spec: The agent's spec.
    """
    if spec.skills:
        names = [s.name for s in spec.skills]
        lines.append(f"Skills: {', '.join(names)}")
    else:
        lines.append("Skills: (none)")
    lines.append("")


def _append_interaction(lines: list[str], spec: AgentSpec) -> None:
    """
    Append the interaction section to the summary.

    :param lines: Mutable list of output lines.
    :param spec: The agent's spec.
    """
    lines.append("Interaction:")
    lines.append(f"  conversational: {spec.interaction.conversational}")
    inp = ", ".join(spec.interaction.modalities.input)
    out = ", ".join(spec.interaction.modalities.output)
    lines.append(f"  input: {inp}")
    lines.append(f"  output: {out}")
    lines.append("")


# ---------------------------------------------------------------------------
# Section resolution
# ---------------------------------------------------------------------------


def _resolve_section(spec: AgentSpec, section: str) -> str:
    """
    Resolve a section path against the spec tree.

    :param spec: The agent's spec.
    :param section: Slash-separated section path.
    :returns: Formatted content for the section.
    """
    parts = [p for p in section.split("/") if p]
    if not parts:
        return _format_summary(spec)

    root = parts[0]

    if root == "instructions":
        return spec.instructions or "(no instructions set)"

    if root == "config":
        return _format_config(spec)

    if root == "skills":
        return _resolve_skills(spec, parts[1:])

    if root == "tools":
        return _format_tools(spec)

    if root == "sub_agents":
        return _resolve_sub_agents(spec, parts[1:])

    valid = "instructions, config, skills, tools, sub_agents"
    return f"Unknown section '{root}'. Valid sections: {valid}"


def _format_config(spec: AgentSpec) -> str:
    """
    Return a YAML-like representation of the spec's key fields.

    :param spec: The agent's spec.
    :returns: Config summary.
    """
    lines: list[str] = [f"spec_version: {spec.spec_version}"]
    if spec.name:
        lines.append(f"name: {spec.name}")
    if spec.description:
        lines.append(f"description: {spec.description}")
    if spec.llm:
        lines.append("llm:")
        lines.append(f"  model: {spec.llm.model}")
    lines.append("executor:")
    lines.append(f"  type: {spec.executor.type or 'llm'}")
    return "\n".join(lines)


def _resolve_skills(spec: AgentSpec, rest: list[str]) -> str:
    """
    Resolve a skills section path.

    :param spec: The agent's spec.
    :param rest: Remaining path parts after ``skills``.
    :returns: Skills list or specific skill content.
    """
    if not rest:
        if not spec.skills:
            return "No skills configured."
        lines: list[str] = []
        for s in spec.skills:
            lines.append(f"- {s.name}: {s.description or '(no description)'}")
        return "\n".join(lines)

    skill_name = rest[0]
    skill = next((s for s in spec.skills if s.name == skill_name), None)
    if skill is None:
        available = ", ".join(s.name for s in spec.skills) or "(none)"
        return f"Skill '{skill_name}' not found. Available: {available}"

    header = f"Skill: {skill.name}\nDescription: {skill.description or ''}\n\n"
    return header + skill.content


def _format_tools(spec: AgentSpec) -> str:
    """
    Return detailed tool information.

    :param spec: The agent's spec.
    :returns: Tool details.
    """
    lines: list[str] = []

    if spec.tools.builtins:
        lines.append("Built-in tools:")
        for b in spec.tools.builtins:
            config_str = f" (config: {b.config})" if b.config else ""
            lines.append(f"  - {b.name}{config_str}")

    if spec.tools.agents:
        lines.append("\nSub-agent tools:")
        for name in spec.tools.agents:
            lines.append(f"  - {name}")

    if spec.mcp_servers:
        lines.append("\nMCP servers:")
        for mcp in spec.mcp_servers:
            lines.append(f"  - {mcp.name}: {mcp.url}")

    if spec.local_tools:
        lines.append("\nLocal tools:")
        for t in spec.local_tools:
            lines.append(f"  - {t.name} ({t.language}): {t.path}")

    if not lines:
        return "No tools configured."
    return "\n".join(lines)


def _resolve_sub_agents(spec: AgentSpec, rest: list[str]) -> str:
    """
    Resolve a sub_agents section path.

    :param spec: The agent's spec.
    :param rest: Remaining path parts after ``sub_agents``.
    :returns: Sub-agent list, summary, or drilled content.
    """
    if not rest:
        if not spec.sub_agents:
            return "No sub-agents configured."
        lines: list[str] = []
        for sa in spec.sub_agents:
            lines.append(f"- {sa.name}: {sa.description or '(no description)'}")
        return "\n".join(lines)

    sa_name = rest[0]
    matched: AgentSpec | None = next(
        (s for s in spec.sub_agents if s.name == sa_name),
        None,
    )
    if matched is None:
        available = ", ".join(s.name or "?" for s in spec.sub_agents) or "(none)"
        return f"Sub-agent '{sa_name}' not found. Available: {available}"

    # Drill further into the sub-agent's spec.
    remaining = rest[1:]
    if not remaining:
        return _format_summary(matched)
    return _resolve_section(matched, "/".join(remaining))
