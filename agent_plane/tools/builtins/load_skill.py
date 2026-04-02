"""Built-in tool: load a skill's instructions by name."""

from __future__ import annotations

import json
from typing import Any

from agent_plane.spec.types import SkillSpec
from agent_plane.tools.base import Tool, ToolContext


class LoadSkillTool(Tool):
    """
    Built-in tool that loads a skill's full instructions by name.

    Looks up the skill in the agent spec, returns the skill
    content with an optional resource file listing appended.

    :param skills: The agent's parsed skill list.
    """

    def __init__(self, skills: list[SkillSpec]) -> None:
        """
        Initialize with the agent's skill list.

        :param skills: Parsed skills from the agent spec, e.g.
            ``[SkillSpec(name="code-review", ...)]``.
        """
        self._skills = skills
        self._skills_by_name: dict[str, SkillSpec] = {s.name: s for s in skills}

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"load_skill"``.
        """
        return "load_skill"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format schema for ``load_skill``.

        The description includes the list of available skill
        names so the LLM knows what it can load.

        :returns: A tool schema dict.
        """
        skill_names = [s.name for s in self._skills]
        return {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": (
                    "Load a skill's full instructions by "
                    "name. Available skills: "
                    f"{', '.join(skill_names)}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": ("The skill name to load"),
                        },
                    },
                    "required": ["name"],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Look up a skill by name and return its content.

        If the skill has bundled resource files, appends
        a listing of available files to the content.

        :param arguments: JSON with ``"name"`` key, e.g.
            ``'{"name": "code-review"}'``.
        :param ctx: Server-side execution context (unused by
            skill tools, required by the :class:`Tool` interface).
        :returns: The skill content string, or an error
            message if the skill is not found.
        """
        args: dict[str, str] = json.loads(arguments)
        skill_name = args.get("name")
        if skill_name is None:
            return "Error: missing required 'name' argument"
        skill = self._skills_by_name.get(skill_name)
        if skill is None:
            available = list(self._skills_by_name.keys())
            return f"Error: skill {skill_name!r} not found. Available skills: {available}"
        resources = list_skill_resources(skill)
        return _format_skill_content(skill, resources)


def list_skill_resources(skill: SkillSpec) -> list[str]:
    """
    List resource files in a skill's directory.

    Scans ``references/``, ``scripts/``, and ``assets/``
    subdirectories. Returns relative paths suitable for
    ``read_skill_file``.

    :param skill: The skill to scan.
    :returns: Sorted list of relative path strings, e.g.
        ``["references/style-guide.md"]``. Empty if the
        skill has no ``skill_dir`` or no resource files.
    """
    if skill.skill_dir is None:
        return []
    files: list[str] = []
    for subdir_name in ("references", "scripts", "assets"):
        subdir = skill.skill_dir / subdir_name
        if not subdir.is_dir():
            continue
        for fp in sorted(subdir.rglob("*")):
            if fp.is_file():
                rel = str(fp.relative_to(skill.skill_dir))
                files.append(rel)
    return files


def _format_skill_content(
    skill: SkillSpec,
    resource_files: list[str],
) -> str:
    """
    Format the skill content for the LLM, appending a
    resource listing if the skill has bundled files.

    :param skill: The skill to format.
    :param resource_files: List of relative paths to
        bundled resource files, e.g.
        ``["references/style-guide.md"]``.
    :returns: The skill content, optionally followed by
        an ``## Available files`` section.
    """
    if not resource_files:
        return skill.content

    lines = [
        skill.content,
        "",
        "## Available files",
        "Use the read_skill_file tool to read these:",
    ]
    for path in resource_files:
        lines.append(f"- {path}")
    return "\n".join(lines)
