"""Built-in tools for agent-plane skills.

Public API:
- ``LoadSkillTool``: Loads a skill's instructions by name.
- ``ReadSkillFileTool``: Reads files from a skill's directory.
- ``any_skill_has_resources``: Checks if any skill has bundled
  resource files (used by ToolManager to decide whether to
  register ReadSkillFileTool).
- ``list_skill_resources``: Lists resource files in a skill's
  directory (used by LoadSkillTool to append file listings).
"""

from __future__ import annotations

from agent_plane.spec.types import SkillSpec
from agent_plane.tools.builtins.load_skill import (
    LoadSkillTool,
    list_skill_resources,
)
from agent_plane.tools.builtins.read_skill_file import (
    ReadSkillFileTool,
)

__all__ = [
    "LoadSkillTool",
    "ReadSkillFileTool",
    "any_skill_has_resources",
    "list_skill_resources",
]


def any_skill_has_resources(
    skills: list[SkillSpec],
) -> bool:
    """
    Check whether any skill has bundled resource files.

    :param skills: The agent's skill list, e.g.
        ``[SkillSpec(name="code-review", ...)]``.
    :returns: ``True`` if at least one skill has a
        ``skill_dir`` with files in references/, scripts/,
        or assets/.
    """
    return any(list_skill_resources(s) for s in skills)
