"""Verify backward-compatible re-exports from tool_manager.

The actual tool tests live in tests/tools/. This file only
checks that the old import path still works.
"""

from __future__ import annotations


def test_reexports_are_importable() -> None:
    """
    The backward-compat re-exports in
    ``agent_plane.runtime.tool_manager`` resolve to the
    real classes in ``agent_plane.tools``.
    """
    from agent_plane.runtime.tool_manager import (  # noqa: F401
        LoadSkillTool,
        ReadSkillFileTool,
        Tool,
        ToolManager,
    )
    from agent_plane.tools import Tool as RealTool
    from agent_plane.tools import ToolManager as RealManager
    from agent_plane.tools.builtins import (
        LoadSkillTool as RealLoad,
    )
    from agent_plane.tools.builtins import (
        ReadSkillFileTool as RealRead,
    )

    assert Tool is RealTool
    assert ToolManager is RealManager
    assert LoadSkillTool is RealLoad
    assert ReadSkillFileTool is RealRead
