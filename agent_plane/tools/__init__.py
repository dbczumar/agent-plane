"""Tools package — Tool ABC, ToolManager, and built-in tools.

Public API:
- ``Tool``: Abstract base class for all agent tools.
- ``ToolManager``: Registry-based tool dispatch for workflows.
"""

from agent_plane.tools.base import Tool
from agent_plane.tools.manager import ToolManager

__all__ = ["Tool", "ToolManager"]
