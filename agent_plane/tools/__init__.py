"""Tools package — Tool ABC, ToolManager, and built-in tools.

Public API:
- ``Tool``: Abstract base class for all agent tools.
- ``ToolManager``: Registry-based tool dispatch for workflows.
- ``CallbackTool``: A tool that executes via HTTP callback.
- ``CallbackToolSpec``: Configuration for a callback tool.
"""

from agent_plane.tools.base import Tool
from agent_plane.tools.client_specified import CallbackTool, CallbackToolSpec
from agent_plane.tools.manager import ToolManager

__all__ = ["CallbackTool", "CallbackToolSpec", "Tool", "ToolManager"]
