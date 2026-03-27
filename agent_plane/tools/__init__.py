"""Tools package — Tool ABC, ToolManager, and built-in tools.

Public API:
- ``Tool``: Abstract base class for all agent tools.
- ``ToolManager``: Registry-based tool dispatch for workflows.
- ``ClientSideTool``: A tool presented to the LLM but executed by the caller.
- ``ClientSideToolSpec``: Configuration for a client-side tool.
"""

from agent_plane.tools.base import Tool
from agent_plane.tools.client_specified import ClientSideTool, ClientSideToolSpec
from agent_plane.tools.manager import ToolManager

__all__ = ["ClientSideTool", "ClientSideToolSpec", "Tool", "ToolManager"]
