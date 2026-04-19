"""Tools package — Tool ABC, ToolManager, decorator, and built-in tools.

Public API (lazy):

- ``Tool``: Abstract base class for class-based tools.
- ``ToolManager``: Registry-based tool dispatch for workflows.
- ``ClientSideTool``: A tool presented to the LLM but executed by the caller.
- ``ClientSideToolSpec``: Configuration for a client-side tool.
- ``LocalPythonTool``: A tool backed by a local Python file in the agent image.
- ``tool``: Decorator for authoring custom Python tools as typed functions.
- ``ToolMetadata``: The metadata object the decorator attaches.

Imports are lazy so that loading ``agent_plane.tools`` does not pull
in heavy submodules (``manager`` transitively imports ``mcp``, which
clashes with the upstream ``mcp`` pip package when a custom tool
file is reloaded inside the subprocess runner). The Pythonic
``module.__getattr__`` hook resolves names on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-checker view: real names with their actual types.
    from agent_plane.tools.base import Tool, ToolContext
    from agent_plane.tools.client_specified import (
        ClientSideTool,
        ClientSideToolSpec,
    )
    from agent_plane.tools.decorator import ToolMetadata, tool
    from agent_plane.tools.local import LocalPythonTool
    from agent_plane.tools.manager import ToolManager


# Map exported name → (submodule path, attribute name).
# Add an entry here when adding a new public re-export.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Tool": ("agent_plane.tools.base", "Tool"),
    "ToolContext": ("agent_plane.tools.base", "ToolContext"),
    "ClientSideTool": ("agent_plane.tools.client_specified", "ClientSideTool"),
    "ClientSideToolSpec": (
        "agent_plane.tools.client_specified",
        "ClientSideToolSpec",
    ),
    "ToolMetadata": ("agent_plane.tools.decorator", "ToolMetadata"),
    "tool": ("agent_plane.tools.decorator", "tool"),
    "LocalPythonTool": ("agent_plane.tools.local", "LocalPythonTool"),
    "ToolManager": ("agent_plane.tools.manager", "ToolManager"),
}


def __getattr__(name: str) -> Any:
    """
    Resolve public re-exports lazily on attribute access.

    Importing one symbol from this package no longer drags in
    every submodule. In particular, subprocess-loaded tool files
    can ``from agent_plane.tools import tool`` without triggering
    the ``ToolManager`` → ``mcp`` import chain (which conflicts
    with the upstream ``mcp`` pip package in subprocess
    environments — see ``list_builtin_tools.py`` for context).

    :param name: The attribute name being accessed.
    :returns: The resolved attribute from the appropriate
        submodule.
    :raises AttributeError: If ``name`` is not in
        :data:`_LAZY_EXPORTS`.
    """
    if name in _LAZY_EXPORTS:
        import importlib

        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClientSideTool",
    "ClientSideToolSpec",
    "LocalPythonTool",
    "Tool",
    "ToolContext",
    "ToolManager",
    "ToolMetadata",
    "tool",
]
