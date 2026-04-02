"""
Registry of client-side tool sets for the TUI.

Each tool set is a module in this package that exports:

- ``TOOLS``: List of OpenAI-format tool schema dicts.
- ``execute_tool(name, arguments) -> str``: Local execution function.

Usage::

    from tool_sets import get_tool_set
    tool_set = get_tool_set("coder")
    schemas = tool_set.TOOLS
    result = tool_set.execute_tool("Read", {"file_path": "/tmp/foo.py"})
"""

from __future__ import annotations

import importlib
from types import ModuleType


def get_tool_set(name: str) -> ModuleType:
    """
    Load a tool set module by name.

    :param name: Tool set name, e.g. ``"coder"``. Must correspond
        to a module in this package (``tool_sets.coder``).
    :returns: The tool set module with ``TOOLS`` and ``execute_tool``.
    :raises SystemExit: If the tool set is not found.
    """
    try:
        return importlib.import_module(f".{name}", package=__name__)
    except ModuleNotFoundError:
        import sys

        print(f"Error: unknown tool set {name!r}")
        print("Available tool sets: coder")
        sys.exit(1)
