"""List all built-in tools available in agent plane.

Returns the live registry of builtin tool names and their
descriptions, so the onboarding assistant always recommends
from the current set — not a stale hardcoded list.

Each tool class is imported individually from its own module to
avoid importing the ``agent_plane.tools.builtins`` package (which
transitively pulls in modules that conflict with the ``mcp`` pip
package in subprocess environments).
"""

from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_builtin_tools",
        "description": (
            "List all built-in tools available in agent plane. "
            "Returns tool names and descriptions. Call this before "
            "recommending tools for a new agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# Maps every builtin tool name to (module_path, class_name).
# This is the sole source of truth — when a new builtin is added,
# add it here. Each module is imported individually to avoid the
# transitive import chain from agent_plane.tools.builtins.__init__.
_TOOL_CLASSES: dict[str, tuple[str, str]] = {
    "download_file": ("agent_plane.tools.builtins.download_file", "DownloadFileTool"),
    "export_agent": ("agent_plane.tools.builtins.export_agent", "ExportAgentTool"),
    "introspect": ("agent_plane.tools.builtins.introspect", "IntrospectTool"),
    "list_files": ("agent_plane.tools.builtins.list_files", "ListFilesTool"),
    "search_conversations": (
        "agent_plane.tools.builtins.search_conversations",
        "SearchConversationsTool",
    ),
    "terminal_close": ("agent_plane.tools.builtins.terminal", "TerminalCloseTool"),
    "terminal_list": ("agent_plane.tools.builtins.terminal", "TerminalListTool"),
    "terminal_run": ("agent_plane.tools.builtins.terminal", "TerminalRunTool"),
    "upload_file": ("agent_plane.tools.builtins.upload_file", "UploadFileTool"),
    "web_fetch": ("agent_plane.tools.builtins.web_fetch", "WebFetchTool"),
    "web_search": ("agent_plane.tools.builtins.web_search", "WebSearchTool"),
}


async def run(arguments: dict[str, Any]) -> str:
    """
    Import each builtin tool class and return names + descriptions.

    :param arguments: Unused (no parameters).
    :returns: Formatted list of available builtin tools.
    """
    import importlib

    lines: list[str] = []
    for name in sorted(_TOOL_CLASSES):
        module_path, class_name = _TOOL_CLASSES[name]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        lines.append(f"- {name}: {cls.description()}")

    return "\n".join(lines)
