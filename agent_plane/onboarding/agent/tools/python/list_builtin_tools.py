"""List all built-in tools available in agent plane.

Returns the live registry of builtin tool names and their
descriptions, so the onboarding assistant always recommends
from the current set — not a stale hardcoded list.
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

# Maps builtin names to their Tool class import paths so we can
# call cls.description() without instantiation.
_TOOL_CLASSES: dict[str, tuple[str, str]] = {
    "code_sandbox": ("agent_plane.tools.builtins.code_sandbox", "CodeSandboxTool"),
    "download_file": ("agent_plane.tools.builtins.download_file", "DownloadFileTool"),
    "export_agent": ("agent_plane.tools.builtins.export_agent", "ExportAgentTool"),
    "introspect": ("agent_plane.tools.builtins.introspect", "IntrospectTool"),
    "list_files": ("agent_plane.tools.builtins.list_files", "ListFilesTool"),
    "search_conversations": (
        "agent_plane.tools.builtins.search_conversations",
        "SearchConversationsTool",
    ),
    "upload_file": ("agent_plane.tools.builtins.upload_file", "UploadFileTool"),
    "web_fetch": ("agent_plane.tools.builtins.web_fetch", "WebFetchTool"),
    "web_search": ("agent_plane.tools.builtins.web_search", "WebSearchTool"),
}


async def run(arguments: dict[str, Any]) -> str:
    """
    Query the builtin tool registry and return names + descriptions.

    :param arguments: Unused (no parameters).
    :returns: Formatted list of available builtin tools.
    """
    import importlib

    from agent_plane.tools.builtins import BUILTIN_NAMES

    lines: list[str] = []
    for name in sorted(BUILTIN_NAMES):
        class_info = _TOOL_CLASSES.get(name)
        if class_info is not None:
            module_path, class_name = class_info
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            lines.append(f"- {name}: {cls.description()}")
        else:
            lines.append(f"- {name}: (no description)")

    return "\n".join(lines)
