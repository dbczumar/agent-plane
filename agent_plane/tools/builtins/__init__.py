"""Built-in tools for agent-plane.

Public API:
- ``LoadSkillTool``: Loads a skill's instructions by name.
- ``ReadSkillFileTool``: Reads files from a skill's directory.
- ``any_skill_has_resources``: Checks if any skill has bundled
  resource files (used by ToolManager to decide whether to
  register ReadSkillFileTool).
- ``list_skill_resources``: Lists resource files in a skill's
  directory (used by LoadSkillTool to append file listings).
- ``get_builtin_tool``: Instantiate a built-in tool by name.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_plane.spec.types import SkillSpec
from agent_plane.tools.base import Tool
from agent_plane.tools.builtins.load_skill import (
    LoadSkillTool,
    list_skill_resources,
)
from agent_plane.tools.builtins.read_skill_file import (
    ReadSkillFileTool,
)
from agent_plane.tools.builtins.spawn import (
    CancelSubAgentTool,
    CheckSubAgentsTool,
    SpawnTool,
)
from agent_plane.tools.builtins.web_search_google import (
    WebSearchGoogleTool,
)
from agent_plane.tools.builtins.web_search_openai import (
    WebSearchOpenAITool,
)
from agent_plane.tools.builtins.web_search_perplexity import (
    WebSearchPerplexityTool,
)

__all__ = [
    "CancelSubAgentTool",
    "CheckSubAgentsTool",
    "LoadSkillTool",
    "ReadSkillFileTool",
    "SpawnTool",
    "WebSearchGoogleTool",
    "WebSearchOpenAITool",
    "WebSearchPerplexityTool",
    "any_skill_has_resources",
    "get_builtin_tool",
    "list_skill_resources",
]

# Lazy imports for code_sandbox and upload_file to avoid
# circular imports — they are registered in the factory below.

# Factory type: each constructor accepts a config dict and returns
# a Tool. Callable is used instead of type[Tool] because the base
# Tool.__init__ does not declare a config parameter — only the
# web search subclasses do.
_BuiltinFactory = Callable[[dict[str, str]], Tool]


# Registry of built-in tools that agents can enable via
# ``tools.builtins`` in config.yaml. Keyed by the name string
# users write in the spec.
def _create_code_sandbox(config: dict[str, str]) -> Tool:
    """
    Lazy factory for CodeSandboxTool.

    :param config: Tool config (unused).
    :returns: A CodeSandboxTool instance.
    """
    from agent_plane.tools.builtins.code_sandbox import CodeSandboxTool

    return CodeSandboxTool()


def _create_upload_file(config: dict[str, str]) -> Tool:
    """
    Lazy factory for UploadFileTool.

    :param config: Tool config (unused).
    :returns: An UploadFileTool instance.
    """
    from agent_plane.tools.builtins.upload_file import UploadFileTool

    return UploadFileTool()


def _create_search_conversations(config: dict[str, str]) -> Tool:
    """
    Lazy factory for SearchConversationsTool.

    :param config: Tool config (unused).
    :returns: A SearchConversationsTool instance.
    """
    from agent_plane.tools.builtins.search_conversations import (
        SearchConversationsTool,
    )

    return SearchConversationsTool()


def _create_list_files(config: dict[str, str]) -> Tool:
    """
    Lazy factory for ListFilesTool.

    :param config: Tool config (unused).
    :returns: A ListFilesTool instance.
    """
    from agent_plane.tools.builtins.list_files import ListFilesTool

    return ListFilesTool()


def _create_download_file(config: dict[str, str]) -> Tool:
    """
    Lazy factory for DownloadFileTool.

    :param config: Tool config (unused).
    :returns: A DownloadFileTool instance.
    """
    from agent_plane.tools.builtins.download_file import DownloadFileTool

    return DownloadFileTool()


_BUILTIN_REGISTRY: dict[str, _BuiltinFactory] = {
    "web_search_openai": WebSearchOpenAITool,
    "web_search_google": WebSearchGoogleTool,
    "web_search_perplexity": WebSearchPerplexityTool,
    "code_sandbox": _create_code_sandbox,
    "upload_file": _create_upload_file,
    "list_files": _create_list_files,
    "download_file": _create_download_file,
    "search_conversations": _create_search_conversations,
}


def get_builtin_tool(
    name: str,
    config: dict[str, str] | None = None,
) -> Tool | None:
    """
    Instantiate a built-in tool by name with optional config.

    :param name: The tool name from ``tools.builtins`` in
        config.yaml, e.g. ``"web_search_openai"``.
    :param config: Tool-specific key-value pairs from the spec,
        e.g. ``{"api_key": "sk-...", "engine_id": "abc"}``.
        ``None`` or empty dict means no spec-level config was
        provided.
    :returns: A :class:`Tool` instance, or ``None`` if the
        name is not recognized.
    """
    factory = _BUILTIN_REGISTRY.get(name)
    if factory is None:
        return None
    return factory(config or {})


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
