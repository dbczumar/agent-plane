"""Built-in tool: export a generated agent from the sandbox workspace.

Used by the onboarding assistant to copy a completed agent directory
from the per-conversation sandbox workspace to a user-specified
location on disk.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Any: the OpenAI tool schema is a heterogeneous dict with string
# keys and mixed value types (str, dict, list).
from typing import Any

from agent_plane.tools.base import Tool, ToolContext

_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "export_agent",
        "description": (
            "Copy a generated agent directory from the sandbox workspace "
            "to a target path on the user's filesystem. Use this after "
            "creating an agent with code_sandbox to place it where the "
            "user wants it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Path to the agent directory inside the workspace, "
                        "relative to the workspace root. "
                        "Example: 'my-research-agent'"
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Absolute path where the agent directory should be "
                        "copied to on the user's filesystem. "
                        "Example: '/home/user/my-research-agent'"
                    ),
                },
            },
            "required": ["source", "target"],
        },
    },
}


class ExportAgentTool(Tool):
    """
    Copy a generated agent directory from the sandbox to a target path.

    The onboarding assistant generates agent files inside the
    code_sandbox workspace. This tool copies the result to the
    user's chosen location. The target directory must not already
    exist (to prevent accidental overwrites).
    """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"export_agent"``.
        """
        return "export_agent"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema.

        :returns: The schema dict.
        """
        return _SCHEMA

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Copy a directory from the workspace to a target path.

        :param arguments: JSON with ``source`` and ``target`` keys.
        :param ctx: Execution context with ``workspace`` path.
        :returns: Success message or error string.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        source_rel = parsed.get("source", "")
        target_str = parsed.get("target", "")

        if not source_rel:
            return "Error: 'source' parameter is required."
        if not target_str:
            return "Error: 'target' parameter is required."

        if ctx.workspace is None:
            return "Error: no workspace available."

        source = ctx.workspace / source_rel
        target = Path(target_str)

        if not source.exists():
            return f"Error: source directory '{source_rel}' not found in workspace."

        if not source.is_dir():
            return f"Error: source '{source_rel}' is not a directory."

        if target.exists():
            return (
                f"Error: target '{target}' already exists. "
                f"Choose a different path to avoid overwriting."
            )

        shutil.copytree(str(source), str(target))
        return f"Exported agent to {target}"
