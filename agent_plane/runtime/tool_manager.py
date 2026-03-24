"""Tool lifecycle and dispatch for agent execution.

Each workflow creates its own ToolManager, connects MCP servers at
start, and tears them down in finally. No cross-execution reuse.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent_plane.spec import AgentSpec

_logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages tool discovery and dispatch for a single workflow execution.

    Currently supports:
    - load_skill built-in (if the agent has skills)

    Not yet implemented:
    - MCP server connections (stdio / http)
    - Local tool execution (Python / TypeScript)
    """

    def __init__(self, spec: AgentSpec, work_dir: Path) -> None:
        self._spec = spec
        self._work_dir = work_dir
        self._started = False

    def start(self) -> None:
        """
        Connect to MCP servers and discover tools.
        Currently a no-op — MCP support is not yet implemented.
        """
        if self._spec.mcp_servers:
            _logger.warning(
                "Agent has %d MCP server(s) configured but MCP support "
                "is not yet implemented — tools will not be available",
                len(self._spec.mcp_servers),
            )
        self._started = True

    def shutdown(self) -> None:
        """
        Disconnect from all MCP servers.
        Currently a no-op — MCP support is not yet implemented.
        """
        self._started = False

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Return OpenAI-format tool schemas for all available tools.
        """
        schemas: list[dict[str, Any]] = []

        # Built-in: load_skill (only if the agent has skills)
        if self._spec.skills:
            skill_names = [s.name for s in self._spec.skills]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "load_skill",
                        "description": (
                            "Load a skill's full instructions by name. "
                            f"Available skills: {', '.join(skill_names)}"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The skill name to load",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                }
            )

        return schemas

    def call_tool(self, name: str, arguments: str) -> str:
        """
        Route a tool call to the appropriate handler.
        Returns the tool output as a string.
        """
        if name == "load_skill":
            return self._load_skill(arguments)

        return f"Error: tool {name!r} not found or not yet supported"

    def _load_skill(self, arguments: str) -> str:
        """Built-in: look up a skill by name and return its content."""
        args: dict[str, Any] = json.loads(arguments)
        skill_name = args.get("name")
        if skill_name is None:
            return "Error: missing required 'name' argument"

        for skill in self._spec.skills:
            if skill.name == skill_name:
                return skill.content

        available = [s.name for s in self._spec.skills]
        return f"Error: skill {skill_name!r} not found. Available skills: {available}"
