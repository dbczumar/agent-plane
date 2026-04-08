"""Code sandbox built-in tool.

Executes arbitrary shell commands in a persistent, per-conversation
workspace directory. Optionally sandboxed via ``srt`` for OS-level
filesystem and network restrictions.

The agent uses this like a terminal — ``cat``, ``python``, ``ls``,
``echo >``, ``pip install``, etc. all work. Output is captured and
returned as a string.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from agent_plane.tools.base import Tool, ToolContext

_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "code_sandbox",
        "description": (
            "Execute a shell command in a persistent workspace. "
            "Use for reading files (cat), writing files (echo >), "
            "running scripts (python), installing packages (pip), "
            "and any other shell operation. The workspace persists "
            "across turns within the conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
            },
            "required": ["command"],
        },
    },
}


class CodeSandboxTool(Tool):
    """
    Execute shell commands in the per-conversation workspace.

    When ``srt`` is available and sandbox is enabled, the command
    is wrapped with ``srt -c`` for OS-level sandboxing. Otherwise,
    plain ``bash -c`` is used.

    :param srt_available: Whether ``srt`` is on PATH.
    :param sandbox_enabled: Whether sandboxing is enabled in config.
    """

    def __init__(
        self,
        srt_available: bool = False,
        sandbox_enabled: bool = True,
    ) -> None:
        """
        Initialize the code sandbox tool.

        :param srt_available: Whether ``srt`` is on PATH.
        :param sandbox_enabled: Whether sandboxing is enabled.
        """
        self._srt_available = srt_available
        self._sandbox_enabled = sandbox_enabled
        self._proc: subprocess.Popen[bytes] | None = None

    @classmethod
    def name(cls) -> str:
        """
        Tool name for dispatch and schema registration.

        :returns: ``"code_sandbox"``.
        """
        return "code_sandbox"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema.

        :returns: The schema dict.
        """
        return _SCHEMA

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Execute a shell command in the workspace.

        :param arguments: JSON with ``"command"`` key.
        :param ctx: Execution context with ``workspace`` path.
        :returns: Combined stdout + stderr output, or error string.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        command = parsed.get("command", "")
        if not command:
            return "Error: empty command"

        if ctx.workspace is None:
            return "Error: no workspace available"

        cmd = self._build_command(command)
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(ctx.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            stdout, _ = self._proc.communicate()
            output = stdout.decode(errors="replace")
            if self._proc.returncode != 0:
                return f"{output}\n[exit code {self._proc.returncode}]"
            return output
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        finally:
            self._proc = None

    def cancel(self) -> None:
        """
        Kill the subprocess on timeout.
        """
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _build_command(self, command: str) -> list[str]:
        """
        Build the subprocess command, optionally wrapping with srt.

        :param command: The shell command string.
        :returns: The command list for ``Popen``.
        """
        if self._srt_available and self._sandbox_enabled:
            return ["srt", "-c", command]
        return ["bash", "-c", command]
