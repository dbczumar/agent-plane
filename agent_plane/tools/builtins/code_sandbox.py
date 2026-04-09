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
import os
import subprocess
from pathlib import Path
from typing import Any

from agent_plane.tools.base import Tool, ToolContext

# Cached srt settings file path — reused across invocations for
# the same workspace to avoid creating a temp file per call.
_srt_settings_cache: dict[str, str] = {}


def _write_srt_settings(workspace: Path) -> str:
    """
    Write an srt settings file that allows writes only to the workspace.

    Cached per workspace path so repeated invocations reuse the
    same file.

    :param workspace: The workspace directory to allow writes to.
    :returns: Path to the settings JSON file.
    """
    import tempfile

    ws_str = str(workspace)
    if ws_str in _srt_settings_cache:
        return _srt_settings_cache[ws_str]

    settings = {
        "network": {
            # Allow package registries so agents can pip/npm install.
            "allowedDomains": [
                "pypi.org",
                "files.pythonhosted.org",
                "registry.npmjs.org",
            ],
            "deniedDomains": [],
        },
        "filesystem": {
            "allowWrite": [ws_str],
            "denyRead": [],
            "denyWrite": [],
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json", prefix="srt-ap-")
    with os.fdopen(fd, "w") as f:
        json.dump(settings, f)
    _srt_settings_cache[ws_str] = path
    return path


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

        cmd = self._build_command(command, ctx.workspace)
        try:
            # Set package install targets to workspace so pip/npm
            # install locally (not to system site-packages).
            # Packages persist across turns within a conversation.
            ws = str(ctx.workspace)
            env = {
                **os.environ,
                "PIP_TARGET": f"{ws}/.pip",
                "PIP_CACHE_DIR": f"{ws}/.cache/pip",
                "PYTHONPATH": f"{ws}/.pip",
                "NODE_PATH": f"{ws}/node_modules",
                "npm_config_prefix": ws,
                "npm_config_cache": f"{ws}/.cache/npm",
            }
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(ctx.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
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

    def _build_command(
        self,
        command: str,
        workspace: Path,
    ) -> list[str]:
        """
        Build the subprocess command, optionally wrapping with srt.

        When srt is active, writes a temporary settings file that
        allows writes to the workspace directory and ``/tmp``
        (needed for bash heredocs).

        :param command: The shell command string.
        :param workspace: The workspace directory path.
        :returns: The command list for ``Popen``.
        """
        if self._srt_available and self._sandbox_enabled:
            settings_path = _write_srt_settings(workspace)
            # TMPDIR=workspace so bash heredocs stay in the sandbox.
            wrapped = f"TMPDIR={workspace} {command}"
            return ["srt", "--settings", settings_path, "-c", wrapped]
        return ["bash", "-c", command]
