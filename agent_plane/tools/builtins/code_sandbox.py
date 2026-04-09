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
import platform
import subprocess
from pathlib import Path
from typing import Any

from agent_plane.tools.base import Tool, ToolContext

# Cached srt settings file path — reused across invocations for
# the same workspace to avoid creating a temp file per call.
srt_settings_cache: dict[str, str] = {}


def system_read_allowlist() -> list[str]:
    """
    Derive the set of system directories that must remain readable
    for shell commands to work inside the sandbox.

    On macOS, ``denyRead: ["/"]`` blocks ALL ``file-read*``
    syscalls (including PATH resolution for ``/usr/bin/cat``).
    We need to explicitly re-allow system directories.

    The list is built dynamically rather than hardcoded:

    1. Walk ``$PATH`` (via :func:`os.get_exec_path`), resolve
       each entry, discard anything under the user home tree,
       and collect the top-level root directory (e.g.
       ``/opt/homebrew/bin`` → ``/opt``).
    2. Add ``/dev`` (device nodes), ``/private/etc`` (macOS
       ``/etc`` symlink target), and ``/private/var/run``
       (runtime sockets).

    On Linux this is unused — srt's bubblewrap starts with
    ``--ro-bind / /`` so system paths are already readable.

    **Known limitation (macOS only):** This is best-effort.
    macOS sandbox-exec has no equivalent of bubblewrap's
    ``--ro-bind / /`` (read-only root mount), so there is no
    way to allow all system paths without also enumerating
    them. The allowlist covers everything reachable via
    ``$PATH`` plus essential OS plumbing, but a tool that
    reads from an unlisted system directory (e.g. a
    non-standard ``/srv`` or ``/opt/custom``) will be denied.
    Sensitive directories (user home, ``/tmp``,
    ``/private/tmp``, ``/private/var/folders``) are
    intentionally excluded.

    :returns: Sorted list of top-level system directories,
        e.g. ``["/Applications", "/System", "/bin", ...]``.
    """
    home_root = Path.home().parent
    roots: set[str] = set()
    for p in os.get_exec_path():
        resolved = Path(p).resolve()
        # Skip anything under the user home tree — that's the
        # data we're trying to protect.
        if resolved.is_relative_to(home_root):
            continue
        if resolved.exists() and len(resolved.parts) >= 2:
            roots.add("/" + resolved.parts[1])
    # /dev for device nodes (e.g. /dev/null, /dev/urandom).
    # /private/etc is the real path behind the macOS /etc symlink.
    # /private/var/run is needed for runtime sockets/daemons.
    # NOT /private (contains /private/tmp) or /private/var
    # (contains /private/var/folders, the per-user temp cache).
    roots.update(["/dev", "/private/etc", "/private/var/run"])
    return sorted(roots)


def write_srt_settings(workspace: Path) -> str:
    """
    Write an srt settings file for sandbox read/write isolation.

    Uses ``denyRead: ["/"]`` on all platforms to block reads from
    the entire filesystem. On macOS, system directories needed
    for shell command execution are re-allowed via
    :func:`system_read_allowlist`. On Linux, srt's bubblewrap
    handles this automatically (``--ro-bind / /``).

    Cached per workspace path so repeated invocations reuse
    the same file.

    :param workspace: The workspace directory. Resolved to its
        canonical path to handle macOS ``/var`` → ``/private/var``
        symlinks.
    :returns: Path to the settings JSON file.
    """
    import tempfile

    # Resolve symlinks so the allowRead/allowWrite paths
    # match what srt sees on the real filesystem. On macOS,
    # /var is a symlink to /private/var — without resolving,
    # the workspace path won't match the deny rules.
    resolved = str(workspace.resolve())
    if resolved in srt_settings_cache:
        return srt_settings_cache[resolved]

    allow_read = [resolved]
    if platform.system() == "Darwin":
        # macOS sandbox-exec needs explicit read allowances for
        # system paths; Linux bwrap doesn't (ro-bind root).
        allow_read += system_read_allowlist()

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
            "allowWrite": [resolved],
            "denyRead": ["/"],
            "allowRead": allow_read,
            "denyWrite": [],
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json", prefix="srt-ap-")
    with os.fdopen(fd, "w") as f:
        json.dump(settings, f)
    srt_settings_cache[resolved] = path
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
        denies reads outside the workspace and system directories.

        :param command: The shell command string.
        :param workspace: The workspace directory path.
        :returns: The command list for ``Popen``.
        """
        if self._srt_available and self._sandbox_enabled:
            settings_path = write_srt_settings(workspace)
            # Resolve so TMPDIR matches the allowWrite path in
            # srt settings (macOS /var → /private/var symlink).
            resolved = workspace.resolve()
            # TMPDIR=workspace so bash heredocs stay in the sandbox.
            wrapped = f"TMPDIR={resolved} {command}"
            return ["srt", "--settings", settings_path, "-c", wrapped]
        return ["bash", "-c", command]
