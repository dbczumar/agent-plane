"""Code sandbox built-in tool.

Executes arbitrary shell commands in a persistent, per-conversation
workspace directory. Optionally sandboxed via ``srt`` for OS-level
filesystem restrictions with unrestricted network access.

The agent uses this like a terminal — ``cat``, ``python``, ``ls``,
``echo >``, ``pip install``, ``curl``, etc. all work. Output is
captured and returned as a string.

Sandboxing uses a Node.js wrapper (``_srt_wrap.mjs``) that calls
srt's ``SandboxManager`` library API. The wrapper disables network
restriction by calling ``updateConfig()`` without ``allowedDomains``
— a workaround for srt's CLI requiring a network allowlist with no
"allow all" option. See ``_srt_wrap.mjs`` for details.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from agent_plane.tools.base import Tool, ToolContext

# Absolute path to the Node.js wrapper script that invokes srt's
# library API with filesystem-only sandboxing (no network restriction).
_SRT_WRAP_PATH = str(Path(__file__).parent / "_srt_wrap.mjs")


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


def deny_read_paths() -> list[str]:
    """
    Return the directories to deny reads from.

    On **Linux**, srt uses bubblewrap which starts with
    ``--ro-bind / /`` (mount entire root read-only) and then
    overlays ``tmpfs`` on denied dirs, bind-mounting ``allowRead``
    paths back through. So ``denyRead: ["/"]`` works correctly —
    system binaries remain readable from the initial ro-bind,
    and network is unaffected.

    On **macOS**, srt uses sandbox-exec seatbelt rules.
    ``denyRead: ["/"]`` blocks all ``file-read*`` syscalls
    under ``/``, which breaks both shell PATH resolution AND
    network (the TLS/DNS stack needs to read system files that
    may not be in our ``allowRead`` list). So we deny only the
    user-data and temp roots, derived from the system:

    - **Home root**: ``Path.home().parent`` (``/Users``).
    - **Temp dir**: ``/tmp`` plus its resolved form
      (``/private/tmp`` on macOS).

    :returns: List of paths to deny reads from.
    """
    if platform.system() != "Darwin":
        return ["/"]

    home_root = str(Path.home().parent)
    tmp = Path("/tmp")
    deny = {home_root, str(tmp), str(tmp.resolve())}
    return sorted(deny)


def build_srt_config(workspace: Path) -> str:
    """
    Build the JSON config string for the srt wrapper script.

    Filesystem isolation denies reads from user-data and temp
    directories. On macOS, system directories needed for shell
    command execution are left readable (targeted deny). On
    Linux, ``denyRead: ["/"]`` is used (bwrap handles it).

    Network is left unrestricted — the wrapper script handles
    this by calling ``SandboxManager.updateConfig()`` to remove
    ``allowedDomains``, which disables srt's network proxy.

    :param workspace: The workspace directory. Resolved to its
        canonical path to handle macOS ``/var`` → ``/private/var``
        symlinks.
    :returns: JSON string for the ``_srt_wrap.mjs`` config arg.
    """
    resolved = str(workspace.resolve())

    allow_read = [resolved]
    if platform.system() == "Darwin":
        allow_read += system_read_allowlist()

    config = {
        "filesystem": {
            "allowWrite": [resolved],
            "denyRead": deny_read_paths(),
            "allowRead": allow_read,
            "denyWrite": [],
        },
    }
    return json.dumps(config)


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
    is wrapped via ``_srt_wrap.mjs`` for OS-level filesystem
    sandboxing with unrestricted network access. Otherwise,
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

        When srt is active, invokes ``_srt_wrap.mjs`` which uses
        srt's library API to sandbox the command with filesystem
        isolation and unrestricted network access.

        :param command: The shell command string.
        :param workspace: The workspace directory path.
        :returns: The command list for ``Popen``.
        """
        if self._srt_available and self._sandbox_enabled:
            config_json = build_srt_config(workspace)
            resolved = workspace.resolve()
            # TMPDIR=workspace so bash heredocs stay in the sandbox.
            wrapped = f"TMPDIR={resolved} {command}"
            return ["node", _SRT_WRAP_PATH, config_json, wrapped]
        return ["bash", "-c", command]
