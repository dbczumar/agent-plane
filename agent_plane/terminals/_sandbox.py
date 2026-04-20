"""srt sandbox configuration for the persistent terminal tool.

Builds the JSON config passed to :file:`_srt_shell.mjs`.

The sandbox denies reads of user-data and temp dirs (home, ``/tmp``)
while allowing the workspace to be read+written. Network is left
unrestricted — srt's normal filesystem-AND-network posture doesn't
fit the "agent needs internet to ``pip install``" case, so we use
srt's library API with ``allowedDomains`` undefined to disable the
network proxy entirely (see ``_srt_shell.mjs`` for the trick).

See ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.5, §6.8.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

# Absolute path to the PTY-compatible srt wrapper. Used by
# :meth:`Shell.spawn` when sandbox wrapping is enabled.
_SRT_SHELL_PATH = str(Path(__file__).parent / "_srt_shell.mjs")


def is_srt_available() -> bool:
    """Whether srt + node are both on PATH.

    Both are needed: srt's library is loaded by the Node wrapper.
    Either binary missing = can't sandbox.

    :returns: True iff sandboxing is usable in this environment.
    """
    return shutil.which("srt") is not None and shutil.which("node") is not None


def srt_shell_path() -> str:
    """Return the absolute path to the PTY-compatible srt wrapper.

    :returns: Filesystem path as a string, ready to pass to
        ``pexpect.spawn``.
    """
    return _SRT_SHELL_PATH


def system_read_allowlist() -> list[str]:
    """
    Derive the set of system directories that must remain readable
    for shell commands to work inside the sandbox.

    On macOS, ``denyRead: ["/"]`` blocks ALL ``file-read*`` syscalls
    (including PATH resolution for ``/usr/bin/cat``). We need to
    explicitly re-allow system directories. The list is built
    dynamically from ``$PATH`` entries (discarding anything under
    the user home tree) plus essential OS plumbing.

    On Linux this is unused — srt's bubblewrap starts with
    ``--ro-bind / /`` so system paths are already readable.

    :returns: Sorted list of top-level system directories, e.g.
        ``["/Applications", "/System", "/bin", ...]``.
    """
    home_root = Path.home().parent
    roots: set[str] = set()
    for p in os.get_exec_path():
        resolved = Path(p).resolve()
        # Skip anything under the user home tree — that's the data
        # we're trying to protect.
        if resolved.is_relative_to(home_root):
            continue
        if resolved.exists() and len(resolved.parts) >= 2:
            roots.add("/" + resolved.parts[1])
    # Essential dirs beyond PATH. /dev device nodes, /etc system
    # config, /run runtime state (DNS needs /run/systemd/resolve),
    # /proc and /sys on Linux, and the /private aliases on macOS.
    roots.update(["/dev", "/etc", "/run", "/proc", "/sys", "/private/etc", "/private/var/run"])
    return sorted(roots)


def deny_read_paths() -> list[str]:
    """
    Return the directories to deny reads from.

    On **Linux**, srt's bubblewrap starts with ``--ro-bind / /``
    (root read-only) and overlays ``tmpfs`` on denied dirs. So
    ``denyRead: ["/"]`` works — system binaries remain readable.

    On **macOS**, sandbox-exec seatbelt doesn't have a ``--ro-bind``
    equivalent; ``denyRead: ["/"]`` breaks shell PATH resolution
    AND the TLS/DNS stack. We deny only user-data and temp roots,
    derived from the system: ``Path.home().parent`` (``/Users``)
    and ``/tmp`` (plus its macOS-resolved ``/private/tmp``).

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
    Build the JSON config string for :file:`_srt_shell.mjs`.

    Filesystem isolation denies reads from user-data and temp dirs
    (per :func:`deny_read_paths`) and allows reads+writes within
    the conversation's workspace. On macOS, system dirs needed
    for shell PATH and network stack are additionally re-allowed
    via :func:`system_read_allowlist`. Network is left unrestricted
    by the wrapper (see :file:`_srt_shell.mjs`).

    The directory containing the bundled shell-integration snippet
    (``agent_plane/terminals/_integration.sh``) is always added to
    ``allowRead``. When agent-plane is installed under the user's
    home dir (editable / venv install), that directory is inside a
    denied tree — without this allowlist entry, bash can't source
    the rcfile and the shell never emits its initial OSC 633 marker.

    :param workspace: The per-conversation workspace directory.
        Resolved to its canonical path to handle macOS
        ``/var`` → ``/private/var`` symlinks.
    :returns: JSON string for :file:`_srt_shell.mjs`'s config arg.
    """
    resolved = str(workspace.resolve())

    # Directory that holds ``_integration.sh`` (and this file). In a
    # normal install that's the ``agent_plane/terminals`` package
    # dir. Must be readable inside the sandbox so ``bash --rcfile``
    # can actually source the snippet.
    terminals_pkg_dir = str(Path(__file__).resolve().parent)

    allow_read = [resolved, terminals_pkg_dir]
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
