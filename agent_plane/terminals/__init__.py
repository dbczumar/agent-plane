"""Persistent-terminal subsystem.

PTY-backed terminal tool that maintains persistent bash subprocesses
across conversation turns. See
``designs/PERSISTENT_TERMINAL_RESEARCH.md`` for the full design.

Three-layer stack:

- :class:`Shell` — one bash subprocess with OSC 633 completion
  detection, ring buffer, ANSI stripping, disk persistence on
  overflow, timeout enforcement, and crash detection.
- :class:`TerminalManager` — per-conversation shell registry with
  name validation and per-conversation shell cap.
- :class:`TerminalManagerRegistry` — server-resident registry keyed
  by ``conversation_id``, idle reaper, shutdown coordination.

Agent-facing tools (``terminal_run``, ``terminal_list``,
``terminal_close``) live in :mod:`agent_plane.tools.builtins.terminal`.
"""

from agent_plane.terminals.manager import (
    ShellCapExceeded,
    ShellNameInvalid,
    TerminalManager,
)
from agent_plane.terminals.registry import TerminalManagerRegistry
from agent_plane.terminals.shell import RunResult, Shell

__all__ = [
    "RunResult",
    "Shell",
    "ShellCapExceeded",
    "ShellNameInvalid",
    "TerminalManager",
    "TerminalManagerRegistry",
]
