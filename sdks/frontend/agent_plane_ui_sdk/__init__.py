"""agent-plane UI SDK — terminal UI components for agent-plane frontends.

Built on top of :mod:`agent_plane_client`. This package provides
Rich-based block formatting and a prompt_toolkit-based terminal host
for building REPLs. For the headless client (HTTP, SSE, blocks),
import from :mod:`agent_plane_client` directly.

Usage::

    from agent_plane_client import AgentPlaneClient, BlockStream
    from agent_plane_ui_sdk import RichBlockFormatter, TerminalHost
"""

from .terminal import PendingAttachment, RichBlockFormatter, StreamingText, TerminalHost

__all__ = [
    "PendingAttachment",
    "RichBlockFormatter",
    "StreamingText",
    "TerminalHost",
]
