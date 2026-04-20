"""Private module-level state for the runtime.

Never import this module outside of agent_plane.runtime.
Use the public getter functions in runtime/__init__.py instead.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

from agent_plane.runtime.caps import RuntimeCaps

if TYPE_CHECKING:
    from agent_plane.runtime.agent_cache import AgentCache
    from agent_plane.stores import (
        AgentStore,
        ArtifactStore,
        ConversationStore,
        FileStore,
        TaskStore,
    )
    from agent_plane.terminals import TerminalManagerRegistry
    from agent_plane.tools import ToolManager

_conversation_store: ConversationStore | None = None
_task_store: TaskStore | None = None
_agent_store: AgentStore | None = None
_agent_cache: AgentCache | None = None
_file_store: FileStore | None = None
_artifact_store: ArtifactStore | None = None
_caps: RuntimeCaps = RuntimeCaps()

# Server-resident terminal-shell registry. Initialized in
# :func:`init` and accessed via ``get_terminal_registry()`` in
# ``agent_plane.runtime``. Conversation-scoped: one
# ``TerminalManager`` per ``conversation_id``, shells persist
# across task workflows within a conversation. See
# ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.4, §6.9.
_terminal_registry: TerminalManagerRegistry | None = None

# Per-workflow tool manager. ContextVar ensures thread-safe isolation —
# DBOS runs each workflow in its own thread, and contextvars are
# per-task/per-thread safe.
_tool_manager_var: ContextVar[ToolManager | None] = ContextVar(
    "_tool_manager",
    default=None,
)


def init(
    *,
    conversation_store: ConversationStore,
    task_store: TaskStore,
    agent_store: AgentStore,
    agent_cache: AgentCache,
    file_store: FileStore | None = None,
    artifact_store: ArtifactStore | None = None,
    caps: RuntimeCaps | None = None,
) -> None:
    """
    Set the runtime's store references. Called once at server
    startup.

    :param conversation_store: The ConversationStore instance
        for persisting conversation items.
    :param task_store: The TaskStore instance for managing
        task lifecycle and durable execution.
    :param agent_store: The AgentStore instance for CRUD
        operations on registered agents.
    :param agent_cache: The AgentCache instance for loading
        and caching parsed agent specs.
    :param file_store: The FileStore instance for file
        metadata lookups during content resolution.
        ``None`` disables multimodal file_id resolution.
    :param artifact_store: The ArtifactStore instance for
        fetching file binary content during content
        resolution. ``None`` disables multimodal file_id
        resolution.
    :param caps: Operator-configured execution ceiling.
        ``None`` uses :class:`RuntimeCaps` defaults.
    """
    from agent_plane.terminals import TerminalManagerRegistry

    global _conversation_store, _task_store, _agent_store  # noqa: PLW0603
    global _agent_cache, _file_store, _artifact_store, _caps  # noqa: PLW0603
    global _terminal_registry  # noqa: PLW0603
    _conversation_store = conversation_store
    _task_store = task_store
    _agent_store = agent_store
    _agent_cache = agent_cache
    _file_store = file_store
    _artifact_store = artifact_store
    _caps = caps if caps is not None else RuntimeCaps()
    # Terminal registry: server-resident, conversation-scoped shell
    # management (see §6.4/§6.9 of PERSISTENT_TERMINAL_RESEARCH.md).
    # Sandbox policy mirrors ``RuntimeCaps.sandbox_enabled`` — when
    # True (default), every shell spawned by this registry is wrapped
    # via srt (§6.5). Timeouts use the §7.1 hardcoded defaults.
    _terminal_registry = TerminalManagerRegistry(
        sandbox_enabled=_caps.sandbox_enabled,
    )
