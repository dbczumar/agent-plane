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
    from agent_plane.stores import AgentStore, ConversationStore, TaskStore
    from agent_plane.tools import ToolManager

_conversation_store: ConversationStore | None = None
_task_store: TaskStore | None = None
_agent_store: AgentStore | None = None
_agent_cache: AgentCache | None = None
_caps: RuntimeCaps = RuntimeCaps()

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
    :param caps: Operator-configured execution ceiling.
        ``None`` uses :class:`RuntimeCaps` defaults.
    """
    global _conversation_store, _task_store, _agent_store, _agent_cache, _caps
    _conversation_store = conversation_store
    _task_store = task_store
    _agent_store = agent_store
    _agent_cache = agent_cache
    _caps = caps if caps is not None else RuntimeCaps()
