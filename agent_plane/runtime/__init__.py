"""Runtime package — public API for accessing runtime state.

Workflow code imports getter functions from here rather than
touching _globals directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_plane.runtime import _globals

if TYPE_CHECKING:
    from agent_plane.runtime.agent_cache import AgentCache
    from agent_plane.runtime.tool_manager import ToolManager
    from agent_plane.stores import AgentStore, ConversationStore, TaskStore


def init(
    *,
    conversation_store: ConversationStore,
    task_store: TaskStore,
    agent_store: AgentStore,
    agent_cache: AgentCache,
) -> None:
    """
    Initialize the runtime with store references.
    Called once at server startup before any workflows run.
    """
    _globals.init(
        conversation_store=conversation_store,
        task_store=task_store,
        agent_store=agent_store,
        agent_cache=agent_cache,
    )


def get_conversation_store() -> ConversationStore:
    """Return the canonical ConversationStore instance."""
    store = _globals._conversation_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_task_store() -> TaskStore:
    """Return the canonical TaskStore instance."""
    store = _globals._task_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_agent_store() -> AgentStore:
    """Return the canonical AgentStore instance."""
    store = _globals._agent_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_agent_cache() -> AgentCache:
    """Return the canonical AgentCache instance."""
    cache = _globals._agent_cache
    if cache is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return cache


def get_tool_manager() -> ToolManager:
    """
    Return the current workflow's ToolManager from the ContextVar.
    Must be called within a workflow that has set the tool manager.
    """
    mgr = _globals._tool_manager_var.get()
    if mgr is None:
        raise RuntimeError("no ToolManager set for this workflow")
    return mgr


def set_tool_manager(mgr: ToolManager | None) -> None:
    """Set or clear the per-workflow ToolManager ContextVar."""
    _globals._tool_manager_var.set(mgr)
