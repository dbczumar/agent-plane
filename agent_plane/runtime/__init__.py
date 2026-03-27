"""Runtime package — public API for accessing runtime state.

Workflow code imports getter functions from here rather than
touching _globals directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_plane.runtime import _globals
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
    from agent_plane.tools import ToolManager


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
    Initialize the runtime with store references.
    Called once at server startup before any workflows run.

    :param conversation_store: The ConversationStore instance
        for persisting conversation items.
    :param task_store: The TaskStore instance for managing
        task lifecycle and durable execution.
    :param agent_store: The AgentStore instance for
        CRUD operations on registered agents.
    :param agent_cache: The AgentCache instance for
        loading and caching parsed agent specs.
    :param file_store: The FileStore instance for file
        metadata lookups during content resolution.
        ``None`` disables multimodal file_id resolution.
    :param artifact_store: The ArtifactStore instance for
        fetching file binary content during resolution.
        ``None`` disables multimodal file_id resolution.
    :param caps: Operator-configured execution ceiling.
        ``None`` uses :class:`RuntimeCaps` defaults.
    """
    _globals.init(
        conversation_store=conversation_store,
        task_store=task_store,
        agent_store=agent_store,
        agent_cache=agent_cache,
        file_store=file_store,
        artifact_store=artifact_store,
        caps=caps,
    )


def get_conversation_store() -> ConversationStore:
    """
    Return the canonical ConversationStore instance.

    :returns: The ConversationStore set during :func:`init`.
    :raises RuntimeError: If the runtime has not been
        initialized.
    """
    store = _globals._conversation_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_task_store() -> TaskStore:
    """
    Return the canonical TaskStore instance.

    :returns: The TaskStore set during :func:`init`.
    :raises RuntimeError: If the runtime has not been
        initialized.
    """
    store = _globals._task_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_agent_store() -> AgentStore:
    """
    Return the canonical AgentStore instance.

    :returns: The AgentStore set during :func:`init`.
    :raises RuntimeError: If the runtime has not been
        initialized.
    """
    store = _globals._agent_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_file_store() -> FileStore | None:
    """
    Return the FileStore instance, or ``None`` if not configured.

    Returns ``None`` (rather than raising) because file_store is
    optional — multimodal file_id resolution is simply skipped
    when no file store is available.

    :returns: The FileStore set during :func:`init`, or ``None``.
    """
    return _globals._file_store


def get_artifact_store() -> ArtifactStore | None:
    """
    Return the ArtifactStore instance, or ``None`` if not configured.

    Returns ``None`` (rather than raising) because artifact_store
    is optional — multimodal file_id resolution is simply skipped
    when no artifact store is available.

    :returns: The ArtifactStore set during :func:`init`, or ``None``.
    """
    return _globals._artifact_store


def get_agent_cache() -> AgentCache:
    """
    Return the canonical AgentCache instance.

    :returns: The AgentCache set during :func:`init`.
    :raises RuntimeError: If the runtime has not been
        initialized.
    """
    cache = _globals._agent_cache
    if cache is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return cache


def get_tool_manager() -> ToolManager:
    """
    Return the current workflow's ToolManager from the
    ContextVar. Must be called within a workflow that has
    set the tool manager.

    :returns: The ToolManager for the current workflow.
    :raises RuntimeError: If no ToolManager has been set for
        the current workflow context.
    """
    mgr = _globals._tool_manager_var.get()
    if mgr is None:
        raise RuntimeError("no ToolManager set for this workflow")
    return mgr


def set_tool_manager(mgr: ToolManager | None) -> None:
    """
    Set or clear the per-workflow ToolManager ContextVar.

    :param mgr: The ToolManager for the current workflow,
        or ``None`` to clear the binding (e.g. in a
        ``finally`` block after the workflow completes).
    """
    _globals._tool_manager_var.set(mgr)


def get_caps() -> RuntimeCaps:
    """
    Return the runtime caps set during :func:`init`.

    :returns: The :class:`RuntimeCaps` instance. Always
        non-None (defaults are used if none were provided).
    """
    return _globals._caps
