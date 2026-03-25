"""DBOS integration — the ONLY file that imports dbos.

All DBOS-specific imports are isolated here. Other modules import
from this file rather than importing dbos directly. This keeps the
DBOS dependency contained and makes it easy to swap or mock.
"""

from __future__ import annotations

import threading

from dbos import (
    DBOS,
    DBOSConfig,
    SetWorkflowID,
    WorkflowHandle,
    WorkflowHandleAsync,
    WorkflowStatus,
    WorkflowStatusString,
)

# ── Singleton initialization ────────────────────────────

_dbos_initialized = False
_init_lock = threading.Lock()


def ensure_dbos(uri: str) -> None:
    """
    Initialize the DBOS singleton. Idempotent — safe to call
    repeatedly.

    :param uri: PostgreSQL connection URI for the DBOS system
        database, e.g.
        ``"postgresql://user:pass@localhost:5432/dbos"``.
    """
    global _dbos_initialized
    if _dbos_initialized:
        return
    with _init_lock:
        if _dbos_initialized:
            return
        DBOS(config=DBOSConfig(name="agent-plane", system_database_url=uri))
        DBOS.launch()
        _dbos_initialized = True


def destroy_dbos() -> None:
    """Tear down the DBOS singleton. Used by tests."""
    global _dbos_initialized
    with _init_lock:
        if _dbos_initialized:
            DBOS.destroy()
            _dbos_initialized = False


def is_dbos_initialized() -> bool:
    """
    Return whether the DBOS singleton has been initialized.
    """
    return _dbos_initialized


# ── Re-exported DBOS decorators and utilities ───────────
# No other file should import from dbos directly.

# Sync APIs (for use outside async contexts)
workflow = DBOS.workflow
step = DBOS.step
start_workflow = DBOS.start_workflow
retrieve_workflow = DBOS.retrieve_workflow
get_workflow_status = DBOS.get_workflow_status
cancel_workflow = DBOS.cancel_workflow
read_stream = DBOS.read_stream
write_stream = DBOS.write_stream
close_stream = DBOS.close_stream

# Async APIs (for use inside async methods — DBOS raises if you
# call the sync version while an event loop is running)
retrieve_workflow_async = DBOS.retrieve_workflow_async
get_workflow_status_async = DBOS.get_workflow_status_async
cancel_workflow_async = DBOS.cancel_workflow_async
read_stream_async = DBOS.read_stream_async


def get_workflow_id() -> str:
    """
    Return the current workflow's ID. Must be called within
    a DBOS workflow context.

    :returns: The workflow ID string, e.g. ``"task_abc123"``.
    """
    return str(DBOS.workflow_id)


__all__ = [
    "SetWorkflowID",
    "WorkflowHandle",
    "WorkflowHandleAsync",
    "WorkflowStatus",
    "WorkflowStatusString",
    "cancel_workflow",
    "cancel_workflow_async",
    "close_stream",
    "destroy_dbos",
    "ensure_dbos",
    "get_workflow_id",
    "get_workflow_status",
    "get_workflow_status_async",
    "is_dbos_initialized",
    "read_stream",
    "read_stream_async",
    "retrieve_workflow",
    "retrieve_workflow_async",
    "start_workflow",
    "step",
    "workflow",
    "write_stream",
]
