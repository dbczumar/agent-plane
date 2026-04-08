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


def ensure_dbos(
    uri: str,
    *,
    application_version: str | None = None,
) -> None:
    """
    Initialize the DBOS singleton. Idempotent — safe to call
    repeatedly.

    :param uri: Database connection URI for the DBOS system
        database, e.g.
        ``"postgresql://user:pass@localhost:5432/dbos"``
        or ``"sqlite:///path/to/db.db"``.
    :param application_version: Optional fixed application
        version string, e.g. ``"v1"``. When omitted, DBOS
        auto-computes a version from registered workflow
        source code. Pinning the version is required for
        crash recovery: the restarted instance must use
        the same version as the crashed one so DBOS can
        identify pending workflows to recover.
    """
    global _dbos_initialized
    if _dbos_initialized:
        return
    with _init_lock:
        if _dbos_initialized:
            return
        config = DBOSConfig(
            name="agent-plane",
            system_database_url=uri,
        )
        if application_version is not None:
            config["application_version"] = application_version
        DBOS(config=config)
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

    :returns: ``True`` if ``ensure_dbos`` has been called and
        ``destroy_dbos`` has not been called since.
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
dbos_sleep = DBOS.sleep
dbos_recv = DBOS.recv
read_stream = DBOS.read_stream


def send_direct(
    destination_id: str,
    message: object,
    topic: str | None = None,
) -> None:
    """
    Send a DBOS message without requiring a workflow context.

    Used by HTTP handlers to wake parked sub-agent workflows.
    Internally uses ``_sys_db.send_direct`` which provides
    idempotency via a generated message UUID.

    :param destination_id: The target workflow ID,
        e.g. ``"task_sub1"``.
    :param message: The message payload (any serializable value).
    :param topic: Optional topic string for routing,
        e.g. ``"tool_result"``.
    """
    # Access internal API — DBOS doesn't expose send_direct
    # on the public DBOS class. This is the only way to send
    # from outside a workflow context.
    from dbos._dbos import _get_dbos_instance

    _get_dbos_instance()._sys_db.send_direct(destination_id, message, topic=topic)


write_stream = DBOS.write_stream
close_stream = DBOS.close_stream

# Async APIs (for use inside async methods — DBOS raises if you
# call the sync version while an event loop is running)
retrieve_workflow_async = DBOS.retrieve_workflow_async
get_workflow_status_async = DBOS.get_workflow_status_async
cancel_workflow_async = DBOS.cancel_workflow_async
read_stream_async = DBOS.read_stream_async

# Async APIs for parallel tool execution
# (see designs/PARALLEL_TOOL_CALLS.md)
dbos_recv_async = DBOS.recv_async
dbos_sleep_async = DBOS.sleep_async
asyncio_wait = DBOS.asyncio_wait
close_stream_async = DBOS.close_stream_async


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
    "asyncio_wait",
    "dbos_recv",
    "dbos_recv_async",
    "dbos_sleep",
    "dbos_sleep_async",
    "send_direct",
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
