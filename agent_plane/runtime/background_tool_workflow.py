"""
DBOS workflow for ``@tool(synchronous=False)`` execution.

The parent workflow dispatches an async tool by starting one of
these child workflows (pinned to a fresh ``task_id`` representing
the background task in ``task_store``). The child:

1. Runs the tool subprocess via the existing ``LocalPythonTool``
   path (one DBOS ``@step`` so DBOS caches the outcome on replay
   per G31).
2. Truncates the result to ~10k chars at the LLM boundary
   (B5 / G44). Full output is preserved separately in
   ``task_store``.
3. Signals the parent via
   ``DBOS.send(parent_task_id, payload, topic="async_work_complete")``
   so the parent's drain (G19) wakes and injects the result.
4. Returns the same payload as the workflow's terminal output —
   the row in ``task_store`` carries the full record for
   ``check_task`` lookups.

Failure path (G86): on any exception during step execution, the
finalization sends an ``async_work_complete`` payload with
``status="failed"`` and a truncated traceback, ensuring the parent
drain still wakes and removes the task from ``pending_tasks``
(otherwise auto-collect could hang forever).
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import Any

from agent_plane.runtime.durability import (
    dbos_send,
    get_workflow_id,
    step,
    workflow,
)

_logger = logging.getLogger(__name__)

# B5/G44 — Python `len(str)` budget for the result text the LLM
# sees. Full output is preserved in task_store; this is purely the
# LLM-facing truncation point.
_RESULT_CHAR_BUDGET = 10_000

# B2 — keep tracebacks bounded so a single failure can't blow out
# the parent LLM's context.
_TRACEBACK_LINE_BUDGET = 30


# ── Topic name (matches the design doc's drain protocol) ────


# Drain channel used by the parent workflow's between-iteration
# drain and end-of-turn auto-collect. ALL background work
# (sub-agents, async tools, async client tools) signals on this
# single topic so the parent has one wake point regardless of kind.
ASYNC_WORK_COMPLETE_TOPIC = "async_work_complete"


@dataclass(frozen=True)
class AsyncWorkCompletePayload:
    """
    Payload shape for the ``async_work_complete`` signal.

    Sent on every terminal transition (completed, failed,
    cancelled) so the parent's drain wakes and removes the
    task from ``pending_tasks``. G86 — without this, a cancelled
    or failed task would never wake the parent and end-of-turn
    auto-collect would hang indefinitely.

    :param task_id: The child task's ID (== DBOS workflow_id;
        G56).
    :param kind: The task kind discriminator, one of
        ``"tool"``, ``"sub_agent"``, ``"client_tool"``. Stored
        so the parent's auto-delivery formatter can build a
        kind-appropriate ``[System: ...]`` system message.
    :param status: Terminal status, one of ``"completed"``,
        ``"failed"``, ``"cancelled"``.
    :param output: For ``"completed"``, the truncated tool
        result text (B5). For ``"failed"``, a short error
        message (the truncated traceback lives in ``error``).
        For ``"cancelled"``, the empty string.
    :param error: Failure detail dict with ``"message"`` and
        ``"traceback"`` keys, or ``None`` for non-failed.
    """

    task_id: str
    kind: str
    status: str
    output: str
    error: dict[str, str] | None


def truncate_for_llm(text: str, *, budget: int = _RESULT_CHAR_BUDGET) -> str:
    """
    Truncate a result string to ``budget`` Python ``len()`` units.

    Code-point-counted (not bytes / not graphemes — see B5/G44).
    Appends a marker indicating how many chars were dropped so the
    LLM knows the result was truncated rather than thinking the
    tool returned a short response.

    :param text: The full result text to truncate.
    :param budget: Maximum allowed char count, e.g. ``10000``.
    :returns: Either the original text (if under budget) or the
        first ``budget`` chars followed by a truncation marker
        like ``"\\n[...12345 more chars truncated...]"``.
    """
    if len(text) <= budget:
        return text
    dropped = len(text) - budget
    return f"{text[:budget]}\n[...{dropped} more chars truncated...]"


def truncate_traceback(
    tb_text: str,
    *,
    line_budget: int = _TRACEBACK_LINE_BUDGET,
) -> str:
    """
    Cap a traceback at ``line_budget`` lines.

    Keeps the first ``line_budget`` lines (which include the
    exception type and the deepest call stack — the most
    diagnostically useful part). Appends a marker indicating
    how many lines were dropped.

    :param tb_text: The full traceback text from
        ``traceback.format_exc()``.
    :param line_budget: Maximum number of lines to keep,
        e.g. ``30``.
    :returns: The truncated traceback text, with a marker
        when truncation occurred.
    """
    lines = tb_text.splitlines()
    if len(lines) <= line_budget:
        return tb_text
    dropped = len(lines) - line_budget
    head = "\n".join(lines[:line_budget])
    return f"{head}\n[...{dropped} more lines truncated...]"


def format_failure_payload(exc: BaseException) -> dict[str, str]:
    """
    Build the ``error`` field of an ``async_work_complete`` payload
    for a tool that raised.

    :param exc: The exception caught from the tool body.
    :returns: A dict with ``"message"`` (``"<ExcType>: <text>"``)
        and ``"traceback"`` (truncated to the line budget) keys.
        Both values are JSON-safe strings.
    """
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "message": f"{type(exc).__name__}: {exc}",
        "traceback": truncate_traceback(tb_text),
    }


@step()
async def _execute_tool_step(
    module_path: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Single DBOS ``@step`` that runs the tool subprocess.

    Cached on replay (G31): if the parent workflow crashes after
    this step succeeds, replay returns the cached result without
    re-executing the subprocess. If the step itself was mid-flight
    when the crash happened, the subprocess is re-spawned on
    restart — non-idempotent tool bodies may double-execute (G54;
    documented in the ``@tool`` decorator's docstring).

    :param module_path: Absolute path to the tool's Python file,
        e.g. ``"/tmp/cache/ag_abc/tools/python/my_tools.py"``.
    :param tool_name: The decorated function's ``__name__``,
        e.g. ``"train_model"``.
    :param arguments: Deserialized argument dict from the LLM.
    :returns: The tool's result as a string (subprocess returns
        a JSON string by convention; see ``_runner._serialize_result``).
    """
    # Lazy imports to avoid pulling LocalPythonTool/ToolManager into
    # this module's import chain (the runner subprocess re-imports
    # this module's parent via @workflow, and ToolManager
    # transitively imports `mcp` which conflicts in subprocess).
    from agent_plane.tools._runner_invoke import invoke_runner_subprocess

    return await invoke_runner_subprocess(
        module_path=module_path,
        tool_name=tool_name,
        arguments=arguments,
    )


@workflow()
async def background_tool_workflow(
    parent_task_id: str,
    module_path: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Run an async ``@tool(synchronous=False)`` invocation.

    Lifecycle:
    1. Run :func:`_execute_tool_step` (the actual tool body).
    2. On success, build an :class:`AsyncWorkCompletePayload` with
       ``status="completed"`` and the truncated result.
    3. On failure, build the same payload with ``status="failed"``
       and a truncated traceback (G86 — must signal even on
       failure so the parent drain wakes).
    4. ``DBOS.send`` the payload to the parent on
       :data:`ASYNC_WORK_COMPLETE_TOPIC`.
    5. Return the payload so DBOS persists it as the workflow's
       terminal output (``check_task`` reads from here).

    :param parent_task_id: The task_id of the parent workflow that
        is awaiting this tool's completion. Receives the
        ``async_work_complete`` signal.
    :param module_path: Absolute path to the tool's source file.
    :param tool_name: The decorated function's ``__name__``.
    :param arguments: Deserialized argument dict from the LLM.
    :returns: The :class:`AsyncWorkCompletePayload` as a dict
        (DBOS serializes the workflow output as JSON; dataclasses
        roundtrip cleanly via field access).
    """
    task_id = get_workflow_id()

    try:
        raw_result = await _execute_tool_step(
            module_path=module_path,
            tool_name=tool_name,
            arguments=arguments,
        )
        truncated = truncate_for_llm(raw_result)
        payload = AsyncWorkCompletePayload(
            task_id=task_id,
            kind="tool",
            status="completed",
            output=truncated,
            error=None,
        )
    except BaseException as exc:  # noqa: BLE001 — workflow boundary
        # Catch BaseException (not just Exception) because DBOS
        # cancellation propagates as BaseException; we still want
        # to signal the parent so its drain wakes and removes us
        # from pending_tasks. Re-raise after sending so DBOS records
        # the failure on the workflow.
        error = format_failure_payload(exc)
        payload = AsyncWorkCompletePayload(
            task_id=task_id,
            kind="tool",
            status="failed",
            output=error["message"],
            error=error,
        )
        _send_payload(parent_task_id, payload)
        raise

    _send_payload(parent_task_id, payload)
    return _payload_to_dict(payload)


def _send_payload(parent_task_id: str, payload: AsyncWorkCompletePayload) -> None:
    """
    Wrap the ``DBOS.send`` call so the workflow body stays readable.

    :param parent_task_id: The parent workflow's task_id.
    :param payload: The completion payload to deliver.
    """
    dbos_send(
        parent_task_id,
        _payload_to_dict(payload),
        topic=ASYNC_WORK_COMPLETE_TOPIC,
    )


def _payload_to_dict(payload: AsyncWorkCompletePayload) -> dict[str, Any]:
    """
    Serialize the dataclass to a plain dict for DBOS.

    Avoids a JSON-roundtrip hop — DBOS handles serialization on
    its own; we just hand it a dict with primitive values.

    :param payload: The completion payload.
    :returns: A JSON-safe dict.
    """
    return {
        "task_id": payload.task_id,
        "kind": payload.kind,
        "status": payload.status,
        "output": payload.output,
        "error": payload.error,
    }
