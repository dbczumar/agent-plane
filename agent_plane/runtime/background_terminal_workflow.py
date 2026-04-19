"""
DBOS workflow for ``terminal_run(synchronous=False)`` execution.

The parent workflow dispatches an async terminal command by
starting one of these child workflows (pinned to a fresh ``task_id``
with ``kind="terminal"``). The child:

1. Registers the task → shell mapping on the
   :class:`TerminalManager` so ``cancel_task`` and ``check_task``
   can find the running command.
2. Runs ``manager.run_sync`` in a DBOS ``@step`` (which dispatches
   the blocking call to the thread pool via ``asyncio.to_thread``).
3. Translates the :class:`RunResult` into an
   :class:`~agent_plane.runtime.background_tool_workflow.AsyncWorkCompletePayload`:

   - ``status="completed"`` / exit_code 0 → payload status ``"completed"``.
   - ``status="completed"`` / exit_code != 0 → payload status
     ``"completed"`` with the exit code in the output (the command
     ran and finished on its own — non-zero is the command's result,
     not an infrastructure failure).
   - ``status="killed"`` → payload status ``"cancelled"`` (timeout
     or interrupt — agent-visible cancel either way).
   - ``status="shell_crashed"`` → payload status ``"failed"``.
   - ``status="shell_busy"`` → payload status ``"failed"`` (defensive;
     should not happen since the tool layer pre-validates by creating
     a fresh task for each async command).
4. Sends the payload to the parent via
   ``DBOS.send(parent_task_id, payload, topic="async_work_complete")``.
5. Returns the payload as the workflow output so
   ``check_task`` can read it back on terminal task inspection.

Failure path (G86): on any unexpected exception, the finalization
sends a ``status="failed"`` payload with a truncated traceback and
re-raises. This guarantees the parent's drain wakes even if the
terminal step blows up in an unforeseen way.

See ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.11 for the full
design.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import Any

from agent_plane.runtime.background_tool_workflow import (
    ASYNC_WORK_COMPLETE_TOPIC,
    AsyncWorkCompletePayload,
    format_failure_payload,
    truncate_for_llm,
)
from agent_plane.runtime.durability import (
    dbos_send_async,
    get_workflow_id,
    step,
    workflow,
)

_logger = logging.getLogger(__name__)

# Kind string recorded on the task row. Distinct from ``"tool"``
# (local Python @tool(synchronous=False) tasks) and ``"sub_agent"``
# (sub-agent tasks). Used by task_lifecycle's check_task / cancel_task
# to dispatch kind-specific behavior.
TERMINAL_KIND = "terminal"


@step()
async def _run_terminal_step(
    task_id: str,
    conversation_id: str,
    shell_name: str,
    command: str,
    timeout_ms: int | None,
    workspace_path: str,
) -> dict[str, Any]:
    """Execute the terminal command inside a DBOS ``@step``.

    Resolves the :class:`TerminalManager` from the runtime registry,
    registers the task so cancel/check can locate the running shell,
    invokes ``run_sync`` in a thread-pool thread (via
    ``asyncio.to_thread``), and unregisters in a ``finally``.

    The registration happens even though DBOS may re-execute this
    step on replay — ``register_running_task`` is idempotent
    (overwriting the mapping is fine).

    :param task_id: The child workflow's task_id. Used to register
        the running command so cancel_task/check_task can find the
        shell. E.g. ``"resp_abc123"``.
    :param conversation_id: The owning conversation's id. Looked up
        in the registry to find the right :class:`TerminalManager`.
        E.g. ``"conv_xyz789"``.
    :param shell_name: Name of the shell to run the command on,
        e.g. ``"default"``.
    :param command: Bash command text, e.g. ``"npm run dev"``.
    :param timeout_ms: Max milliseconds the command may run. ``None``
        means no bound.
    :param workspace_path: Per-conversation workspace directory
        (used by the registry to construct the manager on first
        use), e.g. ``"/home/user/.agent-plane/executor_storage/
        conv_xyz789/archer/workspace"``.
    :returns: ``dataclasses.asdict(result)`` of the
        :class:`~agent_plane.terminals.RunResult`. DBOS serializes
        this as the step's cached output.
    """
    from agent_plane.runtime import get_terminal_registry

    registry = get_terminal_registry()
    manager = registry.for_conversation(conversation_id, Path(workspace_path))
    manager.register_running_task(task_id, shell_name)
    try:
        # Pre-run cancel: if ``cancel_task`` was dispatched before
        # this step started (race with parent's dispatch_async),
        # honor it without running the command. Synthetic "killed"
        # result with exit 130 so ``_result_to_payload`` maps it to
        # status="cancelled" like a SIGINT'd run.
        if manager.is_cancel_requested(task_id):
            from agent_plane.terminals import RunResult

            cancelled = RunResult(
                stdout="",
                exit_code=130,
                status="killed",
                shell=shell_name,
            )
            return dataclasses.asdict(cancelled)

        # Run the command in a thread-pool thread (manager.run_sync
        # is blocking). Pass a ``cancel_predicate`` that the shell's
        # own read-loop thread polls — when it returns True, the
        # read loop sends SIGINT itself (from the thread that owns
        # the pexpect state), avoiding the cross-thread pexpect
        # races that a direct ``shell.interrupt()`` from a different
        # thread would trigger.
        #
        # The predicate closes over ``task_id`` and the manager; it
        # reads ``is_cancel_requested`` which is guarded by the
        # manager's own lock, so safe to call from any thread.
        result = await asyncio.to_thread(
            manager.run_sync,
            shell_name,
            command,
            timeout_ms,
            lambda: manager.is_cancel_requested(task_id),
        )
    finally:
        manager.unregister_running_task(task_id)
    # dataclasses.asdict makes the RunResult JSON-serializable so
    # DBOS's step-cache can persist it for replay / check_task.
    return dataclasses.asdict(result)


def _result_to_payload(
    task_id: str,
    result: dict[str, Any],
) -> AsyncWorkCompletePayload:
    """Map a :class:`RunResult` dict onto an ``AsyncWorkCompletePayload``.

    Terminal-specific translation of status axes (see module
    docstring):

    - ``"completed"``: forwarded verbatim; stdout + exit code flow
      into ``output`` as formatted text.
    - ``"killed"``: remapped to ``"cancelled"`` — the agent's
      ``cancel_task`` (or the ``timeout_ms`` bound) is responsible,
      so the agent-facing axis is "cancelled" rather than "failed."
    - ``"shell_crashed"``: ``"failed"`` — infrastructure failure.
    - ``"shell_busy"``: ``"failed"`` with a diagnostic message.
      Defensive; shouldn't happen in the normal async path.

    :param task_id: The background task's id — echoed on the
        payload so the parent drain's formatter has the id for the
        system message.
    :param result: The :class:`RunResult` as a dict (from
        :func:`dataclasses.asdict`). Expected keys: ``stdout``,
        ``exit_code``, ``status``, ``shell``.
    :returns: The constructed :class:`AsyncWorkCompletePayload`.
    """
    status = result.get("status")
    stdout = result.get("stdout") or ""
    exit_code = result.get("exit_code")
    shell_name = result.get("shell")
    # Render a compact, LLM-friendly output. Full stdout is already
    # capped at ~30 KB inline (§6.7); truncate_for_llm adds a safety
    # margin to match the @tool path's budget.
    body = (
        f"shell={shell_name!r} exit_code={exit_code} status={status!r}\n"
        f"stdout:\n{stdout}"
    )
    output = truncate_for_llm(body)

    if status == "completed":
        return AsyncWorkCompletePayload(
            task_id=task_id,
            kind=TERMINAL_KIND,
            status="completed",
            output=output,
            error=None,
        )
    if status == "killed":
        return AsyncWorkCompletePayload(
            task_id=task_id,
            kind=TERMINAL_KIND,
            status="cancelled",
            output=output,
            error=None,
        )
    # shell_crashed or shell_busy → infrastructure failure.
    error = {
        "message": f"terminal command {status}",
        "traceback": output,
    }
    return AsyncWorkCompletePayload(
        task_id=task_id,
        kind=TERMINAL_KIND,
        status="failed",
        output=output,
        error=error,
    )


@workflow()
async def background_terminal_workflow(
    parent_task_id: str,
    conversation_id: str,
    shell_name: str,
    command: str,
    timeout_ms: int | None,
    workspace_path: str,
) -> dict[str, Any]:
    """
    Run a ``terminal_run(synchronous=False)`` command in a child workflow.

    Lifecycle (matches ``background_tool_workflow`` for consistency
    with the existing drain protocol):

    1. Run :func:`_run_terminal_step` — a DBOS ``@step`` wrapping
       the blocking ``manager.run_sync`` call. Exceptions from here
       bubble to the ``except`` clause below.
    2. Translate the :class:`RunResult` into an
       :class:`AsyncWorkCompletePayload` via :func:`_result_to_payload`.
    3. On success: ``DBOS.send`` the payload to the parent on
       :data:`ASYNC_WORK_COMPLETE_TOPIC`.
    4. On exception: build a ``status="failed"`` payload via the
       shared :func:`format_failure_payload` (same traceback format
       the tool path uses), send it, then re-raise so DBOS records
       the workflow failure.
    5. Return the payload as the workflow output so
       :func:`check_task` can read it back on lookup.

    :param parent_task_id: The parent workflow's task_id, used as
        the destination of the ``async_work_complete`` signal.
    :param conversation_id: The owning conversation's id.
    :param shell_name: Name of the shell to run the command on,
        e.g. ``"default"`` or ``"dev"``.
    :param command: Bash command text, e.g. ``"sleep 30"``.
    :param timeout_ms: Max milliseconds the command may run.
    :param workspace_path: Per-conversation workspace directory.
    :returns: The payload as a plain dict (DBOS serializes the
        workflow output as JSON).
    """
    task_id = get_workflow_id()

    try:
        raw_result = await _run_terminal_step(
            task_id=task_id,
            conversation_id=conversation_id,
            shell_name=shell_name,
            command=command,
            timeout_ms=timeout_ms,
            workspace_path=workspace_path,
        )
        payload = _result_to_payload(task_id, raw_result)
    except BaseException as exc:  # noqa: BLE001 — workflow boundary
        # Matches background_tool_workflow: catch BaseException so
        # DBOS cancellation still signals the parent's drain, then
        # re-raise after sending so DBOS records the failure.
        error = format_failure_payload(exc)
        payload = AsyncWorkCompletePayload(
            task_id=task_id,
            kind=TERMINAL_KIND,
            status="failed",
            output=error["message"],
            error=error,
        )
        await _send_payload(parent_task_id, payload)
        raise

    await _send_payload(parent_task_id, payload)
    return _payload_to_dict(payload)


async def _send_payload(
    parent_task_id: str, payload: AsyncWorkCompletePayload
) -> None:
    """Send the completion payload to the parent workflow's drain.

    Thin wrapper around ``dbos_send_async`` so the workflow body
    stays readable. Async variant because the workflow body runs
    on the asyncio event loop.

    :param parent_task_id: The parent workflow's task_id (the
        destination of the signal).
    :param payload: The completion payload to deliver.
    """
    await dbos_send_async(
        parent_task_id,
        _payload_to_dict(payload),
        topic=ASYNC_WORK_COMPLETE_TOPIC,
    )


def _payload_to_dict(payload: AsyncWorkCompletePayload) -> dict[str, Any]:
    """Serialize the payload dataclass to a JSON-safe dict.

    DBOS serializes workflow outputs and signals as JSON; the
    dataclass must be rendered as a plain dict.

    :param payload: The :class:`AsyncWorkCompletePayload` to
        serialize.
    :returns: A dict with the payload's field values.
    """
    return dataclasses.asdict(payload)
