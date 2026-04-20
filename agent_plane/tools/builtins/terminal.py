"""Terminal builtin tools.

Three tools that together give agents persistent bash-shell access:
``terminal_run``, ``terminal_list``, ``terminal_close``. They operate
on conversation-scoped shells via the
:class:`TerminalManagerRegistry` set up at server startup (see
``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.4 and §6.9).

Current scope (synchronous only — §6.1 Option D, sync path only):

- ``terminal_run(command, shell="default", timeout_ms=None)`` —
  runs one command, blocks until the OSC 633 ``D`` marker, returns
  ``{stdout, exit_code, status, shell}``.
- ``terminal_list()`` — enumerate the current conversation's shells.
- ``terminal_close(shell="default")`` — kill and remove a named shell.

Not yet (Phase 2):

- ``synchronous=False`` on ``terminal_run`` (returns a task handle
  routed through the unified task lifecycle). Requires the
  async-task infrastructure from ``session_model_notes.md``.
- Mid-stream interactive input (``terminal_send`` / stdin).
"""

from __future__ import annotations

import json
from typing import Any

from agent_plane.runtime import get_terminal_registry
from agent_plane.terminals.manager import ShellCapExceeded, ShellNameInvalid
from agent_plane.tools.base import Tool, ToolContext

# Shared "description" text for the shell parameter, referenced in
# every tool's schema.
_SHELL_PARAM_DESC = (
    "Name of the shell to target. Names must start with a letter "
    "and contain only letters, digits, underscores, or hyphens "
    "(up to 64 chars). Defaults to 'default'. Use a custom name "
    "(e.g. 'dev', 'test') to create a parallel shell with isolated "
    "state from the default."
)


# ── terminal_run ──────────────────────────────────────────────────

_RUN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "terminal_run",
        "description": (
            "Run a shell command in a persistent bash shell scoped to "
            "this conversation. Shell state (cwd, environment variables, "
            "sourced scripts) persists across calls within the same "
            "conversation — `cd /tmp` in one call is still in /tmp on "
            "the next. By default blocks until the command completes "
            "and returns captured stdout plus exit code. Pass "
            "synchronous=false for long-running commands to get a "
            "task_id back immediately; then use check_task(task_id) to "
            "poll progress and cancel_task(task_id) to stop. Large "
            "outputs are head+tail truncated with the full log written "
            "to .agent_plane/terminal/<shell>-<n>.log in the workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "shell": {
                    "type": "string",
                    "description": _SHELL_PARAM_DESC,
                    "default": "default",
                },
                "timeout_ms": {
                    "type": ["integer", "null"],
                    "description": (
                        "Maximum milliseconds the command may run. When "
                        "the timeout fires, the command is Ctrl-C'd "
                        "(SIGINT) and the response status is 'killed'. "
                        "The shell stays alive for subsequent commands. "
                        "Null (default) means no bound."
                    ),
                    "default": None,
                },
                "synchronous": {
                    "type": "boolean",
                    "description": (
                        "When true (default) block until the command "
                        "completes and return stdout + exit code in the "
                        "tool result. When false, return immediately "
                        "with a task_id handle — use check_task(task_id) "
                        "to poll partial stdout and cancel_task(task_id) "
                        "to stop. When the command eventually completes, "
                        "its result auto-delivers as a system message "
                        "between LLM iterations so the agent doesn't "
                        "have to poll to pick it up."
                    ),
                    "default": True,
                },
            },
            "required": ["command"],
        },
    },
}


class TerminalRunTool(Tool):
    """Run a command in a per-conversation persistent bash shell.

    See module docstring. Requires ``ctx.conversation_id`` and
    ``ctx.workspace`` to be populated — tests or code paths that
    construct a bare :class:`ToolContext` will get a clear error.
    """

    @classmethod
    def name(cls) -> str:
        """Return the tool's fixed name.

        :returns: ``"terminal_run"``.
        """
        return "terminal_run"

    @classmethod
    def description(cls) -> str:
        """Return a human-readable description for tool discovery.

        :returns: Same prose as the LLM-facing schema description.
        """
        return str(_RUN_SCHEMA["function"]["description"])

    def get_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema.

        :returns: The schema dict with ``type``, ``function``,
            ``parameters`` fields.
        """
        return _RUN_SCHEMA

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """Run the command and return a JSON-encoded result.

        Only handles the synchronous path. Async calls
        (``synchronous=false``) bypass this method — :meth:`is_async`
        returns True and the workflow body routes to
        :meth:`dispatch_async`.

        :param arguments: JSON with keys ``command``, optional
            ``shell``, optional ``timeout_ms``.
        :param ctx: Must have ``conversation_id`` and ``workspace``
            populated.
        :returns: JSON string encoding ``{stdout, exit_code, status,
            shell}`` or an error payload ``{status, error}``.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        command = parsed.get("command", "")
        shell_name = parsed.get("shell", "default")
        timeout_ms = parsed.get("timeout_ms")

        err = _validate_ctx(ctx)
        if err is not None:
            return err
        if not command:
            return json.dumps({"status": "error", "error": "empty command"})

        registry = get_terminal_registry()
        assert ctx.conversation_id is not None  # _validate_ctx guarantees
        assert ctx.workspace is not None
        manager = registry.for_conversation(ctx.conversation_id, ctx.workspace)

        try:
            result = manager.run_sync(shell_name, command, timeout_ms=timeout_ms)
        except ShellNameInvalid as exc:
            return json.dumps({"status": "shell_name_invalid", "error": str(exc)})
        except ShellCapExceeded as exc:
            return json.dumps({"status": "shell_cap_exceeded", "error": str(exc)})

        return json.dumps(
            {
                "stdout": result.stdout,
                "exit_code": result.exit_code,
                "status": result.status,
                "shell": result.shell,
            }
        )

    def is_async(self, arguments: str | None = None) -> bool:
        """Return True iff the call explicitly set ``synchronous=False``.

        Default is synchronous — matches prior behavior and the
        common case (``echo``, ``ls``, short-lived commands). The
        LLM has to opt in for a command it knows will run long
        (``npm run dev``, ``pytest --watch``).

        :param arguments: JSON-encoded argument string. ``None`` or
            unparseable → treat as synchronous (safe default; a
            malformed call will fail loud in ``invoke`` anyway).
        :returns: True iff ``synchronous`` is present and false-y.
        """
        if not arguments:
            return False
        try:
            parsed: dict[str, Any] = json.loads(arguments)
        except (ValueError, TypeError):
            return False
        sync_flag = parsed.get("synchronous", True)
        return sync_flag is False

    async def dispatch_async(
        self,
        *,
        parent_task_id: str,
        parent_conversation_id: str,
        agent_id: str,
        agent_name: str,
        arguments: str,
        workspace_path: str | None,
    ) -> Any:
        """Start a ``background_terminal_workflow`` and return a handle.

        Called by the workflow body when :meth:`is_async` returned
        True. Creates a ``kind="terminal"`` task row pinned to a
        fresh task_id, launches the child workflow, and returns an
        ``_AsyncToolHandle`` that the parent's tool-dispatch loop
        serializes into the LLM's function_call_output.

        Fails loud on missing ``parent_conversation_id`` or
        ``workspace_path`` — the terminal tool can't run without
        them, and silently succeeding would produce a ghost task.

        :param parent_task_id: The parent workflow's task_id.
        :param parent_conversation_id: The owning conversation id.
            Must be non-empty.
        :param agent_id: The owning agent id.
        :param agent_name: Tool name — recorded on the child task
            row. Passed in (rather than read from ``self.name()``)
            because the runtime already has it and it matches the
            other tool paths' contract.
        :param arguments: JSON argument string from the LLM.
        :param workspace_path: Per-conversation workspace directory
            as a string. Must be non-empty.
        :returns: An ``_AsyncToolHandle`` with the new task id +
            LLM-facing message.
        :raises RuntimeError: When the required context is missing —
            would indicate a framework bug.
        """
        from agent_plane.runtime import get_task_store
        from agent_plane.runtime.background_terminal_workflow import (
            TERMINAL_KIND,
            background_terminal_workflow,
        )
        from agent_plane.runtime.durability import (
            SetWorkflowID,
            start_workflow,
        )
        from agent_plane.runtime.workflow import (
            _async_handle_message,
            _AsyncToolHandle,
            _to_thread,
        )

        if not parent_conversation_id:
            raise RuntimeError(
                "terminal_run async dispatch requires a conversation_id "
                "— parent task's context did not propagate one"
            )
        if not workspace_path:
            raise RuntimeError(
                "terminal_run async dispatch requires a workspace_path "
                "— parent task's context did not propagate one"
            )

        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        command = parsed.get("command", "")
        shell_name = parsed.get("shell", "default")
        timeout_ms = parsed.get("timeout_ms")
        if not command:
            # Fail fast with a handle-shaped response so the LLM sees
            # a clear error rather than a silently failed background
            # task. We still create a handle so the message path is
            # consistent, but the task itself never starts.
            raise RuntimeError("terminal_run requires a non-empty command")

        # Register the task BEFORE starting the child workflow so a
        # racing ``cancel_task`` that arrives before the child's own
        # register_running_task finds the task id and can mark it
        # cancelled via ``request_cancel``. The child workflow's
        # step checks ``is_cancel_requested`` before running the
        # command — eliminating the "cancel lost in race" failure
        # mode without needing the shell to pre-exist.
        #
        # NOTE: we DO NOT pre-spawn the shell here. Doing so from
        # the parent workflow's thread caused cross-thread pexpect
        # state issues (SIGINT from the cancel thread racing the
        # child thread's read loop). The child spawns lazily on its
        # first ``manager.run_sync`` call, which owns the shell
        # from a single thread.
        from pathlib import Path as _Path

        from agent_plane.runtime import get_terminal_registry

        registry = get_terminal_registry()
        manager = registry.for_conversation(parent_conversation_id, _Path(workspace_path))
        task_store = get_task_store()

        def _create_row_and_register() -> Any:
            row = task_store.create(
                conversation_id=parent_conversation_id,
                agent_id=agent_id,
                agent_name=agent_name,
                root_task_id=parent_task_id,
                kind=TERMINAL_KIND,
            )
            # Register the task → shell mapping so cancel_task can
            # find it. The shell itself doesn't exist yet; that's
            # fine — request_cancel only checks the task's
            # registration, not the shell's existence.
            manager.register_running_task(row.id, shell_name)
            return row

        new_task = await _to_thread(_create_row_and_register)

        def _start() -> None:
            # Pin the DBOS workflow_uuid to the new task_id so
            # check_task / cancel_task can look up the workflow by
            # task_id (same pattern used by the @tool path).
            with SetWorkflowID(new_task.id):
                start_workflow(
                    background_terminal_workflow,
                    parent_task_id,
                    parent_conversation_id,
                    shell_name,
                    command,
                    timeout_ms,
                    workspace_path,
                )

        await _to_thread(_start)

        return _AsyncToolHandle(
            task_id=new_task.id,
            tool_name=self.name(),
            status="in_progress",
            message=_async_handle_message(new_task.id, self.name()),
        )


# ── terminal_list ─────────────────────────────────────────────────

_LIST_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "terminal_list",
        "description": (
            "List the names of all currently-open shells in this "
            "conversation. Returns an empty list if no shells have been "
            "spawned yet."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


class TerminalListTool(Tool):
    """Enumerate the current conversation's shells by name."""

    @classmethod
    def name(cls) -> str:
        """:returns: ``"terminal_list"``."""
        return "terminal_list"

    @classmethod
    def description(cls) -> str:
        """:returns: Same prose as the LLM-facing schema description."""
        return str(_LIST_SCHEMA["function"]["description"])

    def get_schema(self) -> dict[str, Any]:
        """:returns: The OpenAI schema dict."""
        return _LIST_SCHEMA

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """Return JSON-encoded ``{"shells": [name, ...]}``.

        If the conversation has no manager (never touched terminal_run),
        returns an empty list rather than creating an empty manager —
        no side effects from a list call.

        :param arguments: Unused — the schema has no parameters.
        :param ctx: Must have ``conversation_id``.
        :returns: JSON string with the shell names in insertion order.
        """
        err = _validate_ctx(ctx, require_workspace=False)
        if err is not None:
            return err
        registry = get_terminal_registry()
        assert ctx.conversation_id is not None
        # Peek without creating a manager: active_conversation_ids()
        # only reveals existence, but we need shell names. For "no
        # shells yet" we can short-circuit by checking the active list.
        if ctx.conversation_id not in registry.active_conversation_ids():
            return json.dumps({"shells": []})
        # A manager exists; fetch it (workspace is only used at create
        # time, which already happened, so ``ctx.workspace`` is
        # ignored — but pass it for consistency). If ``ctx.workspace``
        # is None we use a sentinel ``Path(".")`` that won't actually
        # be used.
        from pathlib import Path

        ws = ctx.workspace if ctx.workspace is not None else Path(".")
        manager = registry.for_conversation(ctx.conversation_id, ws)
        return json.dumps({"shells": manager.list_shells()})


# ── terminal_close ────────────────────────────────────────────────

_CLOSE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "terminal_close",
        "description": (
            "Close a shell and discard its state. Frees a slot under "
            "the per-conversation 10-shell cap so a new named shell "
            "can be created. Idempotent — closing a shell that doesn't "
            "exist is not an error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "shell": {
                    "type": "string",
                    "description": _SHELL_PARAM_DESC,
                    "default": "default",
                },
            },
        },
    },
}


class TerminalCloseTool(Tool):
    """Kill and remove a named shell from the current conversation."""

    @classmethod
    def name(cls) -> str:
        """:returns: ``"terminal_close"``."""
        return "terminal_close"

    @classmethod
    def description(cls) -> str:
        """:returns: Same prose as the LLM-facing schema description."""
        return str(_CLOSE_SCHEMA["function"]["description"])

    def get_schema(self) -> dict[str, Any]:
        """:returns: The OpenAI schema dict."""
        return _CLOSE_SCHEMA

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """Close the named shell and return JSON ``{"closed": bool}``.

        ``closed`` is True if a shell was actually killed, False if
        no shell with that name existed (idempotent).

        :param arguments: JSON with optional ``shell`` key; defaults
            to ``"default"``.
        :param ctx: Must have ``conversation_id``.
        :returns: JSON string with the closed-or-not outcome.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        shell_name = parsed.get("shell", "default")

        err = _validate_ctx(ctx, require_workspace=False)
        if err is not None:
            return err
        registry = get_terminal_registry()
        assert ctx.conversation_id is not None
        if ctx.conversation_id not in registry.active_conversation_ids():
            # No manager = no shell to close; idempotent success.
            return json.dumps({"closed": False, "shell": shell_name})

        from pathlib import Path

        ws = ctx.workspace if ctx.workspace is not None else Path(".")
        manager = registry.for_conversation(ctx.conversation_id, ws)
        closed = manager.close(shell_name)
        return json.dumps({"closed": closed, "shell": shell_name})


# ── terminal_send_input ───────────────────────────────────────────

# Default wait-for-output budgets. These match OpenAI Agents SDK's
# ``write_stdin`` defaults (250 ms typing, 5 s polling) — the LLM-
# ergonomics story is covered in
# designs/PERSISTENT_TERMINAL_RESEARCH.md §6.12. ``None`` on the
# ``yield_time_ms`` parameter picks whichever default applies based
# on whether ``chars`` is empty; explicit values skip the auto-bump.
_SEND_INPUT_DEFAULT_YIELD_TYPING_MS = 250
_SEND_INPUT_DEFAULT_YIELD_POLLING_MS = 5_000
_SEND_INPUT_YIELD_FLOOR_MS = 50
_SEND_INPUT_YIELD_CEILING_MS = 30_000
# Inner quiescence window — if no bytes arrive for this long, we
# break out of the yield loop early. Avoids waiting the full budget
# when a fast command (``ls``) has already answered. Set lower than
# the floor to ensure the quiescence path fires for small typing
# defaults too.
_SEND_INPUT_QUIESCENCE_WINDOW_MS = 40

_SEND_INPUT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "terminal_send_input",
        "description": (
            "Send keystrokes/bytes to the stdin of a running async "
            "terminal_run task — enables driving interactive programs "
            "(vim, less, sqlite3, read prompts, password dialogs, "
            "REPLs). Only works on task_ids from "
            "terminal_run(synchronous=false). After writing, waits up "
            "to yield_time_ms for the program to react, then returns "
            "the streaming stdout delta AND the rendered screen so "
            "you can see what changed in one call. Common escape "
            "sequences (all JSON-encodable strings): '\\u0003' for "
            "Ctrl-C, '\\u0004' for Ctrl-D / EOF, '\\u001b' for "
            "Escape, '\\u001b[A'/'B'/'C'/'D' for Up/Down/Right/Left "
            "arrows, '\\t' for Tab, '\\n' for Enter, '\\u007f' for "
            "Backspace. For programs that want text followed by "
            "Enter, send e.g. '\\n'-terminated text in a single "
            "call. Pass chars='' to just poll for output without "
            "typing anything (useful when a command is slowly "
            "producing output and you want to wait for more)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": (
                        "Task id returned by a prior "
                        "terminal_run(synchronous=false) call. The "
                        "target must still be running — after the "
                        "task enters a terminal status, further "
                        "terminal_send_input calls return "
                        "{'delivered': false, 'reason': "
                        "'task_no_longer_running'}."
                    ),
                },
                "chars": {
                    "type": "string",
                    "description": (
                        "Bytes (as a string) to write to the task's "
                        "stdin. Pass '' (empty) to poll for output "
                        "without typing. Control characters and "
                        "escape sequences pass through literally — "
                        "see the tool description for common ones."
                    ),
                },
                "yield_time_ms": {
                    "type": ["integer", "null"],
                    "description": (
                        "How long (milliseconds) to wait after "
                        "writing for the program to produce output "
                        "before returning. Null (default) picks 250 "
                        "for non-empty chars (typing latency) or "
                        "5000 for empty chars (pure poll). Explicit "
                        "values are clamped to [50, 30000]; explicit "
                        "values skip the empty-chars auto-bump."
                    ),
                    "default": None,
                },
            },
            "required": ["task_id", "chars"],
        },
    },
}


def _resolve_yield_time_ms(yield_time_ms: int | None, *, chars_empty: bool) -> int:
    """Pick the effective yield-time given the caller's argument.

    :param yield_time_ms: Caller's raw input. ``None`` means
        "pick a default based on intent."
    :param chars_empty: Whether the caller passed ``chars=""``
        (pure poll). Used only for the default path; explicit
        numeric values are honored verbatim (clamped to the
        module ceiling/floor).
    :returns: The effective wait budget in milliseconds.
    """
    if yield_time_ms is None:
        return (
            _SEND_INPUT_DEFAULT_YIELD_POLLING_MS
            if chars_empty
            else _SEND_INPUT_DEFAULT_YIELD_TYPING_MS
        )
    return max(
        _SEND_INPUT_YIELD_FLOOR_MS,
        min(_SEND_INPUT_YIELD_CEILING_MS, yield_time_ms),
    )


class TerminalSendInputTool(Tool):
    """Write input to a running async ``terminal_run`` task's stdin."""

    @classmethod
    def name(cls) -> str:
        """:returns: ``"terminal_send_input"``."""
        return "terminal_send_input"

    @classmethod
    def description(cls) -> str:
        """:returns: Same prose as the LLM-facing schema description."""
        return str(_SEND_INPUT_SCHEMA["function"]["description"])

    def get_schema(self) -> dict[str, Any]:
        """:returns: The OpenAI schema dict."""
        return _SEND_INPUT_SCHEMA

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """Write input, wait for a quiescence-bounded response, return state.

        Dispatches via ``TerminalManager.send_input_to_task``, then
        loops polling the stdout ring buffer until either the
        ``yield_time_ms`` budget expires or output has gone silent
        for :data:`_SEND_INPUT_QUIESCENCE_WINDOW_MS`. Returns a
        shape that mirrors the terminal-kind ``check_task`` payload
        with an extra ``delivered`` boolean up front so the LLM can
        quickly see whether the input was even writable.

        :param arguments: JSON with ``task_id`` (required), ``chars``
            (required), and optional ``yield_time_ms``.
        :param ctx: Must have ``conversation_id`` populated.
        :returns: JSON string of shape
            ``{delivered, task_id, recent_activity?, screen?,
            reason?}``. ``reason`` is set only when
            ``delivered=False``.
        """
        try:
            parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"delivered": False, "reason": f"invalid arguments: {exc}"})

        task_id = parsed.get("task_id")
        chars = parsed.get("chars", "")
        yield_time_ms_raw = parsed.get("yield_time_ms")
        if not isinstance(task_id, str) or not task_id:
            return json.dumps({"delivered": False, "reason": "task_id is required"})
        if not isinstance(chars, str):
            return json.dumps({"delivered": False, "reason": "chars must be a string"})

        err = _validate_ctx(ctx, require_workspace=False)
        if err is not None:
            return err
        assert ctx.conversation_id is not None

        registry = get_terminal_registry()
        if ctx.conversation_id not in registry.active_conversation_ids():
            return json.dumps(
                {
                    "delivered": False,
                    "task_id": task_id,
                    "reason": "shell_unavailable",
                }
            )
        from pathlib import Path

        ws = ctx.workspace if ctx.workspace is not None else Path(".")
        manager = registry.for_conversation(ctx.conversation_id, ws)

        shell = manager.send_input_to_task(task_id, chars)
        if shell is None:
            # Either the task was never registered or its shell was
            # closed. Either way nothing to write to.
            return json.dumps(
                {
                    "delivered": False,
                    "task_id": task_id,
                    "reason": "task_no_longer_running",
                }
            )

        effective_yield_ms = _resolve_yield_time_ms(yield_time_ms_raw, chars_empty=not chars)
        self._wait_for_quiescence(manager, task_id, effective_yield_ms)

        # Render via the :class:`Shell` we captured at send time
        # rather than looking up by task_id again. Fast
        # interactions (e.g. ``read`` + ``echo`` completing inside
        # our yield window) cause the background workflow to
        # unregister the task before we get here; the shell object
        # itself outlives the task so its screen + ring buffer
        # still reflect the final state. This is the whole reason
        # ``send_input_to_task`` returns the Shell rather than a
        # bool. peek_task_stdout still goes through the manager so
        # it can advance the per-task cursor when the task is
        # still registered; when unregistered we accept losing
        # that advancement and the LLM leans on ``screen``.
        delta = manager.peek_task_stdout(task_id)
        screen = shell.rendered_screen()
        payload: dict[str, Any] = {
            "delivered": True,
            "task_id": task_id,
        }
        if delta is not None:
            text = delta.text
            if delta.lost_bytes > 0:
                text = f"[... {delta.lost_bytes} bytes evicted before this poll ...]\n{text}"
            payload["recent_activity"] = text
        payload["screen"] = screen
        return json.dumps(payload)

    def _wait_for_quiescence(
        self,
        manager: Any,
        task_id: str,
        budget_ms: int,
    ) -> None:
        """Spin until ``budget_ms`` elapses or stdout goes idle.

        Uses the ring buffer's monotonic ``total_bytes_written``
        counter to detect new output. Each iteration: sample the
        counter, if it grew since the last sample reset the
        quiescence timer, else if the timer has elapsed return
        early. This does NOT advance the task's peek cursor, so
        the final :meth:`peek_task_stdout` call in
        :meth:`invoke` still sees every byte that arrived during
        the wait.

        Without an early-return the LLM waits the full budget for
        fast commands; without a minimum quiescence window we
        return before even the first byte arrives for slow-to-
        react programs.

        :param manager: The :class:`TerminalManager` the task lives
            under.
        :param task_id: The task to watch.
        :param budget_ms: Maximum wait in milliseconds.
        """
        import time as _time

        deadline = _time.monotonic() + budget_ms / 1000
        poll_interval = 0.02  # 20 ms; fast enough to catch echoes
        quiescence_s = _SEND_INPUT_QUIESCENCE_WINDOW_MS / 1000
        last_count = manager.total_bytes_written_for_task(task_id)
        if last_count is None:
            # Task unknown at the start — nothing to wait on.
            return
        last_activity = _time.monotonic()
        while True:
            now = _time.monotonic()
            if now >= deadline:
                return
            current = manager.total_bytes_written_for_task(task_id)
            if current is None:
                # Task unregistered mid-wait.
                return
            if current > last_count:
                last_count = current
                last_activity = now
            elif now - last_activity >= quiescence_s:
                return
            _time.sleep(poll_interval)


# ── shared helpers ────────────────────────────────────────────────


def _validate_ctx(ctx: ToolContext, *, require_workspace: bool = True) -> str | None:
    """Validate the tool context has what the terminal tools need.

    Returns None if valid, otherwise a JSON-encoded error payload
    suitable for returning directly from ``invoke``.

    :param ctx: The :class:`ToolContext` to check.
    :param require_workspace: When True (default, for
        ``terminal_run``), ``ctx.workspace`` must be non-None.
        When False (for ``terminal_list`` / ``terminal_close``),
        only ``conversation_id`` is required.
    :returns: ``None`` if the context is valid, or a JSON-encoded
        error payload string.
    """
    if ctx.conversation_id is None:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "terminal tools require a conversation_id on the "
                    "ToolContext. The workflow path that invoked this "
                    "tool didn't populate it."
                ),
            }
        )
    if require_workspace and ctx.workspace is None:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "terminal_run requires a workspace path on the "
                    "ToolContext. The workflow path that invoked this "
                    "tool didn't populate it."
                ),
            }
        )
    return None
