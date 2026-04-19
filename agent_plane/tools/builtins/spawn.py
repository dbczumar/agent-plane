"""Spawn and check tools for sub-agent lifecycle management.

SpawnTool launches sub-agents as independent tasks via the
TaskStore interface. CheckSubAgentsTool returns their current
status without blocking. CancelSubAgentTool stops a running
sub-agent. See designs/STEERABLE_SUBAGENTS.md for the full design.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_plane.entities import (
    ConversationItem,
    MessageData,
    NewConversationItem,
    Task,
)
from agent_plane.entities.task import TERMINAL_STATUSES
from agent_plane.spec import AgentSpec
from agent_plane.stores import ConversationStore
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

# Maximum number of recent conversation items to include in
# check_sub_agents activity for non-completed sub-agents.
_ACTIVITY_TAIL = 5

# Maximum characters per content field in activity items.
# Longer content is truncated with a " [truncated]" suffix.
# Enough to capture a meaningful tool call or result (e.g. a
# code snippet, search output, or structured JSON arguments)
# without bloating the parent's prompt. At 5 items × 2000 chars
# the activity section is bounded to ~10k chars.
_ACTIVITY_MAX_CHARS = 2000


def _extract_output_text(output: list[dict[str, Any]]) -> str:
    """
    Extract final text from a task's output items.

    Walks the output list looking for assistant message items,
    then extracts ``output_text`` blocks from their ``content``
    arrays.

    :param output: The task's output items list, e.g.
        ``[{"type": "message", "role": "assistant",
        "content": [{"type": "output_text",
        "text": "Hello"}]}]``.
    :returns: Concatenated text, or empty string if no text
        items found.
    """
    parts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                text = block.get("text")
                if text is not None and text:
                    parts.append(text)
    return "\n\n".join(parts)


class SpawnSubAgentTool(Tool):
    """
    Launch one sub-agent as an asynchronous background task.

    Phase 3 replacement for ``SpawnTool`` (the old
    batch-spawn tool). The LLM calls ``spawn_sub_agent`` once
    per sub-agent it wants to dispatch — multiple parallel
    sub-agents come from the LLM emitting multiple tool_calls
    in one response, exactly like ``@tool(synchronous=False)``
    dispatch in Phase 2.

    Returns a JSON handle in the same shape as
    :class:`agent_plane.runtime.workflow._AsyncToolHandle`:
    ``{task_id, kind: "sub_agent", type, status, message}``.
    The result eventually auto-delivers via the unified
    ``async_work_complete`` topic — the LLM can also poll with
    ``check_task`` or abort with ``cancel_task``.

    :param sub_specs: Name-to-AgentSpec mapping for available
        sub-agents, e.g. ``{"researcher": AgentSpec(...)}``.
    """

    @classmethod
    def name(cls) -> str:
        """:returns: ``"spawn_sub_agent"`` (singular)."""
        return "spawn_sub_agent"

    @classmethod
    def description(cls) -> str:
        """:returns: Human-readable description of the tool."""
        return (
            "Launch ONE sub-agent as an asynchronous background "
            "task. Returns a handle immediately; the result "
            "auto-delivers as a system message when ready. To "
            "spawn multiple sub-agents in parallel, emit "
            "multiple spawn_sub_agent tool_calls in the same "
            "response — they dispatch concurrently."
        )

    def __init__(self, sub_specs: dict[str, AgentSpec]) -> None:
        """
        Initialize with the agent's available sub-agent specs.

        :param sub_specs: Name-to-AgentSpec mapping, e.g.
            ``{"researcher": AgentSpec(...)}``.
        """
        self._sub_specs = sub_specs

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema with dynamic ``type`` enum.

        The ``type`` enum is the list of sub-agent names the
        agent declares; the LLM can only dispatch declared
        types.

        :returns: Dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_spawn_sub_agent_schema(self._sub_specs)

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Spawn a single named sub-agent and return its handle.

        Phase 4: ``name`` is REQUIRED. The sub-agent's child
        conversation is titled ``"<type>:<name>"`` and parented
        to the caller's conversation, which lets later turns use
        ``send_to_sub_agent`` to continue the same conversation.
        Names must be unique within a parent (G36 enforced by the
        partial unique index in the migration).

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"type": "researcher", "name": "auth", "input":
            "find X"}'``.
        :param ctx: Server-side execution context with task and
            agent identity.
        :returns: JSON handle string with ``task_id``, ``kind``,
            ``type``, ``name``, ``status``, and ``message`` keys.
            On a duplicate name, returns
            ``{"error": "name_already_exists", ...}`` instead so
            the LLM can recover (call ``send_to_sub_agent`` or
            choose a different name).
        """
        args = _parse_spawn_sub_agent_args(arguments)
        if isinstance(args, str):
            return args
        sa_type: str = args["type"]
        sa_name: str = args["name"]
        sa_input: str = args["input"]
        if sa_type not in self._sub_specs:
            return json.dumps(
                {"error": f"unknown sub-agent type: {sa_type!r}"},
            )

        # _call_tool injects client_tools into the arguments JSON
        # before invoke — extract and remove (same convention as
        # the old SpawnTool).
        client_tools: list[dict[str, Any]] = args.get("client_tools", []) or []

        root_task_id = _resolve_root_task_id(ctx.task_id)
        parent_conversation_id = _resolve_parent_conversation_id(ctx.task_id)
        try:
            task_id = _spawn_one(
                agent_id=ctx.agent_id,
                agent_name=sa_type,
                sa_name=sa_name,
                user_input=sa_input,
                root_task_id=root_task_id,
                parent_conversation_id=parent_conversation_id,
                client_tools=client_tools,
            )
        except _NameAlreadyExistsToolError as exc:
            # Surface the partial-unique-index violation as a
            # clean LLM-facing error. The LLM can recover by
            # calling send_to_sub_agent on the same name OR by
            # picking a different name.
            return json.dumps(
                {
                    "error": "name_already_exists",
                    "message": str(exc),
                    "type": sa_type,
                    "name": sa_name,
                }
            )
        return json.dumps(
            {
                "task_id": task_id,
                "kind": "sub_agent",
                "type": sa_type,
                "name": sa_name,
                "status": "in_progress",
                "message": _spawn_handle_message(task_id, sa_type, sa_name),
            }
        )


def _build_spawn_sub_agent_schema(
    sub_specs: dict[str, AgentSpec],
) -> dict[str, Any]:
    """
    Build the OpenAI function schema for ``spawn_sub_agent``.

    The ``type`` parameter's enum is dynamic — derived from the
    keys of ``sub_specs`` so the LLM only sees the sub-agents
    the parent agent actually declares.

    :param sub_specs: Name-to-AgentSpec mapping.
    :returns: OpenAI function-format schema dict.
    """
    type_enum = sorted(sub_specs.keys())
    type_descriptions = {
        name: (spec.description or f"Sub-agent {name!r}.") for name, spec in sub_specs.items()
    }
    type_desc_text = "\n".join(f"  {name}: {desc}" for name, desc in type_descriptions.items())
    return {
        "type": "function",
        "function": {
            "name": SpawnSubAgentTool.name(),
            "description": SpawnSubAgentTool.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": type_enum,
                        "description": (
                            "The sub-agent type to dispatch. "
                            "Must be one of the declared "
                            "sub-agent names. Available "
                            f"types:\n{type_desc_text}"
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "A unique-within-this-parent label "
                            "for the sub-agent instance, e.g. "
                            "'auth' or 'payments'. Lets later "
                            "turns reuse the same conversation "
                            "via send_to_sub_agent. Names must "
                            "be distinct under one parent — a "
                            "duplicate (type, name) returns "
                            "name_already_exists; recover by "
                            "calling send_to_sub_agent OR "
                            "picking a different name."
                        ),
                    },
                    "input": {
                        "type": "string",
                        "description": (
                            "The user-input message to send "
                            "to the sub-agent. The sub-agent "
                            "treats this as the first user "
                            "turn in its conversation."
                        ),
                    },
                },
                "required": ["type", "name", "input"],
                "additionalProperties": False,
            },
        },
    }


def _parse_spawn_sub_agent_args(
    arguments: str,
) -> dict[str, Any] | str:
    """
    Parse and validate ``SpawnSubAgentTool`` arguments.

    :param arguments: Raw JSON string from the LLM.
    :returns: Parsed dict on success, or a JSON error string
        on failure (returned verbatim to the LLM so it can
        correct and retry).
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"error": f"invalid arguments: {exc}"})
    if not isinstance(args, dict):
        return json.dumps({"error": "arguments must be a JSON object"})
    for required in ("type", "name", "input"):
        if required not in args:
            return json.dumps({"error": f"missing required field: {required}"})
    return args


def _spawn_handle_message(task_id: str, sa_type: str, sa_name: str) -> str:
    """
    Build the LLM-facing instruction text on a fresh sub-agent handle.

    Mirrors ``_async_handle_message`` in workflow.py — the LLM
    needs the literal task_id, the name (so it can call
    ``send_to_sub_agent`` later in this conversation), and a
    pointer at ``check_task`` / ``cancel_task`` so it knows the
    result is not in this string.

    :param task_id: The new sub-agent task's ID.
    :param sa_type: The dispatched sub-agent's type name.
    :param sa_name: The dispatched sub-agent's instance name.
    :returns: A compact instruction string.
    """
    return (
        f"Sub-agent {sa_type}:{sa_name} dispatched. The result "
        f"will be auto-delivered as a system message when ready. "
        f"To continue this conversation in a later turn call "
        f"send_to_sub_agent(type={sa_type!r}, name={sa_name!r}, "
        f"input=...). To poll call check_task with "
        f"task_id={task_id!r}; to abort call cancel_task."
    )


class _NameAlreadyExistsToolError(Exception):
    """
    Internal-only exception so ``_spawn_one`` can signal a
    duplicate name without leaking SqlAlchemy's IntegrityError
    or the store's :class:`NameAlreadyExistsError` to the tool
    invocation layer (which would re-wrap and obscure the
    error). Caught and translated to a JSON tool-result by
    ``SpawnSubAgentTool.invoke``.
    """


# ── Spawn implementation ──────────────────────────────


def _resolve_root_task_id(task_id: str) -> str:
    """
    Determine the root task ID for sub-agent spawning.

    If the current task already has a ``root_task_id`` (it is
    itself a sub-agent), that value is propagated so
    ``root_task_id`` always points to the original top-level
    task regardless of nesting depth. Otherwise, the current
    task's own ID is used as the root.

    :param task_id: The current task/workflow ID,
        e.g. ``"task_abc123"``.
    :returns: The root task ID for spawned children.
    """
    from agent_plane.runtime import get_task_store

    task = get_task_store().get_sync(task_id)
    if task is not None and task.root_task_id is not None:
        return task.root_task_id
    return task_id


def _resolve_parent_conversation_id(task_id: str) -> str:
    """
    Return the conversation_id of the task that's calling spawn.

    Phase 4: child sub-agent conversations point at their
    immediate parent (not the root). For nested sub-agents (a
    sub-agent calling spawn_sub_agent), this returns the
    spawning sub-agent's own conversation, so
    ``list_sub_agents`` from inside that sub-agent surfaces its
    own children rather than the root's.

    :param task_id: The currently-executing task's id (the one
        whose tool ``invoke`` was called).
    :returns: The conversation_id of that task.
    :raises RuntimeError: If the task row cannot be found —
        means the framework's invariant (every tool runs inside
        an active task) is broken.
    """
    from agent_plane.runtime import get_task_store

    task = get_task_store().get_sync(task_id)
    if task is None:
        raise RuntimeError(
            f"task {task_id!r} not found — spawn must run inside an active workflow",
        )
    return task.conversation_id


def _spawn_one(
    *,
    agent_id: str,
    agent_name: str,
    sa_name: str,
    user_input: str,
    root_task_id: str,
    parent_conversation_id: str,
    client_tools: list[dict[str, Any]] | None = None,
) -> str:
    """
    Create a named child conversation, append the user input,
    create a task, and start execution.

    Phase 4: the child conversation's title is
    ``"<agent_name>:<sa_name>"`` and its
    ``parent_conversation_id`` points at the caller's
    conversation. The conv-store-level partial unique index
    rejects ``(parent, title)`` collisions; that error is
    re-raised as :class:`_NameAlreadyExistsToolError` for the
    invoker to translate to a clean tool result.

    :param agent_id: The root registered agent ID.
    :param agent_name: The sub-agent's TYPE (e.g. ``"coder"``).
    :param sa_name: The sub-agent's instance NAME (e.g.
        ``"auth"``). Combined with ``agent_name`` to form the
        conversation title.
    :param user_input: The user's input string for the
        sub-agent.
    :param root_task_id: The top-level task ID for tunneling.
    :param parent_conversation_id: The owning parent
        conversation's id (powers list_sub_agents + cascade
        delete).
    :param client_tools: Optional client-side tool schemas to
        propagate to the sub-agent.
    :returns: The created task ID (doubles as response_id).
    :raises _NameAlreadyExistsToolError: On duplicate
        ``(parent_conversation_id, title)``.
    """
    from agent_plane.runtime import (
        get_conversation_store,
        get_task_store,
    )
    from agent_plane.stores.conversation_store import (
        NameAlreadyExistsError,
    )

    conv_store = get_conversation_store()
    task_store = get_task_store()

    title = f"{agent_name}:{sa_name}"
    try:
        conv = conv_store.create_conversation(
            kind="sub_agent",
            title=title,
            parent_conversation_id=parent_conversation_id,
        )
    except NameAlreadyExistsError as exc:
        raise _NameAlreadyExistsToolError(
            f"a sub-agent of type {agent_name!r} with name "
            f"{sa_name!r} already exists in this conversation"
        ) from exc

    task = task_store.create(
        conversation_id=conv.id,
        agent_id=agent_id,
        agent_name=agent_name,
        root_task_id=root_task_id,
        # G74: explicitly mark spawned tasks so the parent loop's
        # auto-collect path can distinguish sub-agents from
        # top-level user turns and from async @tool work items.
        kind="sub_agent",
    )

    # Append user input as the first message in the sub-agent's
    # isolated conversation. response_id = task.id so the message
    # is associated with this sub-agent's response.
    conv_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id=task.id,
                data=MessageData(
                    role="user",
                    content=[
                        {"type": "input_text", "text": user_input},
                    ],
                ),
            ),
        ],
    )

    # Propagate client-side tool schemas so the sub-agent's LLM
    # knows about them. Tool calls are tunneled to the root
    # response via the park mechanism.
    task_store.start(task.id, tools=client_tools or None)
    return task.id


# ── Phase 4: send_to_sub_agent ────────────────────────


class SendToSubAgentTool(Tool):
    """
    Continue an existing named sub-agent's conversation.

    Phase 4 — strict continue-only counterpart to
    ``spawn_sub_agent``. The LLM identifies an existing child
    by ``(type, name)``; ``send_to_sub_agent`` looks up the
    titled child conversation, appends the user message as the
    first item of a NEW task on that conversation, and starts
    the sub-agent workflow. The sub-agent's
    ``_load_initial_history`` will see the full prior history
    plus the new message, so it "remembers" earlier turns.

    Errors:
    * ``sub_agent_not_found`` — no child conversation matches
      the ``(type, name)`` pair under this parent. The LLM
      should call ``spawn_sub_agent`` first.
    * ``sub_agent_busy`` — a task is already running on that
      child conversation. The LLM should wait for the current
      task's completion (it will arrive via the
      async_work_complete drain) before sending again.

    :param sub_specs: Name-to-AgentSpec mapping for available
        sub-agent types (same dict the spawn tool gets — used
        only to validate the requested ``type``).
    """

    @classmethod
    def name(cls) -> str:
        """:returns: ``"send_to_sub_agent"``."""
        return "send_to_sub_agent"

    @classmethod
    def description(cls) -> str:
        """:returns: Human-readable description of the tool."""
        return (
            "Continue an existing named sub-agent's conversation. "
            "The sub-agent must have been previously created with "
            "spawn_sub_agent(type=..., name=...) — strict continue-"
            "only. Returns sub_agent_not_found if no such child "
            "exists, or sub_agent_busy if another task is currently "
            "running on that conversation."
        )

    def __init__(self, sub_specs: dict[str, AgentSpec]) -> None:
        """
        Initialize with the agent's available sub-agent specs.

        :param sub_specs: Name-to-AgentSpec mapping.
        """
        self._sub_specs = sub_specs

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: Dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_send_to_sub_agent_schema(self._sub_specs)

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Continue an existing sub-agent's conversation.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"type": "coder", "name": "auth",
            "input": "now add tests"}'``.
        :param ctx: Server-side execution context.
        :returns: JSON handle string with ``task_id``, ``kind``,
            ``type``, ``name``, ``status``, and ``message``
            keys; on error, ``{"error": "...", ...}``.
        """
        args = _parse_send_to_sub_agent_args(arguments)
        if isinstance(args, str):
            return args
        sa_type: str = args["type"]
        sa_name: str = args["name"]
        sa_input: str = args["input"]
        if sa_type not in self._sub_specs:
            return json.dumps(
                {"error": f"unknown sub-agent type: {sa_type!r}"},
            )

        client_tools: list[dict[str, Any]] = args.get("client_tools", []) or []
        root_task_id = _resolve_root_task_id(ctx.task_id)
        parent_conversation_id = _resolve_parent_conversation_id(ctx.task_id)
        try:
            task_id = _send_to_one(
                agent_id=ctx.agent_id,
                agent_name=sa_type,
                sa_name=sa_name,
                user_input=sa_input,
                root_task_id=root_task_id,
                parent_conversation_id=parent_conversation_id,
                client_tools=client_tools,
            )
        except _SubAgentNotFoundError as exc:
            return json.dumps(
                {
                    "error": "sub_agent_not_found",
                    "message": str(exc),
                    "type": sa_type,
                    "name": sa_name,
                }
            )
        except _SubAgentBusyError as exc:
            return json.dumps(
                {
                    "error": "sub_agent_busy",
                    "message": str(exc),
                    "type": sa_type,
                    "name": sa_name,
                }
            )
        return json.dumps(
            {
                "task_id": task_id,
                "kind": "sub_agent",
                "type": sa_type,
                "name": sa_name,
                "status": "in_progress",
                "message": _spawn_handle_message(task_id, sa_type, sa_name),
            }
        )


def _build_send_to_sub_agent_schema(
    sub_specs: dict[str, AgentSpec],
) -> dict[str, Any]:
    """
    Build the OpenAI function schema for ``send_to_sub_agent``.

    :param sub_specs: Name-to-AgentSpec mapping.
    :returns: OpenAI function-format schema dict.
    """
    type_enum = sorted(sub_specs.keys())
    return {
        "type": "function",
        "function": {
            "name": SendToSubAgentTool.name(),
            "description": SendToSubAgentTool.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": type_enum,
                        "description": (
                            "The sub-agent type. Must match the "
                            "type passed to the original "
                            "spawn_sub_agent call."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "The instance name from the original "
                            "spawn_sub_agent call. Returns "
                            "sub_agent_not_found if no such "
                            "child exists."
                        ),
                    },
                    "input": {
                        "type": "string",
                        "description": (
                            "The new user-input message to "
                            "append to the sub-agent's "
                            "conversation. The sub-agent's "
                            "next turn sees the full prior "
                            "history plus this message."
                        ),
                    },
                },
                "required": ["type", "name", "input"],
                "additionalProperties": False,
            },
        },
    }


def _parse_send_to_sub_agent_args(
    arguments: str,
) -> dict[str, Any] | str:
    """
    Parse and validate ``SendToSubAgentTool`` arguments.

    :param arguments: Raw JSON string from the LLM.
    :returns: Parsed dict on success, or a JSON error string on
        failure (returned verbatim to the LLM so it can
        correct and retry).
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"error": f"invalid arguments: {exc}"})
    if not isinstance(args, dict):
        return json.dumps({"error": "arguments must be a JSON object"})
    for required in ("type", "name", "input"):
        if required not in args:
            return json.dumps({"error": f"missing required field: {required}"})
    return args


class _SubAgentNotFoundError(Exception):
    """Raised by ``_send_to_one`` when no matching child exists."""


class _SubAgentBusyError(Exception):
    """Raised by ``_send_to_one`` when the child has an in-flight task."""


def _send_to_one(
    *,
    agent_id: str,
    agent_name: str,
    sa_name: str,
    user_input: str,
    root_task_id: str,
    parent_conversation_id: str,
    client_tools: list[dict[str, Any]] | None = None,
) -> str:
    """
    Look up an existing named child conversation, append the
    user message, create a new task on it, and start execution.

    Phase 4 strict continue-only path. Resolves the child via
    ``(parent_conversation_id, title="<type>:<name>")``. Errors
    when the child is missing or has a non-terminal task already
    running.

    :param agent_id: The root registered agent ID.
    :param agent_name: The sub-agent's TYPE.
    :param sa_name: The sub-agent's instance NAME.
    :param user_input: The user's new input message.
    :param root_task_id: The top-level task ID for tunneling.
    :param parent_conversation_id: The owning parent
        conversation's id (must match the parent that spawned
        the original child).
    :param client_tools: Optional client-side tool schemas.
    :returns: The new task ID.
    :raises _SubAgentNotFoundError: If no child matches.
    :raises _SubAgentBusyError: If the child has a non-terminal
        task in flight.
    """
    from agent_plane.entities.task import TERMINAL_STATUSES
    from agent_plane.runtime import (
        get_conversation_store,
        get_task_store,
    )

    conv_store = get_conversation_store()
    task_store = get_task_store()

    title = f"{agent_name}:{sa_name}"
    children = conv_store.list_conversations(
        kind="sub_agent",
        parent_conversation_id=parent_conversation_id,
        # We do NOT need pagination here — same parent typically
        # has a small number of named sub-agents. The default
        # limit (20) plus an explicit bump-to-100 covers the
        # realistic worst case without complicating the lookup.
        limit=100,
    )
    match = next((c for c in children.data if c.title == title), None)
    if match is None:
        raise _SubAgentNotFoundError(
            f"no sub-agent of type {agent_name!r} named {sa_name!r} exists in this conversation",
        )

    # Reject if the child has a non-terminal task running.
    # Otherwise the new task would race with the in-flight one,
    # both producing assistant responses on the same conversation
    # and confusing the LLM.
    existing_tasks = task_store.list_tasks_sync(conversation_id=match.id)
    iter_existing = existing_tasks.data if hasattr(existing_tasks, "data") else existing_tasks
    for t in iter_existing:
        if t.status not in TERMINAL_STATUSES:
            raise _SubAgentBusyError(
                f"sub-agent {agent_name}:{sa_name} is busy with task {t.id!r}; "
                f"wait for its completion (it will auto-deliver via the drain) "
                f"before sending again",
            )

    task = task_store.create(
        conversation_id=match.id,
        agent_id=agent_id,
        agent_name=agent_name,
        root_task_id=root_task_id,
        kind="sub_agent",
    )

    conv_store.append(
        match.id,
        [
            NewConversationItem(
                type="message",
                response_id=task.id,
                data=MessageData(
                    role="user",
                    content=[
                        {"type": "input_text", "text": user_input},
                    ],
                ),
            ),
        ],
    )

    task_store.start(task.id, tools=client_tools or None)
    return task.id


# ── Phase 4: list_sub_agents ──────────────────────────


class ListSubAgentsTool(Tool):
    """
    List the named sub-agents that already exist under this conversation.

    Phase 4 — purely informational. The LLM uses the result to
    decide whether to call ``send_to_sub_agent`` (existing) or
    ``spawn_sub_agent`` (new). The output is intentionally a
    minimal ``{type, name}`` list — no timestamps, no task
    counts, no statuses — so it doesn't grow large or shift
    across turns for trivial reasons.
    """

    @classmethod
    def name(cls) -> str:
        """:returns: ``"list_sub_agents"``."""
        return "list_sub_agents"

    @classmethod
    def description(cls) -> str:
        """:returns: Human-readable description of the tool."""
        return (
            "List named sub-agents already created under this "
            "conversation. Returns {sub_agents: [{type, name}, "
            "...]}. Empty list when no sub-agents have been "
            "spawned. Use the result to decide between "
            "send_to_sub_agent (continue) and spawn_sub_agent "
            "(new)."
        )

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: Dict with ``"type": "function"`` and a
            ``"function"`` sub-dict; no parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": ListSubAgentsTool.name(),
                "description": ListSubAgentsTool.description(),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Return the named sub-agents under the caller's conversation.

        :param arguments: Ignored (the schema declares no
            parameters; the LLM occasionally passes ``"{}"``).
        :param ctx: Server-side execution context.
        :returns: JSON ``{"sub_agents": [{"type": ...,
            "name": ...}, ...]}``.
        """
        from agent_plane.runtime import get_conversation_store

        parent_conversation_id = _resolve_parent_conversation_id(ctx.task_id)
        conv_store = get_conversation_store()
        children = conv_store.list_conversations(
            kind="sub_agent",
            parent_conversation_id=parent_conversation_id,
            # 100 is a safe ceiling — agents that need more named
            # sub-agents than this are an antipattern; the LLM
            # would lose track regardless.
            limit=100,
        )
        result: list[dict[str, str]] = []
        for child in children.data:
            # Title is "<type>:<name>" — split into the LLM-
            # friendly fields. Skip rows whose title doesn't
            # match the convention (defensive — Phase-3
            # anonymous spawns left None titles, but those have
            # NULL parent_conversation_id and won't appear in
            # this query at all).
            if child.title is None or ":" not in child.title:
                continue
            sa_type, _, sa_name = child.title.partition(":")
            result.append({"type": sa_type, "name": sa_name})
        return json.dumps({"sub_agents": result})


# ── Check / result helpers ────────────────────────────


def _format_terminal_message(task: Task) -> str:
    """
    Build a human-readable message for a terminal sub-agent.

    Includes error details when available so the parent LLM can
    decide whether to retry, adjust input, or give up.

    :param task: A task in a terminal status (``"failed"``,
        ``"cancelled"``, etc.).
    :returns: A message like ``"Sub-agent 'X' failed:
        context_window_exceeded — Input too long."``.
    """
    # task.status is typed str but some construction paths assign the
    # TaskStatus enum directly; .value gives the lowercase form
    # consistently for f-string interpolation.
    status_str = task.status.value if hasattr(task.status, "value") else task.status
    base = f"Sub-agent {task.agent_name!r} finished with status: {status_str}."
    if task.error:
        # Defensive: task.error is dict[str, str] but may come
        # from DB JSON — missing keys get readable fallbacks so
        # the formatted message is always valid for the parent LLM.
        code = task.error.get("code", "unknown")
        message = task.error.get("message", "")
        # Append error details so the parent LLM can make an
        # informed decision (retry with shorter input, skip, etc.).
        base += f" Error: {code} — {message}" if message else f" Error: {code}."
    return base


def _task_to_result(task: Task) -> dict[str, str]:
    """
    Convert a :class:`Task` to a status-result dict.

    Returns the real task status (e.g. ``"in_progress"``,
    ``"completed"``, ``"failed"``). For completed tasks with
    output, extracts the final text. For all others, returns
    a descriptive message.

    :param task: The task, possibly in a terminal or
        non-terminal state.
    :returns: A dict with ``response_id``, ``agent_name``,
        ``status``, and ``output`` keys.
    """
    if task.status == "completed" and task.output:
        output_text = _extract_output_text(task.output)
    elif task.status in TERMINAL_STATUSES:
        output_text = _format_terminal_message(task)
    else:
        output_text = f"Sub-agent {task.agent_name!r} is still running."

    return {
        "response_id": task.id,
        "agent_name": task.agent_name,
        "status": task.status,
        "output": output_text,
    }


def _task_to_check_result(
    task: Task,
    conv_store: ConversationStore,
) -> dict[str, Any]:
    """
    Build a check result for a single sub-agent.

    Completed sub-agents get extracted output text and no
    activity. All others get recent conversation activity
    so the parent LLM can see what the sub-agent is doing
    (or what it was doing when it failed).

    :param task: The sub-agent's task.
    :param conv_store: For fetching recent conversation items.
    :returns: Result dict with ``response_id``, ``agent_name``,
        ``status``, ``output``, and ``recent_activity`` fields.
    """
    if task.status == "completed" and task.output:
        return {
            "response_id": task.id,
            "agent_name": task.agent_name,
            "status": task.status,
            "output": _extract_output_text(task.output),
            "recent_activity": None,
        }

    # Non-completed: include recent activity
    activity = _get_recent_activity(
        task.conversation_id,
        conv_store,
    )
    output: str | None = None
    if task.status in TERMINAL_STATUSES:
        output = _format_terminal_message(task)

    return {
        "response_id": task.id,
        "agent_name": task.agent_name,
        "status": task.status,
        "output": output,
        "recent_activity": activity,
    }


def _get_recent_activity(
    conversation_id: str,
    conv_store: ConversationStore,
) -> list[dict[str, str | None]]:
    """
    Fetch the last few conversation items and project them
    into a compact format for the parent LLM.

    :param conversation_id: The sub-agent's conversation ID,
        e.g. ``"conv_sub1"``.
    :param conv_store: For fetching items.
    :returns: List of compact activity dicts, chronological
        order, each content field truncated to
        ``_ACTIVITY_MAX_CHARS``.
    """
    page = conv_store.list_items(
        conversation_id,
        limit=_ACTIVITY_TAIL,
        order="desc",
    )
    # Reverse to chronological order (list_items desc gives
    # newest first).
    items = list(reversed(page.data))
    return [_project_activity_item(item) for item in items]


def _project_activity_item(
    item: ConversationItem,
) -> dict[str, str | None]:
    """
    Project a conversation item into a compact dict.

    Handles three item types: messages (user/assistant text),
    function calls (tool name + args), and function call
    outputs (tool name + result). All content fields are
    truncated to ``_ACTIVITY_MAX_CHARS``.

    :param item: A conversation item from the sub-agent's
        conversation.
    :returns: A compact dict with ``role``, ``type``, and
        content fields.
    """
    # Convert Pydantic model to dict so .get() works uniformly
    # across all data types (MessageData, FunctionCallData, etc.).
    data = item.data.model_dump()
    if item.type == "function_call":
        return {
            "role": "assistant",
            "type": "tool_call",
            "name": data.get("name"),
            "args": _truncate(
                data.get("arguments", ""),
            ),
        }
    if item.type == "function_call_output":
        return {
            "role": "tool",
            "type": "tool_result",
            "name": data.get("name"),
            "content": _truncate(
                data.get("output", ""),
            ),
        }
    # Message item — extract role and text content.
    role = data.get("role", "unknown")
    text_parts: list[str] = []
    for block in data.get("content", []):
        if isinstance(block, dict):
            text = block.get("text") or block.get("output_text")
            if text:
                text_parts.append(text)
        elif isinstance(block, str):
            text_parts.append(block)
    return {
        "role": role,
        "type": "text",
        "content": _truncate("\n".join(text_parts)),
    }


def _truncate(text: str) -> str:
    """
    Truncate text to ``_ACTIVITY_MAX_CHARS``.

    :param text: The input string.
    :returns: The original string if short enough, or a
        truncated version with ``" [truncated]"`` suffix.
    """
    if len(text) <= _ACTIVITY_MAX_CHARS:
        return text
    return text[:_ACTIVITY_MAX_CHARS] + " [truncated]"
