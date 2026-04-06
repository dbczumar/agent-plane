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
_ACTIVITY_MAX_CHARS = 300


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


class SpawnTool(Tool):
    """
    Launch sub-agents as independent tasks via the TaskStore.

    The LLM calls ``spawn_sub_agents`` with a list of
    ``{name, input}`` pairs. Each sub-agent gets its own
    conversation and task. Returns response IDs immediately.
    Use ``check_sub_agents`` to retrieve results.

    :param sub_specs: Name-to-AgentSpec mapping for available
        sub-agents, e.g. ``{"researcher": AgentSpec(...)}``.
    """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"spawn_sub_agents"``.
        """
        return "spawn_sub_agents"

    def __init__(self, sub_specs: dict[str, AgentSpec]) -> None:
        """
        Initialize the spawn tool.

        :param sub_specs: Name-to-AgentSpec mapping for available
            sub-agents, e.g. ``{"researcher": AgentSpec(...)}``.
        """
        self._sub_specs = sub_specs

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema with dynamic
        sub-agent names and descriptions.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_spawn_schema(self._sub_specs)

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Spawn sub-agents from the LLM's tool call arguments.

        Parses the arguments JSON, validates sub-agent names,
        creates a conversation and task for each, and starts
        execution. Server-side identity (``task_id``,
        ``agent_id``) comes from ``ctx``, not the LLM arguments.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"agents": [{"name": "researcher",
            "input": "find X"}]}'``.
        :param ctx: Server-side execution context with task
            and agent identity.
        :returns: JSON with response IDs, e.g.
            ``'{"response_ids": ["resp_child1"]}'``.
        """
        args = _parse_spawn_args(arguments)
        if isinstance(args, str):
            return args

        # _call_tool injects client_tools into the arguments
        # JSON before calling invoke — extract and remove it.
        client_tools: list[dict[str, Any]] = args.pop("client_tools", [])

        root_task_id = _resolve_root_task_id(ctx.task_id)
        return _invoke_spawn(
            args,
            self._sub_specs,
            root_task_id=root_task_id,
            agent_id=ctx.agent_id,
            client_tools=client_tools,
        )


class CheckSubAgentsTool(Tool):
    """
    Non-blocking status check for specified sub-agents.

    Checks only the sub-agents whose response IDs are passed
    in — the caller need not check all spawned sub-agents at
    once. Returns immediately with each sub-agent's current
    status. Completed sub-agents include their extracted
    output text; in-progress sub-agents include recent
    conversation activity.
    """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"check_sub_agents"``.
        """
        return "check_sub_agents"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_check_schema()

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Check status of specified sub-agents.

        Returns immediately with each sub-agent's current
        status. No blocking, no waiting.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"response_ids": ["resp_1", "resp_2"]}'``.
        :param ctx: Server-side execution context (unused by
            check, required by the :class:`Tool` interface).
        :returns: JSON with results per sub-agent.
        """
        args = _parse_check_args(arguments)
        if isinstance(args, str):
            return args

        response_ids: list[str] = args["response_ids"]

        from agent_plane.runtime import (
            get_conversation_store,
            get_task_store,
        )

        task_store = get_task_store()
        conv_store = get_conversation_store()

        results: list[dict[str, Any]] = []
        for tid in response_ids:
            task = task_store.get_sync(tid)
            if task is None:
                results.append(
                    {
                        "response_id": tid,
                        "status": "not_found",
                        "output": None,
                        "recent_activity": None,
                    }
                )
                continue
            results.append(
                _task_to_check_result(task, conv_store),
            )
        return json.dumps({"results": results})


class CancelSubAgentTool(Tool):
    """
    Cancel a running sub-agent task.

    Delegates to ``task_store.cancel`` — non-blocking.
    The sub-agent workflow observes the cancellation on its
    next DBOS checkpoint and winds down.
    """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"cancel_sub_agent"``.
        """
        return "cancel_sub_agent"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_cancel_schema()

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Cancel a running sub-agent.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"response_id": "resp_1"}'``.
        :param ctx: Server-side execution context (unused by
            cancel, required by the :class:`Tool` interface).
        :returns: JSON with cancellation confirmation.
        """
        try:
            args = json.loads(arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            return json.dumps({"error": f"invalid arguments: {exc}"})

        response_id = args.get("response_id")
        if not response_id:
            return json.dumps(
                {"error": "missing required field: response_id"},
            )

        from agent_plane.runtime import get_task_store

        task_store = get_task_store()
        task_store.cancel(response_id)
        return json.dumps(
            {
                "status": "cancelled",
                "response_id": response_id,
            }
        )


# ── Schema builders ───────────────────────────────────


def _build_spawn_schema(
    sub_specs: dict[str, AgentSpec],
) -> dict[str, Any]:
    """
    Build the OpenAI-format schema for spawn_sub_agents.

    :param sub_specs: Name-to-AgentSpec mapping.
    :returns: The tool schema dict.
    """
    agent_names = sorted(sub_specs.keys())
    desc_lines = [
        "Launch one or more sub-agents as independent "
        "parallel tasks. Returns response IDs immediately. "
        "Use check_sub_agents() to retrieve results.",
        "",
        "Available sub-agents:",
    ]
    for agent_name in agent_names:
        spec = sub_specs[agent_name]
        label = spec.description or "No description."
        desc_lines.append(f"- {agent_name}: {label}")

    return {
        "type": "function",
        "function": {
            "name": "spawn_sub_agents",
            "description": "\n".join(desc_lines),
            "parameters": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": agent_names,
                                    "description": ("Sub-agent name."),
                                },
                                "input": {
                                    "type": "string",
                                    "description": ("The task or question for the sub-agent."),
                                },
                            },
                            "required": ["name", "input"],
                        },
                        "description": ("List of sub-agents to spawn with their inputs."),
                    },
                },
                "required": ["agents"],
            },
        },
    }


def _build_check_schema() -> dict[str, Any]:
    """
    Build the OpenAI-format schema for check_sub_agents.

    :returns: The tool schema dict.
    """
    return {
        "type": "function",
        "function": {
            "name": "check_sub_agents",
            "description": (
                "Check the current status of one or more spawned "
                "sub-agent tasks. Pass only the response IDs you "
                "want to check — you do not need to check all "
                "spawned sub-agents at once. Returns immediately "
                "with each specified sub-agent's current status, "
                "output (if completed), and recent conversation "
                "activity (if still running). Does not wait."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One or more response IDs returned by spawn_sub_agents to check."
                        ),
                    },
                },
                "required": ["response_ids"],
            },
        },
    }


def _build_cancel_schema() -> dict[str, Any]:
    """
    Build the OpenAI-format schema for cancel_sub_agent.

    :returns: The tool schema dict.
    """
    return {
        "type": "function",
        "function": {
            "name": "cancel_sub_agent",
            "description": (
                "Cancel a running sub-agent task. The sub-agent "
                "will stop execution and its status will become "
                "'cancelled'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response_id": {
                        "type": "string",
                        "description": ("The response ID of the sub-agent to cancel."),
                    },
                },
                "required": ["response_id"],
            },
        },
    }


# ── Argument parsing ──────────────────────────────────


def _parse_spawn_args(
    arguments: str,
) -> dict[str, Any] | str:
    """
    Parse and validate SpawnTool arguments.

    :param arguments: Raw JSON string from the LLM.
    :returns: Parsed dict on success, or a JSON error string
        on failure.
    """
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"error": f"invalid arguments: {exc}"})

    if "agents" not in args:
        return json.dumps({"error": "missing required field: agents"})
    result: dict[str, Any] = args
    return result


def _parse_check_args(
    arguments: str,
) -> dict[str, Any] | str:
    """
    Parse and validate CheckSubAgentsTool arguments.

    :param arguments: Raw JSON string from the LLM.
    :returns: Parsed dict on success, or a JSON error string
        on failure.
    """
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"error": f"invalid arguments: {exc}"})

    if "response_ids" not in args:
        return json.dumps({"error": "missing required field: response_ids"})
    result: dict[str, Any] = args
    return result


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


def _invoke_spawn(
    args: dict[str, Any],
    sub_specs: dict[str, AgentSpec],
    *,
    root_task_id: str,
    agent_id: str,
    client_tools: list[dict[str, Any]],
) -> str:
    """
    Execute the spawn logic for validated arguments.

    :param args: Parsed arguments dict with ``agents`` list.
    :param sub_specs: Name-to-AgentSpec mapping.
    :param root_task_id: The resolved root task ID for
        tunneling, e.g. ``"task_root1"``.
    :param agent_id: The registered agent ID,
        e.g. ``"ag_xyz789"``.
    :param client_tools: OpenAI-format schemas for client-side
        tools to propagate to sub-agents, e.g.
        ``[{"type": "function", "function": {...}}]``.
    :returns: JSON with response IDs.
    """
    agents_list: list[dict[str, str]] = args["agents"]

    response_ids: list[str] = []
    for entry in agents_list:
        sa_name: str = entry["name"]
        sa_input: str = entry["input"]

        if sa_name not in sub_specs:
            return json.dumps({"error": f"unknown sub-agent: {sa_name!r}"})

        task_id = _spawn_one(
            agent_id=agent_id,
            agent_name=sa_name,
            user_input=sa_input,
            root_task_id=root_task_id,
            client_tools=client_tools,
        )
        response_ids.append(task_id)

    return json.dumps({"response_ids": response_ids})


def _spawn_one(
    *,
    agent_id: str,
    agent_name: str,
    user_input: str,
    root_task_id: str,
    client_tools: list[dict[str, Any]] | None = None,
) -> str:
    """
    Create a conversation, append the user input, create a task,
    and start execution for a single sub-agent.

    :param agent_id: The root registered agent ID,
        e.g. ``"ag_xyz789"``.
    :param agent_name: The sub-agent name,
        e.g. ``"researcher"``.
    :param user_input: The user's input string for the
        sub-agent.
    :param root_task_id: The top-level task ID for tunneling,
        e.g. ``"task_abc123"``.
    :param client_tools: Optional list of client-side tool schemas
        (OpenAI format) to propagate to the sub-agent. The
        sub-agent's LLM needs these schemas so it can invoke
        client-side tools; calls are tunneled back to the root
        response via ``root_task_id``.
    :returns: The created task ID (doubles as response_id).
    """
    # Lazy imports to avoid circular dependency — these modules
    # import from runtime which imports from tools.
    from agent_plane.runtime import (
        get_conversation_store,
        get_task_store,
    )

    conv_store = get_conversation_store()
    task_store = get_task_store()

    conv = conv_store.create_conversation(kind="sub_agent")

    task = task_store.create(
        conversation_id=conv.id,
        agent_id=agent_id,
        agent_name=agent_name,
        root_task_id=root_task_id,
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
    base = f"Sub-agent {task.agent_name!r} finished with status: {task.status}."
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
