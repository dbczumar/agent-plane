"""Spawn and collect tools for sub-agent lifecycle management.

SpawnTool launches sub-agents as independent tasks via the
TaskStore interface. CollectTool waits for spawned sub-agents to
complete and returns their results. See designs/SUBAGENT.md for
the full design.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_plane.entities import (
    MessageData,
    NewConversationItem,
    Task,
)
from agent_plane.entities.task import TERMINAL_STATUSES
from agent_plane.spec import AgentSpec
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

# Polling / wait timeout used by CollectTool's wait_sync calls.
_COLLECT_POLL_INTERVAL_S = 0.5


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
    conversation and task. Returns response IDs immediately —
    use ``collect_sub_agents`` to gather results.

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


class CollectTool(Tool):
    """
    Wait for spawned sub-agent tasks to complete and return
    their results.

    Uses ``TaskStore.wait_sync`` and ``TaskStore.get_sync`` to
    block until sub-agents finish. Runs inside a synchronous
    workflow context.
    """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"collect_sub_agents"``.
        """
        return "collect_sub_agents"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_collect_schema()

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Collect results from spawned sub-agents.

        Blocks until all sub-agents reach a terminal state or
        the timeout expires.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"response_ids": ["resp_1"], "timeout": 60}'``.
        :param ctx: Server-side execution context (unused by
            collect, required by the :class:`Tool` interface).
        :returns: JSON with results per sub-agent.
        """
        args = _parse_collect_args(arguments)
        if isinstance(args, str):
            return args

        response_ids: list[str] = args["response_ids"]
        timeout: float | None = args.get("timeout")

        results = _collect_all(response_ids, timeout)
        return json.dumps({"results": results})


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
        "Use collect_sub_agents() to gather results.",
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


def _build_collect_schema() -> dict[str, Any]:
    """
    Build the OpenAI-format schema for collect_sub_agents.

    :returns: The tool schema dict.
    """
    return {
        "type": "function",
        "function": {
            "name": "collect_sub_agents",
            "description": (
                "Wait for spawned sub-agent tasks to complete and return their results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Response IDs returned by spawn_sub_agents()."),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": ("Maximum seconds to wait. Omit to wait indefinitely."),
                    },
                },
                "required": ["response_ids"],
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


def _parse_collect_args(
    arguments: str,
) -> dict[str, Any] | str:
    """
    Parse and validate CollectTool arguments.

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


# ── Collect implementation ────────────────────────────


def _collect_all(
    response_ids: list[str],
    timeout: float | None,
) -> list[dict[str, str]]:
    """
    Wait for all sub-agent tasks to complete and extract results.

    Delegates to ``TaskStore.wait_sync`` per task, splitting the
    remaining timeout budget across tasks sequentially.

    :param response_ids: List of task/response IDs to collect.
    :param timeout: Maximum seconds to wait across all tasks.
        ``None`` means no deadline.
    :returns: List of result dicts with ``response_id``,
        ``agent_name``, ``status``, and ``output`` keys.
    """
    import time

    from agent_plane.runtime import get_task_store

    task_store = get_task_store()
    deadline = time.monotonic() + timeout if timeout is not None else None

    results: list[dict[str, str]] = []
    for task_id in response_ids:
        remaining = _remaining_timeout(deadline)
        task = task_store.wait_sync(task_id, timeout=remaining)
        results.append(_task_to_result(task))
    return results


def _remaining_timeout(
    deadline: float | None,
) -> float | None:
    """
    Compute remaining seconds until the deadline.

    :param deadline: Absolute monotonic deadline, or ``None``
        for no deadline.
    :returns: Seconds remaining (clamped to 0), or ``None``.
    """
    if deadline is None:
        return None
    import time

    return max(0.0, deadline - time.monotonic())


def _task_to_result(task: Task) -> dict[str, str]:
    """
    Convert a :class:`Task` to a collect-result dict.

    :param task: The enriched task, possibly in a terminal or
        non-terminal state.
    :returns: A dict with ``response_id``, ``agent_name``,
        ``status``, and ``output`` keys.
    """
    if task.status in TERMINAL_STATUSES:
        status = task.status
    else:
        # Timed out — task is still running.
        status = "incomplete"

    if task.status == "completed" and task.output:
        output_text = _extract_output_text(task.output)
    else:
        output_text = f"Sub-agent {task.agent_name!r} did not complete (status: {status})."

    return {
        "response_id": task.id,
        "agent_name": task.agent_name,
        "status": status,
        "output": output_text,
    }
