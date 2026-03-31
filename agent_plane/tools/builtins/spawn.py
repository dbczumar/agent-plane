"""Spawn and collect tools for sub-agent lifecycle management.

SpawnTool launches sub-agents as independent DBOS workflow tasks.
CollectTool waits for spawned sub-agents to complete and returns
their results. See designs/SUBAGENT.md for the full design.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent_plane.entities import (
    MessageData,
    NewConversationItem,
)
from agent_plane.runtime.durability import (
    WorkflowStatus,
    WorkflowStatusString,
    get_workflow_status,
)
from agent_plane.spec import AgentSpec
from agent_plane.tools.base import Tool

_logger = logging.getLogger(__name__)

# Polling interval for CollectTool's sync wait loop, in seconds.
# Uses DBOS.sleep() for replay safety.
_COLLECT_POLL_INTERVAL_S = 0.5

# DBOS statuses that mean the workflow is still running.
_DBOS_ACTIVE = frozenset({WorkflowStatusString.PENDING.value, WorkflowStatusString.ENQUEUED.value})

# Mapping from DBOS status strings to task status strings
# used only by CollectTool for result reporting.
_DBOS_TO_RESULT_STATUS: dict[str, str] = {
    WorkflowStatusString.SUCCESS.value: "completed",
    WorkflowStatusString.ERROR.value: "failed",
    WorkflowStatusString.CANCELLED.value: "cancelled",
    WorkflowStatusString.MAX_RECOVERY_ATTEMPTS_EXCEEDED.value: ("failed"),
}


def _extract_output_text(output: list[dict[str, Any]]) -> str:
    """
    Extract final text from a task's output items.

    Walks the output list, pulls items where ``type ==
    "output_text"``, and concatenates their text content.

    :param output: The task's output items list, e.g.
        ``[{"type": "output_text", "text": "Hello"}]``.
    :returns: Concatenated text, or empty string if no text items.
    """
    parts: list[str] = []
    for item in output:
        if item.get("type") == "output_text":
            text = item.get("text")
            if text is not None and text:
                parts.append(text)
    return "\n\n".join(parts)


class SpawnTool(Tool):
    """
    Launch sub-agents as independent DBOS workflow tasks.

    The LLM calls ``spawn_sub_agents`` with a list of
    ``{name, input}`` pairs. Each sub-agent gets its own
    conversation and task. Returns response IDs immediately —
    use ``collect_sub_agents`` to gather results.

    :param sub_specs: Name-to-AgentSpec mapping for available
        sub-agents, e.g. ``{"researcher": AgentSpec(...)}``.
    """

    def __init__(self, sub_specs: dict[str, AgentSpec]) -> None:
        """
        Initialize the spawn tool.

        :param sub_specs: Name-to-AgentSpec mapping for available
            sub-agents, e.g. ``{"researcher": AgentSpec(...)}``.
        """
        self._sub_specs = sub_specs

    @property
    def name(self) -> str:
        """
        Tool name for dispatch and schema registration.

        :returns: ``"spawn_sub_agents"``.
        """
        return "spawn_sub_agents"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema with dynamic
        sub-agent names and descriptions.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return _build_spawn_schema(self._sub_specs)

    def invoke(self, arguments: str) -> str:
        """
        Spawn sub-agents from the LLM's tool call arguments.

        Parses the arguments JSON, validates sub-agent names,
        creates a conversation and task for each, and starts
        execution. The ``root_task_id`` and ``agent_id`` fields
        are injected by the agent loop before dispatch.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"agents": [...], "root_task_id": "task_abc",
            "agent_id": "ag_xyz"}'``.
        :returns: JSON with response IDs, e.g.
            ``'{"response_ids": ["resp_child1"]}'``.
        """
        args = _parse_spawn_args(arguments)
        if isinstance(args, str):
            return args

        return _invoke_spawn(
            args,
            self._sub_specs,
        )


class CollectTool(Tool):
    """
    Wait for spawned sub-agent tasks to complete and return
    their results.

    Uses sync DBOS primitives (``get_workflow_status``) because
    ``invoke()`` runs inside a DBOS workflow thread. Polls with
    ``DBOS.sleep()`` for replay safety.
    """

    @property
    def name(self) -> str:
        """
        Tool name for dispatch and schema registration.

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

    def invoke(self, arguments: str) -> str:
        """
        Collect results from spawned sub-agents.

        Blocks until all sub-agents reach a terminal state or
        the timeout expires.

        :param arguments: JSON-encoded arguments string, e.g.
            ``'{"response_ids": ["resp_1"], "timeout": 60}'``.
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

    for field in ("agents", "root_task_id", "agent_id"):
        if field not in args:
            return json.dumps({"error": f"missing required field: {field}"})
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


def _invoke_spawn(
    args: dict[str, Any],
    sub_specs: dict[str, AgentSpec],
) -> str:
    """
    Execute the spawn logic for validated arguments.

    :param args: Parsed arguments dict with ``agents``,
        ``root_task_id``, and ``agent_id``.
    :param sub_specs: Name-to-AgentSpec mapping.
    :returns: JSON with response IDs.
    """
    agents_list: list[dict[str, str]] = args["agents"]
    root_task_id: str = args["root_task_id"]
    agent_id: str = args["agent_id"]

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
        )
        response_ids.append(task_id)

    return json.dumps({"response_ids": response_ids})


def _spawn_one(
    *,
    agent_id: str,
    agent_name: str,
    user_input: str,
    root_task_id: str,
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

    task_store.start(task.id)
    return task.id


# ── Collect implementation ────────────────────────────


def _collect_all(
    response_ids: list[str],
    timeout: float | None,
) -> list[dict[str, str]]:
    """
    Wait for all sub-agent tasks to complete and extract results.

    Uses a sync polling loop: ``DBOS.sleep()`` + workflow status
    check + deadline check. ``DBOS.sleep()`` is used (not
    ``time.sleep()``) so DBOS can track the sleep for replay.

    :param response_ids: List of task/response IDs to collect.
    :param timeout: Maximum seconds to wait. ``None`` means no
        deadline.
    :returns: List of result dicts with ``response_id``,
        ``agent_name``, ``status``, and ``output`` keys.
    """
    from dbos import DBOS

    deadline = time.monotonic() + timeout if timeout else None
    pending = set(response_ids)
    collected: dict[str, dict[str, str]] = {}

    while pending:
        if deadline is not None and time.monotonic() >= deadline:
            break
        still_pending: set[str] = set()
        for task_id in pending:
            result = _check_task_status(task_id)
            if result is not None:
                collected[task_id] = result
            else:
                still_pending.add(task_id)
        pending = still_pending
        if pending:
            DBOS.sleep(_COLLECT_POLL_INTERVAL_S)

    # Build results in the original order
    results: list[dict[str, str]] = []
    for task_id in response_ids:
        if task_id in collected:
            results.append(collected[task_id])
        else:
            results.append(_build_timeout_result(task_id))
    return results


def _resolve_agent_name(task_id: str) -> str:
    """
    Look up the agent name for a spawned sub-agent task.

    Reads the task row from the database. The row is always
    present because ``_spawn_one`` creates it before this
    function is ever called.

    :param task_id: The sub-agent's task ID,
        e.g. ``"task_child1"``.
    :returns: The agent name, or ``"unknown"`` if not found.
    """
    from agent_plane.runtime import get_task_store

    task = get_task_store().get_sync(task_id)
    if task is not None:
        return task.agent_name
    return "unknown"


def _check_task_status(
    task_id: str,
) -> dict[str, str] | None:
    """
    Check if a sub-agent workflow has reached a terminal state.

    :param task_id: The sub-agent's task ID.
    :returns: A result dict if terminal, or ``None`` if still
        running.
    """
    wf_status: WorkflowStatus | None = get_workflow_status(task_id)
    if wf_status is None:
        return None

    dbos_str = str(wf_status.status)
    if dbos_str in _DBOS_ACTIVE:
        return None

    mapped = _DBOS_TO_RESULT_STATUS.get(dbos_str, "failed")
    agent_name = _resolve_agent_name(task_id)

    if mapped == "completed" and wf_status.output is not None:
        # DBOS stores the workflow return value in wf_status.output.
        # For agent workflows, this is a dict with an "output" key
        # holding the list of output items. Empty list fallback is
        # safe: _extract_output_text handles it gracefully.
        output_items = wf_status.output.get("output", [])
        output_text = _extract_output_text(output_items)
    else:
        output_text = f"Sub-agent {agent_name!r} did not complete (status: {mapped})."

    return {
        "response_id": task_id,
        "agent_name": agent_name,
        "status": mapped,
        "output": output_text,
    }


def _build_timeout_result(task_id: str) -> dict[str, str]:
    """
    Build a result dict for a timed-out sub-agent.

    :param task_id: The sub-agent's task ID.
    :returns: A result dict with status ``"incomplete"``.
    """
    agent_name = _resolve_agent_name(task_id)

    # Get the current DBOS status for the error message
    current_status = "in_progress"
    wf_status = get_workflow_status(task_id)
    if wf_status is not None:
        current_status = _DBOS_TO_RESULT_STATUS.get(str(wf_status.status), "in_progress")

    return {
        "response_id": task_id,
        "agent_name": agent_name,
        "status": "incomplete",
        "output": (f"Sub-agent {agent_name!r} did not complete (status: {current_status})."),
    }
