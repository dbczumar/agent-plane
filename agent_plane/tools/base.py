"""Abstract base class for agent tools."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from pathlib import Path  # used by ToolContext.workspace type hint
from typing import Any

# Tool name constraint: alphanumeric plus ``_`` and ``-``, up to
# 256 characters. OpenAI enforces 1–64 but other providers allow
# longer names, and client-side tools come from the user.
TOOL_NAME_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]{1,256}$")


def is_valid_tool_name(name: str) -> bool:
    """
    Check whether a tool name is valid: 1–256 characters,
    alphanumeric plus ``_`` and ``-``.

    :param name: The tool name to validate, e.g. ``"get_weather"``.
    :returns: ``True`` if the name is valid, ``False`` otherwise.
    """
    return TOOL_NAME_RE.match(name) is not None


@dataclass(frozen=True)
class ToolContext:
    """
    Execution context passed to every tool invocation.

    Provides server-side metadata that tools may need but
    which the LLM does not supply (task identity, agent
    identity, workspace path). Individual tools read the
    fields they need and ignore the rest.

    :param task_id: The current task/workflow ID,
        e.g. ``"task_abc123"``.
    :param agent_id: The registered agent ID,
        e.g. ``"ag_xyz789"``.
    :param workspace: Per-conversation persistent working
        directory. ``code_sandbox`` uses it as subprocess cwd,
        ``upload_file`` resolves paths against it. ``None``
        when no workspace is available (e.g. tests).
    """

    task_id: str
    agent_id: str
    workspace: Path | None = None


class Tool(abc.ABC):
    """
    Abstract base class for all tools available to the agent.

    Each tool has a unique name, an OpenAI-format schema for the
    LLM, and an ``invoke`` method that executes the tool and
    returns a string result.

    Subclasses must implement ``name()`` as a ``@classmethod``
    (for tools with a fixed name, e.g. ``SpawnTool.name()``)
    or as a regular method (for tools whose name depends on
    instance state, e.g. ``McpTool``).
    """

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """
        Unique tool name used for dispatch and schema registration.

        :returns: The tool name, e.g. ``"load_skill"``.
        """

    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """
        Human-readable description of what the tool does.

        Must be readable without instantiation — used for tool
        discovery (e.g. the onboarding assistant's
        ``list_builtin_tools``) and should match the description
        in :meth:`get_schema`.

        :returns: The tool's description string.
        """

    @abc.abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI Chat Completions tool schema.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict describing the tool's name,
            description, and parameters.
        """

    @abc.abstractmethod
    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Execute the tool with the given arguments.

        :param arguments: JSON-encoded arguments string from
            the LLM, e.g. ``'{"name": "summarize"}'``.
        :param ctx: Server-side execution context with task
            and agent identity.
        :returns: The tool's string result.
        """

    def cancel(self) -> None:
        """
        Cancel an in-progress invocation.

        Called by ``call_tool_with_timeout`` when the deadline
        expires. Subprocess-based tools override this to kill
        the child process. Default is a no-op.
        """

    def is_async(self) -> bool:
        """
        Return ``True`` if this tool runs in a background workflow.

        Built-in and synchronous tools return ``False`` (the
        default) — the runtime calls ``invoke()`` inline and
        passes the string result back to the LLM in the same
        iteration. Tools that override this to return ``True``
        signal the runtime to start a
        :func:`~agent_plane.runtime.background_tool_workflow.background_tool_workflow`,
        return a task handle to the LLM immediately, and deliver
        the eventual result via the ``async_work_complete`` topic.

        :returns: ``False`` by default.
        """
        return False
