"""Client-specified tools — tools whose schemas are supplied by the API caller.

These tools are defined at request time rather than baked into the agent
image. The caller provides standard OpenAI-format function schemas; when
the LLM invokes one, the runtime persists the ``function_call`` output
items, streams them to the client, and completes the response. The client
handles execution externally and continues via ``previous_response_id``.

Public API:
- ``ClientSideTool``: A :class:`~agent_plane.tools.base.Tool` that must
  never be executed server-side — its ``invoke()`` raises ``RuntimeError``.
- ``ClientSideToolSpec``: Configuration for one client-side tool (name
  and schema only — no callback URL or headers).
- ``parse_client_side_tool_spec``: Parse one raw OpenAI tool dict into a
  :class:`ClientSideToolSpec`.
- ``parse_client_side_tool_specs``: Parse a list of raw OpenAI tool dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_plane.tools.base import Tool, ToolContext, is_valid_tool_name


@dataclass
class ClientSideToolSpec:
    """
    Configuration for one client-specified tool.

    Holds the information needed to present the tool to the LLM.
    Execution is handled entirely by the API caller — the runtime
    never invokes client-side tools server-side.

    :param name: Tool function name, e.g. ``"get_weather"``. Must
        match the ``function.name`` in the OpenAI schema.
    :param schema: Standard OpenAI-format function tool object, e.g.
        ``{"type": "function", "function": {"name": "get_weather",
        "description": "...", "parameters": {...}}}``.
    :param synchronous: Phase 5 — when ``True`` (default), the
        runtime parks the workflow until the client PATCHes
        ``tool_results`` (the legacy single-PATCH pattern). When
        ``False``, the runtime instead creates a kind="client_tool"
        task, returns a ``{task_id, kind: "client_tool"}`` handle
        to the LLM inline, and the parent loop's
        ``async_work_complete`` drain delivers the eventual result
        from the client's ``async_tool_results`` PATCH.
    """

    name: str
    schema: dict[str, Any]
    synchronous: bool = True


class ClientSideTool(Tool):
    """
    A tool that is presented to the LLM but executed by the API caller.

    When the LLM invokes this tool, the runtime persists the
    ``function_call`` output items, streams them to the client, and
    completes the response. The client handles execution and continues
    via ``previous_response_id``.

    ``invoke()`` raises ``RuntimeError`` — client-side tools must never
    be dispatched through the tool execution path.

    :param spec: The :class:`ClientSideToolSpec` describing this tool.
    """

    def __init__(self, spec: ClientSideToolSpec) -> None:
        """
        :param spec: The :class:`ClientSideToolSpec` describing this tool.
        """
        self._spec = spec

    def name(self) -> str:  # type: ignore[override]
        """
        :returns: The tool function name, e.g. ``"get_weather"``.
        """
        return self._spec.name

    @classmethod
    def description(cls) -> str:
        """
        :returns: Human-readable description of the tool.
        """
        return "Client-side tool executed by the frontend."

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: The schema dict as supplied by the caller.
        """
        return self._spec.schema

    def is_async(self) -> bool:
        """
        Return ``True`` for client tools the caller marked
        ``"synchronous": false`` in their POST body.

        Phase 5 — async client tools take the
        ``async_work_complete`` drain path instead of the
        legacy parking + ``tool_results`` PATCH path. The
        runtime checks this in ``_handle_tool_calls`` to route
        between the two execution models.

        :returns: ``True`` for ``synchronous=False`` client
            tools, ``False`` (default) for the legacy
            parking model.
        """
        return not self._spec.synchronous

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Raise ``RuntimeError`` — client-side tools must never be executed
        server-side.

        The workflow detects client-side tool calls via
        ``ToolManager.is_client_side_tool()`` before dispatching, so this
        method should never be reached in normal operation.

        :param arguments: JSON-encoded arguments string (unused).
        :param ctx: Server-side execution context (unused).
        :raises RuntimeError: Always — indicates a workflow bug.
        """
        raise RuntimeError(
            f"ClientSideTool {self._spec.name!r} must not be invoked server-side. "
            "The workflow must detect client-side tools via ToolManager.is_client_side_tool() "
            "and complete the response without executing them."
        )


def parse_client_side_tool_spec(raw: dict[str, Any]) -> ClientSideToolSpec:
    """
    Parse a raw OpenAI tool dict into a :class:`ClientSideToolSpec`.

    Validates that the dict is a well-formed OpenAI function tool schema
    with a ``function.name``. Recognizes the optional Phase 5
    ``"synchronous": false`` field; default ``True`` preserves the
    legacy parking behavior.

    :param raw: A dict in standard OpenAI function tool format, e.g.::

            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {...}}
                },
                "synchronous": true   // Phase 5; defaults to true
            }

    :returns: A :class:`ClientSideToolSpec` with the name, schema,
        and synchronous flag.
    :raises ValueError: If ``type`` is not ``"function"``,
        ``function.name`` is missing, or ``synchronous`` is not a
        bool.
    """
    if raw.get("type") != "function":
        raise ValueError(
            f"client-specified tools must have type 'function', got {raw.get('type')!r}"
        )

    func = raw.get("function")
    if not isinstance(func, dict):
        raise ValueError("client-specified tool missing 'function' object")

    name = func.get("name")
    if not name:
        raise ValueError("client-specified tool missing function.name")

    if not is_valid_tool_name(name):
        raise ValueError(f"Invalid tool name {name!r}: must match [a-zA-Z0-9_-]{{1,256}}")

    synchronous_raw = raw.get("synchronous", True)
    if not isinstance(synchronous_raw, bool):
        raise ValueError(
            f"client-specified tool {name!r}: 'synchronous' must be a bool, "
            f"got {type(synchronous_raw).__name__}"
        )

    return ClientSideToolSpec(name=name, schema=raw, synchronous=synchronous_raw)


def parse_client_side_tool_specs(
    raw_tools: list[dict[str, Any]],
) -> list[ClientSideToolSpec]:
    """
    Parse a list of raw tool dicts into :class:`ClientSideToolSpec` objects.

    :param raw_tools: List of raw tool dicts from the API request, each
        in standard OpenAI function format.
    :returns: A list of :class:`ClientSideToolSpec` instances.
    :raises ValueError: If any tool in the list is malformed.
    """
    return [parse_client_side_tool_spec(raw) for raw in raw_tools]


__all__ = [
    "ClientSideTool",
    "ClientSideToolSpec",
    "parse_client_side_tool_spec",
    "parse_client_side_tool_specs",
]
