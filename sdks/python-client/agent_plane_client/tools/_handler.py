"""Adapter that turns ``@tool``-decorated functions into a ToolHandler.

The stream-layer ``ToolHandler`` takes a list of OpenAI-shape JSON
schemas and a single ``execute`` callable. Users who have written
tools with the ``@tool`` decorator (Python functions with type hints
and Google-style docstrings) shouldn't have to hand-roll that shape:
:func:`build_tool_handler` reads each function's tool metadata and
builds the handler for them.

Dispatch is by tool name. Calling an unknown tool raises — the SDK
surfaces the error back to the agent as a tool error.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable
from typing import Any

from .._tool_handler import ToolCallInfo, ToolHandler
from ._decorator import TOOL_MARKER_ATTR, ToolMetadata

# JSON Schema property injected for ``@tool(synchronous=False)``
# tools. Surfaces the per-call async-dispatch choice to the LLM
# inside ``parameters.properties`` (the spec-compliant home for
# tool arguments). The server reads ``arguments["synchronous"]``
# from each call and routes accordingly. The SDK strips the key
# before invoking the user's function so tool authors don't have
# to declare a ``synchronous`` parameter on their signatures.
_SYNCHRONOUS_PROPERTY_NAME = "synchronous"
_SYNCHRONOUS_PROPERTY_SCHEMA: dict[str, object] = {
    "type": "boolean",
    "description": (
        "Set to false to dispatch this call as a background "
        "task; you'll receive a {task_id, kind: 'client_tool'} "
        "handle immediately and the actual result will arrive "
        "as a [System: task ... completed] message in a later "
        "turn. Use false for long-running calls or when you "
        "want to fan out several calls in parallel; use true "
        "(or omit) for quick calls whose result you need before "
        "deciding the next step."
    ),
}


def build_tool_handler(functions: list[Callable[..., Any]]) -> ToolHandler:
    """Build a :class:`ToolHandler` from ``@tool``-decorated functions.

    Each function must carry tool metadata attached by the
    :func:`~agent_plane_client.tool` decorator (checked via
    :data:`TOOL_MARKER_ATTR`). The returned handler exposes the
    OpenAI-shape schemas the SDK sends to the server, and an
    ``execute`` callable that dispatches incoming tool calls by
    name.

    :param functions: List of ``@tool``-decorated Python functions,
        e.g. ``[get_current_time, search_docs]``. Each must be a
        module-level ``def`` or ``async def`` decorated with
        ``@tool``.
    :returns: A :class:`ToolHandler` ready to pass as
        ``session.tool_handler`` or via the ``tools=`` keyword on
        ``AgentPlaneClient.query`` / ``Session.query``.
    :raises TypeError: If any function is missing the ``@tool``
        marker (i.e. wasn't decorated).
    :raises ValueError: If two functions share the same tool name
        — tool names must be unique per handler.
    """
    if not functions:
        raise ValueError("build_tool_handler() requires at least one function")

    schemas: list[dict[str, object]] = []
    funcs_by_name: dict[str, Callable[..., Any]] = {}

    for fn in functions:
        meta: ToolMetadata | None = getattr(fn, TOOL_MARKER_ATTR, None)
        if meta is None:
            raise TypeError(
                f"{fn.__module__}.{fn.__qualname__} is not decorated with "
                f"@tool. Decorate it with `from agent_plane_client import tool` "
                f"and apply @tool above the function definition."
            )
        if meta.name in funcs_by_name:
            raise ValueError(
                f"Duplicate tool name {meta.name!r}: "
                f"{funcs_by_name[meta.name].__qualname__} and "
                f"{fn.__qualname__} both export the same name."
            )
        funcs_by_name[meta.name] = fn
        parameters = meta.json_schema
        # Phase 5 v2: for @tool(synchronous=False), inject a
        # ``synchronous`` boolean property into the parameters
        # schema so the LLM can request async dispatch per call.
        # Spec-compliant — `properties` is exactly where the
        # OpenAI tool-call spec puts argument schemas. Deep-copy
        # so we never mutate the metadata's shared json_schema
        # (other handlers, replays, etc. would observe the
        # mutation otherwise).
        if not meta.synchronous:
            parameters = _inject_synchronous_property(meta.json_schema)
        schema: dict[str, object] = {
            "type": "function",
            "function": {
                "name": meta.name,
                "description": meta.description,
                "parameters": parameters,
            },
        }
        schemas.append(schema)

    async def execute(call: ToolCallInfo) -> str:
        """Dispatch ``call`` to the matching ``@tool`` function.

        Sync functions run inline; async functions are awaited.
        The return value is JSON-serialized unless the function
        already returned a string (which is passed through).

        The ``synchronous`` argument (if present) is a routing
        hint consumed by the server to choose async vs sync
        dispatch — it is stripped before invoking the user's
        function so tool authors don't have to declare it.
        """
        fn = funcs_by_name.get(call.name)
        if fn is None:
            # The SDK will surface this back to the agent as a tool
            # error — this typically means the LLM invented a tool
            # name that wasn't in the schemas we sent.
            raise KeyError(f"Unknown tool {call.name!r}. Registered: {sorted(funcs_by_name)}")
        # Drop the routing-hint key. Use `dict(...)` rather than
        # popping in place so we don't mutate caller state.
        invoke_args = {k: v for k, v in call.arguments.items() if k != _SYNCHRONOUS_PROPERTY_NAME}
        result = fn(**invoke_args)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return result
        # Pydantic models and dataclasses commonly aren't JSON-ready
        # out of the box — ``default=str`` handles datetime/UUID/etc.
        return json.dumps(result, default=str)

    return ToolHandler(schemas=schemas, execute=execute)


def _inject_synchronous_property(json_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``json_schema`` with ``synchronous`` added to its properties.

    Used for ``@tool(synchronous=False)`` tools so the LLM sees
    the per-call async-dispatch choice as a real argument. The
    property is added but NOT marked required — the LLM may omit
    it (defaults to sync server-side).

    :param json_schema: The function's auto-derived JSON Schema
        (an ``object`` schema with ``properties``).
    :returns: A deep copy with ``properties[synchronous]`` set
        to :data:`_SYNCHRONOUS_PROPERTY_SCHEMA`. Never mutates
        the input.
    :raises ValueError: If ``json_schema`` already declares a
        ``synchronous`` property — the tool author would
        otherwise silently lose their declaration to ours, and
        a name collision means their tool can't safely use the
        async-dispatch hint anyway.
    """
    new_schema = copy.deepcopy(json_schema)
    properties = new_schema.setdefault("properties", {})
    if _SYNCHRONOUS_PROPERTY_NAME in properties:
        raise ValueError(
            f"@tool(synchronous=False) cannot be combined with a "
            f"function parameter named {_SYNCHRONOUS_PROPERTY_NAME!r}: "
            f"the SDK injects {_SYNCHRONOUS_PROPERTY_NAME!r} as the "
            f"per-call async-dispatch hint and would shadow the "
            f"tool's own argument. Rename the parameter."
        )
    properties[_SYNCHRONOUS_PROPERTY_NAME] = copy.deepcopy(_SYNCHRONOUS_PROPERTY_SCHEMA)
    return new_schema
