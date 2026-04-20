"""
Async-dispatch demo tool set for ``ap chat --tools async_demo``.

Ships one tool — ``slow_compute`` — whose schema declares a
``synchronous`` boolean in ``parameters.properties``. When the
LLM sets ``synchronous=false``, the server dispatches the call
as a ``kind="client_tool"`` task and the python-client SDK's
D6 lifecycle (``_run_async_tool_body``) runs the tool body on
an ``asyncio.Task`` and PATCHes ``async_tool_results`` when
done. The parent agent then sees
``[System: task ... (client_tool) completed]\\n<body>`` as a
system message on its next turn.

The tool itself is deliberately boring (sleep + echo) — the
point is to show the async protocol end-to-end in the TUI
without needing a real compute workload.

Registered via :func:`agent_plane.client_tools.get_tool_set`,
which ``ap chat`` calls when ``--tools async_demo`` is passed.
"""

from __future__ import annotations

import time
from typing import Any

# Tool schemas in standard OpenAI function-calling format. The
# ``synchronous`` property inside ``parameters.properties`` is
# the ONE thing that lights up the async dispatch path —
# without it, both the server's ``_wants_async_dispatch`` and
# the SDK's ``_is_async_tool_call`` fall through to the legacy
# sync/parking model.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "slow_compute",
            "description": (
                "Simulates a long-running background computation. "
                "ALWAYS call with synchronous=false so the LLM gets "
                "a handle immediately; the real output arrives later "
                "as a [System: task ... completed] user message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": (
                            "How long to sleep before returning, e.g. 3.0. "
                            "Use values between 1 and 30 for demo purposes."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": "A tag to echo in the output.",
                    },
                    "synchronous": {
                        "type": "boolean",
                        "description": (
                            "MUST be set to false. Dispatches this call "
                            "as an async background task and returns a "
                            "{task_id, kind: 'client_tool'} handle. The "
                            "actual result is delivered later as a "
                            "[System: ...] user message."
                        ),
                    },
                },
                "required": ["seconds", "label", "synchronous"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a client-side tool call from ``ap chat``.

    The SDK's ``_run_async_tool_body`` calls this after it has
    spawned us as an ``asyncio.Task`` and waited for the handle
    FCO to set ``task_id``. Our return value is the ``output``
    field of the eventual ``async_tool_results`` PATCH.

    The ``synchronous`` arg is a routing hint consumed by the
    server's dispatch decision — strip it here so the "real"
    args stay clean for anyone reasoning about what the tool
    actually took.

    :param name: Tool name from the LLM's ``function_call``.
        Only ``"slow_compute"`` is registered; anything else
        raises :class:`KeyError` and surfaces as a tool error.
    :param arguments: Arg dict from the LLM, including the
        ``synchronous`` routing hint. ``seconds`` is coerced
        to float; ``label`` is used verbatim.
    :returns: A string describing the completion, e.g.
        ``"finished 'hello' after 3.0s"``.
    :raises KeyError: If ``name`` is not ``"slow_compute"`` —
        the registry only exports one tool.
    """
    if name != "slow_compute":
        raise KeyError(f"async_demo only exports 'slow_compute'; got {name!r}")
    seconds = float(arguments.get("seconds") or 0)
    label = str(arguments.get("label") or "")
    # ``time.sleep`` blocks the current thread. That's fine
    # because the SDK invokes ``execute_tool`` inside an
    # ``asyncio.Task`` — the event loop is free to handle the
    # rest of the stream while this task sleeps. (The @tool
    # decorator in the python-client SDK wraps sync functions
    # in asyncio.to_thread, but this tool-set entry point is
    # called from ``_run_async_tool_body`` directly.)
    time.sleep(seconds)
    return f"finished {label!r} after {seconds}s"
