"""Tool-authoring primitives for agent-plane.

Use the :func:`tool` decorator to mark a module-level Python function
as a tool the agent can call. The decorator derives the LLM-facing
JSON schema from the function's signature and Google-style docstring;
the caller just writes Python::

    from agent_plane_client import tool

    @tool
    def get_current_time() -> dict[str, str]:
        \"\"\"Return the current UTC time as ISO-8601.\"\"\"
        return {"now": datetime.now(timezone.utc).isoformat()}

Pass decorated functions as the ``tools=`` argument to
:meth:`AgentPlaneClient.query` or :meth:`Session.query`.

Server-side runtime (``agent_plane.tools.local``) also consumes this
decorator to load ``@tool``-decorated functions bundled inside agent
images, so the same decorator powers both authoring and runtime.
"""

from ._decorator import TOOL_MARKER_ATTR, ToolMetadata, get_tool_metadata, tool
from ._handler import build_tool_handler

__all__ = [
    "TOOL_MARKER_ATTR",
    "ToolMetadata",
    "build_tool_handler",
    "get_tool_metadata",
    "tool",
]
