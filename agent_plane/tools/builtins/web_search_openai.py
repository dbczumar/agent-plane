"""Built-in tool: OpenAI native web search (passthrough).

This tool adds ``{"type": "web_search_preview"}`` to the tools list
sent to OpenAI's Responses API. The LLM handles search execution
server-side — agent-plane never invokes this tool locally.
"""

from __future__ import annotations

# Any: the OpenAI tool schema is a heterogeneous dict with string
# keys and mixed value types (str, dict, list).
from typing import Any

from agent_plane.tools.base import Tool, ToolContext


class WebSearchOpenAITool(Tool):
    """
    Passthrough to OpenAI's built-in ``web_search_preview`` tool.

    Unlike function tools, this is a provider-native tool type.
    The schema is ``{"type": "web_search_preview"}`` — not a
    function definition. OpenAI handles execution server-side;
    ``invoke()`` should never be called.

    :param config: Unused — OpenAI web search needs no config
        (uses the LLM API key).
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        """
        Create a new OpenAI web search passthrough tool.

        :param config: Unused — accepted for interface
            consistency with other built-in tools.
        """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"web_search_openai"``.
        """
        return "web_search_openai"

    @classmethod
    def description(cls) -> str:
        """
        :returns: Human-readable description of the tool.
        """
        return (
            "Search the web using OpenAI's built-in "
            "web_search_preview tool. The LLM handles "
            "search execution server-side."
        )

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-native web search tool schema.

        This is NOT a function schema — it uses OpenAI's built-in
        ``web_search_preview`` type. The Responses API handles
        search execution server-side.

        :returns: ``{"type": "web_search_preview"}``.
        """
        return {"type": "web_search_preview"}

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Not callable — OpenAI handles execution server-side.

        :param arguments: Unused.
        :param ctx: Unused.
        :raises RuntimeError: Always. This tool is a passthrough.
        """
        raise RuntimeError(
            "web_search_openai is a passthrough tool — "
            "OpenAI handles execution server-side. "
            "invoke() should never be called."
        )
