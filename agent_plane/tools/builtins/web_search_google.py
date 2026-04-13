"""Built-in tool: Google Custom Search.

Requires environment variables:
- ``GOOGLE_SEARCH_API_KEY``: API key from Google Cloud Console.
- ``GOOGLE_SEARCH_ENGINE_ID``: Programmable Search Engine ID.

See https://developers.google.com/custom-search/v1/overview
"""

from __future__ import annotations

import json
import logging

# Any: the OpenAI tool schema is a heterogeneous dict with string
# keys and mixed value types (str, dict, list).
from typing import Any

import httpx

from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"

# Maximum results per query (Google CSE limit is 10 per page).
_MAX_RESULTS: int = 10


class WebSearchGoogleTool(Tool):
    """
    Web search via Google Custom Search API.

    API key and engine ID can be provided via the spec config
    block or via ``GOOGLE_SEARCH_API_KEY`` and
    ``GOOGLE_SEARCH_ENGINE_ID`` environment variables (spec
    config takes precedence).

    :param config: Tool-specific config from the agent spec,
        e.g. ``{"api_key": "AIza...", "engine_id": "abc123"}``.
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        """
        Create a new Google Custom Search tool.

        :param config: Optional spec-level config with
            ``api_key`` and ``engine_id`` keys. Falls back to
            environment variables when not provided.
        """
        self._config = config or {}

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"web_search_google"``.
        """
        return "web_search_google"

    @classmethod
    def description(cls) -> str:
        """
        :returns: Human-readable description of the tool.
        """
        return "Search the web using Google and return results with titles, URLs, and snippets."

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema for Google search.

        :returns: An OpenAI function tool schema dict.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search_google",
                "description": (
                    "Search the web using Google and return results "
                    "with titles, URLs, and snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Execute a Google Custom Search query.

        :param arguments: JSON-encoded dict with a ``query`` key.
        :param ctx: Server-side execution context (unused by
            search tools, required by the :class:`Tool` interface).
        :returns: Formatted search results or an error message.
        """
        parsed: dict[str, Any] = json.loads(arguments)
        query = parsed.get("query")
        if not query:
            return "Error: 'query' parameter is required"
        return _search_google(query, self._config)


def _search_google(
    query: str,
    config: dict[str, str],
) -> str:
    """
    Call the Google Custom Search API and format results.

    :param query: The search query string.
    :param config: Spec-level config; checked for ``api_key``
        and ``engine_id`` before falling back to env vars.
    :returns: Formatted results or an error message.
    """
    api_key = config.get("api_key")
    engine_id = config.get("engine_id")
    if not api_key or not engine_id:
        return (
            "Error: api_key and engine_id must be provided in "
            "the web_search config in config.yaml."
        )
    try:
        resp = httpx.get(
            _GOOGLE_CSE_URL,
            params={
                "key": api_key,
                "cx": engine_id,
                "q": query,
                "num": _MAX_RESULTS,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"Google search error: HTTP {exc.response.status_code}"
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return f"Google search error: {exc}"

    return _format_results(resp.json())


def _format_results(data: dict[str, Any]) -> str:
    """
    Format Google CSE JSON response into readable text.

    :param data: The parsed JSON response from Google CSE.
    :returns: Numbered results with title, link, and snippet.
    """
    items = data.get("items", [])
    if not items:
        return "No results found."
    results: list[str] = []
    for i, item in enumerate(items[:_MAX_RESULTS]):
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        results.append(f"{i + 1}. {title}\n   {link}\n   {snippet}")
    return "\n\n".join(results)
