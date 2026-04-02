"""Built-in tool: Perplexity web search.

Requires environment variable:
- ``PERPLEXITY_API_KEY``: API key from Perplexity.

Uses the Perplexity Chat Completions API with an online model
to perform grounded web search. Returns the answer with citations.

See https://docs.perplexity.ai/
"""

from __future__ import annotations

import json
import logging
import os

# Any: the OpenAI tool schema is a heterogeneous dict with string
# keys and mixed value types (str, dict, list).
from typing import Any

import httpx

from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

_PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Perplexity model optimized for web search with citations.
_PERPLEXITY_MODEL = "sonar"


class WebSearchPerplexityTool(Tool):
    """
    Web search via Perplexity's online models.

    Sends the query to Perplexity's Chat Completions API using
    an online model that performs grounded web search. Returns
    the answer text with source citations.

    API key can be provided via the spec config block or via
    the ``PERPLEXITY_API_KEY`` environment variable (spec config
    takes precedence).

    :param config: Tool-specific config from the agent spec,
        e.g. ``{"api_key": "pplx-..."}``.
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        """
        Create a new Perplexity web search tool.

        :param config: Optional spec-level config with an
            ``api_key`` key. Falls back to environment variable
            when not provided.
        """
        self._config = config or {}

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"web_search_perplexity"``.
        """
        return "web_search_perplexity"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema for Perplexity search.

        :returns: An OpenAI function tool schema dict.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search_perplexity",
                "description": (
                    "Search the web using Perplexity AI. Returns "
                    "a grounded answer with source citations."
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
        Execute a Perplexity web search query.

        :param arguments: JSON-encoded dict with a ``query`` key.
        :param ctx: Server-side execution context (unused by
            search tools, required by the :class:`Tool` interface).
        :returns: Perplexity's answer with citations, or an error.
        """
        parsed: dict[str, Any] = json.loads(arguments)
        query = parsed.get("query")
        if not query:
            return "Error: 'query' parameter is required"
        return _search_perplexity(query, self._config)


def _search_perplexity(
    query: str,
    config: dict[str, str],
) -> str:
    """
    Call the Perplexity Chat Completions API with an online model.

    :param query: The search query string.
    :param config: Spec-level config; checked for ``api_key``
        before falling back to the env var.
    :returns: The answer text with citations, or an error message.
    """
    api_key = config.get("api_key") or os.environ.get(
        "PERPLEXITY_API_KEY",
    )
    if not api_key:
        return (
            "Error: api_key must be provided in the spec "
            "config or via PERPLEXITY_API_KEY environment "
            "variable."
        )
    try:
        resp = httpx.post(
            _PERPLEXITY_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _PERPLEXITY_MODEL,
                "messages": [
                    {"role": "user", "content": query},
                ],
            },
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"Perplexity search error: HTTP {exc.response.status_code}"
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return f"Perplexity search error: {exc}"

    return _format_response(resp.json())


def _format_response(data: dict[str, Any]) -> str:
    """
    Extract the answer and citations from Perplexity's response.

    :param data: The parsed JSON response from Perplexity.
    :returns: The answer text followed by numbered citations.
    """
    choices = data.get("choices", [])
    if not choices:
        return "No answer returned."
    message = choices[0].get("message", {})
    content: str = str(message.get("content", ""))
    # Perplexity includes citation URLs in the response metadata.
    citations = data.get("citations", [])
    if citations:
        citation_lines = [f"[{i + 1}] {url}" for i, url in enumerate(citations)]
        content += "\n\nSources:\n" + "\n".join(citation_lines)
    return content
