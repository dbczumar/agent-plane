"""Web search via DuckDuckGo.

Returns a list of search results (title, URL, snippet) using
the DuckDuckGo HTML search page — no API key required.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus, unquote

import httpx

# Maximum number of results to return.
_MAX_RESULTS: int = 10

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo and return results "
            "with titles, URLs, and snippets. No API key required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query, e.g. "
                        "'Python asyncio tutorial'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def run(arguments: dict[str, Any]) -> str:
    """
    Search the web and return formatted results.

    :param arguments: Must contain ``query``.
    :returns: Formatted search results, one per line.
    """
    query = arguments.get("query")
    if not query:
        return "Error: 'query' parameter is required"
    return _search_duckduckgo(query)


def _search_duckduckgo(query: str) -> str:
    """
    Fetch search results from DuckDuckGo's HTML page.

    :param query: The search query string.
    :returns: Formatted results or an error message.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "agent-plane/1.0"},
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as exc:
        return f"Search error: {exc}"

    return _parse_results(resp.text)


def _parse_results(html: str) -> str:
    """
    Parse DuckDuckGo HTML search results into formatted text.

    :param html: The raw HTML response body.
    :returns: Formatted results, one per block, or a no-results
        message.
    """
    # DuckDuckGo HTML wraps each result in <a class="result__a">
    # with a snippet in <a class="result__snippet">.
    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    if not links:
        return "No results found."

    results: list[str] = []
    for i, (raw_url, raw_title) in enumerate(links[:_MAX_RESULTS]):
        title = _strip_html(raw_title).strip()
        href = _extract_url(raw_url)
        snippet = _strip_html(snippets[i]).strip() if i < len(snippets) else ""
        results.append(f"{i + 1}. {title}\n   {href}\n   {snippet}")

    return "\n\n".join(results)


def _strip_html(text: str) -> str:
    """
    Remove HTML tags from a string.

    :param text: HTML string.
    :returns: Plain text with tags removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def _extract_url(raw_url: str) -> str:
    """
    Extract the actual URL from a DuckDuckGo redirect link.

    DuckDuckGo wraps result URLs in a redirect like
    ``//duckduckgo.com/l/?uddg=https%3A%2F%2F...&rut=...``.

    :param raw_url: The href from the search result anchor tag.
    :returns: The decoded destination URL.
    """
    # DuckDuckGo redirect format: //duckduckgo.com/l/?uddg=<encoded>&...
    match = re.search(r"uddg=([^&]+)", raw_url)
    if match:
        return unquote(match.group(1))
    return raw_url
