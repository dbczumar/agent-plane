"""Fetch the text content of a URL.

Returns the raw text body (HTML, JSON, etc.) truncated to a
reasonable size for LLM consumption.
"""

from __future__ import annotations

from typing import Any

import httpx

# Maximum characters returned to avoid overwhelming the LLM context.
_MAX_RESPONSE_CHARS: int = 40_000

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch the content of a URL and return the response body as text. "
            "Useful for reading documentation, API responses, raw files, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The URL to fetch, e.g. "
                        "'https://docs.python.org/3/library/json.html'."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}


def run(arguments: dict[str, Any]) -> str:
    """
    Fetch a URL and return the response text.

    :param arguments: Must contain ``url``.
    :returns: The response body text, truncated to
        ``_MAX_RESPONSE_CHARS``.
    """
    url = arguments.get("url")
    if not url:
        return "Error: 'url' parameter is required"
    try:
        resp = httpx.get(
            url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "agent-plane/1.0"},
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return f"Connection error: {exc}"

    text = resp.text
    if len(text) > _MAX_RESPONSE_CHARS:
        return (
            text[:_MAX_RESPONSE_CHARS]
            + f"\n\n... (truncated — {len(text)} chars total, "
            f"showing first {_MAX_RESPONSE_CHARS})"
        )
    return text
