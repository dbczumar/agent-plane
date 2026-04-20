"""Shared helpers for e2e response-body extraction.

These three helpers were originally duplicated in several terminal
e2e files (``test_terminal_async.py``, ``test_terminal_interactive.py``,
``test_terminal_hierarchy.py``, ``test_archer_terminal.py``,
``test_terminal.py``). Promoted here per the testing skill's
"shared helpers go in helpers.py, not conftest" rule. The older
copies remain inline for now; new files should import from here.
"""

from __future__ import annotations

import json
from typing import Any


def get_output_items(
    body: dict[str, Any],
    item_type: str,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter ``response.output`` by type and optional tool name.

    :param body: Response body from ``GET /v1/responses/{id}``.
    :param item_type: Item type to keep, e.g. ``"function_call"``
        or ``"function_call_output"``.
    :param name: Optional tool-name filter, e.g. ``"terminal_run"``
        for function_call items. When ``None`` every item of the
        matching type is kept.
    :returns: Matching items in original order. Empty list if none
        match.
    """
    items = body.get("output", [])
    filtered = [i for i in items if i.get("type") == item_type]
    if name is not None:
        filtered = [i for i in filtered if i.get("name") == name]
    return filtered


def parse_function_call_outputs(
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse the JSON string inside every ``function_call_output``.

    Each ``function_call_output`` item's ``output`` field is a JSON
    string encoding the tool's structured result. This helper
    decodes them all at once. Bad JSON or non-dict payloads are
    skipped silently — they're irrelevant to most assertions, and
    raising would obscure the real thing the caller is looking
    for.

    :param body: Response body from ``GET /v1/responses/{id}``.
    :returns: Each parsed payload dict, preserving original order.
        Items that failed to decode are dropped.
    """
    results: list[dict[str, Any]] = []
    for item in get_output_items(body, "function_call_output"):
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
    return results


def final_assistant_text(body: dict[str, Any]) -> str:
    """Concatenate every assistant message's ``output_text`` blocks.

    A single response may contain multiple assistant messages
    (one per iteration of the LLM loop); their text concatenated
    with double newlines is usually what the user sees. Used by
    tests that want the "final user-facing text" without worrying
    about which iteration produced it.

    :param body: Response body from ``GET /v1/responses/{id}``.
    :returns: The assistant's text content, ``"\\n\\n"``-joined
        across messages. Empty string if no assistant text is
        present.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n\n".join(parts)
