"""Count words, characters, and lines in text.

A simple example of a local Python tool with no external dependencies.
Demonstrates the ``SCHEMA`` + ``async def run()`` contract.
"""

from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "word_count",
        "description": (
            "Count words, characters, and lines in a block of text. "
            "Useful for analyzing document length or meeting word limits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze.",
                },
            },
            "required": ["text"],
        },
    },
}


async def run(arguments: dict[str, Any]) -> str:
    """
    Count words, characters, and lines.

    :param arguments: Must contain ``"text"`` (str).
    :returns: JSON string with word_count, char_count, line_count.
    """
    import json

    text = arguments.get("text", "")
    words = len(text.split())
    chars = len(text)
    lines = text.count("\n") + (1 if text else 0)
    return json.dumps(
        {
            "word_count": words,
            "char_count": chars,
            "line_count": lines,
        }
    )
