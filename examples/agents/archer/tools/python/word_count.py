# /// script
# dependencies = ["ftfy>=6.0"]
# ///
"""Count words, characters, and lines in text.

Uses ``ftfy`` to fix encoding issues (mojibake, broken Unicode)
before counting, so garbled input produces accurate results.
Demonstrates PEP 723 inline dependencies — ``uv`` auto-installs
``ftfy`` on first invocation.
"""

from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "word_count",
        "description": (
            "Count words, characters, and lines in a block of text. "
            "Fixes encoding issues (mojibake) before counting. "
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
    Fix text encoding then count words, characters, and lines.

    :param arguments: Must contain ``"text"`` (str).
    :returns: JSON string with word_count, char_count, line_count.
    """
    import json

    import ftfy

    text = ftfy.fix_text(arguments.get("text", ""))
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
