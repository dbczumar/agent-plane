# /// script
# dependencies = ["ftfy>=6.0"]
# ///
"""Count words, characters, and lines in text.

Uses ``ftfy`` to fix encoding issues (mojibake, broken Unicode)
before counting, so garbled input produces accurate results.
Demonstrates PEP 723 inline dependencies — ``uv`` auto-installs
``ftfy`` on first invocation.
"""

from agent_plane.tools import tool


@tool
def word_count(text: str) -> dict[str, int]:
    """
    Fix text encoding then count words, characters, and lines.

    Args:
        text: The text to analyze.
    """
    import ftfy

    fixed = ftfy.fix_text(text)
    return {
        "word_count": len(fixed.split()),
        "char_count": len(fixed),
        "line_count": fixed.count("\n") + (1 if fixed else 0),
    }
