"""
Minimal echo tool for the ``e2e-tool-gate`` fixture.

A deterministic tool the LLM can call to exercise the
TOOL_CALL enforcement site end-to-end. The tool simply
reflects its argument so the test can assert on the tool
output surfaced to the LLM.
"""

from __future__ import annotations

from agent_plane_client import tool


@tool
def echo(message: str) -> str:
    """
    Return the input message unchanged.

    Args:
        message: The text to echo back.

    Returns:
        The same string, prefixed so downstream asserts can
        distinguish tool output from raw LLM text.
    """
    return f"echo: {message}"
