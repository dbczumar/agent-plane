"""Echo tool for the toolworker sub-agent."""

from __future__ import annotations

from agent_plane_client import tool


@tool
def echo(message: str) -> str:
    """
    Return the input message unchanged.

    Args:
        message: Text to echo.

    Returns:
        The same string, prefixed for clarity.
    """
    return f"echo: {message}"
