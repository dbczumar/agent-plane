"""Lightweight filesystem MCP server for the archer agent.

Exposes read-only filesystem tools (list directory, read file,
search for text) via HTTP (SSE) transport. Uses FastMCP for
minimal boilerplate.

Usage (start the server, then point the agent config at the URL):
    python filesystem_server.py
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# port=8100 matches the url in filesystem.yaml
mcp = FastMCP("filesystem", port=8100)

# Root directory the server is allowed to access. Defaults to
# the current working directory when launched by agent-plane.
_ALLOWED_ROOT = Path(os.getcwd()).resolve()


def _safe_resolve(path: str) -> Path:
    """
    Resolve a user-provided path and verify it's within the
    allowed root directory.

    :param path: A relative or absolute path string, e.g.
        ``"src/main.py"`` or ``"/etc/passwd"``.
    :returns: The resolved absolute ``Path``.
    :raises ValueError: If the resolved path escapes the
        allowed root directory.
    """
    resolved = (_ALLOWED_ROOT / path).resolve()
    if not resolved.is_relative_to(_ALLOWED_ROOT):
        raise ValueError(f"Path {path!r} is outside the allowed directory: {_ALLOWED_ROOT}")
    return resolved


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """
    List files and directories at the given path.

    Returns one entry per line, with directories suffixed by '/'.

    :param path: Relative path from the working directory,
        e.g. ``"src"`` or ``"."`` for the root.
    """
    resolved = _safe_resolve(path)
    if not resolved.is_dir():
        raise ValueError(f"Not a directory: {path}")
    entries: list[str] = []
    for entry in sorted(resolved.iterdir()):
        name = entry.name
        if entry.is_dir():
            name += "/"
        entries.append(name)
    return "\n".join(entries) if entries else "(empty directory)"


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the contents of a text file.

    :param path: Relative path to the file, e.g.
        ``"config.yaml"`` or ``"src/main.py"``.
    """
    resolved = _safe_resolve(path)
    if not resolved.is_file():
        raise ValueError(f"Not a file: {path}")
    return resolved.read_text()


@mcp.tool()
def search_files(pattern: str, path: str = ".") -> str:
    """
    Search for files matching a glob pattern.

    Returns matching file paths relative to the search root,
    one per line.

    :param pattern: Glob pattern, e.g. ``"**/*.py"`` or
        ``"*.yaml"``.
    :param path: Directory to search in, relative to the
        working directory.
    """
    resolved = _safe_resolve(path)
    if not resolved.is_dir():
        raise ValueError(f"Not a directory: {path}")
    matches = sorted(resolved.glob(pattern))
    # Filter to only files within the allowed root.
    safe_matches = [m for m in matches if m.is_file() and m.is_relative_to(_ALLOWED_ROOT)]
    if not safe_matches:
        return "(no matches)"
    return "\n".join(str(m.relative_to(_ALLOWED_ROOT)) for m in safe_matches)


if __name__ == "__main__":
    # Serve over HTTP (SSE) on port 8100 — matches filesystem.yaml url.
    mcp.run(transport="sse")
