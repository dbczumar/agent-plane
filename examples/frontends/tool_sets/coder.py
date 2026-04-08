"""
Client-side tool set for the ``coder`` agent.

Provides 7 coding tools: Read, Write, Edit, Glob, Grep, Bash, LSP.
All tools execute locally on the caller's machine.

Used by ``terminal.py --tools coder`` and ``examples/agents/coder/client.py``.
"""

from __future__ import annotations

import glob as glob_mod
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# Maximum characters returned from any tool execution.
# Prevents TUI freezes when tools produce huge output
# (e.g. globbing a large repo, running verbose commands).
_MAX_OUTPUT_CHARS = 20_000

# Maximum number of file paths returned by Glob.
_MAX_GLOB_RESULTS = 200

# ── Tool schemas (OpenAI function format) ────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": (
                "Read the contents of a file. Returns the file text "
                "with line numbers. Supports text files, images, and PDFs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the file to read, e.g. '/home/user/project/main.py'."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Line number to start reading from (1-based). "
                            "Only needed for large files."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of lines to read. Only needed for large files."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": (
                "Create a new file or overwrite an existing file. "
                "Prefer Edit for modifying existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": (
                "Make targeted string replacements in an existing file. "
                "The old_string must appear exactly once in the file "
                "unless replace_all is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "If true, replace all occurrences of old_string. Defaults to false."
                        ),
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": (
                "Find files matching a glob pattern. "
                "Returns matching file paths sorted by modification time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": ("Glob pattern to match, e.g. '**/*.py' or 'src/**/*.ts'."),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory to search in. Defaults to the current working directory."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": (
                "Search file contents using regex. Built on ripgrep. "
                "Returns matching file paths or line content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Regex pattern to search for, e.g. 'def main' or 'import\\s+asyncio'."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "File or directory to search in. "
                            "Defaults to the current working directory."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": (
                            "Glob pattern to filter files, e.g. '*.py' or '*.{ts,tsx}'."
                        ),
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": (
                            "Output mode: 'content' shows matching lines, "
                            "'files_with_matches' shows file paths (default), "
                            "'count' shows match counts."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": (
                "Execute a shell command and return its output. "
                "Use for running tests, git operations, builds, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The shell command to execute, "
                            "e.g. 'pytest tests/ -x' or 'git status'."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Timeout in milliseconds. Defaults to 120000 (2 minutes)."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "LSP",
            "description": (
                "Code intelligence via language servers. "
                "Jump to definitions, find references, get type info, "
                "list symbols, find implementations, and trace call "
                "hierarchies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "definition",
                            "references",
                            "hover",
                            "symbols",
                            "implementations",
                            "diagnostics",
                        ],
                        "description": (
                            "The LSP operation to perform: "
                            "'definition' jumps to a symbol's definition, "
                            "'references' finds all usages, "
                            "'hover' gets type info, "
                            "'symbols' lists symbols in a file, "
                            "'implementations' finds interface implementations, "
                            "'diagnostics' returns type errors and warnings."
                        ),
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "line": {
                        "type": "integer",
                        "description": (
                            "1-based line number of the symbol. "
                            "Required for definition, references, hover, "
                            "and implementations."
                        ),
                    },
                    "character": {
                        "type": "integer",
                        "description": (
                            "0-based character offset within the line. "
                            "Required for definition, references, hover, "
                            "and implementations."
                        ),
                    },
                },
                "required": ["action", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time. Returns an "
                "ISO-formatted timestamp."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ── Local tool execution ─────────────────────────────────


def _truncate(output: str) -> str:
    """
    Truncate tool output to ``_MAX_OUTPUT_CHARS``.

    Appends a notice when truncation occurs so the LLM knows
    the output was cut short.

    :param output: Raw tool output string.
    :returns: The output, possibly truncated with a notice.
    """
    if len(output) <= _MAX_OUTPUT_CHARS:
        return output
    return (
        output[:_MAX_OUTPUT_CHARS] + f"\n\n... (truncated — {len(output)} chars total, "
        f"showing first {_MAX_OUTPUT_CHARS})"
    )


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a client-side tool locally and return the result string.

    Output is truncated to ``_MAX_OUTPUT_CHARS`` to prevent TUI
    freezes and LLM context overflow on large results.

    :param name: Tool function name, e.g. ``"Read"`` or ``"Bash"``.
    :param arguments: Parsed arguments dict from the LLM's function call.
    :returns: The tool's output as a string, truncated if needed.
    """
    executors = {
        "Read": _execute_read,
        "Write": _execute_write,
        "Edit": _execute_edit,
        "Glob": _execute_glob,
        "Grep": _execute_grep,
        "Bash": _execute_bash,
        "LSP": _execute_lsp,
        "get_current_time": _execute_get_current_time,
    }
    executor = executors.get(name)
    if executor is None:
        return f"Unknown tool: {name}"
    return _truncate(executor(arguments))


def _execute_read(args: dict[str, Any]) -> str:
    """
    Read a file's contents, optionally with offset and limit.

    :param args: Must contain ``file_path``; may contain ``offset``
        (1-based line number) and ``limit`` (max lines).
    :returns: Numbered lines of the file, or an error message.
    """
    file_path = args["file_path"]
    try:
        text = Path(file_path).read_text()
    except (OSError, UnicodeDecodeError) as exc:
        return f"Error reading {file_path}: {exc}"
    lines = text.splitlines()
    offset = args.get("offset", 1)
    limit = args.get("limit", len(lines))
    # Convert to 0-based index for slicing.
    start = max(0, offset - 1)
    selected = lines[start : start + limit]
    numbered = [f"{start + i + 1}\t{line}" for i, line in enumerate(selected)]
    return "\n".join(numbered)


def _execute_write(args: dict[str, Any]) -> str:
    """
    Write content to a file, creating parent directories if needed.

    :param args: Must contain ``file_path`` and ``content``.
    :returns: Confirmation message or error.
    """
    file_path = Path(args["file_path"])
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(args["content"])
    except OSError as exc:
        return f"Error writing {file_path}: {exc}"
    return f"Successfully wrote {file_path}"


def _execute_edit(args: dict[str, Any]) -> str:
    """
    Replace a string in a file.

    :param args: Must contain ``file_path``, ``old_string``,
        ``new_string``; may contain ``replace_all``.
    :returns: Confirmation with replacement count, or error.
    """
    file_path = Path(args["file_path"])
    try:
        text = file_path.read_text()
    except OSError as exc:
        return f"Error reading {file_path}: {exc}"
    old = args["old_string"]
    new = args["new_string"]
    replace_all = args.get("replace_all", False)
    count = text.count(old)
    if count == 0:
        return f"Error: old_string not found in {file_path}"
    if not replace_all and count > 1:
        return (
            f"Error: old_string appears {count} times (expected 1). "
            f"Use replace_all=true or provide more context."
        )
    result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    file_path.write_text(result)
    replacements = count if replace_all else 1
    return f"Replaced {replacements} occurrence(s) in {file_path}"


def _execute_glob(args: dict[str, Any]) -> str:
    """
    Find files matching a glob pattern.

    Results are capped at ``_MAX_GLOB_RESULTS`` to avoid
    freezing on large directories.

    :param args: Must contain ``pattern``; may contain ``path``.
    :returns: Newline-separated matching file paths.
    """
    pattern = args["pattern"]
    base = args.get("path", ".")
    matches = sorted(
        glob_mod.glob(os.path.join(base, pattern), recursive=True),
    )
    if not matches:
        return "No files matched."
    total = len(matches)
    if total > _MAX_GLOB_RESULTS:
        truncated = matches[:_MAX_GLOB_RESULTS]
        return (
            "\n".join(truncated) + f"\n\n... ({total} total matches, "
            f"showing first {_MAX_GLOB_RESULTS})"
        )
    return "\n".join(matches)


def _execute_grep(args: dict[str, Any]) -> str:
    """
    Search file contents using ripgrep (falls back to ``grep -r``).

    :param args: Must contain ``pattern``; may contain ``path``,
        ``glob``, ``output_mode``.
    :returns: Matching lines or file paths.
    """
    pattern = args["pattern"]
    path = args.get("path", ".")
    cmd = ["rg", pattern, path, "--no-heading"]
    file_glob = args.get("glob")
    if file_glob:
        cmd.extend(["--glob", file_glob])
    mode = args.get("output_mode", "files_with_matches")
    if mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif mode == "count":
        cmd.append("--count")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or "No matches found."
    except FileNotFoundError:
        # ripgrep not installed — fall back to grep.
        grep_cmd = ["grep", "-r", pattern, path]
        result = subprocess.run(
            grep_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or "No matches found."


def _execute_bash(args: dict[str, Any]) -> str:
    """
    Execute a shell command.

    :param args: Must contain ``command``; may contain ``timeout``
        (milliseconds, default 120000).
    :returns: Combined stdout and stderr, or a timeout message.
    """
    command = args["command"]
    timeout_ms = args.get("timeout", 120_000)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000,
        )
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_ms}ms"


def _execute_lsp(args: dict[str, Any]) -> str:
    """
    Stub for LSP tool — requires a running language server.

    :param args: LSP action arguments (action, file_path, line,
        character).
    :returns: A message indicating LSP is not yet implemented.
    """
    return (
        f"LSP not implemented in this client. "
        f"Action: {args.get('action')}, file: {args.get('file_path')}"
    )


def _execute_get_current_time(args: dict[str, Any]) -> str:
    """
    Return the current date and time as an ISO-formatted string.

    :param args: Unused — no arguments required.
    :returns: ISO timestamp, e.g. ``"2026-04-08T14:30:00"``.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S")
