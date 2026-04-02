"""
Example client for the ``coder`` agent with client-side tools.

Demonstrates:
1. Uploading the agent bundle.
2. Creating a response with client-side tool schemas.
3. Polling for completion and executing tool calls locally.
4. Continuing the conversation with tool results via
   ``previous_response_id``.

Usage::

    # Start the agent-plane server, then:
    export AGENT_PLANE_URL=http://localhost:8000
    python client.py "Find all Python files that import asyncio"

Requirements: ``httpx``, ``pyyaml``
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

# Maximum characters returned from any tool execution.
# Prevents context overflow when tools produce huge output
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

    Output is truncated to ``_MAX_OUTPUT_CHARS`` to prevent
    context overflow on large results.

    :param name: Tool function name, e.g. ``"Read"`` or ``"Bash"``.
    :param arguments: Parsed arguments dict from the LLM's function call.
    :returns: The tool's output as a string, truncated if needed.
    """
    if name == "Read":
        return _truncate(_execute_read(arguments))
    if name == "Write":
        return _truncate(_execute_write(arguments))
    if name == "Edit":
        return _truncate(_execute_edit(arguments))
    if name == "Glob":
        return _truncate(_execute_glob(arguments))
    if name == "Grep":
        return _truncate(_execute_grep(arguments))
    if name == "Bash":
        return _truncate(_execute_bash(arguments))
    if name == "LSP":
        return f"LSP not implemented in this example client. Args: {arguments}"
    return f"Unknown tool: {name}"


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
            "Use replace_all=true or provide more context."
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
    matches = sorted(glob_mod.glob(os.path.join(base, pattern), recursive=True))
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


# ── Agent-plane API interaction ──────────────────────────

# Any: API response bodies are heterogeneous dicts.
ResponseBody = dict[str, Any]

BASE_URL = os.environ.get("AGENT_PLANE_URL", "http://localhost:8000")


def upload_agent(client: httpx.Client) -> str:
    """
    Upload the coder agent bundle and return the agent name.

    Builds the bundle from the config.yaml and AGENTS.md in this
    directory, uploads it, and returns the agent name used for
    subsequent API calls.

    :param client: HTTP client pointed at the agent-plane server.
    :returns: The agent name string, e.g. ``"coder"``.
    """
    import io
    import tarfile

    bundle_dir = Path(__file__).parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(bundle_dir / "config.yaml", arcname="config.yaml")
        tf.add(bundle_dir / "AGENTS.md", arcname="AGENTS.md")
        # Include sub-agent directories if present.
        agents_dir = bundle_dir / "agents"
        if agents_dir.is_dir():
            for child in sorted(agents_dir.iterdir()):
                if (child / "config.yaml").exists():
                    tf.add(
                        child / "config.yaml",
                        arcname=f"agents/{child.name}/config.yaml",
                    )
    buf.seek(0)
    resp = client.post(
        f"{BASE_URL}/api/agents",
        files={"bundle": ("agent.tar.gz", buf, "application/gzip")},
    )
    if resp.status_code == 409:
        # Agent already exists — reuse it.
        print("Agent already exists, reusing.")
        return "coder"
    resp.raise_for_status()
    print(f"Agent uploaded: {resp.json()['name']}")
    return resp.json()["name"]


def create_response(
    client: httpx.Client,
    model: str,
    input_text: str,
    previous_response_id: str | None = None,
) -> ResponseBody:
    """
    Create a response and poll until completion.

    Sends the request with ``background=True``, then polls
    ``GET /v1/responses/{id}`` until the task reaches a terminal
    state (``completed`` or ``failed``).

    :param client: HTTP client pointed at the agent-plane server.
    :param model: Agent name to use, e.g. ``"coder"``.
    :param input_text: User message text or stringified tool results.
    :param previous_response_id: ID of the previous response for
        multi-turn continuation.
    :returns: The completed response body dict.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "background": True,
        "tools": TOOLS,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    resp = client.post(f"{BASE_URL}/v1/responses", json=payload)
    resp.raise_for_status()
    response_id = resp.json()["id"]
    print(f"Response created: {response_id}")

    # Poll until terminal state. While polling, check for
    # tunneled client tool calls from sub-agents and execute
    # them mid-flight.
    while True:
        poll = client.get(f"{BASE_URL}/v1/responses/{response_id}")
        poll.raise_for_status()
        body = poll.json()
        status = body.get("status")
        if status in ("completed", "failed"):
            return body
        # Check for tunneled tool calls needing client execution.
        _handle_tunneled_tool_calls(client, response_id, body)
        time.sleep(0.5)


def _handle_tunneled_tool_calls(
    client: httpx.Client,
    response_id: str,
    body: dict[str, Any],
) -> None:
    """
    Execute tunneled client-side tool calls from sub-agents
    and PATCH results back to the server.

    Scans the response output for ``function_call`` items with
    ``status: "action_required"``, executes each locally, and
    submits results via ``PATCH /v1/responses/{id}``.

    :param client: HTTP client.
    :param response_id: The root response ID for PATCH.
    :param body: The polled response body dict.
    """
    output = body.get("output", [])
    action_required = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("status") == "action_required"
    ]
    if not action_required:
        return
    tool_results = []
    for fc in action_required:
        name = fc["name"]
        call_id = fc["call_id"]
        arguments = json.loads(fc.get("arguments", "{}"))
        print(f"\n> Tunneled tool: {name}({json.dumps(arguments, indent=2)})")
        result = execute_tool(name, arguments)
        display = result[:500] + "..." if len(result) > 500 else result
        print(f"  Result: {display}")
        tool_results.append({"call_id": call_id, "output": result})
    # PATCH results back to the server.
    patch_resp = client.patch(
        f"{BASE_URL}/v1/responses/{response_id}",
        json={"tool_results": tool_results},
    )
    if patch_resp.status_code != 200:
        print(f"  PATCH failed: {patch_resp.text[:200]}")


def _print_text_output(output: list[dict[str, Any]]) -> None:
    """
    Print any assistant text content from a response's output items.

    :param output: List of output item dicts from the API response.
    """
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    print(f"\nAssistant: {content['text']}")


def _execute_tool_calls(
    function_calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Execute tool calls locally and return ``function_call_output`` items.

    :param function_calls: List of ``function_call`` item dicts,
        each with ``name``, ``call_id``, ``arguments``.
    :returns: List of ``function_call_output`` dicts to send back.
    """
    results: list[dict[str, str]] = []
    for fc in function_calls:
        name = fc["name"]
        call_id = fc["call_id"]
        arguments = json.loads(fc["arguments"])
        print(f"\n> Executing tool: {name}({json.dumps(arguments, indent=2)})")
        result = execute_tool(name, arguments)
        # Truncate long results for display.
        display = result[:500] + "..." if len(result) > 500 else result
        print(f"  Result: {display}")
        results.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            }
        )
    return results


def run_conversation(model: str, user_input: str) -> None:
    """
    Run a multi-turn conversation, executing tool calls locally.

    Loops until the agent produces a final text response with no
    pending tool calls. Each iteration:

    1. Sends the user input (or tool results) to the agent.
    2. Checks the response output for ``function_call`` items.
    3. Executes each tool call locally via :func:`execute_tool`.
    4. Sends results back as ``function_call_output`` items.

    :param model: Agent name, e.g. ``"coder"``.
    :param user_input: The initial user message.
    """
    with httpx.Client(timeout=300) as client:
        upload_agent(client)
        previous_id: str | None = None
        current_input: str | list[dict[str, Any]] = user_input

        while True:
            resp = create_response(
                client,
                model=model,
                input_text=current_input,
                previous_response_id=previous_id,
            )
            previous_id = resp["id"]
            if resp.get("status") == "failed":
                print(f"Agent failed: {resp.get('error', 'unknown error')}")
                return

            output = resp.get("output", [])
            # Server-side tools already have function_call_output
            # in the output — only execute client-side calls that
            # the server left for us to handle.
            completed_ids = {
                item["call_id"] for item in output if item.get("type") == "function_call_output"
            }
            function_calls = [
                item
                for item in output
                if item.get("type") == "function_call" and item.get("call_id") not in completed_ids
            ]
            _print_text_output(output)

            if not function_calls:
                break

            current_input = _execute_tool_calls(function_calls)


def main() -> None:
    """
    Parse CLI arguments and start the conversation loop.
    """
    parser = argparse.ArgumentParser(
        description="Coder agent — coding assistant with client-side tools",
    )
    parser.add_argument(
        "prompt",
        help="The coding task or question to send to the agent.",
    )
    args = parser.parse_args()
    run_conversation(model="coder", user_input=args.prompt)


if __name__ == "__main__":
    main()
