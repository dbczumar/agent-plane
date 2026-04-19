"""
Async wrapper around the subprocess runner for use from DBOS workflows.

The synchronous tool path (``LocalPythonTool.invoke``) calls
``subprocess.Popen.communicate`` directly. For background tools
(``@tool(synchronous=False)``) we need an awaitable so the DBOS
workflow's event loop stays responsive — wrapping the same
synchronous invocation in ``asyncio.to_thread`` is the simplest
correct approach.

Lives in its own module so the background-tool workflow can
import it without pulling in ``LocalPythonTool``'s broader
dependencies (the runner subprocess re-imports the tool's own
file; we don't need ``ToolManager`` here either).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Absolute path to the runner script. Resolved once at import time
# so subprocess invocations don't depend on cwd. Mirrors the
# constant in agent_plane.tools.local.
_RUNNER_PATH = str(Path(__file__).parent / "_runner.py")

# Maximum bytes to read from the fd 3 response pipe (1 MiB).
# Same cap as the synchronous path — keeps the two callers
# producing identical wire behavior.
_MAX_RESPONSE_BYTES = 1024 * 1024


async def invoke_runner_subprocess(
    *,
    module_path: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Run a ``@tool``-decorated function in a subprocess and return its result.

    Spawns the runner process via ``asyncio.to_thread`` so the
    surrounding async context (DBOS workflow event loop) doesn't
    block while ``subprocess.communicate`` waits for the child.

    Uses the fd-3 response protocol — the runner writes its JSON
    response to file descriptor 3, leaving stdout/stderr free for
    tool ``print()`` debugging.

    :param module_path: Absolute path to the tool's Python file,
        e.g. ``"/tmp/cache/ag_abc/tools/python/my_tools.py"``.
    :param tool_name: The decorated function's ``__name__``,
        e.g. ``"train_model"``.
    :param arguments: Deserialized argument dict from the LLM.
    :returns: The tool's result as a JSON-encoded string (the
        runner wraps return values via ``_serialize_result``).
    :raises RuntimeError: If the subprocess exits non-zero, the
        response is missing or malformed, or the runner reports
        an error in its JSON envelope.
    """
    request = json.dumps(
        {
            "module_path": module_path,
            "tool_name": tool_name,
            "arguments": arguments,
        }
    ).encode()

    return await asyncio.to_thread(_invoke_sync, request)


def _invoke_sync(request: bytes) -> str:
    """
    Synchronous fd-3 subprocess call.

    Used internally by :func:`invoke_runner_subprocess` via
    ``asyncio.to_thread``. Kept as a plain function (rather than a
    coroutine) so the thread executor can schedule it without
    any async overhead.

    :param request: The JSON-encoded request body to write to the
        subprocess's stdin.
    :returns: The tool's result string.
    :raises RuntimeError: On any subprocess or response error.
    """
    read_fd, write_fd = os.pipe()
    proc: subprocess.Popen[bytes] | None = None
    try:
        env = {**os.environ, "_AP_RESPONSE_FD": str(write_fd)}
        proc = subprocess.Popen(
            [sys.executable, _RUNNER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(write_fd,),
            env=env,
        )
        os.close(write_fd)
        write_fd = -1

        _stdout, stderr = proc.communicate(input=request)
        return _read_fd3_response(read_fd, proc.returncode, stderr)
    finally:
        if write_fd != -1:
            os.close(write_fd)
        os.close(read_fd)


def _read_fd3_response(
    read_fd: int,
    returncode: int,
    stderr: bytes,
) -> str:
    """
    Read and parse the JSON response written by the runner to fd 3.

    Mirrors ``agent_plane.tools.local._read_fd3_response`` —
    duplicated rather than imported to keep this module's import
    surface minimal (the ``local`` module pulls in
    ``LocalPythonTool`` and its config types, neither needed
    here).

    :param read_fd: The read end of the fd 3 pipe.
    :param returncode: The subprocess exit code.
    :param stderr: Captured stderr bytes for error context.
    :returns: The tool's result string.
    :raises RuntimeError: On non-zero exit, missing response, or
        an explicit ``"error"`` envelope from the runner.
    """
    raw = os.read(read_fd, _MAX_RESPONSE_BYTES)
    if not raw:
        stderr_text = stderr.decode(errors="replace").strip()
        if returncode != 0:
            raise RuntimeError(
                f"tool subprocess exited with code {returncode}: {stderr_text}"
            )
        raise RuntimeError(f"tool produced no response. stderr: {stderr_text}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response from tool: {exc}") from exc

    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return str(data.get("result", ""))
