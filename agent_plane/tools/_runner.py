"""Subprocess entry point for local Python tool execution.

Invoked by ``LocalPythonTool.invoke()`` as a child process.
Reads a JSON request from stdin, dynamically imports the tool module,
calls ``await module.run(arguments)``, and writes a JSON response to
file descriptor 3.

The fd 3 protocol keeps stdout/stderr free for tool debugging
(``print()`` statements in tool code). In Docker mode (where fd 3 is
not available), the ``_AP_RESPONSE_MODE=stdout`` env var switches to
a stdout-based protocol with a ``__AP_RESPONSE__:`` prefix.

Request format (stdin)::

    {"module_path": "/abs/path/to/tool.py", "arguments": {"key": "value"}}

Response format (fd 3 or stdout)::

    {"result": "tool output string"}
    {"error": "TypeError: missing required argument"}
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import traceback
from types import ModuleType

_RESPONSE_FD = 3
_STDOUT_PREFIX = "__AP_RESPONSE__:"


def main() -> None:
    """
    Entry point for the tool runner subprocess.

    Reads a JSON request from stdin, imports the tool module,
    executes ``run(arguments)``, and writes the result to fd 3.
    """
    raw = sys.stdin.buffer.read()
    try:
        request = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        _write_error(f"Invalid request JSON: {exc}")
        return

    module_path: str = request.get("module_path", "")
    arguments: dict = request.get("arguments", {})  # type: ignore[type-arg]

    module = _load_module(module_path)
    if module is None:
        return

    try:
        result = module.run(arguments)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except Exception as exc:
        traceback.print_exc()
        _write_error(f"{type(exc).__name__}: {exc}")
        return

    if not isinstance(result, str):
        result = str(result)
    _write_response({"result": result})


def _load_module(path: str) -> ModuleType | None:
    """
    Import a Python file as a standalone module.

    :param path: Absolute path to the tool Python file.
    :returns: The loaded module, or ``None`` on failure
        (error written to fd 3).
    """
    if not path:
        _write_error("Empty module_path in request")
        return None
    spec = importlib.util.spec_from_file_location("_tool_module", path)
    if spec is None or spec.loader is None:
        _write_error(f"Cannot create module spec from {path}")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        traceback.print_exc()
        _write_error(f"Import error: {type(exc).__name__}: {exc}")
        return None
    if not hasattr(module, "run") or not callable(module.run):
        _write_error(f"Module {path} has no callable run()")
        return None
    return module


def _write_response(data: dict) -> None:  # type: ignore[type-arg]
    """
    Write a JSON response to the output channel.

    :param data: The response dict (must contain ``"result"``
        or ``"error"``).
    """
    encoded = json.dumps(data).encode()
    fd = _get_output_fd()
    if fd == sys.stdout.fileno():
        # Docker mode: prefix so parent can find the response
        # in stdout mixed with tool debug output.
        sys.stdout.buffer.write(
            f"{_STDOUT_PREFIX}".encode() + encoded + b"\n",
        )
        sys.stdout.buffer.flush()
    else:
        os.write(fd, encoded)
        os.close(fd)


def _write_error(message: str) -> None:
    """
    Write an error response to the output channel.

    :param message: Human-readable error description.
    """
    _write_response({"error": message})


def _get_output_fd() -> int:
    """
    Return the file descriptor for writing the response.

    Reads from ``_AP_RESPONSE_FD`` env var (set by the parent
    to the actual fd number passed via ``pass_fds``). Falls back
    to fd 3 if not set. When ``_AP_RESPONSE_MODE=stdout``, returns
    stdout's fd instead (Docker mode).

    :returns: The file descriptor number.
    """
    if os.environ.get("_AP_RESPONSE_MODE") == "stdout":
        return sys.stdout.fileno()
    return int(os.environ.get("_AP_RESPONSE_FD", str(_RESPONSE_FD)))


if __name__ == "__main__":
    main()
