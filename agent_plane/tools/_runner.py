"""Subprocess entry point for local Python tool execution.

Invoked by ``LocalPythonTool.invoke()`` as a child process.
Reads a JSON request from stdin, dynamically imports the tool
module, looks up the target ``@tool``-decorated function by name,
calls it with the deserialized arguments (wrapping plain ``def``
in ``asyncio.to_thread`` so it doesn't block the event loop),
serializes the return value, and writes a JSON response to file
descriptor 3.

The fd 3 protocol keeps stdout/stderr free for tool debugging
(``print()`` statements in tool code). In Docker mode (where fd 3
is not available), the ``_AP_RESPONSE_MODE=stdout`` env var
switches to a stdout-based protocol with a ``__AP_RESPONSE__:``
prefix.

Request format (stdin)::

    {
        "module_path": "/abs/path/to/tool.py",
        "tool_name": "word_count",
        "arguments": {"text": "..."}
    }

Response format (fd 3 or stdout)::

    {"result": "tool output as JSON-serialized string"}
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
from typing import Any

_RESPONSE_FD = 3
_STDOUT_PREFIX = "__AP_RESPONSE__:"

# The marker attribute the framework's @tool decorator attaches.
# We import the constant lazily inside main() to avoid importing
# the full agent_plane.tools package in the subprocess (it pulls
# in heavy deps); the constant is a string literal anyway.
_TOOL_MARKER_ATTR = "_ap_tool_metadata"


def main() -> None:
    """
    Entry point for the tool runner subprocess.

    Reads a JSON request from stdin, imports the tool module,
    dispatches to the named ``@tool`` function, serializes the
    return value, and writes the result to fd 3.
    """
    raw = sys.stdin.buffer.read()
    try:
        request = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        _write_error(f"Invalid request JSON: {exc}")
        return

    module_path: str = request.get("module_path", "")
    tool_name: str = request.get("tool_name", "")
    arguments: dict[str, Any] = request.get("arguments", {})

    if not tool_name:
        _write_error("Request missing 'tool_name' field — runner cannot dispatch.")
        return

    module = _load_module(module_path)
    if module is None:
        return

    target = _resolve_tool_function(module, tool_name)
    if target is None:
        return

    try:
        result = _invoke_tool(target, arguments)
    except Exception as exc:
        traceback.print_exc()
        _write_error(f"{type(exc).__name__}: {exc}")
        return

    serialized = _serialize_result(target, result)
    _write_response({"result": serialized})


def _load_module(path: str) -> ModuleType | None:
    """
    Import a Python file as a standalone module.

    :param path: Absolute path to the tool Python file.
    :returns: The loaded module, or ``None`` on failure (error
        already written to fd 3).
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
    return module


def _resolve_tool_function(module: ModuleType, tool_name: str) -> Any:
    """
    Find the ``@tool``-decorated function named ``tool_name`` in ``module``.

    Looks up by attribute name first; if found, verifies the
    object carries the ``@tool`` marker attribute (defense in
    depth — prevents accidentally invoking a non-tool function
    with the same name).

    :param module: The loaded tool module.
    :param tool_name: The function name to dispatch to.
    :returns: The callable if found and properly decorated, else
        ``None`` (error already written to fd 3).
    """
    target = getattr(module, tool_name, None)
    if target is None:
        _write_error(f"Tool function '{tool_name}' not found in module.")
        return None
    if not callable(target):
        _write_error(f"Object '{tool_name}' in module is not callable.")
        return None
    if not hasattr(target, _TOOL_MARKER_ATTR):
        _write_error(f"Function '{tool_name}' is not decorated with @tool.")
        return None
    return target


def _invoke_tool(target: Any, arguments: dict[str, Any]) -> Any:
    """
    Call the tool function with deserialized arguments.

    For ``async def`` bodies, schedules the coroutine on a fresh
    event loop. For plain ``def`` bodies, calls directly (we're
    already in a subprocess; blocking is fine here — the parent
    framework wraps the subprocess invocation in
    ``asyncio.to_thread`` for event-loop friendliness).

    :param target: The ``@tool``-decorated callable.
    :param arguments: Deserialized argument dict (already validated
        on the parent side via Pydantic during decoration).
    :returns: The function's return value.
    """
    result = target(**arguments)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return result


def _serialize_result(target: Any, result: Any) -> str:
    """
    Serialize the tool's return value to a JSON string.

    Tries ``pydantic.TypeAdapter`` keyed on the function's
    declared return annotation (handles BaseModel, dataclasses,
    primitives, datetime, UUID natively). Falls back to
    ``json.dumps(value, default=str)`` for un-annotated returns or
    when TypeAdapter rejects the value (e.g. open file handles).

    :param target: The decorated function — its
        ``ToolMetadata.return_annotation`` drives serialization.
    :param result: The function's return value.
    :returns: A JSON string suitable for the LLM-facing tool result.
    """
    metadata = getattr(target, _TOOL_MARKER_ATTR, None)
    return_annotation = (
        getattr(metadata, "return_annotation", None) if metadata is not None else None
    )

    # If the return is already a string, pass it through unchanged
    # (avoids wrapping JSON-string returns in extra quoting). This
    # mirrors how authors of "stringly-typed" tools expect their
    # output to appear in the LLM context.
    if isinstance(result, str):
        return result

    if return_annotation is not None:
        try:
            from pydantic import TypeAdapter  # local import — heavy module

            adapter = TypeAdapter(return_annotation)
            return adapter.dump_json(result).decode("utf-8")
        except Exception:
            # Fall through to json.dumps fallback.
            pass

    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError) as exc:
        return f"<unserializable return value: {type(result).__name__}: {exc}>"


def _write_response(data: dict[str, Any]) -> None:
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
