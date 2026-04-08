"""Local Python tool execution via subprocess.

Loads Python tool files from the agent's ``tools/python/`` directory
and exposes them as :class:`Tool` instances. Each Python file must
export:

- ``SCHEMA``: An OpenAI function-format dict with ``"type": "function"``
  and a ``"function"`` sub-dict containing ``name``, ``description``,
  and ``parameters``.
- ``async def run(arguments: dict) -> str``: The async callable that
  executes the tool.

Tool code runs in a **subprocess** (not in-process) for crash
isolation. Communication uses the fd 3 pipe protocol — see
``_runner.py`` for the child side.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

# Any: OpenAI function schemas contain heterogeneous values
# (strings, ints, nested objects, arrays) — no specific type fits.
from typing import Any

from agent_plane.spec.types import LocalToolInfo
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

# Absolute path to the runner script. Resolved once at import time
# so subprocess invocations don't depend on cwd.
_RUNNER_PATH = str(Path(__file__).parent / "_runner.py")

# Maximum bytes to read from the fd 3 response pipe (1 MiB).
_MAX_RESPONSE_BYTES = 1024 * 1024


class LocalPythonTool(Tool):
    """
    A tool backed by a local Python file, executed in a subprocess.

    The Python file must export ``SCHEMA`` (OpenAI function schema
    dict) and ``async def run(arguments: dict) -> str``.

    :param info: The discovered :class:`LocalToolInfo` from the
        agent spec parser.
    :param schema: The validated SCHEMA dict from the module.
    :param module_path: Absolute path to the tool Python file.
    """

    def __init__(
        self,
        info: LocalToolInfo,
        schema: dict[str, Any],
        module_path: Path,
    ) -> None:
        """
        Initialize from a validated tool file.

        :param info: The :class:`LocalToolInfo` with name and path.
        :param schema: The validated SCHEMA dict.
        :param module_path: Absolute path to the tool file, e.g.
            ``Path("/tmp/cache/ag_abc/tools/python/my_tool.py")``.
        """
        self._info = info
        self._schema = schema
        self._name: str = info.name
        self._module_path = module_path
        self._proc: subprocess.Popen[bytes] | None = None

    def name(self) -> str:  # type: ignore[override]
        """
        Tool name derived from the filename.

        :returns: The tool name, e.g. ``"arxiv_search"``.
        """
        return self._name

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema from the module's ``SCHEMA``.

        :returns: The schema dict, e.g.
            ``{"type": "function", "function": {...}}``.
        """
        return self._schema

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Execute the tool in a subprocess via the fd 3 protocol.

        Spawns ``python _runner.py``, sends the request on stdin,
        and reads the JSON response from fd 3. stdout/stderr are
        captured for debugging but not returned to the LLM.

        :param arguments: JSON-encoded arguments string from the LLM.
        :param ctx: Server-side execution context (unused by
            local tools, required by the :class:`Tool` interface).
        :returns: The tool's string result, or an error string
            if the subprocess fails.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        request = json.dumps(
            {
                "module_path": str(self._module_path),
                "arguments": parsed,
            }
        ).encode()

        read_fd, write_fd = os.pipe()
        try:
            env = {**os.environ, "_AP_RESPONSE_FD": str(write_fd)}
            self._proc = subprocess.Popen(
                [sys.executable, _RUNNER_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
                env=env,
            )
            # Parent closes write end — only the child writes.
            os.close(write_fd)
            write_fd = -1

            stdout, stderr = self._proc.communicate(input=request)
            return _read_response(read_fd, self._proc.returncode, stderr)
        finally:
            self._proc = None
            if write_fd != -1:
                os.close(write_fd)
            os.close(read_fd)

    def cancel(self) -> None:
        """
        Kill the subprocess on timeout.

        Called by ``call_tool_with_timeout`` when the deadline
        expires. Sends SIGKILL — the subprocess dies immediately.
        """
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def _read_response(
    read_fd: int,
    returncode: int,
    stderr: bytes,
) -> str:
    """
    Read and parse the JSON response from the fd 3 pipe.

    :param read_fd: The read end of the fd 3 pipe.
    :param returncode: The subprocess exit code.
    :param stderr: Captured stderr bytes for error reporting.
    :returns: The tool's result string, or an error string.
    """
    raw = os.read(read_fd, _MAX_RESPONSE_BYTES)
    if not raw:
        # No response on fd 3 — subprocess crashed or forgot to write.
        stderr_text = stderr.decode(errors="replace").strip()
        if returncode != 0:
            return f"Error: tool subprocess exited with code {returncode}: {stderr_text}"
        return f"Error: tool produced no response. stderr: {stderr_text}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON response from tool: {exc}"

    if "error" in data:
        return f"Error: {data['error']}"
    result: str = data.get("result", "")
    return result


# ── Module loading and validation (for SCHEMA extraction) ───


def load_local_python_tools(
    local_tools: list[LocalToolInfo],
    workdir: Path,
) -> list[LocalPythonTool]:
    """
    Load and validate local Python tools from the agent image.

    Each tool file is imported to extract and validate its ``SCHEMA``.
    The module is NOT stored — tool execution happens in a subprocess
    via ``_runner.py``, not in-process.

    :param local_tools: Discovered :class:`LocalToolInfo` entries
        from the agent spec.
    :param workdir: The agent image's extracted directory on disk,
        e.g. ``Path("/tmp/agent-cache/ag_abc123")``.
    :returns: List of successfully validated :class:`LocalPythonTool`
        instances.
    """
    tools: list[LocalPythonTool] = []
    for info in local_tools:
        if info.language != "python":
            continue
        tool_path = workdir / info.path
        if not tool_path.is_file():
            _logger.warning(
                "Local tool %r: file not found at %s — skipping",
                info.name,
                tool_path,
            )
            continue
        module = _load_module(info.name, tool_path)
        if module is None:
            continue
        if not _validate_module(info.name, module):
            continue
        tools.append(
            LocalPythonTool(
                info=info,
                schema=module.SCHEMA,
                module_path=tool_path.resolve(),
            )
        )
    return tools


def _load_module(tool_name: str, path: Path) -> ModuleType | None:
    """
    Import a Python file as a standalone module.

    Used only for SCHEMA extraction and validation at load time.
    The module is not stored — execution happens in a subprocess.

    :param tool_name: The tool name for error messages.
    :param path: Absolute path to the Python file.
    :returns: The loaded module, or ``None`` on import error.
    """
    module_name = f"_agent_tool_{tool_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _logger.warning(
            "Local tool %r: failed to create module spec from %s — skipping",
            tool_name,
            path,
        )
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _logger.exception(
            "Local tool %r: import error from %s — skipping",
            tool_name,
            path,
        )
        return None
    return module


def _validate_schema(tool_name: str, schema: Any) -> bool:
    """
    Validate the structure of a local tool's ``SCHEMA`` export.

    Must be a dict with a ``"function"`` key containing at least
    ``"name"`` (str) and ``"parameters"`` (dict).

    :param tool_name: The tool name for error messages.
    :param schema: The ``SCHEMA`` value from the module.
    :returns: ``True`` if the schema is well-formed.
    """
    if not isinstance(schema, dict):
        _logger.warning(
            "Local tool %r: SCHEMA must be a dict, got %s — skipping",
            tool_name,
            type(schema).__name__,
        )
        return False
    func = schema.get("function")
    if not isinstance(func, dict):
        _logger.warning(
            "Local tool %r: SCHEMA missing 'function' dict — skipping",
            tool_name,
        )
        return False
    if not isinstance(func.get("name"), str):
        _logger.warning(
            "Local tool %r: SCHEMA.function.name must be a string — skipping",
            tool_name,
        )
        return False
    if not isinstance(func.get("parameters"), dict):
        _logger.warning(
            "Local tool %r: SCHEMA.function.parameters must be a dict — skipping",
            tool_name,
        )
        return False
    return True


def _validate_module(tool_name: str, module: ModuleType) -> bool:
    """
    Validate that a loaded module has the required exports.

    Checks for ``SCHEMA`` (dict with ``"function"`` key containing
    ``"name"`` and ``"parameters"``) and ``run`` (must be
    ``async def``).

    :param tool_name: The tool name for error messages.
    :param module: The loaded Python module.
    :returns: ``True`` if the module passes all checks.
    """
    if not hasattr(module, "SCHEMA"):
        _logger.warning(
            "Local tool %r: module missing SCHEMA — skipping",
            tool_name,
        )
        return False
    if not _validate_schema(tool_name, module.SCHEMA):
        return False
    schema_name = module.SCHEMA["function"]["name"]
    if schema_name != tool_name:
        _logger.warning(
            "Local tool %r: SCHEMA.function.name is %r but filename "
            "derives %r — the LLM calls the schema name, so these "
            "must match. Skipping",
            tool_name,
            schema_name,
            tool_name,
        )
        return False
    if not hasattr(module, "run"):
        _logger.warning(
            "Local tool %r: module missing run() function — skipping",
            tool_name,
        )
        return False
    if not callable(module.run):
        _logger.warning(
            "Local tool %r: run is not callable — skipping",
            tool_name,
        )
        return False
    if not inspect.iscoroutinefunction(module.run):
        _logger.warning(
            "Local tool %r: run() must be async def — skipping",
            tool_name,
        )
        return False
    return True
