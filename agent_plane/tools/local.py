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

Execution tiers (in priority order):

1. **Docker** — ``docker run`` with network disabled. Used when
   ``sandbox.docker_image`` is configured.
2. **srt + uv** — ``srt uv run --with ... -- python _runner.py``.
   Used when srt is on PATH, sandbox enabled, and tool has PEP 723
   inline deps.
3. **srt** — ``srt python _runner.py``. Used when srt is on PATH
   and sandbox enabled.
4. **uv** — ``uv run --with ... -- python _runner.py``. Used when
   tool has PEP 723 inline deps and uv is available.
5. **plain** — ``python _runner.py``. Default fallback.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

# Any: OpenAI function schemas contain heterogeneous values
# (strings, ints, nested objects, arrays) — no specific type fits.
from typing import Any

from agent_plane.spec.types import LocalToolInfo, SandboxConfig
from agent_plane.tools._pep723 import parse_inline_metadata
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

# Absolute path to the runner script. Resolved once at import time
# so subprocess invocations don't depend on cwd.
_RUNNER_PATH = str(Path(__file__).parent / "_runner.py")

# Maximum bytes to read from the fd 3 response pipe (1 MiB).
_MAX_RESPONSE_BYTES = 1024 * 1024

# Prefix used by the runner in Docker/stdout mode.
_STDOUT_RESPONSE_PREFIX = "__AP_RESPONSE__:"


class LocalPythonTool(Tool):
    """
    A tool backed by a local Python file, executed in a subprocess.

    The Python file must export ``SCHEMA`` (OpenAI function schema
    dict) and ``async def run(arguments: dict) -> str``.

    :param info: The discovered :class:`LocalToolInfo` from the
        agent spec parser.
    :param schema: The validated SCHEMA dict from the module.
    :param module_path: Absolute path to the tool Python file.
    :param sandbox_config: Sandbox settings from the agent spec.
    :param srt_available: Whether ``srt`` is on PATH.
    :param uv_available: Whether ``uv`` is on PATH.
    """

    def __init__(
        self,
        info: LocalToolInfo,
        schema: dict[str, Any],
        module_path: Path,
        sandbox_config: SandboxConfig,
        srt_available: bool,
        uv_available: bool,
        sandbox_enabled: bool = True,
    ) -> None:
        """
        Initialize from a validated tool file.

        :param info: The :class:`LocalToolInfo` with name and path.
        :param schema: The validated SCHEMA dict.
        :param module_path: Absolute path to the tool file, e.g.
            ``Path("/tmp/cache/ag_abc/tools/python/my_tool.py")``.
        :param sandbox_config: Agent-level sandbox settings
            (docker_image).
        :param srt_available: Whether ``srt`` is on PATH.
        :param uv_available: Whether ``uv`` is on PATH.
        :param sandbox_enabled: Runtime policy for srt sandboxing.
        """
        self._info = info
        self._schema = schema
        self._name: str = info.name
        self._module_path = module_path
        self._sandbox_config = sandbox_config
        self._sandbox_enabled = sandbox_enabled
        self._srt_available = srt_available
        self._uv_available = uv_available
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

        Builds the command via :meth:`_build_command`, spawns the
        subprocess, sends the request on stdin, and reads the JSON
        response from fd 3 (or stdout in Docker mode).

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

        # srt and Docker both wrap the command in their own process
        # chain, so the fd 3 pipe doesn't survive to the inner
        # Python process. Use the stdout protocol instead.
        srt_active = self._srt_available and self._sandbox_enabled
        use_stdout = self._sandbox_config.docker_image is not None or srt_active
        if use_stdout:
            return self._invoke_stdout(request)
        return self._invoke_subprocess(request)

    def _invoke_subprocess(self, request: bytes) -> str:
        """
        Run the tool via a local subprocess with fd 3 pipe.

        :param request: JSON-encoded request bytes.
        :returns: Tool result or error string.
        """
        read_fd, write_fd = os.pipe()
        try:
            env = {**os.environ, "_AP_RESPONSE_FD": str(write_fd)}
            self._proc = subprocess.Popen(
                self._build_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
                env=env,
            )
            os.close(write_fd)
            write_fd = -1

            stdout, stderr = self._proc.communicate(input=request)
            return _read_fd3_response(
                read_fd,
                self._proc.returncode,
                stderr,
            )
        finally:
            self._proc = None
            if write_fd != -1:
                os.close(write_fd)
            os.close(read_fd)

    def _invoke_stdout(self, request: bytes) -> str:
        """
        Run the tool via stdout protocol (for srt and Docker).

        When the command is wrapped by srt or Docker, the fd 3 pipe
        doesn't survive to the inner Python process. The runner
        writes the response to stdout with a ``__AP_RESPONSE__:``
        prefix instead.

        :param request: JSON-encoded request bytes.
        :returns: Tool result or error string.
        """
        try:
            env = {**os.environ, "_AP_RESPONSE_MODE": "stdout"}
            self._proc = subprocess.Popen(
                self._build_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            stdout, stderr = self._proc.communicate(input=request)
            return _read_stdout_response(
                stdout,
                self._proc.returncode,
                stderr,
            )
        finally:
            self._proc = None

    def _build_command(self) -> list[str]:
        """
        Build the subprocess command based on execution tier.

        Priority: Docker > srt+uv > srt > uv > plain.

        :returns: The command list for ``subprocess.Popen``.
        """
        if self._sandbox_config.docker_image is not None:
            return self._build_docker_command()

        base = [sys.executable, _RUNNER_PATH]
        # When both uv and srt are active, uv must run OUTSIDE
        # srt (it needs network access to pypi and write access
        # to its cache). srt wraps only the inner python command.
        if self._info.has_inline_deps and self._uv_available:
            return self._build_uv_command(base)
        base = self._prepend_srt(base)
        return base

    def _build_uv_command(self, base: list[str]) -> list[str]:
        """
        Build a ``uv run --with`` command for tools with PEP 723 deps.

        When srt is also active, uv runs OUTSIDE srt (it needs
        network for pypi and write access to its cache). srt wraps
        only the inner ``python _runner.py`` via uv's ``--``
        separator. Without srt, uv wraps the plain python command.

        Uses ``python`` (not ``sys.executable``) so uv's ephemeral
        venv Python is used and can see installed deps.

        :param base: The base command ``[sys.executable, _runner]``
            (unused — replaced with ``python`` for uv).
        :returns: The uv command list.
        """
        uv_args: list[str] = ["uv", "run"]
        for dep in self._info.inline_deps or []:
            uv_args.extend(["--with", dep])
        if self._srt_available and self._sandbox_enabled:
            # uv runs outside srt; srt wraps the inner python.
            # srt -c receives the python command as a quoted string.
            import shlex

            inner = shlex.join(["python", _RUNNER_PATH])
            uv_args.extend(["--", "srt", "-c", inner])
        else:
            uv_args.extend(["--", "python", _RUNNER_PATH])
        return uv_args

    def _prepend_srt(self, cmd: list[str]) -> list[str]:
        """
        Prepend ``srt`` if sandbox is enabled and available.

        Uses ``srt -c '<command>'`` instead of ``srt arg1 arg2``
        because srt's default mode joins args into ``bash -c``
        without proper quoting, which misinterprets PEP 508
        specifiers like ``>=6.0`` as shell redirects.

        :param cmd: The base command to wrap.
        :returns: The wrapped command, or ``cmd`` unchanged.
        """
        if not (self._srt_available and self._sandbox_enabled):
            return cmd
        # Use srt -c with a properly quoted command string so
        # shell metacharacters in args (e.g. ">=6.0") are preserved.
        import shlex

        return ["srt", "-c", shlex.join(cmd)]

    def _build_docker_command(self) -> list[str]:
        """
        Build a ``docker run`` command for container execution.

        The container runs with network disabled, stdin piped,
        and ``_AP_RESPONSE_MODE=stdout`` so the runner writes
        the response to stdout instead of fd 3.

        :returns: The docker run command list.
        """
        image = self._sandbox_config.docker_image
        return [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "-e",
            "_AP_RESPONSE_MODE=stdout",
            image or "",
            "python",
            "-c",
            # Inline the runner as a one-liner that reads stdin
            # and writes to stdout. The full _runner.py is not
            # available inside the container.
            (
                "import sys,json,importlib.util,asyncio,os;"
                "os.environ['_AP_RESPONSE_MODE']='stdout';"
                f"exec(open('{_RUNNER_PATH}').read())"
            ),
        ]

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


def _read_fd3_response(
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


def _read_stdout_response(
    stdout: bytes,
    returncode: int,
    stderr: bytes,
) -> str:
    """
    Read and parse the JSON response from stdout (Docker mode).

    Scans stdout for the ``__AP_RESPONSE__:`` prefix line.

    :param stdout: Captured stdout bytes.
    :param returncode: The subprocess exit code.
    :param stderr: Captured stderr bytes for error reporting.
    :returns: The tool's result string, or an error string.
    """
    for line in stdout.decode(errors="replace").splitlines():
        if line.startswith(_STDOUT_RESPONSE_PREFIX):
            payload = line[len(_STDOUT_RESPONSE_PREFIX) :]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                return f"Error: invalid JSON in stdout response: {exc}"
            if "error" in data:
                return f"Error: {data['error']}"
            result: str = data.get("result", "")
            return result

    stderr_text = stderr.decode(errors="replace").strip()
    if returncode != 0:
        return f"Error: tool subprocess exited with code {returncode}: {stderr_text}"
    return f"Error: tool produced no response. stderr: {stderr_text}"


# ── Module loading and validation (for SCHEMA extraction) ───


def load_local_python_tools(
    local_tools: list[LocalToolInfo],
    workdir: Path,
    sandbox_config: SandboxConfig | None = None,
    srt_available: bool | None = None,
    uv_available: bool | None = None,
    sandbox_enabled: bool = True,
) -> list[LocalPythonTool]:
    """
    Load and validate local Python tools from the agent image.

    Each tool file is imported to extract and validate its ``SCHEMA``.
    The module is NOT stored — tool execution happens in a subprocess
    via ``_runner.py``, not in-process. Tool source is scanned for
    PEP 723 inline metadata to detect dependencies.

    :param local_tools: Discovered :class:`LocalToolInfo` entries
        from the agent spec.
    :param workdir: The agent image's extracted directory on disk,
        e.g. ``Path("/tmp/agent-cache/ag_abc123")``.
    :param sandbox_config: Sandbox settings. ``None`` uses defaults.
    :param srt_available: Whether ``srt`` is on PATH. ``None``
        auto-detects.
    :param uv_available: Whether ``uv`` is on PATH. ``None``
        auto-detects.
    :returns: List of successfully validated :class:`LocalPythonTool`
        instances.
    """
    effective_sandbox = sandbox_config or SandboxConfig()
    effective_srt = srt_available if srt_available is not None else shutil.which("srt") is not None
    effective_uv = uv_available if uv_available is not None else shutil.which("uv") is not None

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

        # Scan for PEP 723 inline metadata before loading the module.
        _scan_inline_metadata(info, tool_path)

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
                sandbox_config=effective_sandbox,
                srt_available=effective_srt,
                uv_available=effective_uv,
                sandbox_enabled=sandbox_enabled,
            )
        )
    return tools


def _scan_inline_metadata(info: LocalToolInfo, path: Path) -> None:
    """
    Scan a tool file for PEP 723 inline script metadata.

    Mutates ``info.has_inline_deps`` and ``info.inline_deps``
    in place if dependencies are found.

    :param info: The :class:`LocalToolInfo` to update.
    :param path: Path to the Python file.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return
    metadata = parse_inline_metadata(source)
    if metadata is not None:
        info.has_inline_deps = True
        info.inline_deps = metadata.dependencies


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
