"""Local Python tool execution via subprocess.

Loads ``@tool``-decorated functions from the agent's
``tools/python/`` directory and exposes each as a
:class:`LocalPythonTool` instance. A single Python file may
export multiple tools (one per ``@tool`` function); the loader
expands one :class:`LocalToolInfo` (file-level) into N
``LocalPythonTool`` instances.

Tool code runs in a **subprocess** (not in-process) for crash
isolation. Communication uses the fd 3 pipe protocol — see
``_runner.py`` for the child side. The subprocess invocation
identifies the target ``@tool`` function by name, since one
file may host several.

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
from agent_plane.tools.decorator import ToolMetadata, get_tool_metadata

_logger = logging.getLogger(__name__)

# Absolute path to the runner script. Resolved once at import time
# so subprocess invocations don't depend on cwd.
_RUNNER_PATH = str(Path(__file__).parent / "_runner.py")

# Maximum bytes to read from the fd 3 response pipe (1 MiB).
_MAX_RESPONSE_BYTES = 1024 * 1024

# Prefix used by the runner in Docker/stdout mode.
_STDOUT_RESPONSE_PREFIX = "__AP_RESPONSE__:"


class LocalToolLoadError(Exception):
    """
    Raised when an agent image's local tool files fail to load.

    Surfaces a single actionable error per agent image. Carries
    enough context (agent name, file path, function name, cause)
    that authors can fix the offending file without further
    debugging.
    """


class LocalPythonTool(Tool):
    """
    A tool backed by a ``@tool``-decorated function in a local Python file.

    One file may export multiple tools; the framework instantiates
    one :class:`LocalPythonTool` per decorated function. The
    subprocess runner re-imports the file and dispatches to the
    named function.

    :param info: The discovered :class:`LocalToolInfo` for the
        file this tool lives in.
    :param metadata: The :class:`ToolMetadata` extracted from the
        ``@tool``-decorated function at agent-image load time.
    :param module_path: Absolute path to the tool Python file.
    :param sandbox_config: Sandbox settings from the agent spec.
    :param srt_available: Whether ``srt`` is on PATH.
    :param uv_available: Whether ``uv`` is on PATH.
    :param sandbox_enabled: Runtime policy for srt sandboxing.
    """

    def __init__(
        self,
        info: LocalToolInfo,
        metadata: ToolMetadata,
        module_path: Path,
        sandbox_config: SandboxConfig,
        srt_available: bool,
        uv_available: bool,
        sandbox_enabled: bool = True,
    ) -> None:
        """
        Initialize from a discovered ``@tool`` function.

        :param info: The :class:`LocalToolInfo` for the source file.
        :param metadata: The :class:`ToolMetadata` produced by
            ``@tool`` at decoration time.
        :param module_path: Absolute path to the tool file, e.g.
            ``Path("/tmp/cache/ag_abc/tools/python/my_tools.py")``.
        :param sandbox_config: Agent-level sandbox settings
            (docker_image).
        :param srt_available: Whether ``srt`` is on PATH.
        :param uv_available: Whether ``uv`` is on PATH.
        :param sandbox_enabled: Runtime policy for srt sandboxing.
        """
        self._info = info
        self._metadata = metadata
        self._module_path = module_path
        self._sandbox_config = sandbox_config
        self._sandbox_enabled = sandbox_enabled
        self._srt_available = srt_available
        self._uv_available = uv_available
        self._proc: subprocess.Popen[bytes] | None = None
        self._workspace: Path | None = None

    def name(self) -> str:  # type: ignore[override]
        """
        Tool name derived from the ``@tool``-decorated function's ``__name__``.

        :returns: The tool name as the LLM sees it, e.g. ``"word_count"``.
        """
        return self._metadata.name

    def is_async(self) -> bool:
        """
        Return ``True`` for ``@tool(synchronous=False)`` functions.

        Reads the flag captured at decoration time. Async tools
        bypass the inline ``invoke()`` path entirely — the runtime
        starts a background workflow and returns a handle to the
        LLM (D2/D3); the actual tool body still runs in a
        subprocess, just inside the background workflow's
        ``@step`` rather than the parent workflow.

        :returns: ``True`` if the tool was decorated with
            ``synchronous=False``.
        """
        return not self._metadata.synchronous

    def module_path(self) -> str:
        """
        Return the absolute path to the tool's source file.

        Used by the background-tool-workflow dispatch path so the
        runner subprocess knows which file to import. Exposed here
        rather than reading ``_module_path`` directly from outside
        the class.

        :returns: Absolute path string, e.g.
            ``"/tmp/cache/ag_abc/tools/python/my_tools.py"``.
        """
        return str(self._module_path)

    @classmethod
    def description(cls) -> str:
        """
        :returns: Generic class-level description; per-instance
            descriptions come from each function's docstring via
            :meth:`get_schema`.
        """
        return "Custom local Python tool."

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function-format tool schema.

        Composes the metadata's name + description + JSON schema
        into the wire-format the framework's tool-dispatch layer
        expects.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict.
        """
        return {
            "type": "function",
            "function": {
                "name": self._metadata.name,
                "description": self._metadata.description,
                "parameters": self._metadata.json_schema,
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Execute the tool in a subprocess via the fd 3 protocol.

        Builds the command via :meth:`_build_command`, spawns the
        subprocess, sends the request on stdin, and reads the JSON
        response from fd 3 (or stdout in Docker mode). The request
        carries the target function name so the runner knows which
        ``@tool`` to dispatch (one file may export several).

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
                "tool_name": self._metadata.name,
                "arguments": parsed,
            }
        ).encode()

        # Pass workspace to the subprocess so local tools can resolve
        # relative paths (e.g. validate_agent resolving sandbox dirs).
        self._workspace = ctx.workspace

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
            if self._workspace is not None:
                env["_AP_WORKSPACE"] = str(self._workspace)
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
            if self._workspace is not None:
                env["_AP_WORKSPACE"] = str(self._workspace)
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
    if not stdout:
        stderr_text = stderr.decode(errors="replace").strip()
        if returncode != 0:
            return f"Error: tool subprocess exited with code {returncode}: {stderr_text}"
        return f"Error: tool produced no stdout. stderr: {stderr_text}"

    text = stdout.decode(errors="replace")
    for line in text.splitlines():
        if line.startswith(_STDOUT_RESPONSE_PREFIX):
            payload = line[len(_STDOUT_RESPONSE_PREFIX) :]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                return f"Error: invalid JSON response from tool: {exc}"
            if "error" in data:
                return f"Error: {data['error']}"
            return str(data.get("result", ""))

    stderr_text = stderr.decode(errors="replace").strip()
    return (
        f"Error: tool produced no recognized response (no "
        f"{_STDOUT_RESPONSE_PREFIX} prefix found). "
        f"exit={returncode} stderr={stderr_text!r}"
    )


def load_local_python_tools(
    local_tools: list[LocalToolInfo],
    workdir: Path,
    sandbox_config: SandboxConfig | None = None,
    srt_available: bool | None = None,
    uv_available: bool | None = None,
    sandbox_enabled: bool = True,
    *,
    agent_name: str | None = None,
    builtin_tool_names: frozenset[str] | None = None,
) -> list[LocalPythonTool]:
    """
    Load and validate local Python tools from the agent image.

    Each file is imported once at agent-image load time. Every
    ``@tool``-decorated function in the module produces one
    :class:`LocalPythonTool`. Names are validated against any
    builtin names provided (collisions fail loud per G27) and
    against each other (two custom tools sharing a name across
    files fail loud).

    :param local_tools: Discovered :class:`LocalToolInfo` entries
        from the agent spec parser (one per file).
    :param workdir: The agent image's extracted directory on disk,
        e.g. ``Path("/tmp/agent-cache/ag_abc123")``.
    :param sandbox_config: Sandbox settings. ``None`` uses defaults.
    :param srt_available: Whether ``srt`` is on PATH. ``None``
        auto-detects.
    :param uv_available: Whether ``uv`` is on PATH. ``None``
        auto-detects.
    :param sandbox_enabled: Runtime policy for srt sandboxing.
    :param agent_name: The agent's name, used in error messages.
        ``None`` falls back to the workdir basename.
    :param builtin_tool_names: Names of framework-provided built-in
        tools enabled for this agent. Used for collision detection
        (G27). ``None`` means skip the builtin-collision check
        (caller already validated, or no builtins active).
    :returns: List of successfully loaded :class:`LocalPythonTool`
        instances, one per ``@tool`` function across all files.
    :raises LocalToolLoadError: If any file fails to load (import
        error, no decorated functions, name collision).
    """
    effective_sandbox = sandbox_config or SandboxConfig()
    effective_srt = srt_available if srt_available is not None else shutil.which("srt") is not None
    effective_uv = uv_available if uv_available is not None else shutil.which("uv") is not None
    effective_agent_name = agent_name or workdir.name

    # Discover decorated functions per file. Track tool name -> source so
    # we can detect cross-file collisions and surface them with both paths.
    discovered: dict[str, _DiscoveredTool] = {}

    for info in local_tools:
        if info.language != "python":
            continue
        tool_path = workdir / info.path
        if not tool_path.is_file():
            raise LocalToolLoadError(
                f"Agent {effective_agent_name!r}: tool file declared at "
                f"{info.path!r} but not found on disk."
            )

        # Scan for PEP 723 inline metadata before loading the module.
        _scan_inline_metadata(info, tool_path)

        module = _import_tool_module(
            agent_name=effective_agent_name,
            tool_path=tool_path,
        )
        functions = _extract_decorated_functions(
            agent_name=effective_agent_name,
            tool_path=tool_path,
            module=module,
        )

        for tool_name, metadata in functions:
            # Detect collision with another custom tool already discovered.
            existing = discovered.get(tool_name)
            if existing is not None:
                raise LocalToolLoadError(
                    f"Tool name collision in agent {effective_agent_name!r}: "
                    f"'{tool_name}' is defined in both "
                    f"{existing.info.path!r} and {info.path!r}. "
                    f"Rename one of the @tool functions so each name is unique."
                )
            # Detect collision with a builtin.
            if builtin_tool_names is not None and tool_name in builtin_tool_names:
                raise LocalToolLoadError(
                    f"Tool name collision in agent {effective_agent_name!r}: "
                    f"custom tool '{tool_name}' (defined in {info.path!r}) "
                    f"conflicts with built-in tool '{tool_name}'. "
                    f"Rename the custom tool or remove the conflicting builtin "
                    f"from config.yaml's tools.builtins list."
                )
            discovered[tool_name] = _DiscoveredTool(
                info=info,
                metadata=metadata,
                module_path=tool_path.resolve(),
            )

    return [
        LocalPythonTool(
            info=disc.info,
            metadata=disc.metadata,
            module_path=disc.module_path,
            sandbox_config=effective_sandbox,
            srt_available=effective_srt,
            uv_available=effective_uv,
            sandbox_enabled=sandbox_enabled,
        )
        for disc in discovered.values()
    ]


class _DiscoveredTool:
    """
    Internal record produced during loader discovery, before the
    final :class:`LocalPythonTool` instances are constructed.

    Lives only inside :func:`load_local_python_tools`; not part
    of the public API.

    :param info: The :class:`LocalToolInfo` for the source file.
    :param metadata: The :class:`ToolMetadata` from the ``@tool``
        decoration.
    :param module_path: Resolved absolute path to the source file.
    """

    __slots__ = ("info", "metadata", "module_path")

    def __init__(
        self,
        info: LocalToolInfo,
        metadata: ToolMetadata,
        module_path: Path,
    ) -> None:
        self.info = info
        self.metadata = metadata
        self.module_path = module_path


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


def _import_tool_module(
    *,
    agent_name: str,
    tool_path: Path,
) -> ModuleType:
    """
    Import a tool file as a standalone module.

    The module is held only long enough to discover decorated
    functions; subsequent invocations re-import in the subprocess
    runner. Failures raise :class:`LocalToolLoadError` with full
    context (agent name, file path, cause).

    :param agent_name: The agent's name, for error messages.
    :param tool_path: Absolute path to the Python file.
    :returns: The loaded module.
    :raises LocalToolLoadError: If the module fails to import.
    """
    module_name = f"_agent_tool_{tool_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    if spec is None or spec.loader is None:
        raise LocalToolLoadError(
            f"Agent {agent_name!r}: cannot create module spec for {tool_path}."
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LocalToolLoadError(
            f"Agent {agent_name!r}: failed to import tool file {tool_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return module


def _extract_decorated_functions(
    *,
    agent_name: str,
    tool_path: Path,
    module: ModuleType,
) -> list[tuple[str, ToolMetadata]]:
    """
    Find every ``@tool``-decorated function defined in ``module``.

    Iterates ``module.__dict__`` looking for callables carrying
    the ``TOOL_MARKER_ATTR`` attribute. Filters to functions
    actually defined IN the module (not re-imported from elsewhere)
    by checking ``__module__`` matches the loaded module's name.

    :param agent_name: The agent's name, for error messages.
    :param tool_path: Path to the tool file (used in errors).
    :param module: The loaded Python module to scan.
    :returns: List of ``(tool_name, ToolMetadata)`` tuples, one
        per decorated function. Empty if none found, in which case
        this function raises (a tool file with no decorated
        functions is a load error).
    :raises LocalToolLoadError: If the module exports no
        ``@tool``-decorated functions.
    """
    found: list[tuple[str, ToolMetadata]] = []
    for value in module.__dict__.values():
        # Only consider objects defined in THIS module (not imports).
        # Re-imported decorated functions would otherwise be doubly
        # registered.
        if not callable(value):
            continue
        if getattr(value, "__module__", None) != module.__name__:
            continue
        metadata = get_tool_metadata(value)
        if metadata is None:
            continue
        found.append((metadata.name, metadata))

    if found:
        return found

    # No decorated functions found. Surface an actionable error so the
    # author knows the file needs to use @tool from agent_plane.tools.
    raise LocalToolLoadError(
        f"Agent {agent_name!r}: tool file {tool_path} exports no "
        f"@tool-decorated functions. Decorate at least one module-level "
        f"function with @tool from agent_plane.tools."
    )
