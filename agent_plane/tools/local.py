"""Local Python tool execution.

Loads Python tool files from the agent's ``tools/python/`` directory
and exposes them as :class:`Tool` instances. Each Python file must
export:

- ``SCHEMA``: An OpenAI function-format dict with ``"type": "function"``
  and a ``"function"`` sub-dict containing ``name``, ``description``,
  and ``parameters``.
- ``run(arguments: dict) -> str``: The callable that executes the tool.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType

# Any: OpenAI function schemas contain heterogeneous values
# (strings, ints, nested objects, arrays) — no specific type fits.
from typing import Any

from agent_plane.spec.types import LocalToolInfo
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)


class LocalPythonTool(Tool):
    """
    A tool backed by a local Python file in the agent image.

    The Python file must export ``SCHEMA`` (OpenAI function schema
    dict) and ``run(arguments: dict) -> str``.

    :param info: The discovered :class:`LocalToolInfo` from the
        agent spec parser.
    :param module: The loaded Python module containing ``SCHEMA``
        and ``run``.
    """

    def __init__(self, info: LocalToolInfo, module: ModuleType) -> None:
        """
        Initialize from a loaded Python module.

        :param info: The :class:`LocalToolInfo` with name and path.
        :param module: The imported module with ``SCHEMA`` and ``run``.
        """
        self._info = info
        self._module = module
        self._schema: dict[str, Any] = module.SCHEMA
        self._run_fn = module.run
        self._name: str = info.name

    def name(self) -> str:  # type: ignore[override]
        """
        Tool name derived from the filename.

        :returns: The tool name, e.g. ``"web.fetch"``.
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
        Execute the tool by calling the module's ``run`` function.

        :param arguments: JSON-encoded arguments string from the LLM.
        :param ctx: Server-side execution context (unused by
            local tools, required by the :class:`Tool` interface).
        :returns: The tool's string result.
        """
        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        result: str = self._run_fn(parsed)
        return result


def load_local_python_tools(
    local_tools: list[LocalToolInfo],
    workdir: Path,
) -> list[LocalPythonTool]:
    """
    Load all local Python tools from the agent's working directory.

    Each tool file is imported as a standalone module. Files that
    fail to load (missing ``SCHEMA``, missing ``run``, import errors)
    are logged and skipped.

    :param local_tools: Discovered :class:`LocalToolInfo` entries
        from the agent spec, e.g.
        ``[LocalToolInfo(name="web.fetch", path="tools/python/web_fetch.py", ...)]``.
    :param workdir: The agent image's extracted directory on disk,
        e.g. ``Path("/tmp/agent-cache/ag_abc123")``.
    :returns: List of successfully loaded :class:`LocalPythonTool`
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
        tools.append(LocalPythonTool(info=info, module=module))
    return tools


def _load_module(tool_name: str, path: Path) -> ModuleType | None:
    """
    Import a Python file as a standalone module.

    :param tool_name: The tool name for error messages.
    :param path: Absolute path to the Python file.
    :returns: The loaded module, or ``None`` on import error.
    """
    module_name = f"_agent_tool_{tool_name.replace('.', '_')}"
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


def _validate_module(tool_name: str, module: ModuleType) -> bool:
    """
    Validate that a loaded module has the required exports.

    :param tool_name: The tool name for error messages.
    :param module: The loaded Python module.
    :returns: ``True`` if the module has ``SCHEMA`` and ``run``.
    """
    if not hasattr(module, "SCHEMA"):
        _logger.warning(
            "Local tool %r: module missing SCHEMA — skipping",
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
    return True
