"""Tests for agent_plane.tools.local (LocalPythonTool)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_plane.spec.types import LocalToolInfo
from agent_plane.tools.base import ToolContext
from agent_plane.tools.local import (
    LocalPythonTool,
    load_local_python_tools,
)


def _write_tool_file(
    tools_dir: Path,
    filename: str,
    schema_name: str,
    body: str = 'return f"result: {arguments}"',
) -> None:
    """
    Write a minimal local tool Python file.

    :param tools_dir: The ``tools/python/`` directory to write into.
    :param filename: File name, e.g. ``"web_fetch.py"``.
    :param schema_name: The ``name`` field in the SCHEMA dict.
    :param body: The body of the ``run`` function.
    """
    tools_dir.mkdir(parents=True, exist_ok=True)
    code = f'''
"""Test tool."""
from typing import Any

SCHEMA: dict[str, Any] = {{
    "type": "function",
    "function": {{
        "name": "{schema_name}",
        "description": "A test tool.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "input": {{"type": "string", "description": "Input value."}},
            }},
            "required": ["input"],
        }},
    }},
}}

def run(arguments: dict[str, Any]) -> str:
    """Execute the tool."""
    {body}
'''
    (tools_dir / filename).write_text(code)


def test_load_valid_tool(tmp_path: Path, tool_ctx: ToolContext) -> None:
    """
    A valid Python tool file with SCHEMA and run() is loaded
    and callable.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(py_dir, "echo_tool.py", "echo_tool")
    info = LocalToolInfo(
        name="echo.tool",
        path="tools/python/echo_tool.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    # 1 tool loaded successfully.
    assert len(tools) == 1, (
        f"Expected 1 tool loaded, got {len(tools)}. "
        f"If 0, the loader failed to find or validate the file."
    )
    tool = tools[0]
    assert tool.name() == "echo.tool"
    schema = tool.get_schema()
    # Schema name matches what the file defined.
    assert schema["function"]["name"] == "echo_tool"
    result = tool.invoke(json.dumps({"input": "hello"}), tool_ctx)
    # run() was called with the parsed arguments.
    assert "hello" in result, (
        f"Expected 'hello' in tool output, got {result!r}. "
        f"If missing, the arguments were not passed to run()."
    )


def test_load_skips_missing_file(tmp_path: Path) -> None:
    """
    A LocalToolInfo pointing to a nonexistent file is skipped
    without error.
    """
    info = LocalToolInfo(
        name="missing.tool",
        path="tools/python/missing_tool.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    # Missing file is skipped, not an error.
    assert tools == [], f"Expected empty list for missing file, got {len(tools)} tool(s)."


def test_load_skips_missing_schema(tmp_path: Path) -> None:
    """
    A Python file without a SCHEMA export is skipped.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "no_schema.py").write_text('def run(args):\n    return "ok"\n')
    info = LocalToolInfo(
        name="no.schema",
        path="tools/python/no_schema.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == [], "Tool without SCHEMA should be skipped."


def test_load_skips_missing_run(tmp_path: Path) -> None:
    """
    A Python file without a run() function is skipped.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "no_run.py").write_text('SCHEMA = {"type": "function", "function": {"name": "x"}}\n')
    info = LocalToolInfo(
        name="no.run",
        path="tools/python/no_run.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == [], "Tool without run() should be skipped."


def test_load_skips_import_error(tmp_path: Path) -> None:
    """
    A Python file that raises on import is skipped.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "broken.py").write_text('raise RuntimeError("broken on import")\n')
    info = LocalToolInfo(
        name="broken",
        path="tools/python/broken.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == [], "Tool with import error should be skipped."


def test_load_skips_typescript(tmp_path: Path) -> None:
    """
    TypeScript local tools are skipped (not yet supported).
    """
    info = LocalToolInfo(
        name="ts.tool",
        path="tools/typescript/ts_tool.ts",
        language="typescript",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == [], "TypeScript tools should be skipped by Python loader."


def test_load_multiple_tools(tmp_path: Path) -> None:
    """
    Multiple valid tool files are all loaded.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(py_dir, "tool_a.py", "tool_a")
    _write_tool_file(py_dir, "tool_b.py", "tool_b")
    infos = [
        LocalToolInfo(name="tool.a", path="tools/python/tool_a.py", language="python"),
        LocalToolInfo(name="tool.b", path="tools/python/tool_b.py", language="python"),
    ]
    tools = load_local_python_tools(infos, tmp_path)
    assert len(tools) == 2, f"Expected 2 tools loaded, got {len(tools)}."
    names = {t.name() for t in tools}
    assert names == {"tool.a", "tool.b"}


def test_invoke_passes_empty_args_for_empty_string(tool_ctx: ToolContext) -> None:
    """
    invoke('') passes an empty dict to run(), not raising
    JSONDecodeError.
    """
    import types

    module = types.ModuleType("test_mod")
    module.SCHEMA = {"type": "function", "function": {"name": "t"}}  # type: ignore[attr-defined]
    module.run = lambda args: f"got: {args}"  # type: ignore[attr-defined]
    info = LocalToolInfo(name="t", path="t.py", language="python")
    tool = LocalPythonTool(info=info, module=module)
    result = tool.invoke("", tool_ctx)
    # Empty string arguments treated as empty dict.
    assert result == "got: {}"
