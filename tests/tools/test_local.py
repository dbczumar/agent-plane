"""Tests for agent_plane.tools.local (LocalPythonTool subprocess execution)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_plane.spec.types import LocalToolInfo
from agent_plane.tools.base import ToolContext
from agent_plane.tools.local import (
    load_local_python_tools,
)


def _write_tool_file(
    tools_dir: Path,
    filename: str,
    schema_name: str,
    body: str = 'return f"result: {arguments}"',
) -> None:
    """
    Write a minimal local tool Python file with async run().

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

async def run(arguments: dict[str, Any]) -> str:
    """Execute the tool."""
    {body}
'''
    (tools_dir / filename).write_text(code)


# ── Subprocess invocation ──────────────────────────────


def test_invoke_subprocess_success(tmp_path: Path, tool_ctx: ToolContext) -> None:
    """
    A valid tool executes in a subprocess and returns its result
    via the fd 3 protocol.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(py_dir, "echo_tool.py", "echo_tool")
    info = LocalToolInfo(
        name="echo_tool",
        path="tools/python/echo_tool.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert len(tools) == 1
    tool = tools[0]
    result = tool.invoke(json.dumps({"input": "hello"}), tool_ctx)
    assert "hello" in result, f"Expected 'hello' in tool output, got {result!r}."


def test_invoke_subprocess_crash_isolation(
    tmp_path: Path,
    tool_ctx: ToolContext,
) -> None:
    """
    A tool that calls ``os._exit(1)`` kills only the subprocess,
    not the server. The parent gets an error string.

    **What breaks if wrong**: in-process execution would kill the
    entire server process.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(
        py_dir,
        "crasher.py",
        "crasher",
        body="import os; os._exit(1)",
    )
    info = LocalToolInfo(
        name="crasher",
        path="tools/python/crasher.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert len(tools) == 1
    result = tools[0].invoke(json.dumps({}), tool_ctx)
    # The server is still alive (we're executing this assertion).
    # The tool returned an error string.
    assert "Error" in result, f"Expected error string from crashed subprocess, got {result!r}"


def test_invoke_subprocess_exception(
    tmp_path: Path,
    tool_ctx: ToolContext,
) -> None:
    """
    A tool that raises an exception returns a structured error
    string (not a crash).
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(
        py_dir,
        "raiser.py",
        "raiser",
        body='raise ValueError("bad input")',
    )
    info = LocalToolInfo(
        name="raiser",
        path="tools/python/raiser.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert len(tools) == 1
    result = tools[0].invoke(json.dumps({}), tool_ctx)
    assert "ValueError" in result
    assert "bad input" in result


def test_invoke_empty_args(tmp_path: Path, tool_ctx: ToolContext) -> None:
    """
    invoke('') passes an empty dict to run(), not raising
    JSONDecodeError.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(
        py_dir,
        "empty_args.py",
        "empty_args",
        body='return f"got: {arguments}"',
    )
    info = LocalToolInfo(
        name="empty_args",
        path="tools/python/empty_args.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    result = tools[0].invoke("", tool_ctx)
    assert "got: {}" in result


def test_cancel_kills_subprocess(tmp_path: Path, tool_ctx: ToolContext) -> None:
    """
    cancel() sends SIGKILL to the subprocess. After cancel, the
    subprocess is dead and communicate() unblocks.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(
        py_dir,
        "sleeper.py",
        "sleeper",
        body="import time; time.sleep(60); return 'done'",
    )
    info = LocalToolInfo(
        name="sleeper",
        path="tools/python/sleeper.py",
        language="python",
    )
    # Explicitly disable srt so we test the fd 3 subprocess path.
    # srt wrapping uses a different process tree that is tested
    # separately via integration tests.
    tools = load_local_python_tools(
        [info],
        tmp_path,
        srt_available=False,
        uv_available=False,
    )
    tool = tools[0]

    import threading

    result_holder: list[str] = []

    def _invoke() -> None:
        result_holder.append(tool.invoke(json.dumps({}), tool_ctx))

    t = threading.Thread(target=_invoke)
    t.start()

    # Give the subprocess time to start, then cancel.
    import time

    time.sleep(0.5)
    tool.cancel()
    t.join(timeout=5.0)
    assert not t.is_alive(), "invoke() should have unblocked after cancel()"
    assert len(result_holder) == 1
    assert "Error" in result_holder[0]


# ── Loading and validation ─────────────────────────────


def test_load_multiple_tools(tmp_path: Path) -> None:
    """
    Multiple valid tool files are all loaded.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(py_dir, "tool_a.py", "tool_a")
    _write_tool_file(py_dir, "tool_b.py", "tool_b")
    infos = [
        LocalToolInfo(name="tool_a", path="tools/python/tool_a.py", language="python"),
        LocalToolInfo(name="tool_b", path="tools/python/tool_b.py", language="python"),
    ]
    tools = load_local_python_tools(infos, tmp_path)
    assert len(tools) == 2
    names = {t.name() for t in tools}
    assert names == {"tool_a", "tool_b"}


def test_load_skips_missing_file(tmp_path: Path) -> None:
    """
    A LocalToolInfo pointing to a nonexistent file is skipped.
    """
    info = LocalToolInfo(
        name="missing_tool",
        path="tools/python/missing_tool.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


def test_load_skips_missing_schema(tmp_path: Path) -> None:
    """
    A Python file without a SCHEMA export is skipped.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "no_schema.py").write_text('async def run(args):\n    return "ok"\n')
    info = LocalToolInfo(
        name="no_schema",
        path="tools/python/no_schema.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


def test_load_skips_missing_run(tmp_path: Path) -> None:
    """
    A Python file without a run() function is skipped.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "no_run.py").write_text(
        'SCHEMA = {"type": "function", "function": '
        '{"name": "no_run", "parameters": {"type": "object"}}}\n'
    )
    info = LocalToolInfo(
        name="no_run",
        path="tools/python/no_run.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


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
    assert tools == []


def test_load_skips_typescript(tmp_path: Path) -> None:
    """
    TypeScript local tools are skipped (not yet supported).
    """
    info = LocalToolInfo(
        name="ts_tool",
        path="tools/typescript/ts_tool.ts",
        language="typescript",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


def test_load_skips_sync_run(tmp_path: Path) -> None:
    """
    A Python file with a sync ``def run()`` is rejected.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "sync_tool.py").write_text(
        'SCHEMA = {"type": "function", "function": '
        '{"name": "sync_tool", "parameters": {"type": "object"}}}\n'
        'def run(args):\n    return "sync"\n'
    )
    info = LocalToolInfo(
        name="sync_tool",
        path="tools/python/sync_tool.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


def test_load_skips_schema_missing_function_key(tmp_path: Path) -> None:
    """
    SCHEMA without a ``"function"`` key is rejected.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "bad_schema.py").write_text(
        'SCHEMA = {"type": "function"}\nasync def run(args):\n    return "ok"\n'
    )
    info = LocalToolInfo(
        name="bad_schema",
        path="tools/python/bad_schema.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


def test_load_skips_schema_name_mismatch(tmp_path: Path) -> None:
    """
    SCHEMA.function.name differs from filename-derived name -> rejected.
    """
    py_dir = tmp_path / "tools" / "python"
    _write_tool_file(py_dir, "my_tool.py", "different_name")
    info = LocalToolInfo(
        name="my_tool",
        path="tools/python/my_tool.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert tools == []


# ── Command construction tiers ──────────────────────────


def test_build_command_plain(tmp_path: Path) -> None:
    """
    Default tier: plain ``python _runner.py``.
    """
    from agent_plane.spec.types import SandboxConfig
    from agent_plane.tools.local import _RUNNER_PATH, LocalPythonTool

    info = LocalToolInfo(name="t", path="t.py", language="python")
    tool = LocalPythonTool(
        info=info,
        schema={"type": "function", "function": {"name": "t", "parameters": {}}},
        module_path=tmp_path / "t.py",
        sandbox_config=SandboxConfig(),
        srt_available=False,
        uv_available=False,
    )
    cmd = tool._build_command()
    assert cmd == [sys.executable, _RUNNER_PATH]


def test_build_command_with_uv(tmp_path: Path) -> None:
    """
    When tool has PEP 723 deps and uv is available, command
    is wrapped with ``uv run --with``.
    """
    from agent_plane.spec.types import SandboxConfig
    from agent_plane.tools.local import _RUNNER_PATH, LocalPythonTool

    info = LocalToolInfo(
        name="t",
        path="t.py",
        language="python",
        has_inline_deps=True,
        inline_deps=["requests>=2.28", "pandas"],
    )
    tool = LocalPythonTool(
        info=info,
        schema={"type": "function", "function": {"name": "t", "parameters": {}}},
        module_path=tmp_path / "t.py",
        sandbox_config=SandboxConfig(),
        srt_available=False,
        uv_available=True,
    )
    cmd = tool._build_command()
    assert cmd[:2] == ["uv", "run"]
    assert "--with" in cmd
    assert "requests>=2.28" in cmd
    assert "pandas" in cmd
    # uv replaces sys.executable with "python" so the venv's
    # Python is used and can see installed deps.
    assert cmd[-2:] == ["python", _RUNNER_PATH]


def test_build_command_with_srt(tmp_path: Path) -> None:
    """
    When srt is available and sandbox enabled, command is
    wrapped with ``srt``.
    """
    from agent_plane.spec.types import SandboxConfig
    from agent_plane.tools.local import _RUNNER_PATH, LocalPythonTool

    info = LocalToolInfo(name="t", path="t.py", language="python")
    tool = LocalPythonTool(
        info=info,
        schema={"type": "function", "function": {"name": "t", "parameters": {}}},
        module_path=tmp_path / "t.py",
        sandbox_config=SandboxConfig(enabled=True),
        srt_available=True,
        uv_available=False,
    )
    cmd = tool._build_command()
    # srt uses -c with a quoted command string to avoid shell
    # metacharacter issues.
    assert cmd[0] == "srt"
    assert cmd[1] == "-c"
    assert sys.executable in cmd[2]
    assert _RUNNER_PATH in cmd[2]


def test_build_command_srt_disabled(tmp_path: Path) -> None:
    """
    When sandbox.enabled is False, srt is NOT prepended even
    when available on PATH.
    """
    from agent_plane.spec.types import SandboxConfig
    from agent_plane.tools.local import _RUNNER_PATH, LocalPythonTool

    info = LocalToolInfo(name="t", path="t.py", language="python")
    tool = LocalPythonTool(
        info=info,
        schema={"type": "function", "function": {"name": "t", "parameters": {}}},
        module_path=tmp_path / "t.py",
        sandbox_config=SandboxConfig(enabled=False),
        srt_available=True,
        uv_available=False,
    )
    cmd = tool._build_command()
    # Plain command, no srt prefix
    assert cmd == [sys.executable, _RUNNER_PATH]


def test_build_command_uv_outside_srt_inside(tmp_path: Path) -> None:
    """
    When both srt and uv are active with inline deps, uv runs
    outside srt (needs network for pypi) and srt wraps the inner
    python command: ``uv run --with ... -- srt -c 'python ...'``.
    """
    from agent_plane.spec.types import SandboxConfig
    from agent_plane.tools.local import LocalPythonTool

    info = LocalToolInfo(
        name="t",
        path="t.py",
        language="python",
        has_inline_deps=True,
        inline_deps=["httpx"],
    )
    tool = LocalPythonTool(
        info=info,
        schema={"type": "function", "function": {"name": "t", "parameters": {}}},
        module_path=tmp_path / "t.py",
        sandbox_config=SandboxConfig(enabled=True),
        srt_available=True,
        uv_available=True,
    )
    cmd = tool._build_command()
    # uv runs first (outside srt), srt wraps the inner python
    assert cmd[0] == "uv"
    assert "srt" in cmd
    assert "-c" in cmd


def test_build_command_docker(tmp_path: Path) -> None:
    """
    When docker_image is configured, command uses ``docker run``.
    """
    from agent_plane.spec.types import SandboxConfig
    from agent_plane.tools.local import LocalPythonTool

    info = LocalToolInfo(name="t", path="t.py", language="python")
    tool = LocalPythonTool(
        info=info,
        schema={"type": "function", "function": {"name": "t", "parameters": {}}},
        module_path=tmp_path / "t.py",
        sandbox_config=SandboxConfig(docker_image="python:3.12-slim"),
        srt_available=True,
        uv_available=True,
    )
    cmd = tool._build_command()
    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--network" in cmd
    assert "none" in cmd
    assert "python:3.12-slim" in cmd


def test_pep723_scanning_at_load_time(tmp_path: Path) -> None:
    """
    Tool files with PEP 723 inline metadata have has_inline_deps
    set at load time.
    """
    py_dir = tmp_path / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "with_deps.py").write_text("""\
# /// script
# dependencies = ["requests"]
# ///

from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "with_deps",
        "description": "Test.",
        "parameters": {"type": "object"},
    },
}

async def run(args: dict[str, Any]) -> str:
    return "ok"
""")
    info = LocalToolInfo(
        name="with_deps",
        path="tools/python/with_deps.py",
        language="python",
    )
    tools = load_local_python_tools([info], tmp_path)
    assert len(tools) == 1
    assert info.has_inline_deps is True
    assert info.inline_deps == ["requests"]


# ── Runner subprocess (direct invocation) ──────────────


def test_runner_valid_tool(tmp_path: Path) -> None:
    """
    The runner subprocess executes a valid tool and writes the
    result to fd 3.
    """
    tool_file = tmp_path / "tool.py"
    tool_file.write_text(
        "async def run(args):\n    return f\"hello {args.get('name', 'world')}\"\n"
    )

    read_fd, write_fd = os.pipe()
    env = {**os.environ, "_AP_RESPONSE_FD": str(write_fd)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_plane.tools._runner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        env=env,
    )
    os.close(write_fd)
    request = json.dumps(
        {
            "module_path": str(tool_file),
            "arguments": {"name": "test"},
        }
    ).encode()
    proc.communicate(input=request)

    raw = os.read(read_fd, 1024 * 1024)
    os.close(read_fd)
    data = json.loads(raw)
    assert data == {"result": "hello test"}


def test_runner_import_error(tmp_path: Path) -> None:
    """
    The runner returns a structured error when the tool fails
    to import.
    """
    tool_file = tmp_path / "bad_tool.py"
    tool_file.write_text('raise RuntimeError("broken")\n')

    read_fd, write_fd = os.pipe()
    env = {**os.environ, "_AP_RESPONSE_FD": str(write_fd)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_plane.tools._runner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        env=env,
    )
    os.close(write_fd)
    request = json.dumps(
        {
            "module_path": str(tool_file),
            "arguments": {},
        }
    ).encode()
    proc.communicate(input=request)

    raw = os.read(read_fd, 1024 * 1024)
    os.close(read_fd)
    data = json.loads(raw)
    assert "error" in data
    assert "Import error" in data["error"]


def test_runner_runtime_error(tmp_path: Path) -> None:
    """
    The runner returns a structured error when run() raises.
    """
    tool_file = tmp_path / "raiser.py"
    tool_file.write_text('async def run(args):\n    raise TypeError("bad type")\n')

    read_fd, write_fd = os.pipe()
    env = {**os.environ, "_AP_RESPONSE_FD": str(write_fd)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_plane.tools._runner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        env=env,
    )
    os.close(write_fd)
    request = json.dumps(
        {
            "module_path": str(tool_file),
            "arguments": {},
        }
    ).encode()
    proc.communicate(input=request)

    raw = os.read(read_fd, 1024 * 1024)
    os.close(read_fd)
    data = json.loads(raw)
    assert "error" in data
    assert "TypeError" in data["error"]
    assert "bad type" in data["error"]
