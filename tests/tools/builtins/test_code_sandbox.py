"""Unit tests for the code_sandbox built-in tool.

Tests the srt settings generation and sandbox read/write isolation
without needing a real LLM or server.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_plane.tools.base import ToolContext
from agent_plane.tools.builtins.code_sandbox import (
    CodeSandboxTool,
    srt_settings_cache,
    system_read_allowlist,
    write_srt_settings,
)


@pytest.fixture(autouse=True)
def _clear_srt_cache() -> None:
    """
    Clear the module-level srt settings cache before each test
    so cached files from previous tests don't interfere.
    """
    srt_settings_cache.clear()


def test_srt_settings_deny_read_is_root(
    tmp_path: Path,
) -> None:
    """
    ``denyRead`` must be ``["/"]`` — deny everything, then
    selectively allow back. An empty ``denyRead`` (the original
    bug) allows agents to read any file on the host.

    **What breaks if wrong**: Agent can ``cat /tmp/secrets``,
    ``ls ~/Documents``, etc.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    settings_path = write_srt_settings(workspace)
    with open(settings_path) as f:
        settings = json.load(f)

    assert settings["filesystem"]["denyRead"] == ["/"], (
        f"denyRead must be ['/'], got: {settings['filesystem']['denyRead']}"
    )


def test_srt_settings_workspace_in_allow_read(
    tmp_path: Path,
) -> None:
    """
    The workspace must appear in ``allowRead`` so the agent
    can read its own files within the denied region.

    **What breaks if wrong**: Agent can't read files it created
    in its own workspace.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    settings_path = write_srt_settings(workspace)
    with open(settings_path) as f:
        settings = json.load(f)

    resolved_ws = str(workspace.resolve())
    assert resolved_ws in settings["filesystem"]["allowRead"], (
        f"allowRead must include resolved workspace '{resolved_ws}'"
    )


def test_srt_settings_restrict_writes_to_workspace(
    tmp_path: Path,
) -> None:
    """
    The srt settings must only allow writes to the workspace.

    **What breaks if wrong**: An agent could write files anywhere
    on the host filesystem.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    settings_path = write_srt_settings(workspace)
    with open(settings_path) as f:
        settings = json.load(f)

    resolved_ws = str(workspace.resolve())
    assert settings["filesystem"]["allowWrite"] == [resolved_ws], (
        f"allowWrite must be exactly [resolved_workspace], "
        f"got: {settings['filesystem']['allowWrite']}"
    )


def test_srt_settings_resolve_symlinks(
    tmp_path: Path,
) -> None:
    """
    Workspace paths must be resolved to canonical form. On macOS,
    ``/var`` → ``/private/var`` — without resolving, the srt
    allow paths won't match the real filesystem.

    **What breaks if wrong**: Workspace reads/writes fail because
    the allow path doesn't match the resolved path srt checks.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    settings_path = write_srt_settings(workspace)
    with open(settings_path) as f:
        settings = json.load(f)

    resolved = str(workspace.resolve())
    assert settings["filesystem"]["allowWrite"] == [resolved]
    assert resolved in settings["filesystem"]["allowRead"]


def test_system_read_allowlist_excludes_user_home() -> None:
    """
    The system read allowlist must not include the user's home
    directory tree. That's the primary attack surface.

    **What breaks if wrong**: Agent can read ``~/Documents``,
    ``~/.ssh``, etc.
    """
    allow = system_read_allowlist()
    home_root = str(Path.home().parent)

    # The home root (e.g. /Users) must not be in the allowlist.
    assert home_root not in allow, f"allowlist must not include user home root '{home_root}'"


def test_system_read_allowlist_includes_bin_dirs() -> None:
    """
    The allowlist must include top-level dirs that contain
    system binaries (derived from PATH), so shell commands
    resolve via PATH.

    **What breaks if wrong**: ``cat``, ``ls``, ``python3`` all
    fail with "command not found."
    """
    allow = system_read_allowlist()
    # /usr covers /usr/bin, /usr/local/bin, etc.
    assert "/usr" in allow or "/bin" in allow, (
        f"allowlist must include /usr or /bin for PATH resolution, got: {allow}"
    )


@pytest.mark.skipif(
    shutil.which("srt") is None,
    reason="srt not on PATH",
)
def test_sandbox_blocks_read_outside_workspace(
    tmp_path: Path,
) -> None:
    """
    With srt enabled, reading a file outside the workspace must
    fail. Directly invokes the tool (no LLM needed).

    **What breaks if wrong**: ``cat /tmp/sentinel`` succeeds and
    returns the file content — sandbox read isolation is broken.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sentinel = f"LEAK_{os.getpid()}"
    fd, sentinel_path = tempfile.mkstemp(
        prefix="ap_sandbox_unit_",
        dir="/tmp",
    )
    try:
        os.write(fd, sentinel.encode())
        os.close(fd)

        tool = CodeSandboxTool(
            srt_available=True,
            sandbox_enabled=True,
        )
        ctx = ToolContext(
            task_id="test_task",
            agent_id="test_agent",
            workspace=workspace,
        )
        # Use absolute path to bypass PATH resolution — tests
        # the filesystem read restriction directly.
        result = tool.invoke(
            json.dumps({"command": f"/bin/cat {sentinel_path}"}),
            ctx,
        )

        assert sentinel not in result, (
            f"SECURITY: sandbox allowed reading {sentinel_path} "
            f"outside workspace. Output: {result[:200]}. "
            f"srt denyRead is not blocking reads."
        )
    finally:
        try:
            os.unlink(sentinel_path)
        except OSError:
            pass


@pytest.mark.skipif(
    shutil.which("srt") is None,
    reason="srt not on PATH",
)
def test_sandbox_allows_read_inside_workspace(
    tmp_path: Path,
) -> None:
    """
    With srt enabled, reading a file inside the workspace must
    succeed. Ensures the allowRead carve-out works.

    **What breaks if wrong**: The deny rules are too broad and
    also block reads within the workspace.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sentinel = f"INSIDE_{os.getpid()}"
    inner_file = workspace / "test_file.txt"
    inner_file.write_text(sentinel)

    tool = CodeSandboxTool(
        srt_available=True,
        sandbox_enabled=True,
    )
    ctx = ToolContext(
        task_id="test_task",
        agent_id="test_agent",
        workspace=workspace,
    )
    result = tool.invoke(
        json.dumps({"command": f"cat {inner_file}"}),
        ctx,
    )

    assert sentinel in result, (
        f"Workspace file should be readable but wasn't. "
        f"Output: {result[:200]}. The allowRead carve-out "
        f"may be missing or wrong."
    )


@pytest.mark.skipif(
    shutil.which("srt") is None,
    reason="srt not on PATH",
)
def test_sandbox_shell_commands_resolve_via_path(
    tmp_path: Path,
) -> None:
    """
    Shell commands like ``ls`` must resolve via PATH inside the
    sandbox. The system allowRead dirs must cover PATH entries.

    **What breaks if wrong**: Every unqualified command fails
    with "command not found."
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tool = CodeSandboxTool(
        srt_available=True,
        sandbox_enabled=True,
    )
    ctx = ToolContext(
        task_id="test_task",
        agent_id="test_agent",
        workspace=workspace,
    )
    # ls is an external binary that requires PATH resolution.
    result = tool.invoke(
        json.dumps({"command": "ls -d ."}),
        ctx,
    )

    assert "." in result, (
        f"'ls' should resolve via PATH but failed. "
        f"Output: {result[:200]}. System allowRead paths "
        f"may be missing."
    )
