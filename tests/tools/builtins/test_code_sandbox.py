"""Unit tests for the code_sandbox built-in tool.

Tests the srt config generation, filesystem isolation, and
unrestricted network access without needing a real LLM or server.
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
    build_srt_config,
    deny_read_paths,
    system_read_allowlist,
)

# ── Config generation tests (no srt needed) ────────────────


def test_srt_config_deny_read_is_non_empty(tmp_path: Path) -> None:
    """
    ``denyRead`` must not be empty — that was the original bug
    that allowed agents to read any file on the host.

    **What breaks if wrong**: Agent can ``cat /tmp/secrets``,
    ``ls ~/Documents``, etc.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = json.loads(build_srt_config(workspace))

    assert len(config["filesystem"]["denyRead"]) > 0, (
        "denyRead must not be empty — that was the original bug"
    )


def test_srt_config_deny_read_covers_tmp(tmp_path: Path) -> None:
    """
    ``/tmp`` must be denied on all platforms. This was the
    user's original bug report.

    **What breaks if wrong**: Agent reads /tmp files containing
    secrets, API keys, or other user data.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = json.loads(build_srt_config(workspace))
    deny = config["filesystem"]["denyRead"]

    # /tmp must be blocked, either directly or via denying "/".
    tmp_blocked = "/" in deny or "/tmp" in deny or "/private/tmp" in deny
    assert tmp_blocked, f"denyRead must block /tmp (directly or via '/'), got: {deny}"


def test_deny_read_paths_covers_user_home() -> None:
    """
    ``deny_read_paths`` must cover either ``/`` (Linux) or
    the user home root (macOS).

    **What breaks if wrong**: No read restrictions at all.
    """
    paths = deny_read_paths()
    assert len(paths) > 0
    covers_root = "/" in paths
    covers_users = "/Users" in paths or "/home" in paths
    assert covers_root or covers_users, f"Must deny either '/' or user-data roots, got: {paths}"


def test_srt_config_workspace_in_allow_read(tmp_path: Path) -> None:
    """
    The workspace must appear in ``allowRead`` so the agent
    can read its own files within the denied region.

    **What breaks if wrong**: Agent can't read files it created
    in its own workspace.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = json.loads(build_srt_config(workspace))
    resolved_ws = str(workspace.resolve())

    assert resolved_ws in config["filesystem"]["allowRead"], (
        f"allowRead must include resolved workspace '{resolved_ws}'"
    )


def test_srt_config_restrict_writes_to_workspace(
    tmp_path: Path,
) -> None:
    """
    The srt config must only allow writes to the workspace.

    **What breaks if wrong**: An agent could write files anywhere
    on the host filesystem.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = json.loads(build_srt_config(workspace))
    resolved_ws = str(workspace.resolve())

    assert config["filesystem"]["allowWrite"] == [resolved_ws], (
        f"allowWrite must be exactly [resolved_workspace], "
        f"got: {config['filesystem']['allowWrite']}"
    )


def test_srt_config_no_network_key(tmp_path: Path) -> None:
    """
    The config must NOT include a ``network`` key. Network
    restriction is disabled by the wrapper script via
    ``updateConfig``.

    **What breaks if wrong**: The wrapper might re-apply network
    restrictions from the config.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = json.loads(build_srt_config(workspace))

    assert "network" not in config, f"Config must not include 'network' key, got: {config.keys()}"


def test_srt_config_resolve_symlinks(tmp_path: Path) -> None:
    """
    Workspace paths must be resolved to canonical form. On macOS,
    ``/var`` → ``/private/var`` — without resolving, the srt
    allow paths won't match the real filesystem.

    **What breaks if wrong**: Workspace reads/writes fail because
    the allow path doesn't match the resolved path srt checks.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = json.loads(build_srt_config(workspace))
    resolved = str(workspace.resolve())

    assert config["filesystem"]["allowWrite"] == [resolved]
    assert resolved in config["filesystem"]["allowRead"]


def test_system_read_allowlist_excludes_user_home() -> None:
    """
    The system read allowlist must not include the user's home
    directory tree. That's the primary attack surface.

    **What breaks if wrong**: Agent can read ``~/Documents``,
    ``~/.ssh``, etc.
    """
    allow = system_read_allowlist()
    home_root = str(Path.home().parent)

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

    assert "/usr" in allow or "/bin" in allow, (
        f"allowlist must include /usr or /bin for PATH resolution, got: {allow}"
    )


# ── srt integration tests (require srt on PATH) ───────────


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
    sandbox. System allowRead dirs must cover PATH entries.

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
    result = tool.invoke(
        json.dumps({"command": "ls -d ."}),
        ctx,
    )

    assert "." in result, (
        f"'ls' should resolve via PATH but failed. "
        f"Output: {result[:200]}. System allowRead paths "
        f"may be missing."
    )


@pytest.mark.skipif(
    shutil.which("srt") is None,
    reason="srt not on PATH",
)
def test_sandbox_allows_network_access(
    tmp_path: Path,
) -> None:
    """
    Network access must be unrestricted inside the sandbox.
    The wrapper script removes ``allowedDomains`` from the srt
    config so no network proxy is activated.

    **What breaks if wrong**: ``curl`` fails with connection
    errors because srt's network proxy is blocking traffic.
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
    result = tool.invoke(
        json.dumps(
            {
                "command": (
                    "curl -sk https://example.com "
                    "-o /dev/null -w '%{http_code}' "
                    "--connect-timeout 5"
                ),
            }
        ),
        ctx,
    )

    # HTTP 200 proves the request reached example.com and
    # got a response — network is not restricted.
    assert "200" in result, (
        f"Expected HTTP 200 from example.com proving network "
        f"access works. Got: {result[:200]}. The srt wrapper "
        f"may not be removing network restrictions."
    )
