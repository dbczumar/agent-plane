"""Happy-path tests for the ``Shell`` primitive.

These tests exercise a real ``bash`` subprocess spawned via pexpect.
No mocks — the whole point of slice 1 is to prove the OSC 633
integration actually works end-to-end.

Scope matches slice 1 of the implementation phasing: spawn, simple
command, state persistence (cwd / env / sourced scripts), exit codes,
close. Slice 2 will add tests for timeouts, size caps, ANSI, crashes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_plane.terminals import Shell

# The ``shell`` fixture lives in tests/terminals/conftest.py — shared
# with test_shell_robustness.py.


def test_spawn_exposes_name(tmp_path: Path) -> None:
    """``Shell.name`` round-trips the name passed to ``spawn``."""
    s = Shell.spawn("myshell", tmp_path, sandbox_enabled=False)
    try:
        assert s.name == "myshell"
    finally:
        s.close()


def test_echo_hello(shell: Shell) -> None:
    """A trivial ``echo`` command captures output and exit 0.

    Failure of this test indicates the OSC 633 D marker parsing is
    broken at the most basic level — all other tests depend on it.
    """
    result = shell.run_sync("echo hello")
    assert "hello" in result.stdout
    assert result.exit_code == 0
    assert result.status == "completed"
    assert result.shell == "default"


def test_nonzero_exit_code(shell: Shell) -> None:
    """The ``false`` builtin returns exit code 1.

    Verifies the D marker encodes exit codes correctly, not just 0.
    """
    result = shell.run_sync("false")
    assert result.exit_code == 1


def test_cwd_persists_across_commands(shell: Shell, tmp_path: Path) -> None:
    """``cd`` in one call persists to the next.

    This is THE defining property of a persistent shell versus
    ``code_sandbox``'s stateless ``Popen``. If this fails, the whole
    design is moot — shells aren't actually persisting state.
    """
    # Initial cwd is the workspace
    r1 = shell.run_sync("pwd")
    # pexpect may produce \r\n line endings on the PTY; match loosely.
    assert str(tmp_path) in r1.stdout

    # cd /tmp (/tmp always exists)
    shell.run_sync("cd /tmp")

    # pwd should now report /tmp
    r2 = shell.run_sync("pwd")
    assert "/tmp" in r2.stdout


def test_env_persists_across_commands(shell: Shell) -> None:
    """``export`` persists across commands.

    Same as cwd persistence but for environment variables. Another
    defining property of persistence.
    """
    shell.run_sync("export AP_TEST_VAR=banana")
    result = shell.run_sync("echo $AP_TEST_VAR")
    assert "banana" in result.stdout
    assert result.exit_code == 0


def test_sourced_function_persists(shell: Shell) -> None:
    """A function defined via ``source`` is callable in the next command.

    Exercises the same persistence axis as env vars but for shell
    functions — which is what ``source venv/bin/activate`` and similar
    real-world patterns actually produce.
    """
    # Define a function inline (standing in for `source <file>` — same
    # mechanism; no need for a real sourced file in slice 1 tests).
    shell.run_sync("greet() { echo greetings, $1; }")
    result = shell.run_sync("greet world")
    assert "greetings, world" in result.stdout


def test_workspace_relative_env_set(shell: Shell, tmp_path: Path) -> None:
    """Workspace-relative env vars are set at shell launch.

    Verifies that `pip install foo` in the agent's first command would
    target the workspace — these env vars are part of the shell launch
    contract (§6.8), not applied per-command like ``code_sandbox``.
    """
    result = shell.run_sync("echo $PIP_TARGET")
    assert f"{tmp_path}/.pip" in result.stdout


def test_multiple_simple_commands_one_call(shell: Shell) -> None:
    """Semicolon-chained simple commands all run, one D marker at the end.

    Exit code is ``$?`` of the LAST simple command, per bash standard.
    """
    result = shell.run_sync("echo a; echo b; echo c")
    assert "a" in result.stdout
    assert "b" in result.stdout
    assert "c" in result.stdout
    assert result.exit_code == 0


def test_exit_code_is_last_command(shell: Shell) -> None:
    """When chained with ``;``, exit code reflects the last command."""
    result = shell.run_sync("true; false")
    assert result.exit_code == 1

    result = shell.run_sync("false; true")
    assert result.exit_code == 0


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Calling ``close`` twice doesn't raise."""
    s = Shell.spawn("default", tmp_path, sandbox_enabled=False)
    s.close()
    s.close()  # should not raise


def test_spawn_fails_loud_when_bash_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing bash surfaces as a clean RuntimeError, not a pexpect explosion.

    Emulates the "bash not installed" scenario by making ``shutil.which``
    return ``None``. Important because §6.8's launch-failure contract
    says spawn failures should be loud and clear.
    """
    monkeypatch.setattr("agent_plane.terminals.shell.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="bash is not installed"):
        Shell.spawn("default", tmp_path, sandbox_enabled=False)


def test_spawn_fails_loud_when_snippet_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vanished snippet file produces a clear error, not an opaque timeout.

    The snippet ships as a data file alongside shell.py. If a packaging
    bug or accidental delete removes it, we should say so explicitly
    rather than letting bash launch against a nonexistent --rcfile and
    then timing out waiting for an initial marker that will never come.
    """
    monkeypatch.setattr(
        "agent_plane.terminals.shell._INTEGRATION_SNIPPET",
        tmp_path / "does-not-exist.sh",
    )
    with pytest.raises(RuntimeError, match="integration snippet missing"):
        Shell.spawn("default", tmp_path, sandbox_enabled=False)


# ── sandbox-on tests ─────────────────────────────────────────────
#
# These are the only tests in the unit-test suite that actually
# invoke srt + node. They verify the §6.5 filesystem-isolation
# promise: the sandbox allows reads+writes in the workspace, blocks
# reads of the user's home, and preserves state across commands in
# the same shell. If srt is not available in the test environment,
# these skip loudly rather than silently passing.


def _srt_available() -> bool:
    """Whether srt + node are both on PATH in this test environment.

    Kept as a local helper (not imported from ``agent_plane.terminals
    ._sandbox``) so these tests stay self-contained and the import
    path to the sandbox helpers can change without updating them.

    :returns: True iff both ``srt`` and ``node`` can be located.
    """
    import shutil as _shutil

    return _shutil.which("srt") is not None and _shutil.which("node") is not None


@pytest.mark.skipif(
    not _srt_available(),
    reason="srt or node not available; skipping sandbox behavior test",
)
def test_sandboxed_shell_allows_workspace_io(tmp_path: Path) -> None:
    """Inside the sandbox, workspace reads/writes work normally.

    Sanity check before any "blocks" assertions: confirm the sandbox
    isn't just breaking everything. If this fails, subsequent deny
    tests are invalid (they'd pass for the wrong reason).
    """
    s = Shell.spawn("default", tmp_path, sandbox_enabled=True)
    try:
        r = s.run_sync("echo hello > greet.txt && cat greet.txt")
        # echo + cat both ran inside the sandbox; the file survived
        # the round-trip. If writes were blocked, exit_code would be
        # nonzero and stdout would be empty.
        assert r.exit_code == 0, (
            f"Sandbox blocked workspace write/read (exit {r.exit_code}). stdout: {r.stdout!r}"
        )
        assert "hello" in r.stdout
    finally:
        s.close()


@pytest.mark.skipif(
    not _srt_available(),
    reason="srt or node not available; skipping sandbox behavior test",
)
def test_sandboxed_shell_blocks_home_reads(tmp_path: Path) -> None:
    """Reads of the user's home directory are denied inside the sandbox.

    The design's core sandbox promise (§6.5 + ``deny_read_paths``):
    the bash subprocess cannot read ``$HOME`` — this is what keeps a
    rogue agent from exfiltrating ``~/.aws/credentials`` etc. The
    test probes ``~/.bashrc`` which exists on most Linuxes; any
    nonzero exit proves the block held.
    """
    s = Shell.spawn("default", tmp_path, sandbox_enabled=True)
    try:
        r = s.run_sync("cat ~/.bashrc 2>&1 || echo BLOCKED")
        # The sandbox must deny the read. Either cat fails (non-zero)
        # and the `|| echo BLOCKED` fires, or the filesystem layer
        # transparently returns no content. Either way, the file's
        # real contents must not appear.
        assert "BLOCKED" in r.stdout or r.exit_code != 0, (
            f"Sandbox did NOT block home dir read — cat ~/.bashrc succeeded. stdout: {r.stdout!r}"
        )
    finally:
        s.close()


@pytest.mark.skipif(
    not _srt_available(),
    reason="srt or node not available; skipping sandbox behavior test",
)
def test_sandboxed_shell_preserves_state_across_commands(
    tmp_path: Path,
) -> None:
    """Persistence still works inside the sandbox.

    Sanity check that the sandbox wrapper doesn't accidentally make
    shells one-shot — ``cd`` in one call must survive to the next,
    same as without the sandbox. Regression guard against bugs where
    the Node wrapper forks a fresh bash per command.
    """
    (tmp_path / "sub").mkdir()
    s = Shell.spawn("default", tmp_path, sandbox_enabled=True)
    try:
        s.run_sync("cd sub")
        r = s.run_sync("pwd")
        assert "/sub" in r.stdout, (
            f"Sandboxed shell lost cwd across commands — pwd stdout: {r.stdout!r}"
        )
    finally:
        s.close()


def test_sandbox_enabled_without_srt_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sandbox_enabled=True`` with srt missing must raise immediately.

    Per Principle #3 (fail loud), when the operator has configured
    sandboxing but srt/node aren't on PATH, we refuse to launch
    rather than silently running unsandboxed. Emulates the "srt
    uninstalled" scenario by patching the is_srt_available helper.
    """
    monkeypatch.setattr(
        "agent_plane.terminals.shell.is_srt_available",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="sandbox_enabled=True but srt"):
        Shell.spawn("default", tmp_path, sandbox_enabled=True)
