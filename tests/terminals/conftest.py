"""Shared fixtures for the ``tests/terminals/`` suite.

Only fixtures used across multiple test files live here — per
CLAUDE.md's fixture-locality rule, anything used in a single file
stays in that file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_plane.terminals import Shell


@pytest.fixture
def shell(tmp_path: Path) -> Iterator[Shell]:
    """Spawn a default-named shell in a fresh workspace; kill on teardown.

    Sandbox is **disabled** for this shared fixture. Most unit tests
    assert behavior of the shell itself (OSC 633, ring buffer, timeout
    handling, ...) which doesn't depend on srt wrapping. Sandbox-
    specific behavior is covered in dedicated tests (see
    ``test_shell.py::test_sandbox_*``) that opt in explicitly.

    :param tmp_path: Pytest's per-test tmpdir, used as the shell's
        workspace. Disk-overflow logs (if any) land under
        ``tmp_path/.agent_plane/terminal/``.
    :yields: A ready-to-use :class:`Shell` named ``"default"``.
    """
    s = Shell.spawn("default", tmp_path, sandbox_enabled=False)
    try:
        yield s
    finally:
        s.close()
