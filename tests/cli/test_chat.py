"""Tests for agent_plane.chat — ap chat CLI logic."""

from __future__ import annotations

from pathlib import Path

from agent_plane.chat import _extract_agent_name, _is_url

# ── _is_url ──────────────────────────────────────────


def test_is_url_http() -> None:
    """HTTP URLs are detected."""
    assert _is_url("http://localhost:8000") is True


def test_is_url_https() -> None:
    """HTTPS URLs are detected."""
    assert _is_url("https://my-server.example.com") is True


def test_is_url_path() -> None:
    """Filesystem paths are not URLs."""
    assert _is_url("./my-agent/") is False


def test_is_url_relative() -> None:
    """Relative paths are not URLs."""
    assert _is_url("examples/agents/archer") is False


def test_is_url_absolute() -> None:
    """Absolute paths are not URLs."""
    assert _is_url("/home/user/my-agent") is False


# ── _extract_agent_name ──────────────────────────────


def test_extract_name_from_config(tmp_path: Path) -> None:
    """Reads agent name from config.yaml."""
    agent_dir = tmp_path / "test-agent"
    agent_dir.mkdir()
    (agent_dir / "config.yaml").write_text("spec_version: 1\nname: my-cool-agent\n")
    assert _extract_agent_name(agent_dir) == "my-cool-agent"


def test_extract_name_falls_back_to_dirname(tmp_path: Path) -> None:
    """Falls back to directory name when config has no name."""
    agent_dir = tmp_path / "fallback-agent"
    agent_dir.mkdir()
    (agent_dir / "config.yaml").write_text("spec_version: 1\n")
    assert _extract_agent_name(agent_dir) == "fallback-agent"


def test_extract_name_no_config(tmp_path: Path) -> None:
    """Falls back to directory name when no config.yaml exists."""
    agent_dir = tmp_path / "no-config"
    agent_dir.mkdir()
    assert _extract_agent_name(agent_dir) == "no-config"
