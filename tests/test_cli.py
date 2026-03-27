"""Tests for agent_plane.cli — bundle env var resolution."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_plane.cli import (
    _bundle,
    _expand_config_env_vars,
    _resolve_bundle_env_vars,
)
from agent_plane.errors import AgentPlaneError


def _write_config(
    agent_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Write a config.yaml to the agent directory.

    :param agent_dir: The agent image directory.
    :param config: The config dict to serialize.
    """
    (agent_dir / "config.yaml").write_text(
        yaml.dump(config, default_flow_style=False),
    )


def _write_mcp_config(
    agent_dir: Path,
    name: str,
    config: dict[str, Any],
) -> None:
    """
    Write an MCP server YAML file under tools/mcp/.

    :param agent_dir: The agent image directory.
    :param name: The MCP config filename (without .yaml).
    :param config: The MCP config dict to serialize.
    """
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    (mcp_dir / f"{name}.yaml").write_text(
        yaml.dump(config, default_flow_style=False),
    )


# ── _expand_config_env_vars ──────────────────────────


def test_expand_config_expands_llm_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_expand_config_env_vars`` resolves ``${VAR}`` in
    ``llm.connection`` values.
    """
    monkeypatch.setenv("TEST_API_KEY", "sk-resolved-123")
    from agent_plane.spec import expand_env_vars

    raw: dict[str, Any] = {
        "spec_version": 1,
        "llm": {
            "model": "gpt-5.4",
            "connection": {"api_key": "${TEST_API_KEY}"},
        },
    }
    changed = _expand_config_env_vars(raw, expand_env_vars)

    assert changed is True
    # The resolved value should replace the ${VAR} reference.
    assert raw["llm"]["connection"]["api_key"] == "sk-resolved-123", (
        "llm.connection.api_key should be expanded from env var"
    )


def test_expand_config_expands_builtin_tool_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_expand_config_env_vars`` resolves ``${VAR}`` in
    ``tools.builtins`` dict-entry config fields.
    """
    monkeypatch.setenv("PPLX_KEY", "pplx-resolved")
    from agent_plane.spec import expand_env_vars

    raw: dict[str, Any] = {
        "spec_version": 1,
        "tools": {
            "builtins": [
                "web_search_openai",
                {"name": "web_search_perplexity", "api_key": "${PPLX_KEY}"},
            ],
        },
    }
    changed = _expand_config_env_vars(raw, expand_env_vars)

    assert changed is True
    # String entries are untouched.
    assert raw["tools"]["builtins"][0] == "web_search_openai"
    # Dict entry api_key should be expanded.
    entry = raw["tools"]["builtins"][1]
    assert entry["api_key"] == "pplx-resolved", (
        "builtin tool api_key should be expanded from env var"
    )
    # 'name' is preserved.
    assert entry["name"] == "web_search_perplexity"


def test_expand_config_no_env_vars_returns_false() -> None:
    """
    ``_expand_config_env_vars`` returns ``False`` when the
    config has no fields that need expansion.
    """
    from agent_plane.spec import expand_env_vars

    raw: dict[str, Any] = {
        "spec_version": 1,
        "name": "simple-agent",
    }
    changed = _expand_config_env_vars(raw, expand_env_vars)
    assert changed is False


def test_expand_config_unresolved_var_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_expand_config_env_vars`` raises ``AgentPlaneError``
    when a ``${VAR}`` reference cannot be resolved.
    """
    monkeypatch.delenv("MISSING_KEY_12345", raising=False)
    from agent_plane.spec import expand_env_vars

    raw: dict[str, Any] = {
        "llm": {
            "model": "gpt-5.4",
            "connection": {"api_key": "${MISSING_KEY_12345}"},
        },
    }
    with pytest.raises(AgentPlaneError, match="MISSING_KEY_12345"):
        _expand_config_env_vars(raw, expand_env_vars)


# ── _resolve_bundle_env_vars ─────────────────────────


def test_resolve_bundle_expands_config_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_resolve_bundle_env_vars`` returns resolved
    ``config.yaml`` content with expanded env vars.
    """
    monkeypatch.setenv("BUNDLE_TEST_KEY", "resolved-value")
    _write_config(
        tmp_path,
        {
            "spec_version": 1,
            "llm": {
                "model": "gpt-5.4",
                "connection": {"api_key": "${BUNDLE_TEST_KEY}"},
            },
        },
    )

    resolved = _resolve_bundle_env_vars(tmp_path)

    assert "config.yaml" in resolved
    # Parse the resolved YAML and verify the value.
    parsed = yaml.safe_load(resolved["config.yaml"])
    assert parsed["llm"]["connection"]["api_key"] == "resolved-value"


def test_resolve_bundle_expands_mcp_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_resolve_bundle_env_vars`` returns resolved MCP config
    YAML with expanded header env vars.
    """
    monkeypatch.setenv("MCP_TOKEN", "tok-abc")
    _write_mcp_config(
        tmp_path,
        "github",
        {
            "name": "github",
            "transport": "http",
            "url": "http://localhost:9000/mcp",
            "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
        },
    )
    # config.yaml must exist (even if empty) for a valid agent dir.
    _write_config(tmp_path, {"spec_version": 1})

    resolved = _resolve_bundle_env_vars(tmp_path)

    arcname = "tools/mcp/github.yaml"
    assert arcname in resolved
    parsed = yaml.safe_load(resolved[arcname])
    assert parsed["headers"]["Authorization"] == "Bearer tok-abc"


def test_resolve_bundle_no_env_vars_returns_empty(
    tmp_path: Path,
) -> None:
    """
    ``_resolve_bundle_env_vars`` returns an empty dict when
    the config has no env var references.
    """
    _write_config(
        tmp_path,
        {
            "spec_version": 1,
            "name": "plain-agent",
        },
    )

    resolved = _resolve_bundle_env_vars(tmp_path)
    assert resolved == {}


def test_resolve_bundle_missing_env_var_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_resolve_bundle_env_vars`` raises ``AgentPlaneError``
    when a config.yaml env var cannot be resolved.
    """
    monkeypatch.delenv("NONEXISTENT_DEPLOY_KEY", raising=False)
    _write_config(
        tmp_path,
        {
            "spec_version": 1,
            "tools": {
                "builtins": [
                    {
                        "name": "web_search_google",
                        "api_key": "${NONEXISTENT_DEPLOY_KEY}",
                    },
                ],
            },
        },
    )

    with pytest.raises(AgentPlaneError, match="NONEXISTENT_DEPLOY_KEY"):
        _resolve_bundle_env_vars(tmp_path)


# ── _bundle integration tests ──────────────────────────


def _extract_yaml_from_bundle(
    bundle_bytes: bytes,
    arcname: str,
) -> dict[str, Any]:
    """
    Extract and parse a YAML file from a tar.gz bundle.

    :param bundle_bytes: The gzipped tarball bytes.
    :param arcname: The archive member name, e.g.
        ``"config.yaml"`` or ``"tools/mcp/github.yaml"``.
    :returns: The parsed YAML content as a dict.
    """
    with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz") as tf:
        member = tf.getmember(arcname)
        extracted = tf.extractfile(member)
        assert extracted is not None, f"Expected {arcname!r} to be a regular file in the bundle"
        return yaml.safe_load(extracted.read())


def test_bundle_resolves_config_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_bundle`` produces a tarball where ``config.yaml`` has
    ``${VAR}`` references replaced with resolved values.

    Verifies the end-to-end path: write agent dir with env var
    refs → call ``_bundle`` → extract tarball → assert resolved.
    """
    monkeypatch.setenv("BUNDLE_LLM_KEY", "sk-live-abc123")
    monkeypatch.setenv("BUNDLE_PPLX_KEY", "pplx-live-xyz")
    _write_config(
        tmp_path,
        {
            "spec_version": 1,
            "name": "env-test-agent",
            "llm": {
                "model": "openai/gpt-4o",
                "connection": {"api_key": "${BUNDLE_LLM_KEY}"},
            },
            "tools": {
                "builtins": [
                    "web_search_openai",
                    {
                        "name": "web_search_perplexity",
                        "api_key": "${BUNDLE_PPLX_KEY}",
                    },
                ],
            },
        },
    )

    bundle_bytes = _bundle(tmp_path)
    parsed = _extract_yaml_from_bundle(bundle_bytes, "config.yaml")

    # LLM connection key must be resolved — if still "${BUNDLE_LLM_KEY}",
    # the server would receive an unresolved reference it can't expand.
    assert parsed["llm"]["connection"]["api_key"] == "sk-live-abc123", (
        "LLM api_key should be resolved in the bundle tarball"
    )
    # Builtin tool config key must be resolved.
    perplexity_entry = parsed["tools"]["builtins"][1]
    assert perplexity_entry["api_key"] == "pplx-live-xyz", (
        "Builtin tool api_key should be resolved in the bundle tarball"
    )
    assert perplexity_entry["name"] == "web_search_perplexity", (
        "Builtin tool name must be preserved after expansion"
    )
    # String entries pass through unchanged.
    assert parsed["tools"]["builtins"][0] == "web_search_openai", (
        "String builtin entries should be unchanged in the bundle"
    )
    # Non-secret fields survive bundling.
    assert parsed["name"] == "env-test-agent"
    assert parsed["llm"]["model"] == "openai/gpt-4o"


def test_bundle_resolves_mcp_header_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_bundle`` produces a tarball where MCP server YAML files
    have ``${VAR}`` references in headers replaced with resolved
    values.
    """
    monkeypatch.setenv("BUNDLE_GH_TOKEN", "ghp-secret-tok")
    _write_config(tmp_path, {"spec_version": 1, "name": "mcp-agent"})
    _write_mcp_config(
        tmp_path,
        "github",
        {
            "name": "github",
            "transport": "http",
            "url": "http://localhost:9000/mcp",
            "headers": {"Authorization": "Bearer ${BUNDLE_GH_TOKEN}"},
        },
    )

    bundle_bytes = _bundle(tmp_path)
    parsed = _extract_yaml_from_bundle(bundle_bytes, "tools/mcp/github.yaml")

    # Header must be resolved — an unresolved "${BUNDLE_GH_TOKEN}"
    # would cause MCP auth failures on the server.
    assert parsed["headers"]["Authorization"] == "Bearer ghp-secret-tok", (
        "MCP header env var should be resolved in the bundle tarball"
    )
    # Non-header fields survive bundling.
    assert parsed["name"] == "github"
    assert parsed["url"] == "http://localhost:9000/mcp"


def test_bundle_no_env_vars_preserves_files(
    tmp_path: Path,
) -> None:
    """
    ``_bundle`` produces a valid tarball even when no env vars
    need expansion — files are included as-is.
    """
    _write_config(
        tmp_path,
        {
            "spec_version": 1,
            "name": "plain-agent",
            "llm": {"model": "openai/gpt-4o"},
        },
    )

    bundle_bytes = _bundle(tmp_path)
    parsed = _extract_yaml_from_bundle(bundle_bytes, "config.yaml")

    # Config content should be preserved exactly.
    assert parsed["name"] == "plain-agent"
    assert parsed["llm"]["model"] == "openai/gpt-4o"


def test_bundle_passthrough_existing_tarball(
    tmp_path: Path,
) -> None:
    """
    ``_bundle`` returns the raw bytes of an existing ``.tar.gz``
    file without modification (env var expansion only applies to
    directories).
    """
    # Build a tarball with an unresolved env var reference.
    config_bytes = yaml.dump(
        {
            "spec_version": 1,
            "llm": {"connection": {"api_key": "${SHOULD_NOT_EXPAND}"}},
        }
    ).encode()
    tarball_path = tmp_path / "agent.tar.gz"
    with tarfile.open(tarball_path, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))

    bundle_bytes = _bundle(tarball_path)

    # Passthrough: bytes must match the original file exactly.
    assert bundle_bytes == tarball_path.read_bytes(), (
        "Existing tarball should be returned as-is without expansion"
    )


def test_bundle_missing_env_var_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_bundle`` raises ``AgentPlaneError`` when the agent
    directory contains an unresolvable ``${VAR}`` reference.
    """
    monkeypatch.delenv("NONEXISTENT_BUNDLE_KEY", raising=False)
    _write_config(
        tmp_path,
        {
            "spec_version": 1,
            "llm": {
                "model": "openai/gpt-4o",
                "connection": {"api_key": "${NONEXISTENT_BUNDLE_KEY}"},
            },
        },
    )

    with pytest.raises(AgentPlaneError, match="NONEXISTENT_BUNDLE_KEY"):
        _bundle(tmp_path)
