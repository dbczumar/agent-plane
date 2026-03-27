"""Tests for agent_plane.cli — bundle env var resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_plane.cli import _expand_config_env_vars, _resolve_bundle_env_vars
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
