"""Tests for agent_plane.onboarding.provider_selection — selection logic."""

from __future__ import annotations

import pytest
from click import ClickException

from agent_plane.onboarding.provider_selection import (
    ProviderSelection,
    resolve_provider_from_model,
)

# ── resolve_provider_from_model ────────────────────────


def test_resolve_parses_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid litellm format should parse into provider + full model string."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    selection = resolve_provider_from_model("anthropic/claude-sonnet-4-20250514")
    assert selection.provider == "anthropic"
    assert selection.model == "anthropic/claude-sonnet-4-20250514"
    assert isinstance(selection, ProviderSelection)


def test_resolve_reads_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials should be read from the provider's env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    selection = resolve_provider_from_model("openai/gpt-5.4")
    assert selection.credentials["api_key"] == "sk-openai-test"


def test_resolve_missing_env_var_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing env var should raise a clear error."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ClickException, match="ANTHROPIC_API_KEY"):
        resolve_provider_from_model("anthropic/claude-sonnet-4-20250514")


def test_resolve_rejects_model_without_slash() -> None:
    """Model string without provider/ prefix should raise."""
    with pytest.raises(ClickException, match="provider/model_name"):
        resolve_provider_from_model("gpt-5.4")


def test_resolve_handles_nested_slashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model strings with multiple slashes should split on the first only."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    selection = resolve_provider_from_model("deepseek/deepseek-chat/v2")
    assert selection.provider == "deepseek"
    # Full model string preserved.
    assert selection.model == "deepseek/deepseek-chat/v2"
