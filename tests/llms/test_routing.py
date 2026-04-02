"""Tests for llms.routing — model string parsing."""

import pytest

from agent_plane.errors import AgentPlaneError
from agent_plane.llms.routing import RoutedModel, parse_model_string


@pytest.mark.parametrize(
    ("model_string", "expected"),
    [
        (
            "anthropic/claude-sonnet-4-20250514",
            RoutedModel(provider="anthropic", model="claude-sonnet-4-20250514"),
        ),
        (
            "openai/gpt-5.4",
            RoutedModel(provider="openai", model="gpt-5.4"),
        ),
        (
            "groq/llama-3.1-70b",
            RoutedModel(provider="groq", model="llama-3.1-70b"),
        ),
        (
            "deepseek/deepseek-chat",
            RoutedModel(provider="deepseek", model="deepseek-chat"),
        ),
        (
            "xai/grok-2",
            RoutedModel(provider="xai", model="grok-2"),
        ),
        (
            "openrouter/meta-llama/llama-3.1-70b",
            RoutedModel(
                provider="openrouter",
                model="meta-llama/llama-3.1-70b",
            ),
        ),
        (
            "ollama/llama3",
            RoutedModel(provider="ollama", model="llama3"),
        ),
        (
            "gemini/gemini-2.5-pro",
            RoutedModel(provider="gemini", model="gemini-2.5-pro"),
        ),
        (
            "bedrock/anthropic.claude-3-sonnet",
            RoutedModel(provider="bedrock", model="anthropic.claude-3-sonnet"),
        ),
        (
            "vertex/gemini-2.5-pro",
            RoutedModel(provider="vertex", model="gemini-2.5-pro"),
        ),
        (
            "databricks/my-endpoint",
            RoutedModel(provider="databricks", model="my-endpoint"),
        ),
    ],
)
def test_parse_with_provider_prefix(
    model_string: str,
    expected: RoutedModel,
) -> None:
    assert parse_model_string(model_string) == expected


def test_parse_without_prefix_defaults_to_openai() -> None:
    result = parse_model_string("gpt-5.4")
    assert result == RoutedModel(provider="openai", model="gpt-5.4")


def test_unknown_provider_raises() -> None:
    with pytest.raises(AgentPlaneError, match="Unknown provider 'foobar'"):
        parse_model_string("foobar/some-model")
