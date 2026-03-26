"""
Provider routing — parse model strings and resolve adapters.

Model strings use ``"provider/model-name"`` format, e.g.
``"anthropic/claude-sonnet-4-20250514"``. If no provider prefix
is given, defaults to ``"openai"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_plane.errors import AgentPlaneError, ErrorCode

# Known providers and their default base URLs.
# API keys come from connection_params at call time (llm.connection config),
# not from environment variables. Providers that require connection_params
# for their base URL (Bedrock, Vertex, Databricks) have None here.
PROVIDER_CONFIGS: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "bedrock": None,
    "vertex": None,
    "databricks": None,
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

_DEFAULT_PROVIDER = "openai"


@dataclass
class RoutedModel:
    """
    A parsed model string split into provider and model name.

    :param provider: The provider identifier, e.g. ``"anthropic"``.
    :param model: The model name without prefix, e.g.
        ``"claude-sonnet-4-20250514"``.
    """

    provider: str
    model: str


def parse_model_string(model: str) -> RoutedModel:
    """
    Parse a ``"provider/model-name"`` string into its components.

    If no ``"/"`` is present, the provider defaults to ``"openai"``
    for backward compatibility.

    :param model: The model string, e.g.
        ``"anthropic/claude-sonnet-4-20250514"`` or ``"gpt-5.4"``.
    :returns: A :class:`RoutedModel` with ``provider`` and ``model``.
    :raises AgentPlaneError: If the provider prefix is not recognized.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
    else:
        provider = _DEFAULT_PROVIDER
        model_name = model

    if provider not in PROVIDER_CONFIGS:
        raise AgentPlaneError(
            f"Unknown provider {provider!r}. Known providers: {sorted(PROVIDER_CONFIGS)}",
            code=ErrorCode.INVALID_INPUT,
        )

    return RoutedModel(provider=provider, model=model_name)
