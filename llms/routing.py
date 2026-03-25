"""
Provider routing — parse model strings and resolve adapters.

Model strings use ``"provider/model-name"`` format, e.g.
``"anthropic/claude-sonnet-4-20250514"``. If no provider prefix
is given, defaults to ``"openai"``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Known providers and their default configurations.
# Each entry maps provider name -> (base_url, api_key_env_var).
# api_key_env of None means no auth required (e.g. Ollama).
PROVIDER_CONFIGS: dict[str, tuple[str, str | None]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta",
        "GOOGLE_API_KEY",
    ),
    "bedrock": ("", None),
    "vertex": ("", None),
    "databricks": ("", "DATABRICKS_TOKEN"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
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
    :raises ValueError: If the provider prefix is not recognized.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
    else:
        provider = _DEFAULT_PROVIDER
        model_name = model

    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider {provider!r}. Known providers: {sorted(PROVIDER_CONFIGS)}"
        )

    return RoutedModel(provider=provider, model=model_name)
