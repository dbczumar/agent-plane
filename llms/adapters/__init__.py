"""
Adapter registry — maps provider names to adapter instances.
"""

from __future__ import annotations

from typing import Any

from llms.adapters.base import BaseAdapter

# Lazy-initialized adapter cache. Each provider gets at most one
# adapter instance per process.
_adapter_cache: dict[str, BaseAdapter] = {}


def get_adapter(provider: str, **kwargs: Any) -> BaseAdapter:
    """
    Return an adapter instance for the given provider.

    Adapters are cached — the first call creates the instance and
    subsequent calls return the same one.

    :param provider: The provider identifier, e.g. ``"anthropic"``.
    :param kwargs: Extra keyword arguments forwarded to the adapter
        constructor (used by tests to override config).
    :returns: A :class:`BaseAdapter` subclass instance.
    :raises ValueError: If the provider is not supported.
    """
    if provider in _adapter_cache and not kwargs:
        return _adapter_cache[provider]

    adapter = _create_adapter(provider, **kwargs)
    if not kwargs:
        _adapter_cache[provider] = adapter
    return adapter


def _create_adapter(provider: str, **kwargs: Any) -> BaseAdapter:
    """
    Instantiate the correct adapter for the provider.

    Imports are lazy to avoid pulling in optional dependencies
    (boto3, google-auth) when they're not needed.

    :param provider: The provider identifier.
    :param kwargs: Extra kwargs for the adapter constructor.
    :returns: A :class:`BaseAdapter` instance.
    """
    # OpenAI-compatible providers
    openai_compat_providers = {
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "ollama": ("http://localhost:11434/v1", None),
    }

    if provider in openai_compat_providers:
        from llms.adapters.openai import OpenAICompatibleAdapter

        base_url, api_key_env = openai_compat_providers[provider]
        return OpenAICompatibleAdapter(
            base_url=kwargs.get("base_url", base_url),
            api_key_env=kwargs.get("api_key_env", api_key_env),
        )

    if provider == "anthropic":
        from llms.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter(**kwargs)

    if provider == "gemini":
        from llms.adapters.gemini import GeminiAdapter

        return GeminiAdapter(**kwargs)

    if provider == "bedrock":
        from llms.adapters.bedrock import BedrockAdapter

        return BedrockAdapter(**kwargs)

    if provider == "vertex":
        from llms.adapters.vertex import VertexAdapter

        return VertexAdapter(**kwargs)

    if provider == "databricks":
        from llms.adapters.databricks import DatabricksAdapter

        return DatabricksAdapter(**kwargs)

    all_providers = sorted(
        openai_compat_providers.keys() | {"anthropic", "gemini", "bedrock", "vertex", "databricks"}
    )
    raise ValueError(f"Unknown provider {provider!r}. Supported: {all_providers}")


def clear_cache() -> None:
    """
    Clear the adapter cache. Useful for tests.
    """
    _adapter_cache.clear()
