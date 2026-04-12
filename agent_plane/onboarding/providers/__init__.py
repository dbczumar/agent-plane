"""
Provider catalog and model discovery for onboarding.

Copied and trimmed from ``mlflow/utils/providers.py`` and
``mlflow/utils/model_catalog/``. MLflow is **not** a dependency
of agent-plane — this is a standalone copy of the catalog data
and the minimal loading/query logic needed for provider selection
during ``ap create``.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    """
    Flat model metadata loaded from a catalog JSON file.

    :param name: The model identifier, e.g. ``"claude-sonnet-4-20250514"``.
    :param provider: The provider name, e.g. ``"anthropic"``.
    :param mode: The model mode, e.g. ``"chat"``, ``"embedding"``, or ``None``.
    :param supports_function_calling: Whether the model supports tool use.
    :param max_input_tokens: Maximum input context window size, or ``None``.
    :param max_output_tokens: Maximum output tokens, or ``None``.
    """

    name: str
    provider: str
    mode: str | None = None
    supports_function_calling: bool = False
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass
class AuthField:
    """
    A single credential field required by a provider's auth mode.

    :param name: Field identifier, e.g. ``"api_key"``.
    :param description: Human-readable label, e.g. ``"Anthropic API Key"``.
    :param secret: Whether the value should be masked in display.
    :param required: Whether the field is mandatory.
    """

    name: str
    description: str
    secret: bool
    required: bool


@dataclass
class AuthMode:
    """
    An authentication mode for a provider (e.g. API key, access keys, IAM role).

    :param mode_id: Short identifier, e.g. ``"api_key"``, ``"access_keys"``.
    :param display_name: Human-readable name, e.g. ``"API Key"``.
    :param description: Help text for the user.
    :param fields: Credential fields the user must supply.
    :param is_default: Whether this is the recommended default mode.
    """

    mode_id: str
    display_name: str
    description: str
    fields: list[AuthField]
    is_default: bool = False


@dataclass
class ProviderConfig:
    """
    Full auth configuration for a provider, with one or more auth modes.

    :param auth_modes: Available authentication modes.
    :param default_mode: The ``mode_id`` of the recommended default.
    """

    auth_modes: list[AuthMode]
    default_mode: str


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def _catalog_dir() -> Path:
    """
    Return the path to the bundled model_catalog directory.

    :returns: Absolute path to the ``model_catalog/`` package directory.
    """
    # importlib.resources.files returns Traversable; cast to Path
    # since we only use it with Path operations (glob, read_text).
    return Path(str(importlib.resources.files(__package__).joinpath("model_catalog")))


@functools.lru_cache(maxsize=1)
def _list_provider_names() -> list[str]:
    """
    Return provider names from the bundled catalog (directory listing).

    :returns: Sorted list of provider stem names, e.g. ``["anthropic", "openai", ...]``.
    """
    try:
        return sorted(p.stem for p in Path(_catalog_dir()).glob("*.json") if p.is_file())
    except (FileNotFoundError, TypeError):
        return []


@functools.lru_cache(maxsize=128)
def _load_provider_catalog(provider: str) -> dict[str, Any]:
    """
    Load a single provider's catalog JSON from bundled package resources.

    :param provider: Provider name, e.g. ``"anthropic"``.
    :returns: Parsed JSON dict with ``schema_version`` and ``models`` keys,
        or empty dict if the file is missing.
    """
    resource_path = _catalog_dir() / f"{provider}.json"
    try:
        result: dict[str, Any] = json.loads(resource_path.read_text("utf-8"))
        return result
    except (FileNotFoundError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Provider consolidation (e.g. vertex_ai-* → vertex_ai)
# ---------------------------------------------------------------------------

_EXCLUDED_PROVIDERS = {"bedrock_converse"}

_PROVIDER_CONSOLIDATION: dict[str, Callable[[str], bool]] = {
    "vertex_ai": lambda p: p == "vertex_ai" or p.startswith("vertex_ai-"),
}


def _normalize_provider(provider: str) -> str:
    """
    Normalize provider name by consolidating variants into a single provider.

    For example, ``vertex_ai-llama_models`` becomes ``vertex_ai``.

    :param provider: Raw provider name from the catalog.
    :returns: Normalized provider name.
    """
    for normalized, matcher in _PROVIDER_CONSOLIDATION.items():
        if matcher(provider):
            return normalized
    return provider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Popular providers shown first in selection UI, matching MLflow AI Gateway.
# Remaining providers follow in alphabetical order.
COMMON_PROVIDERS: list[str] = [
    "openai",
    "anthropic",
    "databricks",
    "bedrock",
    "gemini",
    "vertex_ai",
    "azure",
    "xai",
    "mistral",
    "groq",
    "deepseek",
    "openrouter",
    "ollama",
    "together_ai",
    "cohere",
    "fireworks_ai",
]


def get_all_providers() -> list[str]:
    """
    Return all available provider names from the bundled catalog.

    Popular providers (from :data:`COMMON_PROVIDERS`) are listed first,
    followed by the rest in alphabetical order. This matches the MLflow
    AI Gateway UI ordering so users see the most common choices at the
    top. Provider variants are consolidated (e.g. all ``vertex_ai-*``
    become ``vertex_ai``). Excluded providers (e.g. ``bedrock_converse``)
    are filtered out.

    :returns: Deduplicated list of provider names, popular first.
    """
    all_names: set[str] = set()
    for name in _list_provider_names():
        if name in _EXCLUDED_PROVIDERS:
            continue
        all_names.add(_normalize_provider(name))

    # Popular providers first (in COMMON_PROVIDERS order), then
    # remaining providers alphabetically.
    popular = [p for p in COMMON_PROVIDERS if p in all_names]
    rest = sorted(all_names - set(popular))
    return popular + rest


def get_models(provider: str) -> list[ModelInfo]:
    """
    Return all models for a provider, loaded from the catalog JSON files.

    For consolidated providers (e.g. ``vertex_ai``), models from all
    variant files are included.

    :param provider: Provider name, e.g. ``"anthropic"``.
    :returns: List of :class:`ModelInfo` for all models under that provider.
    """
    matching_files = [
        p
        for p in _list_provider_names()
        if _normalize_provider(p) == provider and p not in _EXCLUDED_PROVIDERS
    ]

    models: list[ModelInfo] = []
    seen: set[str] = set()

    for file_provider in matching_files:
        catalog = _load_provider_catalog(file_provider)
        for model_name, entry in catalog.get("models", {}).items():
            # Strip provider prefix if present (e.g. "gemini/gemini-2.5-flash")
            if model_name.startswith(f"{provider}/"):
                model_name = model_name.removeprefix(f"{provider}/")

            # Skip fine-tuned variants
            if model_name.startswith("ft:"):
                continue

            if model_name in seen:
                continue
            seen.add(model_name)

            context = entry.get("context_window", {})
            capabilities = entry.get("capabilities", {})

            models.append(
                ModelInfo(
                    name=model_name,
                    provider=provider,
                    mode=entry.get("mode"),
                    supports_function_calling=capabilities.get(
                        "function_calling",
                        False,
                    ),
                    max_input_tokens=context.get("max_input"),
                    max_output_tokens=context.get("max_output"),
                )
            )

    return models


def get_chat_models(provider: str) -> list[ModelInfo]:
    """
    Return only chat-capable models for a provider, newest first.

    Filters to ``mode="chat"`` and sorts by version number
    (descending), then release date (newest first), matching the
    MLflow AI Gateway UI ordering.

    :param provider: Provider name, e.g. ``"anthropic"``.
    :returns: Sorted list of chat-mode :class:`ModelInfo` instances.
    """
    chat = [m for m in get_models(provider) if m.mode == "chat"]
    return _sort_models_newest_first(chat)


# Substrings that indicate a model is audio, realtime, image,
# embedding, or otherwise not suitable as a text chat agent brain.
_NON_TEXT_CHAT_PATTERNS = (
    "audio",
    "realtime",
    "tts",
    "whisper",
    "dall-e",
    "embedding",
    "moderation",
    "transcription",
    "speech",
    "vision-preview",
    "container",
)


def get_onboarding_models(provider: str) -> list[ModelInfo]:
    """
    Return models suitable for powering the onboarding agent.

    Filters to chat-mode models that support function calling
    (the onboarding agent needs tools) and excludes audio,
    realtime, image, embedding, and other non-text-chat models.
    Results are sorted newest first.

    :param provider: Provider name, e.g. ``"openai"``.
    :returns: Sorted list of eligible :class:`ModelInfo` instances.
    """
    eligible = [
        m
        for m in get_models(provider)
        if m.mode == "chat" and m.supports_function_calling and not _is_non_text_chat(m.name)
    ]
    return _sort_models_newest_first(eligible)


def _is_non_text_chat(model_name: str) -> bool:
    """
    Check if a model name indicates a non-text-chat model.

    :param model_name: The model name to check.
    :returns: ``True`` if the model should be excluded from
        onboarding model selection.
    """
    lower = model_name.lower()
    return any(pattern in lower for pattern in _NON_TEXT_CHAT_PATTERNS)


# ---------------------------------------------------------------------------
# Model sorting — newest/best models first
# ---------------------------------------------------------------------------

# Matches version-like numbers in model names: gpt-4 → 4, claude-3.5 → 3.5,
# o1 → 1, gpt-4.1 → 4.1, llama-4 → 4
_VERSION_PATTERN = re.compile(
    r"(?:^|[-/])"  # start of string or separator
    r"(?:gpt-?|o|claude-?|llama-?|gemini-?|deepseek-?v?)?"
    r"(\d+(?:\.\d+)?)"  # version number (e.g. 4, 3.5, 4.1)
)

# Matches dates: 2025-04-14, 20250414, 20241022
_DATE_PATTERN = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})")


def _extract_model_version(name: str) -> float:
    """
    Extract the primary version number from a model name.

    :param name: Model name, e.g. ``"gpt-4.1-2025-04-14"``.
    :returns: Version as float, or ``0.0`` if none found.
    """
    match = _VERSION_PATTERN.search(name)
    if match:
        return float(match.group(1))
    return 0.0


def _extract_model_date(name: str) -> int:
    """
    Extract a date as an integer from a model name for sorting.

    :param name: Model name, e.g. ``"gpt-4-2024-08-06"``.
    :returns: Date as YYYYMMDD integer, or ``0`` if none found.
    """
    match = _DATE_PATTERN.search(name)
    if match:
        return int(match.group(1) + match.group(2) + match.group(3))
    return 0


def _sort_models_newest_first(models: list[ModelInfo]) -> list[ModelInfo]:
    """
    Sort models by version (descending), date (newest first), then name.

    Matches MLflow AI Gateway's ``sortModelsByDate()`` logic so that
    newer, more capable models appear at the top of the selection list.

    :param models: Unsorted model list.
    :returns: Sorted model list, newest/highest version first.
    """
    return sorted(
        models,
        key=lambda m: (
            -_extract_model_version(m.name),
            -_extract_model_date(m.name),
            m.name,
        ),
    )


# ---------------------------------------------------------------------------
# Auth mode definitions
# ---------------------------------------------------------------------------

# Providers with multiple auth modes. For simple API-key providers,
# a default mode is generated dynamically by get_provider_config().
_PROVIDER_AUTH_MODES: dict[str, dict[str, dict[str, Any]]] = {
    "bedrock": {
        "api_key": {
            "display_name": "API Key",
            "description": "Use Amazon Bedrock API Key (bearer token)",
            "default": True,
            "fields": [
                {
                    "name": "api_key",
                    "description": "Amazon Bedrock API Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region",
                    "secret": False,
                    "required": True,
                },
            ],
        },
        "access_keys": {
            "display_name": "Access Keys",
            "description": "Use AWS Access Key ID and Secret Access Key",
            "fields": [
                {
                    "name": "aws_access_key_id",
                    "description": "AWS Access Key ID",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "description": "AWS Secret Access Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region (e.g., us-east-1)",
                    "secret": False,
                    "required": False,
                },
            ],
        },
    },
    "azure": {
        "api_key": {
            "display_name": "API Key",
            "description": "Use Azure OpenAI API Key",
            "default": True,
            "fields": [
                {
                    "name": "api_key",
                    "description": "Azure OpenAI API Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "api_base",
                    "description": "Azure OpenAI endpoint URL",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "api_version",
                    "description": "API version (e.g., 2024-02-01)",
                    "secret": False,
                    "required": True,
                },
            ],
        },
    },
    "vertex_ai": {
        "service_account_json": {
            "display_name": "Service Account JSON",
            "description": "Use GCP Service Account credentials (JSON key file contents)",
            "default": True,
            "fields": [
                {
                    "name": "vertex_credentials",
                    "description": "Service Account JSON key file contents",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "vertex_project",
                    "description": "GCP Project ID",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "vertex_location",
                    "description": "GCP Region (e.g., us-central1)",
                    "secret": False,
                    "required": False,
                },
            ],
        },
    },
    "databricks": {
        "pat_token": {
            "display_name": "Personal Access Token",
            "description": "Use Databricks Personal Access Token",
            "default": True,
            "fields": [
                {
                    "name": "api_key",
                    "description": "Databricks Personal Access Token",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "api_base",
                    "description": "Databricks workspace URL",
                    "secret": False,
                    "required": True,
                },
            ],
        },
    },
    "sagemaker": {
        "access_keys": {
            "display_name": "Access Keys",
            "description": "Use AWS Access Key ID and Secret Access Key",
            "default": True,
            "fields": [
                {
                    "name": "aws_access_key_id",
                    "description": "AWS Access Key ID",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "description": "AWS Secret Access Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region (e.g., us-east-1)",
                    "secret": False,
                    "required": True,
                },
            ],
        },
    },
}

# Display names for providers that don't title-case cleanly.
# Copied from MLflow AI Gateway's PROVIDER_DISPLAY_NAMES.
# Providers not in this dict fall back to .replace("_", " ").title().
_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "bedrock": "Amazon Bedrock",
    "gemini": "Google Gemini",
    "vertex_ai": "Google Vertex AI",
    "azure": "Azure OpenAI",
    "groq": "Groq",
    "databricks": "Databricks",
    "xai": "xAI",
    "cohere": "Cohere",
    "mistral": "Mistral AI",
    "together_ai": "Together AI",
    "fireworks_ai": "Fireworks AI",
    "replicate": "Replicate",
    "huggingface": "Hugging Face",
    "ai21": "AI21",
    "perplexity": "Perplexity",
    "deepinfra": "DeepInfra",
    "cerebras": "Cerebras",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
    "ollama": "Ollama",
}


def format_provider_name(provider: str) -> str:
    """
    Return a human-readable display name for a provider.

    Uses a lookup table for providers that don't title-case cleanly
    (e.g. ``"openai"`` → ``"OpenAI"``). Falls back to
    ``provider.replace("_", " ").title()`` for unknown providers.

    :param provider: Provider identifier, e.g. ``"openai"``.
    :returns: Display name, e.g. ``"OpenAI"``.
    """
    if provider in _PROVIDER_DISPLAY_NAMES:
        return _PROVIDER_DISPLAY_NAMES[provider]
    return provider.replace("_", " ").title()


# Env var names for simple API-key providers (used for non-interactive mode).
PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHERAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "ai21": "AI21_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "cloudflare": "CLOUDFLARE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "novita": "NOVITA_API_KEY",
}


def get_provider_config(provider: str) -> ProviderConfig:
    """
    Return the auth configuration for a provider.

    For providers with multiple auth modes (bedrock, azure, vertex_ai,
    databricks, sagemaker), returns the full structure. For simple
    API-key providers, returns a single default auth mode.

    :param provider: Provider name, e.g. ``"openai"`` or ``"bedrock"``.
    :returns: :class:`ProviderConfig` with available auth modes.
    """
    if provider in _PROVIDER_AUTH_MODES:
        modes: list[AuthMode] = []
        default_mode_id: str | None = None
        for mode_id, mode_def in _PROVIDER_AUTH_MODES[provider].items():
            fields = [
                AuthField(
                    name=f["name"],
                    description=f["description"],
                    secret=f["secret"],
                    required=f["required"],
                )
                for f in mode_def["fields"]
            ]
            is_default = mode_def.get("default", False)
            if is_default:
                default_mode_id = mode_id
            modes.append(
                AuthMode(
                    mode_id=mode_id,
                    display_name=mode_def["display_name"],
                    description=mode_def["description"],
                    fields=fields,
                    is_default=is_default,
                )
            )
        return ProviderConfig(
            auth_modes=modes,
            default_mode=default_mode_id or modes[0].mode_id,
        )

    # Simple API-key provider — generate a default mode.
    display = format_provider_name(provider)
    return ProviderConfig(
        auth_modes=[
            AuthMode(
                mode_id="api_key",
                display_name="API Key",
                description=f"Use {display} API Key",
                fields=[
                    AuthField(
                        name="api_key",
                        description=f"{display} API Key",
                        secret=True,
                        required=True,
                    ),
                ],
                is_default=True,
            ),
        ],
        default_mode="api_key",
    )
