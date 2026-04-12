"""
Interactive provider and model selection for ``ap create``.

Prompts the user to pick a provider, supply credentials, and
select a chat-capable model. Uses Rich for polished terminal output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent_plane.onboarding.providers import (
    COMMON_PROVIDERS,
    PROVIDER_ENV_VARS,
    AuthField,
    ModelInfo,
    get_all_providers,
    get_onboarding_models,
    get_provider_config,
)

console = Console()


@dataclass
class ProviderSelection:
    """
    Result of the provider/model selection flow.

    :param provider: Provider name, e.g. ``"anthropic"``.
    :param model: Full litellm model string, e.g.
        ``"anthropic/claude-sonnet-4-20250514"``.
    :param credentials: Credential key-value pairs collected from
        the user, e.g. ``{"api_key": "sk-ant-..."}``.
    """

    provider: str
    model: str
    credentials: dict[str, str]


def select_provider_interactive() -> ProviderSelection:
    """
    Run the full interactive provider -> credentials -> model flow.

    Displays a numbered list of providers, prompts the user to pick
    one, collects credentials based on the provider's auth mode,
    then lists chat-capable models and lets the user choose.

    :returns: A :class:`ProviderSelection` with the chosen provider,
        model, and credentials.
    :raises click.Abort: If the user cancels at any prompt.
    """
    provider = _prompt_provider()
    credentials = _prompt_credentials(provider)
    model = _prompt_model(provider)
    return ProviderSelection(
        provider=provider,
        model=f"{provider}/{model}",
        credentials=credentials,
    )


def resolve_provider_from_model(model_string: str) -> ProviderSelection:
    """
    Build a :class:`ProviderSelection` from a ``--model`` flag value.

    Parses the litellm ``provider/model_name`` format and reads
    credentials from environment variables.

    :param model_string: Model in litellm format, e.g.
        ``"anthropic/claude-sonnet-4-20250514"``.
    :returns: A :class:`ProviderSelection` with credentials from env.
    :raises click.ClickException: If the model string is malformed or
        the required env var is not set.
    """
    if "/" not in model_string:
        raise click.ClickException(
            f"Model must be in provider/model_name format, got: {model_string!r}"
        )

    provider, _ = model_string.split("/", 1)
    credentials = _read_credentials_from_env(provider)
    return ProviderSelection(
        provider=provider,
        model=model_string,
        credentials=credentials,
    )


# ---------------------------------------------------------------------------
# Provider prompt
# ---------------------------------------------------------------------------

_COLUMN_COUNT = 3


def _prompt_provider() -> str:
    """
    Display providers in a Rich panel and prompt the user to pick one.

    Popular providers are listed first (matching MLflow AI Gateway).

    :returns: The selected provider name.
    """
    providers = get_all_providers()
    if not providers:
        raise click.ClickException("No providers found in the model catalog.")

    # Build numbered entries with popular providers highlighted.
    popular_set = set(COMMON_PROVIDERS)
    entries: list[Text] = []
    for i, name in enumerate(providers, 1):
        entry = Text()
        entry.append(f" {i:>3}. ", style="dim")
        if name in popular_set:
            entry.append(name, style="bold cyan")
        else:
            entry.append(name)
        entries.append(entry)

    console.print()
    console.print(
        Panel(
            Columns(entries, column_first=True, padding=(0, 2)),
            title="[bold]Select a provider for the onboarding assistant[/bold]",
            subtitle=f"[dim]{len(providers)} providers available[/dim]",
            border_style="blue",
        )
    )

    return _collect_provider_choice(providers)


def _collect_provider_choice(providers: list[str]) -> str:
    """
    Prompt until the user enters a valid provider number or name.

    :param providers: The full ordered provider list.
    :returns: The selected provider name.
    """
    while True:
        raw = str(click.prompt("Provider", default="1"))
        try:
            choice = int(raw)
            if 1 <= choice <= len(providers):
                return providers[choice - 1]
            console.print(f"  [red]Enter a number between 1 and {len(providers)}.[/red]")
        except ValueError:
            name = raw.strip().lower()
            if name in providers:
                return name
            console.print(f"  [red]Unknown provider:[/red] {raw!r}")


# ---------------------------------------------------------------------------
# Credentials prompt
# ---------------------------------------------------------------------------


def _prompt_credentials(provider: str) -> dict[str, str]:
    """
    Prompt the user for credentials based on the provider's auth config.

    :param provider: Provider name, e.g. ``"bedrock"``.
    :returns: Dict of credential field name to value.
    """
    config = get_provider_config(provider)
    default_mode = next(
        (m for m in config.auth_modes if m.mode_id == config.default_mode),
        config.auth_modes[0],
    )

    credentials: dict[str, str] = {}
    for field in default_mode.fields:
        if not field.required:
            continue
        value = _prompt_field(field, provider)
        credentials[field.name] = value

    return credentials


def _prompt_field(field: AuthField, provider: str) -> str:
    """
    Prompt for a single credential field with env var hint.

    :param field: The auth field to prompt for.
    :param provider: Provider name (for env var lookup).
    :returns: The user-supplied value.
    """
    env_val = _check_env_for_field(field.name, provider)

    while True:
        if env_val:
            masked = env_val[:8] + "..." if len(env_val) > 8 else env_val
            console.print(
                f"  [dim]Found in env:[/dim] {masked}",
                highlight=False,
            )
            value = str(
                click.prompt(
                    field.description,
                    default=env_val,
                    hide_input=field.secret,
                    show_default=False,
                )
            )
        else:
            value = str(
                click.prompt(
                    field.description,
                    hide_input=field.secret,
                )
            )

        if value.strip():
            return value.strip()
        console.print("  [red]Value cannot be empty.[/red]")


def _check_env_for_field(field_name: str, provider: str) -> str | None:
    """
    Check environment for a credential value.

    :param field_name: The field name, e.g. ``"api_key"``.
    :param provider: The provider name for provider-specific vars.
    :returns: The env var value if found, or ``None``.
    """
    # Check provider-specific env var first.
    if field_name == "api_key":
        provider_var = PROVIDER_ENV_VARS.get(provider)
        if provider_var:
            val = os.environ.get(provider_var)
            if val:
                return val

    # Try exact uppercase match.
    val = os.environ.get(field_name.upper())
    if val:
        return val
    return None


# ---------------------------------------------------------------------------
# Model prompt
# ---------------------------------------------------------------------------


def _prompt_model(provider: str) -> str:
    """
    List models suitable for the onboarding assistant and let the user pick.

    Only shows text chat models with function calling support (the
    onboarding assistant needs tools). Excludes audio, realtime, image,
    embedding, and other non-text-chat models. Defaults to the first
    model (newest/best).

    :param provider: Provider name, e.g. ``"anthropic"``.
    :returns: The model name (without provider prefix).
    """
    models = get_onboarding_models(provider)
    if not models:
        # Fall back to all chat models if none support function calling.
        from agent_plane.onboarding.providers import get_chat_models

        models = get_chat_models(provider)

    if not models:
        console.print(f"\n  [yellow]No models found for {provider}.[/yellow]")
        return str(click.prompt("Enter model name manually"))

    if len(models) == 1:
        console.print(f"\n  Model: [bold]{models[0].name}[/bold]")
        return models[0].name

    _display_model_list(models)
    return _collect_model_choice(models)


def _display_model_list(models: list[ModelInfo]) -> None:
    """
    Display models as a compact numbered list, newest first.

    :param models: Sorted model list.
    """
    display_limit = 15
    entries: list[Text] = []
    for i, m in enumerate(models[:display_limit], 1):
        entry = Text()
        entry.append(f" {i:>3}. ", style="dim")
        entry.append(m.name, style="cyan")
        entries.append(entry)

    console.print()
    console.print(
        Panel(
            Columns(entries, column_first=True, padding=(0, 2)),
            title="[bold]Select a model for the onboarding assistant[/bold]",
            subtitle=f"[dim]{len(models)} models · newest first[/dim]",
            border_style="blue",
        )
    )

    if len(models) > display_limit:
        console.print(
            f"  [dim]{len(models) - display_limit} more available — "
            f'type a name to search (e.g. "gpt-5")[/dim]'
        )
    console.print(f"  [dim]Default:[/dim] [bold]{models[0].name}[/bold]")


def _collect_model_choice(models: list[ModelInfo]) -> str:
    """
    Prompt the user to select a model by number or name search.

    Defaults to 1 (the newest model).

    :param models: The full sorted list of available models.
    :returns: The chosen model name.
    """
    while True:
        raw = str(click.prompt("Model", default="1"))
        try:
            choice = int(raw)
            if 1 <= choice <= len(models):
                return models[choice - 1].name
            console.print(f"  [red]Enter a number between 1 and {len(models)}.[/red]")
        except ValueError:
            query = raw.strip().lower()
            matches = [m for m in models if query in m.name.lower()]
            if len(matches) == 1:
                return matches[0].name
            if len(matches) > 1:
                console.print(
                    f"  [yellow]Multiple matches:[/yellow] "
                    f"{', '.join(m.name for m in matches[:5])}"
                )
            else:
                return raw.strip()


# ---------------------------------------------------------------------------
# Non-interactive credential resolution
# ---------------------------------------------------------------------------


def _read_credentials_from_env(provider: str) -> dict[str, str]:
    """
    Read credentials from environment variables for non-interactive mode.

    :param provider: Provider name, e.g. ``"anthropic"``.
    :returns: Dict with credential fields from env vars.
    :raises click.ClickException: If required env vars are missing.
    """
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var:
        value = os.environ.get(env_var)
        if value:
            return {"api_key": value}
        raise click.ClickException(
            f"Non-interactive mode requires {env_var} for provider {provider!r}."
        )

    # Complex providers — check default auth mode fields.
    config = get_provider_config(provider)
    default_mode = next(
        (m for m in config.auth_modes if m.mode_id == config.default_mode),
        config.auth_modes[0],
    )
    return _collect_env_credentials(provider, default_mode.fields)


def _collect_env_credentials(
    provider: str,
    fields: list[AuthField],
) -> dict[str, str]:
    """
    Collect required credential values from environment variables.

    :param provider: Provider name for error messages.
    :param fields: Auth fields to check.
    :returns: Dict of field name to env var value.
    :raises click.ClickException: If any required field is missing.
    """
    credentials: dict[str, str] = {}
    missing: list[str] = []
    for field in fields:
        if not field.required:
            continue
        val = os.environ.get(field.name.upper())
        if val:
            credentials[field.name] = val
        else:
            missing.append(field.name.upper())

    if missing:
        raise click.ClickException(
            f"Non-interactive mode requires these env vars for provider "
            f"{provider!r}: {', '.join(missing)}"
        )
    return credentials
