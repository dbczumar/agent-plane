"""CLI entry point for agent-plane."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml


# Any: YAML configs have heterogeneous value types (str, int, list, etc.)
def _load_config(path: str | None) -> dict[str, Any]:
    """
    Load and return config from a YAML file.
    Returns an empty dict if no path is provided.
    """
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


_DEFAULT_DB_URI = "sqlite:///agent_plane.db"
_DEFAULT_ARTIFACT_LOCATION = "./artifacts"


@click.group()
def cli() -> None:
    """agent-plane CLI."""


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind to.",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--database-uri",
    default=None,
    help=f"Database URI for stores.  [default: {_DEFAULT_DB_URI}]",
)
@click.option(
    "--artifact-location",
    default=None,
    help=f"Path for artifact storage.  [default: {_DEFAULT_ARTIFACT_LOCATION}]",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file.",
)
def server(
    host: str,
    port: int,
    database_uri: str | None,
    artifact_location: str | None,
    config_path: str | None,
) -> None:
    """Start the agent-plane server."""
    import uvicorn

    from agent_plane.server.app import create_app
    from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from agent_plane.stores.artifact_store.local import LocalArtifactStore
    from agent_plane.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )
    from agent_plane.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
    from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore

    cfg = _load_config(config_path)

    # CLI args take precedence over config file, which takes precedence
    # over defaults.
    db_uri = database_uri or cfg.get("database_uri", _DEFAULT_DB_URI)
    art_loc = artifact_location or cfg.get("artifact_location", _DEFAULT_ARTIFACT_LOCATION)

    # Resolve relative artifact location against config file's directory
    # (only when the value came from the config file, not CLI).
    if config_path and artifact_location is None and not Path(art_loc).is_absolute():
        art_loc = str(Path(config_path).parent / art_loc)

    app = create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        task_store=SqlAlchemyTaskStore(db_uri),
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=LocalArtifactStore(art_loc),
    )

    click.echo(f"Starting agent-plane server on {host}:{port}")
    click.echo(f"  database:  {db_uri}")
    click.echo(f"  artifacts: {art_loc}")

    uvicorn.run(app, host=host, port=port)
