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
@click.option(
    "--execution-timeout",
    default=None,
    type=int,
    help="Max wall-clock seconds per agent execution.  [default: 7200]",
)
def server(
    host: str,
    port: int,
    database_uri: str | None,
    artifact_location: str | None,
    config_path: str | None,
    execution_timeout: int | None,
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

    agent_store = SqlAlchemyAgentStore(db_uri)
    file_store = SqlAlchemyFileStore(db_uri)
    task_store = SqlAlchemyTaskStore(db_uri)
    conversation_store = SqlAlchemyConversationStore(db_uri)
    artifact_store = LocalArtifactStore(art_loc)

    # Initialize the runtime with store references so workflow code
    # can access them via getter functions (get_agent_cache(), etc.).
    from agent_plane.runtime import init as init_runtime
    from agent_plane.runtime.agent_cache import AgentCache
    from agent_plane.runtime.caps import RuntimeCaps

    agent_cache = AgentCache(
        artifact_store=artifact_store,
        cache_dir=Path(art_loc) / ".cache",
    )
    # CLI flag > config file > RuntimeCaps default (7200s = 2 hours).
    # 7200 matches RuntimeCaps.execution_timeout default.
    effective_timeout = execution_timeout or cfg.get("execution_timeout") or 7200
    caps = RuntimeCaps(execution_timeout=int(effective_timeout))
    init_runtime(
        conversation_store=conversation_store,
        task_store=task_store,
        agent_store=agent_store,
        agent_cache=agent_cache,
        caps=caps,
    )

    app = create_app(
        agent_store=agent_store,
        file_store=file_store,
        task_store=task_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
    )

    click.echo(f"Starting agent-plane server on {host}:{port}")
    click.echo(f"  database:  {db_uri}")
    click.echo(f"  artifacts: {art_loc}")

    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--server",
    "-s",
    default="http://localhost:8000",
    show_default=True,
    help="Agent-plane server URL.",
)
def deploy(path: str, server: str) -> None:
    """Deploy an agent image to the server."""
    import httpx

    bundle_bytes = _bundle(Path(path))
    url = f"{server.rstrip('/')}/api/agents"

    # 120s timeout: agent bundles can be large and the server
    # needs time to extract and register the image.
    resp = httpx.post(
        url,
        files={"bundle": ("agent.tar.gz", bundle_bytes, "application/gzip")},
        timeout=120.0,
    )

    if resp.status_code == 201:
        body = resp.json()
        click.echo(f"Deployed agent '{body['name']}' (id: {body['id']})")
    else:
        click.echo(f"Deploy failed ({resp.status_code}): {resp.text}", err=True)
        raise SystemExit(1)


def _bundle(source: Path) -> bytes:
    """
    Produce a tar.gz bundle from a directory or pass through an existing tarball.
    """
    if source.is_file():
        return source.read_bytes()

    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                # Relative path inside the bundle (e.g. "config.yaml")
                arcname = str(file_path.relative_to(source))
                tf.add(str(file_path), arcname=arcname)
    return buf.getvalue()


if __name__ == "__main__":
    cli()
