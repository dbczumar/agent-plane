"""CLI entry point for agent-plane."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import BaseModel, ConfigDict


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
        file_store=file_store,
        artifact_store=artifact_store,
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
    Produce a tar.gz bundle from a directory or pass
    through an existing tarball.

    Environment variable references (``${VAR}``) in
    ``config.yaml`` and ``tools/mcp/*.yaml`` are expanded
    using the client's environment before bundling. This
    ensures the server receives resolved secrets rather
    than unresolved ``${VAR}`` references it cannot
    resolve.

    :param source: Path to an agent image directory or an
        existing ``.tar.gz`` bundle file.
    :returns: The gzipped tarball bytes.
    :raises AgentPlaneError: If a required env var is
        missing during expansion.
    """
    if source.is_file():
        return source.read_bytes()

    import io
    import tarfile

    # Pre-resolve env vars in YAML files that contain secrets.
    resolved = _resolve_bundle_env_vars(source)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                arcname = str(file_path.relative_to(source))
                if arcname in resolved:
                    # Write the resolved YAML instead of the
                    # original file (which has ${VAR} refs).
                    data = resolved[arcname].encode("utf-8")
                    info = tarfile.TarInfo(name=arcname)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
                else:
                    tf.add(str(file_path), arcname=arcname)
    return buf.getvalue()


def _resolve_bundle_env_vars(source: Path) -> dict[str, str]:
    """
    Expand ``${VAR}`` references in YAML files that contain
    secrets, using the client's environment.

    Returns a mapping of ``arcname → resolved YAML text`` for
    files that were modified. Files without env var references
    are omitted (bundled as-is).

    Expanded fields:

    - ``config.yaml``: ``llm.connection.*`` values and
      ``tools.builtins[*]`` dict-entry values (except ``name``)
    - ``tools/mcp/*.yaml``: ``headers.*`` values

    :param source: The agent image directory.
    :returns: ``{arcname: resolved_yaml_text}`` for files
        that had env vars expanded.
    :raises AgentPlaneError: If a ``${VAR}`` reference
        cannot be resolved from the environment.
    """
    from agent_plane.spec import expand_env_vars

    resolved: dict[str, str] = {}

    # ── config.yaml ──────────────────────────────────
    config_path = source / "config.yaml"
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text())
        if isinstance(raw, dict):
            changed = _expand_config_env_vars(raw, expand_env_vars)
            if changed:
                resolved["config.yaml"] = yaml.dump(
                    raw,
                    default_flow_style=False,
                )

    # ── tools/mcp/*.yaml ─────────────────────────────
    mcp_dir = source / "tools" / "mcp"
    if mcp_dir.is_dir():
        for yaml_file in sorted(mcp_dir.glob("*.yaml")):
            raw = yaml.safe_load(yaml_file.read_text())
            if isinstance(raw, dict) and "headers" in raw:
                headers = raw.get("headers")
                if isinstance(headers, dict):
                    raw["headers"] = expand_env_vars(
                        {str(k): str(v) for k, v in headers.items()},
                    )
                    arcname = str(yaml_file.relative_to(source))
                    resolved[arcname] = yaml.dump(
                        raw,
                        default_flow_style=False,
                    )

    return resolved


class _LLMDeploy(BaseModel):
    """
    Pydantic model for the ``llm:`` block during deploy-time
    env var expansion.

    :param connection: Key-value pairs for LLM connection
        config, e.g. ``{"api_key": "${OPENAI_API_KEY}"}``.
    """

    model_config = ConfigDict(extra="allow")
    connection: dict[str, str] | None = None


class _BuiltinEntry(BaseModel):
    """
    Pydantic model for a single dict entry in
    ``tools.builtins`` during deploy-time env var expansion.

    :param name: The built-in tool name, e.g.
        ``"web_search_google"``.
    """

    model_config = ConfigDict(extra="allow")
    name: str


class _ToolsDeploy(BaseModel):
    """
    Pydantic model for the ``tools:`` block during deploy-time
    env var expansion.

    :param builtins: Mixed list of string tool names and dict
        entries with config fields, e.g.
        ``["web_search_openai", {"name": "web_search_google",
        "api_key": "${KEY}"}]``.
    """

    model_config = ConfigDict(extra="allow")
    builtins: list[str | dict[str, Any]] | None = None


class _DeployConfig(BaseModel):
    """
    Pydantic model for the top-level config.yaml structure
    during deploy-time env var expansion.

    Only the fields containing secrets (``llm``, ``tools``)
    are modeled; all other fields pass through via
    ``extra="allow"``.

    :param llm: The LLM configuration block, or ``None``
        if absent.
    :param tools: The tools configuration block, or ``None``
        if absent.
    """

    model_config = ConfigDict(extra="allow")
    llm: _LLMDeploy | None = None
    tools: _ToolsDeploy | None = None


def _expand_config_env_vars(
    raw: dict[str, Any],
    expand_fn: Callable[[dict[str, str]], dict[str, str]],
) -> bool:
    """
    Expand ``${VAR}`` references in-place in a parsed
    ``config.yaml`` dict. Returns ``True`` if any field
    was expanded.

    Expanded fields:

    - ``llm.connection`` — all values
    - ``tools.builtins[*]`` — dict-entry values except ``name``

    :param raw: The parsed config.yaml dict (modified in-place).
    :param expand_fn: Callable that expands env var references
        in a string-to-string dict, e.g.
        :func:`agent_plane.spec.expand_env_vars`.
    :returns: ``True`` if any values were expanded.
    """
    cfg = _DeployConfig.model_validate(raw)
    changed = False

    if cfg.llm is not None and cfg.llm.connection is not None:
        raw["llm"]["connection"] = expand_fn(cfg.llm.connection)
        changed = True

    if cfg.tools is not None and cfg.tools.builtins is not None:
        changed = (
            _expand_builtin_env_vars(
                raw["tools"]["builtins"],
                cfg.tools.builtins,
                expand_fn,
            )
            or changed
        )

    return changed


def _expand_builtin_env_vars(
    raw_builtins: list[str | dict[str, Any]],
    parsed_builtins: list[str | dict[str, Any]],
    expand_fn: Callable[[dict[str, str]], dict[str, str]],
) -> bool:
    """
    Expand ``${VAR}`` references in dict entries of
    ``tools.builtins``, modifying *raw_builtins* in-place.

    String entries are skipped (no config to expand). Dict
    entries have all fields except ``name`` expanded.

    :param raw_builtins: The mutable builtins list from the
        raw config dict (modified in-place).
    :param parsed_builtins: The Pydantic-parsed builtins list
        used for typed access.
    :param expand_fn: Callable that expands env var references
        in a string-to-string dict.
    :returns: ``True`` if any values were expanded.
    """
    changed = False
    for i, entry in enumerate(parsed_builtins):
        if not isinstance(entry, dict):
            continue
        parsed = _BuiltinEntry.model_validate(entry)
        # Extra fields are the tool-specific config (api_key, etc.).
        config_fields = (
            {str(k): str(v) for k, v in parsed.model_extra.items()} if parsed.model_extra else {}
        )
        if config_fields:
            expanded = expand_fn(config_fields)
            raw_builtins[i] = {"name": parsed.name, **expanded}
            changed = True
    return changed


if __name__ == "__main__":
    cli()
