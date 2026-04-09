# Databricks Apps Deployment

## Problem

Agent-plane's artifact store only supports local filesystem storage
(`LocalArtifactStore`). This prevents deployment to managed
environments like Databricks Apps where:

- Local disk is ephemeral (app restarts lose data)
- Persistent storage is provided by UC Volumes
- The database is Lakebase (managed PostgreSQL), not SQLite

We need a UC Volumes artifact store implementation and an example
deployment template.

---

## Design Decisions

### UC Volumes artifact store selected by URI scheme

The `--artifact-location` value determines which backend to use:

| URI | Backend |
|-----|---------|
| `./artifacts` or `/path/to/dir` | `LocalArtifactStore` |
| `dbfs:/Volumes/catalog/schema/volume/path` | `DatabricksVolumesArtifactStore` |

No new CLI flag. The `dbfs:` scheme is unambiguous — it's the
standard Databricks filesystem URI prefix. Raw `/Volumes/...` paths
are NOT supported (too easy to confuse with local filesystem paths
on machines where FUSE-mounted volumes exist at `/Volumes/`).

**Why `dbfs:/Volumes/...` not `/Volumes/...`:**

- `/Volumes/catalog/schema/volume` is a valid local path on macOS
  (the `/Volumes` directory exists for mounted disks)
- On Databricks clusters with FUSE, `/Volumes/` is a local mount
  point — but Databricks Apps don't have FUSE mounts
- The `dbfs:` prefix makes the intent explicit: this is a remote
  Databricks filesystem path, accessed via the SDK API

### Lakebase requires token injection at connection time

Lakebase uses short-lived OAuth tokens (60-minute expiry) instead
of static passwords. The token is generated via:

```python
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()
credential = wc.postgres.generate_database_credential(
    endpoint=lakebase_endpoint
)
token = credential.token  # Valid for 60 minutes
```

The token must be injected into each new database connection via a
SQLAlchemy `do_connect` event listener. The connection pool must
recycle connections before token expiry (pool_recycle < 3600s).

This is handled entirely in the example's `app.py` entry point —
no changes to agent-plane's SQLAlchemy stores. They already support
PostgreSQL connection strings; the token injection is transparent.

### Agent-plane stores are already Lakebase-compatible

All five SQLAlchemy stores (Agent, Conversation, Task, File,
Artifact metadata) work with any SQLAlchemy-supported database.
Lakebase is PostgreSQL-compatible. The only requirement is a valid
`postgresql+psycopg://` connection string with fresh OAuth tokens.

No store code changes needed for Lakebase.

---

## Implementation

### 1. `DatabricksVolumesArtifactStore`

New file: `agent_plane/stores/artifact_store/databricks_volumes.py`

```python
class DatabricksVolumesArtifactStore(ArtifactStore):
    """
    Artifact store backed by Databricks Unity Catalog Volumes.

    Uses the Databricks SDK WorkspaceClient.files API for all
    operations. Authentication uses ambient workspace credentials
    (automatic in Databricks Apps).

    Storage location format: dbfs:/Volumes/<catalog>/<schema>/<volume>[/<prefix>]
    """
```

**Interface mapping:**

| Method | Implementation |
|--------|---------------|
| `put(key, data)` | `wc.files.upload(path, BytesIO(data), overwrite=True)` |
| `get(key) -> bytes` | `wc.files.download(path).contents.read()` |
| `delete(key)` | `wc.files.delete(path)` (catch `NotFound`) |
| `exists(key) -> bool` | `wc.files.get_status(path)` (catch `NotFound`) |

**Key validation:** Same traversal protection as `LocalArtifactStore`
— reject `..`, backslashes, absolute paths. Keys are appended to
the volume root path.

**Path construction:** Strip the `dbfs:` scheme, yielding
`/Volumes/catalog/schema/volume/prefix`. Append the key:
`/Volumes/catalog/schema/volume/prefix/agents/ag_abc/bundle.tar.gz`.

**Dependency:** `databricks-sdk` (optional — import guarded with
a clear error message if not installed).

### 2. CLI artifact store factory

In `cli.py`, replace the hardcoded `LocalArtifactStore(art_loc)`
with:

```python
def _create_artifact_store(location: str) -> ArtifactStore:
    if location.startswith("dbfs:/Volumes/"):
        from agent_plane.stores.artifact_store.databricks_volumes import (
            DatabricksVolumesArtifactStore,
        )
        return DatabricksVolumesArtifactStore(location)
    return LocalArtifactStore(location)
```

This is a simple scheme-based dispatch. No registry, no plugin
system. Two backends is not enough to justify abstraction.

### 3. Example deployment template

Directory: `examples/hosting/databricks-apps/`

#### `databricks.yml`

Bundle configuration declaring the app, Lakebase database, and
UC Volume:

```yaml
bundle:
  name: agent-plane-server

variables:
  lakebase_branch:
    description: "Lakebase Postgres branch resource name"
  lakebase_database:
    description: "Lakebase Postgres database resource name"
  volume_name:
    description: "UC Volume for artifact storage (catalog.schema.volume)"

resources:
  apps:
    agent-plane:
      name: "agent-plane"
      description: "Agent-plane server with Lakebase + UC Volumes"
      source_code_path: ./
      resources:
        - name: postgres
          postgres:
            branch: ${var.lakebase_branch}
            database: ${var.lakebase_database}
            permission: CAN_CONNECT_AND_CREATE
        - name: artifact_volume
          uc_securable:
            securable_full_name: ${var.volume_name}
            securable_type: VOLUME
            permission: WRITE_VOLUME
```

#### `app.yaml`

Runtime configuration — maps resources to environment variables:

```yaml
command: ["python", "app.py"]
env:
  - name: AP_LAKEBASE_ENDPOINT
    valueFrom: postgres
  - name: AP_ARTIFACT_VOLUME_PATH
    valueFrom: artifact_volume
```

#### `app.py`

Entry point that:

1. Registers a class-level SQLAlchemy `do_connect` event listener
   for Lakebase token injection — **before** importing agent-plane
2. Reads Lakebase connection env vars (`PGHOST`, `PGPORT`, etc.)
3. Builds the database URI and artifact location
4. Starts the agent-plane server in-process via `create_app()` +
   `uvicorn.run()`

**Critical design: class-level event hook.**

The `do_connect` hook is registered on the `Engine` class (not a
specific engine instance). This means it fires for every engine
created in the process — including agent-plane's internal engine
and DBOS's engine. The hook checks the hostname to avoid injecting
tokens into non-Lakebase connections (e.g. SQLite for DBOS).

This requires **no changes to agent-plane's engine creation code**.
The hook is registered at module import time, before any engine
exists. Verified: SQLAlchemy class-level event listeners fire for
all subsequently-created engine instances.

The entry point must start agent-plane **in-process** (not via
`subprocess.run(["ap", "server", ...])`) because Python event
listeners don't cross process boundaries.

```python
"""Databricks Apps entry point for agent-plane with Lakebase."""

import os

from databricks.sdk import WorkspaceClient
from sqlalchemy import Engine, event

# ── Lakebase token injection ──────────────────────────────
#
# Register BEFORE importing agent-plane so the hook is active
# when agent-plane creates its SQLAlchemy engine.

LAKEBASE_ENDPOINT = os.environ["AP_LAKEBASE_ENDPOINT"]
_wc = WorkspaceClient()


def _get_lakebase_token() -> str:
    return _wc.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT,
    ).token


@event.listens_for(Engine, "do_connect")
def _inject_lakebase_token(dialect, conn_rec, cargs, cparams):
    # Only inject for Lakebase connections — skip SQLite (DBOS)
    # and any other non-PostgreSQL engines.
    host = cparams.get("host", "")
    if host and "localhost" not in host and "127.0.0.1" not in host:
        cparams["password"] = _get_lakebase_token()


# ── Build configuration ──────────────────────────────────

VOLUME_PATH = os.environ["AP_ARTIFACT_VOLUME_PATH"]
PORT = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))

pg_host = os.environ["PGHOST"]
pg_port = os.environ["PGPORT"]
pg_db = os.environ["PGDATABASE"]
pg_user = os.environ["PGUSER"]
db_uri = f"postgresql+psycopg://{pg_user}@{pg_host}:{pg_port}/{pg_db}"
artifact_uri = f"dbfs:{VOLUME_PATH}"

# ── Start agent-plane in-process ─────────────────────────

from agent_plane.server import create_app
from agent_plane.stores.artifact_store.databricks_volumes import (
    DatabricksVolumesArtifactStore,
)
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.agent_store.sqlalchemy_store import (
    SqlAlchemyAgentStore,
)
from agent_plane.stores.file_store.sqlalchemy_store import (
    SqlAlchemyFileStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import (
    SqlAlchemyTaskStore,
)
import uvicorn

artifact_store = DatabricksVolumesArtifactStore(artifact_uri)
agent_store = SqlAlchemyAgentStore(db_uri)
conversation_store = SqlAlchemyConversationStore(db_uri)
file_store = SqlAlchemyFileStore(db_uri)
task_store = SqlAlchemyTaskStore(db_uri)

app = create_app(
    agent_store=agent_store,
    file_store=file_store,
    task_store=task_store,
    conversation_store=conversation_store,
    artifact_store=artifact_store,
)

uvicorn.run(app, host="0.0.0.0", port=PORT)
```

**Token lifecycle:** The `do_connect` hook injects a fresh token on
every new database connection. SQLAlchemy's connection pool creates
new connections as needed. Connections should be recycled before the
60-minute token expiry — set `pool_recycle=3000` (50 minutes) on
the engine. This is configured in `db/utils.py`'s engine creation
(needs a one-line change to set `pool_recycle` when the URI is
PostgreSQL).

#### `requirements.txt`

```
agent-plane
databricks-sdk>=0.40.0
psycopg[binary]>=3.1
```

---

## What changes in existing code

| File | Change | Risk |
|------|--------|------|
| `agent_plane/stores/artifact_store/databricks_volumes.py` | New file | None (additive) |
| `agent_plane/cli.py` | Factory dispatch on URI scheme | Low (existing path unchanged) |

**What does NOT change:**

- `ArtifactStore` abstract interface
- Any SQLAlchemy store implementation
- Agent specs, workflow, executors, prompt builder
- Server routes, SSE streaming

---

## Open Questions

1. **Engine `pool_recycle` for Lakebase.** Agent-plane's
   `db/utils.py` creates engines without `pool_recycle`. For
   Lakebase (60-minute token expiry), connections must recycle
   before expiry. Needs a one-line change: pass
   `pool_recycle=3000` when the URI is PostgreSQL. Alternatively,
   `app.py` could monkey-patch the engine after creation, but
   that's fragile.

2. **Executor storage on UC Volumes.** The `_EXECUTOR_STORAGE_BASE`
   is currently `~/.agent-plane/executor_storage/` (local disk). On
   Databricks Apps, local disk is ephemeral. The artifact store
   snapshot (which uses UC Volumes) restores on server restart, so
   ephemeral local storage + artifact backup should be sufficient.
   No change needed unless we want to eliminate the local disk
   dependency entirely.

3. **DBOS system database.** DBOS creates its own `agent_plane.db`
   SQLite database for workflow state. On Databricks Apps, this
   needs to be either: (a) pointed at a Lakebase-compatible
   PostgreSQL URI, or (b) kept on ephemeral local disk (workflow
   state is reconstructed on restart). DBOS supports PostgreSQL
   natively — this just needs the right configuration.
