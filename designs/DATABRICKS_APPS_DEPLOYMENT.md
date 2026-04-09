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

1. Reads Lakebase connection env vars (`PGHOST`, `PGPORT`, etc.)
2. Registers a SQLAlchemy `do_connect` event listener for token
   injection
3. Builds the database URI and artifact location
4. Starts the agent-plane server via subprocess or direct import

```python
import os
import subprocess
from databricks.sdk import WorkspaceClient

LAKEBASE_ENDPOINT = os.environ["AP_LAKEBASE_ENDPOINT"]
VOLUME_PATH = os.environ["AP_ARTIFACT_VOLUME_PATH"]
PORT = os.environ.get("DATABRICKS_APP_PORT", "8000")

# Build connection string
pg_host = os.environ["PGHOST"]
pg_port = os.environ["PGPORT"]
pg_db = os.environ["PGDATABASE"]
pg_user = os.environ["PGUSER"]
db_uri = f"postgresql+psycopg://{pg_user}@{pg_host}:{pg_port}/{pg_db}"

# Artifact location as dbfs: URI
artifact_uri = f"dbfs:{VOLUME_PATH}"

# Inject Lakebase OAuth tokens into SQLAlchemy connections
from sqlalchemy import event, create_engine

def _get_token():
    wc = WorkspaceClient()
    return wc.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT
    ).token

engine = create_engine(
    db_uri,
    pool_recycle=3000,  # Recycle before 60-min token expiry
)

@event.listens_for(engine, "do_connect")
def inject_token(dialect, conn_rec, cargs, cparams):
    cparams["password"] = _get_token()

# Start agent-plane
subprocess.run([
    "ap", "server",
    "--host", "0.0.0.0",
    "--port", PORT,
    "--database-uri", db_uri,
    "--artifact-location", artifact_uri,
])
```

**Note:** The `do_connect` hook injects a fresh token on every new
connection. The pool recycles connections every 3000 seconds (50
minutes) so tokens never expire mid-session.

**Open question:** The `do_connect` hook operates on `engine`, but
agent-plane creates its own engine internally. The example may need
to set `AP_DB_URI` as an env var and patch the engine creation in
agent-plane to accept the pre-configured engine, OR agent-plane's
engine creation needs to support a `do_connect` hook registration
callback. The MLflow example handles this by registering the hook
before importing MLflow (which creates the engine at import time).
Agent-plane's engine creation happens in `db/utils.py` — we may
need a similar hook point.

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

1. **Engine hook for token injection.** Agent-plane creates its own
   SQLAlchemy engine in `db/utils.py`. The Lakebase token injection
   hook needs to be registered on THAT engine, not a separate one.
   Options:
   - (a) Accept a `connect_hook` callback in `db/utils.py`'s engine
     creation and register it via `event.listens_for`
   - (b) Export the engine after creation so `app.py` can register
     on it
   - (c) Set env vars and have `db/utils.py` detect Databricks
     context and auto-register the hook

   Option (a) is cleanest — explicit, no magic detection.

2. **Executor storage on UC Volumes.** The `_EXECUTOR_STORAGE_BASE`
   is currently `~/.agent-plane/executor_storage/` (local disk). On
   Databricks Apps, local disk is ephemeral. Should executor storage
   also move to UC Volumes? Or is the artifact store snapshot
   (which already uses UC Volumes) sufficient for crash recovery?
   The artifact store snapshot restores on server restart, so
   ephemeral local storage + artifact backup should be sufficient.

3. **DBOS system database.** DBOS creates its own `agent_plane.db`
   SQLite database for workflow state. On Databricks Apps, this
   needs to be either: (a) pointed at a Lakebase-compatible
   PostgreSQL URI, or (b) kept on ephemeral local disk (workflow
   state is reconstructed on restart). DBOS supports PostgreSQL
   natively — this just needs the right configuration.
