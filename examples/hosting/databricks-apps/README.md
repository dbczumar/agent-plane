# Deploying Agent-Plane on Databricks Apps

This example deploys agent-plane to [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) with:

- **Lakebase** (managed PostgreSQL) as the database for all stores
- **UC Volumes** as the artifact store for agent bundles and executor storage snapshots

## Prerequisites

1. A Databricks workspace with Lakebase and UC Volumes enabled
2. [Databricks SDK](https://docs.databricks.com/en/dev-tools/sdk-python.html) installed (`pip install databricks-sdk`)
3. `DATABRICKS_HOST` and `DATABRICKS_TOKEN` configured for your workspace

## Step 1: Create Infrastructure

### Lakebase

Create a Lakebase project with a branch, endpoint, and database. You can
do this via the workspace UI (**SQL** > **Lakebase**) or the SDK:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import Project, Database, DatabaseDatabaseSpec

wc = WorkspaceClient()

# Create project (requires workspace admin)
wc.postgres.create_project(project=Project(), project_id="agent-plane")

# The project comes with a production branch and primary endpoint.
# Create a database:
wc.postgres.create_database(
    parent="projects/agent-plane/branches/production",
    database=Database(spec=DatabaseDatabaseSpec(postgres_database="agent_plane")),
    database_id="agent-plane-db",
)
```

Note the full resource names — you'll need them when creating the app:
- Branch: `projects/agent-plane/branches/production`
- Database: `projects/agent-plane/branches/production/databases/agent-plane-db`

### UC Volume

```sql
-- In a Databricks notebook or SQL editor
CREATE CATALOG IF NOT EXISTS my_catalog;
CREATE SCHEMA IF NOT EXISTS my_catalog.agent_plane;
CREATE VOLUME IF NOT EXISTS my_catalog.agent_plane.artifacts;
```

### Secrets (for LLM API keys)

Agents that call external LLMs (e.g. OpenAI) need API keys. Store
them as Databricks secrets:

```python
wc = WorkspaceClient()
wc.secrets.create_scope(scope="agent-plane")
wc.secrets.put_secret(scope="agent-plane", key="openai-api-key", string_value="sk-...")
```

## Step 2: Build the Wheel

Agent-plane is not yet published to PyPI. Build a wheel from source:

```bash
pip install build
python -m build --wheel --outdir dist/ --no-isolation
```

This produces `dist/agent_plane-X.Y.Z-py3-none-any.whl`.

## Step 3: Create and Deploy the App

The `databricks bundle deploy` CLI does not yet support the `postgres`
resource type. Use the Databricks SDK to create the app and deploy:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App, AppResource, AppResourcePostgres,
    AppResourcePostgresPostgresPermission,
    AppResourceUcSecurable,
    AppResourceUcSecurableUcSecurablePermission,
    AppResourceUcSecurableUcSecurableType,
    AppResourceSecret,
    AppResourceSecretSecretPermission,
)

wc = WorkspaceClient()

# Create the app with all resources
wait = wc.apps.create(app=App(
    name="agent-plane",
    description="Agent-plane server with Lakebase + UC Volumes",
    resources=[
        AppResource(
            name="postgres",
            postgres=AppResourcePostgres(
                branch="projects/agent-plane/branches/production",
                database="projects/agent-plane/branches/production/databases/agent-plane-db",
                permission=AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE,
            ),
        ),
        AppResource(
            name="artifact_volume",
            uc_securable=AppResourceUcSecurable(
                securable_full_name="my_catalog.agent_plane.artifacts",
                securable_type=AppResourceUcSecurableUcSecurableType.VOLUME,
                permission=AppResourceUcSecurableUcSecurablePermission.WRITE_VOLUME,
            ),
        ),
        AppResource(
            name="openai_key",
            secret=AppResourceSecret(
                scope="agent-plane",
                key="openai-api-key",
                permission=AppResourceSecretSecretPermission.READ,
            ),
        ),
    ],
))
app = wait.result()
print(f"App created: {app.url}")
```

### Upload source files and deploy

```python
from databricks.sdk.service.workspace import ImportFormat
from databricks.sdk.service.apps import AppDeployment, EnvVar
from pathlib import Path
import io

ws_base = "/Workspace/Users/you@company.com/agent-plane-app"

# Upload app files
for f in ["app.py", "app.yaml", "requirements.txt"]:
    content = Path(f).read_bytes()
    wc.workspace.mkdirs(ws_base)
    wc.workspace.upload(
        path=f"{ws_base}/{f}",
        content=io.BytesIO(content),
        format=ImportFormat.AUTO,
        overwrite=True,
    )

# Upload the wheel
wheel = Path("dist/agent_plane-0.1.4-py3-none-any.whl")
wc.workspace.upload(
    path=f"{ws_base}/{wheel.name}",
    content=io.BytesIO(wheel.read_bytes()),
    format=ImportFormat.AUTO,
    overwrite=True,
)

# Update requirements.txt to reference the wheel
wc.workspace.upload(
    path=f"{ws_base}/requirements.txt",
    content=io.BytesIO(f"./{wheel.name}[databricks]\n".encode()),
    format=ImportFormat.AUTO,
    overwrite=True,
)

# Deploy
wait = wc.apps.deploy(
    app_name="agent-plane",
    app_deployment=AppDeployment(
        source_code_path=ws_base,
        env_vars=[
            EnvVar(name="AP_LAKEBASE_ENDPOINT", value_from="postgres"),
            EnvVar(name="AP_ARTIFACT_VOLUME_PATH", value_from="artifact_volume"),
            EnvVar(name="OPENAI_API_KEY", value_from="openai_key"),
        ],
    ),
)
dep = wait.result()
print(f"Deployed: {dep.status}")
```

## Step 4: Register Agents

The app self-registers agents bundled alongside `app.py`. To add
an agent, upload its tarball and include registration code in `app.py`
(see the `_register_agents` thread in `app.py`).

Or upload agents via the API after authenticating:

```bash
# From a browser session or using the TUI
curl -X POST https://your-app.databricksapps.com/api/agents \
  -F "bundle=@archer.tar.gz"
```

## Step 5: Connect the TUI

Connect the terminal UI to the remote app:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com \
./scripts/connect-remote.sh \
  https://your-app.databricksapps.com \
  archer
```

This opens a browser for Databricks OAuth consent, then launches the
TUI connected to the remote server. The `archer` argument is the
agent name (must already be registered on the app).

With client-side tools:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com \
./scripts/connect-remote.sh \
  https://your-app.databricksapps.com \
  coder --client-tools coder
```

## How It Works

### Authentication

The app runs as a Databricks service principal. Credentials are
managed automatically:

- **Lakebase**: OAuth tokens generated via `WorkspaceClient.postgres.generate_database_credential()`, injected into every new SQLAlchemy connection via a class-level `do_connect` event hook
- **UC Volumes**: Workspace credentials used by the Databricks SDK (ambient in Apps)
- **TUI access**: Browser-based OAuth using the `databricks-cli` OIDC client with PKCE

### Token Lifecycle

Lakebase OAuth tokens expire after 60 minutes. The SQLAlchemy
connection pool recycles connections every 5 minutes by default
(configurable via `AP_POOL_RECYCLE_SECONDS`), ensuring fresh
tokens on new connections.

### Architecture

```
Databricks App (service principal)
  └── app.py
       ├── Lakebase token hook (do_connect on Engine class)
       ├── agent-plane server (uvicorn + FastAPI)
       │   ├── SQLAlchemy stores → Lakebase PostgreSQL
       │   ├── ArtifactStore → UC Volumes (dbfs:/Volumes/...)
       │   └── DBOS workflow engine → same Lakebase database
       └── Agent bundles registered on startup or via API

TUI (your laptop)
  └── terminal.py --server https://app.databricksapps.com archer
       ├── OAuth browser flow → access token
       └── SSE streaming ← agent-plane responses
```

### Storage

| Component | Backend | Purpose |
|-----------|---------|---------|
| Agent specs, tasks, conversations | Lakebase (PostgreSQL) | Durable metadata |
| Agent bundles, executor snapshots | UC Volumes | Binary blob storage |
| DBOS workflow state | Lakebase (same DB) | Workflow recovery |
| Executor working dirs | Local ephemeral disk | Cache (restored from UC Volumes) |

## Configuration

| Environment Variable | Source | Description |
|---------------------|--------|-------------|
| `PGHOST` | Databricks runtime | Lakebase hostname |
| `PGPORT` | Databricks runtime | Lakebase port (default 5432) |
| `PGDATABASE` | Databricks runtime | Lakebase database name |
| `PGUSER` | Databricks runtime | Lakebase user (service principal) |
| `PGSSLMODE` | Databricks runtime | SSL mode (default "require") |
| `AP_LAKEBASE_ENDPOINT` | app.yaml valueFrom | Lakebase endpoint for token generation |
| `AP_ARTIFACT_VOLUME_PATH` | app.yaml valueFrom | UC Volume path for artifacts |
| `OPENAI_API_KEY` | app.yaml valueFrom | LLM API key (from Databricks secret) |
| `DATABRICKS_APP_PORT` | Databricks runtime | App port (default 8000) |
| `AP_POOL_RECYCLE_SECONDS` | Optional | Connection pool recycle interval (default 300) |

## Troubleshooting

### `permission denied for table agents`

The app's service principal doesn't own the database tables. This
happens when you test locally against the same Lakebase database
before deploying the app. Fix: connect as your user and drop the
tables so the app's SP can recreate them:

```sql
DROP TABLE IF EXISTS pending_tool_calls, tasks, conversation_items,
  conversations, files, agents, alembic_version CASCADE;
```

### `schema "dbos" already exists`

Same root cause — the `dbos` schema was created by a different user.
Drop it: `DROP SCHEMA IF EXISTS dbos CASCADE;`

### Agent registration returns 400

Check that `OPENAI_API_KEY` (or other required env vars) is set.
Agent specs with `${OPENAI_API_KEY}` in their config are expanded
at registration time and fail if the variable is missing.

### Search returns syntax error (`MATCH`)

The FTS5 `MATCH` syntax is SQLite-only. On PostgreSQL (Lakebase),
agent-plane uses `ILIKE` as a fallback. If you see this error,
update to the latest version which includes the PostgreSQL fix.
