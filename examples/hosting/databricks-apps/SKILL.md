---
name: deploy-databricks-app
description: Deploy agent-plane to Databricks Apps with Lakebase and UC Volumes. Use when setting up infrastructure, deploying, redeploying, troubleshooting, or connecting to a remote agent-plane instance on Databricks.
---

# Deploy Agent-Plane to Databricks Apps

## Prerequisites

- `databricks-sdk` installed
- `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars set
- Workspace admin for creating Lakebase projects (or use existing ones)

## Infrastructure Setup

### 1. Create Lakebase project + database

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import Project, Database, DatabaseDatabaseSpec

wc = WorkspaceClient()
wc.postgres.create_project(project=Project(), project_id="agent-plane")
wc.postgres.create_database(
    parent="projects/agent-plane/branches/production",
    database=Database(spec=DatabaseDatabaseSpec(postgres_database="agent_plane")),
    database_id="agent-plane-db",
)
```

### 2. Create UC Volume

```sql
CREATE SCHEMA IF NOT EXISTS my_catalog.agent_plane;
CREATE VOLUME IF NOT EXISTS my_catalog.agent_plane.artifacts;
```

### 3. Store API keys as secrets

```python
wc.secrets.create_scope(scope="agent-plane")
wc.secrets.put_secret(scope="agent-plane", key="openai-api-key", string_value="sk-...")
```

## Build and Deploy

### Build the wheel

```bash
python -m build --wheel --outdir dist/ --no-isolation
```

### Create the app

```python
from databricks.sdk.service.apps import (
    App, AppResource, AppResourcePostgres,
    AppResourcePostgresPostgresPermission,
    AppResourceUcSecurable,
    AppResourceUcSecurableUcSecurablePermission,
    AppResourceUcSecurableUcSecurableType,
    AppResourceSecret, AppResourceSecretSecretPermission,
)

wait = wc.apps.create(app=App(
    name="agent-plane",
    resources=[
        AppResource(name="postgres", postgres=AppResourcePostgres(
            branch="projects/agent-plane/branches/production",
            database="projects/agent-plane/branches/production/databases/agent-plane-db",
            permission=AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE,
        )),
        AppResource(name="artifact_volume", uc_securable=AppResourceUcSecurable(
            securable_full_name="my_catalog.agent_plane.artifacts",
            securable_type=AppResourceUcSecurableUcSecurableType.VOLUME,
            permission=AppResourceUcSecurableUcSecurablePermission.WRITE_VOLUME,
        )),
        AppResource(name="openai_key", secret=AppResourceSecret(
            scope="agent-plane", key="openai-api-key",
            permission=AppResourceSecretSecretPermission.READ,
        )),
    ],
))
app = wait.result()
```

### Upload and deploy

```python
from databricks.sdk.service.workspace import ImportFormat
from databricks.sdk.service.apps import AppDeployment, EnvVar
from pathlib import Path
import io

ws_base = "/Workspace/Users/you@company.com/agent-plane-app"
wc.workspace.mkdirs(ws_base)

# Upload app.py, app.yaml, wheel
for f in ["app.py", "app.yaml"]:
    wc.workspace.upload(path=f"{ws_base}/{f}",
        content=io.BytesIO(Path(f).read_bytes()),
        format=ImportFormat.AUTO, overwrite=True)

wheel = Path("dist/agent_plane-X.Y.Z-py3-none-any.whl")
wc.workspace.upload(path=f"{ws_base}/{wheel.name}",
    content=io.BytesIO(wheel.read_bytes()),
    format=ImportFormat.AUTO, overwrite=True)
wc.workspace.upload(path=f"{ws_base}/requirements.txt",
    content=io.BytesIO(f"./{wheel.name}[databricks]\n".encode()),
    format=ImportFormat.AUTO, overwrite=True)

# Deploy
wait = wc.apps.deploy(app_name="agent-plane",
    app_deployment=AppDeployment(
        source_code_path=ws_base,
        env_vars=[
            EnvVar(name="AP_LAKEBASE_ENDPOINT", value_from="postgres"),
            EnvVar(name="AP_ARTIFACT_VOLUME_PATH", value_from="artifact_volume"),
            EnvVar(name="OPENAI_API_KEY", value_from="openai_key"),
        ],
    ))
dep = wait.result()
```

### Redeploy after code changes

Bump version in `pyproject.toml` (pip caches by version), rebuild wheel, re-upload, redeploy. Same deploy code as above.

## Connect TUI

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com \
./scripts/connect-remote.sh https://your-app.databricksapps.com archer
```

Opens browser for OAuth, then launches TUI.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `permission denied for table agents` | Tables owned by wrong user (local testing polluted DB) | `DROP TABLE ... CASCADE` as the owning user, redeploy |
| `schema "dbos" already exists` | Same — `dbos` schema owned by wrong user | `DROP SCHEMA dbos CASCADE`, redeploy |
| Agent registration 400 `Unresolved environment variable` | `OPENAI_API_KEY` not in env vars | Add secret resource + `EnvVar(value_from=...)` in deploy |
| `MATCH` syntax error on search | SQLite FTS5 syntax on PostgreSQL | Update to latest version (has ILIKE fallback) |
| `ModuleNotFoundError: No module named 'mcp'` | Missing dependency in wheel | Rebuild wheel from latest source (mcp added to deps) |
| App starts then crashes | Check logs via workspace UI: App > Logs tab | Usually a missing env var or dependency |

## Key Architecture

- **app.py** registers a class-level `do_connect` SQLAlchemy hook BEFORE importing agent-plane — this injects Lakebase OAuth tokens into every DB connection
- **UC Volumes** artifact store uses `dbfs:/Volumes/...` URI scheme
- **DBOS** shares the same Lakebase database (creates `dbos` schema)
- **OAuth** for TUI access uses `databricks-cli` OIDC client with PKCE + localhost redirect
