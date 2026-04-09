# Deploying Agent-Plane on Databricks Apps

This example deploys agent-plane to [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) with:

- **Lakebase** (managed PostgreSQL) as the database for all stores
- **UC Volumes** as the artifact store for agent bundles and executor storage snapshots

## Prerequisites

1. A Databricks workspace with Lakebase and UC Volumes enabled
2. [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/install.html) installed and configured
3. A Lakebase Postgres project with a branch, endpoint, and database
4. A UC Volume for artifact storage

### Create Lakebase Resources

If you don't have an existing Lakebase project:

1. Go to **SQL** > **Lakebase** in your Databricks workspace
2. Create a new project
3. Note the branch name (e.g., `projects/my-project/branches/production`)
4. Create a database in the branch
5. Note the database name (e.g., `projects/my-project/branches/production/databases/my-db`)

### Create UC Volume

```sql
-- In a Databricks notebook or SQL editor
CREATE CATALOG IF NOT EXISTS my_catalog;
CREATE SCHEMA IF NOT EXISTS my_catalog.agent_plane;
CREATE VOLUME IF NOT EXISTS my_catalog.agent_plane.artifacts;
```

## Deploy

```bash
# From this directory
databricks bundle deploy \
  --var lakebase_branch=projects/my-project/branches/production \
  --var lakebase_database=projects/my-project/branches/production/databases/my-db \
  --var volume_name=my_catalog.agent_plane.artifacts
```

## How It Works

### Authentication

The app runs as a Databricks service principal. Credentials are
managed automatically:

- **Lakebase**: OAuth tokens generated via `WorkspaceClient.postgres.generate_database_credential()`, injected into every new SQLAlchemy connection via a `do_connect` event hook
- **UC Volumes**: Workspace credentials used by the Databricks SDK (ambient in Apps)

### Token Lifecycle

Lakebase OAuth tokens expire after 60 minutes. The SQLAlchemy
connection pool recycles connections every 5 minutes (configurable
via `AP_POOL_RECYCLE_SECONDS`), ensuring fresh tokens on new
connections.

### Architecture

```
Databricks App (service principal)
  └── app.py
       ├── Lakebase token hook (do_connect)
       ├── agent-plane server (uvicorn + FastAPI)
       │   ├── SQLAlchemy stores → Lakebase PostgreSQL
       │   ├── ArtifactStore → UC Volumes (dbfs:/Volumes/...)
       │   └── DBOS workflow engine → same Lakebase database
       └── Agent bundles uploaded via POST /api/agents
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
| `DATABRICKS_APP_PORT` | Databricks runtime | App port (default 8000) |
| `AP_POOL_RECYCLE_SECONDS` | Optional | Connection pool recycle interval (default 300) |
