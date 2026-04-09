"""Databricks Apps entry point for agent-plane.

Starts agent-plane with Lakebase (managed PostgreSQL) as the
database and UC Volumes as the artifact store.

The Lakebase OAuth token injection hook MUST be registered before
importing agent-plane — it fires for all SQLAlchemy engines
created in the process, including agent-plane's internal engine
and DBOS's engine.
"""

from __future__ import annotations

import logging
import os

import sqlalchemy
from databricks.sdk import WorkspaceClient

# ── Configuration ──────────────────────────────────────────

LAKEBASE_ENDPOINT = os.environ["AP_LAKEBASE_ENDPOINT"]
VOLUME_PATH = os.environ["AP_ARTIFACT_VOLUME_PATH"]
PORT = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))

PGHOST = os.environ["PGHOST"]
PGPORT = os.environ.get("PGPORT", "5432")
PGDATABASE = os.environ["PGDATABASE"]
PGUSER = os.environ["PGUSER"]
PGSSLMODE = os.environ.get("PGSSLMODE", "require")

# Recycle DB connections before Lakebase token expiry (60 min).
# Default 300s (5 min) — conservative, negligible overhead.
POOL_RECYCLE_SECONDS = int(
    os.environ.get("AP_POOL_RECYCLE_SECONDS", "300")
)

logger = logging.getLogger(__name__)

# ── Lakebase token injection ──────────────────────────────
#
# Registered at the Engine CLASS level so it fires for every
# engine created in this process. Must be done BEFORE importing
# agent-plane (which creates its own engine at startup).

_workspace_client = WorkspaceClient()


def _get_lakebase_token() -> str:
    """
    Generate a fresh OAuth token for Lakebase.

    Tokens expire after 60 minutes. Called on every new database
    connection (not on pooled connection reuse).

    :returns: The OAuth token string.
    """
    credential = _workspace_client.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT,
    )
    return credential.token


@sqlalchemy.event.listens_for(sqlalchemy.engine.Engine, "do_connect")
def _inject_lakebase_credentials(
    dialect: sqlalchemy.engine.interfaces.Dialect,
    conn_rec: sqlalchemy.pool._ConnectionRecord,
    cargs: list,
    cparams: dict,
) -> None:
    """
    SQLAlchemy event listener that injects a fresh Lakebase OAuth
    token into every new PostgreSQL connection.

    Only intercepts connections to the Lakebase host — skips
    SQLite (DBOS) and any other non-Lakebase engines.

    :param dialect: The SQLAlchemy dialect.
    :param conn_rec: The connection record.
    :param cargs: Positional connection arguments.
    :param cparams: Keyword connection parameters (modified in-place).
    """
    if cparams.get("host") != PGHOST:
        return
    cparams["password"] = _get_lakebase_token()
    cparams["sslmode"] = PGSSLMODE
    logger.debug("Injected fresh Lakebase OAuth token for %s", PGHOST)


# ── Start agent-plane ─────────────────────────────────────
#
# Import AFTER the do_connect hook is registered so agent-plane's
# engine creation picks it up.

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import uvicorn  # noqa: E402

from agent_plane.runtime import init as init_runtime  # noqa: E402
from agent_plane.runtime.agent_cache import AgentCache  # noqa: E402
from agent_plane.runtime.caps import RuntimeCaps  # noqa: E402
from agent_plane.server.app import create_app  # noqa: E402
from agent_plane.stores.agent_store.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyAgentStore,
)
from agent_plane.stores.artifact_store.databricks_volumes import (  # noqa: E402
    DatabricksVolumesArtifactStore,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyConversationStore,
)
from agent_plane.stores.file_store.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyFileStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyTaskStore,
)

DB_URI = f"postgresql+psycopg://{PGUSER}@{PGHOST}:{PGPORT}/{PGDATABASE}"
ARTIFACT_URI = f"dbfs:{VOLUME_PATH}"
# Local cache directory for extracted agent bundles. Ephemeral —
# the artifact store (UC Volumes) is the durable backing store.
CACHE_DIR = Path(tempfile.mkdtemp(prefix="ap_cache_"))

# Create stores
agent_store = SqlAlchemyAgentStore(DB_URI)
file_store = SqlAlchemyFileStore(DB_URI)
task_store = SqlAlchemyTaskStore(DB_URI)
conversation_store = SqlAlchemyConversationStore(DB_URI)
artifact_store = DatabricksVolumesArtifactStore(ARTIFACT_URI)

# Initialize runtime
init_runtime(
    agent_cache=AgentCache(artifact_store=artifact_store, cache_dir=CACHE_DIR),
    caps=RuntimeCaps(),
    agent_store=agent_store,
    file_store=file_store,
    task_store=task_store,
    conversation_store=conversation_store,
    artifact_store=artifact_store,
)

# Create FastAPI app
app = create_app(
    agent_store=agent_store,
    file_store=file_store,
    task_store=task_store,
    conversation_store=conversation_store,
    artifact_store=artifact_store,
)

if __name__ == "__main__":
    logger.info(
        "Starting agent-plane on 0.0.0.0:%d (Lakebase + UC Volumes)",
        PORT,
    )
    uvicorn.run(app, host="0.0.0.0", port=PORT)
