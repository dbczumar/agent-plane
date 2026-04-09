"""Databricks Apps entry point for agent-plane.

Starts agent-plane with Lakebase (managed PostgreSQL) as the
database and UC Volumes as the artifact store.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
logger = logging.getLogger("agent-plane-app")

try:
    import sqlalchemy
    from databricks.sdk import WorkspaceClient

    # ── Configuration ──────────────────────────────────────────

    LAKEBASE_ENDPOINT = os.environ.get("AP_LAKEBASE_ENDPOINT", "")
    VOLUME_PATH = os.environ.get("AP_ARTIFACT_VOLUME_PATH", "")
    PORT = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))

    PGHOST = os.environ.get("PGHOST", "")
    PGPORT = os.environ.get("PGPORT", "5432")
    PGDATABASE = os.environ.get("PGDATABASE", "")
    PGUSER = os.environ.get("PGUSER", "")
    PGSSLMODE = os.environ.get("PGSSLMODE", "require")
    POOL_RECYCLE_SECONDS = int(
        os.environ.get("AP_POOL_RECYCLE_SECONDS", "300")
    )

    logger.info("Config: PGHOST=%s PGDATABASE=%s PGUSER=%s VOLUME=%s PORT=%d",
                PGHOST, PGDATABASE, PGUSER, VOLUME_PATH, PORT)

    if not PGHOST:
        logger.error("PGHOST not set — is the postgres resource configured?")
    if not VOLUME_PATH:
        logger.error("AP_ARTIFACT_VOLUME_PATH not set — is the volume resource configured?")

    # ── Lakebase token injection ──────────────────────────────

    _workspace_client = WorkspaceClient()

    @sqlalchemy.event.listens_for(sqlalchemy.engine.Engine, "do_connect")
    def _inject_lakebase_credentials(dialect, conn_rec, cargs, cparams):
        if cparams.get("host") != PGHOST:
            return
        cparams["password"] = _workspace_client.postgres.generate_database_credential(
            endpoint=LAKEBASE_ENDPOINT,
        ).token
        cparams["sslmode"] = PGSSLMODE

    # ── Start agent-plane ─────────────────────────────────────

    import tempfile
    from pathlib import Path

    import uvicorn

    from agent_plane.runtime import init as init_runtime
    from agent_plane.runtime.agent_cache import AgentCache
    from agent_plane.runtime.caps import RuntimeCaps
    from agent_plane.server.app import create_app
    from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
    from agent_plane.stores.artifact_store.databricks_volumes import (
        DatabricksVolumesArtifactStore,
    )
    from agent_plane.stores.conversation_store.sqlalchemy_store import (
        SqlAlchemyConversationStore,
    )
    from agent_plane.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
    from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore

    DB_URI = f"postgresql+psycopg://{PGUSER}@{PGHOST}:{PGPORT}/{PGDATABASE}"
    ARTIFACT_URI = f"dbfs:{VOLUME_PATH}"
    CACHE_DIR = Path(tempfile.mkdtemp(prefix="ap_cache_"))

    logger.info("DB_URI: %s", DB_URI[:80])
    logger.info("ARTIFACT_URI: %s", ARTIFACT_URI)

    agent_store = SqlAlchemyAgentStore(DB_URI)
    file_store = SqlAlchemyFileStore(DB_URI)
    task_store = SqlAlchemyTaskStore(DB_URI)
    conversation_store = SqlAlchemyConversationStore(DB_URI)
    artifact_store = DatabricksVolumesArtifactStore(ARTIFACT_URI)

    init_runtime(
        agent_cache=AgentCache(artifact_store=artifact_store, cache_dir=CACHE_DIR),
        caps=RuntimeCaps(),
        agent_store=agent_store,
        file_store=file_store,
        task_store=task_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
    )

    app = create_app(
        agent_store=agent_store,
        file_store=file_store,
        task_store=task_store,
        conversation_store=conversation_store,
        artifact_store=artifact_store,
    )

    if __name__ == "__main__":
        logger.info("Starting agent-plane on 0.0.0.0:%d", PORT)
        uvicorn.run(app, host="0.0.0.0", port=PORT)

except Exception:
    logger.error("FATAL: agent-plane failed to start:\n%s", traceback.format_exc())
    # Keep the process alive briefly so logs can be captured
    import time
    time.sleep(30)
    sys.exit(1)
