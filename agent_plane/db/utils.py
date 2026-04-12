"""Database utilities — engine caching, session management, helpers."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from agent_plane.entities import NewConversationItem

_logger = logging.getLogger(__name__)

# A callable that returns a context manager yielding a Session.
ManagedSessionMaker = Callable[[], AbstractContextManager[Session]]

# ── Engine caching ─────────────────────────────────────

_engine_cache: dict[str, Engine] = {}
_engine_lock = threading.Lock()


def _create_engine(db_uri: str) -> Engine:
    """
    Create a SQLAlchemy engine with connection pool configuration.

    SQLite uses ``StaticPool`` (single connection, no pooling) with
    ``busy_timeout`` for write contention. Non-SQLite databases use
    connection pooling with ``pool_pre_ping`` to verify connections
    before use.

    :param db_uri: SQLAlchemy database connection string, e.g.
        ``"sqlite:///mydb.db"`` or
        ``"postgresql://user:pass@host/dbname"``.
    :returns: A configured :class:`~sqlalchemy.engine.Engine`.
    """
    is_sqlite = db_uri.startswith("sqlite")
    if is_sqlite:
        return create_engine(db_uri)
    return create_engine(
        db_uri,
        # Verify connections are alive before checking them out
        # from the pool. Prevents "server has gone away" errors
        # after idle periods.
        pool_pre_ping=True,
        # Recycle connections older than 30 minutes. Prevents
        # stale connections when the database server restarts
        # or closes idle connections.
        pool_recycle=1800,
    )


def get_or_create_engine(db_uri: str) -> Engine:
    """
    Return a cached engine for the given URI, creating one if needed.

    On first creation, runs Alembic migrations to ensure the schema is
    up to date. Subsequent calls with the same URI return the cached
    engine without re-running migrations.

    :param db_uri: SQLAlchemy database connection string, e.g.
        ``"sqlite:///mydb.db"`` or
        ``"postgresql://user:pass@host/dbname"``.
    :returns: A :class:`~sqlalchemy.engine.Engine` for the given URI.
    """
    if db_uri not in _engine_cache:
        with _engine_lock:
            if db_uri not in _engine_cache:
                engine = _create_engine(db_uri)
                _run_migrations(engine, db_uri)
                _engine_cache[db_uri] = engine
    return _engine_cache[db_uri]


def _run_migrations(engine: Engine, db_uri: str) -> None:
    """
    Run Alembic migrations against the database if the schema is not
    current. Checks whether our application tables exist first to
    avoid unnecessary work on already-initialized databases.

    :param engine: The SQLAlchemy engine to inspect and migrate.
    :param db_uri: Database connection string forwarded to Alembic's
        ``sqlalchemy.url`` config option, e.g.
        ``"sqlite:///mydb.db"``.
    """
    from alembic import command
    from alembic.config import Config

    from agent_plane.db.db_models import Base

    expected_tables = {table.name for table in Base.metadata.sorted_tables}
    actual_tables = set(inspect(engine).get_table_names())
    if expected_tables.issubset(actual_tables):
        return

    _logger.info("Initializing application database tables...")
    alembic_ini = Path(__file__).parent / "alembic.ini"
    config = Config(str(alembic_ini))
    config.set_main_option("sqlalchemy.url", db_uri)
    # Pass a shared connection so Alembic operates within the same
    # engine (required for SQLite in-memory databases, and avoids
    # creating a second connection pool).
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


def clear_engine_cache() -> None:
    """
    Dispose of all cached engines and clear the engine cache.

    Intended for test teardown to ensure a fresh database state
    between test runs.
    """
    with _engine_lock:
        for engine in _engine_cache.values():
            engine.dispose()
        _engine_cache.clear()


# ── Managed session ────────────────────────────────────


def make_managed_session_maker(
    engine: Engine,
) -> ManagedSessionMaker:
    """
    Create a context-manager factory for database sessions.

    Sessions auto-commit on success and auto-rollback on failure.
    When the underlying dialect is SQLite, each session
    additionally enables ``PRAGMA foreign_keys`` and sets a 20-second
    ``busy_timeout``.

    :param engine: The SQLAlchemy engine to bind sessions to.
    :returns: A callable that, when invoked, returns a context
        manager yielding a :class:`~sqlalchemy.orm.Session`.
    """
    factory = sessionmaker(bind=engine)
    is_sqlite = engine.dialect.name == "sqlite"

    @contextmanager
    def managed_session() -> Iterator[Session]:
        """
        Yield a managed :class:`~sqlalchemy.orm.Session`.

        Commits on clean exit, rolls back on exception. For SQLite
        backends, enables foreign key enforcement and sets a
        busy timeout before yielding.
        """
        with factory() as session:
            try:
                if is_sqlite:
                    session.execute(text("PRAGMA foreign_keys = ON"))
                    session.execute(text("PRAGMA busy_timeout = 20000"))  # 20s
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    return managed_session


# ── ID generation ──────────────────────────────────────

_ITEM_TYPE_PREFIX: dict[str, str] = {
    "message": "msg_",
    "function_call": "fc_",
    "function_call_output": "fco_",
    "reasoning": "rs_",
    "compaction": "cmp_",
    "native_tool": "nt_",
}


def generate_agent_id() -> str:
    """
    Generate a unique agent identifier.

    :returns: A string of the form ``"ag_<32-char hex>"``,
        e.g. ``"ag_0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c"``.
    """
    return f"ag_{uuid.uuid4().hex}"


def generate_file_id() -> str:
    """
    Generate a unique file identifier.

    :returns: A string of the form ``"file_<32-char hex>"``,
        e.g. ``"file_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"``.
    """
    return f"file_{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    """
    Generate a unique conversation identifier.

    :returns: A string of the form ``"conv_<32-char hex>"``,
        e.g. ``"conv_e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9"``.
    """
    return f"conv_{uuid.uuid4().hex}"


def generate_task_id() -> str:
    """
    Generate a unique task (response) identifier.

    :returns: A string of the form ``"resp_<32-char hex>"``,
        e.g. ``"resp_d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3"``.
    """
    return f"resp_{uuid.uuid4().hex}"


def generate_item_id(item_type: str) -> str:
    """
    Generate a unique conversation-item identifier.

    The prefix is determined by the item type:

    - ``"message"`` -> ``"msg_"``
    - ``"function_call"`` -> ``"fc_"``
    - ``"function_call_output"`` -> ``"fco_"``
    - ``"reasoning"`` -> ``"rs_"``
    - ``"compaction"`` -> ``"cmp_"``

    :param item_type: One of ``"message"``, ``"function_call"``,
        ``"function_call_output"``, ``"reasoning"``, or
        ``"compaction"``.
    :returns: A prefixed identifier, e.g. ``"msg_a1b2c3d4..."``.
    :raises ValueError: If *item_type* is not a recognised type.
    """
    prefix = _ITEM_TYPE_PREFIX.get(item_type)
    if prefix is None:
        raise ValueError(f"unknown item type: {item_type!r}")
    return f"{prefix}{uuid.uuid4().hex}"


# ── FTS (SQLite FTS5) ─────────────────────────────────

_FTS_TABLE = "conversation_items_fts"

_CREATE_FTS = text(
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5("
    "item_id UNINDEXED, conversation_id UNINDEXED, search_text)"
)


def ensure_fts_table(engine: Engine) -> None:
    """
    Create the FTS5 virtual table if on SQLite. Idempotent.

    On non-SQLite dialects this is a no-op.

    :param engine: The SQLAlchemy engine whose dialect is inspected.
        If SQLite, the ``conversation_items_fts`` virtual table is
        created (if it does not already exist).
    """
    if engine.dialect.name == "sqlite":
        with engine.connect() as conn:
            conn.execute(_CREATE_FTS)
            conn.commit()


def insert_fts(
    session: Session,
    item_id: str,
    conversation_id: str,
    search_text: str,
) -> None:
    """
    Dual-write a row into the FTS5 table (SQLite only).

    On non-SQLite dialects this is a no-op.

    :param session: An active SQLAlchemy session. Its bound engine's
        dialect is checked to decide whether to write.
    :param item_id: The conversation-item ID to index, e.g.
        ``"msg_a1b2c3d4..."``.
    :param conversation_id: The parent conversation ID, e.g.
        ``"conv_e4f5a6b7..."``.
    :param search_text: Plain-text content to store in the FTS
        index for this item.
    """
    if session.bind and session.bind.dialect.name == "sqlite":
        session.execute(
            text(
                f"INSERT INTO {_FTS_TABLE}"
                "(item_id, conversation_id, search_text) "
                "VALUES (:item_id, :cid, :st)"
            ),
            {"item_id": item_id, "cid": conversation_id, "st": search_text},
        )


def delete_fts_by_conversation(session: Session, conversation_id: str) -> None:
    """
    Remove all FTS rows for a conversation (SQLite only).

    On non-SQLite dialects this is a no-op.

    :param session: An active SQLAlchemy session. Its bound engine's
        dialect is checked to decide whether to delete.
    :param conversation_id: The conversation whose FTS rows should be
        removed, e.g. ``"conv_e4f5a6b7..."``.
    """
    if session.bind and session.bind.dialect.name == "sqlite":
        session.execute(
            text(f"DELETE FROM {_FTS_TABLE} WHERE conversation_id = :cid"),
            {"cid": conversation_id},
        )


# ── Search text extraction ─────────────────────────────


def extract_search_text(item: NewConversationItem) -> str:
    """
    Extract plain text for FTS from an item's data, per DBSPEC.

    The item has already been Pydantic-validated, so required fields
    (content, name, arguments, output, summary) are guaranteed
    present. We use direct dict access to fail loud if that
    assumption is ever violated.

    Content/summary blocks are heterogeneous (text, image, etc.)
    so we filter to only text-bearing blocks via ``.get("text")``.

    :param item: A Pydantic-validated conversation item whose
        ``type`` is one of ``"message"``, ``"function_call"``,
        ``"function_call_output"``, ``"reasoning"``, or
        ``"compaction"``.
    :returns: A single plain-text string suitable for FTS indexing.
    :raises ValueError: If *item.type* is not a recognised type.
    """
    from agent_plane.entities.conversation import CompactionData

    data = item.data.model_dump()
    if item.type == "message":
        return " ".join(
            block["text"]
            for block in data["content"]
            if isinstance(block, dict) and block.get("text")
        )
    if item.type == "function_call":
        return f"{data['name']} {data['arguments']}"
    if item.type == "function_call_output":
        return str(data["output"])
    if item.type == "reasoning":
        return " ".join(
            block["text"]
            for block in data["summary"]
            if isinstance(block, dict) and block.get("text")
        )
    if item.type == "compaction":
        assert isinstance(item.data, CompactionData)
        return item.data.summary
    if item.type == "native_tool":
        # Native tool items are opaque provider dicts — no
        # meaningful text to index for search.
        return ""
    raise ValueError(f"unknown item type: {item.type!r}")


# ── Timestamp ──────────────────────────────────────────


def now_epoch() -> int:
    """
    Return the current time as Unix epoch seconds (integer).

    :returns: Seconds since 1970-01-01 00:00:00 UTC, truncated to
        an integer.
    """
    return int(time.time())
