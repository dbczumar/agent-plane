"""Database utilities — engine caching, session management, helpers."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from agent_plane.entities import NewConversationItem

# A callable that returns a context manager yielding a Session.
ManagedSessionMaker = Callable[[], AbstractContextManager[Session]]

# ── Engine caching ─────────────────────────────────────

_engine_cache: dict[str, Engine] = {}
_engine_lock = threading.Lock()


def get_or_create_engine(db_uri: str) -> Engine:
    """Return a cached engine for the given URI, creating one if needed."""
    if db_uri not in _engine_cache:
        with _engine_lock:
            if db_uri not in _engine_cache:
                _engine_cache[db_uri] = create_engine(db_uri)
    return _engine_cache[db_uri]


def clear_engine_cache() -> None:
    """Clear the engine cache. Used by tests."""
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
    Sessions auto-commit on success, auto-rollback on failure.
    SQLite gets PRAGMA foreign_keys and busy_timeout.
    """
    factory = sessionmaker(bind=engine)
    is_sqlite = engine.dialect.name == "sqlite"

    @contextmanager
    def managed_session() -> Iterator[Session]:
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
}


def generate_agent_id() -> str:
    return f"ag_{uuid.uuid4().hex}"


def generate_file_id() -> str:
    return f"file_{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex}"


def generate_task_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def generate_item_id(item_type: str) -> str:
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
    """Create the FTS5 virtual table if on SQLite. Idempotent."""
    if engine.dialect.name == "sqlite":
        with engine.connect() as conn:
            conn.execute(_CREATE_FTS)
            conn.commit()


def insert_fts(session: Session, item_id: str, conversation_id: str, search_text: str) -> None:
    """Dual-write a row into the FTS5 table (SQLite only)."""
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
    """Remove all FTS rows for a conversation (SQLite only)."""
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
    (content, name, arguments, output, summary) are guaranteed present.
    We use direct dict access to fail loud if that assumption is ever
    violated.
    """
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
    return ""


# ── Timestamp ──────────────────────────────────────────


def now_epoch() -> int:
    """Current time as Unix epoch seconds."""
    return int(time.time())
