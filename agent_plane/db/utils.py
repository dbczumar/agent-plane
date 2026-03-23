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
    prefix = _ITEM_TYPE_PREFIX.get(item_type, "item_")
    return f"{prefix}{uuid.uuid4().hex}"


# ── Search text extraction ─────────────────────────────


def extract_search_text(item: NewConversationItem) -> str:
    """
    Extract plain text for FTS from an item's data, per DBSPEC.
    """
    data = item.data.model_dump()
    if item.type == "message":
        content: list[dict[str, str]] = data.get("content", [])
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        )
    if item.type == "function_call":
        return f"{data.get('name', '')} {data.get('arguments', '')}"
    if item.type == "function_call_output":
        return str(data.get("output", ""))
    if item.type == "reasoning":
        summary: list[dict[str, str]] = data.get("summary", [])
        return " ".join(
            block.get("text", "")
            for block in summary
            if isinstance(block, dict) and block.get("text")
        )
    return ""


# ── Timestamp ──────────────────────────────────────────


def now_epoch() -> int:
    """Current time as Unix epoch seconds."""
    return int(time.time())
