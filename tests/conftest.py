"""Shared test fixtures — file-based SQLite database with schema applied."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from agent_plane.db.db_models import Base
from agent_plane.db.utils import _engine_cache, _engine_lock
from agent_plane.runtime.durability import destroy_dbos


@pytest.fixture()
def db_uri(tmp_path: Path) -> str:
    """
    Return a test database URI backed by a file in tmp_path, and
    register a fresh engine in the engine cache so stores find it.
    File-based (not in-memory) because DBOS needs a real file to
    create its system tables. Cleaned up after each test.
    """
    db_path = tmp_path / "test.db"
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri)
    Base.metadata.create_all(engine)

    with _engine_lock:
        _engine_cache[uri] = engine

    yield uri

    # Tear down DBOS singleton so the next test can re-initialize
    # with a fresh database.
    destroy_dbos()

    with _engine_lock:
        _engine_cache.pop(uri, None)
    engine.dispose()
