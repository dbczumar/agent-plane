"""Shared test fixtures — in-memory SQLite database with schema applied."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from agent_plane.db.db_models import Base
from agent_plane.db.utils import _engine_cache, _engine_lock

_TEST_DB_URI = "sqlite://"


@pytest.fixture()
def db_uri() -> str:
    """
    Return a test database URI and register a fresh in-memory engine
    in the engine cache so stores find it. Cleaned up after each test.
    """
    engine = create_engine(_TEST_DB_URI)
    Base.metadata.create_all(engine)

    with _engine_lock:
        _engine_cache[_TEST_DB_URI] = engine

    yield _TEST_DB_URI

    with _engine_lock:
        _engine_cache.pop(_TEST_DB_URI, None)
    engine.dispose()
