"""Tests for database engine pool configuration (agent_plane/db/utils.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.db.utils import clear_engine_cache, get_or_create_engine


@pytest.fixture(autouse=True)
def _clean_engine_cache() -> None:
    """
    Clear the module-level engine cache before each test
    so that each test creates a fresh engine.
    """
    clear_engine_cache()


def test_non_sqlite_engine_has_pool_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Non-SQLite engines must be created with pool_pre_ping=True and
    pool_recycle=1800 to prevent stale/dead connections.
    """
    captured_kwargs: dict[str, Any] = {}
    mock_engine = MagicMock()

    def _capturing_create_engine(uri: str, **kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return mock_engine

    monkeypatch.setattr(
        "agent_plane.db.utils.create_engine",
        _capturing_create_engine,
    )
    # Skip migrations -- we only care about engine creation kwargs.
    monkeypatch.setattr(
        "agent_plane.db.utils._run_migrations",
        lambda engine, db_uri: None,
    )

    get_or_create_engine("postgresql://user:pass@localhost/testdb")

    # pool_pre_ping=True prevents "server has gone away" errors
    # after idle periods. Failure means dead connections won't be
    # detected before checkout, causing intermittent query failures.
    assert captured_kwargs.get("pool_pre_ping") is True

    # pool_recycle=1800 (30 min) prevents stale connections when
    # the database server restarts or closes idle connections.
    # Failure means connections could persist indefinitely and break.
    assert captured_kwargs.get("pool_recycle") == 1800


def test_sqlite_engine_no_pool_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    SQLite engines must NOT receive pool settings -- SQLite uses a
    single connection with StaticPool and pooling options would be
    meaningless or harmful.
    """
    captured_kwargs: dict[str, Any] = {}
    mock_engine = MagicMock()

    def _capturing_create_engine(uri: str, **kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return mock_engine

    monkeypatch.setattr(
        "agent_plane.db.utils.create_engine",
        _capturing_create_engine,
    )
    monkeypatch.setattr(
        "agent_plane.db.utils._run_migrations",
        lambda engine, db_uri: None,
    )

    get_or_create_engine("sqlite:///test.db")

    # SQLite uses StaticPool (single connection, no pooling).
    # pool_pre_ping would be wasteful, and pool_recycle is
    # meaningless. Failure would mean SQLite engines get
    # pool options intended for multi-connection databases.
    assert "pool_pre_ping" not in captured_kwargs
    assert "pool_recycle" not in captured_kwargs
