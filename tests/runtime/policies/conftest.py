"""
Fixtures for runtime policy tests.

Re-imports the `conversation_store` fixture (and its
underlying `db_uri`) from `tests/stores/conftest.py` so tests
here can exercise the real persistence layer without
duplicating setup.
"""

from __future__ import annotations

import pytest

from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


@pytest.fixture()
def conversation_store(db_uri: str) -> SqlAlchemyConversationStore:
    """
    Conversation store backed by a per-test SQLite DB.

    Mirrors the fixture in `tests/stores/conftest.py` — kept
    local so runtime policy tests can evolve their dependency
    surface without coupling to store-test internals.
    """
    return SqlAlchemyConversationStore(db_uri)
