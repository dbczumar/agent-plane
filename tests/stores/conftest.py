"""Shared fixtures for store tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_plane.runtime import init as init_runtime
from agent_plane.runtime.agent_cache import AgentCache
from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore


@pytest.fixture()
def agent_store(db_uri: str) -> SqlAlchemyAgentStore:
    return SqlAlchemyAgentStore(db_uri)


@pytest.fixture()
def conversation_store(db_uri: str) -> SqlAlchemyConversationStore:
    return SqlAlchemyConversationStore(db_uri)


@pytest.fixture()
def artifact_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture()
def task_store(
    db_uri: str,
    agent_store: SqlAlchemyAgentStore,
    conversation_store: SqlAlchemyConversationStore,
    artifact_store: LocalArtifactStore,
    tmp_path: Path,
) -> SqlAlchemyTaskStore:
    """
    Task store with runtime initialized. Required because task_store.start()
    launches the real agent workflow, which needs runtime getters.
    """
    agent_cache = AgentCache(
        artifact_store=artifact_store,
        cache_dir=tmp_path / ".cache",
    )
    init_runtime(
        conversation_store=conversation_store,
        task_store=SqlAlchemyTaskStore(db_uri),
        agent_store=agent_store,
        agent_cache=agent_cache,
    )
    return SqlAlchemyTaskStore(db_uri)
