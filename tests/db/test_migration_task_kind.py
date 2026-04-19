"""Tests for the ``tasks.kind`` column added to the initial schema migration."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from agent_plane.db.utils import clear_engine_cache, get_or_create_engine


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    """
    Spin up a fresh SQLite DB with the full alembic chain applied.

    :param tmp_path: pytest-managed scratch dir.
    :returns: A SQLAlchemy ``Engine`` pointed at an empty SQLite
        file with the full schema applied.
    """
    db_path = tmp_path / "test.db"
    uri = f"sqlite:///{db_path}"
    # get_or_create_engine runs migrations internally on first call
    # for a given URI.
    engine = get_or_create_engine(uri)
    try:
        yield engine
    finally:
        # Drop the engine from the module-level cache so a fresh
        # fixture in another test gets its own DB.
        clear_engine_cache()


def test_migration_adds_kind_column_with_default_backfill(
    db_engine: Engine,
) -> None:
    """
    A fresh DB has the ``tasks.kind`` column. New rows inserted
    WITHOUT specifying ``kind`` get the server-side default
    ``"agent_task"`` — this is the behavior pre-existing rows in
    older databases would receive when the migration runs.
    """
    with db_engine.connect() as conn:
        # Verify the column exists with the right type.
        cols = sa.inspect(db_engine).get_columns("tasks")
        kind_cols = [c for c in cols if c["name"] == "kind"]
        # Exactly one kind column; if zero, the migration didn't run;
        # if more than one, the schema is inconsistent.
        assert len(kind_cols) == 1, (
            f"Expected exactly one 'kind' column on tasks, got {len(kind_cols)}. "
            f"If 0, the migration didn't include the column."
        )
        kind_col = kind_cols[0]
        assert not kind_col["nullable"], "tasks.kind must be NOT NULL"

        # Insert a row without specifying kind to verify the server
        # default backfills it. Use raw SQL to bypass the ORM's own
        # default, simulating what happens to pre-existing rows.
        conn.execute(
            sa.text(
                """
                INSERT INTO agents (id, created_at, name, bundle_location, version)
                VALUES ('ag_test', 0, 'test-agent', 'loc', 1)
                """
            )
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO conversations (id, created_at, updated_at, kind)
                VALUES ('conv_test', 0, 0, 'default')
                """
            )
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO tasks (
                    id, agent_id, conversation_id, created_at,
                    inbox_closed, agent_name, background
                ) VALUES (
                    'tsk_test', 'ag_test', 'conv_test', 0,
                    0, 'test-agent', 0
                )
                """
            )
        )
        conn.commit()

        row = conn.execute(sa.text("SELECT kind FROM tasks WHERE id = 'tsk_test'")).one()
        # Server default applies when the INSERT omits 'kind'.
        # If this returns NULL, the server_default isn't set — meaning
        # pre-existing rows after migration would also be NULL, breaking
        # the backfill invariant.
        assert row[0] == "agent_task", (
            f"Expected default 'agent_task' for missing kind, got {row[0]!r}. "
            f"If NULL, the server_default isn't applied — existing rows "
            f"would not be backfilled."
        )


def test_migration_kind_check_constraint_rejects_invalid(
    db_engine: Engine,
) -> None:
    """
    The CHECK constraint rejects values outside the documented set.

    Invalid kinds would let the LLM see misclassified tasks via
    ``list_tasks`` / ``check_task`` — the constraint is the
    structural defense.
    """
    with db_engine.connect() as conn:
        conn.execute(
            sa.text(
                """
                INSERT INTO agents (id, created_at, name, bundle_location, version)
                VALUES ('ag_x', 0, 'x', 'loc', 1)
                """
            )
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO conversations (id, created_at, updated_at, kind)
                VALUES ('conv_x', 0, 0, 'default')
                """
            )
        )
        conn.commit()

        with pytest.raises(sa.exc.IntegrityError):
            # 'gibberish' isn't in the documented set; the CHECK
            # constraint must reject it. If this insert succeeds,
            # the constraint isn't enforced and bad data could leak.
            conn.execute(
                sa.text(
                    """
                    INSERT INTO tasks (
                        id, agent_id, conversation_id, created_at,
                        inbox_closed, agent_name, background, kind
                    ) VALUES (
                        'tsk_bad', 'ag_x', 'conv_x', 0,
                        0, 'x', 0, 'gibberish'
                    )
                    """
                )
            )
            conn.commit()


@pytest.mark.parametrize(
    "kind",
    ["agent_task", "tool", "sub_agent", "client_tool"],
)
def test_migration_kind_check_constraint_accepts_documented_values(
    db_engine: Engine,
    kind: str,
) -> None:
    """All four documented kind values are accepted by the constraint."""
    with db_engine.connect() as conn:
        conn.execute(
            sa.text(
                """
                INSERT INTO agents (id, created_at, name, bundle_location, version)
                VALUES (:id, 0, :name, 'loc', 1)
                """
            ),
            {"id": f"ag_{kind}", "name": f"agent-{kind}"},
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO conversations (id, created_at, updated_at, kind)
                VALUES (:id, 0, 0, 'default')
                """
            ),
            {"id": f"conv_{kind}"},
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO tasks (
                    id, agent_id, conversation_id, created_at,
                    inbox_closed, agent_name, background, kind
                ) VALUES (
                    :id, :agent_id, :conv_id, 0,
                    0, :name, 0, :kind
                )
                """
            ),
            {
                "id": f"tsk_{kind}",
                "agent_id": f"ag_{kind}",
                "conv_id": f"conv_{kind}",
                "name": f"agent-{kind}",
                "kind": kind,
            },
        )
        conn.commit()

        # Verify the row stored the kind we passed.
        row = conn.execute(
            sa.text("SELECT kind FROM tasks WHERE id = :id"),
            {"id": f"tsk_{kind}"},
        ).one()
        assert row[0] == kind


def test_migration_kind_index_exists(db_engine: Engine) -> None:
    """
    An index on ``tasks.kind`` exists so ``list_tasks`` filtering
    by kind doesn't scan the full table.
    """
    indexes = sa.inspect(db_engine).get_indexes("tasks")
    kind_indexes = [i for i in indexes if i["column_names"] == ["kind"]]
    assert len(kind_indexes) == 1, (
        f"Expected one index on tasks.kind for filter performance, "
        f"got {len(kind_indexes)}. Indexes present: "
        f"{[i['name'] for i in indexes]}"
    )
