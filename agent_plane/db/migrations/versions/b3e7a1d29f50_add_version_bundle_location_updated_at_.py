"""add version, bundle_location, updated_at to agents

Revision ID: b3e7a1d29f50
Revises: 43fb65b29464
Create Date: 2026-04-14 17:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3e7a1d29f50"
down_revision: str | None = "43fb65b29464"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("agents") as batch_op:
        batch_op.add_column(
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        )
        # Backfill existing rows: bundle was stored under the agent id
        # as the artifact key (no version prefix).
        batch_op.add_column(
            sa.Column("bundle_location", sa.String(length=512), nullable=True),
        )
        batch_op.add_column(
            sa.Column("updated_at", sa.Integer(), nullable=True),
        )

    # Backfill bundle_location for existing agents: the old artifact
    # key format was just the agent id.
    op.execute(
        sa.text("UPDATE agents SET bundle_location = id WHERE bundle_location IS NULL")
    )

    # Now make bundle_location NOT NULL.
    with op.batch_alter_table("agents") as batch_op:
        batch_op.alter_column("bundle_location", nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("agents") as batch_op:
        batch_op.drop_column("updated_at")
        batch_op.drop_column("bundle_location")
        batch_op.drop_column("version")
