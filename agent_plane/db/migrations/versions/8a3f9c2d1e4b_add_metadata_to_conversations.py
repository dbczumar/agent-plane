"""add metadata column to conversations

Revision ID: 8a3f9c2d1e4b
Revises: 43fb65b29464
Create Date: 2026-04-20 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8a3f9c2d1e4b"
down_revision: str | None = "43fb65b29464"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("metadata", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversations", "metadata")
