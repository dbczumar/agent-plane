"""SQLAlchemy table definitions for the agent-plane database."""

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all agent-plane tables."""


class SqlAgent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_agents_created_at", "created_at"),)


class SqlFile(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(512))
    bytes: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str | None] = mapped_column(String(256), nullable=True)

    __table_args__ = (Index("ix_files_created_at", "created_at"),)


class SqlConversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_conversations_created_at", "created_at"),)


class SqlTask(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), ForeignKey("agents.id"))
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id"))
    previous_response_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[int] = mapped_column(Integer)
    inbox_closed: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_tasks_conversation_id", "conversation_id"),
        Index("ix_tasks_agent_id", "agent_id"),
        Index("ix_tasks_created_at", "created_at"),
    )


class SqlConversationItem(Base):
    __tablename__ = "conversation_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id"))
    response_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    position: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32))
    data: Mapped[str] = mapped_column(Text)
    search_text: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        Index(
            "ix_conversation_items_conversation_id_position",
            "conversation_id",
            "position",
        ),
        Index("ix_conversation_items_response_id", "response_id"),
    )
