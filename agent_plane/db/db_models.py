"""SQLAlchemy table definitions for the agent-plane database."""

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all agent-plane tables."""


class SqlAgent(Base):
    """
    SQLAlchemy model for the ``agents`` table.

    Each row represents a registered agent in the system.

    :param id: Unique agent identifier, e.g. ``"ag_0f1a2b3c..."``.
    :param created_at: Unix epoch seconds when the agent was created.
    :param name: Human-readable agent name. Must be unique across all
        agents, max 256 characters.
    :param description: Optional free-text description of the agent's
        purpose. ``None`` when not provided.
    """

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_agents_created_at", "created_at"),)


class SqlFile(Base):
    """
    SQLAlchemy model for the ``files`` table.

    Each row represents an uploaded file tracked by the system.

    :param id: Unique file identifier, e.g. ``"file_a1b2c3d4..."``.
    :param created_at: Unix epoch seconds when the file record was
        created.
    :param filename: Original filename as provided by the uploader,
        max 512 characters. e.g. ``"report.pdf"``.
    :param bytes: Size of the file in bytes.
    :param content_type: MIME type of the file, e.g.
        ``"application/pdf"``. ``None`` when not provided.
    """

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(512))
    bytes: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str | None] = mapped_column(String(256), nullable=True)

    __table_args__ = (Index("ix_files_created_at", "created_at"),)


class SqlConversation(Base):
    """
    SQLAlchemy model for the ``conversations`` table.

    Each row represents a conversation thread that contains one or
    more conversation items.

    :param id: Unique conversation identifier, e.g.
        ``"conv_e4f5a6b7..."``.
    :param created_at: Unix epoch seconds when the conversation was
        created.
    :param title: Optional human-readable title for the conversation.
        ``None`` when not provided.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_conversations_created_at", "created_at"),)


class SqlTask(Base):
    """
    SQLAlchemy model for the ``tasks`` table.

    Each row represents a task (also referred to as a "response")
    assigned to an agent within a conversation.

    :param id: Unique task identifier, e.g. ``"resp_d8e9f0a1..."``.
    :param agent_id: Foreign key to :class:`SqlAgent.id`. Cascades
        on delete.
    :param conversation_id: Foreign key to
        :class:`SqlConversation.id`. Cascades on delete.
    :param previous_response_id: ID of the preceding task in the
        conversation, or ``None`` if this is the first task.
    :param created_at: Unix epoch seconds when the task was created.
    :param inbox_closed: Whether the agent's inbox for this task is
        closed. Defaults to ``False``.
    :param agent_name: Denormalized copy of the agent's name at
        task-creation time.
    :param background: Whether this task runs in the background.
        Defaults to ``False``.
    """

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), ForeignKey("agents.id", ondelete="CASCADE"))
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    previous_response_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[int] = mapped_column(Integer)
    inbox_closed: Mapped[bool] = mapped_column(Boolean, default=False)
    agent_name: Mapped[str] = mapped_column(String(256))
    background: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_tasks_conversation_id", "conversation_id"),
        Index("ix_tasks_agent_id", "agent_id"),
        Index("ix_tasks_created_at", "created_at"),
    )


class SqlConversationItem(Base):
    """
    SQLAlchemy model for the ``conversation_items`` table.

    Each row represents a single item (message, function call,
    function call output, or reasoning block) within a conversation.

    :param id: Unique item identifier with a type-based prefix,
        e.g. ``"msg_a1b2c3..."``, ``"fc_d4e5f6..."``.
    :param conversation_id: Foreign key to
        :class:`SqlConversation.id`. Cascades on delete.
    :param response_id: The task/response ID this item belongs to,
        e.g. ``"resp_d8e9f0a1..."``.
    :param created_at: Unix epoch seconds when the item was created.
    :param status: Item status string. Defaults to ``"completed"``.
    :param position: Zero-based ordering index within the
        conversation. Used for deterministic item ordering.
    :param type: Item type discriminator, one of ``"message"``,
        ``"function_call"``, ``"function_call_output"``,
        ``"reasoning"``.
    :param data: JSON-serialized item payload. Structure varies by
        ``type``.
    :param search_text: Plain-text extraction of ``data`` used for
        full-text search indexing.
    """

    __tablename__ = "conversation_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE")
    )
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
            unique=True,
        ),
        Index("ix_conversation_items_response_id", "response_id"),
    )
