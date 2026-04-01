"""SQLAlchemy table definitions for the agent-plane database."""

from __future__ import annotations

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, Integer, String, Text
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
    :param kind: Conversation type. ``"default"`` for user-initiated,
        ``"sub_agent"`` for sub-agent execution conversations.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(String(32), default="default")

    __table_args__ = (
        CheckConstraint("kind IN ('default', 'sub_agent')", name="ck_conversations_kind"),
        Index("ix_conversations_created_at", "created_at"),
        Index("ix_conversations_kind", "kind"),
    )


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
    :param root_task_id: ID of the top-level task that initiated
        this sub-agent's spawn tree, or ``None`` for top-level
        tasks.
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
    root_task_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True
    )

    __table_args__ = (
        Index("ix_tasks_conversation_id", "conversation_id"),
        Index("ix_tasks_agent_id", "agent_id"),
        Index("ix_tasks_created_at", "created_at"),
        Index("ix_tasks_root_task_id", "root_task_id"),
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


class SqlPendingToolCall(Base):
    """
    SQLAlchemy model for the ``pending_tool_calls`` table.

    Tracks the full lifecycle of a tunneled client-side tool call --
    from a sub-agent parking to the client delivering the result.

    :param call_id: Tool call ID (PK), matches the LLM-generated
        call ID. e.g. ``"call_abc123"``.
    :param root_task_id: The top-level task whose response output
        contains the ``function_call`` item.
    :param task_id: The parked sub-agent's task ID.
    :param tool_name: The tool function name, e.g. ``"Read"``.
    :param arguments: JSON-encoded arguments from the LLM,
        e.g. ``'{"file_path": "/tmp/foo.py"}'``.
    :param status: ``"action_required"`` or ``"completed"``.
    :param result: The tool's string output from the client.
        ``None`` until the client PATCHes.
    :param created_at: Unix epoch when the sub-agent parked.
    :param completed_at: Unix epoch when the client PATCHed.
        ``None`` until completed.
    """

    __tablename__ = "pending_tool_calls"

    call_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    root_task_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tasks.id", ondelete="CASCADE")
    )
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey("tasks.id", ondelete="CASCADE"))
    tool_name: Mapped[str] = mapped_column(String(256))
    arguments: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer)
    completed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('action_required', 'completed')",
            name="ck_pending_tool_calls_status",
        ),
        Index("ix_pending_tool_calls_root_task_id", "root_task_id"),
        Index("ix_pending_tool_calls_task_id", "task_id"),
    )
