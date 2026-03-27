"""Task store — manages task lifecycle and durable execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from agent_plane.entities import (
    ConversationItem,
    NewConversationItem,
    Task,
)


class TaskStore(ABC):
    """
    Abstract base for task persistence and durable execution.

    Manages the full task lifecycle: creation, DBOS-backed execution
    (start, stream, wait), the steering handshake (try_deliver,
    close_inbox), cancellation, and deletion.
    """

    def __init__(self, storage_location: str) -> None:
        """
        Initialize the task store.

        The ``storage_location`` is a database URI. Concrete
        implementations also initialize the DBOS durable execution
        engine via ``ensure_dbos(storage_location)``.

        :param storage_location: Database URI,
            e.g. ``"sqlite:///tasks.db"`` or
            ``"postgresql://user:pass@host/db"``.
        """
        self.storage_location = storage_location

    @abstractmethod
    def create(
        self,
        conversation_id: str,
        agent_id: str,
        agent_name: str,
        previous_response_id: str | None = None,
        background: bool = False,
    ) -> Task:
        """
        Create a new task for executing an agent in the given
        conversation. Generates a unique task_id (which doubles as
        the response_id), stores the task record with
        status="queued", and returns the Task. Does not start
        execution -- call :meth:`start` to begin.

        ``agent_name`` is persisted so the API can return the
        original model name even if the agent is later renamed
        or deleted.

        ``instructions`` and ``reasoning`` are NOT stored in the
        task row -- they are pure workflow inputs passed to
        :meth:`start`.

        :param conversation_id: ID of the conversation this task
            belongs to, e.g. ``"conv_abc123"``.
        :param agent_id: ID of the agent to execute,
            e.g. ``"agent_xyz789"``.
        :param agent_name: Human-readable agent name at creation
            time, e.g. ``"code-assistant"``.
        :param previous_response_id: ID of the prior response in
            the thread, or ``None`` for the first turn,
            e.g. ``"resp_def456"``.
        :param background: Whether this is a background task
            (``True``) or blocking (``False``).
        :returns: The newly created :class:`Task` with status
            ``"queued"``.
        """
        ...

    @abstractmethod
    def start(
        self,
        task_id: str,
        instructions: str | None = None,
        reasoning: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Begin execution of a previously created task.

        Launches the DBOS workflow asynchronously and returns
        immediately -- the task remains "queued" until the workflow
        actually begins running, at which point it transitions to
        "in_progress". The task must exist and be in "queued"
        status.

        ``instructions``, ``reasoning``, and ``tools`` are passed
        directly to the DBOS workflow as inputs (stored by DBOS,
        not in the tasks table).

        Enforces the task/workflow invariant via a compensating
        transaction: if the DBOS workflow fails to start, the task
        row is deleted so neither artifact exists.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param instructions: Optional per-request system
            instructions override.
        :param reasoning: Optional reasoning configuration,
            e.g. ``{"effort": "medium"}``.
        :param tools: Optional list of client-specified tool dicts
            in OpenAI function format extended with ``agent_plane``
            callback metadata. Each entry must include
            ``agent_plane.callback.url``. ``None`` and ``[]`` are
            equivalent (no client tools), e.g.
            ``[{"type": "function", "function": {...},
            "agent_plane": {"callback": {"url": "..."}}}]``.
        """
        ...

    @abstractmethod
    def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """
        Yield streaming events as they are produced by the runtime.

        Awaits until the next event is available. The iterator ends
        when the task completes or is cancelled. Each event is a
        dict with a ``"type"`` field (e.g. ``"text_delta"``,
        ``"tool_call"``). Backed by ``DBOS.read_stream()``. Async
        because it long-polls for events.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: An async iterator of event dicts.
        """
        ...

    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """
        Return a snapshot of the task's current state.

        Output is populated only when status is "completed". For
        all other terminal states (failed, incomplete, cancelled),
        output is empty -- intermediate work is captured in the
        DBOS stream, not in the task output. Returns the task
        regardless of status. Returns ``None`` if the task does
        not exist (deleted by user or cleaned up by system).

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The :class:`Task` snapshot, or ``None``.
        """
        ...

    @abstractmethod
    async def wait(self, task_id: str) -> Task:
        """
        Await until the task reaches a terminal state and return
        the final Task.

        Terminal states: completed, failed, incomplete, or
        cancelled. Used by the server for blocking mode
        (``background=False``). Internally calls
        ``DBOS.retrieve_workflow(task_id).get_result()``.
        Async because it blocks until completion.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The final :class:`Task` in a terminal state.
        """
        ...

    @abstractmethod
    def try_deliver(
        self,
        task_id: str,
        conversation_id: str,
        item: NewConversationItem,
    ) -> bool:
        """
        Atomically deliver a steering message to a running task,
        or report that the inbox is already closed.

        Single transaction: if the agent's inbox is still open,
        appends the item to the conversation and returns ``True``.
        If the agent has already closed its inbox (finishing up),
        returns ``False`` -- the caller should create a new
        response instead.

        Server-side half of the steering handshake.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param conversation_id: ID of the conversation to append
            the item to, e.g. ``"conv_abc123"``.
        :param item: The :class:`NewConversationItem` to deliver
            to the running agent.
        :returns: ``True`` if the item was delivered, ``False``
            if the inbox was already closed.
        """
        ...

    @abstractmethod
    def close_inbox(
        self,
        task_id: str,
        conversation_id: str,
        last_seen_item_id: str | None,
    ) -> list[ConversationItem]:
        """
        Atomically attempt to close the inbox for a finishing task.

        Within a single transaction: queries for items in the
        conversation newer than ``last_seen_item_id`` (or all items
        if ``None``). If found, returns them (inbox stays open --
        agent must continue). If none found, sets
        ``inbox_closed=True`` and returns empty list (agent may
        complete). This is the agent-side half of the steering
        handshake.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :param conversation_id: ID of the conversation to check
            for new items, e.g. ``"conv_abc123"``.
        :param last_seen_item_id: ID of the last item the agent
            has processed, or ``None`` to check all items,
            e.g. ``"msg_xyz789"``.
        :returns: List of unseen :class:`ConversationItem` objects
            (empty if inbox was successfully closed).
        """
        ...

    @abstractmethod
    async def cancel(self, task_id: str) -> Task:
        """
        Stop execution and mark the task as cancelled.

        If in progress, stops the DBOS workflow and waits for the
        finally block to complete (close_stream, inbox drain).
        Sets status to "cancelled". The task record is preserved
        -- :meth:`get` still works, and the response can be
        referenced as ``previous_response_id`` to continue or
        redirect the conversation. Async because stopping an
        in-progress workflow may block while the finally block
        runs.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        :returns: The cancelled :class:`Task`.
        """
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """
        Remove a task record entirely.

        If in progress, stops the DBOS workflow first and waits
        for the finally block to complete. Then deletes the record
        -- subsequent :meth:`get` returns ``None``. Async because
        stopping an in-progress workflow may block while the
        finally block runs.

        :param task_id: Unique task identifier,
            e.g. ``"task_abc123"``.
        """
        ...

    async def delete_all(
        self,
        *,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """
        Cancel and delete all tasks matching the filter.

        Delegates to :meth:`list_tasks` + :meth:`delete` so DBOS
        workflow cleanup is honoured.

        :param agent_id: Optional agent ID filter,
            e.g. ``"agent_xyz789"``.
        :param conversation_id: Optional conversation ID filter,
            e.g. ``"conv_abc123"``.
        """
        for task in await self.list_tasks(agent_id=agent_id, conversation_id=conversation_id):
            await self.delete(task.id)

    @abstractmethod
    async def list_tasks(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[Task]:
        """
        Return tasks matching the given filters.

        All filters are optional and combined with AND. Used by
        the route layer to find in-flight tasks for cancellation
        (e.g. before deleting an agent or conversation).

        :param conversation_id: Optional conversation ID filter,
            e.g. ``"conv_abc123"``.
        :param agent_id: Optional agent ID filter,
            e.g. ``"agent_xyz789"``.
        :returns: A list of matching :class:`Task` objects,
            ordered by ``created_at`` descending.
        """
        ...
