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
    def __init__(self, uri: str) -> None:
        """
        Initialize the task store and the underlying DBOS durable execution
        engine. Calls ensure_dbos(uri) to initialize the DBOS singleton if
        it hasn't been already.
        """

    @abstractmethod
    def create(
        self,
        conversation_id: str,
        agent_id: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        background: bool = False,
    ) -> Task:
        """
        Create a new task for executing an agent in the given conversation.
        Generates a unique task_id (which doubles as the response_id),
        stores the task record with status="queued", and returns the Task.
        Does not start execution -- call start() to begin.
        """
        ...

    @abstractmethod
    def start(self, task_id: str) -> None:
        """
        Begin execution of a previously created task. Launches the DBOS
        workflow asynchronously and returns immediately -- the task
        remains "queued" until the workflow actually begins running,
        at which point it transitions to "in_progress". The task must
        exist and be in "queued" status.
        """
        ...

    @abstractmethod
    def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """
        Yield streaming events as they are produced by the runtime. Awaits
        until the next event is available. The iterator ends when the task
        completes or is cancelled. Each event is a dict with a "type" field
        (e.g. "text_delta", "tool_call"). Backed by DBOS.read_stream().
        Async because it long-polls for events.
        """
        ...

    @abstractmethod
    def get(self, task_id: str) -> Task | None:
        """
        Return a snapshot of the task's current state. Output is populated
        only when status is "completed". For all other terminal states
        (failed, incomplete, cancelled), output is empty -- intermediate
        work is captured in the DBOS stream, not in the task output.
        Returns the task regardless of status. Returns None if the task
        does not exist (deleted by user or cleaned up by system).
        """
        ...

    @abstractmethod
    async def wait(self, task_id: str) -> Task:
        """
        Await until the task reaches a terminal state (completed, failed,
        incomplete, or cancelled) and return the final Task. Used by the
        server for blocking mode (background=false). Internally calls
        DBOS.retrieve_workflow(task_id).get_result(). Async because it
        blocks until completion.
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
        Atomically deliver a steering message to a running task, or
        report that the inbox is already closed.

        Single transaction: if the agent's inbox is still open, appends
        the item to the conversation and returns True. If the agent has
        already closed its inbox (finishing up), returns False -- the
        caller should create a new response instead.

        Server-side half of the steering handshake.
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
        Within a single transaction: queries for items in the conversation
        newer than last_seen_item_id (or all items if None). If found,
        returns them (inbox stays open -- agent must continue). If none
        found, sets inbox_closed=True and returns empty list (agent may
        complete). This is the agent-side half of the steering handshake.
        """
        ...

    @abstractmethod
    async def cancel(self, task_id: str) -> Task:
        """
        Stop execution and mark the task as cancelled. If in progress,
        stops the DBOS workflow and waits for the finally block to
        complete (close_stream, inbox drain). Sets status to "cancelled".
        The task record is preserved -- get() still works, and the response
        can be referenced as previous_response_id to continue or redirect
        the conversation. Async because stopping an in-progress workflow
        may block while the finally block runs.
        """
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """
        Remove a task record entirely. If in progress, stops the DBOS
        workflow first and waits for the finally block to complete. Then
        deletes the record -- subsequent get() returns None. Async because
        stopping an in-progress workflow may block while the finally block
        runs.
        """
        ...

    @abstractmethod
    def list_tasks(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[Task]:
        """
        Return tasks matching the given filters. All filters are
        optional and combined with AND. Used by the route layer to
        find in-flight tasks for cancellation (e.g. before deleting
        an agent or conversation).
        """
        ...
