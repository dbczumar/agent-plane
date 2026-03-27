"""Routes for the /v1/responses endpoints (OpenResponses-compatible)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from agent_plane.entities import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    MessageData,
    NewConversationItem,
    Task,
    TaskStatus,
)
from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.runtime.live_stream import register as _live_register
from agent_plane.runtime.live_stream import subscribe as _live_subscribe
from agent_plane.runtime.live_stream import unregister as _live_unregister
from agent_plane.server.schemas import (
    ConversationRef,
    CreateResponseRequest,
    ErrorDetail,
    IncompleteDetails,
    ResponseDeleted,
    ResponseObject,
    Usage,
)
from agent_plane.stores import AgentStore, ConversationStore, TaskStore
from agent_plane.tools.client_specified import parse_callback_tool_specs


def _build_response_object(task: Task) -> ResponseObject:
    """
    Convert a runtime Task into an API-layer ResponseObject.

    Uses ``task.agent_name`` (persisted at creation) for the model
    field so the value is stable even if the agent is renamed or
    deleted.

    :param task: The runtime task entity to convert.
    :returns: A fully populated :class:`ResponseObject`.
    """
    return ResponseObject(
        id=task.id,
        status=task.status,
        model=task.agent_name,
        created_at=task.created_at,
        completed_at=task.completed_at,
        # Only completed tasks surface output; failed/incomplete/cancelled
        # return [] per the OpenResponses spec.
        output=task.output if task.status == TaskStatus.COMPLETED else [],
        background=task.background,
        previous_response_id=task.previous_response_id,
        conversation=ConversationRef(id=task.conversation_id),
        instructions=task.instructions,
        reasoning=task.reasoning,
        usage=Usage(**task.usage) if task.usage else None,
        error=(ErrorDetail(**task.error) if task.error else None),
        incomplete_details=(
            IncompleteDetails(**task.incomplete_details) if task.incomplete_details else None
        ),
    )


def _normalize_input(
    raw_input: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize the request input into a list of content parts.

    A plain string is converted into a single ``input_text``
    content block.

    :param raw_input: Either a plain string (e.g. ``"Hello"``)
        or a list of content-block dicts, e.g.
        ``[{"type": "input_text", "text": "Hello"}]``.
    :returns: A list of content-block dicts.
    """
    if isinstance(raw_input, str):
        return [{"type": "input_text", "text": raw_input}]
    return raw_input


def _format_sse(event_type: str, data: dict[str, Any] | str) -> str:
    """
    Format a single Server-Sent Event string.

    :param event_type: SSE event name, e.g.
        ``"response.created"``, ``"response.completed"``.
    :param data: Payload to serialize. Dicts are JSON-encoded;
        strings are sent as-is.
    :returns: A complete SSE frame (``event: ...\\ndata: ...\\n\\n``).
    """
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _poll_disconnect(request: Request) -> None:
    """
    Block until the client disconnects.

    Polls every 0.5 seconds and returns once the client
    has closed the connection.

    :param request: The incoming FastAPI request to monitor.
    """
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)


_logger = logging.getLogger(__name__)


async def _resolve_conversation(
    req: CreateResponseRequest,
    content: list[dict[str, Any]],
    task_store: TaskStore,
    conversation_store: ConversationStore,
) -> ResponseObject | str:
    """
    Handle the ``previous_response_id`` path: resolve
    ``conversation_id``, validate the conversation/response
    relationship, attempt steering delivery, and wait if the
    inbox is closed.

    :param req: The incoming create-response request.
    :param content: Normalized user input content blocks,
        e.g. ``[{"type": "input_text", "text": "Hello"}]``.
    :param task_store: Store for task lifecycle operations.
    :param conversation_store: Store for conversation
        persistence.
    :returns: A :class:`ResponseObject` if steering succeeded
        (caller should return it immediately) or a
        ``conversation_id`` string for normal task-creation
        flow.
    :raises AgentPlaneError: If ``previous_response_id`` cannot
        be resolved.
    """
    if not req.previous_response_id:
        # No previous response — create a fresh conversation
        conversation = conversation_store.create_conversation()
        return conversation.id

    # Resolve conversation via durable path (queries items by
    # response_id). Raises internally if not found.
    try:
        conversation_id = conversation_store.get_conversation_id(req.previous_response_id)
    except LookupError:
        raise AgentPlaneError(
            "invalid previous_response_id",
            code=ErrorCode.INVALID_INPUT,
        )

    _validate_conversation_relationship(req, conversation_id, conversation_store)

    return await _attempt_steering(req.previous_response_id, content, conversation_id, task_store)


def _validate_conversation_relationship(
    req: CreateResponseRequest,
    conversation_id: str,
    conversation_store: ConversationStore,
) -> None:
    """
    Validate that ``conversation`` and ``previous_response_id``
    are consistent: same conversation and no fork.

    :param req: The incoming create-response request containing
        both ``conversation`` and ``previous_response_id``.
    :param conversation_id: The resolved conversation ID from
        the ``previous_response_id`` lookup.
    :param conversation_store: Store for conversation queries.
    :raises AgentPlaneError: If the conversation reference does
        not match the resolved conversation or if the
        ``previous_response_id`` is not the latest response
        (fork detected).
    """
    if not req.conversation:
        return

    # Validate conversation / response relationship
    if conversation_id != req.conversation.id:
        raise AgentPlaneError(
            "previous_response_id does not belong to the specified conversation",
            code=ErrorCode.INVALID_INPUT,
        )
    latest = conversation_store.get_latest_response_id(conversation_id)
    if latest != req.previous_response_id:
        raise AgentPlaneError(
            "conversation provided with a fork -- previous_response_id is not the latest response",
            code=ErrorCode.INVALID_INPUT,
        )


async def _attempt_steering(
    previous_response_id: str,
    content: list[dict[str, Any]],
    conversation_id: str,
    task_store: TaskStore,
) -> ResponseObject | str:
    """
    Check if the previous response is still active and try to
    deliver the message to the running agent's inbox.

    :param previous_response_id: ID of the prior response,
        e.g. ``"resp_abc123"``.
    :param content: Normalized user input content blocks,
        e.g. ``[{"type": "input_text", "text": "..."}]``.
    :param conversation_id: The resolved conversation ID.
    :param task_store: Store for task lifecycle operations.
    :returns: A :class:`ResponseObject` if steering succeeded
        (message delivered to the running agent), otherwise
        the ``conversation_id`` string for normal
        task-creation flow.
    :raises AgentPlaneError: If the previous response does
        not exist (deleted).
    """
    # Check if previous response exists (None means deleted).
    # Must use get_async — this is an async handler with a running
    # event loop, and DBOS status lookup has distinct sync/async APIs.
    prev_task = await task_store.get(previous_response_id)
    if not prev_task:
        raise AgentPlaneError(
            "previous_response_id not found",
            code=ErrorCode.INVALID_INPUT,
        )

    # Steering: if the previous response is still running, try
    # to deliver the message to the running agent's inbox.
    if prev_task.status in ACTIVE_STATUSES:
        message = NewConversationItem(
            type="message",
            response_id=previous_response_id,
            data=MessageData(role="user", content=content),
        )
        delivered = task_store.try_deliver(
            previous_response_id,
            conversation_id,
            message,
        )
        if delivered:
            # Message accepted by running agent — return the
            # existing in-progress response.
            return _build_response_object(prev_task)
        # Inbox closed — agent is finishing. Wait for
        # completion so assistant output is in the conversation
        # before the new response loads history.
        await task_store.wait(previous_response_id)

    return conversation_id


def _create_task(
    req: CreateResponseRequest,
    conversation_id: str,
    agent_id: str,
    agent_name: str,
    content: list[dict[str, Any]],
    task_store: TaskStore,
    conversation_store: ConversationStore,
) -> Task:
    """
    Create a new task and append the user message to the
    conversation.

    Does NOT start the workflow -- the caller must call
    :func:`_start_task` separately. This split allows the
    caller to register the live stream between create and
    start, eliminating the race where early events are lost.

    :param req: The incoming create-response request.
    :param conversation_id: ID of the conversation to append
        to, e.g. ``"conv_abc123"``.
    :param agent_id: ID of the agent executing the task.
    :param agent_name: Denormalized agent name persisted on the
        task for stable API responses, e.g.
        ``"research-agent"``.
    :param content: Normalized user input content blocks,
        e.g. ``[{"type": "input_text", "text": "..."}]``.
    :param task_store: Store for task creation.
    :param conversation_store: Store for appending conversation
        items.
    :returns: The newly created (but not yet started)
        :class:`Task`.
    """
    task = task_store.create(
        conversation_id=conversation_id,
        agent_id=agent_id,
        agent_name=agent_name,
        previous_response_id=req.previous_response_id,
        background=req.background,
    )
    conversation_store.append(
        conversation_id,
        [
            NewConversationItem(
                type="message",
                response_id=task.id,
                data=MessageData(role="user", content=content),
            )
        ],
    )
    return task


def _start_task(
    task: Task,
    req: CreateResponseRequest,
    task_store: TaskStore,
    tools: list[dict[str, Any]] | None,
) -> None:
    """
    Start the DBOS workflow for a previously created task and
    set workflow inputs on the task entity.

    :param task: The task entity returned by :func:`_create_task`.
    :param req: The incoming create-response request (provides
        ``instructions`` and ``reasoning``).
    :param task_store: Store used to launch the DBOS workflow.
    :param tools: Validated client-specified tool dicts to pass
        as workflow inputs, or ``None`` if the request had no
        client tools.
    """
    task_store.start(
        task.id,
        instructions=req.instructions,
        reasoning=req.reasoning,
        tools=tools,
    )
    # Set workflow inputs on the task entity for the initial response.
    # Subsequent get() calls restore them from DBOS workflow inputs.
    task.instructions = req.instructions
    task.reasoning = req.reasoning
    task.tools = tools


@dataclass
class _InitialEvents:
    """
    Result of building the initial SSE events (created, queued,
    in_progress).

    Carries the serialized events and the initial response dict
    needed for the terminal fallback.

    :param sse_strings: Pre-formatted SSE frame strings ready to
        yield to the client.
    :param initial_dict: Serialized :class:`ResponseObject` dict
        for the initial (queued) state, reused when building
        the terminal fallback on error.
    """

    sse_strings: list[str] = field(default_factory=list)
    initial_dict: dict[str, Any] = field(default_factory=dict)


def _build_initial_events(
    task: Task,
    background: bool,
) -> _InitialEvents:
    """
    Build the ``response.created``, optional ``response.queued``,
    and ``response.in_progress`` SSE events.

    :param task: The newly created task entity.
    :param background: Whether this is a background task.
        Background tasks include a ``response.queued`` event.
    :returns: An :class:`_InitialEvents` containing the
        formatted SSE strings and the initial response dict.
    """
    initial = _build_response_object(task)
    initial_dict = initial.model_dump()
    events: list[str] = []

    events.append(
        _format_sse(
            "response.created",
            {
                "type": "response.created",
                "response": initial_dict,
                "sequence_number": 0,
            },
        )
    )

    if background:
        events.append(
            _format_sse(
                "response.queued",
                {
                    "type": "response.queued",
                    "response": initial_dict,
                    "sequence_number": 1,
                },
            )
        )

    in_progress_dict = {
        **initial_dict,
        "status": TaskStatus.IN_PROGRESS,
    }
    events.append(
        _format_sse(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": in_progress_dict,
                # 2 when background (created, queued, in_progress);
                # 1 when foreground (created, in_progress)
                "sequence_number": 2 if background else 1,
            },
        )
    )

    return _InitialEvents(
        sse_strings=events,
        initial_dict=initial_dict,
    )


@dataclass
class _TerminalEvent:
    """
    Result of building the terminal SSE event.

    :param event_type: SSE event name, e.g.
        ``"response.completed"``, ``"response.failed"``.
    :param response_dict: Serialized :class:`ResponseObject`
        dict for the final state.
    """

    event_type: str
    response_dict: dict[str, Any]


async def _build_terminal_event(
    task_id: str,
    initial_dict: dict[str, Any],
    task_store: TaskStore,
) -> _TerminalEvent:
    """
    Wait for the task workflow to fully exit and build the
    terminal SSE event.

    Falls back to a minimal failed response on error so the
    client receives a clean SSE close instead of a dropped
    connection.

    :param task_id: ID of the task to wait on,
        e.g. ``"resp_abc123"``.
    :param initial_dict: Serialized initial response dict used
        as the base for the fallback error payload.
    :param task_store: Store for waiting on task completion.
    :returns: A :class:`_TerminalEvent` with the final event
        type and response dict.
    """
    # Stream ended — wait for the workflow to
    # fully exit (the finally block may still be
    # running after close_stream).
    try:
        final_task = await task_store.wait(task_id)
        final_resp = _build_response_object(final_task)
        return _TerminalEvent(
            event_type=f"response.{final_task.status}",
            response_dict=final_resp.model_dump(),
        )
    except Exception:
        _logger.exception(
            "failed to build terminal event for task %s",
            task_id,
        )
        # Build a minimal failed response so the client
        # gets a clean SSE close instead of a dropped
        # connection.
        return _TerminalEvent(
            event_type="response.failed",
            response_dict={
                **initial_dict,
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "Failed to retrieve final response",
                },
            },
        )


async def _stream_events(
    task: Task,
    task_store: TaskStore,
    background: bool,
) -> AsyncIterator[str]:
    """
    Async generator that yields SSE strings for a streaming
    response.

    Reads from the in-process live stream for real-time token
    delivery. The caller MUST register the live stream before
    starting the task to guarantee no events are lost (no race
    condition).

    On foreground tasks, cancels the task if the client
    disconnects before the stream completes.

    :param task: The task entity to stream events for.
    :param task_store: Store for task lifecycle operations
        (wait, cancel, get).
    :param background: Whether this is a background task.
        Background tasks are not cancelled on disconnect.
    :yields: Formatted SSE frame strings.
    """
    completed_normally = False
    try:
        initial = _build_initial_events(task, background)
        for sse in initial.sse_strings:
            yield sse
        seq = len(initial.sse_strings)

        async for event in _live_subscribe(task.id):
            if "type" not in event:
                raise ValueError(
                    "stream event missing 'type' field",
                )
            event["sequence_number"] = seq
            yield _format_sse(event["type"], event)
            seq += 1

        terminal = await _build_terminal_event(
            task.id,
            initial.initial_dict,
            task_store,
        )
        yield _format_sse(
            terminal.event_type,
            {
                "type": terminal.event_type,
                "response": terminal.response_dict,
                "sequence_number": seq,
            },
        )

        yield "data: [DONE]\n\n"
        completed_normally = True
    finally:
        _live_unregister(task.id)
        # Foreground streaming: cancel on disconnect
        if not background and not completed_normally:
            current = await task_store.get(task.id)
            if current and current.status not in TERMINAL_STATUSES:
                await asyncio.shield(task_store.cancel(task.id))


async def _handle_blocking_wait(
    task: Task,
    request: Request,
    task_store: TaskStore,
) -> ResponseObject:
    """
    Race task completion against client disconnect so foreground
    requests are cancelled when the client drops.

    :param task: The task entity to wait on.
    :param request: The incoming FastAPI request, used to detect
        client disconnect.
    :param task_store: Store for waiting on and cancelling the
        task.
    :returns: A :class:`ResponseObject` for the completed (or
        cancelled) task.
    """
    # -- background=false, stream=false: blocking wait --
    # Race task completion against client disconnect so
    # foreground requests are cancelled when the client drops.
    wait_coro = asyncio.create_task(task_store.wait(task.id))
    disconnect_coro = asyncio.create_task(_poll_disconnect(request))
    done, pending = await asyncio.wait(
        {wait_coro, disconnect_coro},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()

    if wait_coro in done:
        return _build_response_object(wait_coro.result())

    # Client disconnected — cancel the foreground task
    cancelled = await task_store.cancel(task.id)
    return _build_response_object(cancelled)


def create_responses_router(
    task_store: TaskStore,
    conversation_store: ConversationStore,
    agent_store: AgentStore,
) -> APIRouter:
    """
    Factory that builds the responses router.

    Stores are closed over -- no dependency injection.

    :param task_store: Store for task lifecycle operations.
    :param conversation_store: Store for conversation
        persistence.
    :param agent_store: Store for agent lookups (model
        resolution).
    :returns: A configured :class:`APIRouter` with all
        ``/responses`` endpoints.
    """
    router = APIRouter()

    # ── POST /responses ──────────────────────────────────────────

    # response_model=None: this endpoint returns either a Pydantic model
    # or a StreamingResponse; FastAPI can't auto-generate a schema for
    # that union, so we disable automatic response model inference.
    @router.post("/responses", response_model=None)
    async def create_response(
        req: CreateResponseRequest, request: Request
    ) -> ResponseObject | StreamingResponse:
        """
        Create a new response (task execution).

        Validates the request, resolves the conversation (or
        steers an active response), creates and starts the
        task, and returns the result according to the
        ``stream`` and ``background`` flags.

        :param req: The create-response request body.
        :param request: The raw FastAPI request, used for
            disconnect detection on blocking waits.
        :returns: A :class:`ResponseObject` (blocking or
            background) or a :class:`StreamingResponse` (SSE).
        :raises AgentPlaneError: On validation failures or
            unknown model.
        """
        # -- Validate store --
        if not req.store:
            raise AgentPlaneError(
                "store: false is not supported",
                code=ErrorCode.INVALID_INPUT,
            )

        # -- Validate model exists --
        agent = agent_store.get_by_name(req.model)
        if agent is None:
            raise AgentPlaneError("Unknown model", code=ErrorCode.NOT_FOUND)

        # -- Validate conversation without previous_response_id --
        if req.conversation and not req.previous_response_id:
            raise AgentPlaneError(
                "conversation provided without previous_response_id",
                code=ErrorCode.INVALID_INPUT,
            )

        # -- Validate and parse client-specified tools --
        client_tools: list[dict[str, Any]] | None = None
        if req.tools:
            try:
                # Validate structure upfront; parse_callback_tool_specs
                # raises ValueError on malformed entries.
                parse_callback_tool_specs(req.tools)
            except ValueError as exc:
                raise AgentPlaneError(str(exc), code=ErrorCode.INVALID_INPUT)
            client_tools = req.tools

        content = _normalize_input(req.input)

        result = await _resolve_conversation(req, content, task_store, conversation_store)
        # Steering succeeded — return the existing response
        if isinstance(result, ResponseObject):
            return result
        conversation_id: str = result

        # -- Normal flow: create task, then start --
        # Split into create → register live stream → start to
        # guarantee no streaming events are lost (no race).
        task = _create_task(
            req,
            conversation_id,
            agent.id,
            agent.name,
            content,
            task_store,
            conversation_store,
        )

        if req.stream:
            # Register BEFORE start so the live stream captures
            # every event the workflow produces.
            _live_register(task.id, asyncio.get_running_loop())

        _start_task(task, req, task_store, client_tools)

        # -- background=true, stream=false: return immediately --
        if req.background and not req.stream:
            return _build_response_object(task)

        # -- streaming (both background and foreground) --
        if req.stream:
            return StreamingResponse(
                _stream_events(task, task_store, req.background),
                media_type="text/event-stream",
            )

        return await _handle_blocking_wait(task, request, task_store)

    # ── GET /responses/{response_id} ─────────────────────────────

    @router.get("/responses/{response_id}")
    async def get_response(response_id: str) -> ResponseObject:
        """
        Retrieve a single response by ID.

        :param response_id: The response/task identifier,
            e.g. ``"resp_abc123"``.
        :returns: The matching :class:`ResponseObject`.
        :raises AgentPlaneError: If the response is not found.
        """
        task = await task_store.get(response_id)
        if not task:
            raise AgentPlaneError("Response not found", code=ErrorCode.NOT_FOUND)
        return _build_response_object(task)

    # ── POST /responses/{response_id}/cancel ─────────────────────

    @router.post("/responses/{response_id}/cancel")
    async def cancel_response(
        response_id: str,
    ) -> ResponseObject:
        """
        Cancel an in-progress response.

        If the response is already in a terminal state, returns
        it unchanged.

        :param response_id: The response/task identifier,
            e.g. ``"resp_abc123"``.
        :returns: The cancelled (or already-terminal)
            :class:`ResponseObject`.
        :raises AgentPlaneError: If the response is not found.
        """
        task = await task_store.get(response_id)
        if not task:
            raise AgentPlaneError("Response not found", code=ErrorCode.NOT_FOUND)
        if task.status in TERMINAL_STATUSES:
            return _build_response_object(task)
        cancelled_task = await task_store.cancel(response_id)
        return _build_response_object(cancelled_task)

    # ── DELETE /responses/{response_id} ──────────────────────────

    @router.delete("/responses/{response_id}")
    async def delete_response(
        response_id: str,
    ) -> ResponseDeleted:
        """
        Delete a response by ID.

        :param response_id: The response/task identifier,
            e.g. ``"resp_abc123"``.
        :returns: A :class:`ResponseDeleted` confirmation.
        :raises AgentPlaneError: If the response is not found.
        """
        task = await task_store.get(response_id)
        if not task:
            raise AgentPlaneError("Response not found", code=ErrorCode.NOT_FOUND)
        await task_store.delete(response_id)
        return ResponseDeleted(id=response_id)

    return router
