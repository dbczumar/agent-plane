"""Routes for the /v1/responses endpoints (OpenResponses-compatible)."""

import asyncio
import json
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from agent_plane.runtime.models import (
    MessageData,
    NewConversationItem,
    Task,
)
from agent_plane.stores import SessionStore, TaskStore
from agent_plane.server.models import (
    AgentObject,
    ConversationRef,
    CreateResponseRequest,
    ErrorDetail,
    IncompleteDetails,
    ResponseDeleted,
    ResponseObject,
    Usage,
)

_TERMINAL_STATUSES = {"completed", "failed", "incomplete", "cancelled"}


def _build_response_object(task: Task) -> ResponseObject:
    """
    Convert a runtime Task into an API-layer ResponseObject.
    All fields are read from the task — the task store persists
    everything needed to reconstruct the response on GET.
    """
    return ResponseObject(
        id=task.task_id,
        status=task.status,
        model=task.agent,
        created_at=task.created_at,
        completed_at=task.completed_at,
        output=task.output if task.status == "completed" else [],
        background=task.background,
        previous_response_id=task.previous_response_id,
        conversation=ConversationRef(id=task.session_id),
        instructions=task.instructions,
        metadata=task.metadata,
        usage=Usage(**task.usage) if task.usage else None,
        error=(
            ErrorDetail(**task.error) if task.error else None
        ),
        incomplete_details=(
            IncompleteDetails(**task.incomplete_details)
            if task.incomplete_details
            else None
        ),
    )


def _normalize_input(raw_input: str | list) -> list:
    """
    Normalize the request input into a list of content parts. A plain
    string is converted into a single input_text content block.
    """
    if isinstance(raw_input, str):
        return [{"type": "input_text", "text": raw_input}]
    return raw_input


def _format_sse(event_type: str, data: dict | str) -> str:
    """Format a single SSE event string."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _poll_disconnect(request: Request) -> None:
    """Block until the client disconnects."""
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)


def create_responses_router(
    task_store: TaskStore,
    session_store: SessionStore,
    get_agent_by_name: Callable[[str], AgentObject | None],
) -> APIRouter:
    """
    Factory that builds the responses router. Stores and the agent
    lookup function are closed over — no dependency injection.
    """
    router = APIRouter()

    # ── POST /responses ──────────────────────────────────────────

    @router.post("/responses")
    async def create_response(
        req: CreateResponseRequest, request: Request
    ):
        # -- Validate store --
        if not req.store:
            raise HTTPException(
                status_code=400, detail="store: false is not supported"
            )

        # -- Validate input type --
        if not isinstance(req.input, (str, list)):
            raise HTTPException(
                status_code=400, detail="input must be a string or array"
            )

        # -- Validate model exists --
        agent = get_agent_by_name(req.model)
        if agent is None:
            raise HTTPException(
                status_code=404, detail="Unknown model"
            )

        # -- Validate conversation without previous_response_id --
        if req.conversation and not req.previous_response_id:
            raise HTTPException(
                status_code=400,
                detail="conversation provided without previous_response_id",
            )

        content = _normalize_input(req.input)

        if req.previous_response_id:
            # Resolve session via durable path (queries messages by
            # response_id). Raises internally if not found.
            try:
                session_id = session_store.get_session_id(
                    req.previous_response_id
                )
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="invalid previous_response_id",
                )

            # Validate conversation / response relationship
            if req.conversation:
                if session_id != req.conversation.id:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "previous_response_id does not belong to "
                            "the specified conversation"
                        ),
                    )
                latest = session_store.get_latest_response_id(session_id)
                if latest != req.previous_response_id:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "conversation provided with a fork -- "
                            "previous_response_id is not the latest "
                            "response"
                        ),
                    )

            # Check if previous response exists (None means deleted)
            prev_task = task_store.get(req.previous_response_id)
            if not prev_task:
                raise HTTPException(
                    status_code=400,
                    detail="previous_response_id not found",
                )

            # Steering: if the previous response is still running, try
            # to deliver the message to the running agent's inbox.
            if prev_task.status in ("in_progress", "queued"):
                message = NewConversationItem(
                    type="message",
                    response_id=req.previous_response_id,
                    data=MessageData(
                        role="user", content=content
                    ),
                )
                delivered = task_store.try_deliver(
                    req.previous_response_id, session_id, message
                )
                if delivered:
                    # Message accepted by running agent — return the
                    # existing in-progress response.
                    return _build_response_object(prev_task)
                # Inbox closed — agent is finishing. Wait for
                # completion so assistant output is in the session
                # before the new response loads history.
                await task_store.wait(req.previous_response_id)
        else:
            # No previous response — create a fresh session
            session = session_store.create_session()
            session_id = session.id

        # -- Normal flow: create a new response --
        task = task_store.create(
            session_id=session_id,
            agent=req.model,
            instructions=req.instructions,
            metadata=req.metadata,
            previous_response_id=req.previous_response_id,
            background=req.background,
        )
        session_store.append(
            session_id,
            [
                NewConversationItem(
                    type="message",
                    response_id=task.task_id,
                    data=MessageData(
                        role="user", content=content
                    ),
                )
            ],
        )
        task_store.start(task.task_id)

        # -- background=true, stream=false: return immediately --
        if req.background and not req.stream:
            return _build_response_object(task)

        # -- streaming (both background and foreground) --
        if req.stream:

            async def event_generator():
                completed_normally = False
                try:
                    seq = 0
                    initial = _build_response_object(task)
                    initial_dict = initial.model_dump()

                    # response.created
                    yield _format_sse(
                        "response.created",
                        {
                            "type": "response.created",
                            "response": initial_dict,
                            "sequence_number": seq,
                        },
                    )
                    seq += 1

                    # response.queued (background only)
                    if req.background:
                        yield _format_sse(
                            "response.queued",
                            {
                                "type": "response.queued",
                                "response": initial_dict,
                                "sequence_number": seq,
                            },
                        )
                        seq += 1

                    # response.in_progress
                    in_progress_dict = {
                        **initial_dict,
                        "status": "in_progress",
                    }
                    yield _format_sse(
                        "response.in_progress",
                        {
                            "type": "response.in_progress",
                            "response": in_progress_dict,
                            "sequence_number": seq,
                        },
                    )
                    seq += 1

                    # Stream events from the task store
                    async for event in task_store.stream(
                        task.task_id
                    ):
                        event_type = event.get("type", "unknown")
                        event["sequence_number"] = seq
                        yield _format_sse(event_type, event)
                        seq += 1

                    # Stream ended — wait for the workflow to
                    # fully exit (the finally block may still be
                    # running after close_stream).
                    final_task = await task_store.wait(
                        task.task_id
                    )
                    final_resp = _build_response_object(
                        final_task
                    )
                    final_dict = final_resp.model_dump()

                    # Terminal status event
                    terminal_event = (
                        f"response.{final_task.status}"
                    )
                    yield _format_sse(
                        terminal_event,
                        {
                            "type": terminal_event,
                            "response": final_dict,
                            "sequence_number": seq,
                        },
                    )

                    # End-of-stream sentinel
                    yield "data: [DONE]\n\n"
                    completed_normally = True
                finally:
                    # Foreground streaming: cancel on disconnect
                    if (
                        not req.background
                        and not completed_normally
                    ):
                        current = task_store.get(task.task_id)
                        if (
                            current
                            and current.status
                            not in _TERMINAL_STATUSES
                        ):
                            await asyncio.shield(
                                task_store.cancel(task.task_id)
                            )

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        # -- background=false, stream=false: blocking wait --
        # Race task completion against client disconnect so
        # foreground requests are cancelled when the client drops.
        wait_coro = asyncio.create_task(
            task_store.wait(task.task_id)
        )
        disconnect_coro = asyncio.create_task(
            _poll_disconnect(request)
        )
        done, pending = await asyncio.wait(
            {wait_coro, disconnect_coro},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if wait_coro in done:
            return _build_response_object(wait_coro.result())

        # Client disconnected — cancel the foreground task
        cancelled = await task_store.cancel(task.task_id)
        return _build_response_object(cancelled)

    # ── GET /responses/{response_id} ─────────────────────────────

    @router.get("/responses/{response_id}")
    async def get_response(response_id: str):
        task = task_store.get(response_id)
        if not task:
            raise HTTPException(
                status_code=404, detail="Response not found"
            )
        return _build_response_object(task)

    # ── POST /responses/{response_id}/cancel ─────────────────────

    @router.post("/responses/{response_id}/cancel")
    async def cancel_response(response_id: str):
        task = task_store.get(response_id)
        if not task:
            raise HTTPException(
                status_code=404, detail="Response not found"
            )
        if task.status in _TERMINAL_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Response is already in terminal status: "
                    f"{task.status}"
                ),
            )
        cancelled_task = await task_store.cancel(response_id)
        return _build_response_object(cancelled_task)

    # ── DELETE /responses/{response_id} ──────────────────────────

    @router.delete("/responses/{response_id}")
    async def delete_response(response_id: str):
        task = task_store.get(response_id)
        if not task:
            raise HTTPException(
                status_code=404, detail="Response not found"
            )
        await task_store.delete(response_id)
        return ResponseDeleted(id=response_id)

    return router
