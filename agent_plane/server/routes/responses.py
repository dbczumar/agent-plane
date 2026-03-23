"""Routes for the /v1/responses endpoints (OpenResponses-compatible)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from agent_plane.entities import (
    MessageData,
    NewConversationItem,
    Task,
)
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

_TERMINAL_STATUSES = {"completed", "failed", "incomplete", "cancelled"}


def _build_response_object(
    task: Task,
    agent_store: AgentStore,
) -> ResponseObject:
    """
    Convert a runtime Task into an API-layer ResponseObject.
    Looks up the agent name from agent_id for the model field.
    """
    agent = agent_store.get(task.agent_id)
    return ResponseObject(
        id=task.task_id,
        status=task.status,
        model=agent.name if agent else task.agent_id,
        created_at=task.created_at,
        completed_at=task.completed_at,
        output=task.output if task.status == "completed" else [],
        background=task.background,
        previous_response_id=task.previous_response_id,
        conversation=ConversationRef(id=task.conversation_id),
        instructions=task.instructions,
        usage=Usage(**task.usage) if task.usage else None,
        error=(ErrorDetail(**task.error) if task.error else None),
        incomplete_details=(
            IncompleteDetails(**task.incomplete_details) if task.incomplete_details else None
        ),
    )


def _normalize_input(raw_input: str | list[Any]) -> list[Any]:
    """
    Normalize the request input into a list of content parts. A plain
    string is converted into a single input_text content block.
    """
    if isinstance(raw_input, str):
        return [{"type": "input_text", "text": raw_input}]
    return raw_input


def _format_sse(event_type: str, data: dict[str, Any] | str) -> str:
    """Format a single SSE event string."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _poll_disconnect(request: Request) -> None:
    """Block until the client disconnects."""
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)


def create_responses_router(
    task_store: TaskStore,
    conversation_store: ConversationStore,
    agent_store: AgentStore,
) -> APIRouter:
    """
    Factory that builds the responses router. Stores are closed
    over — no dependency injection.
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
        # -- Validate store --
        if not req.store:
            raise HTTPException(status_code=400, detail="store: false is not supported")

        # -- Validate model exists --
        agent = agent_store.get_by_name(req.model)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown model")

        # -- Validate conversation without previous_response_id --
        if req.conversation and not req.previous_response_id:
            raise HTTPException(
                status_code=400,
                detail="conversation provided without previous_response_id",
            )

        content = _normalize_input(req.input)

        if req.previous_response_id:
            # Resolve conversation via durable path (queries items by
            # response_id). Raises internally if not found.
            try:
                conversation_id = conversation_store.get_conversation_id(req.previous_response_id)
            except LookupError:
                raise HTTPException(
                    status_code=400,
                    detail="invalid previous_response_id",
                )

            # Validate conversation / response relationship
            if req.conversation:
                if conversation_id != req.conversation.id:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "previous_response_id does not belong to the specified conversation"
                        ),
                    )
                latest = conversation_store.get_latest_response_id(conversation_id)
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
                    data=MessageData(role="user", content=content),
                )
                delivered = task_store.try_deliver(
                    req.previous_response_id,
                    conversation_id,
                    message,
                )
                if delivered:
                    # Message accepted by running agent — return the
                    # existing in-progress response.
                    return _build_response_object(prev_task, agent_store)
                # Inbox closed — agent is finishing. Wait for
                # completion so assistant output is in the conversation
                # before the new response loads history.
                await task_store.wait(req.previous_response_id)
        else:
            # No previous response — create a fresh conversation
            conversation = conversation_store.create_conversation()
            conversation_id = conversation.id

        # -- Normal flow: create a new response --
        task = task_store.create(
            conversation_id=conversation_id,
            agent_id=agent.id,
            instructions=req.instructions,
            previous_response_id=req.previous_response_id,
            background=req.background,
        )
        conversation_store.append(
            conversation_id,
            [
                NewConversationItem(
                    type="message",
                    response_id=task.task_id,
                    data=MessageData(role="user", content=content),
                )
            ],
        )
        task_store.start(task.task_id)

        # -- background=true, stream=false: return immediately --
        if req.background and not req.stream:
            return _build_response_object(task, agent_store)

        # -- streaming (both background and foreground) --
        if req.stream:

            async def event_generator() -> AsyncIterator[str]:
                completed_normally = False
                try:
                    seq = 0
                    initial = _build_response_object(task, agent_store)
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
                    async for event in task_store.stream(task.task_id):
                        event_type = event.get("type", "unknown")
                        event["sequence_number"] = seq
                        yield _format_sse(event_type, event)
                        seq += 1

                    # Stream ended — wait for the workflow to
                    # fully exit (the finally block may still be
                    # running after close_stream).
                    final_task = await task_store.wait(task.task_id)
                    final_resp = _build_response_object(final_task, agent_store)
                    final_dict = final_resp.model_dump()

                    # Terminal status event
                    terminal_event = f"response.{final_task.status}"
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
                    if not req.background and not completed_normally:
                        current = task_store.get(task.task_id)
                        if current and current.status not in _TERMINAL_STATUSES:
                            await asyncio.shield(task_store.cancel(task.task_id))

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        # -- background=false, stream=false: blocking wait --
        # Race task completion against client disconnect so
        # foreground requests are cancelled when the client drops.
        wait_coro = asyncio.create_task(task_store.wait(task.task_id))
        disconnect_coro = asyncio.create_task(_poll_disconnect(request))
        done, pending = await asyncio.wait(
            {wait_coro, disconnect_coro},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if wait_coro in done:
            return _build_response_object(wait_coro.result(), agent_store)

        # Client disconnected — cancel the foreground task
        cancelled = await task_store.cancel(task.task_id)
        return _build_response_object(cancelled, agent_store)

    # ── GET /responses/{response_id} ─────────────────────────────

    @router.get("/responses/{response_id}")
    async def get_response(response_id: str) -> ResponseObject:
        task = task_store.get(response_id)
        if not task:
            raise HTTPException(status_code=404, detail="Response not found")
        return _build_response_object(task, agent_store)

    # ── POST /responses/{response_id}/cancel ─────────────────────

    @router.post("/responses/{response_id}/cancel")
    async def cancel_response(response_id: str) -> ResponseObject:
        task = task_store.get(response_id)
        if not task:
            raise HTTPException(status_code=404, detail="Response not found")
        if task.status in _TERMINAL_STATUSES:
            return _build_response_object(task, agent_store)
        cancelled_task = await task_store.cancel(response_id)
        return _build_response_object(cancelled_task, agent_store)

    # ── DELETE /responses/{response_id} ──────────────────────────

    @router.delete("/responses/{response_id}")
    async def delete_response(response_id: str) -> ResponseDeleted:
        task = task_store.get(response_id)
        if not task:
            raise HTTPException(status_code=404, detail="Response not found")
        await task_store.delete(response_id)
        return ResponseDeleted(id=response_id)

    return router
