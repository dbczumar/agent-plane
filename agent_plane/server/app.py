"""FastAPI application — main entry point for the agent-plane server."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent_plane.errors import AgentPlaneError
from agent_plane.server.routes.agents import create_agents_router
from agent_plane.server.routes.conversations import create_conversations_router
from agent_plane.server.routes.files import create_files_router
from agent_plane.server.routes.responses import create_responses_router
from agent_plane.stores import (
    AgentStore,
    ArtifactStore,
    ConversationStore,
    FileStore,
    TaskStore,
)

_logger = logging.getLogger(__name__)


def create_app(
    agent_store: AgentStore,
    file_store: FileStore,
    task_store: TaskStore,
    conversation_store: ConversationStore,
    artifact_store: ArtifactStore,
) -> FastAPI:
    """
    Build and return the FastAPI application with all routes mounted.

    Stores are injected here and passed to route factories. Each store
    is forwarded to the router factories that need it; the app itself
    only wires them together.

    :param agent_store: Store for agent CRUD operations.
    :param file_store: Store for uploaded-file metadata.
    :param task_store: Store for response/task lifecycle and
        workflow execution.
    :param conversation_store: Store for conversation and
        conversation-item persistence.
    :param artifact_store: Store for binary blobs (agent bundles,
        file content).
    :returns: A fully configured :class:`FastAPI` application.
    """
    app = FastAPI(title="Agent Plane Server")

    @app.exception_handler(AgentPlaneError)
    async def _handle_agent_plane_error(request: Request, exc: AgentPlaneError) -> JSONResponse:
        if exc.http_status >= 500:
            _logger.error("Internal error: %s", exc.message, exc_info=True)
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    app.include_router(
        create_agents_router(agent_store, task_store, artifact_store),
        prefix="/api",
        tags=["agents"],
    )
    app.include_router(
        create_responses_router(task_store, conversation_store, agent_store),
        prefix="/v1",
        tags=["responses"],
    )
    app.include_router(
        create_conversations_router(conversation_store, task_store),
        prefix="/v1",
        tags=["conversations"],
    )
    app.include_router(
        create_files_router(file_store, artifact_store),
        prefix="/v1",
        tags=["files"],
    )

    return app
