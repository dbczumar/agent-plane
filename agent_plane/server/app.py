"""FastAPI application — main entry point for the agent-plane server."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.runtime.agent_cache import AgentCache
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
    agent_cache: AgentCache,
) -> FastAPI:
    """
    Build and return the FastAPI application with all routes mounted.

    Stores and cache are injected here and passed to route factories.
    Each dependency is forwarded to the router factories that need it;
    the app itself only wires them together.

    :param agent_store: Store for agent CRUD operations.
    :param file_store: Store for uploaded-file metadata.
    :param task_store: Store for response/task lifecycle and
        workflow execution.
    :param conversation_store: Store for conversation and
        conversation-item persistence.
    :param artifact_store: Store for binary blobs (agent bundles,
        file content).
    :param agent_cache: Cache for loaded agent specs and working
        directories.
    :returns: A fully configured :class:`FastAPI` application.
    """
    app = FastAPI(title="Agent Plane Server")

    @app.exception_handler(AgentPlaneError)
    async def _handle_agent_plane_error(request: Request, exc: AgentPlaneError) -> JSONResponse:
        """
        Convert application errors to structured JSON responses.

        :param request: The incoming request.
        :param exc: The application error.
        :returns: A JSON response with the error code and message.
        """
        if exc.http_status >= 500:
            _logger.error("Internal error: %s", exc.message, exc_info=True)
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def _handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all for unhandled exceptions (e.g. database
        OperationalError). Returns the standard JSON error schema
        so clients always get a consistent response format.

        :param request: The incoming request.
        :param exc: The unhandled exception.
        :returns: A 500 JSON response with ``internal_error`` code.
        """
        _logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR,
                    "message": "An internal error occurred.",
                },
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """
        Liveness check — returns 200 when the server is running.

        :returns: ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    app.include_router(
        create_agents_router(agent_store, task_store, artifact_store, agent_cache),
        prefix="/api",
        tags=["agents"],
    )
    app.include_router(
        create_responses_router(task_store, conversation_store, agent_store, file_store),
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
