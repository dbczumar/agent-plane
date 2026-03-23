"""FastAPI application — main entry point for the agent-plane server."""

from fastapi import FastAPI

from agent_plane.runtime.stores import ArtifactStore, SessionStore, TaskStore
from agent_plane.server.routes.agents import create_agents_router
from agent_plane.server.routes.conversations import create_conversations_router
from agent_plane.server.routes.files import create_files_router
from agent_plane.server.routes.responses import create_responses_router


def create_app(
    task_store: TaskStore,
    session_store: SessionStore,
    artifact_store: ArtifactStore,
) -> FastAPI:
    """
    Build and return the FastAPI application with all routes mounted.
    Stores are injected here and passed to route factories.
    """
    app = FastAPI(title="Agent Plane Server")

    agents_router = create_agents_router(task_store, artifact_store)
    app.include_router(
        agents_router,
        prefix="/api",
        tags=["agents"],
    )

    app.include_router(
        create_responses_router(
            task_store,
            session_store,
            get_agent_by_name=agents_router.get_agent_by_name,
        ),
        prefix="/v1",
        tags=["responses"],
    )
    app.include_router(
        create_conversations_router(session_store),
        prefix="/v1",
        tags=["conversations"],
    )
    app.include_router(
        create_files_router(artifact_store),
        prefix="/v1",
        tags=["files"],
    )

    return app
