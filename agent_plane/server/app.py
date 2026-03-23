"""FastAPI application — main entry point for the agent-plane server."""

from fastapi import FastAPI

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


def create_app(
    agent_store: AgentStore,
    file_store: FileStore,
    task_store: TaskStore,
    conversation_store: ConversationStore,
    artifact_store: ArtifactStore,
) -> FastAPI:
    """
    Build and return the FastAPI application with all routes mounted.
    Stores are injected here and passed to route factories.
    """
    app = FastAPI(title="Agent Plane Server")

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
