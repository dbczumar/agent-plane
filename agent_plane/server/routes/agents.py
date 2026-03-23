"""Routes for the /api/agents endpoints."""

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from agent_plane.entities import Agent
from agent_plane.server.schemas import AgentDeleted, AgentObject, PaginatedList
from agent_plane.stores import AgentStore, ArtifactStore, TaskStore


def _to_agent_object(agent: Agent) -> AgentObject:
    return AgentObject(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        created_at=agent.created_at,
    )


def create_agents_router(
    agent_store: AgentStore,
    task_store: TaskStore,
    artifact_store: ArtifactStore,
) -> APIRouter:
    router = APIRouter()

    # ── POST /agents ───────────────────────────────────────────────

    @router.post("/agents", status_code=201)
    async def create_agent(
        bundle: UploadFile = File(...),
        name: str = Form(...),
        description: str | None = Form(default=None),
    ) -> AgentObject:
        if agent_store.get_by_name(name) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Agent with name '{name}' already exists",
            )

        bundle_bytes = await bundle.read()
        agent = agent_store.create(
            name=name,
            description=description,
        )
        artifact_store.put(agent.id, bundle_bytes)

        return _to_agent_object(agent)

    # ── GET /agents ────────────────────────────────────────────────

    @router.get("/agents")
    async def list_agents(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ) -> PaginatedList:
        page = agent_store.list(limit=limit, after=after, before=before, order=order)
        data = [_to_agent_object(a) for a in page.data]
        return PaginatedList(
            data=data,
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )

    # ── GET /agents/{agent_id} ─────────────────────────────────────

    @router.get("/agents/{agent_id}")
    async def get_agent(agent_id: str) -> AgentObject:
        agent = agent_store.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return _to_agent_object(agent)

    # ── DELETE /agents/{agent_id} ──────────────────────────────────

    @router.delete("/agents/{agent_id}")
    async def delete_agent(agent_id: str) -> AgentDeleted:
        agent = agent_store.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        for task in task_store.list_tasks(agent_id=agent_id):
            await task_store.cancel(task.id)
        artifact_store.delete(agent_id)
        agent_store.delete(agent_id)

        return AgentDeleted(id=agent_id)

    return router
