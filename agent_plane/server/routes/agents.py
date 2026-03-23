"""Routes for the /api/agents endpoints."""

import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from agent_plane.runtime.stores import ArtifactStore, TaskStore
from agent_plane.server.models import AgentDeleted, AgentObject, PaginatedList


def create_agents_router(
    task_store: TaskStore,
    artifact_store: ArtifactStore,
) -> APIRouter:
    """
    Factory that builds the agents router. Agent storage is in-memory
    (a simple dict inside this closure) as a placeholder until real
    storage is implemented.
    """
    router = APIRouter()

    # In-memory storage, keyed by agent id.
    agents_by_id: dict[str, AgentObject] = {}
    # Name-to-id index for uniqueness checks.
    agents_by_name: dict[str, str] = {}

    # Expose storage on the router so other routers (e.g. responses)
    # can look up agents by name.
    router.agents_by_id = agents_by_id  # type: ignore[attr-defined]
    router.agents_by_name = agents_by_name  # type: ignore[attr-defined]

    def get_agent_by_name(name: str) -> AgentObject | None:
        """
        Look up an agent by its unique name. Returns None if no
        agent with that name exists.
        """
        agent_id = agents_by_name.get(name)
        if agent_id is None:
            return None
        return agents_by_id.get(agent_id)

    router.get_agent_by_name = get_agent_by_name  # type: ignore[attr-defined]

    # ── POST /agents ───────────────────────────────────────────────

    @router.post("/agents", status_code=201)
    async def create_agent(
        bundle: UploadFile = File(...),
        name: str = Form(...),
        description: str | None = Form(default=None),
    ) -> AgentObject:
        if name in agents_by_name:
            raise HTTPException(
                status_code=409,
                detail=f"Agent with name '{name}' already exists",
            )

        agent_id = f"ag_{uuid.uuid4().hex[:12]}"
        bundle_bytes = await bundle.read()
        artifact_store.put(agent_id, bundle_bytes)

        agent = AgentObject(
            id=agent_id,
            name=name,
            description=description,
            created_at=int(time.time()),
        )

        agents_by_id[agent_id] = agent
        agents_by_name[name] = agent_id

        return agent

    # ── GET /agents ────────────────────────────────────────────────

    @router.get("/agents")
    async def list_agents(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ) -> PaginatedList:
        sorted_agents = sorted(
            agents_by_id.values(),
            key=lambda a: a.created_at,
            reverse=(order == "desc"),
        )

        # Apply cursor-based pagination.
        if after is not None:
            idx = next(
                (
                    i
                    for i, a in enumerate(sorted_agents)
                    if a.id == after
                ),
                None,
            )
            if idx is not None:
                sorted_agents = sorted_agents[idx + 1 :]

        if before is not None:
            idx = next(
                (
                    i
                    for i, a in enumerate(sorted_agents)
                    if a.id == before
                ),
                None,
            )
            if idx is not None:
                sorted_agents = sorted_agents[:idx]

        has_more = len(sorted_agents) > limit
        page = sorted_agents[:limit]

        return PaginatedList(
            data=page,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=has_more,
        )

    # ── GET /agents/{agent_id} ─────────────────────────────────────

    @router.get("/agents/{agent_id}")
    async def get_agent(agent_id: str) -> AgentObject:
        agent = agents_by_id.get(agent_id)
        if agent is None:
            raise HTTPException(
                status_code=404, detail="Agent not found"
            )
        return agent

    # ── DELETE /agents/{agent_id} ──────────────────────────────────

    @router.delete("/agents/{agent_id}")
    async def delete_agent(agent_id: str) -> AgentDeleted:
        agent = agents_by_id.pop(agent_id, None)
        if agent is None:
            raise HTTPException(
                status_code=404, detail="Agent not found"
            )

        agents_by_name.pop(agent.name, None)
        await task_store.cancel_by_agent(agent.name)
        artifact_store.delete(agent_id)

        return AgentDeleted(id=agent_id)

    return router
