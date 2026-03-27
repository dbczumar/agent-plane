"""Routes for the /api/agents endpoints."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile

from agent_plane.entities import Agent
from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.server.schemas import AgentDeleted, AgentObject, PaginatedList
from agent_plane.spec import ExtractionError, load
from agent_plane.stores import AgentStore, ArtifactStore, TaskStore


def _to_agent_object(agent: Agent) -> AgentObject:
    """
    Convert a runtime Agent entity to an API-layer AgentObject.

    :param agent: The runtime agent entity.
    :returns: A :class:`AgentObject` for the API response.
    """
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
    """
    Factory that builds the agents router.

    Stores are closed over -- no dependency injection.

    :param agent_store: Store for agent CRUD operations.
    :param task_store: Store for task deletion when an agent is
        deleted.
    :param artifact_store: Store for agent bundle binary blobs.
    :returns: A configured :class:`APIRouter` with all
        ``/agents`` endpoints.
    """
    router = APIRouter()

    # ── POST /agents ───────────────────────────────────────────────

    @router.post("/agents", status_code=201)
    async def create_agent(
        bundle: UploadFile = File(...),
    ) -> AgentObject:
        """
        Create a new agent from an uploaded bundle archive.

        Validates the bundle, extracts the agent spec, ensures
        the agent name is unique, persists metadata, and stores
        the bundle binary.

        :param bundle: Uploaded ``.tar.gz`` agent bundle file
            containing a ``config.yaml`` agent spec.
        :returns: The newly created :class:`AgentObject`.
        :raises AgentPlaneError: If the bundle is invalid, the
            spec is missing a name, or an agent with the same
            name already exists.
        """
        bundle_bytes = await bundle.read()

        # Validate bundle and extract agent spec
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                spec = load(bundle_bytes, dest=Path(tmpdir) / "agent")
        except AgentPlaneError:
            raise
        except ExtractionError as exc:
            raise AgentPlaneError(str(exc), code=ErrorCode.INVALID_INPUT) from exc

        if spec.name is None:
            raise AgentPlaneError(
                "agent spec must include a name",
                code=ErrorCode.INVALID_INPUT,
            )

        if agent_store.get_by_name(spec.name) is not None:
            raise AgentPlaneError(
                f"Agent with name '{spec.name}' already exists",
                code=ErrorCode.ALREADY_EXISTS,
            )

        agent = agent_store.create(
            name=spec.name,
            description=spec.description,
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
        """
        List agents with cursor-based pagination.

        :param limit: Maximum number of items to return
            (1-100).
        :param after: Cursor — return items after this agent
            ID, e.g. ``"ag_abc123"``.
        :param before: Cursor — return items before this
            agent ID.
        :param order: Sort order, ``"asc"`` or ``"desc"``.
        :returns: A :class:`PaginatedList` of agents.
        """
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
        """
        Retrieve a single agent by ID.

        :param agent_id: The agent identifier,
            e.g. ``"ag_abc123"``.
        :returns: The matching :class:`AgentObject`.
        :raises AgentPlaneError: If the agent is not found.
        """
        agent = agent_store.get(agent_id)
        if agent is None:
            raise AgentPlaneError("Agent not found", code=ErrorCode.NOT_FOUND)
        return _to_agent_object(agent)

    # ── DELETE /agents/{agent_id} ──────────────────────────────────

    @router.delete("/agents/{agent_id}")
    async def delete_agent(agent_id: str) -> AgentDeleted:
        """
        Delete an agent, its tasks, and its bundle artifact.

        :param agent_id: The agent identifier,
            e.g. ``"ag_abc123"``.
        :returns: An :class:`AgentDeleted` confirmation.
        :raises AgentPlaneError: If the agent is not found.
        """
        agent = agent_store.get(agent_id)
        if agent is None:
            raise AgentPlaneError("Agent not found", code=ErrorCode.NOT_FOUND)

        await task_store.delete_all(agent_id=agent_id)
        artifact_store.delete(agent_id)
        agent_store.delete(agent_id)

        return AgentDeleted(id=agent_id)

    return router
