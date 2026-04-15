"""Routes for the /api/agents endpoints."""

import hashlib
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile

from agent_plane.db.utils import generate_agent_id
from agent_plane.entities import Agent
from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.runtime.agent_cache import AgentCache
from agent_plane.server.schemas import AgentDeleted, AgentObject, PaginatedList
from agent_plane.spec import AgentSpec, ExtractionError, load
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
        version=agent.version,
        description=agent.description,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


def _validate_bundle(bundle_bytes: bytes) -> AgentSpec:
    """
    Validate an agent bundle and return the parsed spec.

    Extracts the tarball to a temp directory, parses the spec,
    and checks that a name is present.

    :param bundle_bytes: Raw bytes of the ``.tar.gz`` bundle.
    :returns: The validated :class:`AgentSpec`.
    :raises AgentPlaneError: If the bundle is invalid or the
        spec is missing a name.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load(bundle_bytes, dest=Path(tmpdir) / "agent")
    except AgentPlaneError:
        raise
    except ExtractionError as exc:
        raise AgentPlaneError(str(exc), code=ErrorCode.INVALID_INPUT) from exc
    except Exception as exc:
        # Catch YAML parse errors and other unexpected failures
        # during spec loading so they surface as 400, not 500.
        raise AgentPlaneError(
            f"invalid agent bundle: {exc}",
            code=ErrorCode.INVALID_INPUT,
        ) from exc

    if spec.name is None:
        raise AgentPlaneError(
            "agent spec must include a name",
            code=ErrorCode.INVALID_INPUT,
        )

    return spec


def _bundle_location(agent_id: str, bundle_bytes: bytes) -> str:
    """
    Compute a content-addressed artifact key for a bundle.

    :param agent_id: The agent's unique identifier,
        e.g. ``"ag_abc123"``.
    :param bundle_bytes: Raw bytes of the bundle.
    :returns: Artifact store key in the form
        ``"{agent_id}/{sha256_hex}"``.
    """
    digest = hashlib.sha256(bundle_bytes).hexdigest()
    return f"{agent_id}/{digest}"


def create_agents_router(
    agent_store: AgentStore,
    task_store: TaskStore,
    artifact_store: ArtifactStore,
    agent_cache: AgentCache,
) -> APIRouter:
    """
    Factory that builds the agents router.

    Stores and cache are closed over -- no dependency injection.

    :param agent_store: Store for agent CRUD operations.
    :param task_store: Store for task deletion when an agent is
        deleted.
    :param artifact_store: Store for agent bundle binary blobs.
    :param agent_cache: Cache for loaded agent specs and working
        directories.
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
        the bundle binary under a content-addressed key.

        :param bundle: Uploaded ``.tar.gz`` agent bundle file
            containing a ``config.yaml`` agent spec.
        :returns: The newly created :class:`AgentObject`.
        :raises AgentPlaneError: If the bundle is invalid, the
            spec is missing a name, or an agent with the same
            name already exists.
        """
        bundle_bytes = await bundle.read()
        spec = _validate_bundle(bundle_bytes)
        # _validate_bundle raises if name is None; assert for mypy
        assert spec.name is not None

        if agent_store.get_by_name(spec.name) is not None:
            raise AgentPlaneError(
                f"Agent with name '{spec.name}' already exists",
                code=ErrorCode.ALREADY_EXISTS,
            )

        agent_id = generate_agent_id()
        loc = _bundle_location(agent_id, bundle_bytes)
        artifact_store.put(loc, bundle_bytes)
        agent = agent_store.create(
            agent_id=agent_id,
            name=spec.name,
            bundle_location=loc,
            description=spec.description,
        )

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

    # ── PUT /agents/{agent_id} ─────────────────────────────────────

    @router.put("/agents/{agent_id}")
    async def update_agent(
        agent_id: str,
        bundle: UploadFile = File(...),
    ) -> AgentObject:
        """
        Update an agent with a new bundle.

        Validates the new bundle, checks that the spec name
        matches the existing agent, stores the bundle under a
        content-addressed key, updates the DB, and warm-swaps
        the cache.

        :param agent_id: The agent identifier,
            e.g. ``"ag_abc123"``.
        :param bundle: Uploaded ``.tar.gz`` agent bundle file.
        :returns: The updated :class:`AgentObject`.
        :raises AgentPlaneError: If the agent is not found, the
            bundle is invalid, or the spec name doesn't match.
        """
        bundle_bytes = await bundle.read()
        spec = _validate_bundle(bundle_bytes)
        # _validate_bundle raises if name is None; assert for mypy
        assert spec.name is not None

        existing = agent_store.get(agent_id)
        if existing is None:
            raise AgentPlaneError("Agent not found", code=ErrorCode.NOT_FOUND)

        if spec.name != existing.name:
            raise AgentPlaneError(
                f"spec name '{spec.name}' does not match agent "
                f"name '{existing.name}'; name is immutable",
                code=ErrorCode.INVALID_INPUT,
            )

        new_loc = _bundle_location(agent_id, bundle_bytes)

        # Idempotency: same bundle content = no-op
        if new_loc == existing.bundle_location:
            return _to_agent_object(existing)

        artifact_store.put(new_loc, bundle_bytes)
        updated = agent_store.update(agent_id, new_loc)
        if updated is None:
            # Agent was deleted between get() and update()
            raise AgentPlaneError("Agent not found", code=ErrorCode.NOT_FOUND)

        agent_cache.replace(agent_id, new_loc, bundle_bytes)

        return _to_agent_object(updated)

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
        artifact_store.delete(agent.bundle_location)
        agent_store.delete(agent_id)
        agent_cache.evict(agent_id)

        return AgentDeleted(id=agent_id)

    return router
