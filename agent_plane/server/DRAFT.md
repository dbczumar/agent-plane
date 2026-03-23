# server/ — One way to HOST agents

Depends on: spec/, runtime/

The agent server. Receives agent bundles, stores them, exposes CRUD and
(eventually) invocation APIs over HTTP. Uses the runtime to execute agents.

## Planned files

### app.py
FastAPI application:
- Lifespan management (startup: init DB, shutdown: cleanup)
- Mount API routers
- Middleware (error handling, request logging)

### config.py
Server configuration:
- Database URI (default: SQLite)
- Bundle storage root path
- Working directory root path
- Max bundle size
- Server host/port

### db.py
SQLAlchemy setup:
- Engine and session factory
- Base model class
- Migration support (alembic — future)

### api/agents.py
REST endpoints:
- `POST /api/agents` — upload bundle + metadata, create agent
  - Multipart: bundle tarball + name + optional description
  - Returns 201 with agent metadata
  - Returns 409 if name already exists
- `GET /api/agents/{agent_id}` — retrieve agent metadata
  - Returns DB metadata (no filesystem access)
- `DELETE /api/agents/{agent_id}` — delete agent
  - Removes workdir, bundle, and DB row
  - Returns 204
- `GET /api/agents` — list all agents (future)

### models/agent.py
SQLAlchemy model:
- `agent_id` (UUID, PK)
- `version` (int, PK) — always 1 for now, forward-compatible
- `name` (unique)
- `description`
- `status` (active, deleting)
- `created_at`, `updated_at`
- Cached spec metadata as JSON columns:
  - `config_json` — parsed config.yaml
  - `instructions` — AGENTS.md content
  - `skills_json` — skill metadata array
  - `mcp_servers_json` — MCP server configs
  - `local_tools_json` — local tool info
- `bundle_path` — pointer to stored tarball
- `workdir_path` — pointer to extracted directory

### schemas/agent.py
Pydantic models for API request/response:
- `CreateAgentResponse` — agent_id, name, version, status, spec summary
- `GetAgentResponse` — full metadata including cached spec fields
- Error responses

### services/agent_service.py
Business logic orchestration:
- `create_agent(name, description, bundle_file)`:
  1. Check name uniqueness
  2. Safe-extract tarball (via spec/tar_utils)
  3. Parse spec (via spec/parser)
  4. Validate spec (via spec/validator)
  5. Store bundle (via storage/bundle_store)
  6. Move extracted dir to workdir location
  7. Write DB row with cached metadata
- `get_agent(agent_id)` — DB lookup
- `delete_agent(agent_id)` — mark deleting, remove files, delete row

### storage/bundle_store.py
Abstract interface + local filesystem implementation:
- `store(agent_id, version, tarball_bytes) → bundle_path`
- `retrieve(bundle_path) → bytes`
- `delete(bundle_path)`
- Local impl stores under `{storage_root}/bundles/{agent_id}/{version}.tar.gz`
- Interface is pluggable for S3/DBFS/GCS later

## Key design decisions
- DB metadata is a read-optimized cache; the bundle is the source of truth
- Working directory is ephemeral and re-extractable from the bundle
- No agent execution APIs yet — just CRUD for agent endpoints
- Server depends on runtime/ but doesn't expose execution APIs in v1

## Not yet (future)
- `POST /api/agents/{agent_id}/invoke` — execute agent
- `POST /api/agents/{agent_id}/stream` — streaming execution
- Conversation management APIs
- Agent listing with filtering
- Health / readiness endpoints
