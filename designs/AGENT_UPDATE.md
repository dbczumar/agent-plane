# Plan: Agent Update Endpoint (`PUT /api/agents/{id}`)

## Context

Agents deployed to agent-plane are currently immutable after creation. The only
lifecycle operations are create and delete. `PUT /api/agents/{id}` is explicitly
reserved in API.md's "Not Yet" section. This plan adds that endpoint so a
deployed agent can be updated with a new bundle without changing its identity
(agent ID stays the same).

## Design Decisions

1. **Full bundle replacement** — the update accepts a new tarball and replaces
   the stored bundle entirely. No partial/field-only updates. This follows the
   spec self-containment principle (behavior is determined by the spec).

2. **Name is immutable** — the agent's name cannot change on update. If the new
   bundle's spec has a different name than the existing agent, the server rejects
   the request with 400. Name is the public API contract (`model` field in
   `POST /v1/responses`) and changing it would break clients.

3. **Active tasks continue on old spec** — unlike DELETE (which cancels all
   tasks), UPDATE lets in-flight tasks finish with their already-loaded spec.
   Only future tasks use the new bundle.

4. **Content-addressed artifact keys** — each bundle is stored at
   `"{agent_id}/{sha256_hex}"` where `sha256_hex` is the hex-encoded SHA-256
   hash of the bundle bytes. The `Agent` record tracks:
   - `version: int` — monotonic counter (starts at 1, incremented on update).
     For display/API purposes.
   - `bundle_location: str` — the artifact store key for the current bundle
     (e.g. `"ag_abc123/a1b2c3..."`). This is what `AgentCache.load()` uses
     to fetch the bundle.

   This provides:
   - **No data loss on failure** — old bundle at its own key is untouched.
   - **No corruption on concurrent updates** — each update writes to a
     content-derived key, so concurrent PUTs never overwrite each other's
     bundles (unless they contain the same content, which is fine).
   - **Idempotent retries** — if the client retries with the same bundle,
     the hash is the same, the artifact write is a no-op (same key, same
     bytes), and the DB update is skipped (bundle_location unchanged).
     No version bump, no orphans, no side effects.
   - **Future rollback** — old bundles remain in the artifact store; point
     `bundle_location` back to a previous key.
   - **Atomic transition** — the DB row update is the single commit point.
   - **Negligible overhead** — SHA-256 of a 100MB bundle takes ~40ms;
     realistic bundles (< 1MB) hash in under 1ms.

   Concurrent updates: last writer wins. The loser's bundle is only an orphan
   if its content differs from the winner's — identical bundles share a key.

5. **Warm cache swap** — instead of evict-then-reload (which causes a cache miss
   latency spike), add a `replace(agent_id, bundle_bytes)` method to
   `AgentCache` that extracts the new bundle to a temp directory, swaps the
   in-memory entry, then renames the directory into the cache location. Concurrent
   readers see either the old spec or the new spec, never an empty cache.

6. **Add `updated_at` field** — nullable `int | None`, `None` for agents never
   updated. Follows the `Conversation.updated_at` precedent.

7. **Pass `AgentCache` to the router** — the router factory already takes stores
   as closure params. Adding `agent_cache` is explicit and avoids a hidden
   runtime dependency.

## Update Flow (step by step)

1. Client sends `PUT /api/agents/{id}` with new tarball
2. Server validates bundle (extract to temp dir, parse spec, check name match)
3. Server reads existing agent from DB
4. Server computes `new_loc = f"{agent_id}/{sha256(bundle_bytes).hex()}"`
5. **Idempotency check**: if `new_loc == existing.bundle_location`, the bundle
   is identical to what's already deployed. Return the current `AgentObject`
   with HTTP 200 — no version bump, no writes, no side effects.
6. Server writes new bundle: `artifact_store.put(new_loc, bytes)` — old bundle
   untouched; concurrent updates with different content write to different keys
7. Server updates DB row: `agent_store.update(agent_id, new_loc)` — bumps
   version, sets bundle_location and updated_at
8. Server warm-swaps the cache
9. Return updated `AgentObject` with HTTP 200

**Failure at step 6** (artifact write fails): DB unchanged, old agent works.
**Failure at step 7** (DB write fails): orphaned bundle at `new_loc`, old
agent works. Orphan is harmless.
**Failure at step 8** (cache swap fails): DB points to new bundle, next
`load()` re-downloads from artifact store (slow but correct).

## Files to Modify

### 1. Entity: `agent_plane/entities/agent.py`
- Add `version: int = 1` to the `Agent` dataclass
- Add `bundle_location: str` to the `Agent` dataclass (set at creation)
- Add `updated_at: int | None = None` to the `Agent` dataclass

### 2. DB model: `agent_plane/db/db_models.py`
- Add `version` Integer column to `SqlAgent` (NOT NULL, server_default=1)
- Add `bundle_location` String column to `SqlAgent` (NOT NULL)
- Add `updated_at` nullable Integer column to `SqlAgent`

### 3. Alembic migration: new file in `agent_plane/db/migrations/versions/`
- `down_revision = "43fb65b29464"`
- Add `version` Integer column (NOT NULL, server_default="1")
- Add `bundle_location` String column (NOT NULL — for existing rows,
  backfill with `agent_id` since that's the current artifact key format)
- Add nullable `updated_at` Integer column

### 4. API schema: `agent_plane/server/schemas.py`
- Add `version: int = 1` to `AgentObject`
- Add `updated_at: int | None = None` to `AgentObject`
- (`bundle_location` is internal — not exposed in the API)

### 5. Abstract store: `agent_plane/stores/agent_store/__init__.py`
- Add abstract method `update(agent_id, bundle_location) -> Agent | None`
  — fetches the row, bumps `version`, sets `bundle_location` and
  `updated_at`. Returns updated Agent, or None if agent doesn't exist.

### 6. SQLAlchemy store: `agent_plane/stores/agent_store/sqlalchemy_store.py`
- Implement `update()`: fetch row, increment version, set bundle_location
  and updated_at, return entity
- Update `_to_entity` to include `version`, `bundle_location`, `updated_at`
- Update `create()` to set initial `bundle_location`

### 7. Agent cache: `agent_plane/runtime/agent_cache.py`
- Change `load(agent_id)` to `load(agent_id, bundle_location)` — uses
  `bundle_location` as the artifact store key
- Add `replace(agent_id, bundle_location, bundle_bytes)` method for warm
  swap: extract new bundle to temp dir, swap in-memory entry, rename into
  cache location, clean up old dir
- Update callers of `load()` to pass `bundle_location` (runtime workflow
  gets it from the agent store)

### 8. Routes: `agent_plane/server/routes/agents.py`
- Extract bundle validation from `create_agent` into
  `_validate_bundle(bundle_bytes) -> AgentSpec`
- Update `create_agents_router` signature to accept `agent_cache: AgentCache`
- Update `create_agent` to compute `bundle_location` from content hash
  and store bundle under that key
- Add `PUT /agents/{agent_id}` handler implementing the update flow above
- Update `_to_agent_object` to include `version` and `updated_at`

### 9. Delete handler: `agent_plane/server/routes/agents.py`
- Update `delete_agent` to delete the bundle at `agent.bundle_location`.
  Old version bundles become orphans — acceptable for now, can add GC later.

### 10. App factory: `agent_plane/server/app.py`
- Add `agent_cache: AgentCache` parameter to `create_app`
- Pass it to `create_agents_router`

### 11. Callers of `create_app` (thread `agent_cache` through)
- `agent_plane/cli.py` (~line 226) — `agent_cache` is already constructed
  at line 210
- `examples/hosting/databricks-apps/app.py` (~line 105) — extract existing
  `AgentCache` at line 96 into a local variable and pass to `create_app`
- `tests/server/conftest.py` (~line 416) — construct `AgentCache` in fixture
- `tests/server/integration/durability_helpers.py` (~line 189) — construct
  `AgentCache`

### 12. Runtime workflow: `agent_plane/runtime/workflow.py`
- Update `agent_cache.load(agent_id)` calls to pass `bundle_location` —
  the workflow looks up the agent from the agent store to get the current
  `bundle_location`

### 13. API spec: `agent_plane/server/API.md`
- Add `PUT /api/agents/{id}` section with request/response docs
- Add `version` and `updated_at` to agent object schema
- Remove `PUT /api/agents/{id}` from "Not Yet" section

### 14. Tests

**Store tests** (`tests/stores/test_agent_store.py`):
- `test_update_agent` — create, update, verify version=2 and new
  bundle_location
- `test_update_nonexistent_agent` — returns None
- `test_update_sets_updated_at` — verify updated_at is populated
- `test_update_increments_version` — create (v1), update (v2), update (v3)
- `test_create_agent_has_version_1` — newly created agents have version=1

**Cache tests** (`tests/runtime/test_agent_cache.py`):
- `test_load_with_bundle_location` — load uses bundle_location as artifact key
- `test_replace_swaps_spec` — replace returns new spec, old spec gone
- `test_replace_concurrent_load` — load during replace returns either old
  or new, never fails

**Route integration tests** (`tests/server/integration/test_routes_agents.py`):
- `test_update_agent` — PUT new bundle, verify 200 with version=2
- `test_update_agent_not_found` — 404
- `test_update_agent_name_mismatch` — PUT bundle with different spec name
  than existing agent, verify 400
- `test_update_agent_invalid_bundle` — corrupt bytes, verify 400
- `test_update_preserves_old_bundle` — after update, old bundle_location
  still exists in artifact store
- `test_update_same_bundle_is_idempotent` — PUT identical bundle twice,
  verify version doesn't bump on the second call
- `test_create_agent_has_null_updated_at` — newly created agents have
  updated_at=null, version=1

**E2E test** (`tests/e2e/test_agent_update.py`):

Uses a real LLM and a real `ap server` subprocess (same pattern as other e2e
tests). The test proves that an in-flight request on the old version completes
successfully, and a new request after the update uses the new version.

- `test_update_agent_zero_downtime`:
  1. Upload archer agent (version 1)
  2. Send a long-running background request to archer (e.g. "Research the
     history and current state of quantum computing. Be thorough — cover
     at least 5 major milestones.")
  3. While that request is still in progress (poll confirms `in_progress`),
     PUT a new bundle for the agent (same name, updated description in
     config.yaml) — this bumps to version 2
  4. Verify the PUT returns 200 with `version=2`
  5. Send a second background request to archer (e.g. "What is 2+2? Be
     brief, one sentence.")
  6. Poll both requests to terminal state
  7. Assert both completed successfully — the old request wasn't disrupted
     by the update, and the new request ran against the updated agent
  8. Verify the agent's GET response shows version=2 and updated_at is set

Helper additions to `tests/e2e/conftest.py`:
- `_update_agent(client, agent_id, agent_dir)` — builds a tarball and
  PUTs it to `/api/agents/{agent_id}`
- A variant of the archer fixture that returns the agent ID (not just name),
  since PUT requires the ID

## Verification

1. Run store-level tests: `pytest tests/stores/test_agent_store.py -xvs`
2. Run cache tests: `pytest tests/runtime/test_agent_cache.py -xvs`
3. Run route integration tests:
   `pytest tests/server/integration/test_routes_agents.py -xvs`
4. Run full test suite: `pytest tests/ -x`
5. Run e2e test:
   `pytest tests/e2e/test_agent_update.py --llm-api-key $LLM_API_KEY -v`
6. Manual curl test against a running server:
   - Create agent, verify `version=1`, `updated_at` is null
   - PUT with new bundle, verify `version=2`, `updated_at` is set
   - Send a request to the agent, verify it uses the new spec
   - Verify old bundle still exists in artifact store
