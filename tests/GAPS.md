# Implementation Gaps

Issues found during spec compliance audit of the FastAPI server against
API.md and RUNTIME.md. Grouped by severity.

---

## Fixed in this pass

These items from AUDIT_RESPONSES.md are now resolved:

1. **`context_management` missing from request model** (audit item 14/54)
   Fixed: added `context_management: list[Any] | None = None` to
   `CreateResponseRequest` in `models.py`.

2. **Missing `response.queued` event for background streaming** (audit item 19)
   Fixed: streaming generator now emits `response.queued` between
   `response.created` and `response.in_progress` when `req.background`
   is true. `responses.py` lines 234-244.

3. **No disconnect cancellation for foreground requests** (audit item 51)
   Fixed in both paths:
   - Streaming: `try/finally` in generator cancels the task via
     `asyncio.shield(task_store.cancel(...))` when the stream ends
     abnormally (client disconnect) and `background` is false.
     `responses.py` lines 297-311.
   - Blocking: `asyncio.wait` races `task_store.wait()` against
     `_poll_disconnect(request)`. On disconnect, calls
     `task_store.cancel()`. `responses.py` lines 318-339.

4. **Missing `order` query parameter on all list endpoints**
   Fixed: added `order` param (asc/desc) to:
   - `GET /api/agents` (default: desc) — `agents.py`
   - `GET /v1/files` (default: desc) — `files.py`
   - `GET /v1/conversations` (default: desc) — `conversations.py`
   - `GET /v1/conversations/{id}/items` (default: asc) — `conversations.py`

---

## Needs runtime support

These require changes to the runtime layer (Task dataclass, store
interfaces, or store implementations) before the route layer can
comply with the spec.

### ~~1. Usage always null (audit item 49)~~ — FIXED

Added `usage: dict | None = None` to `Task` (`runtime/models.py`).
`_build_response_object()` now maps `task.usage` to a `Usage` model
via `Usage(**task.usage)`. Runtime must populate `task.usage` on
completion with `{input_tokens, output_tokens, output_tokens_details,
total_tokens}` from the LLM provider.

### 2. Fork detection / conversation splitting (audit item 53)

**Spec**: API.md lines 215-220 — when `previous_response_id` points to a
non-latest response (no explicit `conversation`), the server creates a new
conversation, copies items up to the fork point, and adds the new response
there.

**Gap**: Currently, forks without an explicit `conversation` field silently
append to the existing conversation. Fork detection only triggers as a 400
error when the caller explicitly passes `conversation`.

**Status**: Already listed in RUNTIME.md "Not Yet" section. Requires:
- `session_store.fork_session(source_session_id, fork_at_response_id)`
  method to copy items and create a new session.
- Route logic in `responses.py` to detect implicit forks (when
  `previous_response_id` is not the latest response and no `conversation`
  is provided) and call the fork method.

### ~~3. `model` field on conversation items~~ — FIXED

### ~~4. Non-message item types in conversation items~~ — FIXED

Both resolved by replacing `NewMessage`/`Message` with a generic
`NewConversationItem`/`ConversationItem` model (`runtime/models.py`).
Items have `type` ("message", "function_call", etc.) and a `data` dict
for type-specific fields. The `model` field lives in `data` only for
model-produced items (assistant messages, function calls, reasoning).
User messages and function call outputs omit it. The `_to_api_item()`
helper in `conversations.py` merges common fields with `data` to
produce the API shape. Store methods renamed: `search_messages` →
`search_items`, `last_seen_message_id` → `last_seen_item_id`.

### ~~4. Non-message item types in conversation items~~ — FIXED (see #3)

### ~~5. `before` pagination on conversation items~~ — FIXED

`search_items` now takes `after`/`before`/`limit` directly (replaced
`page_token`/`max_results`/`order_by`). Route passes both `after` and
`before` from query params through to the store.

### ~~6. GET output empty for non-completed statuses (audit item 27)~~ — FIXED

Route-level enforcement in `_build_response_object()`: returns
`task.output` only when `task.status == "completed"`, otherwise `[]`.

---

## Needs design decision

### ~~7. Agent delete: cancel in-flight responses~~ — FIXED

`task_store` is now passed to `create_agents_router()`. Added
`cancel_by_agent(agent)` to `TaskStore`. `delete_agent` calls
`task_store.cancel_by_agent(agent.name)` before removing the agent.

### ~~8a. Agent bundle storage~~ — FIXED

Bundle bytes are now read from the `UploadFile` and stored via
`artifact_store.put(agent_id, bundle_bytes)` in `create_agent`.
Deleted via `artifact_store.delete(agent_id)` in `delete_agent`.

### 8b. Agent bundle validation

**Spec**: API.md line 57: "400 Bad Request — invalid bundle."

**Gap**: Bundle bytes are stored but never validated. Need to define
and enforce tarball structure requirements (e.g. required files like
`AGENTS.md`, safe extraction, size limits).

### 9. Conversation delete: cancel in-flight responses

**Spec**: API.md line 334 — "Cancels any in-flight responses in the
conversation before deleting."

**Status**: The `session_store.delete_session()` docstring says it "may
need to cancel in-flight responses." This is delegated to the store
implementation. The route layer is correct — it awaits the async
`delete_session()` call. The store implementation must handle cancellation
internally (likely by querying task_store for in-flight tasks in the
session and cancelling them before deleting).

---

## Low priority / cosmetic

### 10. Background streaming: durable execution correctness

**Spec**: API.md line 705 — background execution continues on disconnect.

**Status**: This works accidentally because the streaming path has no
disconnect-cancel logic for background requests (the `finally` block
checks `req.background` and skips cancellation). Now that foreground
disconnect cancellation is implemented, this is explicitly correct — the
`not req.background` guard ensures background tasks are never cancelled
on disconnect.

### 11. `context_management` not passed to runtime

`context_management` is now accepted on the request model but not passed
to `task_store.create()` or the runtime. The field needs to be:
- Added to `task_store.create()` signature
- Added to the `Task` dataclass
- Used by the runtime to configure compaction behavior
