---
name: agent-plane-dev
description: Development guidelines and conventions for the agent-plane project
---

# Agent-Plane Development Guide

## Project Overview

Agent-plane is a FastAPI server that hosts, manages, and executes agents via an
OpenResponses-compatible API. Python 3.10+, Pydantic v2, async FastAPI.

## Architecture

```
agent_plane/
  spec/        # Agent definition contract (what an agent is)
  runtime/     # Execution engine (models, abstract stores)
  server/      # HTTP layer (FastAPI routes, API models)
  client/      # CLI and typed client library
```

### Layered Design

- **Runtime layer** (`runtime/models.py`, `runtime/stores.py`): core data
  models and abstract store interfaces. No HTTP concepts.
- **Server layer** (`server/app.py`, `server/routes/`, `server/models.py`):
  FastAPI routes, request/response Pydantic models, SSE streaming.
- Stores are injected into `create_app()` — route factories close over them.

### Key Files

| File | Purpose |
|------|---------|
| `runtime/models.py` | `Task`, `Session`, `ConversationItem`, item data types |
| `runtime/stores.py` | Abstract `TaskStore` and `SessionStore` interfaces |
| `server/app.py` | `create_app()` factory, route mounting |
| `server/models.py` | API-layer Pydantic models (`ResponseObject`, etc.) |
| `server/routes/responses.py` | `POST/GET/DELETE /v1/responses`, streaming |
| `server/routes/conversations.py` | Conversation CRUD and item listing |
| `server/routes/agents.py` | Agent CRUD (in-memory storage) |
| `server/routes/files.py` | File upload/download (in-memory storage) |
| `server/API.md` | Full API specification (source of truth) |
| `runtime/RUNTIME.md` | Runtime design document |
| `tests/GAPS.md` | Known implementation gaps |
| `tests/AUDIT_RESPONSES.md` | Spec compliance audit results |

## Data Model Conventions

### When to use dataclasses vs Pydantic

- **Dataclasses**: `Task`, `Session`, `PagedList` — simple containers, no
  complex validation needed. Fields use `field(default_factory=...)` for
  mutable defaults.
- **Pydantic BaseModel**: `ConversationItem`, `NewConversationItem`, all
  `*Data` types, API request/response models — need validation,
  `model_dump()`, `model_validator`.

### ConversationItem type system

Items have a `type` field and a typed `data` field. The mapping is enforced
by a `@model_validator`:

| `type` | `data` class | `model` field? |
|--------|-------------|----------------|
| `"message"` (user) | `MessageData` | No |
| `"message"` (assistant) | `MessageData` | Yes (required) |
| `"function_call"` | `FunctionCallData` | Yes |
| `"function_call_output"` | `FunctionCallOutputData` | No |
| `"reasoning"` | `ReasoningData` | Yes |

`ITEM_TYPE_TO_DATA_CLS` maps type strings to data classes.
`parse_item_data(type, raw_dict)` deserializes from DB.

### API shape for items

`_to_api_item()` in `conversations.py` merges common fields (`id`,
`response_id`, `type`, `status`) with `data.model_dump(exclude_none=True)`.
This means `model` only appears when set (not on user messages or
function_call_outputs).

## Store Interface Conventions

- **Sync methods**: `create`, `get`, `start`, `try_deliver`, `close_inbox`,
  `append`, `search_items`, `list_sessions` — backed by DBOS transactions.
- **Async methods**: `wait`, `stream`, `cancel`, `delete`,
  `delete_session` — long-running or need workflow interaction.
- Pagination uses `after`/`before`/`limit` (cursor-based), returns
  `PagedList[T]` with `next_page_token`.

## Testing Practices

### Always add tests when validating

Every time you validate something about the code (running test scripts,
checking behavior manually), add pytest tests for it. Manual verification
is ephemeral — it proves the code works now but doesn't prevent regressions.

### Running tests

```bash
cd /Users/corey.zumar/agent-plane
python -m pytest tests/ -xvs
```

### Test organization

- `tests/test_models.py` — ConversationItem and data model validation
- Add new tests to existing suites when they logically fit
- Use class-based grouping (e.g., `TestMessageData`, `TestParseItemData`)

### Manual testing with a real agent

Always manually test end-to-end with a real agent, not just unit tests or
mocked stores. This is the only way to verify that the full request
lifecycle (routing, store persistence, agent execution, streaming, output
assembly) works correctly.

<!-- TODO: Add detailed instructions for standing up a real agent and
running manual E2E tests once the runtime machinery is in place. -->

### Verifying OpenResponses compatibility

Our API must behave consistently with OpenAI's Responses API. When
implementing or changing response-related behavior, verify against the
real OpenAI API:

1. **Ask the user for an OpenAI API key** if you don't have one
2. **Send the equivalent request to OpenAI** using `curl` or the OpenAI
   Python SDK to see the real behavior (response shape, field values,
   status codes, streaming event sequence, edge case handling)
3. **Compare our output to theirs** — field names, defaults, nullability,
   ordering, and error responses should match
4. **Pay special attention to**: streaming event ordering, terminal status
   shapes, pagination cursor behavior, and how `previous_response_id`
   threading works in practice

This is especially important for ambiguous spec areas where API.md may
not capture every nuance. The real OpenAI API is the ground truth.

## Spec Compliance Workflow

Spec documents (`.md` files in the source tree) are the source of truth for
how the system should behave. Examples include `server/API.md` (HTTP
contract) and `runtime/RUNTIME.md` (runtime design), but more may be added
over time. Always check for relevant spec docs in the area you're changing.

Steps:

1. Find and read the spec docs relevant to your change
2. Check `tests/GAPS.md` for known gaps and their status
3. Check `tests/AUDIT_RESPONSES.md` for prior audit results
4. When fixing a gap, update GAPS.md to mark it as fixed
5. Route-level enforcement preferred over store-level where possible
6. After changes, re-audit the affected section against the relevant specs

## Common Patterns

### Route factories

Each route module exports a `create_*_router()` factory that returns an
`APIRouter`. Stores and lookup functions are passed as arguments and closed
over (no FastAPI dependency injection).

### SSE streaming

`_format_sse(event_type, data)` formats events. Streaming generators yield
lifecycle events (`response.created`, `response.in_progress`, etc.),
then stream from `task_store.stream()`, then emit a terminal event.

### Disconnect handling

- **Foreground streaming**: `try/finally` in generator, `asyncio.shield`
  to cancel on disconnect.
- **Foreground blocking**: `asyncio.wait` racing task vs disconnect poll.
- **Background**: never cancelled on disconnect.
