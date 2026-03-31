# Sub-Agent Client-Side Tools

## Context

This document covers the client-side tool tunneling mechanism for
spawned sub-agents. See `SUBAGENT.md` for the spawn/collect tools
and overall sub-agent execution model.

The spawned model has a hard problem: **client-side tools**. When a
spawned sub-agent needs a client-side tool, the workflow must pause, the
client must execute the tool, and the workflow must resume — all while
`collect_sub_agents` waits for the sub-agent to finish.

---

## Design: Tunneled Tool Calls (Single-Response Model)

Sub-agent client-side tool calls are **tunneled through the parent's
response**. The client interacts with a single response ID (the
parent's). It never needs to discover sub-agent task IDs, subscribe to
sub-agent streams, or manage multiple response lifecycles.

When a sub-agent hits a client-side tool:
1. The `function_call` is published to the **parent's** response output
   with `status: "action_required"` and `model` set to `"parent.child"`
2. The client sees it via `GET /v1/responses/{parent_id}` or the
   parent's SSE stream
3. The client submits tool results via
   `PATCH /v1/responses/{parent_id}`
4. The server routes the result to the correct sub-agent's inbox
5. The sub-agent resumes

### Why tunneled (not per-stream)

| Approach | Client complexity | Server complexity | Streams to manage |
|----------|-------------------|-------------------|-------------------|
| Per-stream | High — N+1 streams, discover sub-agent IDs | Low — self-contained | N+1 |
| **Tunneled** | **Low — one response, one stream** | **Medium — cross-workflow publishing** | **1** |

Per-stream pushes orchestration to the client: discover sub-agent task
IDs from spawn output, subscribe to each stream, track which streams
are done, know when the parent is truly finished. The tunneled approach
keeps this server-side where it belongs. The client just sees tool
calls and responds to them.

---

## Signal: `action_required` Function Call Status

OpenAI's Responses API defines three statuses for `function_call`
output items: `in_progress`, `completed`, `incomplete`. We extend this
with a fourth:

- **`action_required`** — the client must execute this tool call and
  submit the result via `PATCH /v1/responses/{id}`.

This disambiguates from `in_progress` (which means "server is
processing this"). The client's logic:

```
for each function_call in response.output:
    if function_call.status == "action_required":
        execute tool, PATCH result back
```

The `action_required` status is set when a sub-agent's client-side tool
call is published to the parent's output. It transitions to `completed`
when the client submits the result.

---

## Data Model

### `pending_tool_calls` table

Single table tracking the full lifecycle of a tunneled client-side
tool call — from the sub-agent parking to the client delivering the
result.

```
pending_tool_calls:
  call_id         String(64)  PRIMARY KEY
  root_task_id    String(64)  NOT NULL  FK → tasks.id ON DELETE CASCADE
  task_id         String(64)  NOT NULL  FK → tasks.id ON DELETE CASCADE
  status          String(32)  NOT NULL  CHECK(status IN ("action_required", "completed"))
  result          Text        NULLABLE  — NULL until completed
  created_at      Integer     NOT NULL  — Unix epoch, when the sub-agent parked
  completed_at    Integer     NULLABLE  — Unix epoch, NULL until completed

  INDEX(root_task_id)   — PATCH endpoint looks up by root_task_id
  INDEX(task_id)        — park loop polls by task_id + status
```

- **`call_id`**: `String(64)` PK — matches LLM-generated tool call
  IDs (same length as other ID columns)
- **FK cascades**: both FKs use `ON DELETE CASCADE`. If the root task
  or sub-agent task is deleted, pending rows are cleaned up
  automatically. No orphaned rows.
- **`created_at`**: tracks when the sub-agent parked. Useful for
  debugging stale rows and for timeout calculations
  (created_at + timeout = deadline).
- **Indexes**: `root_task_id` for the PATCH endpoint (find all
  pending calls for a root response). `task_id` for the park loop
  (poll for completed results for this sub-agent).

**Lifecycle:**
1. Sub-agent parks → INSERT with `status="action_required"`,
   `result=NULL`, `completed_at=NULL`, `created_at=<now>`
2. Client PATCHes → UPDATE `status="completed"`,
   `result=<tool output>`, `completed_at=<now>`
3. Sub-agent poll loop → SELECT WHERE `task_id=T2 AND
   status="completed"` → finds result, resumes

The PK on `call_id` enforces idempotency — a client retrying PATCH
cannot deliver duplicate results. Status values match the external
API's `function_call` status field.

**Schema migration:** The `pending_tool_calls` table is a new table
added to `db_models.py`. No separate Alembic migration — it's picked
up by `Base.metadata.create_all()`. The `root_task_id` and `kind`
columns on existing tables (`tasks` and `conversations` respectively)
are also added by updating the existing model definitions in
`db_models.py` (see `SUBAGENT.md` schema migration section).

### `PendingToolCall` entity

```python
@dataclass
class PendingToolCall:
    """
    A tunneled client-side tool call awaiting client execution.

    Represents one row in the ``pending_tool_calls`` table. Created
    when a sub-agent parks for a client-side tool, completed when
    the client PATCHes the result.

    :param call_id: Tool call ID (PK), matches the LLM-generated
        call ID, e.g. ``"call_abc123"``.
    :param root_task_id: The top-level task whose response output
        contains the ``function_call`` item, e.g. ``"task_root1"``.
    :param task_id: The parked sub-agent's task ID,
        e.g. ``"task_sub2"``.
    :param status: ``"action_required"`` or ``"completed"``.
    :param result: The tool's string output from the client.
        ``None`` until the client PATCHes.
    :param created_at: Unix epoch when the sub-agent parked.
    :param completed_at: Unix epoch when the client PATCHed.
        ``None`` until completed.
    """

    call_id: str
    root_task_id: str
    task_id: str
    status: str
    result: str | None
    created_at: int
    completed_at: int | None
```

Defined in `entities/pending_tool_call.py`.

### `CompletePendingToolCallResult` — return type for completion

```python
class CompletePendingToolCallResult(str, Enum):
    """
    Outcome of attempting to complete a pending tool call.

    Used by the PATCH handler to map to HTTP status codes.

    :cvar COMPLETED: Row updated from action_required → completed.
    :cvar NOT_FOUND: call_id does not exist in the table.
    :cvar ALREADY_COMPLETED: Row already has status=completed
        (idempotent re-PATCH). First writer wins — the stored
        result is not overwritten.
    :cvar SUB_AGENT_DONE: Row exists but the sub-agent's task has
        reached a terminal status (completed, failed, cancelled).
        The tool result cannot be delivered because no one is
        waiting for it.
    """

    COMPLETED = "completed"
    NOT_FOUND = "not_found"
    ALREADY_COMPLETED = "already_completed"
    SUB_AGENT_DONE = "sub_agent_done"
```

Defined alongside `PendingToolCall` in `entities/pending_tool_call.py`.

### `TaskStore` methods for pending tool calls

The `pending_tool_calls` table is owned by `TaskStore` — it's
task-scoped data (keyed by task IDs, FK to tasks). Three new
**abstract methods** added to the `TaskStore` interface in
`stores/task_store/__init__.py` (alongside `create`, `get`, `wait`,
etc.) with implementations in `sqlalchemy_store.py`:

```python
@abstractmethod
def create_pending_tool_call(
    self,
    call_id: str,
    root_task_id: str,
    task_id: str,
) -> None:
    """
    Insert a routing entry for a tunneled client-side tool call.
    Uses INSERT ON CONFLICT DO NOTHING for DBOS replay safety.

    :param call_id: The tool call ID (PK),
        e.g. ``"call_abc123"``.
    :param root_task_id: The root task whose response output
        contains the function_call item, e.g. ``"task_root1"``.
    :param task_id: The parked sub-agent's task ID,
        e.g. ``"task_sub2"``.
    """

@abstractmethod
def complete_pending_tool_call(
    self,
    call_id: str,
    result: str,
) -> CompletePendingToolCallResult:
    """
    Attempt to mark a pending tool call as completed.

    Checks three conditions in order:
    1. Row exists? If not → NOT_FOUND.
    2. Row already completed? → ALREADY_COMPLETED (no-op,
       first writer wins — stored result is NOT overwritten).
    3. Sub-agent task still running? If terminal →
       SUB_AGENT_DONE (row is NOT updated).
    4. Otherwise → UPDATE to completed, return COMPLETED.

    The sub-agent task status check (step 3) is performed
    inside this method because TaskStore already owns both the
    pending_tool_calls table and the tasks table.

    :param call_id: The tool call ID, e.g. ``"call_abc123"``.
    :param result: The tool's string output from the client.
    :returns: The outcome — caller maps to HTTP status codes.
    """

@abstractmethod
def get_pending_tool_calls(
    self,
    task_id: str,
    status: str | None = None,
) -> list[PendingToolCall]:
    """
    Query pending tool calls for a task, optionally filtered
    by status. The park loop calls this with
    status="completed" to find delivered results.

    :param task_id: The sub-agent's task ID,
        e.g. ``"task_sub2"``.
    :param status: Optional status filter. ``"completed"``
        returns only delivered results. ``"action_required"``
        returns only waiting calls. ``None`` returns all.
    :returns: Matching pending tool call rows.
    """
```

---

## Request Flow

Concrete example: parent agent "orchestrator" spawns a "researcher"
sub-agent. The researcher needs a client-side tool (`search_client_db`).

### Actors

- **Client** — makes HTTP requests, subscribes to SSE streams
- **Server** — HTTP layer, routes to stores/workflows
- **W1** — parent DBOS workflow (orchestrator)
- **W2** — sub-agent DBOS workflow (researcher)
- **LLM** — called by workflows

### Full lifecycle

```
STEP  ACTOR              ACTION
────  ─────              ──────

 1    Client → Server    POST /v1/responses
                         { agent_id: "orchestrator",
                           input: "research quantum computing" }

 2    Server             Creates task T1 (parent), starts DBOS workflow W1.
                         Returns { id: R1, status: "in_progress" }

 3    Client → Server    GET /v1/responses/R1 (poll)   (SSE subscription)

 4    W1 (parent)        Calls LLM. LLM returns:
                           tool_call: spawn_sub_agents("researcher", "find papers")

 5    W1 (parent)        SpawnTool.invoke() executes:
                           a. Creates conversation C2
                           b. Creates task T2 with root_task_id=T1
                           c. Starts DBOS workflow W2
                           d. Returns tool result: '{"response_ids": ["T2"]}'

      Client ← SSE(R1)  ← function_call (spawn) [status: "completed"]
                         ← function_call_output '{"response_ids":["T2"]}'

 6    W1 (parent)        Next LLM turn. LLM returns:
                           tool_call: collect_sub_agents(response_ids=["T2"])
                         CollectTool.invoke() blocks via DBOS handle.
                         W1 blocks here.

      Client ← SSE(R1)  ← function_call (collect) [status: "in_progress"]
```

At this point W2 is running independently:

```
 7    W2 (researcher)    Calls LLM. LLM returns:
                           tool_call: search_client_db(query="quantum")
                         search_client_db is a CLIENT-SIDE tool.

 8    W2 (researcher)    Detects client-side tool call. PARKS:
                           a. Writes routing entry FIRST (via task_store):
                              (call_id="call_abc", root_task_id=T1, task_id=T2)
                           b. Then publishes function_call to PARENT's output
                              (via conv_store.append on root's conversation):
                              { type: "function_call",
                                name: "search_client_db",
                                call_id: "call_abc",
                                status: "action_required",
                                model: "orchestrator.researcher" }
                           c. Enters park loop: poll pending_tool_calls table
                              for T2 every 500ms

      Client ← SSE(R1)  ← function_call (search_client_db)
                           [call_id: "call_abc",
                            arguments: "{\"query\":\"quantum\"}",
                            status: "action_required",
                            model: "orchestrator.researcher"]
```

Client sees the tool call on the PARENT's stream and responds:

```
 9    Client             Sees function_call with status "action_required".
                         Executes search_client_db locally.
                         Gets results: ["paper1.pdf", "paper2.pdf"]

10    Client → Server    PATCH /v1/responses/R1
                         { tool_results: [{
                             call_id: "call_abc",
                             output: '["paper1.pdf","paper2.pdf"]'
                         }] }

11    Server             Looks up call_abc in pending_tool_calls table →
                         task_id = T2.
                         Updates pending_tool_calls row:
                           status="completed",
                           result='["paper1.pdf","paper2.pdf"]',
                           completed_at=<now>
                         Updates function_call item status to "completed"
                         in parent's output.
                         Returns R1 with updated output.
```

W2 resumes:

```
12    W2 (researcher)    Park loop reads inbox → finds result for call_abc.
                         All pending tool results received. Exits park loop.
                         Feeds tool results back to LLM.

13    W2 (researcher)    LLM processes results. Returns final text:
                         "Found 2 relevant papers on quantum computing..."
                         No more tool calls → workflow completes normally.
```

Back to parent:

```
14    W1 (parent)        handle.get_result() returns — T2 completed.
                         CollectTool reads T2's final output.
                         Returns to LLM: '{"results": [{"response_id": "T2",
                           "agent_name": "researcher", "status": "completed",
                           "output": "Found 2 relevant papers..."}]}'

      Client ← SSE(R1)  ← function_call_output (collect) [status: "completed"]

15    W1 (parent)        LLM synthesizes final answer:
                         "Based on the research, here's a summary..."
                         No more tool calls → workflow completes.

      Client ← SSE(R1)  ← message "Based on the research..."
                         ← response.completed
```

---

## What the Client Sees: `GET /v1/responses/R1` at Each Stage

### Stage 1: Parent running, no sub-agent tool calls yet

```json
{
  "status": "in_progress",
  "output": [
    { "type": "message", "content": [{"type": "output_text", "text": "I'll research this..."}] },
    { "type": "function_call", "name": "spawn_sub_agents",
      "call_id": "call_1", "status": "completed" },
    { "type": "function_call_output", "call_id": "call_1",
      "output": "{\"response_ids\": [\"T2\"]}" },
    { "type": "function_call", "name": "collect_sub_agents",
      "call_id": "call_2", "status": "in_progress" }
  ]
}
```

Client sees server-side tools (`completed` and `in_progress`). Nothing
to do — wait for more output.

### Stage 2: Sub-agent needs client tool

```json
{
  "status": "in_progress",
  "output": [
    ...previous items...,
    { "type": "function_call", "name": "search_client_db",
      "call_id": "call_abc", "status": "action_required",
      "model": "orchestrator.researcher",
      "arguments": "{\"query\":\"quantum\"}" }
  ]
}
```

Client sees `status: "action_required"` — execute tool, PATCH result.

### Stage 3: After client PATCHes tool result

```json
{
  "status": "in_progress",
  "output": [
    ...previous items...,
    { "type": "function_call", "name": "search_client_db",
      "call_id": "call_abc", "status": "completed",
      "model": "orchestrator.researcher" },
    { "type": "function_call_output", "call_id": "call_abc",
      "output": "[\"paper1.pdf\",\"paper2.pdf\"]" }
  ]
}
```

Tool call transitioned to `completed`. Client knows it's handled.

### Stage 4: Everything done

```json
{
  "status": "completed",
  "output": [
    { "type": "message", ... },
    { "type": "function_call", "name": "spawn_sub_agents", "status": "completed", ... },
    { "type": "function_call_output", "call_id": "call_1", ... },
    { "type": "function_call", "name": "collect_sub_agents", "status": "completed", ... },
    { "type": "function_call", "name": "search_client_db",
      "status": "completed", "model": "orchestrator.researcher", ... },
    { "type": "function_call_output", "call_id": "call_abc", ... },
    { "type": "function_call_output", "call_id": "call_2",
      "output": "{\"results\": [{...}]}" },
    { "type": "message", "content": [{"type": "output_text",
      "text": "Based on the research..."}] }
  ]
}
```

---

## Multi-Round Client Tools (Sub-Agent Parks Twice)

If the sub-agent needs multiple rounds of client-side tools, the same
park/publish/PATCH cycle repeats:

```
12    W2 (researcher)    Picks up first tool result. Feeds to LLM.
                         LLM returns ANOTHER client-side tool call:
                           tool_call: download_pdf(url="paper1.pdf")

13    W2 (researcher)    Parks again. Same mechanism (routing first, item second):
                           a. Write routing entry for call_def (task_store)
                           b. Publish to parent's output (conv_store.append):
                              function_call (download_pdf)
                              [status: "action_required", model: "orchestrator.researcher"]
                           c. Poll inbox

      Client ← SSE(R1)  ← function_call (download_pdf)
                           [call_id: "call_def",
                            arguments: "{\"url\":\"paper1.pdf\"}",
                            status: "action_required",
                            model: "orchestrator.researcher"]

14    Client             Sees new action_required tool call.
                         Executes download_pdf locally.

15    Client → Server    PATCH /v1/responses/R1
                         { tool_results: [{ call_id: "call_def",
                           output: "<pdf content>" }] }

16    Server             Routes to T2 via pending_tool_calls table, updates row.

17    W2 (researcher)    Picks up result, feeds to LLM, LLM finishes.
                         Workflow completes.

      ... collect returns, parent continues as before ...
```

Key: **same parent response R1** throughout. The client never interacts
with T2 directly.

---

## Parallel Sub-Agents with Independent Client Tools

Multiple sub-agents can independently request client-side tools. All
surface on the parent's output:

```
      W1 spawns T2 (researcher) and T3 (analyst)
      W1 calls collect_sub_agents(["T2", "T3"]) — blocks on both

      T2 parks (needs search_client_db) → publishes to R1's output
      T3 parks (needs query_db) → publishes to R1's output

      Client sees on R1:
        function_call: search_client_db [action_required, model: "orchestrator.researcher"]
        function_call: query_db [action_required, model: "orchestrator.analyst"]

      Client PATCHes both results on R1.
      Server routes each to the correct sub-agent's inbox.

      T2 resumes, completes.    T3 resumes, completes.
      collect returns with both results.
```

The `model` field on each function_call tells the client which
sub-agent is requesting. The `call_id` is globally unique and used for
routing — the client just includes it in the PATCH.

---

## API Changes

### New endpoint: `PATCH /v1/responses/{id}`

Submits tool results for an in-progress response:

```json
PATCH /v1/responses/R1
{
  "tool_results": [
    { "call_id": "call_abc", "output": "..." },
    { "call_id": "call_def", "output": "..." }
  ]
}
```

**Server logic:**

```python
# PATCH handler pseudocode
for tool_result in request.tool_results:
    outcome = task_store.complete_pending_tool_call(
        tool_result.call_id, tool_result.output
    )
    match outcome:
        case CompletePendingToolCallResult.NOT_FOUND:
            return 404, {"error": {"code": "not_found",
                "message": f"Tool call {tool_result.call_id} not found."}}
        case CompletePendingToolCallResult.SUB_AGENT_DONE:
            return 409, {"error": {"code": "conflict",
                "message": f"Sub-agent for {tool_result.call_id} "
                           f"is no longer waiting for results."}}
        case CompletePendingToolCallResult.ALREADY_COMPLETED:
            pass  # Idempotent — no-op, continue
        case CompletePendingToolCallResult.COMPLETED:
            # Update the function_call item status to "completed"
            # Append function_call_output item to parent's output
            update_parent_output(tool_result.call_id, tool_result.output)

return 200, updated_response
```

**HTTP status codes:**

| Status | Condition | `complete_pending_tool_call` result |
|--------|-----------|-------------------------------------|
| `200 OK` | Tool result accepted (new or idempotent re-PATCH). | `COMPLETED` or `ALREADY_COMPLETED` |
| `400 Bad Request` | Malformed body (missing `tool_results`, `call_id`, or `output`). | N/A — validation before store call |
| `404 Not Found` | Response ID does not exist, or `call_id` not in `pending_tool_calls`. | `NOT_FOUND` |
| `409 Conflict` | Sub-agent task reached terminal status (completed, failed, cancelled). No one is waiting for this result. | `SUB_AGENT_DONE` |

**Idempotency:** Re-PATCHing a `call_id` that already has
`status="completed"` returns `200 OK` — it's a no-op. The first
writer wins: the stored result is **not overwritten** by a re-PATCH
with a different `output` value. This means clients can safely retry
PATCH on network failure without risk of double-delivery or result
corruption.

**409 vs idempotent 200:** Both involve a `call_id` that's already
been handled, but the distinction matters:
- **200 (ALREADY_COMPLETED)**: the client's own previous PATCH
  succeeded. Safe to proceed.
- **409 (SUB_AGENT_DONE)**: the sub-agent finished independently
  (timeout, cancellation). The tool result is wasted — the client
  should check the response status via GET.

**Atomicity:** All tool results in a single PATCH are applied in one
database transaction. If any `call_id` fails validation (not found,
conflict), the entire PATCH is rejected — no partial updates.

PATCH is the correct verb — the client is partially updating an
existing in-progress response, not creating a new resource (POST) or
replacing it entirely (PUT).

### New function_call status: `action_required`

Extends the existing `in_progress | completed | incomplete` enum:

| Status | Meaning |
|--------|---------|
| `in_progress` | Server is processing this tool call |
| `completed` | Tool call finished (server or client) |
| `incomplete` | Tool call didn't finish (timeout, filter) |
| **`action_required`** | **Client must execute and PATCH result** |

### Sub-agent attribution via `model` field

```json
{
  "type": "function_call",
  "name": "search_client_db",
  "call_id": "call_abc",
  "status": "action_required",
  "model": "orchestrator.researcher",
  "arguments": "{\"query\":\"quantum\"}"
}
```

The existing `model` field (already present on all function_call
items) carries the dotted form `"root.child"` for tunneled sub-agent
calls. This is always a single dot regardless of nesting depth
(sub-agent names are unique across the spec tree, so
`"orchestrator.summarizer"` is unambiguous even if `summarizer` is a
grandchild). Dots are not allowed in agent names (enforced by the
spec validator).

For parent-level tool calls, `model` remains the root agent name
(e.g. `"orchestrator"`). The client distinguishes tunneled sub-agent
calls by checking for a dot in `model`.

### No new field needed: reuse `model` (FunctionCallData.agent)

`FunctionCallData` already has an `agent` field (serialized as
`"model"` in JSON) that identifies who produced the item. Today it's
set to the task's agent name (e.g. `"orchestrator"`). For tunneled
sub-agent tool calls, set it to the dotted form instead (e.g.
`"orchestrator.researcher"`).

No new entity field is required. The `agent` field is:
- Never compared or filtered on in production code
- Never sent to the LLM (not included in prompt history)
- Not indexed in the database (stored in a JSON blob)
- Purely informational for API clients

For `FunctionCallOutputData`, the `model` field does not exist today
(outputs don't carry agent attribution). For tunneled sub-agent tool
outputs, the attribution is recoverable from the corresponding
`function_call` item via `call_id` — no new field needed on outputs
either.

---

## Internal: Cross-Workflow Publishing

When a sub-agent hits a client-side tool, it publishes the
`function_call` item to the **parent's** response output, not its own.
This is a cross-workflow write — W2 writes to W1's output.

This works because:
- Output items are stored in the database (conversation store)
- SSE events are published via a database-backed channel
- Any process can insert items for any task ID
- W1 is blocked (inside CollectTool), so it's not actively writing
  to its own output — no concurrent writer conflict

The sub-agent knows the root task ID via `root_task_id` on its own
task row.

### Park sequence: write ordering (no transaction needed)

The sub-agent's park sequence has two database writes to two
different stores:
1. INSERT into `pending_tool_calls` (routing entry) — via
   `TaskStore.create_pending_tool_call()`
2. INSERT `function_call` item into root's conversation output — via
   `ConversationStore.append()`

These are **separate store calls, not a single transaction**. The
`pending_tool_calls` table and the `conversation_items` table are
owned by different stores with different sessions. Forcing a
cross-store transaction would violate the existing store abstraction
boundaries.

**Correctness comes from write ordering + DBOS replay, not
transactions.**

#### The rule: routing row FIRST, conversation item SECOND

```python
# In the sub-agent's workflow, when client-side tools are detected:

# Step 1: Write routing row FIRST
#   This tells the PATCH endpoint where to deliver results.
#   The client can't PATCH yet because it hasn't seen the
#   function_call item (that's written in step 2).
task_store.create_pending_tool_call(
    call_id=tc.call_id,
    root_task_id=root_task_id,
    task_id=task_id,  # this sub-agent's task ID
)

# Step 2: Write function_call item to root's conversation SECOND
#   Now the client can see the tool call via GET or SSE.
#   By this point, the routing row already exists, so when the
#   client PATCHes, the server can look up call_id and find it.
conv_store.append(
    root_conversation_id,
    [function_call_item],  # status="action_required"
)

# Step 3: Enter park loop
#   Poll pending_tool_calls for status="completed"
results = park_for_tool_results(task_id, pending_ids)
```

#### Why this ordering is safe

The client can only PATCH after seeing the `function_call` item
(written in step 2). By that point, the routing row (step 1) already
exists. So the client can **never** PATCH a `call_id` that has no
routing row — the 404 race condition is eliminated by ordering.

The reverse ordering (item first, routing second) would be dangerous:
the client could see the item, PATCH immediately, and hit a 404
because the routing row doesn't exist yet.

**PATCH arrives before park loop starts:** The client could see the
`function_call` item (step 2), PATCH the result, and the
`complete_pending_tool_call` UPDATE could land **before** the
sub-agent enters the park loop (step 3). This is safe: the PATCH
updates the `pending_tool_calls` row to `status="completed"`. When
the park loop starts its first poll, it immediately finds the
completed row and returns — no waiting needed. The park loop is
designed to handle results that arrived before polling began.

#### Crash recovery via DBOS replay

DBOS restarts in-progress workflows after a server crash by
re-executing the workflow function from the beginning. The code runs
again — same inputs, same side effects. This means every database
write in the park sequence will be attempted a second time.

**Problem without protection:** If the server crashes after step 1
(routing row written) and DBOS replays, step 1 re-executes:
`INSERT INTO pending_tool_calls (call_id="call_abc", ...)`. That row
already exists — `call_id` is the PK. Without protection, this is a
duplicate key error and the workflow crashes on replay. Same problem
for step 2: `conv_store.append()` re-inserts a conversation item at
a position that's already occupied.

**Fix: `INSERT ... ON CONFLICT DO NOTHING` on both writes.** This
SQL clause means "if the row already exists, silently skip the insert
instead of erroring." The replay becomes a no-op — no error, no
duplicate. The workflow proceeds to the next step as if the write
succeeded.

**Implementation:** `task_store.create_pending_tool_call()` uses
`INSERT ... ON CONFLICT (call_id) DO NOTHING`. `conv_store.append()`
uses `INSERT ... ON CONFLICT (conversation_id, position) DO NOTHING`
(position is unique within a conversation).

**Every crash point converges to correct state:**

| Crash point | What exists in DB | What DBOS replay does |
|-------------|-------------------|----------------------|
| Before step 1 | Nothing | Writes routing row (new), writes item (new). Normal. |
| After step 1, before step 2 | Routing row only | Re-INSERTs routing row → `ON CONFLICT DO NOTHING` (skip). Writes item (new). Both exist. Normal. |
| After step 2, before step 3 | Both rows | Re-INSERTs routing row → skip. Re-INSERTs item → skip. Enters park loop. Normal. |
| During step 3 (park loop) | Both rows + maybe client already PATCHed | Re-INSERTs → skip. Re-enters park loop. If `status="completed"` (client PATCHed during uptime), first poll finds result → resumes immediately. If not, waits for client. Normal. |

No crash scenario produces duplicates, errors, or inconsistent state.
The combination of write ordering (finding #8) and idempotent INSERTs
means correctness without cross-store transactions.

---

## Internal: Park Loop Implementation

Inside the sub-agent's agent loop, when client-side tools are detected:

```python
# In the sub-agent's _handle_pending_tool_calls equivalent:
server_tools, client_tools = partition_tools(pending_tool_calls)

# Execute server-side tools normally
execute_server_tools(server_tools)

if client_tools:
    # 1. Write routing rows FIRST (one per client tool call)
    #    ORDER MATTERS: routing must exist before the client can
    #    see the function_call item and attempt a PATCH.
    for tc in client_tools:
        task_store.create_pending_tool_call(
            call_id=tc.call_id,
            root_task_id=root_task_id,
            task_id=task_id,
        )

    # 2. Write function_call items to ROOT's conversation SECOND
    #    Now the client can see them via GET/SSE and PATCH results.
    #    Routing rows already exist, so PATCH won't 404.
    conv_store.append(root_conversation_id, [
        build_function_call_item(tc, model=dotted_agent_name)
        for tc in client_tools
    ])

    # 3. Park: poll pending_tool_calls until all results arrive
    pending_ids = {tc.call_id for tc in client_tools}
    results = park_for_tool_results(task_id, pending_ids)

    # 4. Feed results back into conversation history
    append_tool_results_to_history(results)

    # 5. Continue agent loop (next LLM call)
```

The `park_for_tool_results` function:

```python
def park_for_tool_results(
    task_id: str,
    pending_ids: set[str],
    poll_interval: float = 0.5,
    timeout: float | None = None,
) -> dict[str, str]:
    """
    Block until all pending tool results are delivered to the inbox.

    Polls the pending_tool_calls table at the given interval. Uses the
    same pattern as the steering inbox — database-backed with atomic
    reads.

    :param task_id: The sub-agent's task ID.
    :param pending_ids: Set of tool_call_ids to wait for.
    :param poll_interval: Seconds between inbox checks. Fixed at 500ms
        — not externally configurable. 500ms balances responsiveness
        with DB load (matches the steering inbox pattern).
    :param timeout: Maximum seconds to wait. None uses the sub-agent's
        remaining execution timeout.
    :returns: Mapping of tool_call_id to result string.
    :raises TimeoutError: If timeout expires before all results arrive.
    """
    collected: dict[str, str] = {}
    deadline = time.monotonic() + timeout if timeout else None

    while pending_ids - collected.keys():
        if deadline and time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for tool results: "
                f"{pending_ids - collected.keys()}"
            )
        DBOS.sleep(poll_interval)
        rows = task_store.get_pending_tool_calls(
            task_id, status="completed",
        )
        for row in rows:
            if row.call_id in pending_ids:
                collected[row.call_id] = row.result

    return collected
```

---

## Client Recovery After Stream Interruption

The SSE stream is a live view of database state — not the source of
truth. All output items, `pending_tool_calls` rows, and workflow state
are persisted. Recovery is: GET the response, process what you see.

There is no stream resumption endpoint — this matches OpenAI's
Responses API, which also only streams on the initial POST. If the
SSE connection drops, clients poll `GET /v1/responses/{id}`.
Stream resumption (`GET /v1/responses/{id}/stream`) is a possible
future optimization but not part of this design.

### Scenario 1: Stream breaks while sub-agent is parked

The sub-agent published `action_required` to the root's output, but
the client lost the SSE connection before seeing it (server restart,
network drop, etc.).

```
1. Client reconnects.
2. GET /v1/responses/R1
   → status: "in_progress"
   → output includes function_call with status: "action_required"
3. Client executes the tool.
4. PATCH /v1/responses/R1 with tool result.
5. Sub-agent resumes, workflow continues.
6. Client polls GET /v1/responses/R1 until status is terminal.
   (No stream resumption — SSE is only on the initial POST.)
```

### Scenario 2: Stream breaks after client PATCHed but before confirmation

The client sent the PATCH but didn't see the response (connection
dropped mid-request).

```
1. Client reconnects.
2. GET /v1/responses/R1
   → If PATCH succeeded: function_call shows status: "completed"
     Client knows it was delivered. Nothing to do.
   → If PATCH didn't reach server: function_call still "action_required"
     Client re-PATCHes. Idempotent — PK on call_id prevents duplicates.
3. Poll GET /v1/responses/R1 until status is terminal.
```

### Scenario 3: Server restarts while sub-agent is parked

DBOS replays workflows on restart. The sub-agent re-enters the park
loop and polls `pending_tool_calls`.

```
1. If client already PATCHed before restart:
   → pending_tool_calls row has status="completed"
   → Park loop finds it immediately on first poll, resumes.
2. If client hasn't PATCHed yet:
   → pending_tool_calls row has status="action_required"
   → function_call item is already in root's output (persisted)
   → Sub-agent parks again, waits.
   → Client does GET, sees action_required, PATCHes normally.
```

### Scenario 4: Sub-agent finishes before client reconnects

The sub-agent completed (either the tool result was delivered before
the stream broke, or the sub-agent timed out).

```
1. Client reconnects.
2. GET /v1/responses/R1
   → status: "completed" (or "incomplete" if timed out)
   → Full output is present.
3. Nothing to do — workflow already finished.
```

### Client recovery algorithm

A general-purpose client handles all scenarios with one loop:

```
while True:
    response = GET /v1/responses/R1
    if response.status in ("completed", "incomplete"):
        break  # done

    for item in response.output:
        if item.type == "function_call"
           and item.status == "action_required"
           and item.call_id not in already_patched:
            result = execute_tool(item.name, item.arguments)
            PATCH /v1/responses/R1 {call_id: item.call_id, output: result}
            already_patched.add(item.call_id)

    # Wait for more output (no stream resumption — poll only)
    sleep(poll_interval)
```

This is stateless on the client side (except the `already_patched`
set, which is an optimization — PATCHing twice is safe). The client
can crash, restart, and resume from the GET.

## Known Pitfalls and Mitigations

### 1. Thread pool exhaustion

**Problem:** Parked workflows hold a DBOS thread. Many concurrent parked
sub-agents could starve the thread pool.

**Mitigation:** This is the same pattern as the steering inbox — the
existing agent loop already parks workflows waiting for steering input.
The thread pool must be sized for the expected concurrency. Future
optimization: async waiting (release the thread while parked).

### 2. Cross-workflow output ordering

**Problem:** Multiple sub-agents publish `function_call` items to the
parent's output concurrently. Items from different sub-agents could
interleave.

**Mitigation:** Each item is appended atomically. Order within a single
sub-agent is preserved (one sub-agent publishes one tool call at a
time). Interleaving between sub-agents is acceptable — the `model`
field disambiguates.

### 3. Race: client PATCHes after sub-agent times out

**Problem:** Sub-agent times out while parked (client was slow). Workflow
completes with `incomplete` status. Client then PATCHes tool results
for a call_id whose sub-agent is already done.

**Mitigation:** PATCH endpoint checks the sub-agent's current status
before writing to inbox. If the sub-agent is no longer parked, return
an error (e.g., 409 Conflict).

### 4. Timeout and cancellation

**Problem:** Client never submits tool results → workflow parks forever.

**Mitigation:** `park_for_tool_results` has a timeout (defaults to the
sub-agent's remaining execution timeout). On timeout, the workflow
completes with an error/incomplete status. `collect_sub_agents` sees this and
reports `status: "incomplete"` to the parent LLM.

**Cancellation:** When the parent task is cancelled, spawned sub-agent
tasks should be cancelled too. The cancel propagation writes a
cancellation signal that the park loop checks alongside tool results.

### 5. DBOS recovery

**Problem:** Server crashes while sub-agent is parked. On restart, DBOS
replays the workflow.

**Mitigation:** The inbox is database-backed. Tool results delivered
during downtime persist in the table. When the workflow replays and
re-enters the park loop, it finds the results on its first poll.
Function_call items were already published to the parent's output
before parking, so they are not re-published.

### 6. Idempotency

**Problem:** Client retries `PATCH /v1/responses/{id}` due to network
error.

**Mitigation:** The `pending_tool_calls` primary key is `call_id`. The UPDATE
from PATCH is idempotent — writing the same result twice leaves the
row unchanged.

### 7. Collect timeout vs. park timeout interaction

**Problem:** The parent's `collect_sub_agents` has a timeout (e.g. 30s).
A sub-agent is parked waiting for a client-side tool result. Which
timeout fires first?

**Answer:** They are independent timeouts on independent workflows:
- **Park timeout** (sub-agent W2): `park_for_tool_results` uses the
  sub-agent's remaining execution timeout. If it fires, W2 completes
  with `status: "incomplete"`.
- **Collect timeout** (parent W1): `CollectTool` uses
  `min(explicit_timeout, parent_remaining_time)`. If it fires, W1
  returns partial results with `status: "incomplete"` for W2.

If collect times out first, W1 proceeds with partial results. W2
continues running — it may still receive the client's PATCH and
complete successfully, but the parent no longer waits for it. If the
park times out first, W2 completes with `incomplete`, and
`collect_sub_agents` returns that status to the parent.

No coordination is needed — both timeouts are fail-safe. The first
one to fire produces a clean result.

---

## How Client-Side Tools Integrate with Spawn/Collect

### `root_task_id` enables tunneling

`SpawnTool` sets `root_task_id` at task creation. The value is
**always the top-level task ID**, regardless of nesting depth:

- **Top-level parent spawns child**: `root_task_id = parent.id`
  (the parent has no `root_task_id` of its own)
- **Child spawns grandchild**: `root_task_id = child.root_task_id`
  (propagated — still points to the original top-level task)

This is handled by the argument injection in `_handle_tool_calls()`
(see `SUBAGENT.md` pseudocode): `task.root_task_id or task.id`.

The sub-agent uses `root_task_id` to:
- Know which root response's output to publish `function_call` items to
- Route via `pending_tool_calls` table (`root_task_id` column)
- Tunnel regardless of nesting depth (always points to root)

### Single execution path

Spawned sub-agents use the real `_run_agent_loop()`, not a simplified
loop. The park branch in `_handle_tool_calls()` activates for
sub-agents with client-side tools:

```python
if client_tools and task.root_task_id is not None:
    # Tunneled: write routing rows, publish to root's output, park
    for tc in client_tools:
        task_store.create_pending_tool_call(...)
    conv_store.append(root_conversation_id, function_call_items)
    results = park_for_tool_results(task_id, pending_ids)
    # Feed results back, continue agent loop
elif client_tools:
    # Top-level task: existing behavior — complete response with
    # function_call items, client responds via new POST
```

Because the branch is inside the existing agent loop, no new execution
path is needed.

### `collect_sub_agents` is unaffected

When a sub-agent parks for client tools, its workflow stays alive —
the DBOS thread is blocked in the park loop, not completed.
`handle.get_result()` still blocks until the workflow truly finishes.
`CollectTool` doesn't change.

### Summary: client-side tool changes

| Change | Scope |
|--------|-------|
| Park branch in `_handle_tool_calls()` | New `elif` branch when `root_task_id IS NOT NULL` |
| `pending_tool_calls` table | New table added to `db_models.py` |
| `TaskStore` pending tool call methods | 3 new methods on existing store |
| `PATCH /v1/responses/{id}` | New endpoint |
| `park_for_tool_results()` | New function |
| `action_required` status | function_call item enum extension |
| Dotted `model` values | Existing field, new format |

---

## Open Questions

### Q1: Multiple tool calls in one LLM turn

If the LLM returns both server-side and client-side tool calls in the
same turn, the workflow must:
1. Execute server-side tools immediately
2. Publish client-side tool calls to parent's output
3. Park for client-side tool results
4. Feed ALL results (server + client) back to the LLM

The ordering matters — server-side results are available immediately,
client-side results arrive later. The park loop only waits for
client-side tool_call_ids.

### Q2: Parent output ordering during concurrent publishing

As the parent workflow runs, output items accumulate. When the parent
blocks on collect, sub-agents publish tool calls to the parent's output.
When collect returns, the parent resumes and appends more items. The
output is an append-only ordered list.

Clients reading `GET /v1/responses/{id}` at different times see
different prefixes of this list. This is consistent with how in-progress
responses work — the output grows over time.

### Q3: Nested sub-agents with client tools (resolved)

All tunneled tool calls go to the **root** response, regardless of
nesting depth. Every sub-agent task has `root_task_id` pointing
directly to the top-level task — no chain-walking needed. The client
always interacts with one response.
