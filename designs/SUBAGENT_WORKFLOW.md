# Spawned Sub-Agent Workflow: Client-Side Tool Handling (Phase 2)

## Context

This is the **Phase 2** design for client-side tool support in spawned
sub-agents. Phase 1 (see `SUBAGENT.md`) implements spawn/collect with
server-side tools only. Phase 1 lays extension points so
that this design can be implemented without refactoring Phase 1 code.

**Prerequisite:** Phase 1 must be complete before implementing Phase 2.

The spawned model has a hard problem: **client-side tools**. When a
spawned sub-agent needs a client-side tool, the workflow must pause, the
client must execute the tool, and the workflow must resume — all while
`collect_sub_agents` waits for the sub-agent to finish.

This document specifies the request flow and internal mechanics for
spawned sub-agents with client-side tools.

---

## Design: Tool Result Inbox (Long-Lived Workflow)

Instead of letting the sub-agent's task complete when it hits client-side
tools (which would create multi-task chains and force `collect_sub_agents` to track
conversations across task boundaries), **the sub-agent workflow stays
alive**. It parks internally, waiting for tool results to be delivered.

From the client's perspective, the protocol is identical to any other
response — `GET /v1/responses/{id}` shows function_call items, and
`POST /v1/responses` with `previous_response_id` submits results. The
difference is purely server-side: instead of creating a new task, the
response creation endpoint delivers tool results to the parked workflow.

### Why this approach

| Approach | Polling? | Multi-task tracking? | `collect_sub_agents` complexity |
|----------|----------|----------------------|----------------------|
| Poll conversations | Yes (external) | Yes | High — chase task chains |
| Completion channel | Yes (external) | Yes | Medium — watch a flag |
| **Tool result inbox** | **Yes (internal, encapsulated)** | **No** | **Trivial — one `wait()`** |

The polling still exists, but it's pushed inside the sub-agent workflow
where it belongs, and hidden from `collect_sub_agents` entirely. From `collect_sub_agents`'s
perspective, it's zero-polling: `task_store.wait(task_id)` blocks until
the workflow truly completes.

---

## Data Model

### `tool_result_inbox` table

```
tool_result_inbox:
  task_id        TEXT     — the parked sub-agent's task ID
  tool_call_id   TEXT     — which tool call this result is for
  result         TEXT     — the client's tool output (JSON string)
  delivered_at   TIMESTAMP
  PRIMARY KEY (task_id, tool_call_id)
```

The primary key on `(task_id, tool_call_id)` enforces idempotency — a
client retrying `POST /v1/responses` cannot deliver duplicate results.

### `spawned_sub_agent` flag (set in Phase 1)

Tasks created by `spawn_sub_agents` are marked with a boolean flag (column or
metadata). **Phase 1 sets this flag but nothing reads it.** Phase 2
uses it so the `POST /v1/responses` endpoint can distinguish:
- **Not a parked sub-agent** → create a new task (existing behavior)
- **Parked sub-agent** → deliver tool results to inbox (new behavior)

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

 3    Client → Server    GET /v1/responses/R1   (SSE stream subscription)

 4    W1 (parent)        Calls LLM. LLM returns:
                           tool_call: spawn_sub_agents("researcher", "find papers")

 5    W1 (parent)        SpawnTool.invoke() executes:
                           a. Creates conversation C2
                           b. Creates task T2, marked as spawned sub-agent
                           c. Starts DBOS workflow W2
                           d. Returns tool result: '{"response_ids": ["T2"]}'

      Client ← SSE(R1)  ← output_item.added  function_call (spawn)
                         ← output_item.added  function_call_output '{"response_ids":["T2"]}'

 6    W1 (parent)        Next LLM turn. LLM returns:
                           tool_call: collect_sub_agents(response_ids=["T2"])
                         CollectTool.invoke() calls task_store.wait(T2).
                         W1 blocks here.

      Client ← SSE(R1)  ← output_item.added  function_call (collect)
                         (no output yet — parent is waiting)
```

At this point W2 is running independently:

```
 7    W2 (researcher)    Calls LLM. LLM returns:
                           tool_call: search_client_db(query="quantum")
                         search_client_db is a CLIENT-SIDE tool.

 8    W2 (researcher)    Detects client-side tool call. PARKS:
                           a. Persists function_call items to conversation C2
                           b. Publishes function_call items to T2's live stream
                           c. Records pending tool_call_ids
                           d. Enters park loop: poll tool_result_inbox
                              for T2 every 500ms

      T2's SSE stream now has function_call items. No one is
      subscribed yet.
```

Client discovers the sub-agent needs input:

```
 9    Client             Client knows about T2 from step 5 (saw task_id in
                         spawn's tool output on parent's SSE stream).
                         Decides to check on it.

10    Client → Server    GET /v1/responses/T2   (poll or SSE subscribe)

11    Server → Client    Returns:
                         { id: R2,
                           status: "in_progress",
                           output: [
                             ...,
                             { type: "function_call",
                               name: "search_client_db",
                               call_id: "call_abc",
                               arguments: '{"query":"quantum"}' }
                           ] }

                         NOTE: status is "in_progress" (not "completed")
                         because W2 is still alive, just parked.
```

Client executes the tool and submits results:

```
12    Client             Sees function_call for search_client_db.
                         Executes it locally.
                         Gets results: ["paper1.pdf", "paper2.pdf"]

13    Client → Server    POST /v1/responses
                         { previous_response_id: R2,
                           input: [{
                             type: "function_call_output",
                             call_id: "call_abc",
                             output: '["paper1.pdf","paper2.pdf"]'
                           }] }

14    Server             Looks up R2 → belongs to task T2 → T2 is marked
                         as spawned sub-agent and is currently parked.

                         BRANCH (instead of creating a new task):
                           a. Writes tool result to tool_result_inbox:
                              (task_id=T2, tool_call_id="call_abc",
                               result='["paper1.pdf","paper2.pdf"]')
                           b. Returns { id: R2, status: "in_progress" }
                              Same response — workflow is still running.
```

W2 resumes:

```
15    W2 (researcher)    Park loop reads inbox → finds result for call_abc.
                         All pending tool results received. Exits park loop.
                         Feeds tool results back to LLM as
                         function_call_output items.

16    W2 (researcher)    LLM processes results. Returns final text:
                         "Found 2 relevant papers on quantum computing..."
                         No more tool calls → workflow completes normally.

      Client ← SSE(R2)  ← output_item.added  message "Found 2 relevant..."
                         ← response.completed
```

Back to parent:

```
17    W1 (parent)        task_store.wait(T2) returns — T2 completed.
                         CollectTool reads T2's final output.
                         Returns to LLM: '{"results": [{"task_id": "T2",
                           "agent_name": "researcher", "status": "completed",
                           "output": "Found 2 relevant papers..."}]}'

      Client ← SSE(R1)  ← output_item.added  function_call_output (collect)

18    W1 (parent)        LLM synthesizes final answer:
                         "Based on the research, here's a summary..."
                         No more tool calls → workflow completes.

      Client ← SSE(R1)  ← output_item.added  message "Based on the research..."
                         ← response.completed
```

---

## Multi-Round Client Tools (Sub-Agent Parks Twice)

If the sub-agent needs multiple rounds of client-side tools, the same
park/resume cycle repeats within the same workflow:

```
15    W2 (researcher)    Picks up first tool result. Feeds to LLM.
                         LLM returns ANOTHER client-side tool call:
                           tool_call: download_pdf(url="paper1.pdf")

16    W2 (researcher)    Parks again. Same mechanism:
                           a. Publish function_call to SSE stream
                           b. Write pending tool_call_ids to inbox
                           c. Poll inbox

      Client ← SSE(R2)  ← output_item.added  function_call (download_pdf)

17    Client             Sees new tool call on T2's stream.
                         Executes download_pdf locally.

18    Client → Server    POST /v1/responses
                         { previous_response_id: R2,
                           input: [{ type: "function_call_output",
                                     call_id: "call_def",
                                     output: "<pdf content>" }] }

19    Server             Delivers to tool_result_inbox for T2. Same R2.

20    W2 (researcher)    Picks up result, feeds to LLM, LLM finishes.
                         Workflow completes.

      ... collect returns, parent continues as before ...
```

Key: **same task T2, same response R2, same workflow W2** throughout.
No new tasks or responses are created for client-side tool round-trips.

---

## Parallel Sub-Agents with Independent Client Tools

Multiple sub-agents can independently request client-side tools without
blocking each other:

```
      W1 spawns T2 (researcher) and T3 (analyst)
      W1 calls collect_sub_agents(["T2", "T3"]) — blocks on both

      T2 parks (needs search_client_db)    T3 parks (needs query_db)

      Client handles both independently:
        GET /v1/responses/T2 → sees search_client_db → submits results
        GET /v1/responses/T3 → sees query_db → submits results

      T2 resumes, completes.               T3 resumes, completes.

      collect returns with both results.
```

The client discovers both task IDs from the `spawn_sub_agents` tool output on the
parent's SSE stream and interacts with each sub-agent independently via
standard API endpoints. No coordination between sub-agents is needed.

---

## Server-Side Branch: `POST /v1/responses`

The response creation endpoint gains a conditional path:

```
POST /v1/responses  { previous_response_id: R }

  1. Look up response R → get task T
  2. Is T a parked spawned sub-agent?

     NO  → existing behavior:
           create new task, start new workflow

     YES → new behavior:
           extract function_call_output items from input
           write each to tool_result_inbox (task_id=T)
           return existing response R with status "in_progress"
```

### Detection: how does the server know T is parked?

The task record has two indicators:
- **`spawned_sub_agent` flag** — set at creation time by `SpawnTool`
- **`parked` status** — set by the workflow when it enters the park loop

Both must be true. A spawned sub-agent that is not currently parked
(e.g. it's between tool calls, actively running LLM) should NOT receive
inbox deliveries — the workflow isn't polling yet. The exact mechanism
(status column, separate flag, metadata field) is an implementation
detail.

---

## SSE Stream Semantics

The sub-agent's SSE stream behaves as follows:

```
WORKFLOW STATE           SSE STREAM BEHAVIOR
─────────────           ───────────────────
Running (LLM + tools)   Events stream normally (deltas, output items)
Parked (waiting)         function_call items published, then stream pauses
                         No response.completed — workflow is still alive
Resumed (tool results)   New events appear on the SAME stream
Truly completed          response.completed sent, stream closes
```

This means:
- `response.completed` is ONLY sent when the workflow truly finishes
- A client subscribed via SSE sees function_call items appear, then
  silence while parked, then new output when resumed
- Status via `GET /v1/responses/{id}` remains `in_progress` while parked
- `in_progress` + function_call items in output = "submit tool results"

### New semantic: `in_progress` with `function_call` items

This combination does not exist today. Currently, function_call items
only appear in completed responses. For parked sub-agents, function_call
items appear while the response is still `in_progress`.

Clients must recognize: **`in_progress` + `function_call` output items =
the sub-agent needs tool results to continue.**

---

## Internal: Park Loop Implementation

Inside the sub-agent's agent loop, when client-side tools are detected:

```python
# In the sub-agent's _handle_tool_calls equivalent:
server_tools, client_tools = partition_tools(tool_calls)

# Execute server-side tools normally
execute_server_tools(server_tools)

if client_tools:
    # 1. Persist function_call items to conversation
    persist_function_call_items(client_tools)

    # 2. Publish to live stream (SSE)
    publish_function_call_items(client_tools)

    # 3. Record what we're waiting for
    pending_ids = {tc.call_id for tc in client_tools}

    # 4. Park: poll inbox until all results arrive
    results = park_for_tool_results(task_id, pending_ids)

    # 5. Feed results back into conversation history
    append_tool_results_to_history(results)

    # 6. Continue agent loop (next LLM call)
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

    Polls the tool_result_inbox table at the given interval. Uses the
    same pattern as the steering inbox — database-backed with atomic
    reads.

    :param task_id: The sub-agent's task ID.
    :param pending_ids: Set of tool_call_ids to wait for.
    :param poll_interval: Seconds between inbox checks. 500ms balances
        responsiveness with DB load.
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
        rows = read_tool_results(task_id, pending_ids)
        for row in rows:
            collected[row.tool_call_id] = row.result

    return collected
```

---

## Known Pitfalls and Mitigations

### 1. Thread pool exhaustion

**Problem:** Parked workflows hold a DBOS thread. Many concurrent parked
sub-agents could starve the thread pool.

**Mitigation:** This is the same pattern as the steering inbox — the
existing agent loop already parks workflows waiting for steering input.
The thread pool must be sized for the expected concurrency. Future
optimization: async waiting (release the thread while parked).

### 2. Response status semantics

**Problem:** `in_progress` with `function_call` output items is a new
combination. Clients that only look for function_calls in completed
responses will miss them.

**Mitigation:** Document the new semantic. Clients interacting with
spawned sub-agents must check for function_call items in `in_progress`
responses.

### 3. Dual-mode branch in `POST /v1/responses`

**Problem:** The same endpoint now does two different things: create a
new task OR deliver to inbox.

**Mitigation:** The branch is clean — check one flag on the task record.
The response to the client is identical in shape either way. The
conditional is small and well-tested.

**Race condition:** Client submits tool results after the workflow times
out and completes. The inbox write succeeds but nobody reads it.
Mitigation: the endpoint checks the task's current status before writing
to the inbox. If the task is no longer parked, fall back to the standard
"create new task" path (or return an error).

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
function_call items were already persisted to the conversation before
parking, so they are not re-published.

### 6. Idempotency

**Problem:** Client retries `POST /v1/responses` due to network error.

**Mitigation:** The `tool_result_inbox` primary key is
`(task_id, tool_call_id)`. Duplicate inserts are upserts (or ignored) —
the same result for the same tool call is written once.

---

## What `POST /v1/responses` Returns

When delivering to a parked sub-agent, the endpoint returns the
**existing response** (same response ID R2). This differs from the
standard path (which returns a new response with a new ID).

Rationale: the workflow is still running under the same task. No new
task or response was created. The response's output will continue to
grow as the workflow resumes and produces more items.

---

## Extension Points Laid in Phase 1

Phase 1 (`SUBAGENT.md`) establishes these extension points that Phase 2
builds on. None require refactoring — Phase 2 changes are purely
additive.

### 1. `spawned_sub_agent` flag on tasks

`SpawnTool` marks every task it creates with `spawned_sub_agent=True`.
Phase 1 sets this flag but nothing reads it. Phase 2 uses it in the
`POST /v1/responses` handler to detect parked sub-agents and route tool
results to the inbox instead of creating a new task.

### 2. `_run_agent_loop()` as single execution path

Spawned sub-agents use the real `_run_agent_loop()`, not a simplified
loop. Phase 2 adds a park branch inside `_handle_tool_calls()`:

```python
# Phase 1 (current):
if client_tools:
    # Sub-agent ToolManager doesn't include client tools,
    # so this branch is never reached for sub-agents

# Phase 2 (added):
if client_tools and task.spawned_sub_agent:
    # Park: publish function_calls, poll inbox, resume
    results = park_for_tool_results(task_id, pending_ids)
    # Feed results back, continue agent loop
elif client_tools:
    # Existing behavior: complete response with function_call items
```

Because the branch is inside the existing agent loop, no new execution
path is needed.

### 3. `collect_sub_agents` uses `task_store.wait(task_id)`

This call is identical in Phase 1 and Phase 2. When a sub-agent parks
for client tools (Phase 2), its workflow stays alive — the DBOS thread
is blocked in the park loop, not completed. `task_store.wait()` still
blocks until the workflow truly finishes. `CollectTool` doesn't change.

### 4. Client-side tool filtering is one line

Phase 1 filters client-side tools from the spawned sub-agent's
ToolManager. Phase 2 removes this filter (one-line change) and relies
on the park branch in `_handle_tool_calls()` to handle them.

### Summary: Phase 2 changes

| Change | Scope | Touches Phase 1 code? |
|--------|-------|----------------------|
| Remove client-tool filter | ToolManager setup | One-line removal |
| Add park branch | `_handle_tool_calls()` | New `elif` branch |
| Add `tool_result_inbox` table | New migration | No existing tables |
| Add inbox delivery branch | `POST /v1/responses` handler | New `if` check |
| Add `park_for_tool_results()` | New function | No |
| Add `read_tool_results()` | New store method | No |

---

## Open Questions

### Q1: Client polling strategy for sub-agents

The client discovers sub-agent task IDs from the parent's `spawn_sub_agents` tool
output. But it has no signal that a sub-agent needs attention until it
actively checks. Options:

- **Eager subscription:** Client subscribes to each sub-agent's SSE
  stream immediately after seeing the spawn output. Sees function_call
  items in real-time.
- **Lazy polling:** Client periodically polls
  `GET /v1/responses/{task_id}` for each sub-agent. Simpler but adds
  latency.
- **Parent-stream hint:** The parent's stream could emit a hint event
  when a sub-agent parks (the sub-agent publishes to its own stream, but
  could also notify the parent). This would alert the client without
  requiring eager subscription. Adds coupling.

Recommendation: eager subscription. The client knows the task IDs
immediately and can subscribe to all of them. The SSE connection is
cheap.

### Q2: Multiple tool calls in one LLM turn

If the LLM returns both server-side and client-side tool calls in the
same turn, the workflow must:
1. Execute server-side tools immediately
2. Park for client-side tool results
3. Feed ALL results (server + client) back to the LLM

The ordering matters — server-side results are available immediately,
client-side results arrive later. The park loop only waits for
client-side tool_call_ids.

### Q3: `GET /v1/responses/{id}` while parked — output ordering

As the workflow runs, output items accumulate. When the workflow parks,
the output includes both the LLM's text/tool responses AND the
function_call items awaiting client input. When the workflow resumes,
more items are appended. The output is an append-only ordered list.

Clients reading `GET /v1/responses/{id}` at different times see
different prefixes of this list. This is consistent with how in-progress
responses work today — the output grows over time.
