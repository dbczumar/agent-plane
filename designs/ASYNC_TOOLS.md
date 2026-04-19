# Async (Background) Tools

Depends on: `RUNTIME.md`, `AGENTLOOP.md`, `SUBAGENT.md`
Ports: G4 from `PORTING_FROM_OMNIAGENTS.md` (§3.1, §Rift 3, Phase 3).

## 1. Overview

Agent-plane today executes every tool call synchronously within one
LLM turn: the LLM requests tools, the workflow runs them, returns
results, and the LLM gets another turn with the results in context.
A single slow tool (a multi-minute web fetch, a long code_sandbox
run) blocks the entire turn. Parallelism exists only *within* a
batch of tool calls emitted by the same LLM response.

Async tools let the agent fire a tool and continue working. When a
result arrives, the **workflow wakes itself** on a durable signal
and gives the LLM another iteration *within the same in-progress
response* — no new user input required, and the client's SSE
stream stays open the whole time. The wake injects a framework
notice announcing how many results are in the inbox; the agent
chooses whether to retrieve them (`read_inbox`), fire more work,
or produce a final response. Between completions the workflow
sleeps — no wasted LLM calls, no polling.

**The LLM never has to wait for the user to "say something" to see
a background result.** The entire cycle (fire → sleep → wake →
notice → LLM iteration → read → ...) happens autonomously inside
one agent-loop execution.

The core mechanism is a **DBOS child workflow per background tool
call**, with a durable `DBOS.send` → `dbos_recv_async` signal on a
shared `async_work_complete` topic. This is the same primitive the
workflow already uses for client-side tool tunneling
(`agent_plane/runtime/workflow.py:2925`), so the integration cost is
small and the durability story is identical.

### How this relates to existing waiting structures

Agent-plane already has three durable "things that accumulate while
the workflow is running" mechanisms. The async-tool inbox is a
fourth, distinguished by who drains it and when:

| Structure | State lives in | Wait mechanism | LLM sees… | LLM controls drain timing? |
| --- | --- | --- | --- | --- |
| **Steering inbox** (existing — `tasks.inbox_closed`, `try_deliver`, `close_inbox`) | `conversation_items` + `inbox_closed` flag on `tasks` | Poll at `close_inbox` checkpoints in the loop | New user messages as normal conversation items on the next loop iteration | No — items auto-appear in history |
| **Pending client-side tool calls** (existing — `pending_tool_calls` table) | Dedicated table, status `action_required` / `completed` | `dbos_recv_async(topic="tool_result")` blocked per call_id via `_wait_for_pending_calls` | A synchronous tool result in the *same* LLM iteration that fired the call | No — workflow parks until all parallel calls in the batch return |
| **Sub-agent auto-collect** (existing) | Sub-agent task rows + conversation items | Poll by parent at collect-check points | Finished sub-agent outputs injected as messages | No — auto-collect logic decides |
| **Async-tool inbox** (this design — `background_tool_calls` table) | Dedicated table, status `running` / `completed` / `failed` / `cancelled` + `read_at` | `dbos_recv_async(topic="async_work_complete")` (any-completion wake) | `handle_id` returned immediately on fire; results retrieved by explicit `read_inbox` call | **Yes — LLM decides when to call `read_inbox`** |

What makes this one different isn't the durability or the wake
infrastructure — both of those patterns already exist. It's that
this is the first mechanism where the LLM *holds handles it can
reference by name* and *drains the queue on its own schedule*.
Steering appears passively in history, client-side tools block the
current iteration, sub-agent collect happens on the runtime's
schedule; none of those give the LLM an explicit "here are N
things waiting, you decide when to look" surface.

The new `async_work_complete` topic is also a minor addition to
the **steering path**: today steering is detected at `close_inbox`
checkpoints (no signal), but because async-tool parks can be
long-running the steering endpoint will additionally
`DBOS.send(topic="async_work_complete", kind="steering")` so the
parked workflow wakes promptly on user steering during a long
background-tool wait. That's the only change to the existing
steering path. See §11.

---

## 2. Goals / Non-Goals

**Goals**

- A tool call can be fired without blocking the current agent-loop
  iteration.
- On completion, the workflow wakes *itself* (no user input
  required) and gives the LLM another iteration inside the same
  `/v1/responses` execution. The framework notice at the top of
  that iteration announces how many results are in the inbox.
- The agent controls what happens next: read the result, queue
  more work, or finalize the response.
- No wasted LLM calls — the workflow blocks on a durable receive,
  not a poll loop. No LLM call happens between completion signals.
- Durable: background calls survive server crashes. On replay, the
  parent rebuilds pending state from the store and re-receives any
  signals delivered during the crash window.

**What "turn" means in this doc.** "Agent-loop iteration" (or "LLM
iteration") = one LLM call + any resulting tool dispatches, all
inside one `/v1/responses` execution. A single user message can
drive many agent-loop iterations. Background tools wake *new
iterations of the current execution*, not new user-driven
requests.

**Non-goals**

- **Mid-step tool cancellation.** DBOS `@step` executions are
  atomic. `cancel_background` only prevents *delivery* of the
  result, not execution of the step itself. Mid-step cancel is a
  separate design (ties into G6, phase-aware cancel).
- **Client-side tools as background tools.** Client-side tools
  already have durable async infrastructure — the
  `pending_tool_calls` table tracks outstanding calls, the client
  PATCHes results back, and the workflow parks on
  `dbos_recv_async(topic="tool_result")` per call_id via
  `_wait_for_pending_calls`. But the LLM-facing semantics are
  synchronous: fire → park → result injected into the *same* LLM
  iteration as a `function_call_output`. The LLM never sees a
  handle_id and can't keep working in the meantime. Routing
  client-side tools through `run_in_background` (so the LLM
  could fire `Bash("npm run dev")` and continue) is a natural
  extension: the PATCH handler would write into
  `background_tool_calls` and signal `async_work_complete`
  instead of writing into `pending_tool_calls` and signaling
  `tool_result`. Deferred — the existing path continues to
  handle the ~95% synchronous case.
- **Sub-agents as background tools.** Sub-agents have their own
  spawn/auto-collect path (`SpawnTool`, G5 named sessions).
  `run_in_background` rejects sub-agent names.
- **Priority or deadlines on individual background calls.** Future.

---

## 3. User-Facing API

Four new builtins, opt-in via `tools.builtins`:

### 3.1 `run_in_background`

```
run_in_background(tool_name: str, arguments: dict) -> {"handle_id": str}
```

Fires `tool_name(arguments)` in a child DBOS workflow. Returns
immediately with an opaque `handle_id` (format: `bg_<32 hex>`).
`tool_name` must resolve in the agent's registered tool set
(builtins, MCP tools, local Python tools). Sub-agent tool names
and client-side tool names are rejected at call time.

**Errors**

- `unknown_tool`: name not registered.
- `ineligible_tool`: tool is a sub-agent or client-side tool.
- `limit_exceeded`: parent has
  `max_concurrent_background_tools` (default 8) already pending.

### 3.2 `read_inbox`

```
read_inbox() -> {
  "items": [InboxItem, ...],
  "truncated": bool,
  "remaining": int,
}
```

Dequeues completed, unread items. Caps per call mirror OmniAgents:
at most `max_inbox_items_per_read` items (default 32) or
`max_inbox_bytes_per_read` bytes (default 16 KiB), whichever comes
first. Items returned are stamped `read_at = now()` so they don't
reappear.

```
InboxItem = {
  "handle_id":    str,
  "tool_name":    str,
  "status":       "completed" | "failed" | "cancelled",
  "result":       Any,    # present iff status == "completed"
  "error":        str,    # present iff status == "failed"
  "started_at":   int,    # unix seconds
  "completed_at": int,
}
```

### 3.3 `list_background`

```
list_background() -> {"handles": [{handle_id, tool_name, status, started_at}, ...]}
```

Non-blocking snapshot of every handle whose row is either
`status=running` or completed-but-unread. Intended for the LLM to
introspect ("am I still waiting on anything?") without pulling
result payloads.

### 3.4 `cancel_background`

```
cancel_background(handle_id: str) -> {"status": str}
```

Sets `cancel_requested=TRUE` on the handle's row. The running
child workflow runs its step to completion (DBOS `@step` is
atomic), then before signaling the parent checks
`cancel_requested`; if set, it posts `status="cancelled"` and
discards the actual result. Returns the *requested* status; the
final delivered status lands in the inbox.

---

## 4. Agent Loop Changes

Three rules, inserted into the existing loop:

1. **Before each LLM iteration**: if
   `unread_inbox_count(task_id) > 0`, prepend one ephemeral
   framework system message to the LLM call's input:
   > `[system] {N} background tool result(s) are in your inbox.
   > Call read_inbox to retrieve them.`
   The notice is ephemeral — not persisted as a conversation item
   — and regenerated each iteration. Once the agent reads the
   inbox the count drops and the notice stops being injected.

2. **After each LLM iteration that produces no tool calls and no
   final output-terminating signal:**
   - If `pending_bg` is empty and the inbox is empty → the task
     completes via the existing finalization path.
   - Otherwise → persist the LLM's text as an interim assistant
     message (streamed to the client over the still-open SSE
     connection), then block on
     `dbos_recv_async(topic="async_work_complete")`. **This wait
     is internal to the workflow; the client's request is still
     in-flight and does NOT require a new user message to
     resume.** On wake, reconcile pending/inbox and loop back to
     rule 1, which triggers the next LLM iteration automatically.

3. **Pending state** lives in a `ContextVar`-scoped `RuntimeState`
   object attached to the workflow. On crash replay, it's rebuilt
   by querying `background_tool_calls WHERE parent_task_id=self
   AND status='running'` — so recovered workflows re-enter the
   wait with the correct set.

Pseudocode (schematic):

```python
async def _run_agent_loop(task_id, agent_id):
    runtime = RuntimeState.for_workflow(task_id, agent_id)
    runtime.pending_bg = bg_store.list_running(task_id)

    while True:
        unread = bg_store.count_unread(task_id)
        if unread > 0:
            _inject_inbox_notice(task_id, unread)

        llm_response = await _call_llm_step(...)

        if llm_response.tool_calls:
            for call in llm_response.tool_calls:
                await _dispatch_tool(runtime, call)  # direct or @step
            continue

        if not runtime.pending_bg and unread == 0:
            await _finalize_task(task_id, llm_response)
            return

        await _persist_interim_assistant_message(task_id, llm_response)
        await _wait_for_async_work(runtime)  # dbos_recv_async + reconcile
```

`_wait_for_async_work` is the one receive site. Steering messages
and (future) named-session completions send to the same topic with
a discriminated `kind` field; the function dispatches on `kind`.

---

## 5. Data Model

One new table. No changes to existing tables.

```sql
CREATE TABLE background_tool_calls (
    handle_id         VARCHAR(64) PRIMARY KEY,
    parent_task_id    VARCHAR(64) NOT NULL REFERENCES tasks(id)
                      ON DELETE CASCADE,
    tool_name         VARCHAR(256) NOT NULL,
    arguments         TEXT NOT NULL,            -- JSON-encoded
    status            VARCHAR(32) NOT NULL,     -- running|completed|failed|cancelled
    result            TEXT,                     -- JSON, null until terminal
    error             TEXT,                     -- null unless status=failed
    cancel_requested  BOOLEAN NOT NULL DEFAULT FALSE,
    started_at        INTEGER NOT NULL,
    completed_at      INTEGER,                  -- null while running
    read_at           INTEGER,                  -- null until dequeued
    CHECK (status IN ('running','completed','failed','cancelled'))
);
CREATE INDEX ix_bg_tool_calls_parent_task_id
    ON background_tool_calls(parent_task_id);
CREATE INDEX ix_bg_tool_calls_parent_unread
    ON background_tool_calls(parent_task_id, completed_at)
    WHERE status != 'running' AND read_at IS NULL;
```

- `handle_id` is the DBOS child workflow ID. Prefixed (`bg_`) for
  readability and to namespace it away from task IDs.
- `arguments` / `result` are stored as JSON text (same encoding as
  existing `conversation_items.data`).
- `read_at` is set by `read_inbox`, which runs in a single
  `UPDATE ... RETURNING` to avoid a read-modify-write race.

A new abstract `BackgroundToolStore` in
`agent_plane/stores/background_tool_store/__init__.py`:

```python
class BackgroundToolStore(ABC):
    @abstractmethod
    def create(self, parent_task_id: str, handle_id: str,
               tool_name: str, arguments: str) -> None: ...
    @abstractmethod
    def mark_completed(self, handle_id: str, result: str) -> None: ...
    @abstractmethod
    def mark_failed(self, handle_id: str, error: str) -> None: ...
    @abstractmethod
    def mark_cancelled(self, handle_id: str) -> None: ...
    @abstractmethod
    def request_cancel(self, handle_id: str) -> None: ...
    @abstractmethod
    def is_cancel_requested(self, handle_id: str) -> bool: ...
    @abstractmethod
    def list_running(self, parent_task_id: str) -> list[str]: ...
    @abstractmethod
    def list_active(self, parent_task_id: str) -> list[BackgroundHandle]: ...
    @abstractmethod
    def count_unread(self, parent_task_id: str) -> int: ...
    @abstractmethod
    def dequeue_unread(self, parent_task_id: str,
                       max_items: int, max_bytes: int,
                       read_at: int) -> tuple[list[InboxItem], int]: ...
```

`dequeue_unread` returns `(items, remaining_unread_after_dequeue)`
in a single transaction: `SELECT ... FOR UPDATE SKIP LOCKED` (or
the SQLite-equivalent `UPDATE ... RETURNING` with a subquery) so
two concurrent `read_inbox` calls (there shouldn't be any under
normal single-writer DBOS semantics, but defense in depth) never
hand the same item to both callers.

SQLAlchemy implementation follows the existing store pattern at
`agent_plane/stores/background_tool_store/sqlalchemy_store.py`.
Alembic migration is a new revision (the normal path; we don't
fold this into the initial migration).

---

## 6. DBOS Integration

### 6.1 Child workflow

```python
@workflow()
async def background_tool_workflow(
    parent_task_id: str,
    handle_id: str,
    agent_id: str,
    tool_name: str,
    arguments_json: str,
) -> None:
    """
    Run one tool, persist the result, signal the parent.
    """
    bg_store = get_background_tool_store()
    arguments = json.loads(arguments_json)

    try:
        result = await _call_tool_step(
            ToolContext(
                task_id=parent_task_id,
                agent_id=agent_id,
                workspace=_workspace_for(parent_task_id),
            ),
            tool_name,
            arguments,
        )
        if bg_store.is_cancel_requested(handle_id):
            bg_store.mark_cancelled(handle_id)
            final_status = "cancelled"
        else:
            bg_store.mark_completed(handle_id, json.dumps(result))
            final_status = "completed"
    except Exception as e:
        bg_store.mark_failed(handle_id, repr(e))
        final_status = "failed"

    DBOS.send(
        destination=parent_task_id,
        message={
            "kind": "background_tool",
            "handle_id": handle_id,
            "status": final_status,
        },
        topic="async_work_complete",
    )
```

Key properties:
- Reuses the existing `_call_tool_step` (`@step`), so tool retries,
  MCP lifecycle, telemetry spans, and spec-level timeouts apply
  exactly as they do for synchronous tool calls.
- The `cancel_requested` check runs *after* the step completes but
  *before* `DBOS.send`. A cancelled handle still incurs the cost of
  the tool call; only the delivery is suppressed.
- `DBOS.send` is durable — redelivered on replay until received.

### 6.2 Parent dispatch (direct tools)

`run_in_background`, `read_inbox`, `list_background`, and
`cancel_background` are **direct tools** (workflow-dispatched),
not `@step` tools. They need access to `RuntimeState.pending_bg`,
the parent's `task_id` / `agent_id`, and must call
`DBOS.start_workflow`, none of which belong inside a `@step`.

This matches the pattern the porting doc calls out in Rift 2 and
that OmniAgents uses for its `sys_*` tools. Agent-plane already
uses workflow-level dispatch for sub-agent spawn/check/cancel, so
the mechanism exists — async tools register in the same dispatch
table.

```python
# agent_plane/tools/direct/run_in_background.py
def run_in_background(
    runtime: RuntimeState,
    tool_name: str,
    arguments: dict,
) -> dict[str, str]:
    if not _eligible(tool_name, runtime):
        raise ToolError("ineligible_tool", f"{tool_name!r} cannot run in background")
    if len(runtime.pending_bg) >= runtime.max_concurrent_background_tools:
        raise ToolError(
            "limit_exceeded",
            f"already {len(runtime.pending_bg)} tools pending",
        )

    handle_id = f"bg_{uuid4().hex}"
    bg_store = get_background_tool_store()
    bg_store.create(
        parent_task_id=runtime.task_id,
        handle_id=handle_id,
        tool_name=tool_name,
        arguments=json.dumps(arguments),
    )
    DBOS.start_workflow(
        background_tool_workflow,
        workflow_id=handle_id,           # handle_id == child workflow id
        parent_task_id=runtime.task_id,
        handle_id=handle_id,
        agent_id=runtime.agent_id,
        tool_name=tool_name,
        arguments_json=json.dumps(arguments),
    )
    runtime.pending_bg.add(handle_id)
    _emit_sse(runtime, "response.background_tool.started", {
        "handle_id": handle_id, "tool_name": tool_name,
    })
    return {"handle_id": handle_id}
```

### 6.3 Shared wake topic

All workflow-wake signals use `topic="async_work_complete"`:

| Payload `kind`    | Source                              | Parent reaction |
| ---               | ---                                 | --- |
| `background_tool` | `background_tool_workflow`          | Remove from `pending_bg`; next-turn inbox notice. |
| `named_session`   | Named-session sub-agent (future G5) | Same envelope; reconcile pending_sessions. |
| `steering`        | User `POST /v1/responses/{id}/steer`| Append steering message, reload history. |

Unifying the topic means the parent has one `dbos_recv_async` call
and one dispatch switch. This is deliberately aligned with the
G5 named-sessions port so both features share infrastructure.

The existing client-side-tool parking uses `topic="tool_result"`,
keyed per call_id. That stays on its own topic — different wait
semantics (per-call blocking for a specific id vs.
any-completion wake).

---

## 7. Error Handling and Cancellation

### 7.1 Tool failures

A raised exception inside `_call_tool_step` results in
`mark_failed` with `repr(exc)` as the error string. The inbox item
has `status="failed"` and the `error` field populated. The LLM
sees the failure and chooses how to react (retry, abandon, surface
to the user). Sibling background tools are unaffected.

### 7.2 Parent task cancellation

When the parent task is cancelled (`POST /v1/responses/{id}/cancel`
or internal cascade):

1. `bg_store.request_cancel_all(parent_task_id)` sets
   `cancel_requested=TRUE` on every `status=running` row.
2. The existing cascade-cancel logic (which already terminates
   sub-agent workflows) additionally terminates background child
   workflows via `DBOS.cancel_workflow(handle_id)` if supported.
3. Any child workflow that is already past its step's execution
   will still mark the row `cancelled` and emit the cancelled
   signal; the parent ignores the signal because the parent task
   is already terminal.

No orphaned rows: `ON DELETE CASCADE` on `parent_task_id` cleans
up `background_tool_calls` when the parent task is deleted.

### 7.3 `cancel_background` semantics

Sets `cancel_requested=TRUE` on one handle. The child either:

- Hasn't finished its step yet → step runs to completion, then the
  post-step check delivers `cancelled`.
- Already finished its step → the `cancel_requested` check happens
  before the terminal UPDATE; the row ends `cancelled` and the
  inbox item is `cancelled`.

In both cases the handle ends in the inbox with `status=cancelled`
and no result payload. The caller sees a terminal entry it can
acknowledge and move on from.

### 7.4 Timeouts

No new timeout field. Each background tool inherits `tools.timeout`
from the agent spec, enforced inside `_call_tool_step`. Timeout
raises → `mark_failed` with a timeout message.

---

## 8. Spec Changes

Opt-in by including the tools in `tools.builtins`. The validator
enforces that an agent declaring `run_in_background` also declares
`read_inbox` (otherwise completions accumulate forever). `list_background`
and `cancel_background` are optional.

```yaml
tools:
  builtins:
    - run_in_background
    - read_inbox
    - list_background       # optional
    - cancel_background     # optional
    - web_search
    - web_fetch
```

Caps are hardcoded defaults matching OmniAgents, overridable at
the top level of `tools`:

```yaml
tools:
  background:
    max_concurrent: 8
    max_inbox_items_per_read: 32
    max_inbox_bytes_per_read: 16384
```

The block is only valid when `run_in_background` is in
`tools.builtins`; otherwise the validator rejects it with a clear
message (fail-loud per Design Principle 5).

---

## 9. SSE Events

Four new event types, additive:

| Event                                   | Payload |
| ---                                     | --- |
| `response.background_tool.started`      | `{handle_id, tool_name}` |
| `response.background_tool.completed`    | `{handle_id, tool_name, status, duration_ms}` |
| `response.inbox.notice`                 | `{unread_count}` — emitted when the framework notice is prepended |
| `response.inbox.read`                   | `{handle_ids: [...]}` — emitted after `read_inbox` dequeues |

Clients render "N in background", "K results waiting", etc. The
terminal REPL's status bar is the first consumer.

---

## 10. Crash Recovery

### Parent crash

1. DBOS replays `agent_execution_workflow` from its last checkpoint.
2. `RuntimeState.pending_bg` is rebuilt by
   `bg_store.list_running(task_id)`.
3. Signals that arrived during the crash window are redelivered
   when the parent next enters `dbos_recv_async`.
4. Completed-but-unread rows persist unchanged; the inbox notice
   count reflects reality.

### Child crash

The child workflow replays from its checkpoint. If the step's
output was checkpointed (the usual case), the replay skips the
step and proceeds to the row update + `DBOS.send`. If the step
hadn't checkpointed, the tool re-executes — the same semantics
that apply today to synchronous `_call_tool_step` executions.

### Idempotency audit

- `bg_store.create` is keyed by PK `handle_id` — on replay we
  attempt `INSERT ... ON CONFLICT DO NOTHING` (SQLAlchemy:
  `insert(...).on_conflict_do_nothing()`).
- `mark_completed` / `mark_failed` / `mark_cancelled` use
  `WHERE status='running'` so a double-transition is a no-op.
- `DBOS.send` is idempotent across replay by DBOS design.
- `dequeue_unread` uses an atomic UPDATE-RETURNING pattern; a
  replay of the dequeue step returns the cached step output, so
  the row is stamped `read_at` exactly once.

---

## 11. Interactions

### Compaction

Unread rows live in `background_tool_calls` and do not enter the
conversation until `read_inbox` dequeues them — at which point
the dequeued payload is appended as a `function_call_output`
conversation item (or a new `background_tool_result` type —
decision pending, see §13). From there, compaction handles it
like any other tool result. The 16 KiB / 32-item per-read cap
bounds single-turn context growth.

### Steering (existing `inbox_closed` + `try_deliver` path)

Today the steering path is poll-based: the server-side
`try_deliver` appends to `conversation_items` and flips no signal;
the workflow detects the new item on its next `close_inbox`
checkpoint. That's sufficient today because the workflow is
rarely parked for long — client-side tool parks are typically
short.

Async-tool parks can be minutes. To keep steering responsive, the
steering endpoint (`POST /v1/responses/{id}/steer`) **also**
sends `DBOS.send(task_id, {"kind": "steering"},
topic="async_work_complete")` after a successful `try_deliver`.
The parked parent wakes, drains the message, and reloads history
— the existing `close_inbox` handshake continues to work
unchanged on its own. This is a small, additive change: the
steering path still writes to `conversation_items` exactly as
today; it just additionally emits a wake signal so long async
parks don't swallow steering latency.

When a bg completion and a steering signal land in the same
receive window, the parent drains whichever comes first, handles
it, re-enters `dbos_recv_async`, and the other wakes it
immediately. Ordering is FIFO by DBOS send timestamp.

### Pending client-side tool calls (existing `tool_result` path)

Unchanged by this design. `_wait_for_pending_calls` continues to
`dbos_recv_async(topic="tool_result")` per call_id, and the PATCH
handler continues to `DBOS.send(topic="tool_result", ...)` per
call_id. The two topics are deliberately disjoint — per-call
blocking semantics vs. any-completion wake — and the LLM-facing
contract of client-side tools (synchronous injection into the
firing iteration) is preserved.

The `run_in_background` tool rejects client-side tool names at
the eligibility check so the LLM can't accidentally route through
the wrong path.

### Sub-agents (existing `SpawnTool` + auto-collect)

`run_in_background` rejects names in the agent's sub-agent
registry. Sub-agents keep their own spawn/auto-collect API
(`SpawnTool`, `CheckSubAgentsTool`, `CancelSubAgentTool`). The
future G5 named-sessions port will route sub-agent completions
through the shared `async_work_complete` topic — at that point
sub-agents will share the wake infrastructure defined here, but
the spawn surface stays distinct because the semantics (long-
lived named actors vs. fire-and-forget tool calls) differ enough
to warrant separate LLM-facing tools.

### Guardrails (G1, future)

When policies land, they evaluate:

- `tool_call` phase on `run_in_background` → DENY prevents
  `DBOS.start_workflow`.
- `tool_result` phase when an item is dequeued by `read_inbox` →
  DENY replaces the item with a policy verdict entry; the LLM
  sees `status="denied"` and the verdict reason.

### Observability (G13)

The child workflow inherits the parent's MLflow context via the
existing `telemetry.py` hooks — each background tool call emits a
TOOL span, nested under the parent AGENT span, with the same
attributes as synchronous tool calls plus
`background=true, handle_id=<id>`.

---

## 12. Worked Example

An agent asked to summarize three GitHub issues in parallel:

```
User: Summarize issues 12, 34, 56.

[LLM turn 1] tool_calls:
  run_in_background("web_fetch", {"url": ".../issues/12"}) → {handle_id: "bg_a1..."}
  run_in_background("web_fetch", {"url": ".../issues/34"}) → {handle_id: "bg_b2..."}
  run_in_background("web_fetch", {"url": ".../issues/56"}) → {handle_id: "bg_c3..."}
  text: "Fetching all three in parallel."

[Loop] pending_bg = {a1, b2, c3}, inbox = []. LLM produced text and
       the three calls returned synchronously (run_in_background is
       instant). No further tool calls → persist interim text,
       block on dbos_recv_async.

[bg_b2 completes] signal received. pending_bg = {a1, c3}. inbox = [b2].
  → framework notice: "1 background tool result(s) in your inbox."

[LLM turn 2] tool_calls:
  read_inbox() → {items: [{handle_id: "bg_b2...", status: "completed",
                           result: <issue 34>}], ...}
  text: "Got issue 34. Waiting for 12 and 56."

[Loop] no more tool calls → persist interim, block again.

[bg_a1 + bg_c3 complete, two signals] pending_bg = {}, inbox = [a1, c3].
  → framework notice: "2 background tool result(s) in your inbox."

[LLM turn 3] tool_calls:
  read_inbox() → {items: [a1, c3], ...}
  text: <final combined summary>

[Loop] pending_bg = {}, inbox = [] → finalize task.
```

Three parallel fetches took `max(t1, t2, t3)` wall time instead of
`t1 + t2 + t3`, consuming three LLM turns regardless of fetch
latency.

---

## 13. Prior Art

Survey of how other agent frameworks handle (or decline to handle)
LLM-visible async tools, conducted April 2026 against published
docs, source code, and GitHub issues. Informs what this design
borrows, what's novel, and where to harden against known failure
modes.

### 13.1 Industry scan

| Framework | LLM-visible async? | Scope | Retrieval | Handle surface |
| --- | --- | --- | --- | --- |
| **Claude Code** (Anthropic) | Yes | `Bash` only | `BashOutput(bash_id)` per-handle poll + persistent system-reminder | `bash_id` |
| **Codex CLI** (OpenAI) | Yes | PTY sessions only | `write_stdin(session_id, "")` yield/poll on same session | `process_id` (20 concurrent cap, 10k-line buffer) |
| **DeepAgents** (LangChain) | Yes | Remote sub-agents only | `check_async_task(task_id)` per-handle poll; also `update_async_task`, `cancel_async_task`, `list_async_tasks` | `task_id` |
| **Microsoft Agent Framework** | Whole-response level | Agent run, not tool | `continuation_token` poll | — |
| **LangGraph** | Interrupt-based | Node-level | External `Command(resume=…)`; caller-driven | — |
| **OpenAI Agents SDK** | No | — | Blocks | — |
| **AutoGen / Agent Framework** | No | — | Blocks | — |
| **OpenHands** | Partial / ad-hoc | `&`-backgrounded Bash | Polling inside exec tool | — |
| **Gemini CLI** | Partial | Shell PIDs visible in output | No LLM-facing handle API | — |
| **smolagents** (HF) | No | — | Blocks (async support is feature request #334) | — |
| **Goose** (Block) | No LLM-visible | — | Blocks | — |
| **Agent-plane (this design)** | **Yes** | **Any registered tool** | **`read_inbox()` batch drain + autonomous wake notice** | **`handle_id` per call** |

**Bottom line:** LLM-visible async is the minority position — 6 of 10 surveyed frameworks expose zero LLM-level async. Of the 4 that do, each ties async to a single tool type (Bash, PTY, or remote subagent). No surveyed framework exposes a generic `run_in_background(any_tool, args)` across arbitrary tool kinds.

### 13.2 Three ways this design is unusual

1. **Per-tool generality.** `run_in_background(tool_name, args)` working across any registered tool (builtins, MCP, local) is more ambitious than any comparable framework. The closest precedent is DeepAgents, and it's scoped to remote agent runs over Agent Protocol. Genericity aligns naturally with agent-plane's existing tool model — all tools already flow through `_call_tool_step` — but LLMs have no training-data precedent for a single handle-based API spanning tool kinds. Tool descriptions and example invocations in agent prompts will matter more than they would for a per-tool naming scheme.

2. **Batch drain via `read_inbox`.** Every prior-art framework uses per-handle polling: `BashOutput(bash_id)`, `check_async_task(task_id)`, `write_stdin(session_id, "")`. None drains many completions in one call. Arguments for our approach: fewer tool calls when N tasks finish near-simultaneously, naturally bounded context growth via per-read byte/item caps, simpler prompt surface (LLM doesn't need to remember all outstanding handles). Argument against: LLMs trained through 2024–2025 have seen per-handle polling but not batch drain. A companion per-handle `read_background(handle_id)` may be worth offering post-MVP; treat as an open question.

3. **Autonomous wake + framework notice.** Runtime-injected reminders when new results arrive is the pattern-match to Claude Code's system-reminder behavior — and **exactly the pattern that caused multiple high-severity bugs in Claude Code**: infinite reminders, context exhaustion, stuck loops (Claude Code issues #11190, #11716, #12302, #13847). Every other framework with async tools (DeepAgents, Codex) makes the agent or user explicitly decide when to check — DeepAgents docs actively warn against over-polling. §13.3 below lists the specific hardening this informs.

### 13.3 Risks flagged by prior art

- **Notice-loop bugs.** Claude Code's reminder-injection pattern has caused context exhaustion at scale. Before shipping, §4 rule 1 should be hardened with:
  - The notice must **not** re-emit if the LLM's immediately prior action was `read_inbox` and the inbox is now empty — prevents oscillation.
  - The notice string is byte-capped and identical across iterations (no accumulating numbers, no listing each handle by name).
  - An e2e test covers the pathological "drain → new completion arrives → drain again" interleaving repeated many times per task, asserting no context bloat.

- **Novel batch-drain idiom.** Frontier LLMs have not been prompted on `read_inbox()` returning a list. Expect to iterate on tool description wording and example invocations in agent prompts during rollout. A per-handle `read_background(handle_id)` companion tool may prove necessary if LLMs pattern-match the Claude Code / DeepAgents idiom too strongly. Post-MVP question.

- **Mid-flight argument updates.** DeepAgents exposes `update_async_task(task_id, new_instructions)` for amending a running call mid-flight. This doesn't map obviously onto tool calls (you can't usually change the arguments to an in-flight tool call), but once future work pushes sub-agents or long-lived work into this inbox (G5 named sessions), the surface gap reappears. Noted for future reference.

- **Concurrency cap.** Our default of 8 is on the conservative end — Codex settled on 20 after iteration. Not wrong, but if production agents hit the limit early, raising to 16–20 is consistent with prior art.

### 13.4 Validated choices

Prior art supports several of our decisions:

- **"Cancel delivery, not execution"** (§7.3) matches the industry consensus. Neither DeepAgents (`cancel_async_task`) nor Codex (`kill_shell`) claim mid-step cooperative cancel — both operate on handles. Our semantics is the de facto position; mid-step cancel is unsolved everywhere.

- **Opaque `bg_<hex>` handle format** is consistent with Codex's hex-based shell IDs. DeepAgents uses `task_id`, Claude Code uses `bash_id`. The specific prefix isn't load-bearing.

- **Workflow-driven autonomy over application-driven polling.** Microsoft Agent Framework's `continuation_token` pattern is application-driven; LangGraph's interrupt is caller-driven; our design is workflow-driven (the parked workflow resumes itself on signal). This is closer to Claude Code's background-reminder model, which — notice-loop bugs aside — is the right choice for a runtime that aims to be autonomous during an agent-loop execution.

- **Reusing existing durability primitives.** DBOS `DBOS.send` + `dbos_recv_async` is the same pair agent-plane already uses for client-side tool tunneling; no new durability mechanism invented.

### 13.5 Evidence trail

- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview); Claude Code issues [#9905](https://github.com/anthropics/claude-code/issues/9905), [#19173](https://github.com/anthropics/claude-code/issues/19173) (BashOutput→TaskOutput rename), [#11190](https://github.com/anthropics/claude-code/issues/11190), [#11716](https://github.com/anthropics/claude-code/issues/11716), [#12302](https://github.com/anthropics/claude-code/issues/12302) (notice-loop bugs).
- OpenAI Codex issue [#6404](https://github.com/openai/codex/issues/6404); source: `codex-rs/core/src/tools/handlers/unified_exec.rs`.
- DeepAgents [async-subagents docs](https://docs.langchain.com/oss/python/deepagents/async-subagents); [v0.5 blog](https://blog.langchain.com/deep-agents-v0-5/).
- [OpenAI Agents SDK tools docs](https://openai.github.io/openai-agents-python/tools/).
- [LangGraph interrupts docs](https://docs.langchain.com/oss/python/langgraph/interrupts).
- [Microsoft Agent Framework background responses](https://learn.microsoft.com/en-us/agent-framework/agents/background-responses).
- OpenHands issue [#7869](https://github.com/OpenHands/OpenHands/issues/7869); smolagents issue [#334](https://github.com/huggingface/smolagents/issues/334); Gemini CLI issues [#5941](https://github.com/google-gemini/gemini-cli/issues/5941), [#13594](https://github.com/google-gemini/gemini-cli/issues/13594).

---

## 14. Open Questions

1. **Inbox ordering** — FIFO by `completed_at` is the proposed
   default. LIFO might fit "newest on top" intuition but makes
   iteration order non-deterministic across retries. Recommend
   FIFO.

2. **Inbox overflow** — if unread items exceed some threshold (say
   128), should `run_in_background` start failing with
   `inbox_full` to apply backpressure? Current proposal: no hard
   cap on unread count; the notice repeats until drained. Revisit
   if we observe runaway agents.

3. **Conversation item type for dequeued results** — options are
   (a) append each dequeued item as a `function_call_output`
   (piggy-back on the existing type), (b) introduce a new
   `background_tool_result` item type. (b) is cleaner for
   observability and compaction tuning but adds a store-layer
   type and schema change. Recommend starting with (a) and
   revisiting if we need to distinguish.

4. **Explicit `wait_for_inbox` tool** — the current design has
   implicit waiting (no-tool-calls → block). If the LLM tends to
   produce noisy "still waiting" filler on each wake, an explicit
   `wait_for_inbox(timeout_seconds)` tool would let it yield
   without text. Defer until we see the behavior in practice.

5. **Result visibility in `list_background`** — should
   already-read handles remain in `list_background` output
   (as `status=read`) for introspection? Proposed: exclude read
   items; they live in conversation history for that purpose.

6. **Mid-step cancellation** — deferred. Requires either DBOS
   step-level cancel (not currently exposed) or the tool runner
   cooperatively checking a cancel flag. Tie this in with G6
   (phase-aware cancel).

7. **Budget / cost accounting** — tools consume tokens and API
   calls. A cumulative "background cost so far" surfaced to the
   LLM would let it reason about when to cancel runaway work.
   Not in scope for v1.
