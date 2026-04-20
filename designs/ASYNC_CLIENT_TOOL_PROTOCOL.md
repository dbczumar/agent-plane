# Async Client-Side Tool Protocol

End-to-end interaction pattern for `synchronous: false` client-side
tools as currently implemented (Phase 5 server-side + SDK
wire-protocol slice).

This describes what's on the wire **today**. The SDK lifecycle
that automates the client side (D6) is deferred — see
`tests/_adherence/phase5.md` for what's missing.

---

## Quick mental model

| Concept | Sync client tool | Async client tool (Phase 5) |
|---|---|---|
| How LLM picks | Default (no `synchronous` in args) | LLM sets `arguments.synchronous = false` per call |
| Server behavior | Parks the workflow on a `function_call` with `status: "action_required"` | Dispatches as a `kind="client_tool"` task and **keeps streaming** |
| LLM sees | Nothing yet — call is pending | A `function_call_output` with `{task_id, kind: "client_tool", ...}` handle inline, immediately |
| Client unblocks server via | `PATCH /v1/responses/{id}` `tool_results` | `PATCH /v1/responses/{id}` `async_tool_results` |
| Result delivery to LLM | Workflow resumes, FCO is the tool result | Drain auto-delivers `[System: task ... completed]` user message on next iteration |
| Cancel | Standard cancel flow | New SSE event `response.client_task.cancel` so client can abort local work |

The async path piggybacks on the same `async_work_complete`
drain channel used by `@tool(synchronous=False)` and async
sub-agents. One mechanism, three producers.

---

## 1. Setup — client declares the tool

Client POSTs `/v1/responses` with the tool schema. Async
dispatch is a **per-call** choice expressed by the LLM as a
real argument — the tool's schema surfaces the option by
declaring `synchronous` inside `parameters.properties`:

```json
{
  "model": "my-agent",
  "input": "...",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "long_compute",
        "description": "...",
        "parameters": {
          "type": "object",
          "properties": {
            "n": {"type": "integer"},
            "synchronous": {
              "type": "boolean",
              "description": "Set false to dispatch as a background task..."
            }
          },
          "required": ["n"]
        }
      }
    }
  ]
}
```

**Spec-compliant.** `parameters` is JSON Schema; `properties`
is exactly where the OpenAI tool spec puts argument schemas.
Nothing extends the OpenResponses tool wrapper; no top-level
non-OpenAI field; nothing inside `function` outside the
canonical shape. Tools that don't declare `synchronous` in
properties have no async option — every call falls through to
the sync (parking) path.

Server side:
- `parse_client_side_tool_spec`
  (`agent_plane/tools/client_specified/__init__.py`) just
  validates the OpenAI shape; it doesn't read any
  Phase-5-specific field at parse time.
- `_wants_async_dispatch`
  (`agent_plane/runtime/workflow.py`) reads
  `arguments.synchronous` from each LLM-emitted `function_call`
  and routes per-call — `False` ⇒ async dispatch, anything else
  (omitted, `True`, malformed) ⇒ sync.

SDK side:
- `@tool(synchronous=False)` on a Python function attaches
  `ToolMetadata(synchronous=False)`
  (`sdks/python-client/agent_plane_client/tools/_decorator.py`).
- `build_tool_handler`
  (`sdks/python-client/agent_plane_client/tools/_handler.py`)
  injects a `synchronous: {type: boolean, description: ...}`
  property into the parameters schema only when the tool
  actually opts in — sync tools' schemas stay pristine.
- The SDK strips `arguments.synchronous` before invoking the
  user's function so tool authors don't have to declare a
  `synchronous` parameter on their signatures (and a name
  collision is rejected at handler-build time).

---

## 2. Turn 1 — LLM calls the async tool

LLM emits a `function_call` with `arguments` containing
`synchronous: false`. The workflow's `_handle_tool_calls`
splits client tools by `_wants_async_dispatch(tc)` (which
reads the per-call argument) and routes the async ones to
`_dispatch_async_client_tools`.

For each async call, `_dispatch_async_client_tools`:

1. **Creates a `kind="client_tool"` task row** in `task_store`
   anchored to the parent's conversation. **No DBOS workflow
   is attached** — `task_store.start` is NOT called. The
   client owns execution; the server only owns the row.

2. **Persists a `function_call_output`** whose `output` is JSON:

   ```json
   {
     "task_id": "task_xyz",
     "kind": "client_tool",
     "tool_name": "long_compute",
     "status": "in_progress",
     "message": "Client-side tool 'long_compute' dispatched as task 'task_xyz'. Result will auto-deliver as a system message when the client PATCHes back. To poll call check_task; to abort call cancel_task."
   }
   ```

3. **Streams both items out** via the standard SSE flow —
   `function_call` then `function_call_output` with the handle
   in its `output` field.

The server **does not park**. The LLM sees the handle inline as
the FCO and continues — possibly calling more tools, generating
text, or completing the turn. The parent's
`agent_execution_workflow` loop continues to next iteration and
may drain async work if it next blocks waiting for it.

### What the SSE stream looks like to the client

```
event: response.output_item.done
data: {"item": {"type": "function_call",
                "name": "long_compute",
                "arguments": "{\"n\": 5}",
                "call_id": "call_abc",
                "status": "completed",
                "model": "my-agent"}}

event: response.output_item.done
data: {"item": {"type": "function_call_output",
                "call_id": "call_abc",
                "output": "{\"task_id\": \"task_xyz\", \"kind\": \"client_tool\", ...}"}}

(...stream continues with text deltas, more tool calls, etc.)
```

The SDK parses these into `ToolCall` and `ToolResult` events
respectively (no Phase-5-specific event for the dispatch
itself — the handle is just the FCO output).

---

## 3. Out-of-band — client executes the tool

The client (today: hand-rolled, since SDK lifecycle is
deferred) sees `ToolCall` + `ToolResult` events in its SSE
stream and:

1. Detects from its own tool registry that this call's tool
   was declared `synchronous: false`.
2. Parses the `ToolResult.output` as JSON to extract `task_id`.
3. Runs the actual tool body however it wants —
   `asyncio.create_task`, a worker pool, a subprocess, a
   distributed job, etc. The server does not care how
   execution happens.

The server is meanwhile free to keep streaming: the LLM may
call other tools, emit text, or complete the turn while the
async tool body is still running on the client.

---

## 4. Tool finishes — client PATCHes the result

Client sends:

```
PATCH /v1/responses/{root_response_id}
Content-Type: application/json

{
  "async_tool_results": [
    {
      "task_id": "task_xyz",
      "status": "completed",
      "output": "compute result text"
    }
  ]
}
```

Failure variants:

```json
{"task_id": "task_xyz", "status": "failed",
 "error": {"message": "...", "traceback": "..."}}
```

```json
{"task_id": "task_xyz", "status": "cancelled"}
```

The `output`, `error`, and `traceback` fields are all
optional. **No invented defaults** — `output: None` stays
`None` server-side; `error.traceback` is omitted (not stored
as `""`) when not provided.

### What the server does

`_apply_async_tool_results`
(`agent_plane/server/routes/responses.py`) splits into two
helpers:

`_lookup_client_tool_task`:
1. `task_store.get(task_id)` → 404 if missing.
2. Verify `task.kind == "client_tool"` → 409 if not (prevents
   PATCH from corrupting `agent_task` / `tool` / `sub_agent`
   rows).
3. If `task.status` is already terminal, return `None` →
   handler skips (G3 first-write-wins; late "completed" PATCH
   does NOT override a prior "cancelled").

`_finalize_and_signal`:
1. `task_store.finalize_async_task(task_id, status, output, error)`
   writes `manual_status` / `manual_output` /
   `manual_error_message` / `manual_error_traceback` columns
   on the task row. (Tasks WITH a DBOS workflow shouldn't use
   this path — DBOS itself is the source of truth there.)
2. `dbos_send_async(parent_root_task_id, payload, topic="async_work_complete")`
   delivers the result to the parent loop.

The PATCH may carry both `tool_results` and `async_tool_results`
in the same body — they're processed independently. One PATCH
can complete a sync tool and an async tool simultaneously.

---

## 5. Parent picks up the result

The parent loop's `_drain_async_completions` step
(`agent_plane/runtime/workflow.py`) drains any queued
`async_work_complete` payloads at the **top of every iteration**
(D4) and converts each into a user-role conversation item:

```
[System: task task_xyz (client_tool) completed]
<output content>
```

For failure:
```
[System: task task_xyz (client_tool) failed]
<error message>
<traceback>
```

For cancellation:
```
[System: task task_xyz (client_tool) cancelled]
```

The LLM sees the system message on the next turn and can
react. The drain channel is shared with `@tool(synchronous=False)`
work and async sub-agents — clients should not try to
distinguish producers from the message format alone, but the
`task_id` cross-references the original handle the LLM
received.

If the parent has already finished by the time the PATCH
arrives, the signal is queued in DBOS and harmless.

---

## 6. Cancellation paths

### 6a. Direct cancel of the client_tool task

```
POST /v1/responses/{task_id}/cancel
```

Where `task_id` is the **client_tool task id from the handle**
(NOT the parent response id). Server-side:

1. `task_store.finalize_async_task(task_id=..., status="cancelled")`
   marks the row terminal in-store.
2. Emits an SSE event on the parent's SSE stream:

   ```
   event: response.client_task.cancel
   data: {"task_id": "task_xyz"}
   ```

3. The drain does NOT get a signal here. The expectation is
   that the client cancels its local work and PATCHes back
   `status: "cancelled"`, which then triggers the
   `async_work_complete` signal via the normal path.

### 6b. Parent cancel propagation

```
POST /v1/responses/{parent_response_id}/cancel
```

Triggers `cancel_pending_child_tools`
(`agent_plane/runtime/workflow.py:2953`) BEFORE cancelling the
parent itself (parent cancel must be issued from outside the
workflow — once DBOS marks it CANCELLED, no further `@step`
can run, including `list_tasks`).

For each non-terminal child of the parent:
- `kind="tool"` (server-side `@tool(synchronous=False)`):
  `cancel_workflow_async(child.id)` kills its DBOS workflow.
  Its `background_tool_workflow` enters its
  `except BaseException` block, sends an
  `async_work_complete` payload with `status="cancelled"`, and
  re-raises so DBOS records the workflow as CANCELLED.
- `kind="client_tool"` (Phase 5): `finalize_async_task(status="cancelled")`
  marks the row terminal in-store and emits a
  `response.client_task.cancel` SSE event so the client
  cancels its local asyncio task. As above, the drain doesn't
  get a signal directly — the client's PATCH delivers the
  terminal state via `async_work_complete`.
- `kind="sub_agent"`: handled by its own signaling.

Per-child failures are swallowed so one cancel error does not
block the others.

### 6c. Race: cancel beat the PATCH

If `cancel` finalizes the row as `cancelled` and the client's
late `PATCH async_tool_results` arrives with `status="completed"`:

- `_lookup_client_tool_task` sees `task.status in TERMINAL_STATUSES`
  and returns `None`.
- `_apply_async_tool_results` skips the entry.
- The PATCH succeeds (200 OK) but is a no-op.
- The original `cancelled` status sticks (G3 first-write-wins).

The client may also be late in seeing the
`response.client_task.cancel` SSE event — that's fine; the
late PATCH is still safe.

---

## 7. SDK SSE event surface

The python-client SDK (`agent_plane_client`) parses the
following Phase-5-relevant events:

| SSE event | SDK event class | Notes |
|---|---|---|
| `response.output_item.done` (item.type = `function_call`) | `ToolCall` | Same event as sync tool — distinguish by inspecting your own tool registry |
| `response.output_item.done` (item.type = `function_call_output`) | `ToolResult` | For async tools, `output` is the handle JSON |
| `response.client_task.cancel` | `ClientTaskCancel(task_id)` | Phase 5 only. Malformed frames (missing/empty `task_id`) are dropped silently |

Defined in `sdks/python-client/agent_plane_client/_events.py`
and parsed in `_sse.py`.

---

## 8. Idempotency, failures, edge cases

| Scenario | Behavior |
|---|---|
| PATCH unknown `task_id` | 404 NOT_FOUND |
| PATCH a non-client_tool kind (e.g. `agent_task`, `tool`, `sub_agent`) | 409 CONFLICT — these have their own status sources of truth |
| Repeat PATCH after terminal status | 200 OK no-op (G3 first-write-wins) |
| Empty `tool_results` + populated `async_tool_results` | Both legs accepted |
| Empty both arrays | 200 OK no-op |
| Parent already terminal when PATCH arrives | Signal queued in DBOS, harmless |
| Async tool exceeds 1h on the client | NOT enforced server-side today; D6 SDK lifecycle should add `asyncio.wait_for(timeout=3600)` |
| SSE drops mid-flight | See § 9 — polling/HTTP fallback recovers without server-side state loss |

---

## 9. SSE drops — polling and reconnect

The SSE stream is a *delivery convenience*, not the source of
truth. Every async-tool fact lives in two places that survive
disconnects:

- **The conversation history** (queryable via
  `GET /v1/conversations/{conv_id}/items`). Each dispatched
  async client tool produced a `function_call` and a
  `function_call_output` whose `output` is the handle JSON
  with `{task_id, kind: "client_tool", ...}`. These items are
  durable.
- **The `client_tool` task row** in `task_store` (queryable
  via `GET /v1/responses/{task_id}`). Status is
  `in_progress` until terminal; PATCH and direct cancel both
  finalize it in-store.

So the only thing SSE delivers that polling can't reproduce is
**timing** — a `response.client_task.cancel` event tells the
client *immediately* that the server cancelled the task. With
the SSE channel down, the client has to discover the same
fact by checking task status.

### What survives a disconnect

| Server-side | Affected by SSE drop? |
|---|---|
| Parent agent workflow | No — DBOS-durable, runs to completion regardless |
| `client_tool` task rows | No — same database, same status |
| Drain of `async_work_complete` payloads | No — DBOS topic is durable |
| Outbound SSE events for already-disconnected stream | Yes — events are dropped (no replay buffer); next reconnect must re-derive state via polling |

| Client-side | Affected by SSE drop? |
|---|---|
| In-flight asyncio tool tasks | Only if the client process itself died |
| Outgoing `PATCH async_tool_results` | No — PATCH is a separate HTTP call, idempotent server-side |
| Knowledge of server-side cancellations | **Yes** — without `response.client_task.cancel`, the client doesn't learn until it polls |

### Polling recovery procedure

A client that lost SSE (or a new client process picking up an
existing conversation) recovers with this sweep:

1. **List in-flight async tools.** Fetch the conversation
   items: `GET /v1/conversations/{conv_id}/items?limit=...`.
   Find every `function_call_output` whose `output` JSON
   parses to `{kind: "client_tool", task_id, ...}`.

2. **Check each task's terminal state.** For each `task_id`
   from the handles, call `GET /v1/responses/{task_id}`. Read
   `status`:
   - `in_progress` (or `queued`) → still server-side active.
     If your local asyncio task is still running, keep going.
     If you have no local task (process restart), see step 4.
   - `cancelled` → the server cancelled while you were
     disconnected. Stop your local work; optionally PATCH
     `status: "cancelled"` (idempotent).
   - `completed` / `failed` → already terminal. Don't
     re-PATCH (would no-op anyway).

3. **For tools that finished while disconnected**, send the
   PATCH `async_tool_results` as normal. Server signals the
   parent's drain on receipt.

4. **For tools where you have no local task** (process restart
   mid-flight): the SDK can't safely re-run the body
   (side-effects, idempotency unclear). Two safe options:
   - Mark as failed with a clear error
     (`{status: "failed", error: {message: "Client process
     restarted; tool body lost"}}`) so the parent sees the
     terminal state and can decide what to do.
   - Re-run only if the tool is explicitly marked safe to
     retry (D6 will need a `@tool(retry_on_recovery=True)`
     opt-in or similar; not yet designed).

5. **(Optional) Resume the SSE stream.**
   `GET /v1/responses/{root_response_id}` with
   `Accept: text/event-stream` returns the live tail of
   events. Past events are NOT replayed — that's why steps
   1–4 are mandatory before opening the stream, not after.

### Server-side guarantees this depends on

- `task_store.get(task_id)` always returns the current
  terminal state of a `client_tool` row (no stale caching).
- The conversation-items endpoint includes
  `function_call_output` items as soon as they're persisted —
  before the parent moves on (it can't, the FCO is
  synchronously persisted before the LLM sees the next
  iteration).
- Cancellation finalizes the row *before* emitting the SSE
  event, so a polling client never sees `in_progress` for a
  task the server has already cancelled.

### What's NOT yet built

- **Server-side replay buffer for SSE events.** Reconnects
  start fresh from the live tail; no `Last-Event-ID` /
  back-fill mechanism. Polling recovery is mandatory after
  any disconnect.
- **SDK-side automatic reconnect-and-sweep.** The D6
  lifecycle work needs to bake this into the client so the
  user doesn't write the polling loop by hand. See § 11
  "What's missing."

---

## 10. E2E test plan (for D6 client integration)

The deferred D6 SDK lifecycle work needs end-to-end tests
that prove the python-client SDK correctly drives a real
agent-plane server through every async-client-tool path.
Tests live in `tests/e2e/test_async_client_tool_e2e.py` and
use a real LLM via `--llm-api-key` (existing pattern).

Each test follows the same shape:

1. **Define a `@tool(synchronous=False)` function** that
   produces a deterministic output (e.g. `return f"DONE_{n}"`).
   Use a `threading.Event` or `asyncio.Event` to coordinate
   timing where needed.
2. **Build the handler** with `build_tool_handler([fn])`.
3. **Open `client.responses.stream(...)`** with the handler
   attached.
4. **Wait for the response to reach a terminal state** and
   inspect the persisted conversation items.
5. **Assert** on the LLM's final output text, the tool
   invocation count/args, the persisted `function_call` and
   `function_call_output` items, and the
   `[System: task ... completed]` system messages the drain
   delivered.

### Test inventory

| # | Test name | Production breakage that would fail it |
|---|---|---|
| 1 | `test_async_dispatch_returns_handle_to_llm_e2e` | SDK fails to inject `synchronous` property → LLM never sets it → server takes sync path → `function_call_output` is the parked-tool result instead of a `{task_id, kind: "client_tool"}` handle |
| 2 | `test_async_completion_delivers_system_message_e2e` | SDK forgets to PATCH `async_tool_results` → drain never fires → no `[System: task ... completed]` user message → LLM stalls or hits max_iterations |
| 3 | `test_async_failure_surfaces_traceback_e2e` | Tool body raises; SDK must catch, PATCH `status: "failed"` with `error.message` and `error.traceback` → drain delivers `[System: task ... failed]` with the message → LLM can react |
| 4 | `test_async_parallel_fan_out_e2e` | LLM dispatches 3 async tools in one turn. Tool body sleeps 2s. Total wall time should be ≈ 2s, not 6s. Asserts SDK runs them concurrently as separate `asyncio.Task`s rather than serializing |
| 5 | `test_mixed_sync_and_async_in_same_turn_e2e` | LLM emits one sync tool call (no `synchronous` arg) and one async (`synchronous: false`) in the same turn. SDK must PATCH `tool_results` for the sync one AND fire-and-forget the async one — both legs of the PATCH body coexisting |
| 6 | `test_parent_cancel_propagates_to_async_tool_e2e` | While async tool is running, `POST /cancel` on the parent. Server emits `response.client_task.cancel`. SDK must cancel the matching local `asyncio.Task` AND PATCH `status: "cancelled"`. Asserts: local task's `Cancelled` exception was raised, server-side row is `cancelled`, no `[System: task ... completed]` was delivered |
| 7 | `test_direct_cancel_of_client_tool_task_e2e` | While async tool is running, `POST /v1/responses/{task_id}/cancel` directly on the client_tool task (not the parent). Same expected SDK behavior as #6 |
| 8 | `test_late_patch_after_cancel_is_noop_e2e` | Cancel arrives after the SDK already kicked off PATCH but before it lands. The PATCH (with `status: "completed"`) must hit the G3 first-write-wins no-op path — server keeps `cancelled`, no second drain signal |
| 9 | `test_sse_disconnect_polling_recovery_e2e` | Drop the SSE connection mid-flight (close the httpx response). SDK must (a) not crash, (b) on next `responses.stream` or `responses.poll` call, scan conversation history for in-flight client_tool handles, (c) discover any cancelled-while-disconnected tasks, (d) PATCH any tools that completed locally during the disconnect |
| 10 | `test_max_lifetime_cap_e2e` | Tool body deliberately sleeps 3700s. SDK's `asyncio.wait_for(timeout=3600)` must raise `TimeoutError` at the 1h mark, then PATCH `status: "failed"` with a clear timeout message. Test uses a fast-clock fixture (monkeypatch the cap to 5s) so the test itself runs in ~5s |
| 11 | `test_idempotent_patch_on_network_retry_e2e` | Wrap the SDK's PATCH call to fire twice (simulating network retry). Server must accept both — first one wins, second is a no-op. Asserts only one `[System: task ... completed]` message in the conversation |
| 12 | `test_full_stack_kitchen_sink_e2e` | One turn: sub-agent spawn (Phase 4) + async server-side `@tool` (Phase 2) + async client-side `@tool` (Phase 5), all running in parallel. Cancel the parent while all three are in-flight. Asserts: all three children cancelled, all PATCH idempotency holds, no zombie state in `task_store` |

### Fixture infrastructure needed

- **`@tool(synchronous=False)` test tools** with deterministic
  outputs and `threading.Event`-gated timing — define in
  `tests/e2e/_fixtures/async_client_tools.py`.
- **`make_e2e_client(api_key)`** — already exists.
- **`assert_conversation_items_contain(...)`** helper — match
  on item type + content predicates.
- **`drop_sse_mid_flight(...)`** context manager — wraps the
  httpx transport and forcibly closes the response stream
  partway through.
- **`time_warp_max_lifetime(seconds)`** fixture — monkeypatch
  the SDK's 1h cap to a small value for #10.

### Test-cadence recommendation

- Run the full suite with `pytest tests/e2e/test_async_client_tool_e2e.py
  --llm-api-key $LLM_API_KEY -v` before merging any change to
  `agent_plane/runtime/workflow.py`,
  `agent_plane/server/routes/responses.py`,
  `sdks/python-client/agent_plane_client/_responses.py`, or
  `sdks/python-client/agent_plane_client/tools/_handler.py`.
- Tests #4 and #6 are the most expensive (multi-second sleeps
  and real LLM round-trips, ~30s each); the rest should run in
  ≤10s each. Total suite budget: ~3 min wall-clock.
- Mandatory: also test through the terminal TUI per
  `CLAUDE.md` "Mandatory TUI Verification" — the polling-API
  E2E and the streaming-TUI path are different code paths.

---

## 11. What's missing (D6 deferred work)

Today there is **no SDK helper** that automates the client side.
A user must:
- Maintain a `call_id → tool_name` map by inspecting their own
  schemas to detect async tools.
- Parse the `function_call_output` JSON to extract `task_id`.
- Run the tool body (`asyncio.create_task` or similar).
- Track `call_id → asyncio.Task` and `call_id → task_id`.
- PATCH `async_tool_results` when the task finishes.
- Listen for `ClientTaskCancel` and cancel the matching task.
- Survive across `stream()` calls — the async PATCH may
  complete after the user's stream loop exits.
- Enforce the 1h cap themselves.

The SDK's existing `stream()` execution model (`ResponsesNamespace.stream`
in `sdks/python-client/agent_plane_client/_responses.py:120`)
runs `tool_handler.execute()` synchronously at end-of-turn and
PATCHes `tool_results` before restarting the stream. That model
doesn't fit async tools, where the server has already moved
past them while the body is still running.

D6 (deferred, see Task #38) will replace that model with
fire-and-forget asyncio tasks, automatic
`async_tool_results` PATCH on completion,
`ClientTaskCancel`-driven cancellation, and the 1h cap.

---

## 12. File reference index

| Concern | File:line |
|---|---|
| Tool spec parsing (`synchronous` flag) | `agent_plane/tools/client_specified/__init__.py:135` |
| Async tool dispatch | `agent_plane/runtime/workflow.py:2849` (`_dispatch_async_client_tools`) |
| Tool-call routing (sync vs async) | `agent_plane/runtime/workflow.py` ~2501 (`_handle_tool_calls`) |
| Drain (parent picks up result) | `agent_plane/runtime/workflow.py` (`_drain_async_completions`) |
| PATCH endpoint validation + dispatch | `agent_plane/server/routes/responses.py` (`_lookup_client_tool_task` + `_finalize_and_signal` + `_apply_async_tool_results`) |
| Task-store finalization | `agent_plane/stores/task_store/sqlalchemy_store.py` (`finalize_async_task`) |
| Parent-cancel propagation | `agent_plane/runtime/workflow.py:2953` (`cancel_pending_child_tools`) |
| `client_tool` task `kind` constant | `agent_plane/runtime/workflow.py:37` (`_CLIENT_TOOL_KIND`) |
| API documentation | `agent_plane/server/API.md` (Submit Tool Results section) |
| SDK `@tool` decorator | `sdks/python-client/agent_plane_client/tools/_decorator.py` |
| SDK `build_tool_handler` (emits `synchronous`) | `sdks/python-client/agent_plane_client/tools/_handler.py` |
| SDK SSE parser (`response.client_task.cancel`) | `sdks/python-client/agent_plane_client/_sse.py` |
| SDK `ClientTaskCancel` event class | `sdks/python-client/agent_plane_client/_events.py` |
| SDK `stream()` (current sync execute model) | `sdks/python-client/agent_plane_client/_responses.py:120` |
| Server integration tests | `tests/server/integration/test_async_client_tool_integration.py` (9 tests) |
| SDK wire-protocol unit tests | `tests/frontends/sdk/test_async_client_tool_sdk.py` (6 tests) |
| Adherence checklist | `tests/_adherence/phase5.md` |
