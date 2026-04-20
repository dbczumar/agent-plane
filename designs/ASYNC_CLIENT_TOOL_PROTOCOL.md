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
| Reconnect mid-flight | Client must remember `task_id`s of in-flight async work; no server-side reconnect sweep yet |

---

## 9. What's missing (D6 deferred work)

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

## 10. File reference index

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
