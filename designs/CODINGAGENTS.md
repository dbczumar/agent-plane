# Coding Agents via Claude Agent SDK

## Context

Agent-plane has a working agent execution loop: load agent → build prompt →
call LLM → execute tools → repeat, all durably checkpointed by DBOS. The
loop calls the LLM via agent-plane's own `llms.Client` and executes tools
via `ToolManager`. This works well for generic agentic workflows.

For coding agents, the Claude Agent SDK provides a superior executor: it
runs Claude Code's full agent loop internally — the same system prompt,
built-in tools (Bash, Read, Write, Edit, Glob, Grep), context management,
and model tuning that power Claude Code. Rather than reimplementing coding
tools, we wrap the SDK and let it handle the LLM + tool loop while
agent-plane provides durability, conversation persistence, steering, SSE
streaming, and client-side tool bridging on top.

### What exists

| Component | File | Status |
|-----------|------|--------|
| `agent_execution_workflow` | `runtime/workflow.py:1759` | DBOS `@workflow` entry point |
| `_run_agent_loop` | `runtime/workflow.py:1519` | LLM → tools → repeat loop |
| `_call_llm_streaming` | `runtime/workflow.py` | `@step` — streams tokens, checkpoints result |
| `_call_tool` | `runtime/workflow.py:584` | `@step` — tool dispatch with timeout/retry |
| `_handle_final_response` | `runtime/workflow.py:834` | Persist-first-then-check steering handshake |
| `_handle_tool_calls` | `runtime/workflow.py:1126` | Client/server split, execution, persistence |
| `_write_output` | `runtime/workflow.py:92` | Dual-path SSE (DBOS + live stream) |
| `ToolManager` | `tools/manager.py` | Tool registry, MCP connections, client-side detection |
| `ConversationStore` | `stores/conversation_store/` | Persistent item storage (message, function_call, function_call_output) |
| `TaskStore` | `stores/task_store/` | Steering handshake (try_deliver, close_inbox) |

### What's needed

A way to run the Claude Agent SDK as the executor inside `agent_execution_workflow`,
reusing the existing conversation store, task store, SSE streaming, and steering
handshake — with no changes to those components.

---

## Design Decisions

### One loop, different executors

`agent_execution_workflow` dispatches to `_run_agent_loop` today. With
the executor abstraction, the loop body calls `executor.run_turn()` and
consumes the event stream uniformly — no branching on executor type. The
only change in `agent_execution_workflow` is constructing the right
executor based on `spec.llm.executor`:

```python
executor = _create_executor(spec, tool_mgr)  # DefaultExecutor or ClaudeSDKExecutor
result = _run_agent_loop(..., executor)
```

Same `@workflow`, same task lifecycle, same SSE plumbing, same loop
function. The executor is the only thing that varies.

### The SDK manages the inner tool loop; agent-plane manages the outer turn

The Claude Agent SDK runs Claude Code's full agent loop internally: model
calls tools, observes results, calls more tools, and eventually produces a
final response. From agent-plane's perspective, one SDK "turn" is equivalent
to many iterations of the current `_run_agent_loop`.

The outer structure is:

```
_run_claude_sdk_loop:
  task start:
    1. restore session blob from artifact store → conv_home/
    2. create ClaudeSDKClient with HOME=conv_home
    3. if transcript exists in conv_home:
         use --continue (CLI replays its own transcript)
       else (first turn of conversation):
         load history from conv_store, build prompt
    4. send prompt via client.query()

  subsequent turns (same subprocess, context in memory):
    1. send new user message via client.query()

  every turn:
    5. consume SDK event stream:
       - text deltas → _write_output (SSE)
       - tool_use blocks → conv_store.append (function_call)
       - tool results → conv_store.append (function_call_output)
    6. persist final assistant message to conv_store
    7. steering handshake (close_inbox)
    8. if no more steering → go to task end
       else → loop back to "subsequent turns"

  task end:
    9. tar conv_home/.claude/ → artifact_store.put()
   10. disconnect client (kills subprocess)
   11. return _AgentLoopResult
```

Steps 5–6 are the SDK's internal loop, observed passively through the
event stream. Agent-plane doesn't call LLM or execute tools — it
persists and relays what the SDK reports. On subsequent turns, the SDK
subprocess already has the full conversation context in memory
(including any compaction), so only the new user message is sent.

### No `@step` around the SDK turn

The current loop wraps `_call_llm` and `_call_tool` in `@step` so DBOS
caches their outputs. The SDK turn is long-running (minutes, not seconds)
and internally stateful — checkpointing its output and replaying it would
require serializing the full event history, which is both expensive and
redundant with what conv_store already persists.

Instead, durability comes from conv_store writes. Each tool call and tool
result is a database transaction as it happens. On crash recovery:

1. Conversation history is intact in conv_store (every tool call and result
   persisted individually).
2. SDK client state is lost (in-memory).
3. Recovery: `_run_claude_sdk_loop` loads history from conv_store,
   builds a prompt that includes the already-completed work, and starts a
   fresh SDK session. The model sees the full prior context and continues.

This is strictly weaker than `@step`-level recovery (the SDK may re-execute
the last in-flight tool call), but the tools are idempotent file operations
(Read, Edit, Bash), and the cost of one redundant tool call is negligible
versus the complexity of checkpointing SDK state.

### Steering between turns only (v1)

The current loop checks for steering between every LLM call. The SDK's
internal loop is opaque — we can't inject messages between its internal
tool calls without killing the process.

For v1, steering is checked after the SDK turn completes. The full sequence:

```
  SDK turn completes → persist assistant message → close_inbox check
  If late steering messages found → build new prompt with steering context
                                  → send to SDK (client is persistent)
                                  → consume next turn
```

This reuses `_handle_final_response`'s persist-first-then-check pattern
exactly. The SDK client is persistent across `query()` calls, so the
second turn sees the full conversation context including the steering
message.

Mid-turn steering (interrupting the SDK's internal loop) is deferred.

### Custom tool handlers via `@tool`

The SDK's `@tool` decorator is the only way to register custom tool
handlers — Python functions that run when Claude calls a tool. Under
the hood, the SDK routes these through its in-process MCP protocol,
but from our perspective it's just "register a Python function, SDK
calls it." No network, no separate process, no ports.

```python
@tool(name="Read", description="Read a file", input_schema=...)
async def read_handler(args):
    # park for client, return result
    return {"content": [{"type": "text", "text": file_contents}]}

server = create_sdk_mcp_server("client_tools", tools=[read_handler])
opts = ClaudeAgentOptions(mcp_servers={"client_tools": server})
```

The `ClaudeSDKExecutor` registers two sets of `@tool` handlers:

- **Client-side tools** (Read, Grep, etc. — whatever the client
  registered): Each handler parks for the client to execute the tool
  and return the result via `pending_tool_calls`.
- **`Task` tool** (subagent spawning): The handler creates a child
  task and blocks until it completes.

### Client-side tools via `pending_tool_calls`

Client-side tools use the existing `pending_tool_calls` table and
tunneling mechanism from `SUBAGENT_WORKFLOW.md` — one mechanism for
both top-level SDK tasks and subagents.

When a client-side tool handler is invoked by the SDK, it parks:

1. INSERT into `pending_tool_calls` (`status="action_required"`)
2. Publish `function_call` to the task's response output with
   `status: "action_required"`
3. Enter park loop (poll `pending_tool_calls` for completion)

The client sees the `action_required` item on the SSE stream, executes
the tool locally, and PATCHes the result back:

```
PATCH /v1/responses/{task_id}
Body: {call_id: "abc", output: "file contents..."}
```

Server updates the `pending_tool_calls` row → `status="completed"`.
The `@tool` handler's park loop picks up the result and returns it to
the SDK. The SDK continues reasoning.

**For top-level SDK tasks**: `root_task_id` is the task itself (or
NULL — same behavior). The `function_call` appears on the task's own
response output.

**For subagent SDK tasks**: `root_task_id` points to the parent. The
`function_call` is tunneled to the **root's** response output. The
client PATCHes the root. Same mechanism, same client API, same table.

No in-memory Futures, no dual code paths. One mechanism everywhere.

### Subagents: agent-plane owns orchestration, not the SDK

The Claude SDK has a built-in `Task` tool (previously called `Agent`)
that spawns child Claude processes. If we let the SDK manage subagents
internally, their work is opaque — no granular persistence, no SSE
visibility, no durability for the child's individual tool calls. A crash
mid-subagent loses all of the child's progress.

Instead, we **disable the SDK's built-in `Task` tool** and replace it
with an agent-plane-controlled `@tool` handler of the same name.

**Verified behavior** (Claude Code v2.1.87, SDK v0.1.35): The `Task`
tool bypasses `allowed_tools` (allowlist) but respects
`disallowed_tools` (denylist). Setting `disallowed_tools=["Task"]` on
`ClaudeAgentOptions` successfully prevents the SDK from using the
built-in subagent tool. We then register a custom `Task` `@tool`
handler via `create_sdk_mcp_server()` that agent-plane controls. The
SDK sees a tool named `Task`, calls it, and gets a result — unaware
that agent-plane is orchestrating the child.

When the SDK calls `Task(prompt="research how auth works")`, and the
child needs client-side tools and permissions, everything is **tunneled
through the root parent's response** — the same model as
`SUBAGENT_WORKFLOW.md`. The client watches one stream, responds to one
response ID, and never knows about child task IDs.

Full flow with client-side tool call + permission request:

1. Parent SDK calls `Task(prompt="research how auth works")`
2. The parent's `Task` `@tool` handler creates a child task
   (`task_store.create`) with `root_task_id` set to the
   parent's task ID. Child inherits parent's executor, client-side
   tools, and permission policy.
3. Child `agent_execution_workflow` launches with its own
   `ClaudeSDKExecutor`, conv_store, and `can_use_tool` callback.
4. The `Task` handler blocks (awaits child completion).

5. Child SDK reasons, calls `Grep(pattern="auth", path="src/")`.
   Grep is client-side → the child's `Grep` `@tool` handler:
   - Inserts row into `pending_tool_calls` (`status="action_required"`,
     `root_task_id=parent`, `task_id=child`)
   - Publishes `function_call` to **parent's** response output with
     `status: "action_required"` and `model: "parent.child"`
   - Client sees it on the **parent's** SSE stream
   - Handler enters park loop (polls `pending_tool_calls` for
     completion)

6. Client executes Grep locally, PATCHes result:
   `PATCH /v1/responses/{parent_id}` with `call_id` + output.
   Server updates `pending_tool_calls` row → `status="completed"`.

7. Child's park loop sees completed row, resumes with Grep result.

8. Child SDK calls more tools (server-side ones execute autonomously
   per `allowed_tools`, client-side ones park again). Eventually
   child SDK completes → final response persisted to child conv_store.

10. Parent `Task` handler unblocks, returns child's result to parent SDK.
    Parent persists `function_call(Task)` + `function_call_output(Task)`
    to parent conv_store. Parent SDK continues reasoning.

**Key points**:

- **One stream, one response ID**. Client-side tools and permission
  requests from any depth of subagent are tunneled to the root parent's
  response output. The client interacts with `PATCH /v1/responses/{root_id}`
  only. No per-child stream subscriptions, no child task ID discovery.
  This matches the existing `SUBAGENT_WORKFLOW.md` tunneling model.

- **`pending_tool_calls` everywhere**. Same table and lifecycle as the
  existing subagent design — INSERT on park, UPDATE on PATCH, poll
  loop in the `@tool` handler. Used for both top-level SDK tasks and
  subagents. One mechanism, no dual paths.

- **Nested subagents propagate `root_task_id`**. If child spawns a
  grandchild, `root_task_id` still points to the original root.
  Grandchild tool calls tunnel all the way up to the root's output.

- **Each subagent gets its own session state**. A child task has its
  own `conversation_id`, so it gets its own `conv_home` at
  `{tmpdir}/claude-sessions/{child_conversation_id}/` and its own
  artifact store blob at `claude-sessions/{child_conversation_id}`.
  No special handling — the same per-conversation lifecycle applies.

### Permissions: no new mechanism

There is no server-to-client permission round-trip. Permissions are
handled at two layers that already exist:

**Server-side: developer lists allowed tools in the agent spec.**
The spec's `tools` list defines which built-in tools the SDK can use on
the server. Built-in Claude Code tools use a `claude:` prefix to
distinguish them from custom and client-side tools. If a tool is listed,
it's allowed. If it's not, it's not. In YAML:

```yaml
llm:
  executor: claude_sdk
  model: claude-sonnet-4-20250514
tools:
  - claude:Bash
  - claude:Read
  - claude:Edit
  - claude:Write
  - claude:Glob
  - claude:Grep
```

At executor construction time, `claude:` prefixed tools are stripped to
their bare names and passed as `allowed_tools` on `ClaudeAgentOptions`.
`disallowed_tools=["Task"]` is always set to disable the built-in
subagent tool (see subagents section). The `can_use_tool` callback
provides optional finer-grained guardrails (e.g., block `Write` outside
the working directory). It returns `Allow` or `Deny` synchronously — no
client involvement, no parking.

**Client-side: client controls its own tools.**
The client registers tools in `POST /v1/responses`. When the SDK calls
a client-side tool, it goes through the standard `action_required` →
PATCH flow. The client can implement whatever permission logic it wants
before executing: show a prompt to the user, check a local policy,
auto-approve, etc. That's entirely the client's concern — agent-plane
doesn't prescribe it.

**The result**: if a developer wants Bash to require user approval, they
don't add Bash to the server-side allowed list — instead, the client
registers Bash as a client-side tool and implements approval logic
locally. If the developer trusts Bash on the server, they add it to
`allowed_tools` and it runs autonomously. No new item types, no new
client obligations, no permission-specific parking.

### Executor abstraction

A minimal ABC decouples the workflow loop from how LLM calls and tool
execution are performed. Three event types and one interface:

```python
@dataclass
class TextDelta:
    """A streamed text token from the model."""
    text: str

@dataclass
class ToolCallEvent:
    """A tool call that was already executed by the executor.
    Emitted only by internal executors (e.g. SDK). Both call and
    result are bundled — the workflow just persists them."""
    call_id: str
    name: str
    arguments: str
    result: str

@dataclass
class TurnComplete:
    """End of one executor turn.
    tool_calls is empty for internal executors (they handle tools
    themselves). Non-empty for external executors (workflow must
    execute them via @step and re-invoke)."""
    text: str | None
    tool_calls: list[_ToolCall]

ExecutorEvent: TypeAlias = TextDelta | ToolCallEvent | TurnComplete
```

```python
class Executor(abc.ABC):
    @abc.abstractmethod
    def run_turn(
        self,
        history: list[ConversationItem],
        instructions: str | None,
        tool_schemas: list[dict[str, Any]],
        task_id: str,
    ) -> Iterator[ExecutorEvent]:
        ...
```

One unified `tool_schemas` parameter — no server/client split. The executor
partitions internally if needed: `ToolManager.is_client_side_tool()` is
already available in the workflow context. For the SDK executor, client-side
tools get `@tool` handlers that park; server-side tools get `@tool`
handlers that execute locally. For the default executor, the split happens in the
workflow after `TurnComplete` (unchanged from today).

### Sync/async bridge

The existing `_run_agent_loop` and DBOS `@workflow` are sync. The Claude
Agent SDK client is async (`await client.query()`,
`async for msg in client.receive_response()`). The executor ABC returns
a sync `Iterator[ExecutorEvent]` to match the sync workflow.

`ClaudeSDKExecutor.run_turn()` bridges this:

1. Spawns a background thread running an asyncio event loop.
2. The async loop calls `client.query()` and iterates
   `client.receive_response()`, pushing `ExecutorEvent`s into a
   `queue.Queue` (thread-safe, bounded).
3. The sync `run_turn()` generator yields from the queue, blocking on
   `queue.get()` until the next event arrives or the async loop signals
   completion via a sentinel.

```python
def run_turn(self, history, instructions, tool_schemas, task_id):
    event_queue: queue.Queue[ExecutorEvent | None] = queue.Queue(maxsize=256)

    def _run_async():
        asyncio.run(self._consume_sdk_stream(event_queue, ...))
        event_queue.put(None)  # sentinel: stream done

    thread = threading.Thread(target=_run_async, daemon=True)
    thread.start()

    while True:
        event = event_queue.get()
        if event is None:
            break
        yield event

    thread.join()
```

`DefaultExecutor` has no async code — it wraps the existing sync
`_call_llm_streaming` directly and yields events inline. No thread
needed.

The workflow consumes the event stream uniformly:

```python
for event in executor.run_turn(history, instructions, tool_schemas, task_id):
    if isinstance(event, TextDelta):
        _write_output(task_id, ...)
    elif isinstance(event, ToolCallEvent):
        # Already executed. Persist both sides.
        _persist_and_stream(..., [function_call_item, function_call_output_item], ...)
    elif isinstance(event, TurnComplete):
        if not event.tool_calls:
            return _handle_final_response(...)
        # External executor: execute via @step, update history, loop continues
```

### Agent spec: one new field

`LLMConfig` gets one new optional field:

```python
executor: str | None = None  # "claude_sdk" or None (default)
```

When `None`, `agent_execution_workflow` creates a `DefaultExecutor` (thin
wrapper around the existing `_call_llm_streaming` `@step`). When
`"claude_sdk"`, it creates a `ClaudeSDKExecutor`. Both are passed to the
same workflow loop.

No new item types, no conversation store changes, no task store changes, no
API route changes, no SSE event type changes.

---

## Implementation

### Changed files

**`spec/types.py`** — Add `executor: str | None = None` to `LLMConfig`.

**`runtime/workflow.py`** — Create executor based on spec, pass to loop:

```python
executor = _create_executor(spec, tool_mgr)
result = _run_agent_loop(
    task_id, conversation_id, spec, agent_name, agent_id,
    instructions, tool_mgr, executor,
)
```

The loop body replaces direct `_call_llm_for_iteration` calls with
`executor.run_turn()`. `_handle_final_response`, `_handle_tool_calls`,
`_persist_and_stream`, and the steering handshake remain unchanged —
they operate on the same event/item types regardless of executor.

### New files

**`runtime/executor.py`** — The executor abstraction. Contains:
`TextDelta`, `ToolCallEvent`, `TurnComplete`, `Executor` ABC, and
`DefaultExecutor` (wraps existing `_call_llm_streaming`).

**`runtime/claude_sdk_executor.py`** — `ClaudeSDKExecutor(Executor)`.
Roughly 200–300 lines:

1. `run_turn()` — creates/reuses SDK client, sends prompt, yields events.
2. `_create_sdk_client()` — builds `ClaudeSDKClient` with `ClaudeAgentOptions`.
   Configures:
   - `allowed_tools` from spec (stripped `claude:` prefix, e.g. `["Bash", "Read", "Edit"]`)
   - `mcp_servers={"client_tools": server}` (client-side `@tool` handlers via `create_sdk_mcp_server()`)
   - `permission_mode="bypassPermissions"` (autonomous execution)
   - `system_prompt` from spec instructions
   - `model` from spec.llm.model (optional override)
3. `_consume_sdk_stream()` — iterates the SDK message stream, maps events:

   | SDK event | Action |
   |-----------|--------|
   | `StreamEvent` / `text_delta` | `_write_output(task_id, {"type": "response.output_text.delta", ...})` |
   | `StreamEvent` / `content_block_start` (tool_use) | `conv_store.append(function_call)` + `_write_output` |
   | `UserMessage` / `ToolResultBlock` | `conv_store.append(function_call_output)` + `_write_output` |
   | `ResultMessage` | capture final text |
   | `SystemMessage` / `api_retry` | log warning, detect terminal auth errors |

4. `_build_prompt_from_history()` — converts conv_store items to a text prompt
   for the SDK. Used in two cases: (a) first turn of a conversation — extracts
   just the user's message, and (b) crash recovery without an artifact store
   blob (Path B) — serializes the full conversation history so the SDK sees
   prior context. NOT used for subsequent turns within a task (the subprocess
   already has context in memory) or when an artifact store blob is available
   (the CLI replays its own transcript via `--continue`).
5. `_build_client_tool_handlers()` — registers client-side tools as `@tool`
   handlers via `create_sdk_mcp_server()`. Each handler inserts into
   `pending_tool_calls`, emits an SSE event, polls for completion, and
   returns the result.

**`server/routes/responses.py`** (if not already present) — Add endpoint:

```
POST /api/responses/{task_id}/tool_output
Body: {"call_id": "...", "output": "..."}
```

Updates the `pending_tool_calls` row for the given `call_id` to
`status="completed"`. Returns 404 if no pending call exists (already
completed or timed out).

### Reused without changes

| Component | Why no changes needed |
|-----------|----------------------|
| `ConversationStore` | Existing item types (message, function_call, function_call_output) cover all SDK events |
| `TaskStore` | Steering handshake (close_inbox) works at turn boundaries |
| `_write_output` | SSE dual-path delivery works for any event dict |
| `_handle_final_response` | Persist-first-then-check pattern applies to SDK turns identically |
| `_persist_and_stream` | Persists + streams items — called from the SDK runner |
| `_build_assistant_item` | Builds the final assistant message item — reused |
| `_item_to_output` | Converts ConversationItem to API format — reused |
| `fetch_all_items` | Loads conversation history — reused |
| `ToolManager` | Registers MCP + client-side tools. SDK runner uses it for schemas and client-side detection |
| API routes | `POST /v1/responses` unchanged. Client sends tools, gets SSE events, same contract |
| SSE event format | Same event types: `response.output_text.delta`, `response.output_item.done` |
| DBOS `@workflow` | Same `agent_execution_workflow`, just dispatches to a different loop |

---

## Durability and Recovery

### Normal operation

Every tool call and tool result is persisted to conv_store as an
individual database transaction during the SDK event stream. At task
end, the CLI's session state (transcript JSONL) is persisted to the
artifact store as a tarball.

### Crash recovery

If the server crashes mid-task:

1. Conv_store contains all completed tool calls and results up to the
   crash.
2. The in-flight tool call (if any) may or may not have completed —
   the SDK's in-memory state is lost.
3. DBOS re-invokes `agent_execution_workflow`. Recovery tries two
   paths in order:

   **Path A (artifact store has session blob):** The previous task's
   session state was persisted. Restore blob → temp dir → create new
   subprocess with `--continue`. The CLI replays its own transcript
   and re-compacts — same quality as before the crash. Then
   reconstruct just the delta (items in conv_store that postdate the
   transcript) as the new prompt.

   **Path B (no blob — first task or blob lost):** Load full history
   from conv_store, build a prompt including all prior context, send
   to a fresh subprocess. The SDK compacts if needed.

4. Worst case: the last in-flight tool call is re-executed. Since
   tools are file operations (Read returns the same content, Edit is
   a no-op if the string was already replaced, Bash may re-run a
   command), this is safe.

### What is NOT recoverable

- In-flight text streaming: tokens emitted to SSE before crash are
  lost on the client side (client reconnects). The SDK re-generates
  the response.
- The last artifact store persist (task end step 6) — if the crash
  happened before the blob was written, the next task falls through
  to Path B. No data loss, just one extra compaction pass.

### Comparison with the default loop

| Property | Default loop (`_run_agent_loop`) | Claude SDK loop |
|----------|----------------------------------|-----------------|
| LLM call recovery | `@step` — cached, no re-call | Re-calls LLM (prompt includes prior context) |
| Tool call recovery | `@step` — cached, no re-execute | Re-executes last in-flight tool (idempotent) |
| Conv_store durability | Same | Same |
| SSE recovery | Same (client reconnects) | Same |
| Recovery cost | Near-zero (cached steps) | One artifact restore + transcript replay, or one LLM re-compact |

The SDK loop trades recovery efficiency for implementation simplicity.

---

## Client-Side Tool Flow

### Registration

Client specifies tools in `POST /v1/responses` body (existing contract):

```json
{
  "tools": [
    {"type": "function", "function": {"name": "Glob", "parameters": {...}}},
    {"type": "function", "function": {"name": "Grep", "parameters": {...}}},
    {"type": "function", "function": {"name": "Read", "parameters": {...}}}
  ]
}
```

The SDK runner partitions tools:
- Tools matching `ToolManager.is_client_side_tool()` → `@tool` handlers
  that park (pending_tool_calls)
- All others → `@tool` handlers that execute locally
- Claude Code built-in tools (Bash, Read, Edit, etc.) can be either:
  - **Server-side** (default): SDK executes them on the server
  - **Client-side**: If the client registers tools with the same names,
    the client-side registration takes precedence. The SDK's built-in tools
    for those names are disabled (`allowed_tools` omits them), and the
    client-side `@tool` handler handles them instead.

### Execution

```
Client POSTs /v1/responses with tools=[Glob, Grep, Read, Edit, Bash]
  → SDK runner disables built-in Glob/Grep/Read/Edit/Bash
  → SDK runner registers client-side @tool handlers
  → SDK calls "Read" → @tool handler:
      1. INSERT into pending_tool_calls (status="action_required")
      2. publish function_call to response output with
         status: "action_required"
      3. enter park loop (poll pending_tool_calls for completion)
  → Client receives SSE event, reads local file
  → Client PATCHes /v1/responses/{task_id}
      {call_id: "...", output: "1\t#!/usr/bin/env python\n2\t..."}
  → Server updates pending_tool_calls row → status="completed"
  → @tool handler's park loop picks up result, returns to SDK
  → SDK continues reasoning
```

### Timeout

If the client doesn't respond within 5 minutes, the park loop times
out. The `@tool` handler returns an error string to the SDK. The model sees the
timeout and can retry or ask the user.

### Compaction and session state

The `ClaudeSDKExecutor` creates a single `ClaudeSDKClient` per task.
`connect()` spawns one CLI subprocess; subsequent turns send messages
to the same process via `query()` / `receive_response()`. The
subprocess stays alive until the task completes (`disconnect()`).

**Compaction**: The SDK compacts in memory using Claude Code's
battle-tested, coding-aware logic (preserves file paths, diffs, tool
sequences, reasoning chains). Because the subprocess is long-lived,
compacted state persists across all turns within a task. Agent-plane
writes every event to conv_store as it streams through (durable
mirror) but does not rebuild prompts from conv_store between turns —
the SDK already has the conversation context.

### Session state: local temp dir + artifact store

The CLI subprocess writes transcript JSONL, debug logs, and config
state to `~/.claude/` by default. To isolate per-conversation state,
each conversation sets `HOME` to a deterministic temp path:

```
{tempfile.gettempdir()}/claude-sessions/{conversation_id}/
```

`tempfile.gettempdir()` returns a stable path (typically `/tmp`)
determined by env vars (`TMPDIR` → `TEMP` → `TMP` → platform
default), consistent across processes and restarts on the same
machine.

```python
conv_home = Path(tempfile.gettempdir()) / "claude-sessions" / conversation_id
conv_home.mkdir(parents=True, exist_ok=True)

opts = ClaudeAgentOptions(
    permission_mode="bypassPermissions",
    max_turns=1,
    env={
        "HOME": str(conv_home),
        "ANTHROPIC_API_KEY": api_key,
    },
)
```

**Within a task**: Subprocess alive, compaction in memory, zero cost.
Multiple turns via `query()` / `receive_response()` on the same
subprocess.

**Across tasks**: The subprocess dies when the task ends. On the next
task, the replica has no way of knowing whether it handled the
previous task — so it always restores session state from the artifact
store and always persists it back after:

```
Task start:
  1. conv_home = {tmpdir}/claude-sessions/{conversation_id}/
  2. blob = artifact_store.get(f"claude-sessions/{conversation_id}")
     if blob exists:
       extract blob → conv_home/
  3. create ClaudeSDKClient with HOME=conv_home
  4. if transcript exists in conv_home: use --continue
     else: reconstruct prompt from conv_store (first turn)

Task end:
  5. tar conv_home/.claude/ → blob
  6. artifact_store.put(f"claude-sessions/{conversation_id}", blob)
  7. disconnect client (kills subprocess)
```

The artifact store key is `claude-sessions/{conversation_id}`. The
blob is a tarball of the `.claude/` directory — transcript JSONL plus
whatever else the CLI wrote. Typically small: ~5KB per turn for the
transcript, ~38KB for debug logs.

**Two tiers of session recovery** (best to worst):

| Scenario | What happens | Cost |
|----------|-------------|------|
| Same task (steering turns) | Subprocess alive, context in memory | Zero |
| New task (any replica) | Restore from artifact store → temp dir → `--continue` | One artifact fetch + replay + re-compact |
| Crash / missing blob | Reconstruct from conv_store, fresh session | One LLM call to re-compact |

**Conv_store remains the ultimate source of truth.** The transcript
blob in artifact store is an optimization cache — if it's missing or
corrupt, the workflow falls back to conv_store reconstruction. The
SDK re-compacts the history on the spot, same logic, same quality.
The only cost is one extra compaction pass.

---

## Not Yet

- **Mid-turn steering**: Interrupting the SDK's internal tool loop to inject
  a user message. Requires killing the SDK process and resuming, or SDK-level
  support for message injection. Deferred until there's a concrete use case
  that can't be served by between-turn steering.

- **SDK session resumption via `--resume`**: Superseded by the
  `--continue` approach in "Session state" section. `--continue`
  replays the transcript from the local temp dir (restored from
  artifact store if needed). `--resume` requires a session ID and
  offers no additional benefit over `--continue`.

- **Cost tracking**: The SDK's `ResultMessage` may include token usage and
  cost. Extracting these and writing them to `task.usage` is straightforward
  but not in scope for v1.


- **Multiple SDK models**: The Claude Agent SDK currently targets Claude. If
  it expands to other models, `spec.llm.model` would select the model within
  the SDK. No design changes needed — just pass the model through.

---

## Test Plan

1. **SDK turn completes with text-only response**: Send a prompt that
   requires no tools. Verify: assistant message persisted to conv_store,
   SSE text deltas emitted, `_AgentLoopResult.status == "completed"`.

2. **SDK turn with tool calls**: Send a prompt that triggers built-in tool
   use (e.g., "read file X"). Verify: function_call and function_call_output
   items persisted to conv_store in correct order, SSE events emitted for
   each.

3. **Client-side tool execution**: Register client tools, send a prompt that
   triggers one. Verify: `pending_tool_calls` row inserted with
   `status: "action_required"`, function_call published to response output,
   client PATCHes result → park loop resumes → SDK continues.

4. **Client tool timeout**: Register client tool, send prompt that triggers it,
   don't PATCH result. Verify: park loop times out, error returned to SDK,
   model handles gracefully.

5. **Steering between turns**: Send initial prompt → SDK completes →
   inject steering message via try_deliver → verify SDK starts new turn
   with steering context.

6. **Crash recovery**: Start SDK turn with tool calls → kill workflow
   mid-stream → DBOS re-invokes → verify conv_store has partial results →
   verify new SDK session receives reconstructed prompt → verify completion.

7. **Client-side tool overrides built-in**: Register client tool named "Read".
   Verify: SDK's built-in Read is disabled, client-side `@tool` handler
   handles Read calls instead.

8. **Mixed server + client tools**: Agent has server-side tools and
   client registers tools. Both types called in same turn. Verify: server
   tools execute locally, client tools go through async bridge.

9. **Execution timeout**: Set short execution timeout on spec. Verify:
   SDK turn is killed when timeout expires, incomplete result returned.

10. **Auth error**: Configure invalid API key. Verify: SDK reports auth error
    via SystemMessage, executor returns failed result with clear error message.

11. **Subagent spawning**: Send a prompt that triggers the Task tool. Verify:
    child task created with its own conv_store, child's tool calls persisted
    individually, parent's conv_store has function_call(Task) and
    function_call_output(Task), child's final response returned to parent SDK.

12. **Subagent crash recovery**: Start subagent → kill server mid-child-turn →
    DBOS re-invokes parent → parent re-calls Task tool → child restarts from
    its own conv_store → both complete.

13. **Built-in Task tool is disabled**: Verify `disallowed_tools=["Task"]` is
    set on `ClaudeAgentOptions`. Verify the custom `Task` `@tool` handler is
    registered and invoked instead of the built-in one.

14. **Server-side `can_use_tool` guardrail**: Configure `can_use_tool` to
    deny Write outside working directory. Send prompt that triggers Write
    to a disallowed path. Verify: callback returns Deny synchronously, no
    client round-trip, model sees denial message.
