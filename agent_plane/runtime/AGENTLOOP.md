# Layer 2: Agent Execution Loop

## Context

The DBOS plumbing (Layer 1) is complete — durable workflows, streaming,
steering handshake all work end-to-end. The placeholder workflow returns
hardcoded strings. Layer 2 replaces it with the real agent loop: load
agent → build prompt → call LLM → execute tools → repeat.

The goal is a working agent execution loop that can load an agent spec,
call an LLM, execute MCP tools, handle the steering inbox, persist
output, and stream events — all durably checkpointed by DBOS.

---

## Design Decisions

### LLM client: litellm

`litellm` wraps OpenAI, Anthropic, Cohere, etc. behind a single
`completion()` API. Returns OpenAI-format responses. Matches RUNTIME.md's
reference. `litellm.completion()` is synchronous, which works cleanly
inside DBOS `@step` functions. Add `litellm>=1.40` to dependencies.

### MCP client: the `mcp` package

The `mcp` PyPI package provides stdio and HTTP transport clients. It's
async-only, but DBOS step threads have no running event loop, so
`asyncio.run()` works inside steps. Add `mcp>=1.0` to dependencies.

### Streaming LLM calls inside DBOS steps

`call_llm` uses `litellm.completion(stream=True)`. As token chunks
arrive, they are written to the DBOS stream (for SSE delivery to
clients) **and** accumulated in memory. When the stream finishes, the
full assembled response is returned as the step's output — this is what
DBOS checkpoints.

- **Normal operation**: Clients see tokens arrive incrementally via SSE.
- **Crash recovery**: DBOS returns the cached full response from the
  completed step. The workflow re-emits it as a single output event.
  The client has reconnected anyway, so token-by-token replay isn't
  needed — they get the complete response immediately.
- **Crash mid-stream**: If the step hadn't finished (LLM call was still
  in progress), DBOS re-runs the step from scratch on recovery. The LLM
  call re-executes and streams fresh tokens. This is acceptable — the
  client reconnected and the LLM call is idempotent.

**Open question**: Verify that `write_stream()` works inside a DBOS
`@step` function during implementation. If not, the LLM call moves out
of the step and becomes a plain function call within the workflow body,
with a separate `@step` to checkpoint the assembled response afterward.

### Fresh MCP connections per request

Each workflow execution (i.e. each `POST /v1/responses`) creates its own
`ToolManager`, which connects to the agent's MCP servers at the start of
the workflow and tears them down in `finally`. Connections are not shared
or reused across requests — every execution pays the full MCP startup
cost. This is simple and avoids leaked state or stale-connection bugs,
at the expense of per-request latency for MCP server initialization.

### Agent loading via AgentCache

`AgentCache` (already implemented in `runtime/agent_cache.py`) is a
two-tier cache (memory + disk) backed by `ArtifactStore`. The empty
cache is instantiated at server startup and held in `_globals` — no
agents are loaded until a workflow requests one. When a workflow calls
`agent_cache.load(agent_id)`, the cache checks memory, then disk, then
downloads from `ArtifactStore` on a full miss (extracting, parsing, and
validating via `spec.load()`). Subsequent requests for the same agent
hit the cache.

### Store access via getter functions

`runtime/_globals.py` is a private module that holds store references
and the `AgentCache`, set once at server startup via `init()`.
Workflow code **never imports `_globals` directly**. Instead,
`runtime/__init__.py` exports typed getter functions —
`get_conversation_store()`, `get_task_store()`, `get_agent_cache()`,
etc. — that read from the private module and raise `RuntimeError` if
the runtime hasn't been initialized yet. This keeps the global state
encapsulated and gives callers a clean, discoverable API.

### Tool manager concurrency via contextvars

The `ToolManager` is stored in a `contextvars.ContextVar` (not a plain
global) so concurrent workflows in the same process don't collide — DBOS
runs each workflow in its own thread, and contextvars are
per-task/per-thread safe.

---

## New Files

```
agent_plane/runtime/
  __init__.py      # MODIFY — public getters (get_agent_cache, get_task_store, etc.) + init()
  _globals.py      # NEW — private store globals + init()
  prompt.py        # NEW — prompt construction from spec + history
  tool_manager.py  # NEW — MCP lifecycle, tool routing, built-in tools
  steps.py         # NEW — @step functions (call_llm, call_tool, load_agent, load_history)
  workflow.py      # REPLACE — real agent loop
  agent_cache.py   # EXISTS — two-tier agent cache (memory + disk)
```

---

## File Details

### 1. `_globals.py` + `__init__.py` — Runtime state + public getters

`_globals.py` is private — never imported outside the `runtime` package.

```python
# _globals.py  (private)
_conversation_store: ConversationStore | None = None
_task_store: TaskStore | None = None
_agent_store: AgentStore | None = None
_agent_cache: AgentCache | None = None
_tool_manager: ContextVar[ToolManager | None] = ContextVar(
    "_tool_manager", default=None,
)

def init(conversation_store, task_store, agent_store, agent_cache):
    """Called once at server startup."""
```

`runtime/__init__.py` exports typed getter functions:

```python
# __init__.py  (public API)
def get_conversation_store() -> ConversationStore: ...
def get_task_store() -> TaskStore: ...
def get_agent_store() -> AgentStore: ...
def get_agent_cache() -> AgentCache: ...
def get_tool_manager() -> ToolManager: ...   # reads from ContextVar
```

Each getter raises `RuntimeError` if the value is `None` (runtime not
initialized). `cli.py` calls `init()`; workflow code calls getters.

### 2. `prompt.py` — Prompt construction

Pure functions, no side effects. Key interfaces:

- **`build_system_message(spec, per_request_instructions, tool_schemas)`**
  → system message dict
  - Concatenates: agent instructions (AGENTS.md) + per-request
    instructions + skill metadata (name + description only, so LLM knows
    skills exist and can call `load_skill`)

- **`history_to_messages(items: list[ConversationItem])`** → litellm
  message list
  - Maps conversation items to OpenAI chat format:
    - `message(role=user)` → `{"role": "user", "content": ...}`
    - `message(role=assistant)` → `{"role": "assistant", "content": ...}`
    - `function_call` → `{"role": "assistant", "tool_calls": [...]}`
    - `function_call_output` → `{"role": "tool", "tool_call_id": ..., "content": ...}`
  - Merges consecutive assistant messages with tool_calls into one
    message (litellm expects this)

- **`build_messages(spec, history, instructions, tool_schemas)`** → full
  messages list

### 3. `tool_manager.py` — Tool lifecycle and dispatch

```python
class ToolManager:
    def __init__(self, spec: AgentSpec, work_dir: Path): ...
    def start(self) -> None:          # connect MCP servers, discover tools, register builtins
    def shutdown(self) -> None:       # close all connections
    def get_tool_schemas(self) -> list[dict]:  # OpenAI-format tool schemas
    def call_tool(self, name: str, arguments: str) -> str:  # route and execute
```

- **MCP stdio**: `asyncio.run(stdio_client(...))` inside sync methods —
  safe because DBOS step threads have no event loop
- **MCP HTTP**: `asyncio.run(streamablehttp_client(...))`
- **Built-in `load_skill`**: looks up skill by name in `spec.skills`,
  returns full `SkillSpec.content`
- **Local tools**: deferred — returns error message if called in MVP

### 4. `steps.py` — DBOS-checkpointed operations

```python
@step()
def load_agent(agent_id: str) -> dict:
    # agent_cache.load(agent_id) → LoadedAgent(spec, workdir)
    # Returns {"spec": dataclasses.asdict(spec), "workdir": str(workdir)}
    # (must be JSON-serializable for DBOS checkpointing)

@step()
def load_history(conversation_id: str) -> list[dict]:
    # conversation_store.list_items(conversation_id) → serialize items

@step()
def check_steering(conversation_id: str, after: str | None) -> list[dict]:
    # conversation_store.list_items(conversation_id, after=after)

@step()
def call_llm(messages: list[dict], model: str, tools: list[dict],
             max_tokens: int | None, reasoning_effort: str | None) -> dict:
    # litellm.completion(stream=True, ...) → iterate chunks
    # For each chunk: write_stream("output", token_event) + accumulate
    # Return full assembled response dict (checkpointed by DBOS)

@step()
def call_tool(tool_name: str, arguments: str) -> str:
    # get_tool_manager() → ToolManager (from ContextVar via getter)
    # Routes to MCP server, built-in, or local tool
```

**Serialization boundary**: All `@step` inputs and outputs must be
JSON-serializable. `AgentSpec` is passed as dict between steps;
reconstructed to dataclass in the workflow. `workdir` is passed as a
string path.

**Crash recovery**: On restart, completed steps return cached output. The
`AgentCache`'s disk tier means the bundle doesn't need to be
re-downloaded — just re-parsed from disk. MCP servers are reconnected,
but completed LLM/tool calls are skipped.

### 5. `workflow.py` — The agent loop

Every call is annotated: **[EXISTS]** means it's implemented today,
**[NEW]** means Layer 2 must create it.

```python
@workflow()                                          # [EXISTS] durability.py
def agent_execution_workflow(agent_id, conversation_id,
                             previous_response_id, instructions):
    task_id = get_workflow_id()                       # [EXISTS] durability.py

    # Phase 1: Load
    loaded = get_agent_cache().load(agent_id)        # [EXISTS] AgentCache.load() → LoadedAgent
    spec = loaded.spec                               # [EXISTS] LoadedAgent.spec: AgentSpec
    work_dir = loaded.workdir                        # [EXISTS] LoadedAgent.workdir: Path
    tool_mgr = ToolManager(spec, work_dir)           # [NEW] tool_manager.py
    set_tool_manager(tool_mgr)                       # [NEW] __init__.py (writes ContextVar)

    try:
        tool_mgr.start()                             # [NEW] tool_manager.py
        tool_schemas = tool_mgr.get_tool_schemas()   # [NEW] tool_manager.py

        items = get_conversation_store() \
            .list_items(conversation_id)              # [EXISTS] ConversationStore → PagedList
        history = items.data                         # [EXISTS] PagedList.data: list[ConversationItem]
        last_seen = history[-1].id if history else None
        output_items = []

        # Phase 2: Loop (_MAX_ITERATIONS is a runtime constant)
        for _ in range(_MAX_ITERATIONS):
            # Check steering
            new_page = get_conversation_store() \
                .list_items(conversation_id,
                            after=last_seen)         # [EXISTS] ConversationStore.list_items
            if new_page.data:
                history.extend(new_page.data)
                last_seen = new_page.data[-1].id

            # Call LLM (streams tokens to clients during execution)
            messages = build_messages(                # [NEW] prompt.py
                spec, history, instructions,
                tool_schemas,
            )
            llm_resp = call_llm(                     # [NEW] steps.py @step
                messages, spec.llm.model,
                tool_schemas,
                spec.llm.max_completion_tokens,
                spec.llm.reasoning_effort,
            )

            # If no tool calls → final response
            if not has_tool_calls(llm_resp):          # [NEW] utility
                late = get_task_store().close_inbox(      # [EXISTS] TaskStore.close_inbox
                    task_id, conversation_id,
                    last_seen,
                )
                if late:
                    history.extend(late)
                    last_seen = late[-1].id
                    continue

                # Persist and return
                persist_output(                      # [NEW] utility — appends items
                    conversation_id, task_id,        #   to conversation_store
                    llm_resp, output_items,
                )
                return build_result(                 # [NEW] utility — builds task
                    task_id, "completed",             #   result dict
                    output_items,
                )

            # Execute tool calls
            for tc in get_tool_calls(llm_resp):      # [NEW] utility
                result = call_tool(                  # [NEW] steps.py @step
                    tc.name, tc.arguments,
                )
                history.append(
                    to_tool_output(tc, result)        # [NEW] utility → ConversationItem
                )
                output_items.append(
                    to_output_item(tc, result)        # [NEW] utility → dict for output
                )

        return build_result(
            task_id, "incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
    finally:
        close_stream("output")                       # [EXISTS] durability.py
        tool_mgr.shutdown()                          # [NEW] tool_manager.py
        set_tool_manager(None)                       # [NEW] __init__.py
        drain_inbox(                                 # [NEW] utility — final inbox
            task_id, conversation_id, last_seen,     #   cleanup on exit
        )
```

**Functions that need to be created in Layer 2:**

| Function | Module | Purpose |
|----------|--------|---------|
| `ToolManager` (class) | `tool_manager.py` | MCP lifecycle, tool dispatch, built-in tools |
| `build_messages()` | `prompt.py` | Assemble system + history into litellm message list |
| `call_llm()` | `steps.py` | `@step` — streaming litellm call, checkpointed |
| `call_tool()` | `steps.py` | `@step` — route to ToolManager via ContextVar |
| `has_tool_calls()` | `workflow.py` | Check if LLM response contains tool calls |
| `get_tool_calls()` | `workflow.py` | Extract tool call list from LLM response |
| `persist_output()` | `workflow.py` | Append output items to conversation store |
| `build_result()` | `workflow.py` | Build task result dict |
| `to_tool_output()` | `workflow.py` | LLM tool call + result → `ConversationItem` |
| `to_output_item()` | `workflow.py` | LLM tool call + result → output dict |
| `drain_inbox()` | `workflow.py` | Final inbox cleanup in `finally` block |
| `init()` | `__init__.py` | Set store refs + AgentCache at startup (delegates to `_globals`) |
| `get_agent_cache()` | `__init__.py` | Return canonical `AgentCache` instance |
| `get_conversation_store()` | `__init__.py` | Return canonical `ConversationStore` instance |
| `get_task_store()` | `__init__.py` | Return canonical `TaskStore` instance |
| `get_tool_manager()` | `__init__.py` | Return current workflow's `ToolManager` from `ContextVar` |
| `set_tool_manager()` | `__init__.py` | Set/clear the per-workflow `ToolManager` `ContextVar` |

### 6. `cli.py` — Add runtime init

After constructing stores, before `uvicorn.run()`:

```python
from agent_plane.runtime import init as init_runtime
from agent_plane.runtime.agent_cache import AgentCache

agent_cache = AgentCache(
    artifact_store=artifact_store,
    cache_dir=Path(cache_dir),
)
init_runtime(
    conversation_store=conversation_store,
    task_store=task_store,
    agent_store=agent_store,
    agent_cache=agent_cache,
)
```

### 7. `pyproject.toml` — New dependencies

```
litellm>=1.40
mcp>=1.0
```

---

## Implementation Phases

### Phase A: Foundation (no external deps needed)

1. `_globals.py` — private store globals + `init()`
2. `runtime/__init__.py` — public getter functions + re-export `init`
3. `cli.py` — construct AgentCache, call `init()` at startup
4. `prompt.py` — message construction from spec + history
5. Tests for prompt.py (pure data transformation, no mocks)

### Phase B: LLM integration

1. Add `litellm` dependency
2. `steps.py` — `load_agent` (via AgentCache), `load_history`,
   `check_steering`, `call_llm`
3. Tests for steps (monkeypatch `litellm.completion`)

### Phase C: Tool integration

1. Add `mcp` dependency
2. `tool_manager.py` — MCP client lifecycle, tool routing, `load_skill`
   built-in
3. `steps.py` — add `call_tool` step
4. Tests for tool_manager (mock MCP sessions)

### Phase D: Assemble the loop

1. `workflow.py` — replace placeholder with real agent loop
2. Integration tests (monkeypatch `call_llm` to return canned responses)
3. Test steering handshake end-to-end
4. Test `max_iterations` → incomplete
5. Test error paths (LLM failure, tool failure)

### Phase E: Polish

1. Verify SSE event shapes match OpenAI Responses API format
2. Verify `finally` block runs correctly on normal and error paths

---

## Key Existing Code to Reuse

| Module | What it provides |
|--------|-----------------|
| `runtime/agent_cache.py` | `AgentCache.load(agent_id)` → `LoadedAgent(spec, workdir)` |
| `spec.load(source, dest)` | Extract + parse + validate in one call |
| `entities/conversation.py` | All item data types and `parse_item_data()` |
| `entities/agent.py` | `LoadedAgent` dataclass |
| `runtime/durability.py` | `workflow` / `step` / `write_stream` / `close_stream` |
| `stores/task_store` | `close_inbox`, `try_deliver` (steering handshake) |
| `stores/conversation_store` | `list_items`, `append` |

---

## Verification

1. **Unit tests**: `prompt.py` (pure functions), tool_manager routing
   logic
2. **Integration tests**: Full workflow with monkeypatched `call_llm`
   returning canned responses. Verify: events stream correctly, output
   persisted to conversation, steering works, inbox drained on exit
3. **Manual smoke test**: Start server, register an agent bundle with a
   real LLM config, `POST /v1/responses`, verify response comes back
   with real LLM output
4. **Tool test**: Agent with an MCP server (e.g. a trivial stdio tool),
   verify `function_call` → `function_call_output` round-trip

---

## Not Yet

- MCP connection pooling — keep long-lived MCP server connections across requests instead of
  reconnecting on every workflow execution. Would reduce per-request latency for agents with
  slow-to-start MCP servers (e.g. database tools, heavy stdio processes). Requires health checks,
  reconnection logic, and cleanup on agent eviction.
- Local tool execution — Python/TypeScript tool files bundled with the agent image. Currently
  returns an error if called. Needs sandboxing design (subprocess? container? WASM?).
- Parallel tool calls — execute multiple tool calls from a single LLM response concurrently
  instead of sequentially. Would speed up agents that request several independent tool calls
  in one turn.
- Token usage tracking — extract token counts from litellm response and populate the `usage`
  field on the response object.
- `completed_at` timestamp — populate on terminal task status.
- Cancellation propagation — when a client cancels a response, interrupt the in-flight LLM call
  or tool execution rather than waiting for the current step to finish.
- `Runtime` object — replace `_globals.py` entirely with a proper `Runtime` class that holds
  stores, AgentCache, and configuration. The getter functions in `__init__.py` would delegate to
  the active `Runtime` instance, and `_globals.py` would be deleted — all state lives on the
  `Runtime`. Would make the runtime usable outside the server (CLI-driven execution, testing,
  embedded use) without relying on module-level state set during server startup.
- Shared `ToolManager` — replace per-workflow tool managers with a centralized `ToolManager` that
  holds long-lived MCP connections and is shared across concurrent workflow executions. Would
  eliminate per-request MCP startup cost and the `ContextVar` plumbing. Requires thread-safe
  connection dispatch, per-agent connection sets, lifecycle management (reconnect on failure,
  teardown on agent eviction), and careful isolation so one workflow's tool call doesn't
  interfere with another's.
- Configurable max iterations — currently a hardcoded runtime constant. Could become a per-agent
  setting (formal field on `AgentSpec` or `LLMConfig`) or a server-level config. Not in `params`
  since params are explicitly "not interpreted by the runtime" per AGENTSPEC.md.
