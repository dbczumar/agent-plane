# Programmatic Claude Agent SDK Support

## Problem

Developers who have written code with the Claude Agent SDK want to
deploy it on agent-plane without rewriting it. Today,
`executor.type: claude_sdk` configures the SDK via YAML fields
(`tools.builtins`, `instructions`, `llm.model`). This works for
agents that can be fully described declaratively, but not for agents
with custom `@tool` handlers, custom MCP servers, hooks, sub-agent
definitions, or programmatic `ClaudeAgentOptions` construction.

### What SDK users write today

```python
from claude_agent_sdk import (
    ClaudeAgentOptions, AgentDefinition,
    tool, create_sdk_mcp_server, HookMatcher,
)

@tool("search_docs", "Search documentation", {"query": str})
async def search_docs(args):
    results = my_vector_db.search(args["query"])
    return {"content": [{"type": "text", "text": str(results)}]}

options = ClaudeAgentOptions(
    tools=["Bash", "Read", "Edit"],
    mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
    system_prompt="You are a coding assistant with access to docs.",
    agents={"reviewer": AgentDefinition(
        system_prompt="You review code.",
        allowed_tools=["Read", "Grep"],
    )},
    hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[my_hook])]},
    max_turns=50,
)

# They also write client lifecycle code:
client = ClaudeSDKClient(options)
await client.connect()
await client.query("Fix the auth bug")
async for msg in client.receive_response():
    ...
```

### Goal

The user wraps their options construction in a function and deploys.
No rewriting. No restructuring. Agent-plane handles the rest.

---

## Architecture

### Worker subprocess model

The user's entrypoint code (including `@tool` handlers, hooks, and
`can_use_tool` callbacks) runs in a **dedicated worker subprocess**,
not in the agent-plane server process. The worker speaks the existing
`/v1/turns` SSE protocol over localhost — agent-plane connects via
`RemoteExecutor`.

```
ap server process
├── FastAPI / uvicorn
├── DBOS workflows
├── RemoteExecutor → http://localhost:{port}/v1/turns
│
└── SDK Worker subprocess (per-agent)
    ├── sys.path includes agent's installed deps
    ├── Imports agent.py entrypoint
    ├── ClaudeSDKClient (SDK → CLI subprocess)
    ├── @tool handlers run HERE (user code)
    ├── hooks run HERE (user code)
    ├── can_use_tool runs HERE (user code)
    └── HTTP server: POST /v1/turns → SSE events
```

### Why a subprocess

1. **Dependency isolation**: Each agent installs its own packages
   (from `requirements.txt` or PEP 723). A subprocess has its own
   `sys.path` — no version conflicts between agents, no module
   leaking via `sys.modules`.

2. **Crash isolation**: A buggy `@tool` handler or hook that crashes
   kills the worker, not the server. Agent-plane detects the
   failure and marks the task as failed. DBOS recovery can restart.

3. **No new protocol**: The worker speaks `/v1/turns` — the same
   SSE protocol `RemoteExecutor` already supports. No new IPC
   design, no new executor type.

4. **Generalizes**: Any agent framework can be deployed this way.
   Wrap it in a `/v1/turns` HTTP endpoint, agent-plane connects.

### How it connects to durability

Agent-plane's workflow owns the outer loop. The worker handles the
SDK turn. Events flow:

```
Worker → SSE events → RemoteExecutor → workflow event loop
                                        ├── TextChunk → SSE to client
                                        ├── ToolCallObserved → persist to conv_store
                                        ├── ToolCallRequested → workflow executes
                                        │   ├── spawn_sub_agents → durable sub-agent
                                        │   ├── client tools → park for client
                                        │   └── other tools → ToolManager
                                        └── TurnComplete → persist + steering check
```

SDK-internal tools (Bash, Read, Edit — in `allowed_tools`) execute
inside the worker. Everything else comes back to agent-plane as
`tool_call_requested` events for durable execution.

---

## Agent spec

```yaml
spec_version: 1
name: my-coding-agent

executor:
  type: claude_sdk
  entrypoint: agent.py    # Python module in the bundle
  timeout: 600
  max_iterations: 20

llm:
  model: claude-sonnet-4-20250514   # optional — overrides entrypoint
```

When `entrypoint` is set, agent-plane:
1. Installs deps (if `requirements.txt` or PEP 723 metadata present)
2. Spawns the worker subprocess
3. Creates `RemoteExecutor(endpoint=f"http://localhost:{port}/v1/turns")`
4. Runs the agent loop as normal

---

## Entrypoint contract

The module exports `create_options() -> ClaudeAgentOptions`:

```python
# agent.py
from claude_agent_sdk import (
    ClaudeAgentOptions, AgentDefinition,
    tool, create_sdk_mcp_server, HookMatcher,
)
from my_tools import search_docs, validate_code

def create_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=["Bash", "Read", "Edit", "Write"],
        mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
        system_prompt="You are a senior engineer.",
        agents={
            "reviewer": AgentDefinition(
                system_prompt="You review code for bugs.",
                allowed_tools=["Read", "Grep"],
            ),
        },
        hooks={
            "PreToolUse": [HookMatcher(matcher="Bash", hooks=[validate_code])],
        },
        max_turns=30,
        thinking={"type": "adaptive"},
    )
```

**What the user changes from their existing code:** wrap the options
construction in `create_options()`. Delete client lifecycle code
(connect, query, stream loop, disconnect). That's it.

---

## How agent-plane handles each field

### Fields used as-is

Passed directly to `ClaudeSDKClient` in the worker:

| Field | Behavior |
|-------|----------|
| `tools` / `allowed_tools` | SDK tool configuration |
| `system_prompt` | System instructions |
| `mcp_servers` | User's MCP servers |
| `hooks` | PreToolUse, PostToolUse, etc. — run in worker |
| `can_use_tool` | Permission callback — runs in worker |
| `model` | Model selection (YAML `llm.model` overrides if set) |
| `thinking` | Thinking/reasoning configuration |
| `max_turns` | Turn limit |
| `max_budget_usd` | Cost budget |
| `output_format` | Structured output config |
| `betas` | SDK beta features |
| `plugins` | SDK plugins |

### Fields agent-plane overrides

Set by the worker regardless of what the user provides:

| Field | Worker sets | Reason |
|-------|------------|--------|
| `cwd` | Worker's working directory | Worker manages filesystem |
| `env` | `{"CLAUDECODE": ""}` + overrides | Prevents nested session errors |
| `permission_mode` | `bypassPermissions` | Server-side execution |
| `disallowed_tools` | `["Task"]` (merged) | Agent-plane manages sub-agents |
| `extra_args` | `{"no-session-persistence": None}` | Agent-plane manages state |
| `include_partial_messages` | `True` | Needed for streaming |

### Fields silently ignored

Accepted without error but have no effect:

| Field | Why ignored |
|-------|-------------|
| `continue_conversation` | Agent-plane manages sessions via conversation_id |
| `resume` | Agent-plane manages continuity via storage_dir |
| `fork_session` | Not applicable |
| `setting_sources` | Agent-plane manages settings |
| `settings` | Agent-plane manages settings |
| `sandbox` | Agent-plane controls sandboxing |
| `add_dirs` | Agent-plane controls filesystem |
| `cli_path` | Worker finds claude |
| `user` | Agent-plane manages identity |
| `stderr` | Worker captures stderr |

Worker logs a one-time info message listing ignored fields.

---

## Sub-agent auto-translation

When `options.agents` is present, the worker extracts each
`AgentDefinition` and reports them to agent-plane at startup.
Agent-plane builds in-memory `AgentSpec` objects and registers
them on the parent's spec tree.

### Mapping

| `AgentDefinition` field | `AgentSpec` field |
|---|---|
| `system_prompt` | `instructions` |
| `allowed_tools` | `tools.builtins` (with `claude:` prefix) |
| `model` | `llm.model` |
| (other fields) | Passed to sub-agent's `ClaudeAgentOptions` |

### Runtime flow

1. Parent SDK calls `Task("research X")` via the CLI subprocess
2. Worker yields `tool_call_requested(name="Task", ...)`
3. Agent-plane's workflow receives it, routes to `SpawnTool`
4. `SpawnTool` creates a child `agent_execution_workflow`
5. Child gets its own worker subprocess, conversation store,
   durability — full agent-plane guarantees
6. Child completes → result returned to parent via next
   `/v1/turns` POST
7. Worker feeds result back to the SDK as a tool result

### What users get for free

- Sub-agent tool calls persisted to conversation store
- Sub-agent crashes recovered via DBOS re-invoke
- Sub-agent output visible on parent's SSE stream
- Client-side tools tunnel from sub-agents to root
- Zero changes to their `AgentDefinition` code

### Nested sub-agents

If a sub-agent's `AgentDefinition` itself has `agents={...}`, the
translation recurses. Each level gets its own durable workflow and
worker subprocess.

---

## Client-side tools

Client-side tools from the API request are reported to the worker
at turn start. The worker registers them but does NOT execute them.
When the SDK calls one, the worker yields `tool_call_requested`.
Agent-plane parks it (`action_required`), the client PATCHes the
result, agent-plane sends it back to the worker in the next
`/v1/turns` POST.

Same flow as `RemoteExecutor` — the worker is just another remote
service from agent-plane's perspective.

---

## YAML field interaction

When `executor.entrypoint` is set:

| YAML field | Behavior |
|-----------|----------|
| `llm.model` | Overrides `options.model` if set |
| `tools.builtins` | **Invalid** — entrypoint defines tools |
| `instructions` | **Invalid** — entrypoint sets system_prompt |
| `compaction` | Valid — workflow concern |
| `executor.timeout` | Valid — workflow concern |
| `executor.max_iterations` | Valid — workflow concern |

Validator rejects `tools.builtins` and `instructions` when
`entrypoint` is set.

---

## Dependencies

Dependencies are resolved at **worker startup** using `uv run`, not
at deploy time. This is the same pattern used for local Python tools
with PEP 723 inline metadata.

### How it works

When the worker subprocess starts, agent-plane wraps the launch
command with `uv run`:

```bash
# With requirements.txt:
uv run --requirements {agent_dir}/requirements.txt \
  -- python -m agent_plane.runtime.sdk_worker {agent_dir} --port {port}

# With PEP 723 inline metadata:
uv run --with my-vector-db --with langchain \
  -- python -m agent_plane.runtime.sdk_worker {agent_dir} --port {port}
```

`uv` creates an ephemeral virtual environment, installs deps, and
runs the worker inside it. The venv is cached by `uv` keyed on the
exact dependency set — second startup is near-instant.

### Where deps go

Into `uv`'s global cache (`~/.cache/uv/` by default). NOT into
the workspace or storage_dir. The cache is shared across all
conversations for the same agent (same deps = same cached venv).

No explicit per-agent directory to manage, no cleanup needed — `uv`
handles cache lifecycle.

### Supported formats

| Format | Detection | Launch command |
|--------|-----------|----------------|
| `requirements.txt` in bundle | File exists | `uv run --requirements requirements.txt -- ...` |
| PEP 723 inline metadata in entrypoint | `# /// script` block | `uv run --with dep1 --with dep2 -- ...` |
| No deps declared | Neither present | `python -m agent_plane.runtime.sdk_worker ...` (no uv) |

### Why this works

- **No deploy-time step**: Deps resolve on first worker startup
- **Cached**: `uv` caches aggressively — subsequent starts are fast
- **Isolated**: Each worker runs in its own `uv`-managed venv —
  no version conflicts between agents
- **Consistent**: Same `uv run` pattern used for local Python tools
- **Fallback**: If `uv` is not available, the worker starts without
  dep resolution (pre-installed packages only)

---

## Worker subprocess lifecycle

### Startup

1. Agent-plane finds a free port
2. Detects deps: `requirements.txt` in bundle or PEP 723 in entrypoint
3. Spawns worker:
   - With deps: `uv run --requirements ... -- python -m agent_plane.runtime.sdk_worker {agent_dir} --port {port}`
   - Without deps: `python -m agent_plane.runtime.sdk_worker {agent_dir} --port {port}`
4. Worker imports `agent.py`, calls `create_options()`
5. Extracts `options.agents` → reports sub-agent defs to agent-plane
6. Starts HTTP server on `localhost:{port}`
7. Agent-plane creates `RemoteExecutor(endpoint=...)`

### Per-turn

1. Agent-plane POSTs to `/v1/turns` with `conversation_id` + messages
2. Worker sends prompt to SDK client
3. Worker streams SSE events back:
   - `text_chunk` for streaming text
   - `tool_call_observed` for SDK-internal tools (Bash, Read, etc.)
   - `tool_call_requested` for external tools (Task, client tools)
   - `turn_complete` when done
4. Agent-plane persists events, handles tool requests, sends
   results back in next POST

### Session recovery

If the worker crashes or restarts, it loses the SDK client state
(in-memory conversation context). Agent-plane detects via 404 on
the next `/v1/turns` POST and sends `history` in the retry — same
as `RemoteExecutor`'s recovery handshake.

### Shutdown

Agent-plane sends SIGTERM to the worker on task end. Worker
disconnects SDK client and exits.

---

## Security

All user code runs in the worker subprocess:

| Component | Where it runs | Isolation |
|-----------|--------------|-----------|
| `@tool` handlers | Worker subprocess | Crash-isolated from server |
| `hooks` callbacks | Worker subprocess | Crash-isolated from server |
| `can_use_tool` | Worker subprocess | Crash-isolated from server |
| Imported dependencies | Worker subprocess | No version conflicts |
| SDK CLI (Bash, Read, etc.) | CLI child of worker | Separate process |

Worker crash → agent-plane marks task failed → DBOS can re-invoke.
Server process is unaffected.

The operator trusts the code they deploy — same model as any
container-based deployment. The subprocess boundary prevents
accidental damage (crashes, OOM), not malicious code.

---

## Implementation plan

### Phase 1: Worker subprocess + `/v1/turns` adapter

1. **`runtime/sdk_worker.py`** (new) — Standalone script:
   - Imports entrypoint module, calls `create_options()`
   - Creates `ClaudeSDKClient` with the options
   - Serves `POST /v1/turns` → runs SDK turn → SSE events
   - Handles session recovery (404 → rebuild from history)

2. **`runtime/executors/claude.py`** — When spec has `entrypoint`:
   - Spawn worker subprocess instead of in-process SDK client
   - Create `RemoteExecutor` pointing at worker's localhost port
   - Manage worker lifecycle (start, health check, shutdown)

3. **`spec/types.py`** — Add `entrypoint: str | None = None` to
   `ExecutorSpec`.

4. **`spec/validator.py`** — Reject `tools.builtins` and
   `instructions` when `entrypoint` is set.

### Phase 2: Sub-agent auto-translation

5. **`runtime/sdk_worker.py`** — Extract `options.agents` at
   startup, report to agent-plane via a startup handshake
   (`GET /v1/agent-info` or similar).

6. **`runtime/executors/claude.py`** — Build `AgentSpec` from
   reported `AgentDefinition` data, register on parent spec tree.

### Phase 3: Dependency installation

7. **`runtime/agent_cache.py`** — At bundle extraction:
   - If `requirements.txt` exists: `pip install -r --target {dir}`
   - If entrypoint has PEP 723 metadata: parse and install
   - Pass `{dir}` to worker via env or CLI arg

### Phase 4: Per-agent worker pooling

8. Reuse worker subprocesses across tasks for the same agent.
   One worker per agent, handles multiple conversations via
   `conversation_id` routing. Avoids subprocess startup cost
   per task.

---

## Example deployment

### Bundle structure

```
my-agent/
├── config.yaml
├── agent.py              # entrypoint
├── my_tools.py           # custom tool implementations
└── requirements.txt      # optional (Phase 3)
```

### config.yaml

```yaml
spec_version: 1
name: my-coding-agent

executor:
  type: claude_sdk
  entrypoint: agent.py
  timeout: 600
```

### agent.py

```python
from claude_agent_sdk import (
    ClaudeAgentOptions, AgentDefinition,
    tool, create_sdk_mcp_server, HookMatcher,
)
from my_tools import search_docs, validate_code

def create_options() -> ClaudeAgentOptions:
    doc_server = create_sdk_mcp_server("docs", tools=[search_docs])
    return ClaudeAgentOptions(
        tools=["Bash", "Read", "Edit", "Write"],
        mcp_servers={"docs": doc_server},
        system_prompt="You are a senior engineer.",
        agents={
            "reviewer": AgentDefinition(
                system_prompt="You review code for bugs.",
                allowed_tools=["Read", "Grep"],
            ),
        },
        hooks={
            "PreToolUse": [HookMatcher(matcher="Bash", hooks=[validate_code])],
        },
        max_turns=30,
        thinking={"type": "adaptive"},
    )
```

### What the user adapted

From their existing code, they:
1. Wrapped options construction in `create_options()` — no rewrite
2. Deleted `ClaudeSDKClient` creation — agent-plane manages it
3. Deleted `connect()` / `query()` / stream loop / `disconnect()`
4. Kept ALL `@tool` handlers, MCP servers, hooks, sub-agents, deps

Zero rewriting of agent logic. Only lifecycle code is removed.
