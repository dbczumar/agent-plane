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
                                        ├── ReasoningChunk → SSE to client
                                        ├── ToolCallStarted → SSE to client (UX)
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

The `entrypoint` field is a **file path** relative to the bundle root
(e.g. `agent.py`, `src/my_agent/main.py`). The worker imports it via
`importlib.import_module()` after adding the bundle directory to
`sys.path`:

```python
# Inside sdk_worker.py at startup:
sys.path.insert(0, str(agent_dir))        # bundle dir on sys.path
module_name = Path(entrypoint).stem       # "agent.py" → "agent"
mod = importlib.import_module(module_name)
options = mod.create_options()
```

This means:

- **Local bundle imports work naturally.** If `agent.py` does
  `from my_tools import search_docs` and `my_tools.py` is in the
  bundle, it resolves from the bundle directory on `sys.path`.
- **Third-party packages resolve from uv's venv.** When the worker
  is launched via `uv run --requirements requirements.txt`, all
  declared deps are installed before Python starts — `import chromadb`
  works because it's already on `sys.path`.
- **Package structures work.** A bundle with `my_pkg/__init__.py`
  and `my_pkg/tools.py` imports normally.

The module must export `create_options() -> ClaudeAgentOptions`:

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

`create_options` is a fixed convention (like `main()` in Go or a
WSGI `app`). The function name is not configurable in v1 — this keeps
the spec simple and discoverable.

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

### Fields ignored with warning

Accepted without error but have no effect. At startup, the worker
checks which of these fields are set and emits a **single warning**
listing all of them:

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

Example (one warning, emitted once at startup):

```
WARNING: create_options() sets fields that agent-plane manages
and will ignore: sandbox, settings, add_dirs. These have no
effect — agent-plane controls sandboxing, settings, and
filesystem access. Remove them to silence this warning.
```

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
`entrypoint` is set. Also rejects `entrypoint` when
`executor.type` is not `claude_sdk` (entrypoint is only valid
for Claude SDK executors in v1).

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
   - `reasoning_chunk` for thinking/reasoning deltas
   - `tool_call_started` when SDK begins a tool (before execution)
   - `tool_call_observed` when SDK completes a tool (after execution)
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

## /v1/turns protocol additions

This design requires two new SSE event types in the `/v1/turns`
protocol. These must be added to `RemoteExecutor`, the executor
event types in `base.py`, and the workflow's event handling.

### `tool_call_started`

```json
{"type": "tool_call_started", "call_id": "c1", "name": "Bash", "arguments": {"command": "pip install chromadb"}}
```

Emitted when the SDK **begins** executing a native tool (Bash,
Read, Edit, etc.) — before the tool runs. `tool_call_observed`
fires later with the result. Without this, the client sees nothing
while a long-running tool executes (e.g. a 30-second pip install).

The workflow forwards this to the client's SSE stream for display
but does not persist it — `tool_call_observed` is the durable
record. This is purely a UX event.

### `reasoning_chunk`

```json
{"type": "reasoning_chunk", "delta": "Let me think about...", "event_type": "reasoning_text"}
```

Already defined in the protocol but not yet implemented in the
worker-to-RemoteExecutor path. The SDK streams thinking blocks
when `thinking` is configured. The worker maps SDK thinking
events → `reasoning_chunk` SSE events with `event_type` values:
`reasoning_started`, `reasoning_text`, `reasoning_summary`.

The workflow forwards these to the client's SSE stream. Not
persisted (thinking content is ephemeral).

---

## Open details

### Worker health monitoring

If the worker dies between turns (OOM, segfault), agent-plane
detects it reactively via 404 on the next `/v1/turns` POST. There
is no proactive heartbeat. This is consistent with `RemoteExecutor`
(which also has no health monitoring) but means the failure surfaces
as a turn-level error, not an immediate alert.

For v1 this is acceptable. A health check endpoint
(`GET /v1/health`) on the worker could be added later for
monitoring.

### `storage_dir` and workspace access

The worker subprocess needs access to `storage_dir` for the SDK's
session state (`.claude/` transcripts, `--continue` behavior).
Agent-plane passes it as a CLI argument or environment variable
when spawning the worker. The worker sets `cwd` to the storage_dir
workspace subdirectory — same as today's in-process
`ClaudeAgentsExecutor`.

### Entrypoint error handling

If the entrypoint module has an import error or `create_options()`
raises, the worker crashes before the HTTP server starts. Agent-plane
detects this by:
1. Monitoring the subprocess exit code (non-zero = crash)
2. Timeout on the health/port readiness check

The error is surfaced as a clear message in the task's error field:
`"Entrypoint failed: {exception message}"` — not a generic timeout.

### Relationship to existing `ClaudeAgentsExecutor`

Without `entrypoint`: existing in-process behavior continues
unchanged. `ClaudeAgentsExecutor` constructs `ClaudeAgentOptions`
from YAML fields and runs the SDK client in the server process.

With `entrypoint`: the worker subprocess model kicks in. Two code
paths for the same `executor.type: claude_sdk`, selected by whether
`entrypoint` is set. The in-process path remains the default for
YAML-only agents — no migration needed.

### Compaction

The worker-backed executor returns `max_context_tokens() -> None`
(same as `RemoteExecutor`). The SDK manages its own context window
internally. Agent-plane skips both proactive and reactive compaction
for entrypoint-based agents. This is correct — the SDK's built-in
compaction is better than agent-plane's for Claude models.

### Tool result round-trip

When the worker yields `tool_call_requested` (for `Task` or client
tools), agent-plane executes the tool and sends the result back in
the next `/v1/turns` POST as a `role: "tool"` message. The worker
must translate this back to a tool result for the SDK client.

The SDK client's `query()` method accepts tool results as part of
the message stream. The worker maps `role: "tool"` messages from
the POST body to SDK tool result format before feeding them to
`client.query()`. This is the same translation that
`RemoteExecutor`'s recovery handshake does — the `/v1/turns`
protocol already defines the `role: "tool"` message format.

### API key and connection config

The Claude Agent SDK reads `ANTHROPIC_API_KEY` from the environment
— there is no programmatic way to pass credentials through
`ClaudeAgentOptions`. This is different from the `DefaultExecutor`
(litellm), which supports `llm.connection.api_key` in YAML with
`${ENV_VAR}` expansion.

For the worker subprocess:

- The worker inherits the server process's environment. If
  `ANTHROPIC_API_KEY` is set where agent-plane runs, the worker
  gets it automatically via `subprocess.Popen(env=os.environ)`.
- `llm.connection` remains **forbidden** for `executor.type:
  claude_sdk` (the validator already enforces this) because the
  SDK doesn't accept connection params programmatically.
- The user must ensure `ANTHROPIC_API_KEY` is set in the
  environment where agent-plane runs. This is the only auth
  setup needed.

If the SDK adds programmatic API key support in the future, we
can lift the `llm.connection` restriction and pass it through.

### Model override delivery

When YAML specifies `llm.model`, it overrides whatever model the
user set in `create_options()`. Agent-plane passes the override
to the worker as a CLI argument:

```bash
python -m agent_plane.runtime.sdk_worker {agent_dir} \
  --port {port} \
  --model claude-sonnet-4-20250514
```

The worker applies the override to the options before constructing
the SDK client:

```python
options = mod.create_options()
if args.model:
    options.model = args.model
```

If `llm.model` is not set in YAML, the worker uses whatever model
`create_options()` returns.

### Worker logging

The worker subprocess's stderr is captured by agent-plane and
routed to the server's logger with a `[worker:{agent_name}]`
prefix. This surfaces:

- Entrypoint import errors and `create_options()` exceptions
- SDK warnings (rate limits, retries, deprecations)
- Tool execution output (e.g. Bash stderr)
- Worker HTTP server lifecycle events

Without this, debugging entrypoint-based agents would require
manual subprocess inspection.

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
   startup. Expose `GET /v1/agent-info` endpoint that returns
   a JSON list of sub-agent definitions (name, system_prompt,
   allowed_tools, model). Called once by agent-plane before the
   first turn.

6. **`runtime/executors/claude.py`** — After spawning worker,
   call `GET /v1/agent-info` to discover sub-agents. Build
   in-memory `AgentSpec` for each. Register on parent spec's
   `sub_agents` list and add names to `tools.agents` so
   `SpawnTool` can route to them.

### Phase 3: Dependency installation

7. **`runtime/executors/claude.py`** — Detect deps at worker
   spawn time:
   - If `requirements.txt` in bundle: `uv run --requirements ...`
   - If entrypoint has PEP 723 metadata: parse → `uv run --with ...`
   - If neither: plain `python` (no uv)
   - Fallback: if `uv` not on PATH, start without dep resolution

### Phase 4: Per-agent worker pooling

8. Reuse worker subprocesses across tasks for the same agent.
   One worker per agent, handles multiple conversations via
   `conversation_id` routing in `/v1/turns` requests. The SDK
   client's `session_id` parameter maps to `conversation_id`.
   Avoids subprocess + dep resolution cost per task.

---

## Deployment guide

Step-by-step: how to take existing Claude Agent SDK code and deploy
it to agent-plane.

### Step 1: Set up your environment

Set `ANTHROPIC_API_KEY` in the environment where agent-plane runs.
This is the **only auth setup needed** — the worker subprocess
inherits it automatically.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

There is no `llm.connection` or `api_key` field in the YAML for
`claude_sdk` executors. The SDK reads the key from the environment
directly. This is different from `executor.type: llm` (litellm),
which supports `llm.connection.api_key: ${ENV_VAR}` in YAML.

### Step 2: Wrap your options in `create_options()`

Take your existing `ClaudeAgentOptions` construction and wrap it
in a function named `create_options`. Delete all client lifecycle
code (connect, query, stream, disconnect).

**Before** (your existing code):

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from my_tools import search_docs

options = ClaudeAgentOptions(
    tools=["Bash", "Read", "Edit"],
    mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
    system_prompt="You are a senior engineer.",
    max_turns=30,
)

# Lifecycle code — DELETE THIS
client = ClaudeSDKClient(options)
await client.connect()
await client.query("Fix the auth bug")
async for msg in client.receive_response():
    print(msg)
```

**After** (your entrypoint file):

```python
# agent.py
from claude_agent_sdk import ClaudeAgentOptions
from my_tools import search_docs

def create_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=["Bash", "Read", "Edit"],
        mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
        system_prompt="You are a senior engineer.",
        max_turns=30,
    )
```

**What you keep:** `@tool` handlers, MCP servers, hooks,
`can_use_tool`, sub-agent definitions, all imports.

**What you delete:** `ClaudeSDKClient` creation, `connect()`,
`query()`, stream loop, `disconnect()`. Agent-plane manages all
of this.

### Step 3: Create `config.yaml`

```yaml
spec_version: 1
name: my-coding-agent

executor:
  type: claude_sdk
  entrypoint: agent.py
  timeout: 600
  max_iterations: 20
```

Optional model override (takes precedence over `create_options()`):

```yaml
llm:
  model: claude-sonnet-4-20250514
```

**Fields you cannot set** when `entrypoint` is present:
- `tools.builtins` — the entrypoint defines tools
- `instructions` — the entrypoint sets `system_prompt`

The validator rejects these to prevent conflicting configuration.

### Step 4: Add dependencies (optional)

If your code imports third-party packages, declare them so the
worker can install them at startup:

**Option A — `requirements.txt`:**

```
chromadb>=0.4
langchain>=0.2
```

**Option B — PEP 723 inline metadata** in the entrypoint:

```python
# /// script
# dependencies = ["chromadb>=0.4", "langchain>=0.2"]
# ///

from claude_agent_sdk import ClaudeAgentOptions
...
```

If neither is present, the worker starts without dependency
resolution (uses whatever is already installed).

### Step 5: Bundle and upload

```bash
# Create the bundle
tar -czf my-agent.tar.gz -C my-agent/ .

# Upload to agent-plane
curl -X POST http://localhost:8080/api/agents \
  -F "bundle=@my-agent.tar.gz"
```

Bundle structure:

```
my-agent/
├── config.yaml           # required
├── agent.py              # entrypoint (referenced in config.yaml)
├── my_tools.py           # local modules imported by agent.py
├── my_pkg/               # package structures work too
│   ├── __init__.py
│   └── helpers.py
└── requirements.txt      # optional third-party deps
```

### Step 6: Use the agent

```bash
# Start a conversation
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "my-coding-agent", "input": "Fix the auth bug"}'
```

The agent runs with full agent-plane guarantees: durable execution,
conversation persistence, SSE streaming, steering, and sub-agent
orchestration.

### What you get for free

| Capability | How |
|-----------|-----|
| Durability | DBOS workflow survives crashes |
| Conversation persistence | Conversation store tracks all messages |
| SSE streaming | Text, reasoning, tool calls streamed to client |
| Steering | Client can send mid-turn messages |
| Sub-agent orchestration | `agents` in options → durable child workflows |
| Client-side tools | Tunneled through agent-plane's parking mechanism |
| Crash recovery | Worker crash → task failed → DBOS re-invoke |
| Dependency isolation | Each agent gets its own `uv`-managed venv |

### What changed from your existing code

| You wrote | What happens on agent-plane |
|-----------|---------------------------|
| `ClaudeSDKClient(options)` | Agent-plane creates the client in the worker |
| `client.connect()` | Worker starts SDK client at turn start |
| `client.query("...")` | `/v1/responses` POST triggers the turn |
| `async for msg in ...` | SSE stream to the API client |
| `client.disconnect()` | Worker shuts down on task end |
| `ANTHROPIC_API_KEY` in env | Same — worker inherits it from server |
| `options.model` | YAML `llm.model` overrides if set |
