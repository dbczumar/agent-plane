# SDK Entrypoint: Deploying Claude Agent SDK Code to Agent-Plane

## Problem

Developers who have written code with the Claude Agent SDK want to
deploy it on agent-plane without rewriting it as a declarative YAML
spec. Today, `executor.type: claude_sdk` configures the SDK via YAML
fields (`tools.builtins`, `instructions`, `llm.model`). This works
for agents that can be fully described declaratively, but not for
agents with custom `@tool` handlers, custom MCP servers, runtime
logic, or programmatic `ClaudeAgentOptions` construction.

### What SDK users write today

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, tool

@tool("search_docs", "Search documentation", {...})
async def search_docs(args):
    results = my_vector_db.search(args["query"])
    return {"content": [{"type": "text", "text": str(results)}]}

options = ClaudeAgentOptions(
    tools=["Bash", "Read", "Edit"],
    mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
    system_prompt="You are a coding assistant with access to docs.",
    permission_mode="bypassPermissions",
)

client = ClaudeSDKClient(options)
await client.connect()
await client.query("Fix the auth bug")
async for msg in client.receive_response():
    ...
```

### What we want

```yaml
executor:
  type: claude_sdk
  entrypoint: agent.py
```

Agent-plane loads the module, gets the options, manages the rest.
The developer's `@tool` handlers and MCP servers run as-is.

---

## Proposal

The agent bundle includes a Python file (e.g. `agent.py`). The spec
declares `executor.entrypoint: agent.py`. Agent-plane imports the
module and calls a known export to get `ClaudeAgentOptions`. The
`ClaudeAgentsExecutor` uses these options instead of constructing
them from YAML fields.

---

## Open Questions

### Q1: What does the entrypoint module export?

**Option A: `create_options() -> ClaudeAgentOptions`**

Zero-arg function. Agent-plane calls it once at executor construction
time. Simple, but the function can't access runtime context
(conversation_id, storage_dir, file stores).

```python
def create_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=["Bash", "Read"],
        mcp_servers={"docs": my_mcp_server},
        system_prompt="...",
    )
```

**Option B: `create_options(context) -> ClaudeAgentOptions`**

Receives an agent-plane context object with runtime info. The
function can use `context.storage_dir`, `context.file_store`, etc.
More powerful, but couples the entrypoint to agent-plane's API.

```python
def create_options(context: AgentPlaneContext) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=["Bash", "Read"],
        system_prompt=f"Working in {context.storage_dir}",
    )
```

**Option C: `create_options` is called per-turn, not once**

Called before each `run_turn()`. Can vary options per conversation.
More flexible, but heavier.

**Decision:** ?

---

### Q2: How do user `@tool` handlers interact with agent-plane's tool routing?

Today `ClaudeAgentsExecutor` registers client-side tools as MCP
handlers backed by `context.call_tool` (which parks for the client).
If the user's entrypoint also defines `@tool` handlers:

- **Conflict**: User defines `@tool("Read", ...)` AND the YAML spec
  has `claude:Read`. Which runs?
- **Merge**: User's MCP server tools + agent-plane's client-tool MCP
  server both get registered. The SDK sees both.
- **Replace**: When entrypoint is set, agent-plane does NOT register
  its own MCP server. User owns all tool routing.

**Decision:** ?

---

### Q3: Dependency management

The entrypoint module will import packages (`my_vector_db`,
`langchain`, custom libraries). Options:

- **Pre-installed**: Packages must be on the server already. Simple
  but limits portability.
- **requirements.txt in bundle**: Agent-plane installs deps at deploy
  time (like `pip install -r requirements.txt`). Adds latency on
  first deploy but self-contained.
- **PEP 723 inline metadata**: The entrypoint file declares deps
  inline (same as local Python tools). Consistent with existing
  pattern.
- **Docker**: The bundle includes a Dockerfile or specifies an image.
  Full isolation but heavy.

**Decision:** ?

---

### Q4: Security — arbitrary code in the server process

The entrypoint runs in the main server process. A crash, OOM, or
infinite loop kills the server. Existing precedent:

| Component | Runs in | Isolation |
|-----------|---------|-----------|
| Local Python tools | Subprocess | Crash-isolated, optional srt sandbox |
| code_sandbox | Subprocess | Crash-isolated, optional srt sandbox |
| MCP servers | Separate process | Full process isolation |
| **Entrypoint (proposed)** | **Main process** | **None** |

Options:

- **Accept for v1**: Same trust model as importing any Python
  library. The operator trusts the code they deploy.
- **Subprocess**: Run the entrypoint in a subprocess. But the SDK
  client is stateful and long-lived — subprocess isolation is
  complex for persistent connections.
- **Worker process**: The entrypoint runs in a dedicated worker
  process that agent-plane communicates with via IPC. Crash-isolated
  but complex.

**Decision:** ?

---

### Q5: ClaudeSDKClient lifecycle

Today `ClaudeAgentsExecutor` manages the SDK client — creates it
per-conversation, reuses across turns, disconnects on task end.
Options when an entrypoint is set:

- **Agent-plane manages client**: Entrypoint returns options.
  Agent-plane creates, connects, queries, and disconnects the client.
  User code is only the options factory + `@tool` handlers.
- **User manages client**: Entrypoint returns a connected client (or
  a factory that produces clients). More control for advanced users,
  but agent-plane can't guarantee lifecycle (disconnect on task end,
  crash recovery).
- **Hybrid**: Agent-plane manages the client but calls user hooks
  (`on_connect`, `on_disconnect`, `on_turn_start`).

**Decision:** ?

---

### Q6: How does the entrypoint interact with YAML spec fields?

When `entrypoint: agent.py` is set alongside `executor.type: claude_sdk`:

| YAML field | With entrypoint | Rationale |
|-----------|----------------|-----------|
| `llm.model` | ? | Entrypoint may set model in options |
| `tools.builtins` | ? | Entrypoint defines its own tools |
| `instructions` | ? | Entrypoint sets system_prompt in options |
| `compaction` | ? | Workflow concern, independent of executor |
| `executor.timeout` | Valid | Workflow concern |
| `executor.max_iterations` | Valid | Workflow concern |

Options:

- **Entrypoint replaces all SDK-specific fields**: `llm.model`,
  `tools.builtins`, and `instructions` are invalid when entrypoint
  is set. Validator rejects them. Clean separation.
- **Entrypoint overrides**: YAML fields are defaults. Entrypoint can
  override any of them. Flexible but confusing (which one wins?).
- **Merge**: YAML `tools.builtins` merged with entrypoint's tools.
  Complex, error-prone.

**Decision:** ?

---

### Q7: What about non-Claude-SDK executors?

The entrypoint concept could generalize beyond `claude_sdk`:

- `executor.type: remote` + entrypoint: start a local HTTP server
  from the entrypoint code instead of connecting to an external one.
- `executor.type: llm` + entrypoint: custom pre/post-processing
  around the default LLM call.

Should the design support this from the start, or scope to
`claude_sdk` only for v1?

**Decision:** ?

---

## Strawman Design (for discussion)

Based on the simplest viable option for each question:

```yaml
# config.yaml
spec_version: 1
name: my-coding-agent

executor:
  type: claude_sdk
  entrypoint: agent.py   # relative to bundle root
  timeout: 600

llm:
  model: claude-sonnet-4-20250514   # optional override
```

```python
# agent.py — the entrypoint module
from claude_agent_sdk import ClaudeAgentOptions, tool, create_sdk_mcp_server

@tool("search_docs", "Search the documentation", {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
})
async def search_docs(args):
    # User's custom tool logic
    return {"content": [{"type": "text", "text": "result..."}]}

def create_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=["Bash", "Read", "Edit"],
        mcp_servers={
            "docs": create_sdk_mcp_server("docs", tools=[search_docs]),
        },
        system_prompt="You are a coding assistant with docs access.",
        permission_mode="bypassPermissions",
    )
```

Agent-plane:
1. Extracts the bundle, imports `agent.py`
2. Calls `agent.create_options()` to get `ClaudeAgentOptions`
3. If `llm.model` is set in YAML, overrides `options.model`
4. Creates `ClaudeSDKClient(options)`, manages lifecycle
5. Client-side tools from the API request are merged as an
   additional MCP server (agent-plane's existing mechanism)
6. `@tool` handlers in the entrypoint run in-process

What the user adapts:
- Wrap their options construction in `create_options()`
- Remove their `ClaudeSDKClient` lifecycle code (agent-plane owns it)
- Remove their event streaming loop (agent-plane streams via SSE)
- Keep all `@tool` handlers and MCP servers as-is
