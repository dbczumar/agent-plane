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
    ClaudeAgentOptions, tool, create_sdk_mcp_server,
    AgentDefinition, HookMatcher,
)

@tool("search_docs", "Search documentation", {"query": str})
async def search_docs(args):
    results = my_vector_db.search(args["query"])
    return {"content": [{"type": "text", "text": str(results)}]}

options = ClaudeAgentOptions(
    tools=["Bash", "Read", "Edit"],
    mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
    system_prompt="You are a coding assistant with access to docs.",
    agents={"researcher": AgentDefinition(...)},
    hooks={"PreToolUse": [HookMatcher(...)]},
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

## Design

### Agent spec

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

### Entrypoint contract

The module exports `create_options() -> ClaudeAgentOptions`:

```python
# agent.py
from claude_agent_sdk import ClaudeAgentOptions, tool, create_sdk_mcp_server

@tool("search_docs", "Search documentation", {"query": str})
async def search_docs(args):
    results = my_vector_db.search(args["query"])
    return {"content": [{"type": "text", "text": str(results)}]}

def create_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=["Bash", "Read", "Edit"],
        mcp_servers={"docs": create_sdk_mcp_server("docs", tools=[search_docs])},
        system_prompt="You are a coding assistant with access to docs.",
        agents={"researcher": AgentDefinition(
            system_prompt="You research topics.",
            allowed_tools=["WebSearch", "Read"],
        )},
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[my_hook])]},
        can_use_tool=my_permission_handler,
        max_turns=50,
        max_budget_usd=1.0,
        thinking={"type": "adaptive"},
    )
```

**What the user changes from their existing code:** wrap the options
construction in `create_options()`. Delete client lifecycle code
(connect, query, stream loop, disconnect). That's it.

---

## How agent-plane handles each field

### Fields used as-is

These are passed directly to `ClaudeSDKClient`:

| Field | Behavior |
|-------|----------|
| `tools` / `allowed_tools` | SDK tool configuration |
| `system_prompt` | System instructions for the agent |
| `mcp_servers` | User's MCP servers registered alongside agent-plane's |
| `hooks` | PreToolUse, PostToolUse, etc. — run in-process |
| `can_use_tool` | Permission callback — runs in-process |
| `model` | Model selection (YAML `llm.model` overrides if set) |
| `thinking` | Thinking/reasoning configuration |
| `max_turns` | Turn limit |
| `max_budget_usd` | Cost budget |
| `output_format` | Structured output config |
| `betas` | SDK beta features |
| `plugins` | SDK plugins |

### Fields agent-plane overrides

These are set by agent-plane regardless of what the user provides:

| Field | Agent-plane sets | Reason |
|-------|-----------------|--------|
| `cwd` | `storage_dir` workspace | Agent-plane manages the working directory |
| `env` | `{"CLAUDECODE": ""}` + overrides | Prevents nested session errors |
| `cli_path` | System `claude` binary | Agent-plane finds the CLI |
| `permission_mode` | `bypassPermissions` | Server-side execution is trusted |
| `disallowed_tools` | `["Task"]` (merged) | Agent-plane manages sub-agents |
| `extra_args` | `{"no-session-persistence": None}` | Agent-plane manages session state |
| `include_partial_messages` | `True` | Needed for streaming |

### Fields silently ignored

These don't apply in agent-plane's execution model. They are
accepted without error but have no effect:

| Field | Why ignored |
|-------|-------------|
| `continue_conversation` | Agent-plane manages sessions via `conversation_id` / `previous_response_id` |
| `resume` | Agent-plane manages session continuity via `storage_dir` |
| `fork_session` | Not applicable — each task is a fresh execution |
| `setting_sources` | Agent-plane manages settings |
| `settings` | Agent-plane manages settings |
| `sandbox` | Agent-plane controls sandboxing at the tool level |
| `add_dirs` | Agent-plane controls filesystem access |
| `user` | Agent-plane manages user identity |
| `stderr` | Agent-plane captures stderr |

Agent-plane logs a one-time info message listing which fields were
ignored, so users aren't surprised.

---

## Sub-agent auto-translation

When the entrypoint's options include `agents={...}`, agent-plane
automatically translates each `AgentDefinition` into an agent-plane
sub-agent. The user's sub-agent definitions work as-is — no
rewriting to YAML config files.

### How it works

1. Agent-plane extracts `options.agents` before passing options to
   the SDK client.
2. For each `AgentDefinition`, builds an in-memory `AgentSpec`:

   | `AgentDefinition` field | `AgentSpec` field |
   |---|---|
   | `system_prompt` | `instructions` |
   | `allowed_tools` | `tools.builtins` (with `claude:` prefix) |
   | `model` | `llm.model` |
   | `mcp_servers` | Passed to sub-agent's `ClaudeAgentsExecutor` |
   | `hooks` | Passed to sub-agent's `ClaudeAgentOptions` |
   | `can_use_tool` | Passed to sub-agent's `ClaudeAgentOptions` |

3. Agent-plane sets `disallowed_tools=["Task"]` on the parent.
4. Agent-plane registers its own `Task` MCP handler that routes to
   the auto-generated sub-agent specs.
5. When the parent SDK calls `Task("research X")`, agent-plane's
   handler catches it and spawns a durable `agent_execution_workflow`
   for the sub-agent.
6. Each sub-agent gets its own conversation store, SSE streaming,
   crash recovery — full durability.

### What the user gets for free

- Sub-agent tool calls are persisted to conversation store
- Sub-agent crashes are recovered via DBOS re-invoke
- Sub-agent output is visible on the parent's SSE stream
- Client-side tools tunnel from sub-agents to the root
- No changes to their `AgentDefinition` code

### Nested sub-agents

If a sub-agent's `AgentDefinition` itself has `agents={...}`, the
translation recurses. Each level gets its own durable workflow.

---

## Client-side tools

Client-side tools registered by the API caller (in
`POST /v1/responses`) are merged as an additional MCP server on the
SDK client — same as the existing `ClaudeAgentsExecutor` behavior.
The user's MCP servers from `create_options()` coexist with
agent-plane's client-tool MCP server.

If a user-defined tool name conflicts with a client-side tool name,
the user's MCP tool takes precedence (it's registered first). The
client-side tool is shadowed. This is the expected behavior — the
entrypoint defines the agent's capabilities, the client can't
override them.

---

## YAML field interaction

When `executor.entrypoint` is set:

| YAML field | Behavior |
|-----------|----------|
| `llm.model` | Overrides `options.model` if set |
| `tools.builtins` | **Invalid** — entrypoint defines tools |
| `instructions` | **Invalid** — entrypoint sets `system_prompt` |
| `compaction` | Valid — workflow concern, independent of executor |
| `executor.timeout` | Valid — workflow concern |
| `executor.max_iterations` | Valid — workflow concern |

The validator rejects `tools.builtins` and `instructions` when
`entrypoint` is set. These would conflict with the entrypoint's
options. `llm.model` is allowed as a deployment-time override
(e.g., swap models without changing code).

---

## Dependencies

The entrypoint's imports must be available on the server. Options
for v1 (in priority order):

1. **Pre-installed**: Packages must be on the server. Simplest.
   Works when the operator controls the server environment.

2. **requirements.txt in bundle**: Agent-plane runs
   `pip install -r requirements.txt` at deploy time. Self-contained
   but adds deploy latency.

3. **PEP 723 inline metadata**: The entrypoint file declares deps
   inline. Consistent with local Python tools. Requires `uv`.

**Decision for v1**: Pre-installed. Document it. Add
`requirements.txt` support in a fast follow.

---

## Security

The entrypoint runs in the main server process. Same trust model as
any Python library the operator installs. Crash isolation is NOT
provided for v1 — same as importing any pip package.

| Risk | Mitigation |
|------|-----------|
| Crash kills server | Operator trusts deployed code (same as pip) |
| Infinite loop | `executor.timeout` kills the task |
| Memory leak | Operator monitors (same as any server) |
| Malicious code | Operator controls what's deployed |

Subprocess isolation for entrypoints is deferred — the SDK client
is stateful and long-lived, making subprocess boundaries complex.

---

## Implementation plan

### Phase 1: Entrypoint loading

1. **`spec/types.py`** — Add `entrypoint: str | None = None` to
   `ExecutorSpec`.
2. **`spec/parser.py`** — Parse `executor.entrypoint` from YAML.
3. **`spec/validator.py`** — Reject `tools.builtins` and
   `instructions` when `entrypoint` is set.
4. **`runtime/executors/claude.py`** — If spec has entrypoint:
   - Import the module from the agent's workdir
   - Call `create_options()`
   - Use returned options instead of building from YAML

### Phase 2: Sub-agent auto-translation

5. **`runtime/executors/claude.py`** — Extract `options.agents`:
   - Build in-memory `AgentSpec` for each `AgentDefinition`
   - Register on the parent spec's `sub_agents` list
   - Set `disallowed_tools=["Task"]` on parent options
   - Agent-plane's existing `Task` MCP handler routes to them

### Phase 3: requirements.txt

6. **`runtime/agent_cache.py`** — At bundle extraction time, if
   `requirements.txt` exists, run `pip install -r` into a
   per-agent virtualenv.

---

## Example deployment

### Bundle structure

```
my-agent/
├── config.yaml
├── agent.py          # entrypoint
├── my_tools.py       # custom tool implementations
└── requirements.txt  # optional (Phase 3)
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
    tool, create_sdk_mcp_server,
)
from my_tools import search_docs, validate_code

def create_options() -> ClaudeAgentOptions:
    doc_server = create_sdk_mcp_server(
        "docs", tools=[search_docs],
    )
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
            "PreToolUse": [
                HookMatcher(
                    matcher="Bash",
                    hooks=[validate_code],
                ),
            ],
        },
        max_turns=30,
        thinking={"type": "adaptive"},
    )
```

### What the user adapted

From their existing code, they:
1. Wrapped options construction in `create_options()` ✅
2. Deleted `ClaudeSDKClient` creation ✅
3. Deleted `connect()` / `query()` / stream loop / `disconnect()` ✅
4. Kept ALL `@tool` handlers, MCP servers, hooks, sub-agents ✅

Zero rewriting of agent logic. Only lifecycle code is removed.
