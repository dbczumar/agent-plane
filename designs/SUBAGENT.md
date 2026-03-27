# Sub-Agent Execution

## Context

The agent spec layer already supports sub-agents: `AgentSpec.sub_agents`
holds recursively parsed child specs, `ToolsConfig.agents` lists which
sub-agents this agent can call, the parser discovers them from
`agents/<name>/config.yaml`, and the validator checks that every
referenced name has a corresponding directory. What's missing is the
runtime — no tool exists to actually invoke a sub-agent during
execution. The agent loop ignores `spec.sub_agents` entirely today.

The goal is to make sub-agents callable during the agent loop via
`spawn_sub_agents` and `collect_sub_agents` built-in tools. Each sub-agent runs as an
independent DBOS workflow with its own task ID, conversation, and SSE
stream.

**Phasing:** Phase 1 implements spawn/collect with **server-side tools
only** — no client-side tools in sub-agents. The architecture is
designed so that client-side tool support (Phase 2) can be added
without refactoring Phase 1 code. See `SUBAGENT_WORKFLOW.md` for the
Phase 2 design.

### What exists

| Component | File | Status |
|-----------|------|--------|
| `AgentSpec.sub_agents: list[AgentSpec]` | `spec/types.py:303` | Recursive nested specs |
| `ToolsConfig.agents: list[str]` | `spec/types.py:156` | Declared callable sub-agents |
| `_discover_sub_agents()` | `spec/parser.py:495` | Parses `agents/<name>/config.yaml` recursively |
| `_validate_sub_agents()` | `spec/validator.py:206` | Every name in `tools.agents` has a directory |
| `Tool` ABC | `tools/base.py:25` | `name`, `get_schema()`, `invoke()` |
| `ToolManager` registry | `tools/manager.py:60` | `dict[str, Tool]` dispatch |
| `task_store.start()` | `stores/task_store` | Launch DBOS workflow for a task |
| `task_store.wait()` | `stores/task_store` | Block until task reaches terminal state |

### What's missing

1. `SpawnTool` + `CollectTool` — built-in tools for sub-agent lifecycle
2. Registration in `ToolManager` when sub-agents are declared

---

## Execution Model: Spawn / Collect

The parent calls `spawn_sub_agents` to launch one or more sub-agents as independent
tasks, then calls `collect_sub_agents` to gather their results. Each spawned
sub-agent is a full DBOS workflow with its own task ID, conversation,
and SSE stream.

```
Parent LLM → tool_call("spawn_sub_agents", {agents: [{name: "researcher", input: "..."},
                                           {name: "critic", input: "..."}]})
             → creates task resp_child1 (researcher)
             → creates task resp_child2 (critic)
             → returns {response_ids: ["resp_child1", "resp_child2"]}

[sub-agents run in parallel as independent DBOS workflows]
[each can call server-side tools; client-side tools in Phase 2]

Parent LLM → tool_call("collect_sub_agents", {response_ids: ["resp_child1", "resp_child2"]})
             → blocks until both reach terminal state
             → returns {results: [{task_id: "resp_child1", status: "completed",
                                   output: "RLHF papers..."}, ...]}
Parent LLM → continues with collected results
```

**Properties:**
- Each sub-agent is a full task with its own lifecycle
- Parallel execution — sub-agents run concurrently in separate threads
- SSE streaming per sub-agent — clients can subscribe to each
- Sub-agents use `_run_agent_loop()` — the same agent loop as top-level
  tasks (single execution path, no separate sub-agent loop)
- Parent blocks on `collect_sub_agents`, not on individual sub-agents
- Tasks are marked with a `spawned_sub_agent` flag (extension point
  for Phase 2 client-side tool support)

**Phase 1 limitation:** no client-side tools. Sub-agents can only use
server-side tools (MCP, skills, builtins). Client-side tools are
filtered out of the sub-agent's ToolManager. Phase 2 adds client-side
tool support via the tool result inbox pattern — see
`SUBAGENT_WORKFLOW.md`.

---

## Design Decisions

### Isolated conversations

Each sub-agent invocation creates a **new, isolated conversation**. The
sub-agent cannot see the parent's history or any previous sub-agent
invocations. The parent chooses what context to pass via the `input`
parameter.

Why: shared history would leak the parent's tool calls, instructions,
and other sub-agents' output into the sub-agent's context. Isolation
keeps each sub-agent focused on its specific task. The parent agent is
responsible for summarizing relevant context in the input.

The sub-agent's conversation is persisted in the conversation store for
auditability — you can inspect what the sub-agent saw and produced.

### Spawn depth limit

Sub-agents can themselves have sub-agents (the spec is recursive). A
spawned sub-agent that itself spawns sub-agents creates a tree of
workflows. A hard depth limit of **5** prevents runaway nesting. The
depth counter is threaded through: `SpawnTool` passes `depth + 1` when
creating the sub-agent's `ToolManager`. When the limit is reached,
`SpawnTool.invoke()` returns an error string instead of creating tasks.

Why 5: three levels of delegation (parent → specialist → sub-specialist)
covers practical use cases. Five gives headroom without enabling
pathological recursion. The limit is a runtime constant, not
configurable in the spec — agent authors should not need to think about
recursion depth.

### Timeout

`collect_sub_agents` accepts an optional `timeout` parameter (seconds). If any
sub-agent hasn't completed by the timeout, `collect_sub_agents` returns with
partial results — completed sub-agents have their output, timed-out
sub-agents have `status: "incomplete"`. If no timeout is specified,
`collect_sub_agents` uses the parent's remaining execution timeout (wall-clock
deadline minus elapsed time) to avoid the parent hanging indefinitely.
Each spawned sub-agent also respects its own `execution.timeout`
independently.

### Sub-agent spec loading across the DBOS boundary

Sub-agents are nested `AgentSpec` objects inside the parent's
`spec.sub_agents` — they're not registered in the agent store. But
the DBOS workflow boundary requires serializable inputs (you can't pass
a Python object).

Solution: `task_store.start()` accepts an optional `sub_agent_name`
kwarg. The workflow receives `(agent_id, sub_agent_name)`:

```python
# In agent_execution_workflow:
loaded = get_agent_cache().load(agent_id)
if sub_agent_name is not None:
    # Resolve nested sub-agent spec from parent
    sub_specs = {sa.name: sa for sa in loaded.spec.sub_agents}
    spec = sub_specs[sub_agent_name]
else:
    spec = loaded.spec
```

`agent_id` is always the **root registered agent**. The parent's spec
is already cached (in-memory tier 1 hit in `AgentCache`), so loading
it again is a dict lookup. The resolved sub-agent spec already contains
its own `sub_agents` list — if a sub-agent itself spawns children, the
same mechanism works: pass the root `agent_id` + the child's name,
load parent, extract child. No need for path-style lookups because
each level's `spec.sub_agents` is the full recursive tree.

`sub_agent_name` is also persisted on the task row (new column). This
serves double duty: the workflow uses it for spec loading, and
`CollectTool` reads it via `task_store.get()` to populate the
`agent_name` field in collect results.

### Sub-agent gets its own ToolManager

Each sub-agent invocation creates a fresh `ToolManager` with the
sub-agent's own tools (MCP servers, skills, local tools, and nested
sub-agents). The parent's tools are **not** inherited — isolation is
the default. This matches the existing design where sub-agent specs are
self-contained.

### `spawn_sub_agents` / `collect_sub_agents` are built-in tools

Two generic built-in tools handle the sub-agent lifecycle:

- **`spawn_sub_agents`** — takes a list of `{name, input}` pairs, creates tasks,
  returns task IDs
- **`collect_sub_agents`** — takes task IDs, blocks until complete, returns results

These are registered once in the `ToolManager` when the agent has any
sub-agents declared.

Why generic tools: spawning is a coordination primitive, not a
per-sub-agent concern. A single `spawn_sub_agents` call can launch multiple
different sub-agents. Per-agent spawn tools would clutter the schema and
force sequential spawning.

### `collect_sub_agents` is explicit (LLM calls it)

The LLM explicitly calls `collect_sub_agents` to gather results, rather than
`spawn_sub_agents` blocking until all sub-agents complete. This is modeled after
Temporal's `start_child_workflow()` + `await handle.result()` pattern
and LangGraph's fan-out/fan-in — the two frameworks with robust parallel
HITL support both use explicit collection.

Why explicit: the LLM retains control between `spawn_sub_agents` and `collect_sub_agents`. It
could spawn agents, do other tool calls (e.g. its own research), then
collect results. Implicit blocking would force the parent to idle.
Explicit collection also lets the LLM handle partial results — if
`collect_sub_agents` times out, the LLM can decide whether to retry, proceed with
partial data, or ask for more time.

### Client discovery of spawned sub-agent tasks

No special discovery mechanism is needed. `spawn_sub_agents` is a normal tool call
— the parent's SSE stream emits the standard `function_call` and
`function_call_output` events. The `function_call_output` contains the
task IDs:

```json
{"response_ids": ["resp_child1", "resp_child2"]}
```

The client watches for tool calls named `spawn_sub_agents` in the parent's output
stream, extracts the task IDs from the result, and interacts with each
sub-agent via existing endpoints:
- Subscribe to the sub-agent's SSE stream via
  `GET /v1/responses/{task_id}/stream`
- Poll status via `GET /v1/responses/{task_id}`
- (Phase 2) Respond to client-side tool calls via
  `POST /v1/responses` with `previous_response_id` pointing to the
  sub-agent's response

No new SSE event types, no new response object fields, no new API
routes.

### LLM config is required on sub-agents

A sub-agent without an `llm` block cannot execute. The validator already
requires `spec_version` but not `llm`. For sub-agents referenced in
`tools.agents`, validation should require `llm.model` to be set. This
is a new validation rule.

### Output extraction in `collect_sub_agents`

`collect_sub_agents` returns extracted final text, not the raw
`task.output` list. `task.output` is a list of response items (text
blocks, tool calls, etc.) — the parent LLM doesn't need the
sub-agent's internal tool call chain, just the answer.

Extraction: walk `task.output`, pull items where `type ==
"output_text"`, concatenate their text content. For non-terminal
tasks (incomplete, failed, cancelled), return a descriptive error
string instead: `"Sub-agent 'researcher' did not complete (status:
incomplete)."`. The `output` field is always a string, never null —
the parent LLM gets a uniform type and can reason about failures
from the error message.

Why text-only: matches what other frameworks do (OpenAI Agents SDK,
CrewAI return final text). The full output is persisted on the task
and accessible via `GET /v1/responses/{id}` if anyone needs it.
Edge case: if the sub-agent terminates without a text block (e.g.,
hits timeout mid-tool-call), this falls into the non-terminal path
and the parent gets the error string.

### Client-side tools are excluded from sub-agents (Phase 1)

In Phase 1, client-side tools are filtered out of the sub-agent's
ToolManager. If the sub-agent's LLM attempts to call a client-side
tool, it receives the standard "tool not found" error and must proceed
without it.

Client-side tools are architecturally possible (each sub-agent is an
independent workflow with its own SSE stream) but require the tool
result inbox mechanism described in `SUBAGENT_WORKFLOW.md`. Phase 1
lays the groundwork:

- Spawned sub-agents use `_run_agent_loop()` (the real agent loop),
  not a separate code path — Phase 2 adds a park branch in
  `_handle_tool_calls()` without touching `collect_sub_agents` or `spawn_sub_agents`
- Tasks are marked with `spawned_sub_agent` flag — Phase 2 uses this
  for the `POST /v1/responses` inbox delivery branch
- `collect_sub_agents` uses `task_store.wait(task_id)` — this doesn't change in
  Phase 2 because the sub-agent workflow stays alive while parked

---

## Tool Schemas

### `spawn_sub_agents` tool

```json
{
  "type": "function",
  "function": {
    "name": "spawn_sub_agents",
    "description": "Launch one or more sub-agents as independent parallel tasks. Returns response IDs immediately. Use collect_sub_agents() to gather results.\n\nAvailable sub-agents:\n- researcher: Searches the web and summarizes findings.\n- critic: Reviews text for factual errors.",
    "parameters": {
      "type": "object",
      "properties": {
        "agents": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string",
                "enum": ["researcher", "critic"],
                "description": "Sub-agent name."
              },
              "input": {
                "type": "string",
                "description": "The task or question for the sub-agent."
              }
            },
            "required": ["name", "input"]
          },
          "description": "List of sub-agents to spawn with their inputs."
        }
      },
      "required": ["agents"]
    }
  }
}
```

Both the `enum` on `name` and the sub-agent list in `description` are
built dynamically by `SpawnTool.get_schema()` from `sub_specs`. The
example above shows an agent with two sub-agents: `researcher` and
`critic`. Each sub-agent's description comes from `spec.description`.

### `collect_sub_agents` tool

```json
{
  "type": "function",
  "function": {
    "name": "collect_sub_agents",
    "description": "Wait for spawned sub-agent tasks to complete and return their results.",
    "parameters": {
      "type": "object",
      "properties": {
        "response_ids": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Response IDs returned by spawn_sub_agents()."
        },
        "timeout": {
          "type": "integer",
          "description": "Maximum seconds to wait. Omit to use the remaining execution timeout."
        }
      },
      "required": ["response_ids"]
    }
  }
}
```

**`collect_sub_agents` return value:**

```json
{
  "results": [
    {
      "response_id": "resp_child1",
      "agent_name": "researcher",
      "status": "completed",
      "output": "Recent RLHF papers include: (1) InstructGPT..."
    },
    {
      "response_id": "resp_child2",
      "agent_name": "critic",
      "status": "incomplete",
      "output": "Sub-agent 'critic' did not complete (status: incomplete)."
    }
  ]
}
```

**Output extraction:** `output` is always a string, never null. For
completed tasks, `CollectTool` walks `task.output`, extracts items
where `type == "output_text"`, and concatenates their text content.
For non-completed tasks (incomplete, failed, cancelled), `output` is
a human-readable error string like `"Sub-agent 'critic' did not
complete (status: incomplete)."` — the parent LLM gets a single
string either way and can decide how to react.

---

## New Files

```
agent_plane/
  tools/
    builtins/
      spawn.py         # NEW — SpawnTool + CollectTool
```

## Modified Files

```
agent_plane/
  tools/
    manager.py         # MODIFY — register spawn/collect tools
  spec/
    validator.py       # MODIFY — require llm on callable sub-agents
```

---

## File Details

### 1. `tools/builtins/spawn.py` — SpawnTool + CollectTool

Two tools that manage the lifecycle of spawned sub-agents.

```python
class SpawnTool(Tool):
    """
    Launch sub-agents as independent DBOS workflow tasks.

    :param sub_specs: Name-to-AgentSpec mapping for available
        sub-agents.
    :param parent_task_id: The parent workflow's task ID,
        e.g. ``"resp_abc123"``.
    :param parent_agent_id: The parent agent's ID (sub-agents
        run under the same agent registration).
    :param depth: Current spawn nesting depth (0 = top-level).
    :param max_depth: Maximum allowed nesting depth.
    """

    @property
    def name(self) -> str:
        return "spawn_sub_agents"

    def invoke(self, arguments: str) -> str:
        # 1. Check depth < max_depth (return error string if exceeded)
        # 2. Parse arguments: list of {name, input} dicts
        # 3. For each agent:
        #    a. Validate name is in sub_specs
        #    b. Create conversation via conv_store.create_conversation()
        #    c. Append user input message
        #    d. Create task via task_store.create()
        #       - agent_id = parent_agent_id (root registered agent)
        #       - Mark task with spawned_sub_agent=True
        #    e. task_store.start(task_id, sub_agent_name=name)
        #       Workflow loads parent spec via agent_cache, extracts
        #       sub-agent by name from spec.sub_agents
        #
        # NOTE: Steps 3b-3e are not transactional. A crash between
        # create and start leaves an orphaned QUEUED task. This is
        # acceptable: the parent is a DBOS workflow, so DBOS replays
        # invoke() on recovery, creating a fresh task. The orphan is
        # inert. Same gap exists for top-level task creation today.
        #
        # 4. Return JSON: {response_ids: [...]}
        ...


class CollectTool(Tool):
    """
    Wait for spawned sub-agent tasks to complete and return
    their results.

    :param parent_task_id: The parent workflow's task ID
        (for timeout calculation).
    """

    @property
    def name(self) -> str:
        return "collect_sub_agents"

    def invoke(self, arguments: str) -> str:
        # 1. Parse arguments: response_ids list + optional timeout
        # 2. Calculate effective timeout:
        #    min(explicit_timeout, remaining_parent_execution_time)
        # 3. For each task_id:
        #    a. task_store.wait(task_id, timeout=remaining)
        #    b. task_store.get(task_id) → extract final output text
        # 4. Return JSON with results (status + output per task)
        # 5. Timed-out tasks get status: "incomplete"
        ...
```

### 2. `tools/manager.py` — Register spawn/collect tools

In `ToolManager.__init__`, after registering skill tools and before
registering client tools, register spawn/collect:

```python
def _register_sub_agent_tools(self) -> None:
    """
    Register spawn/collect tools when the agent has sub-agents
    declared in tools.agents.
    """
    if not self._spec.tools.agents:
        return

    # Build name → spec lookup from the recursive sub_agents list
    sub_specs = {sa.name: sa for sa in self._spec.sub_agents}

    self._tools["spawn_sub_agents"] = SpawnTool(
        sub_specs=sub_specs,
        parent_task_id=self._parent_task_id,
        parent_agent_id=self._agent_id,
        depth=self._sub_agent_depth,
    )
    self._tools["collect_sub_agents"] = CollectTool(
        parent_task_id=self._parent_task_id,
    )
```

`ToolManager.__init__` gains new parameters:

- `parent_task_id: str` — the current workflow's task ID
- `sub_agent_depth: int = 0` — current nesting depth
- `agent_id: str` — the parent agent's registered ID

### 3. `spec/validator.py` — Require LLM on callable sub-agents

Add a check in `_validate_sub_agents()`: for each name in
`tools.agents`, the corresponding sub-agent spec must have a non-None
`llm` block with a non-empty `model`.

```python
# In _validate_sub_agents():
for name in spec.tools.agents:
    sub = sub_specs.get(name)
    if sub is None:
        result.add(
            f"tools.agents[{name!r}]",
            "no matching sub-agent directory",
        )
        continue
    if sub.llm is None or not sub.llm.model:
        result.add(
            f"sub_agents[{name!r}].llm",
            "callable sub-agent must have llm.model configured",
        )
```

---

## Walkthrough

### Agent spec structure

```
my-agent/
├── config.yaml
│   spec_version: 1
│   name: orchestrator
│   llm:
│     model: openai/gpt-4o
│   tools:
│     agents: [researcher, critic]
│   instructions: |
│     You are an orchestrator. Spawn sub-agents to gather
│     information and review work in parallel.
└── agents/
    ├── researcher/
    │   └── config.yaml
    │       spec_version: 1
    │       name: researcher
    │       description: Searches the web and summarizes findings.
    │       llm:
    │         model: openai/gpt-4o-mini
    │       instructions: |
    │         You are a research assistant. Search thoroughly
    │         and return a concise summary of your findings.
    └── critic/
        └── config.yaml
            spec_version: 1
            name: critic
            description: Reviews text for factual errors.
            llm:
              model: openai/gpt-4o-mini
            instructions: |
              You are a critic. Review the given text and
              identify any factual errors or weak arguments.
```

### Step 1: Agent loaded, ToolManager created

The workflow loads the orchestrator's spec. `spec.sub_agents` contains
two entries: researcher and critic. `spec.tools.agents` is
`["researcher", "critic"]`.

`ToolManager.__init__` calls `_register_sub_agent_tools()`, which
registers:
- `SpawnTool` — launch sub-agents as independent tasks
- `CollectTool` — wait for results

### Step 2: LLM spawns sub-agents

The LLM decides it needs parallel work:

```json
{"tool_calls": [{"call_id": "call_spawn_sub_agents", "name": "spawn_sub_agents",
  "arguments": "{\"agents\": [
    {\"name\": \"researcher\", \"input\": \"Find papers on RLHF\"},
    {\"name\": \"critic\", \"input\": \"Review this draft for factual errors\"}
  ]}"}]}
```

`SpawnTool.invoke()`:
1. Creates a conversation + task for each sub-agent
2. Starts each task via `task_store.start()` (launches DBOS workflow)
3. Returns immediately:
   ```json
   {"response_ids": ["resp_child1", "resp_child2"]}
   ```

The parent's agent loop persists this as a normal `function_call_output`
item. Streaming clients see the spawn call and its result via the
standard `response.output_item.done` SSE events — no special event
types needed. The client extracts task IDs from the tool output and
interacts with each sub-agent via existing API endpoints.

### Step 3: Sub-agents run in parallel

Both sub-agents execute as independent DBOS workflows via
`_run_agent_loop()` — the same agent loop used by top-level tasks:
- Each has its own conversation, ToolManager, and agent loop
- Each can call its own server-side tools (MCP, skills, etc.)
- Client-side tools are filtered out of the ToolManager (Phase 1)
- Each sub-agent streams output via its own SSE stream

### Step 4: Parent collects results

The parent's next LLM turn calls `collect_sub_agents`:

```json
{"tool_calls": [{"call_id": "call_collect_sub_agents", "name": "collect_sub_agents",
  "arguments": "{\"response_ids\": [\"resp_child1\", \"resp_child2\"]}"}]}
```

`CollectTool.invoke()`:
1. Calls `task_store.wait()` for each task ID
2. Extracts final output text from each completed task
3. Returns:
   ```json
   {"results": [
     {"response_id": "resp_child1", "agent_name": "researcher",
      "status": "completed", "output": "RLHF papers include..."},
     {"response_id": "resp_child2", "agent_name": "critic",
      "status": "completed", "output": "Found 2 factual errors..."}
   ]}
   ```

### Step 5: Parent synthesizes

The parent LLM sees both results and produces a final response that
incorporates the researcher's findings and the critic's feedback.

---

## Implementation Phases

### Phase 1: Server-side tools only

#### Phase 1A: Spawn/collect tools

1. `tools/builtins/spawn.py` — `SpawnTool` + `CollectTool`
2. `tools/manager.py` — add `_register_sub_agent_tools()`, add
   `parent_task_id`, `sub_agent_depth`, and `agent_id` parameters
3. Wire `SpawnTool` to create tasks via `task_store`
   - Mark tasks with `spawned_sub_agent=True` flag
   - Spawned sub-agents use `_run_agent_loop()` (single execution path)
   - Filter out client-side tools from sub-agent ToolManager
4. Wire `CollectTool` to wait via `task_store.wait()` and extract
   results
5. Update `ToolManager` construction sites (workflow.py)
6. Tests: spawn creates tasks, collect waits and returns results,
   timeout handling, depth limit

#### Phase 1B: Validation

1. `spec/validator.py` — require `llm.model` on callable sub-agents
2. Tests: verify validation catches missing LLM config

#### Phase 1C: Integration

1. Integration test: spawn one sub-agent, collect result
2. Integration test: spawn two sub-agents, both complete, collect
   returns both results
3. Integration test: collect with timeout, partial results
4. Integration test: depth limit exceeded → error string
5. Verify sub-agent conversations persisted in store
6. Verify `spawned_sub_agent` flag is set on tasks

#### Phase 1 extension points for Phase 2

These are laid in Phase 1 but not used until Phase 2:

- **`spawned_sub_agent` flag on tasks** — `SpawnTool` sets this at
  creation. Phase 2 uses it in the `POST /v1/responses` endpoint to
  detect parked sub-agents and deliver tool results to the inbox
  instead of creating a new task.
- **`_run_agent_loop()` as single execution path** — spawned sub-agents
  use the real agent loop. Phase 2 adds a park branch inside
  `_handle_tool_calls()` when client-side tools are detected and the
  task has `spawned_sub_agent=True`. No new loop needed.
- **`collect_sub_agents` uses `task_store.wait()`** — this call doesn't change in
  Phase 2. When a sub-agent parks for client tools, its workflow stays
  alive, so `wait()` still blocks until the workflow truly completes.
- **Client-side tool filtering is one line** — Phase 2 removes the
  filter and adds the park logic. No other changes to `ToolManager` or
  tool registration.

### Phase 2: Client-side tools in spawned sub-agents

See `SUBAGENT_WORKFLOW.md` for the full design. Summary of changes:

1. **Remove client-side tool filter** from sub-agent ToolManager setup
   (one-line change)
2. **Add park branch in `_handle_tool_calls()`** — when the task has
   `spawned_sub_agent=True` and client-side tools are detected, park
   the workflow instead of completing the response
3. **Add `tool_result_inbox` table** — new database table for client
   tool result delivery (no migration of existing tables)
4. **Add branch in `POST /v1/responses`** — check `spawned_sub_agent`
   flag + parked status; deliver to inbox instead of creating new task
5. Tests: spawned sub-agent with client-side tool, multi-round parking,
   parallel sub-agents with independent client tools

---

## Verification (Phase 1)

1. **Unit tests**: `SpawnTool` creates tasks correctly with
   `spawned_sub_agent=True`, returns task IDs
2. **Unit tests**: `CollectTool` waits and returns results, timeout
   returns partial results
3. **Unit tests**: spawn depth limit — exceeded returns error string
4. **Validation tests**: sub-agent without `llm.model` fails validation
5. **Integration tests**: spawn + collect flow with parallel mock
   sub-agents (server-side tools only)
6. **Integration tests**: full workflow with parent spawning sub-agent
   via `ControllableMockClient` — verify parent receives sub-agent
   output via collect
7. **Extension point test**: verify `spawned_sub_agent` flag is set on
   tasks created by `SpawnTool`

---

## Open Questions

### Q1: Parent timeout while blocked on `collect_sub_agents`

The parent blocks on `collect_sub_agents` while sub-agents run. If a sub-agent
hangs (e.g. infinite LLM loop, or in Phase 2: waiting on a client-side
tool call that never comes), the parent hangs too.

Mitigation: `collect_sub_agents` defaults to the parent's remaining execution
timeout. If the parent's execution timeout expires, the parent's agent
loop terminates with `status: "incomplete"`. The sub-agent tasks
continue running independently (they have their own execution timeouts).

Future optimization: async waiting (release the DBOS thread while
blocked on `collect_sub_agents`, re-acquire when sub-agents complete). Requires
changes to how DBOS manages workflow threads.

---

## Not Yet

- **Client-side tools in spawned sub-agents (Phase 2)** — the full
  design is in `SUBAGENT_WORKFLOW.md`. Requires: tool result inbox
  table, park branch in `_handle_tool_calls()`, delivery branch in
  `POST /v1/responses`. Phase 1 lays extension points (spawned flag,
  single agent loop, `collect_sub_agents` via `wait()`) so Phase 2 is additive.
- **Shared conversation mode** — allow a sub-agent to see the parent's
  conversation history (opt-in via spec). Useful for "advisor" patterns
  where the sub-agent needs full context.
- **Per-sub-agent timeout/retry** — `ToolsConfig.agents` is currently
  `list[str]`. TIMEOUTS.md shows a richer format with per-agent
  `timeout` and `retry`. Parsing this requires changing `agents` from
  `list[str]` to `list[str | dict]` and updating the parser.
- **Context passing** — structured context injection beyond the `input`
  string. Could allow the parent to pass files, conversation excerpts,
  or typed parameters to the sub-agent.
- **Result format** — currently returns plain text. Could return
  structured output (JSON) if the sub-agent's spec declares an output
  schema.
- **Sub-agent cancellation** — when the parent's task is cancelled,
  propagate cancellation to spawned sub-agent tasks. Currently spawned
  sub-agents would continue running independently.
- **Collect with partial completion** — allow `collect_sub_agents` to return as
  soon as N-of-M sub-agents complete, rather than waiting for all.
  Useful for best-of-N patterns where the parent only needs the first
  good result.
