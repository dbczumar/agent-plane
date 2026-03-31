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

Spawn/collect tools and client-side tool tunneling are implemented
together. See `SUBAGENT_WORKFLOW.md` for the client-side tool
tunneling design.

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
| `task_store.wait()` | `stores/task_store` | Block until task reaches terminal state (async; CollectTool uses sync DBOS handles instead) |

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
[each can call server-side and client-side tools]

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
- Spawned tasks have `root_task_id` set (identifies them as
  sub-agents; enables client-side tool tunneling via
  `SUBAGENT_WORKFLOW.md`)

**Client-side tools:** Sub-agents support client-side tools via the
tunneled model described in `SUBAGENT_WORKFLOW.md`. When a sub-agent
hits a client-side tool, the call is published to the root response's
output and the sub-agent parks until the client responds.

---

## Design Decisions

### Isolated conversations

Each sub-agent invocation creates a **new, isolated conversation** with
`kind="sub_agent"`. The sub-agent cannot see the parent's history or
any previous sub-agent invocations. The parent chooses what context to
pass via the `input` parameter.

Why: shared history would leak the parent's tool calls, instructions,
and other sub-agents' output into the sub-agent's context. Isolation
keeps each sub-agent focused on its specific task. The parent agent is
responsible for summarizing relevant context in the input.

The sub-agent's conversation is persisted in the conversation store for
auditability — you can inspect what the sub-agent saw and produced.
Sub-agent conversations are created with `title=None` (same as
top-level conversations — no auto-generated title). The `kind` field
is sufficient to distinguish them.

`GET /v1/conversations` filters out `kind="sub_agent"` by default so
they don't clutter the listing, but clients can still access them by
ID via `GET /v1/conversations/{id}`.

### Timeout

`collect_sub_agents` accepts an optional `timeout` parameter (seconds).

**Effective timeout calculation:**
`effective = min(explicit_timeout, remaining_parent_execution_time)`.
If no explicit timeout is given, the parent's remaining execution
timeout is used. If the parent has no execution timeout, there is no
deadline — `collect_sub_agents` blocks until all sub-agents complete (or
fail).

**Behavior on timeout:**
`collect_sub_agents` returns with partial results. Each sub-agent in the
results list has one of:
- `status: "completed"`, `output: "<extracted text>"` — finished
  before the deadline
- `status: "incomplete"`, `output: "Sub-agent '<name>' did not
  complete (status: <current_status>)."` — still running when the
  deadline hit

The timed-out sub-agent tasks are **not cancelled** — they continue
running independently with their own execution timeouts. The parent
LLM receives the partial results and decides how to proceed (retry,
proceed without, ask for more time). Cancellation propagation is a
future enhancement (see "Not Yet").

Each spawned sub-agent also respects its own `execution.timeout`
independently — a sub-agent can time out on its own even if the
parent's `collect_sub_agents` deadline hasn't passed.

### Sub-agent spec loading across the DBOS boundary

Sub-agents are nested `AgentSpec` objects inside the parent's
`spec.sub_agents` — they're not registered in the agent store. But
the DBOS workflow boundary requires serializable inputs (you can't pass
a Python object).

Solution: the workflow reads its own task row to determine if it's a
sub-agent. If `root_task_id` is set, the task is a spawned sub-agent —
the workflow uses the existing `agent_name` column (which holds the
sub-agent name, e.g. `"researcher"`) to find the sub-agent spec via
recursive search through the root spec tree:

```python
# In agent_execution_workflow:
loaded = get_agent_cache().load(agent_id)
task = get_task_store().get(task_id)
if task.root_task_id is not None:
    # Spawned sub-agent: find spec by agent_name in the full tree
    spec = _find_sub_agent_spec(loaded.spec, task.agent_name)
else:
    spec = loaded.spec


def _find_sub_agent_spec(spec: AgentSpec, name: str) -> AgentSpec | None:
    """
    Recursively search the spec tree for a sub-agent by name.

    Sub-agent names are validated to be unique across the entire spec
    tree, so this always finds at most one match.

    :param spec: The root agent spec to search.
    :param name: The sub-agent name to find, e.g. ``"researcher"``.
    :returns: The matching sub-agent spec, or None if not found.
    """
    for sa in spec.sub_agents:
        if sa.name == name:
            return sa
        found = _find_sub_agent_spec(sa, name)
        if found is not None:
            return found
    return None
```

`agent_id` is always the **root registered agent**. The spec is
already cached (in-memory tier 1 hit in `AgentCache`), so loading it
is a dict lookup. Sub-agent names are unique across the entire spec
tree (enforced by the validator), so the recursive search always finds
the right spec regardless of nesting depth.

No new workflow parameters are needed — the workflow discovers its
sub-agent status from the task row. `SpawnTool` sets `agent_name` to
the sub-agent name and `root_task_id` to the top-level task's ID at
creation time.

### Schema change: `root_task_id`

One new nullable column on the task table:

```
root_task_id  String(64)  NULLABLE  FK → tasks.id ON DELETE CASCADE
              INDEX(root_task_id)
```

- **Type**: `String(64)` — matches existing ID columns (`SqlTask.id`,
  `SqlTask.agent_id`, etc.)
- **Nullable**: `NULL` for top-level tasks, set for sub-agents
- **FK cascade**: `ON DELETE CASCADE` — if the root task is deleted,
  sub-agent tasks are cleaned up automatically
- **Index**: indexed for `pending_tool_calls` lookups (JOIN on
  root_task_id) and for "find all sub-agents of a root task" queries

Uses:
- **Spec loading**: workflow checks `root_task_id IS NOT NULL`, then
  uses existing `agent_name` to find the sub-agent spec via recursive
  search through the root spec tree.
- **Tunneled tool calls**: sub-agent publishes `function_call`
  items directly to the root task's response output. No chain-walking
  needed regardless of nesting depth.
- **PATCH routing**: `PATCH /v1/responses` checks
  `root_task_id IS NOT NULL` to decide whether to deliver tool
  results to the inbox instead of creating a new task.

`SpawnTool` sets `root_task_id` at creation time. For a top-level
parent spawning a sub-agent, `root_task_id` is the parent's own task
ID. For nested spawns (sub-agent spawning sub-sub-agent),
`root_task_id` is propagated from the parent — always pointing to the
original top-level task.

Backward-compatible — nullable, so existing tasks are unaffected.

### Schema change: conversation `kind`

One new non-nullable column on the conversations table:

```
kind  String(32)  NOT NULL  DEFAULT "default"
      CHECK(kind IN ("default", "sub_agent"))
      INDEX(kind)
```

- **Type**: `String(32)` — short enum-like values, 32 chars leaves
  room for future kinds without a migration
- **Not nullable**: every conversation has a kind. Server default
  ensures existing rows get `"default"` on migration.
- **Check constraint**: restricts to known values. New values require
  a migration (intentional — adding a new kind is a deliberate act).
- **Index**: `list_conversations` filters by kind on every call

Values: `"default"` (user-initiated) or `"sub_agent"` (created by
`SpawnTool` for sub-agent execution).

`SpawnTool` sets `kind="sub_agent"` at conversation creation time.
`GET /v1/conversations` filters to `kind="default"` so sub-agent
conversations don't appear in the listing. Clients can still access
sub-agent conversations directly via `GET /v1/conversations/{id}` —
the `kind` field only affects listing, not access.

Backward-compatible — non-nullable with a server default of
`"default"`, so existing rows are unaffected.

### Schema migration approach

All schema changes (new columns and new tables) are made by updating
the existing model definitions in `db_models.py`:

- **`root_task_id`**: add column to existing `SqlTask` model
- **`kind`**: add column to existing `SqlConversation` model
- **`pending_tool_calls`**: new table added to `db_models.py`

No separate Alembic migration scripts. The existing `Base.metadata`
picks up the changes, and `create_all()` handles schema creation.
Both new columns are backward-compatible (nullable or have server
defaults), so existing data is unaffected.

### Store access in tools

`SpawnTool` and `CollectTool` need access to `task_store` and
`conversation_store` at invoke time. Rather than passing stores as
constructor parameters, tools call runtime getters inside `invoke()`:
`get_task_store()` and `get_conversation_store()`. This matches the
existing pattern — the workflow uses `get_agent_cache()` the same way.
Stores are module-level singletons initialized at server startup, so
the getters are a dict lookup.

### CollectTool uses sync DBOS primitives

`CollectTool.invoke()` is sync (matching the `Tool.invoke()` ABC).
`task_store.wait()` is async (designed for the FastAPI API layer).
CollectTool bypasses `task_store.wait()` and uses sync DBOS primitives
directly: `retrieve_workflow(task_id)` → `handle.get_result()`. Both
are sync and block the DBOS workflow thread until the sub-agent
workflow completes. This is the correct pattern inside a DBOS workflow.

### `task_store.wait()` gets a timeout parameter

`task_store.wait()` currently blocks indefinitely. Add
`timeout: float | None = None`. If the deadline expires before the
workflow completes, `wait()` returns the task in its current
(non-terminal) state instead of raising.

**Two call paths, same semantics:**

- **API layer (async)**: `task_store.wait(task_id, timeout=X)` —
  existing async method, used by `GET /v1/responses` long-poll.
  Implementation: `asyncio.wait_for()` wrapping the existing
  DBOS event wait, catching `TimeoutError` and returning current
  task state.

- **Workflow layer (sync)**: `CollectTool` runs inside a DBOS
  workflow thread and cannot use async. It uses a **polling loop**:
  `DBOS.sleep(interval)` + `task_store.get(task_id)` + deadline
  check. If the task reaches a terminal state, return. If the
  deadline expires, return the task in its current state.
  `DBOS.sleep()` is used (not `time.sleep()`) so DBOS can track
  the sleep for replay.

Both paths return the same thing: a `Task` object. The caller checks
`task.status` to determine whether the task completed or timed out.

### Sub-agent system prompt

Each sub-agent uses its **own** `instructions` from its spec — it does
not inherit the parent's system prompt. This follows from conversation
isolation: the sub-agent gets a fresh conversation with only the
`input` string as context. The parent's instructions, tool
descriptions, and conversation history are not visible.

If a sub-agent's spec has no `instructions` field, the LLM receives
no system prompt (same behavior as a top-level agent with no
instructions). The parent can pass guidance via the `input` parameter
if needed.

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

### Client interaction with sub-agents

The client sees `spawn_sub_agents` and `collect_sub_agents` as normal
tool calls on the parent's SSE stream. Sub-agent client-side tool
calls are **tunneled through the parent's response**: the
`function_call` item appears in the parent response's output with
`status: "action_required"` and `model` set to `"parent.child"`
(e.g. `"orchestrator.researcher"`). The client responds via
`PATCH /v1/responses/{parent_id}` with tool results. The client
never needs to subscribe to sub-agent streams or track sub-agent
task IDs.

See `SUBAGENT_WORKFLOW.md` for the full client-side tool design.

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
Edge cases:
- **Non-terminal** (incomplete, failed, cancelled): the parent gets
  a descriptive error string like `"Sub-agent 'researcher' did not
  complete (status: incomplete)."`.
- **Completed with no `output_text` items** (e.g. sub-agent only
  made tool calls and never produced text): `output` is an empty
  string `""`. The parent LLM can interpret this as "no answer."
- **Timeout mid-tool-call**: falls into the non-terminal path above.

### Client-side tools in sub-agents

Sub-agents support client-side tools via the **tunneled model**
described in `SUBAGENT_WORKFLOW.md`. When a sub-agent hits a
client-side tool, the `function_call` is published to the **root's**
response output (with `status: "action_required"` and
`model: "parent.child"`). The client responds via
`PATCH /v1/responses/{root_id}`.

Key properties:
- Sub-agents use `_run_agent_loop()` (the real agent loop), not a
  separate code path — the park branch in `_handle_tool_calls()`
  activates when client-side tools are detected and
  `root_task_id IS NOT NULL`
- `root_task_id` on the task row tells the sub-agent which root
  response's output to publish to
- `collect_sub_agents` uses sync DBOS `handle.get_result()` — when a
  sub-agent parks for client tools, its workflow stays alive, so
  `get_result()` still blocks until the workflow truly completes
- Client-side tool specs are passed through to the sub-agent's
  `ToolManager` — same `client_tool_specs` as the parent

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
  db/
    db_models.py       # MODIFY — add root_task_id column to SqlTask,
                       #           add kind column to SqlConversation
  entities/
    task.py            # MODIFY — add root_task_id field to Task
    conversation.py    # MODIFY — add kind field to Conversation
  stores/
    task_store/
      __init__.py         # MODIFY — add root_task_id to create(),
                         #           add pending_tool_call methods
      sqlalchemy_store.py # MODIFY — add root_task_id to create(),
                         #           add timeout to wait()
    conversation_store/
      __init__.py        # MODIFY — add kind param to create_conversation()
      sqlalchemy_store.py # MODIFY — filter by kind in list(),
                         #           pass kind in create_conversation()
  runtime/
    workflow.py        # MODIFY — sub-agent spec resolution via
                       #           root_task_id + agent_name
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
    """

    @property
    def name(self) -> str:
        return "spawn_sub_agents"

    def invoke(self, arguments: str) -> str:
        # 1. Parse arguments JSON. Contains:
        #    - agents: list of {name, input} (from LLM)
        #    - root_task_id: str (injected by caller)
        #    - agent_id: str (injected by caller)
        #    Malformed JSON → return error string to LLM:
        #      '{"error": "invalid arguments: <details>"}'
        #
        # 2. For each agent:
        #    a. Validate name is in sub_specs
        #       - Unknown name → return error string to LLM:
        #         '{"error": "unknown sub-agent: <name>"}'
        #       This is a tool-level error, not an exception.
        #       The LLM can retry with a corrected name.
        #       (LLM config validation happens at spec load time
        #       via the validator, not at invoke time.)
        #    b. Create conversation via get_conversation_store()
        #       with kind="sub_agent"
        #    c. Append user input message
        #    d. Create task via get_task_store().create()
        #       - agent_id = args["agent_id"] (root registered agent)
        #       - agent_name = sub-agent name (e.g. "researcher")
        #       - root_task_id = args["root_task_id"]
        #    e. task_store.start(task_id)
        #       Workflow reads its own task row, sees root_task_id
        #       is set, uses agent_name to find sub-agent spec via
        #       recursive search through the root spec tree
        #
        # NOTE: Steps 2b-2e are not transactional. A crash between
        # create and start leaves an orphaned QUEUED task. This is
        # acceptable: the parent is a DBOS workflow, so DBOS replays
        # invoke() on recovery, creating a fresh task. The orphan is
        # inert. Same gap exists for top-level task creation today.
        #
        # 3. Return JSON: {response_ids: [...]}
        ...


class CollectTool(Tool):
    """
    Wait for spawned sub-agent tasks to complete and return
    their results.
    """

    @property
    def name(self) -> str:
        return "collect_sub_agents"

    def invoke(self, arguments: str) -> str:
        # 1. Parse arguments: response_ids list + optional timeout
        # 2. Calculate effective timeout:
        #    min(explicit_timeout, remaining_parent_execution_time)
        # 3. For each task_id:
        #    a. retrieve_workflow(task_id) → handle
        #    b. handle.get_result(timeout=remaining)
        #       (sync DBOS primitive — blocks workflow thread)
        #    c. get_task_store().get(task_id) → extract final output
        #       text from task.output (type == "output_text" items)
        # 4. Return JSON with results (status + output per task)
        # 5. Timed-out tasks get error string as output
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

    self._tools["spawn_sub_agents"] = SpawnTool(sub_specs=sub_specs)
    self._tools["collect_sub_agents"] = CollectTool()
```

**No changes to `ToolManager.__init__`** — it does not receive
`root_task_id` or `agent_id`. Instead, the agent loop injects these
into the `arguments` JSON string before calling `tool.invoke()`:

```python
# In _handle_tool_calls(), before dispatching to SpawnTool:
if tool_name == "spawn_sub_agents":
    args = json.loads(llm_arguments)
    # Inject server-side context that the LLM doesn't know about.
    # The LLM's tool schema does NOT include these fields — they
    # are invisible to the LLM and injected by the agent loop.
    #
    # root_task_id: the TOP-LEVEL task. For a top-level parent,
    #   this is the parent's own task_id. For a nested sub-agent
    #   spawning children, this is propagated from the parent's
    #   task.root_task_id — always points to the original root.
    #   SpawnTool passes this through to task_store.create().
    #
    # agent_id: the registered agent ID (always the root agent).
    #   Sub-agents are not registered in the agent store — they
    #   are nested specs within the root agent's spec tree.
    #   SpawnTool passes this through so the sub-agent's workflow
    #   can load the root spec and find the sub-agent by name.
    if task.root_task_id is not None:
        args["root_task_id"] = task.root_task_id
    else:
        args["root_task_id"] = task.id
    args["agent_id"] = agent_id
    llm_arguments = json.dumps(args)
tool.invoke(llm_arguments)
```

SpawnTool parses `root_task_id` and `agent_id` from the arguments
alongside the LLM-provided `agents` array. This keeps the Tool ABC
unchanged (`invoke(self, arguments: str) -> str`) and ToolManager
generic — no special-casing in the constructor or dispatch.

### 3. `spec/validator.py` — Validation rules

Add three checks in `_validate_sub_agents()`:

1. **LLM required**: for each name in `tools.agents`, the
   corresponding sub-agent spec must have a non-None `llm` block with
   a non-empty `model`.
2. **Unique names**: sub-agent names must be unique across the entire
   spec tree (not just within one level). This enables flat lookup by
   name for spec loading — see "Sub-agent spec loading" above.
3. **No dots in names**: agent names must not contain `.` (dot). The
   dot is reserved as the delimiter in the `model` field on
   tunneled output items (e.g. `"orchestrator.researcher"`).

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

### 4. Entity changes

**`Task` dataclass** — add one field:

```python
root_task_id: str | None = None
```

Docstring: `:param root_task_id: ID of the top-level task that
initiated this sub-agent's spawn tree, or ``None`` for top-level
tasks, e.g. ``"task_abc123"``.`

**`Conversation` dataclass** — add one field:

```python
kind: str = "default"
```

Docstring: `:param kind: Conversation type. ``"default"`` for
user-initiated, ``"sub_agent"`` for sub-agent execution
conversations.`

### 5. Store interface changes

**`TaskStore.create()`** — add `root_task_id` parameter:

```python
@abstractmethod
def create(
    self,
    conversation_id: str,
    agent_id: str,
    agent_name: str,
    previous_response_id: str | None = None,
    background: bool = False,
    root_task_id: str | None = None,  # NEW
) -> Task:
```

`:param root_task_id: ID of the top-level task that initiated this
sub-agent's spawn tree. ``None`` for top-level tasks,
e.g. ``"task_abc123"``.`

Added at the end of the parameter list for backward compatibility.
`SqlAlchemyStore.create()` maps this directly to the new
`root_task_id` column.

**`TaskStore.wait()`** — add `timeout` parameter:

```python
@abstractmethod
async def wait(
    self,
    task_id: str,
    timeout: float | None = None,  # NEW
) -> Task:
```

`:param timeout: Maximum seconds to wait. ``None`` blocks
indefinitely (current behavior). If the deadline expires,
returns the task in its current (non-terminal) state instead
of raising.`

**`ConversationStore.create_conversation()`** — add `kind` parameter:

```python
@abstractmethod
def create_conversation(
    self,
    kind: str = "default",  # NEW
) -> Conversation:
```

`:param kind: Conversation type. ``"default"`` for user-initiated,
``"sub_agent"`` for sub-agent execution conversations.`

**`ConversationStore.list_conversations()`** — add `kind` filter:

```python
@abstractmethod
def list_conversations(
    self,
    limit: int = 20,
    after: str | None = None,
    before: str | None = None,
    order: str = "desc",
    kind: str | None = "default",  # NEW — None = no filter
) -> PagedList[Conversation]:
```

`:param kind: Filter to conversations of this kind. Exact match.
``"default"`` returns only user-initiated conversations.
``"sub_agent"`` returns only sub-agent conversations. ``None``
disables the filter and returns all conversations regardless of
kind. Defaults to ``"default"`` so sub-agent conversations are
hidden from standard listings.`

The type is `kind: str | None = "default"` — nullable to support
the "all" case.

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
- Client-side tools are tunneled via `root_task_id` (see
  `SUBAGENT_WORKFLOW.md`)
- Each sub-agent streams output via its own SSE stream

### Step 4: Parent collects results

The parent's next LLM turn calls `collect_sub_agents`:

```json
{"tool_calls": [{"call_id": "call_collect_sub_agents", "name": "collect_sub_agents",
  "arguments": "{\"response_ids\": [\"resp_child1\", \"resp_child2\"]}"}]}
```

`CollectTool.invoke()`:
1. Calls sync DBOS `retrieve_workflow()` + `handle.get_result()` per task
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

## Implementation Order

### Step 1: Schema and entity changes

1. Add `root_task_id` column to existing `SqlTask` in `db_models.py`
2. Add `kind` column to existing `SqlConversation` in `db_models.py`
3. Add `pending_tool_calls` table to `db_models.py`
4. Update `Task` dataclass: add `root_task_id: str | None = None`
5. Update `Conversation` dataclass: add `kind: str = "default"`
6. Define `PendingToolCall` dataclass (see `SUBAGENT_WORKFLOW.md`)

### Step 2: Store interface changes

1. `TaskStore.create()` — add `root_task_id` parameter
2. `TaskStore.wait()` — add `timeout` parameter
3. `TaskStore` — add 3 pending tool call methods
4. `ConversationStore.create_conversation()` — add `kind` parameter
5. `ConversationStore.list_conversations()` — add `kind` filter
6. SqlAlchemy implementations for all of the above

### Step 3: Spawn/collect tools

1. `tools/builtins/spawn.py` — `SpawnTool` + `CollectTool`
2. `tools/manager.py` — add `_register_sub_agent_tools()`
3. Wire `SpawnTool` to create tasks via `get_task_store()`
   - Set `root_task_id` (propagated: parent's `root_task_id` if set,
     else parent's own task ID — always points to top-level)
   - Set `agent_name` to sub-agent name
   - Spawned sub-agents use `_run_agent_loop()` (single execution path)
4. Wire `CollectTool` to wait via sync DBOS polling loop with
   `DBOS.sleep()` + deadline check, extract results
5. Argument injection in `_handle_tool_calls()` for `root_task_id`
   and `agent_id` (see pseudocode in "ToolManager" section)

### Step 4: Validation

1. `spec/validator.py` — require `llm.model` on callable sub-agents,
   unique names across entire spec tree, no dots in agent names

### Step 5: Client-side tool tunneling

1. Park branch in `_handle_tool_calls()` — detect client-side tools
   when `root_task_id IS NOT NULL`, write routing rows, publish to
   root's output, enter park loop
2. `PATCH /v1/responses/{id}` endpoint — submit tool results
3. `action_required` function_call status enum extension

### Step 6: Testing

1. **Unit**: `SpawnTool` creates tasks with correct `root_task_id`
2. **Unit**: `CollectTool` waits and returns results, timeout returns
   partial results
3. **Unit**: `complete_pending_tool_call` returns correct status for
   not_found / completed / already_completed / sub_agent_done cases
4. **Validation**: sub-agent without `llm.model` fails, duplicate
   names fail, dots in names fail
5. **Integration**: spawn one sub-agent, collect result
6. **Integration**: spawn two sub-agents, collect both
7. **Integration**: collect with timeout, partial results
8. **Integration**: sub-agent with client-side tool, full park/PATCH
   cycle
9. **Integration**: multi-round parking (sub-agent parks twice)
10. **Integration**: parallel sub-agents with independent client tools
11. **Verify**: sub-agent conversations have `kind="sub_agent"`
12. **Verify**: nested spawns propagate `root_task_id` to root

---

## Open Questions

### Q1: Parent timeout while blocked on `collect_sub_agents`

The parent blocks on `collect_sub_agents` while sub-agents run. If a sub-agent
hangs (e.g. infinite LLM loop, or waiting on a client-side tool call
that never comes), the parent hangs too.

Mitigation: `collect_sub_agents` defaults to the parent's remaining execution
timeout. If the parent's execution timeout expires, the parent's agent
loop terminates with `status: "incomplete"`. The sub-agent tasks
continue running independently (they have their own execution timeouts).

Future optimization: async waiting (release the DBOS thread while
blocked on `collect_sub_agents`, re-acquire when sub-agents complete). Requires
changes to how DBOS manages workflow threads.

---

## Not Yet

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
