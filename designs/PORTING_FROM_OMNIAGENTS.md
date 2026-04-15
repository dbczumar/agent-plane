# Porting OmniAgents into Agent-Plane

This document analyzes the feature gap between
`universe2/experimental/matei_data/agent_framework` (OmniAgents) and
`agent-plane`, identifies design philosophy rifts that require special
attention, and proposes a phased porting strategy.

---

## 1. Executive Summary

OmniAgents and agent-plane share the same high-level goal (declarative
agent authoring + multi-harness execution) but diverge sharply in three
areas:

1. **State model** -- OmniAgents is in-memory/ephemeral; agent-plane is
   persisted/durable (DBOS + SQLAlchemy).
2. **Guardrails** -- OmniAgents has a full policy/label system for
   information-flow control; agent-plane has none.
3. **Environment isolation** -- OmniAgents provides per-session OS
   environments with Landlock/seccomp sandboxing and tmux terminals;
   agent-plane provides sandboxed code execution as a builtin tool but
   no per-session environment abstraction.

The porting effort is NOT a code transplant. The two systems share very
few lines of portable code. Instead, this is a **feature-level port**:
understanding what OmniAgents can do that agent-plane cannot, and
designing agent-plane-native implementations of those capabilities.

---

## 2. Architecture Comparison

| Dimension | OmniAgents | Agent-Plane |
|---|---|---|
| **Agent definition** | Single YAML file, inline tools/policies/labels | Directory image: `config.yaml` + `AGENTS.md` + `tools/` + `agents/` |
| **State model** | In-memory `Session` object; optional JSON snapshot | Persisted via SQLAlchemy stores + DBOS durable workflows |
| **Session concept** | Rich runtime object (history, connection, memory, labels, os_env, terminals, runtime, async tasks, named child sessions) | No runtime session object; state reconstructed each iteration from stores |
| **Agent loop** | `Session.stream_turn()` async generator | `agent_execution_workflow()` DBOS workflow with `@step` checkpoints |
| **Tool dispatch** | In-process, session-aware (tools can read/write session state) | DBOS `@step`, isolated (`ToolContext` has only task_id, agent_id, workspace) |
| **Sub-agent model** | Named sessions (send/peek/list), Connection-based message passing | DBOS workflow spawning (SpawnTool), polling (CheckSubAgentsTool), auto-collect |
| **Guardrails** | PolicyEngine with 5 policy types, 4 phases, label schema, monotonic propagation | None |
| **OS environment** | Per-session sandboxed subprocess (read/write/edit/shell), Landlock/seccomp, fork | `code_sandbox` builtin tool, `srt`-based sandboxing |
| **Terminal support** | tmux-based terminal management (5 system tools), per-instance isolation | None |
| **Async tools** | Background tasks with inbox (`sys_call_async`/`sys_read_inbox`) | Parallel tool calls via `asyncio`, but no background task system |
| **Cancellation** | Phase-aware state machine (MODEL/TOOL), executor interrupt, tool cancel, pending notice | Task-level cancel via store, less granular |
| **Streaming** | Direct async generator yield | Dual-channel: DBOS durable stream + in-memory Queue bridge |
| **Compaction** | None | 3-layer: surgical clearing -> LLM summarization -> truncation |
| **Client-side tools** | None (all tools execute server/session-side) | Full tunneling: server returns function_call, client executes, PATCHes result |
| **Multi-provider LLM** | Primarily Databricks (OpenAI-compatible) + env var fallbacks | 7 adapters: OpenAI, Anthropic, Gemini, Bedrock, Vertex, Databricks, OpenAI-compatible |
| **Server** | Starlette (raw ASGI): REST + WebSocket + SSE, Vercel AI SDK event protocol | FastAPI (Starlette + Pydantic): REST + SSE, OpenAI-compatible event protocol |
| **CLI** | argparse + prompt_toolkit + rich, debug overview mode, mascots | Click + rich, REPL with slash commands, UI SDK rendering pipeline |
| **Persistence** | JSON snapshots in `~/.omniagents/sessions/` | Alembic-managed SQLite/PostgreSQL |
| **Durability** | None (crash = lost state) | DBOS crash recovery (replay from checkpoints) |
| **Memory** | In-memory KV store with scopes (per_session, per_user, cross_user) | None (conversation history is the memory) |
| **Credentials** | Token-based with scopes and attenuation | API keys in spec config |
| **Tracing** | MLflow spans (AGENT, CHAT_MODEL, TOOL, GUARDRAIL) | Observability design doc exists but system not implemented |
| **Skills** | SkillTool loads docs into context on demand | Skill system with `load_skill`/`read_skill_file` tools |

---

## 3. Feature Gap Inventory

### 3.1 Major Gaps (new subsystems required)

#### G1: Policy and Label System

OmniAgents has a full guardrails engine:
- **5 policy types**: `FunctionPolicy` (Python callable), `BuiltinPolicy`
  (rate limits), `PromptPolicy` (LLM-evaluated), `CascadePolicy`
  (ordered chain), `LabelPolicy` (condition on labels)
- **4 evaluation phases**: `input`, `output`, `tool_call`, `tool_result`
- **3 actions**: `ALLOW`, `ASK` (interactive approval), `DENY`
- **Label schema**: monotonic constraints (`max`, `min`, `none`),
  parent-child propagation, information flow control
- **Ask handler**: interactive approval with timeout

Agent-plane has **zero guardrails infrastructure**. This is the single
largest feature gap.

Key files: `omniagents/policies.py`, `omniagents/datamodel.py`
(LabelSchemaRule), `omniagents/session.py` (4 evaluation points,
`_apply_root_label_update`, `_propagate_child_labels`).

#### G2: OS Environment Abstraction

OmniAgents provides each agent session with a sandboxed filesystem and
shell:
- `CallerProcessOSEnvironment`: sandboxed helper subprocess for
  read/write/edit/shell
- **Landlock + seccomp**: Linux kernel-level filesystem restriction +
  syscall filtering (no root required)
- **Fork mode**: copies working directory for isolated mutation
- **Cross-session mounts**: parent mounts child env at path prefix
  (`sys_os_mount_env`)
- 6 system tools: `sys_os_read`, `sys_os_write`, `sys_os_edit`,
  `sys_os_shell`, `sys_os_mount_env`, `sys_os_unmount_env`

Agent-plane has `code_sandbox` as a builtin tool and `srt`-based
sandboxing for local Python tools, but no per-session environment
abstraction.

Key files: `omniagents/os_env.py`, `omniagents/sandbox.py`,
`omniagents/landlock_sandbox.py`, `DESIGN_OS_ENV_ACCESS.md`.

#### G3: Terminal Environment (tmux)

OmniAgents provides agents with interactive terminal sessions:
- `TerminalInstance`: per-instance tmux server with isolated socket
- 5 system tools: `sys_terminal_launch`, `sys_terminal_send`,
  `sys_terminal_read`, `sys_terminal_list`, `sys_terminal_close`
- Sandbox wrapping of CLI processes via `create_exec_launcher`
- Fork mode for terminal working directories

Agent-plane has no terminal concept.

Key files: `omniagents/terminal.py`, `DESIGN_TMUX_AGENTS.md`.

#### G4: Background Tool Execution

OmniAgents allows agents to fire tools in the background and collect
results later:
- `sys_call_async`: starts a tool in background, returns `handle_id`
- `sys_cancel_async`: cancels a background tool by handle
- `sys_read_inbox`: reads completed results from inbox (up to 32 items,
  16KB per read)
- Session sleeps (via `_wake_event`) until a result arrives -- no
  wasted LLM calls between completions
- Framework notice injection when unread items exist

Agent-plane has parallel tool calls within a single turn but no
background tool system. The agent loop blocks until all tool calls
in a turn complete before proceeding to the next LLM call.

Key files: `omniagents/session.py` (lines ~2100-2300: `_async_call`,
`_run_async_tool_call`, `_read_inbox`).

#### G5: Named Child Sessions

OmniAgents allows persistent named sessions with child agents:
- `sys_session_send`: sends async turn to a named child session
- `sys_session_peek`: inspects child session's recent events
- `sys_session_list`: lists all active named sessions
- `sys_session_cancel_turn`: cancels a child's current turn
- `sys_session_close`: closes a named session
- `max_sessions` limit per agent tool
- Results flow back via inbox (`sys_read_inbox`)

**Why this matters**: it changes the fundamental metaphor of
sub-agents. Spawn-and-collect treats sub-agents as functions (call
with input, get output). Named sessions treat them as actors
(maintain an ongoing relationship). The actor model is more
expressive -- it enables iterative refinement, parallel
exploration, and long-running collaboration where each sub-agent
maintains accumulated context across many parent turns. Without
named sessions, each "send more work to the coder" requires a
fresh spawn that re-discovers the codebase from scratch.

Agent-plane's sub-agents are DBOS workflows accessed via
`spawn_sub_agents`/`check_sub_agents`/`cancel_sub_agent`. Each
spawn creates a new conversation with no continuity to prior work.

Key files: `omniagents/session.py` (lines ~2400-2800:
`get_or_create_agent_session`, `_ManagedAgentSession`,
`_run_agent_session_turn`).

### 3.2 Medium Gaps (extensions to existing subsystems)

#### G6: Phase-Aware Cancellation

OmniAgents tracks execution phase (`IDLE`, `MODEL`, `TOOL`) and
provides:
- Per-phase cancel behavior (interrupt executor vs cancel tool)
- `_active_tool_call` state tracking with tool-specific cancel functions
- `CancellableFunctionTool` with explicit `runner.cancel()` protocol
- Pending cancellation notice injected at start of next turn
- `CancelResult` with status/phase/reason

Agent-plane has task-level cancellation via the store but no phase-aware
cancel within the agent loop.

#### G7: Memory System

OmniAgents has an in-memory KV store with scopes:
- `per_session`: ephemeral within a session
- `per_user`: persists across sessions for a user
- `cross_user`: shared across all sessions
- `get/set/delete/list_keys/search` operations

Agent-plane has no memory abstraction. Conversation history serves as
implicit memory.

Key file: `omniagents/datamodel.py` (Memory class, line 139).

#### G8: Connection / Bidirectional Messaging

OmniAgents uses `Connection` objects (asyncio Queue pairs) for
bidirectional messaging:
- Primary connection links CLI/server to session
- Multiple typed connections (`chat`, `internal`, `api`)
- `HandoffTool` transfers connection to another agent

Agent-plane uses HTTP request/response + SSE for communication. There
is no persistent bidirectional channel.

#### G9: Credentials Abstraction

OmniAgents has a `Credentials` dataclass:
- `scopes`, `principal`, `expires_at`
- `attenuate()` for scope narrowing in child sessions
- Credential propagation through agent hierarchy

Agent-plane uses API keys in the spec config with no runtime credential
object.

Key file: `omniagents/datamodel.py` (Credentials class, line 178).

#### G10: In-Process Python Runtime

OmniAgents has `sys_runtime_execute` which runs Python code in-process
with access to the session object. Agent-plane has `code_sandbox` which
runs code in an isolated subprocess.

Key file: `omniagents/runtime.py`.

#### G11: Additional Executors (Codex, Pi)

OmniAgents supports two executors agent-plane lacks:
- **CodexExecutor**: long-lived `codex app-server` subprocess,
  JSONL protocol, persistent thread across turns
- **PiExecutor**: `pi --mode rpc` subprocess, TCP socket bridge for
  tools, generated JavaScript extension

Agent-plane has `DefaultExecutor`, `ClaudeAgentsExecutor`,
`AgentsSdkExecutor`, `RemoteExecutor`.

#### G12: InheritedTool / Tool Inheritance

OmniAgents allows child agents to inherit tools from their parent:
```yaml
tools:
  search: inherit
```
Resolved during session init from the parent's tool registry.

Agent-plane does not support tool inheritance. Sub-agents are fully
self-contained (spec self-containment principle).

#### G13: MLflow Tracing Integration

OmniAgents has `TracingContext` with span types: `AGENT`, `CHAT_MODEL`,
`TOOL`, `GUARDRAIL`. Spans are auto-created in the session loop.
Agent-plane has an observability design doc but no implementation.

Key file: `omniagents/tracing.py`.

#### G20: Client-Side Interactive Processes

Agent-plane's client-side Bash tool is one-shot: run a command, get
stdout/stderr, done. This covers ~95% of coding agent needs but cannot
handle:

- **Interactive programs requiring a PTY** -- `gcloud auth login`, SSH
  sessions, `docker exec -it`, installers that prompt yes/no. These
  hang or fail in a non-interactive subprocess.
- **Long-running processes with monitoring** -- start `npm run dev`,
  come back later to check its output, send it Ctrl-C. One-shot Bash
  loses the process after it returns.
- **Persistent shell state** -- cd, env vars, virtualenv activation
  persist in a terminal session. One-shot Bash starts fresh each time.

This is an agent-plane-specific gap (OmniAgents doesn't have
client-side tools at all, so the comparison is moot). But since
agent-plane's coding agents rely heavily on client-side execution,
these gaps matter.

Key files: `agent_plane/client_tools/`, `agent_plane/tools/client_specified/`.

### 3.3 Minor Gaps (small features or UI differences)

#### G14: Session Snapshot/Restore

OmniAgents can serialize sessions to JSON and restore them
(`session_store.py`). Agent-plane persists everything in the database
natively, so this is less necessary -- but the CLI `resume` command
uses it.

#### G15: CLI Debug Overview Mode

OmniAgents CLI has a debug overlay (ctrl-O) showing all active sessions
with scrollable output and session switching. Agent-plane's REPL has
slash commands but no debug overlay.

#### G16: Vercel AI SDK Data Stream Protocol

OmniAgents' server supports the Vercel AI SDK Data Stream Protocol
(`stream_protocol.py`) for SSE. Agent-plane uses OpenAI-compatible SSE
events.

#### G17: WebSocket Support

OmniAgents' server supports WebSocket for chat. Agent-plane uses SSE
only.

#### G18: Mascots

OmniAgents generates procedural symmetric ASCII art mascots. Purely
cosmetic.

#### G19: Unity Catalog Functions

OmniAgents supports UC functions as tools (`uc_tools.py`). Agent-plane
does not.

---

## 4. Design Philosophy Rifts

These are fundamental architectural differences that make naive code
porting impossible. Each requires a deliberate design decision.

### Rift 1: Ephemeral In-Memory State vs Persisted Durable State

**OmniAgents**: Session is a rich in-memory object. Labels, memory,
connections, async tasks, named sessions -- all live as Python objects
in the session's heap. Crash = total state loss.

**Agent-plane**: All state is in the database. The workflow function
reconstructs context from stores each iteration. DBOS checkpoints
every LLM call and tool call. Crash = replay from last checkpoint.

**Why this matters for porting**: Every OmniAgents feature that relies
on in-memory session state (policies reading `session.labels`, async
tasks stored in `_async_tasks` dict, named sessions in
`_agent_sessions`) must be redesigned to either:
- (a) persist in a store (new tables, new store interfaces), or
- (b) be reconstructed from conversation history each iteration, or
- (c) live in a `ContextVar`-scoped runtime container within the DBOS
  workflow (acceptable for ephemeral state that need not survive
  crashes)

**Recommendation**: Create a `RuntimeState` context-scoped object that
holds ephemeral per-workflow state (labels, policy engine, active tool
calls). For features that MUST survive crashes (label values, policy
verdicts), add persistence. For features that can be reconstructed
(named sessions from conversation items), use replay.

### Rift 2: Session-Aware Tools vs Isolated Tool Steps

**OmniAgents**: System tools (`sys_*`) have direct access to the
session object. `sys_os_read` calls `self.os_env.read()`.
`sys_call_async` creates `asyncio.Task` in the session's event loop.
`sys_session_send` accesses `self._agent_sessions`.

**Agent-plane**: Tools are DBOS `@step` functions receiving only
`ToolContext(task_id, agent_id, workspace)`. They cannot access
runtime state, other tools, or the agent loop.

**Why this matters**: Porting sys_ tools requires either:
- (a) expanding `ToolContext` to include a runtime state reference
  (breaks DBOS step isolation), or
- (b) implementing system tools as workflow-level operations (not
  tool calls) that the agent loop dispatches directly, or
- (c) using a shared `ContextVar` that system tools access (works
  because DBOS steps run on the workflow's thread)

**Recommendation**: Option (b) for "system" tools that need runtime
state. The workflow dispatches these directly instead of going through
`_call_tool @step`. This matches OmniAgents' `_DIRECT_TOOL_NAMES`
pattern. Regular tools continue through the `@step` path.

### Rift 3: Async Generator vs DBOS Workflow Thread

**OmniAgents**: `stream_turn()` is an async generator. The caller
iterates it, receiving events. The session owns an asyncio event loop.
Background tasks (`asyncio.create_task`) run concurrently. The session
sleeps (via `_wake_event`) until a result arrives or the user sends a
message -- no wasted LLM calls.

**Agent-plane**: The DBOS workflow runs on a sync thread. Async
bridging happens via `EventLoopThread` for MCP and `asyncio.to_thread`
for blocking tools. There is no persistent event loop for background
tasks.

**Why this matters**: The async tool system (G4) and named sessions
(G5) depend on `asyncio.create_task` for background execution. This
cannot work in the DBOS thread model.

**Recommendation**: Use DBOS child workflows for background tools,
with signal-based wake to avoid wasted LLM calls.

Each `run_in_background("tool_name", {...})` spawns a minimal DBOS
child workflow that runs the tool as a single `@step` (no agent loop,
no LLM call). When the child completes, it signals the parent via
`DBOS.send(parent_workflow_id, result, topic="bg_tool_complete")`.

The parent workflow uses `dbos_recv_async(topic="bg_tool_complete")`
to sleep until a result arrives -- the same mechanism client-side tool
parking already uses. No polling loop, no wasted LLM calls.

The agent loop behavior when background tools are pending:

1. LLM produces a response with no tool calls.
2. Background tools are still running.
3. **The workflow does NOT complete the task.** Instead it waits
   (via `dbos_recv_async`) for the next background tool to finish.
   This wait is free -- no LLM calls, no polling. The SSE stream
   stays open.
4. A background tool completes. Its result is injected as a system
   message.
5. The LLM gets a turn to process the result. It can kick off more
   background work, produce intermediate text, or produce a final
   response.
6. Repeat from step 2 if more tools are pending. Complete the task
   only when the LLM is done AND nothing is pending.

This gives the LLM control over when to work and what to do with
partial results, without burning LLM calls on "still waiting" loops.
The workflow is idle (zero cost) between completions.

This is durable (child workflows survive crashes), reuses existing
DBOS primitives (`dbos_recv_async`, `DBOS.send`), and matches the
existing client-side tool parking pattern.

### Rift 4: Agent Definition Format

**OmniAgents**: Single YAML file with everything inline. Tools are
Python callable references. Policies are inline YAML blocks. Labels
are top-level fields.

**Agent-plane**: Directory-based agent image. Tools are separate files
(`tools/python/*.py`). Instructions are `AGENTS.md`. Sub-agents are
directories.

**Why this matters**: The OmniAgents YAML format is simpler for quick
prototyping. Agent-plane's directory format is better for production
(separate concerns, version control, code review).

**Recommendation**: Extend agent-plane's `config.yaml` to support
policy and label declarations. Add a `policies:` section and a
`labels:` section. Do NOT support the OmniAgents single-YAML format --
it violates agent-plane's spec self-containment principle (tools as
code references imply server-side Python paths). Instead, express
policies within the existing agent image format:
```yaml
# config.yaml additions
policies:
  - name: content_filter
    type: prompt
    on: [input, output]
    prompt: policies/content_filter.md  # bundled file
  - name: rate_limit
    type: builtin
    on: [tool_call]
    config: {max_tool_calls_per_turn: 10}

labels:
  sensitivity: public
  schema:
    sensitivity:
      values: [public, internal, confidential, restricted]
      monotonic: max
```

### Rift 5: Sub-Agent Communication Model

**OmniAgents**: Named sessions with bidirectional messaging. Parent
sends input, peeks at progress, sends more input. Child is a
long-lived session across turns.

**Agent-plane**: Sub-agents are isolated DBOS workflows. Spawned once,
polled for completion, auto-collected. Steering allows one-way message
injection. No persistent sessions across tasks.

**Why this matters**: OmniAgents' named sessions enable interactive
sub-agent patterns (send-peek-send loops, collaborative workflows).
Agent-plane's spawn-poll-collect is more batch-oriented.

**Recommendation**: Named sessions are a significant enhancement to
agent-plane's sub-agent model. They should be implemented as a new
concept in the sub-agent system:
- A `NamedSession` store tracks persistent sub-agent conversations
- `session_send` reuses an existing conversation (no new workflow if
  agent is idle, else steers the running workflow)
- `session_peek` reads recent conversation items
- `session_close` marks the session as closed

This builds on the existing steering infrastructure rather than
replacing it.

### Rift 6: Spec Self-Containment vs Runtime Detection

**OmniAgents**: Freely reads environment variables at runtime
(`DATABRICKS_HOST`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`). Falls
back between credential sources. Detects server capabilities.

**Agent-plane**: Principle #1 is spec self-containment. All
configuration comes from the spec. `${ENV_VAR}` expanded at deploy
time, not runtime.

**Why this matters**: OmniAgents features that rely on runtime env var
detection (executor factory, credential resolution) cannot be ported
as-is. They must be redesigned to use spec-declared configuration.

**Recommendation**: All ported features must declare their
configuration in the spec. No runtime detection. This means:
- Executor selection: spec `executor.type`, not runtime harness
  detection
- Credentials: spec `llm.connection`, not `~/.databrickscfg`
- Sandbox configuration: spec `tools.sandbox`, not OS detection

---

## 5. Per-Feature Porting Strategy

### Phase 1: Guardrails Foundation (G1 -- Policy/Label System)

This is the highest-value, highest-complexity port. It touches the
agent loop, the spec format, and the streaming output.

**New files:**
- `agent_plane/policies/` package:
  - `types.py`: `PolicyAction` enum, `PolicyResult` dataclass,
    `PolicyPhase` literal
  - `base.py`: `Policy` ABC with `evaluate(content, phase) -> PolicyResult`
  - `function_policy.py`: wraps a Python callable
  - `builtin_policy.py`: rate limits, framework-managed policies
  - `prompt_policy.py`: LLM-evaluated guardrails (uses `agent_plane.llms.Client`)
  - `cascade_policy.py`: ordered policy chain
  - `label_policy.py`: condition-on-labels policy
  - `engine.py`: `PolicyEngine` -- iterates policies, accumulates labels, short-circuits
- `agent_plane/labels/` package:
  - `schema.py`: `LabelSchemaRule` with `allows()`, `merged_with_child()`
  - `state.py`: `LabelState` -- per-workflow label storage with schema validation

**Spec changes (`config.yaml`):**
```yaml
policies:
  - name: <string>
    type: function | builtin | prompt | cascade | label
    on: [input, output, tool_call, tool_result]
    # type-specific fields...
labels:
  <key>: <initial_value>
  schema:
    <key>:
      values: [ordered, list, of, valid, values]
      monotonic: max | min | none
```

**Spec types changes (`agent_plane/spec/types.py`):**
- Add `PolicySpec` and `LabelConfig` to `AgentSpec`

**Workflow changes (`workflow.py`):**
- Create `PolicyEngine` at workflow start
- Insert 4 evaluation points:
  1. Before processing user input (input phase)
  2. Before executing each tool call (tool_call phase)
  3. After each tool result (tool_result phase)
  4. After final LLM response (output phase)
- On DENY: persist a system message, emit SSE event, continue/halt
- On ASK: emit SSE event, await client response (new SSE event type),
  or use timeout-based auto-deny
- Label state in `ContextVar`, accessible from workflow-level
  system tools

**SSE changes:**
- New event types: `response.policy.denied`, `response.policy.ask`,
  `response.policy.label_changed`

**Store changes:**
- Labels persisted as conversation metadata (survives compaction)
- Policy verdicts logged as conversation items (type: `policy_verdict`)

**Estimated complexity**: HIGH. Touches spec, workflow, streaming,
stores.

**Key risk**: PromptPolicy requires an LLM call within the agent loop.
In OmniAgents this is a fresh executor instance. In agent-plane, this
must be a separate `@step` or an inline LLM call outside the main
executor turn.

### Phase 2: OS Environment (G2)

**New files:**
- `agent_plane/os_env/` package:
  - `base.py`: `OSEnvironment` ABC (read, write, edit, shell, close)
  - `local.py`: `LocalOSEnvironment` -- sandboxed subprocess (port
    of `_HelperProcessClient`)
  - `sandbox.py`: `SandboxPolicy` dataclass + activation logic
  - `landlock.py`: Linux Landlock/seccomp backend (port of
    `landlock_sandbox.py`)

**Tool changes:**
- New system tools: `os_read`, `os_write`, `os_edit`, `os_shell`
- These are workflow-level dispatched (Rift 2 recommendation),
  not `@step` tools, because they need the per-workflow
  `OSEnvironment` instance

**Spec changes:**
```yaml
os_env:
  type: local  # only option initially
  cwd: .
  fork: false
  sandbox:
    read_paths: [.]
    write_paths: [.]
    allow_network: true
```

**Workflow changes:**
- Create `OSEnvironment` at workflow start (in `_run_agent_loop`)
- Close in finally block
- System tool dispatch: intercept `os_*` tool names before
  `_call_tool @step`

**Estimated complexity**: MEDIUM-HIGH. The helper subprocess and
sandbox are self-contained modules, but integrating into the DBOS
workflow requires care around subprocess lifecycle.

**Key risk**: Landlock/seccomp only works on Linux. Agent-plane must
gracefully handle macOS/non-Linux (where OmniAgents simply skips
sandboxing). This is acceptable per RuntimeCaps pattern.

### Phase 3: Background Tools and Named Sessions (G4, G5)

These two features are tightly coupled in OmniAgents (named session
results arrive in the same inbox as async tool results).

#### Background Tools (G4)

**Core mechanism**: DBOS child workflows with signal-based wake.

**New builtin tool**: `run_in_background`
```
run_in_background(tool_name: str, arguments: dict) -> {handle_id: str}
```
Returns immediately. The tool executes in a child DBOS workflow.

**New DBOS workflow**: `background_tool_workflow`
```python
@workflow()
async def background_tool_workflow(parent_task_id, tool_name, arguments):
    result = await _call_tool(...)    # existing @step
    DBOS.send(parent_task_id, {       # signal parent
        "handle_id": handle_id,
        "tool_name": tool_name,
        "result": result,
    }, topic="bg_tool_complete")
    return result
```

**Agent loop changes** (in `_run_agent_loop`):

Track `pending_bg_tools: set[str]` (child workflow IDs).

After LLM produces a response with no tool calls:
- If `pending_bg_tools` is empty: complete normally (existing path).
- If `pending_bg_tools` is non-empty:
  1. Persist the LLM's text response as an assistant message (stream
     it to the user -- they see intermediate output).
  2. Wait for the next completion:
     `result = await dbos_recv_async(topic="bg_tool_complete")`.
     This is zero-cost (no LLM calls, no polling).
  3. Remove from `pending_bg_tools`. Inject result as a system
     message: `"[Background tool completed: {tool_name}]\n{result}"`.
  4. Give the LLM a turn. It processes the result, may kick off
     more background work, or produce a final response.
  5. Loop back. Task completes only when LLM is done AND
     `pending_bg_tools` is empty.

During the `dbos_recv_async` wait, steering also wakes the workflow
(user sends a message). Handle both wake sources.

**Additional tools**:
- `check_background() -> list[{handle_id, tool_name, status}]`:
  Non-blocking status check. Queries task store for child workflow
  statuses.
- `cancel_background(handle_id)`: Cancels a child workflow via
  task store.

**Key properties**:
- Durable: child workflows survive crashes, parent replays from
  checkpoint and re-waits.
- Zero wasted LLM calls: workflow sleeps between completions.
- Reuses existing primitives: `DBOS.send`/`dbos_recv_async` (same
  as client-side tool parking), `_call_tool @step`, task store.
- SSE stream stays open: user sees intermediate text and tool
  completion events in real-time.

#### Named Sessions (G5)

A named session is a persistent conversation with a sub-agent that
the parent refers to by a human-readable key (e.g. `"coder:auth"`).
Unlike spawn-and-collect, named sessions accumulate context across
multiple parent turns -- the sub-agent remembers prior work without
needing to re-discover it. This enables iterative refinement,
parallel exploration, and long-running collaboration patterns.

**Data model**: no new store, no new entity type. A named session is
just a conversation with one additional nullable column.

```python
@dataclass
class Conversation:
    id: str
    created_at: int
    updated_at: int
    title: str | None
    kind: str
    # new column:
    parent_conversation_id: str | None
```

- `parent_conversation_id` -- the owning conversation. Named
  sessions belong to the parent's **conversation**, not any
  particular task. Tasks are ephemeral (one per turn); conversations
  are the stable identity the sub-session persists against.
- `title` -- doubles as the session key for named sessions
  (e.g. `"coder:auth"`). System-set and immutable for sub-agent
  conversations, so using it as a lookup key is safe.

A conversation is a named session iff `parent_conversation_id IS NOT
NULL`. To "close" a session, set `parent_conversation_id = NULL`.
The conversation and its history remain (audit trail), but it's no
longer findable via the parent lookup. `max_sessions` enforcement
counts only conversations where `parent_conversation_id = mine`.

**New builtin tools** (all workflow-level dispatched, not `@step`,
because they need access to runtime state like `pending_sessions`):

- `session_send(tool, key, input)` -- three cases:
  1. **First send**: create a new conversation with
     `parent_conversation_id = <parent's conversation>` and
     `title = "<tool>:<key>"`. Append the user message. Spawn a
     new sub-agent task on the conversation. Add the task_id to
     `pending_sessions`. Return `{handle_id: <task_id>}`.
  2. **Session exists, sub-agent is running**: steer the message
     into the active task via `task_store.try_deliver()`. The
     active task_id is already in `pending_sessions`. Return
     `{handle_id: <active_task_id>}`.
  3. **Session exists, sub-agent is idle**: append the message to
     the existing conversation. Spawn a new task on the same
     conversation. Add the new task_id to `pending_sessions`.
     Return `{handle_id: <task_id>}`.
- `session_peek(tool, key)` -- read recent items from the named
  session's conversation. Plain query, no DBOS involvement.
- `session_list()` -- list all open named sessions for the parent
  conversation.
- `session_cancel_turn(tool, key)` -- cancel the active task via
  `task_store.cancel()`. The session stays open; the parent can
  send new work afterward.
- `session_close(tool, key)` -- cancel any active task, set
  `parent_conversation_id = NULL` on the child conversation.

**Completion delivery**: unified with background tools via the
`async_work_complete` topic. When a child sub-agent workflow
completes, it signals its parent:

- **If a parent task is currently running**: signal goes through
  `DBOS.send` -> parent's `dbos_recv_async` wake, result injected
  as a system message on the parent's current loop iteration.
- **If no parent task is running** (e.g. parent's task already
  ended, user hasn't sent the next message yet): append the result
  as a message on the parent conversation directly. The next parent
  task picks it up via `_load_initial_history` on startup. No
  orphaned signals, no separate inbox -- the conversation IS the
  inbox.

**Workflow-level state**:

```python
runtime_state.pending_sessions: set[str]  # set of active child task_ids
```

The same `pending_bg_tools` + `pending_sessions` pattern described
in the background tools section. The agent loop blocks task
completion while either set is non-empty and wakes on any
completion signal.

**Crash recovery**: on replay, reconstruct `pending_sessions` by
querying `task_store.list_active_tasks(root_task_id=self.task_id)`
filtered to tasks whose conversation has
`parent_conversation_id = self.conversation_id`. DBOS redelivers
any signals that arrived during the crash window.

**Spec changes**: optional `max_sessions` field on sub-agent
declarations to limit concurrent named sessions per sub-agent:

```yaml
tools:
  agents:
    - name: coder
      max_sessions: 3
    - name: researcher
      max_sessions: 1
```

Enforced in `session_send` case 1 (first send).

**Estimated complexity**: MEDIUM. One column on an existing table.
Five new builtin tools (workflow-level dispatched). Wake mechanism
shared with background tools. The most subtle piece is the
dual completion delivery path (steering into an active task vs
appending to the parent conversation), but both use existing
primitives.

### Phase 4: Terminal Environment (G3, G20)

Terminals replace `code_sandbox` as the primary server-side execution
model, and extend client-side tools with interactive process support.

#### Server-side: Terminals replace code_sandbox

`code_sandbox` is a one-shot subprocess runner. Terminals are strictly
more capable: persistent sessions, interactive programs, multiple
concurrent processes, screen monitoring. Rather than maintaining two
server-side execution models, terminals should subsume code_sandbox.

**Migration path:**
1. Implement the terminal system (below).
2. For agents that currently use `code_sandbox`, the terminal provides
   the same capability: `terminal_send("python script.py")` +
   `terminal_read()` replaces `code_sandbox("python script.py")`.
3. Optionally keep `code_sandbox` as a convenience wrapper that
   internally uses a terminal but returns structured output
   (stdout/stderr/exit_code) instead of raw screen capture. This
   gives agents the clean interface of code_sandbox when they don't
   need interactivity, while using the same underlying mechanism.
4. Long-term: deprecate `code_sandbox` as a separate builtin. Agents
   that need structured output use local Python tools. Agents that
   need execution use terminals.

**Backend choice: `pexpect` + `pyte`** (following Gemini CLI's
pattern).

Gemini CLI's approach -- `node-pty` + headless `xterm.js` -- is the
canonical pattern for standalone agent terminals. In Python, the
equivalent is `pexpect` (PTY allocation) + `pyte` (headless terminal
emulator). Both are pure-pip, no system dependencies, works on
macOS and Linux.

**Why not tmux**: Tmux requires the `tmux` system binary, which
isn't present on fresh Docker images, most macOS installs without
Homebrew, or base Linux distributions. It also has higher
per-operation cost (fork+exec a tmux CLI for every send/read) and
complicates the DBOS durability model (tmux server survives parent
crashes, which conflicts with workflow replay semantics). The
extras tmux provides (multi-pane, detach/reattach) are irrelevant
for agent use cases.

**Why not raw subprocess**: Loses interactivity -- can't handle
interactive prompts, TUI programs, or persistent shell state.

**Why screen rendering at all?** For most agent use cases, raw
stdout with ANSI stripping is sufficient (that's what Claude Code
does). But headless xterm.js / pyte gives you clean rendered output
for programs that update in-place (progress bars, spinners), at
minimal cost. The default should be raw stdout; pyte rendering is
available for tools that explicitly need it.

```python
class TerminalSession:
    """One pexpect-managed PTY session with an optional pyte screen."""
    def launch(command: str, args: list[str], cwd: Path, dimensions: tuple[int, int]) -> None
    def send(text: str, keys: list[str]) -> None
    def read_stdout(lines: int | None) -> str  # raw stdout ring buffer
    def read_screen() -> str  # rendered screen via pyte (optional)
    def is_alive() -> bool
    def close() -> None
```

**Backend policy**: one backend, one path (Design Principle #2 --
no dual modes). If `pexpect` can't spawn a PTY (e.g., Windows
without `winpty`), the terminal system returns an error at launch
time rather than silently falling back to a weaker mode.

**New files:**
- `agent_plane/terminals/` package:
  - `session.py`: `TerminalSession` using `pexpect` + `pyte`
  - `manager.py`: `TerminalManager` -- per-workflow session
    registry, lifecycle management
  - `ring_buffer.py`: bounded output ring buffer with ANSI
    stripping
  - `sandbox_launcher.py`: generates shim script that applies
    Landlock/seccomp before exec'ing the target binary (only
    used when sandbox is enabled)

**Server-side tools** (workflow-level dispatch):
- `terminal_launch(name, command, args)`: start a tmux session
- `terminal_send(name, text, keys)`: send keystrokes
- `terminal_read(name, scrollback)`: capture screen (ANSI-stripped)
- `terminal_list()`: list active terminal instances
- `terminal_close(name)`: kill tmux session, clean up

**Spec changes:**
```yaml
terminals:
  code_runner:
    command: bash
    scrollback: 10000
    sandbox:
      write_paths: [.]
  claude_worker:
    command: claude
    args: [--agent]
    scrollback: 10000
```

**Security model**: each `TerminalSession` is an independent
subprocess (direct `pexpect.spawn` with its own PTY pair). No shared
state between sessions. When sandbox is enabled, the sandbox launcher
shim applies Landlock + seccomp inside the process before `exec`ing
the target binary. No tmux sockets to protect, no per-instance server
isolation to manage.

#### Client-side: Interactive process support (G20)

Agent-plane's client-side Bash tool cannot handle interactive programs,
long-running processes, or persistent shell state. Adding client-side
terminal support fills these gaps.

**New client-side tools** (tunneled like existing Read/Write/Edit/Bash):
- `terminal_launch(command, args)`: client starts a tmux session
  locally, returns `{session_id}`
- `terminal_send(session_id, text, keys)`: client sends keystrokes
- `terminal_read(session_id, scrollback)`: client captures screen
- `terminal_list()`: client lists active local terminals
- `terminal_close(session_id)`: client kills the session

These are client-specified tools: the server presents the schema to
the LLM, the client handles execution, results come back via PATCH.
The client-side terminal manager lives in the frontend SDK or the
REPL code.

**Use cases unlocked:**
- `gcloud auth login` in a terminal, agent reads the screen for
  success/failure
- `npm run dev` in a background terminal, agent checks output
  periodically via `terminal_read`
- Persistent shell session with cd, env vars, virtualenv that
  survive across tool calls

**Estimated complexity**: MEDIUM. Pure-Python pexpect+pyte
implementation is ~200-300 lines. Sandbox launcher shim is a small
addition. Client-side terminals use the same `TerminalSession`
implementation but run on the client machine via tunneled tool
calls.

**Dependencies added**: `pexpect` (~40k downloads/day, stable since
2012), `pyte` (~30k downloads/day, stable). Both pure-Python, no
system binaries.

**Key risk minimal**: no system dependencies. Works on fresh Docker
images. Works on stock macOS. Works on any Linux distro with
Python.

**Industry precedent (surveyed April 2026)**:

| Project | Stars | Execution backend |
|---|---|---|
| **Gemini CLI** (Google) | ~60k | `node-pty` + headless `xterm.js` terminal emulator. Fallback to `child_process` when node-pty unavailable. |
| **OpenHands** | ~50k | `libtmux` (primary) + adding `SubprocessBashSession` fallback. |
| **Cline** | ~45k | VSCode integrated terminal with OSC 633 shell integration. |
| **Aider** | ~40k | Proposes commands, user runs them (no direct execution). |
| **Continue.dev** | ~25k | Node subprocess via `ink`-based CLI. |
| **Goose** (Block) | ~25k | Delegates to MCP "developer" extension (no core shell tool). |
| **Roo Code** | ~20k | Dual: VSCode shell integration OR inline `execa` subprocess. |
| **SWE-agent** | ~18k | Stateless `subprocess.run` / `docker exec`. |
| **OpenAI Codex CLI** | ~15k | Rust native PTY + process manager with session IDs. |
| **Smolagents** (HF) | ~15k | Custom AST Python interpreter; blocks subprocess entirely. |
| **Claude Code** | Closed | Persistent bash subprocess, no tmux. |
| **Cursor** | Closed | VSCode integrated terminal + OSC 633. |
| **Modal** | Infrastructure | `Sandbox.exec(pty=True)` -- PTY as first-class primitive. |
| **E2B** | Infrastructure | `sandbox.pty.create()` -- Firecracker + PTY. |
| **OmniAgents** | Experimental | Raw tmux with per-instance servers. |

**Clear pattern**: PTY + headless terminal emulator is the dominant
choice for modern standalone agents. Gemini CLI's `node-pty` +
`xterm.js` headless is exactly the pexpect + pyte pattern in
JavaScript -- confirming it's the canonical approach.

VSCode OSC 633 shell integration is popular for IDE extensions
(Cursor, Cline, Roo Code) but is IDE-specific and not portable.

Tmux is used by OpenHands and OmniAgents but OpenHands is actively
adding a non-tmux fallback because tmux isn't always available
(OpenHands issue #9971).

Stateless subprocess is a legitimate choice in containerized
environments (SWE-agent, Roo Code's inline mode, Smolagents).

### Phase 5: Cancellation Enhancement (G6)

**Workflow changes:**
- Track `_execution_phase` (IDLE/MODEL/TOOL) in the workflow
- On cancel during MODEL: interrupt executor
  (`executor.interrupt_session` equivalent)
- On cancel during TOOL: cancel running `@step` task
  (`CancellableFunctionTool` protocol -- add `cancel()` to Tool ABC)
- Inject cancellation notice at start of next turn
- Return phase information in cancel response

**Estimated complexity**: LOW-MEDIUM. Mostly workflow logic changes.

### Phase 6: Remaining Medium Gaps (G7-G13)

**G7 Memory**: Add `MemoryStore` with scoped KV operations. Persist
in database. New system tools: `memory_get`, `memory_set`,
`memory_delete`, `memory_list`.

**G8 Connections**: Not a direct port. Agent-plane's SSE + steering
model is fundamentally different. Consider whether bidirectional
WebSocket support is needed (G17). If so, implement as an alternative
transport, not a replacement for SSE.

**G9 Credentials**: Add optional `credentials` section to spec with
scope-based access control. Not urgent unless multi-tenant isolation
is a near-term requirement.

**G10 In-process runtime**: Evaluate whether `code_sandbox` (isolated
subprocess) is sufficient. If session-object access is needed, this
requires careful security design. Likely lower priority.

**G11 Codex/Pi executors**: Port as new `Executor` implementations.
Self-contained. Depends on the binaries being available.

**G12 InheritedTool**: Conflicts with spec self-containment
(Principle 1). Sub-agents should declare their own tools. If tool
sharing is needed, implement via MCP server that both agents connect
to. Do NOT port the `inherit` keyword.

**G13 MLflow tracing**: Port `TracingContext` as an observability
integration. Add span creation hooks at executor turn, tool call,
and policy evaluation points.

---

## 6. Recommended Phasing

```
Phase 1: Guardrails Foundation (G1)              [HIGH priority, HIGH effort]
  ├── PolicyEngine + 5 policy types
  ├── LabelSchemaRule + LabelState
  ├── Spec format extensions
  ├── 4 evaluation points in workflow
  └── SSE event extensions

Phase 2: OS Environment (G2)                     [HIGH priority, MEDIUM effort]
  ├── OSEnvironment ABC + LocalOSEnvironment
  ├── Sandboxed helper subprocess
  ├── Landlock/seccomp backend
  └── System tools (os_read/write/edit/shell)

Phase 3: Async Tools + Named Sessions (G4, G5)   [MEDIUM priority, MEDIUM effort]
  ├── Background tool execution via child workflows
  ├── Inbox as store query
  ├── NamedSessionStore
  └── session_send/peek/list/close tools

Phase 4: Terminal Environment (G3, G20)           [MEDIUM priority, MEDIUM effort]
  ├── Server-side: TerminalSession (pexpect + pyte), sandbox launcher
  ├── Server-side: terminal tools (launch/send/read/list/close)
  ├── Server-side: code_sandbox migration path
  ├── Client-side: terminal tools (tunneled via PATCH)
  └── Client-side: TerminalSession manager in frontend SDK / REPL

Phase 5: Cancellation Enhancement (G6)           [LOW priority, LOW effort]
  ├── Phase-aware cancel in workflow
  ├── Tool cancel protocol
  └── Cancellation notice injection

Phase 6: Remaining Gaps (G7-G13)                 [LOW priority, VARIES]
  ├── Memory system (G7)
  ├── Codex/Pi executors (G11)
  ├── MLflow tracing (G13)
  └── Others as needed
```

**Do NOT port**: G12 (InheritedTool -- violates spec
self-containment), G18 (mascots -- cosmetic).

**Defer or adapt**: G8 (Connections -- agent-plane's SSE+steering is
different by design), G16 (Vercel AI SDK protocol -- agent-plane uses
OpenAI-compatible events), G17 (WebSocket -- not a real gap, SSE+POST
matches the OpenAI Responses API protocol).

---

## 7. Open Questions / Decisions Required

1. ~~**Should background tools be durable?**~~ **RESOLVED**: Yes. Use
   DBOS child workflows with signal-based wake (`dbos_recv_async`).
   Durable, zero wasted LLM calls, reuses existing DBOS primitives.
   See Rift 3 and Phase 3 for full design.

2. **Should policy verdicts be persisted?** If we want audit trails
   and post-hoc analysis of guardrail decisions, persist as
   conversation items. If ephemeral is fine, keep in-memory only.

3. **Should the Ask flow be synchronous or SSE-based?** OmniAgents
   blocks the session loop and waits for the ask handler. In
   agent-plane, the workflow could: (a) emit an SSE event and park
   (like client-side tool tunneling), or (b) complete the task with
   `status: action_required` and require a re-POST.

4. **Should labels propagate to/from sub-agents?** OmniAgents' label
   propagation (`merged_with_child`) is a security feature. If
   agent-plane sub-agents are fully isolated (separate conversations),
   propagation requires reading child conversation metadata after
   completion. Decide whether this is needed.

5. **Should OS env state survive across tasks?** In OmniAgents, the
   OS env is per-session (ephemeral). In agent-plane, conversations
   span multiple tasks. Should file changes persist across tasks
   (like executor storage) or start fresh each task?

6. **Priority ordering**: Which phase should start first? The above
   ordering assumes guardrails are highest value, but if OS
   environment or terminals are needed sooner for specific use cases,
   the order can be adjusted.

7. **OmniAgents YAML compatibility**: Should agent-plane support
   loading OmniAgents YAML files as a convenience? This would require
   a translator from single-YAML to agent-image format. Not
   recommended (adds a second path, violates Principle 2), but worth
   discussing.

---

## 8. User-Facing Feature Diff

The sections above focus on architecture and internals. This section
lists the differences that matter to someone **building or using
agents** -- what each system can do that the other cannot.

### 8.1 Agent-plane has, OmniAgents lacks

| # | Feature | What it means for users |
|---|---------|------------------------|
| 1 | **Client-side tools** | Agents can ask the user's local machine to read/write/edit files and run shell commands. Tools execute on the client, not the server. This is how coding agents work -- the agent reads your local repo, edits files in place, runs your test suite. OmniAgents' `os_env` tools all execute server-side. **Gap**: client-side Bash is one-shot only -- no interactive programs (e.g. `gcloud auth login`), no long-running processes (e.g. `npm run dev` with later monitoring), no persistent shell state. See G20 for proposed client-side terminal support. |
| 2 | **Multi-provider LLM** | Point an agent at OpenAI, Anthropic, Gemini, Bedrock, Vertex, Databricks, ollama, groq, deepseek, xai, openrouter. OmniAgents is essentially Databricks-only for the default executor. |
| 3 | **Compaction** | Long conversations automatically get summarized so they don't hit context window limits. Users can have arbitrarily long sessions. OmniAgents has no compaction -- long conversations eventually fail. |
| 4 | **Crash recovery** | Server crashes mid-conversation, agent picks up where it left off. Transparent to the user. OmniAgents loses all state on crash. |
| 5 | **Conversation forking** | Branch from any mid-conversation point into a new thread. Useful for exploring alternative approaches without losing the original. |
| 6 | **Web search / web fetch** | Builtin tools with multiple search providers (OpenAI, Google, Perplexity). OmniAgents has no builtin web tools (would need MCP or custom tools). |
| 7 | **File upload / download** | Users can send files to agents and retrieve files back. Builtin tools: `upload_file`, `download_file`, `list_files`. |
| 8 | **Conversation search** | Search across all past conversations by keyword. Builtin tool: `search_conversations`. |
| 9 | **Agent deployment** | `ap deploy` ships an agent bundle to a remote server. OmniAgents requires manual server setup or runs locally only. |
| 10 | **Agent scaffolding** | `ap create` interactively walks through creating a new agent (picks provider, model, tools, generates config.yaml + AGENTS.md). |
| 11 | **Multimodal input** | Send images, audio, video, files to agents. Modalities declared in spec. OmniAgents is text-only input. |
| 12 | **Reasoning visibility** | Agent's chain-of-thought (reasoning/thinking tokens) surfaced in the UI as they stream. |
| 13 | **Sub-agent auto-collect** | When sub-agents finish, their results are automatically gathered and injected into the parent's context. No manual "check on sub-agents" calls needed. |
| 14 | **REPL slash commands** | `/new`, `/switch`, `/history`, `/agents`, `/cancel`, `/quit` for managing conversations in the terminal. |
| 15 | **Agent export** | Package an agent as a portable bundle (`export_agent` tool). |

### 8.2 OmniAgents has, agent-plane lacks

| # | Feature | What it means for users |
|---|---------|------------------------|
| 1 | **Guardrails (policies + labels)** | Agent authors can declare input/output/tool-call/tool-result policies that ALLOW, DENY, or ASK for approval. Rate limits, LLM-evaluated content filters, label-based information flow control. Users see "[DENIED by policy]" when guardrails trigger. Agent-plane has zero guardrails. |
| 2 | **OS environment per agent** | Each agent gets its own sandboxed filesystem and shell (read/write/edit/shell). Landlock+seccomp kernel sandbox on Linux. Fork mode copies the working directory for safe mutation. Agent-plane has `code_sandbox` but not per-agent filesystem isolation. |
| 3 | **Terminal sessions (tmux)** | Agents can launch, interact with, and read from interactive terminal processes. An agent can start a `claude` CLI in a tmux pane, send it commands, read its screen output. Each terminal has its own isolated tmux server. Terminals are strictly more capable than agent-plane's `code_sandbox` -- proposed to replace it server-side and extend client-side tools with interactive process support (see Phase 4). |
| 4 | **Background tool execution** | Agents can fire a tool in the background (`sys_call_async`), continue working, and collect results later (`sys_read_inbox`). The session sleeps between completions (no wasted LLM calls). Useful for parallel research tasks. Agent-plane has parallel tool calls within a turn but no fire-and-forget across turns. Proposed design: DBOS child workflows with `dbos_recv_async` signal-based wake (see Rift 3 and Phase 3). |
| 5 | **Named persistent child sessions** | Parent agents can maintain long-lived named sessions with child agents (`sys_session_send/peek/list/close`). A "coding supervisor" can send work to a "coder" session, peek at progress, send more instructions. Sessions persist across parent turns. Agent-plane's sub-agents are one-shot (spawn, collect, done). |
| 6 | **Interactive approval (ASK)** | Policies can require user approval before proceeding. The agent pauses, shows the user what's being attempted, waits for yes/no. Configurable timeout with auto-deny. |
| 7 | **Cross-session environment mounts** | Parent agents can mount a child's filesystem at a path prefix (`sys_os_mount_env`). Inspect diffs, compare parallel approaches, merge files -- all through the standard os_read/os_write tools. |
| 8 | **Phase-aware cancellation** | Cancel targets the right thing: during model inference it interrupts the LLM, during tool execution it cancels the tool. Next turn gets a notice explaining what was cancelled. Agent-plane has coarser task-level cancel. |
| 9 | **In-process Python runtime** | `sys_runtime_execute` runs Python code with direct access to the session object. The agent can programmatically inspect its own history, memory, labels. Agent-plane's `code_sandbox` is isolated with no session access. |
| 10 | **Scoped memory** | Key-value memory with `per_session`, `per_user`, `cross_user` scopes. Agents can remember things across turns (session) or across conversations (user/cross_user). |
| 11 | **Credential attenuation** | Token-based credentials with scopes that narrow as they propagate to child agents. A child agent gets a subset of the parent's permissions. |
| 12 | **Tool inheritance** | Child agents can inherit tools from their parent (`inherit` keyword) instead of declaring their own. |
| 13 | **MLflow tracing** | Automatic span creation (AGENT, CHAT_MODEL, TOOL, GUARDRAIL) integrated with MLflow. |
| 14 | **CLI debug overlay** | Ctrl-O opens a debug view showing all active sessions with scrollable output, session switching (tab/shift-tab), search (ctrl-r). |
| 15 | **WebSocket chat** | Server supports WebSocket as an alternative transport for chat. Not a real gap -- SSE + POST achieves the same thing and matches the OpenAI Responses API protocol. Only relevant if real-time audio (OpenAI Realtime API) is needed. |
| 16 | **Codex / Pi executors** | Run agents through OpenAI Codex (`codex app-server`) or Pi coding agent (`pi --mode rpc`) as harnesses. |
| 17 | **Session snapshot/restore** | Save a session to JSON and resume it later (`omniagents resume`). |
| 18 | **Unity Catalog functions** | Use Databricks UC SQL functions as agent tools. |
| 19 | **Vercel AI SDK protocol** | SSE streaming compatible with Vercel AI SDK frontend components. |
