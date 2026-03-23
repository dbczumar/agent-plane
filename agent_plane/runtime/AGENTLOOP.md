Layer 2: Agent Execution Loop

 Context

 The DBOS plumbing (Layer 1) is complete — durable workflows, streaming, steering handshake all work end-to-end. The
 placeholder workflow returns hardcoded strings. Layer 2 replaces it with the real agent loop: load agent → build prompt →
 call LLM → execute tools → repeat.

 The goal is a working agent execution loop that can load an agent spec, call an LLM, execute MCP tools, handle the
 steering inbox, persist output, and stream events — all durably checkpointed by DBOS.

 Design Decisions

 LLM client: litellm

 litellm wraps OpenAI, Anthropic, Cohere, etc. behind a single completion() API. Returns OpenAI-format responses. Matches
 RUNTIME.md's reference. litellm.completion() is synchronous, which works cleanly inside DBOS @step functions. Add
 litellm>=1.40 to dependencies.

 MCP client: the mcp package

 The mcp PyPI package provides stdio and HTTP transport clients. It's async-only, but DBOS step threads have no running
 event loop, so asyncio.run() works inside steps. Add mcp>=1.0 to dependencies.

 Non-streaming LLM calls for MVP

 DBOS checkpoints @step output on completion. Streaming deltas from the LLM mid-step would bypass checkpointing. MVP uses
 litellm.completion(stream=False) — the complete response is the step output. After the step returns, the workflow writes
 the result to the DBOS stream for SSE. This means clients see the full response arrive at once (no token-by-token
 streaming), but durability is correct. Streaming LLM calls are a follow-up optimization.

 Per-execution tool manager

 Each workflow gets its own ToolManager — connects MCP servers at start, tears them down in finally. No cross-execution
 resource sharing. Simple, no leaked state.

 Agent loading via AgentCache

 AgentCache (already implemented in runtime/agent_cache.py) is a two-tier cache (memory + disk) backed by ArtifactStore.
 On cache miss it downloads the tarball, extracts to disk, parses, and validates via spec.load(). On hit it returns a
 LoadedAgent(spec, workdir) from memory or re-parses from the disk cache. The cache is constructed at server startup and
 held in _globals. Workflows call agent_cache.load(agent_id) — no manual extract/parse/validate steps.

 Store access via module globals

 runtime/_globals.py holds store references and the AgentCache, set once at server startup. Workflow functions read them
 directly, matching the pattern in RUNTIME.md.

 Tool manager concurrency via contextvars

 The ToolManager is stored in a contextvars.ContextVar (not a plain global) so concurrent workflows in the same process
 don't collide — DBOS runs each workflow in its own thread, and contextvars are per-task/per-thread safe.

 New Files

 agent_plane/runtime/
   _globals.py      # NEW — module-level store globals + AgentCache + init()
   prompt.py        # NEW — prompt construction from spec + history
   tool_manager.py  # NEW — MCP lifecycle, tool routing, built-in tools
   steps.py         # NEW — @step functions (call_llm, call_tool, load_agent, load_history)
   workflow.py      # REPLACE — real agent loop
   agent_cache.py   # EXISTS — two-tier agent cache (memory + disk)

 File Details

 1. _globals.py — Store globals + AgentCache

 conversation_store: ConversationStore | None = None
 task_store: TaskStore | None = None
 agent_store: AgentStore | None = None
 agent_cache: AgentCache | None = None

 # Per-workflow tool manager (contextvars for thread safety)
 _tool_manager: ContextVar[ToolManager | None] = ContextVar(default=None)

 def init(conversation_store, task_store, agent_store, agent_cache):
     """Called once at server startup."""

 cli.py constructs the AgentCache (wrapping ArtifactStore) and passes it to init().
 Workflows access agent_cache.load(agent_id) to get LoadedAgent(spec, workdir).

 2. prompt.py — Prompt construction

 Pure functions, no side effects. Key interfaces:

 - build_system_message(spec, per_request_instructions, tool_schemas) → system message dict
   - Concatenates: agent instructions (AGENTS.md) + per-request instructions + skill metadata (name + description only, so
 LLM knows skills exist and can call load_skill)
 - history_to_messages(items: list[ConversationItem]) → litellm message list
   - Maps conversation items to OpenAI chat format:
       - message(role=user) → {"role": "user", "content": ...}
     - message(role=assistant) → {"role": "assistant", "content": ...}
     - function_call → {"role": "assistant", "tool_calls": [...]}
     - function_call_output → {"role": "tool", "tool_call_id": ..., "content": ...}
   - Merges consecutive assistant messages with tool_calls into one message (litellm expects this)
 - build_messages(spec, history, instructions, tool_schemas) → full messages list

 3. tool_manager.py — Tool lifecycle and dispatch

 class ToolManager:
     def __init__(self, spec: AgentSpec, work_dir: Path): ...
     def start(self) -> None:          # connect MCP servers, discover tools, register builtins
     def shutdown(self) -> None:       # close all connections
     def get_tool_schemas(self) -> list[dict]:  # OpenAI-format tool schemas
     def call_tool(self, name: str, arguments: str) -> str:  # route and execute

 - MCP stdio: asyncio.run(stdio_client(...)) inside sync methods — safe because DBOS step threads have no event loop
 - MCP HTTP: asyncio.run(streamablehttp_client(...))
 - Built-in load_skill: looks up skill by name in spec.skills, returns full SkillSpec.content
 - Local tools: deferred — returns error message if called in MVP

 4. steps.py — DBOS-checkpointed operations

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
     # litellm.completion(...) → response dict

 @step()
 def call_tool(tool_name: str, arguments: str) -> str:
     # Reads ToolManager from _globals._tool_manager contextvar
     # Routes to MCP server, built-in, or local tool

 Serialization boundary: All @step inputs and outputs must be JSON-serializable. AgentSpec is passed as dict between steps;
  reconstructed to dataclass in the workflow. workdir is passed as a string path.

 Crash recovery: On restart, completed steps return cached output. The AgentCache's disk tier means the bundle doesn't
 need to be re-downloaded — just re-parsed from disk. MCP servers are reconnected, but completed LLM/tool calls are skipped.

 5. workflow.py — The agent loop

 @workflow()
 def agent_execution_workflow(agent_id, conversation_id, previous_response_id, instructions):
     task_id = get_workflow_id()

     # Phase 1: Load
     agent_dict = load_agent(agent_id)               # @step (cached on recovery)
     spec = reconstruct_spec(agent_dict["spec"])
     work_dir = Path(agent_dict["workdir"])
     tool_mgr = ToolManager(spec, work_dir)
     _globals._tool_manager.set(tool_mgr)

     try:
         tool_mgr.start()
         tool_schemas = tool_mgr.get_tool_schemas()

         history_dicts = load_history(conversation_id)   # @step
         history = [reconstruct_item(d) for d in history_dicts]
         last_seen = history[-1].id if history else None
         output_items = []
         total_usage = {}

         # Phase 2: Loop
         max_iter = spec.params.get("max_iterations", 32)
         for _ in range(max_iter):
             # Check steering
             new = check_steering(conversation_id, last_seen)  # @step
             if new:
                 new_items = [reconstruct_item(d) for d in new]
                 history.extend(new_items)
                 last_seen = new_items[-1].id

             # Call LLM
             messages = build_messages(spec, history, instructions, tool_schemas)
             llm_resp = call_llm(messages, spec.llm.model, tool_schemas,
                                spec.llm.max_completion_tokens,
                                spec.llm.reasoning_effort)         # @step

             # Stream the response
             for event in response_to_stream_events(llm_resp, len(output_items)):
                 write_stream("output", event)

             # If no tool calls → final response
             if not has_tool_calls(llm_resp):
                 late = task_store.close_inbox(task_id, conversation_id, last_seen)
                 if late:
                     history.append(to_history(llm_resp, task_id))
                     history.extend(late)
                     last_seen = late[-1].id
                     continue

                 # Persist and return
                 persist_output(conversation_id, task_id, spec, llm_resp, output_items)
                 return build_result(task_id, "completed", output_items, total_usage)

             # Execute tool calls
             history.append(to_history(llm_resp, task_id))
             for tc in get_tool_calls(llm_resp):
                 result = call_tool(tc.name, tc.arguments)  # @step
                 stream_tool_output(tc, result)
                 history.append(to_tool_output(tc, result))
                 output_items.append(to_output_item(tc, result))

         return build_result(task_id, "incomplete",
                            incomplete_details={"reason": "max_output_tokens"})
     finally:
         close_stream("output")
         tool_mgr.shutdown()
         _globals._tool_manager.set(None)
         drain_inbox(task_id, conversation_id, last_seen)

 6. cli.py — Add runtime init

 After constructing stores, before uvicorn.run():

 from agent_plane.runtime.agent_cache import AgentCache
 from agent_plane.runtime._globals import init as init_runtime

 agent_cache = AgentCache(artifact_store=artifact_store, cache_dir=Path(cache_dir))
 init_runtime(
     conversation_store=conversation_store,
     task_store=task_store,
     agent_store=agent_store,
     agent_cache=agent_cache,
 )

 7. pyproject.toml — New dependencies

 litellm>=1.40
 mcp>=1.0

 Implementation Phases

 Phase A: Foundation (no external deps needed)

 1. _globals.py — store globals + AgentCache + init
 2. runtime/__init__.py — re-export init
 3. cli.py — construct AgentCache, call init at startup
 4. prompt.py — message construction from spec + history
 5. Tests for prompt.py (pure data transformation, no mocks)

 Phase B: LLM integration

 1. Add litellm dependency
 2. steps.py — load_agent (via AgentCache), load_history, check_steering, call_llm
 3. Tests for steps (monkeypatch litellm.completion)

 Phase C: Tool integration

 1. Add mcp dependency
 2. tool_manager.py — MCP client lifecycle, tool routing, load_skill built-in
 3. steps.py — add call_tool step
 4. Tests for tool_manager (mock MCP sessions)

 Phase D: Assemble the loop

 1. workflow.py — replace placeholder with real agent loop
 2. Integration tests (monkeypatch call_llm to return canned responses)
 3. Test steering handshake end-to-end
 4. Test max_iterations → incomplete
 5. Test error paths (LLM failure, tool failure)

 Phase E: Polish

 1. Usage tracking (extract token counts from litellm response)
 2. Verify SSE event shapes match OpenAI Responses API format
 3. Cancellation handling (verify finally block runs correctly)
 4. completed_at timestamp population

 Key Existing Code to Reuse

 - agent_plane/runtime/agent_cache.py — AgentCache.load(agent_id) → LoadedAgent(spec, workdir)
 - agent_plane/spec.load(source, dest) — extract + parse + validate in one call
 - agent_plane/entities/conversation.py — all item data types and parse_item_data()
 - agent_plane/entities/agent.py — LoadedAgent dataclass
 - agent_plane/runtime/durability.py — workflow/step/write_stream/close_stream
 - agent_plane/stores/task_store — close_inbox, try_deliver (steering handshake)
 - agent_plane/stores/conversation_store — list_items, append

 Verification

 1. Unit tests: prompt.py (pure functions), tool_manager routing logic
 2. Integration tests: Full workflow with monkeypatched call_llm returning canned responses. Verify: events stream
 correctly, output persisted to conversation, steering works, inbox drained on exit
 3. Manual smoke test: Start server, register an agent bundle with a real LLM config, POST /v1/responses, verify response
 comes back with real LLM output
 4. Tool test: Agent with an MCP server (e.g. a trivial stdio tool), verify function_call → function_call_output round-trip
