# runtime/ — How an agent RUNS

Depends on: spec/

The execution engine. Takes an `AgentSpec` + a working directory + user inputs,
runs the agent loop, and produces outputs. Knows nothing about HTTP, databases,
or bundle storage.

## Planned files

### executor.py
The core execution interface:
- `AgentExecutor.from_spec(spec, workdir)` — construct an executor from a parsed spec
- `executor.run(messages) → AgentResponse` — synchronous execution
- `executor.stream(messages) → AsyncIterator[AgentEvent]` — streaming execution
- Manages the LLM call loop: prompt construction → inference → tool calls → repeat
- Enforces max iterations, timeouts
- Handles context window overflow (compaction/pruning)

### tool_manager.py
Load and manage tools for an execution:
- Start MCP server connections (stdio subprocesses, HTTP clients)
- Load local tools from working directory (import Python modules, etc.)
- Provide built-in tools: `load_skill()`, `load_reference()`, `load_script()`
- Route tool calls to the correct handler
- Enforce tool-level timeouts and retries
- Sandbox/isolation considerations for local tool code

### skill_manager.py
Progressive skill disclosure:
- At startup: inject skill metadata (name + description only) into system prompt
- On `load_skill(names)`: return full SKILL.md body content
- On `load_reference(skill, ref)`: return reference file content
- On `load_script(skill, script)`: return script file content
- All reads are from the extracted working directory

### session.py
Conversation state for a single execution:
- Message history (user, assistant, tool messages)
- Active skill context
- Execution metadata (iteration count, token usage, timing)
- Checkpointing interface (for pause/resume — future)

## Key design decisions
- The runtime is a library, not a service — it can be embedded anywhere
- The server is the primary host, but CLI direct-run and embedded use are also valid
- No database or network dependencies (except LLM inference + MCP calls)
- Tool manager handles the complexity of different tool types behind one interface

## Not yet (future)
- Sub-agent execution (agent A invokes agent B)
- Execution checkpointing and resume
- Resource limits / sandboxing for tool code
- Identity propagation to tools
