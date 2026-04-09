# OpenAI Agents SDK Executor

## Context

The executor plugin system (`Executor` ABC, `_create_executor` dispatch)
supports three backends today: `DefaultExecutor` (litellm),
`ClaudeAgentsExecutor` (Claude Agent SDK subprocess), and
`RemoteExecutor` (HTTP). All three share the same event protocol and
workflow integration.

The OpenAI Agents SDK (`openai-agents` on PyPI) is OpenAI's Python
framework for building agents. It manages the agent loop internally
(`Runner.run_streamed()`), supports streaming, function tools, MCP
servers, handoffs, and 100+ LLMs via the Responses and Chat Completions
APIs. It's the structural parallel to the Claude Agent SDK —
a proper Python package that owns the loop.

For Codex-specific coding capabilities (sandboxed shell, file edits),
the Codex CLI can be attached as an MCP server (`codex mcp-server`).
This is a configuration choice — the executor itself is the Agents
SDK, not Codex directly.

| Aspect | Claude SDK Executor | OpenAI Agents SDK Executor (proposed) |
|--------|-------------------|--------------------------------------|
| Package | `claude-agent-sdk` | `openai-agents` (PyPI) |
| Loop owner | Claude SDK subprocess | `Runner.run_streamed()` (in-process) |
| Streaming | SDK stream → queue bridge (threading) | Native async iterator |
| Native tools | Bash, Read, Edit, Write, Glob, Grep | Function tools (Python `async def`) |
| Coding tools | Built-in (Claude Code) | Optional: `codex mcp-server` |
| Client-side tools | In-process MCP server bridge | Function tools calling `context.call_tool` |
| Context management | SDK-internal | SDK-internal (compaction, sessions) |
| Auth | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` (or per-provider) |
| Models | Claude family only | OpenAI, Anthropic, 100+ via LiteLLM |
| Conversation state | `.claude/` session transcript | `result.history` or `conversationId` |

### What exists

| Component | File | Status |
|-----------|------|--------|
| `Executor` ABC | `runtime/executors/base.py` | `from_spec`, `run_turn`, lifecycle hooks |
| `ClaudeAgentsExecutor` | `runtime/executors/claude.py` | Reference for SDK-managed executor pattern |
| `_create_executor` dispatch | `runtime/workflow.py:167` | `"llm"`, `"claude_sdk"`, `"remote"` branches |
| `ExecutorSpec.type` | `spec/types.py:57` | Discriminator field |
| `executors/__init__.py` | `runtime/executors/__init__.py` | Public re-exports |

### What's missing

1. `AgentsSdkExecutor` — new executor wrapping `openai-agents`
2. `"agents_sdk"` branch in `_create_executor`
3. Example agent configs (basic + Codex-backed)

---

## Why Agents SDK, Not Raw Codex

We initially considered speaking JSON-RPC directly to `codex
app-server`. The Agents SDK is better on every axis:

| Concern | Raw `codex app-server` | OpenAI Agents SDK |
|---------|----------------------|-------------------|
| **Transport** | Subprocess + JSON-RPC over stdio | Pure Python, `pip install` |
| **Agent loop** | Opaque subprocess, no control | `Runner.run_streamed()` with events |
| **Streaming** | Parse newline-delimited JSON manually | Native Python async iterator |
| **Client-side tools** | `dynamicTools` (experimental API) | Function tools = plain `async def` |
| **Model support** | Codex-supported models only | Any model via Responses/Chat Completions API |
| **Codex features** | Built-in Shell + ApplyPatch | Attach via `codex mcp-server` when needed |
| **Binary prereq** | `npm i -g @openai/codex` required | None (pure Python) |
| **Python SDK** | Experimental, not on PyPI | Official, well-maintained, on PyPI |

The Codex CLI's coding capabilities (sandboxed shell, file edits) are
accessible via MCP when needed — but not every agent needs them. The
Agents SDK is the general-purpose harness; Codex is an optional
attachment.

---

## Event Mapping

The Agents SDK streams two event types from `result.stream_events()`:
`RawResponsesStreamEvent` (token-level) and `RunItemStreamEvent`
(item-level). Both map to existing executor events.

| Agents SDK event | Executor event | Notes |
|-----------------|---------------|-------|
| `raw_response_event` + `ResponseTextDeltaEvent` | `TextChunk(text=delta)` | Streamed text tokens |
| `raw_response_event` + `ResponseReasoningSummaryTextDeltaEvent` | `ReasoningChunk(delta, "reasoning_summary")` | Reasoning model output |
| `run_item_stream_event` + `tool_called` | `ToolCallObserved(...)` | SDK executed a tool internally |
| `run_item_stream_event` + `tool_output` | (absorbed — result captured in `ToolCallObserved`) | |
| `run_item_stream_event` + `mcp_approval_requested` | (auto-approve via approval callback) | Codex MCP tool approvals |
| Run completion (async iterator ends) | `TurnComplete(text=final_output)` | `result.final_output` has the text |
| `MaxTurnsExceeded` exception | `TurnComplete(text=partial)` | SDK hit its turn limit |
| `ModelBehaviorError`, `UserError` | `ExecutorError(message, code)` | Unrecoverable failures |

No new event types needed.

---

## Design Decisions

### SDK manages the full agent loop

When `Runner.run_streamed()` is called, the SDK:
1. Calls the LLM with messages + tools
2. If the LLM requests tool calls, executes them
3. Re-calls the LLM with results
4. Repeats until the LLM produces a final response

This is the same ownership model as `ClaudeAgentsExecutor` — the
executor's `run_turn()` maps to one full SDK run (potentially many
LLM calls internally). The workflow sees a single turn with streamed
events.

`max_context_tokens()` returns `None` — the SDK manages compaction.

### Tools as function tools (no MCP bridge needed)

The Agents SDK supports `@function_tool`-decorated Python functions.
Agent-plane's server-side and client-side tools are registered as
function tools whose implementations call `context.call_tool()`:

```python
from agents import function_tool

def _make_tool(name: str, schema: dict, context: ExecutorContext):
    params = schema["function"]["parameters"]
    desc = schema["function"]["description"]

    @function_tool(name_override=name, description_override=desc)
    async def tool_fn(**kwargs) -> str:
        req = ToolCallRequested(
            call_id=f"call_{uuid4().hex[:12]}",
            name=name,
            arguments=kwargs,
        )
        result = context.call_tool(req)
        return result.content

    return tool_fn
```

This is dramatically simpler than the Claude executor's MCP server
bridge — no in-process MCP server, no callback holder, no threading.
The function tool is a plain async function that blocks on
`call_tool()`, which parks the call and waits for the client to
respond (for client-side tools) or executes immediately (for
server-side tools).

### Codex coding tools via MCP (optional)

When the agent spec declares `codex:Shell` or `codex:ApplyPatch` in
`tools.builtins`, the executor launches `codex mcp-server` as an MCP
server and attaches it to the Agent:

```python
from agents.mcp import MCPServerStdio

codex_mcp = MCPServerStdio(
    name="Codex CLI",
    params={
        "command": "codex",
        "args": ["mcp-server"],
    },
)

agent = Agent(
    name=spec.name,
    instructions=system_prompt,
    model=llm_config.model,
    mcp_servers=[codex_mcp],
    tools=[...function_tools...],
)
```

Codex MCP exposes two tools: `codex` (start session) and
`codex-reply` (continue session). The SDK calls them like any other
tool — the executor sees `ToolCallObserved` events.

This separates concerns cleanly:
- **Agents SDK** = agent loop, LLM calls, tool dispatch, streaming
- **Codex MCP** = coding capabilities (shell, file edits, sandbox)
- **Agent-plane** = persistence, client tunneling, steering, SSE

### Conversation state via history replay

The Agents SDK offers four state strategies: `result.history`,
`session`, `conversationId`, and `previousResponseId`. We use
`result.history` — the simplest. Each `run_streamed()` call receives
the full message history from the workflow (same as `DefaultExecutor`).
The SDK replays it into the LLM context.

No persistent SDK-side state to manage. If the process restarts, the
workflow reloads history from the conversation store and passes it in.
No `.claude/`-style session recovery needed.

### Model from agent spec

The model comes from `spec.llm.model`, same as all executors:

```yaml
llm:
  model: gpt-5.4          # or: openai/o3, anthropic/claude-sonnet-4, etc.
  connection:
    api_key: ${OPENAI_API_KEY}
```

The Agents SDK supports any model accessible via the OpenAI Responses
API or Chat Completions API. For non-OpenAI providers, the SDK's
LiteLLM integration handles routing.

---

## Implementation

### `AgentsSdkExecutor` class

```python
class AgentsSdkExecutor(Executor):
    """
    Executor wrapping the OpenAI Agents SDK.

    The SDK manages the agent loop internally. Server-side and
    client-side tools are registered as function tools. Codex
    coding capabilities are optionally attached via MCP.
    """

    def __init__(
        self,
        *,
        model: str,
        codex_tools: list[str],
        connection: dict[str, str] | None = None,
    ) -> None: ...

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """Extract model, codex tools, and connection from spec."""

    def max_context_tokens(self) -> int | None:
        return None  # SDK manages compaction

    def on_task_start(self, context: ExecutorContext) -> None:
        """Launch Codex MCP server if codex: tools declared."""

    def on_task_end(self, context: ExecutorContext) -> None:
        """Shut down Codex MCP server if running."""

    def run_turn(
        self, messages, tools, system_prompt,
        llm_config, context,
    ) -> Iterator[ExecutorEvent]:
        """Run SDK agent, bridge async stream to sync iterator."""
```

### Process lifecycle

```
from_spec()
  ├── Extract model from spec.llm.model
  ├── Extract codex:-prefixed tool names
  └── Extract connection config (api_key, base_url)

on_task_start()
  └── If codex: tools declared → start MCPServerStdio("codex mcp-server")

run_turn()
  ├── Build Agent(name, instructions, model, tools, mcp_servers)
  ├── Convert agent-plane tools → function_tool wrappers
  ├── Convert messages → Agents SDK input format
  ├── Runner.run_streamed(agent, input, history)
  ├── Consume stream_events() → yield executor events
  └── Return when async iterator ends

on_task_end()
  └── If codex MCP running → stop it
```

### Sync/async bridge

The workflow calls `run_turn()` synchronously (DBOS thread pool). The
Agents SDK is async (`Runner.run_streamed()` is a coroutine). Bridge
via the same event-queue pattern as the Claude executor:

```python
def run_turn(self, messages, tools, system_prompt,
             llm_config, context) -> Iterator[ExecutorEvent]:
    event_queue: queue.Queue[ExecutorEvent | None] = queue.Queue(
        maxsize=256,
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=lambda: loop.run_until_complete(
            _async_run(self, messages, tools, system_prompt,
                       llm_config, context, event_queue)
        ),
        daemon=True,
    )
    thread.start()
    while True:
        event = event_queue.get()
        if event is None:
            break
        yield event
    thread.join(timeout=5.0)
```

### Stream consumption

```python
async def _async_run(
    executor: AgentsSdkExecutor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    from agents import Agent, Runner
    from openai.types.responses import ResponseTextDeltaEvent

    function_tools = [
        _make_function_tool(schema, context)
        for schema in tools
    ]
    mcp_servers = executor._build_mcp_servers()

    agent = Agent(
        name="agent",
        instructions=system_prompt,
        model=llm_config.model,
        tools=function_tools,
        mcp_servers=mcp_servers,
    )

    input_items = _messages_to_input(messages)
    try:
        result = Runner.run_streamed(agent, input=input_items)
        async for event in result.stream_events():
            _map_event(event, event_queue)
        event_queue.put(TurnComplete(text=result.final_output))
    except Exception as exc:
        event_queue.put(ExecutorError(
            message=f"Agents SDK error: {exc}",
            code=type(exc).__name__,
        ))
    finally:
        event_queue.put(None)


def _map_event(
    event: StreamEvent,
    eq: queue.Queue[ExecutorEvent | None],
) -> None:
    from openai.types.responses import ResponseTextDeltaEvent

    if event.type == "raw_response_event":
        if isinstance(event.data, ResponseTextDeltaEvent):
            eq.put(TextChunk(text=event.data.delta))
    elif event.type == "run_item_stream_event":
        if event.name == "tool_called":
            item = event.item
            eq.put(ToolCallObserved(
                call_id=item.raw_item.call_id,
                name=item.raw_item.name,
                arguments=json.loads(item.raw_item.arguments),
                result=str(item.output),
                status="success",
                duration_ms=0.0,
            ))
```

### Client-side tool bridging

Agent-plane tools become Agents SDK function tools whose
implementations call `context.call_tool()`:

```python
def _make_function_tool(
    schema: dict[str, Any],
    context: ExecutorContext,
) -> Any:
    from agents import function_tool

    func_spec = schema["function"]
    name = func_spec["name"]
    desc = func_spec.get("description", "")

    @function_tool(name_override=name, description_override=desc)
    async def tool_fn(**kwargs: Any) -> str:
        req = ToolCallRequested(
            call_id=f"call_{uuid4().hex[:12]}",
            name=name,
            arguments=kwargs,
        )
        # call_tool blocks (parks) for client-side tools,
        # executes immediately for server-side tools.
        result = context.call_tool(req)
        return result.content

    return tool_fn
```

No MCP server bridge, no callback holders, no notification routing.
A function tool is just an async function. When the SDK decides to
call it, it runs, blocks on `call_tool()` if needed, and returns the
result. The SDK handles re-invoking the LLM with the result.

---

## Touchpoints

| File | Change |
|------|--------|
| `runtime/executors/agents_sdk.py` | New file — `AgentsSdkExecutor` |
| `runtime/executors/__init__.py` | Export `AgentsSdkExecutor` |
| `runtime/workflow.py:_create_executor` | Add `"agents_sdk"` branch |
| `spec/types.py:ExecutorSpec` | Update docstring to include `"agents_sdk"` |
| `spec/AGENTSPEC.md` | Document `executor.type: agents_sdk` and `codex:` tool prefix |
| `tests/runtime/test_executor.py` | Unit tests for event mapping |
| `examples/agents/openai-coder/` | Example: Agents SDK + Codex MCP |
| `examples/agents/openai-basic/` | Example: Agents SDK without Codex |

### Example configs

**Basic (no Codex):**

```yaml
# examples/agents/openai-basic/config.yaml
spec_version: 1

name: openai-basic
description: An assistant powered by the OpenAI Agents SDK.

executor:
  type: agents_sdk

llm:
  model: gpt-5.4
  connection:
    api_key: ${OPENAI_API_KEY}

instructions: INSTRUCTIONS.md
```

**With Codex coding tools:**

```yaml
# examples/agents/openai-coder/config.yaml
spec_version: 1

name: openai-coder
description: A coding assistant using OpenAI Agents SDK + Codex.

executor:
  type: agents_sdk

llm:
  model: gpt-5.4
  connection:
    api_key: ${OPENAI_API_KEY}

tools:
  builtins:
    - codex:Shell
    - codex:ApplyPatch
  agents:
    - reviewer

instructions: INSTRUCTIONS.md
```

### Dependency

```
pip install openai-agents
```

Requires Python 3.10+. For Codex MCP tools, `codex` binary must be
on PATH (`npm i -g @openai/codex`).

---

## Open Questions

1. **Conversation state strategy.** We default to history replay
   (pass full message list each turn). The Agents SDK also supports
   `conversationId` (server-managed state on OpenAI's side) and
   `previousResponseId` (lightweight chaining). History replay is
   simplest and matches existing executors, but `conversationId`
   would reduce token costs on long conversations. Worth revisiting
   if token costs become a concern.

2. **Codex MCP lifecycle.** The `codex mcp-server` subprocess is
   started in `on_task_start` and stopped in `on_task_end`. Should it
   persist across tasks (like the Claude SDK subprocess) to preserve
   shell state? Current leaning: start fresh each task — Codex MCP
   is stateless (`codex` tool starts a new session, `codex-reply`
   continues via thread ID).

3. **Non-OpenAI models.** The Agents SDK supports arbitrary providers
   via LiteLLM. Should `executor.type: agents_sdk` be the recommended
   path for all non-Anthropic models, replacing `DefaultExecutor`
   over time? Or keep both — `DefaultExecutor` for simple single-turn
   LLM calls, `AgentsSdkExecutor` for SDK-managed multi-turn loops?
