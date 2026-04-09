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
| Streaming | SDK stream → queue bridge (threading) | Native async iterator (still needs queue bridge) |
| Native tools | Bash, Read, Edit, Write, Glob, Grep | Function tools (Python `async def`) |
| Coding tools | Built-in (Claude Code) | Optional: `codex mcp-server` |
| Client-side tools | In-process MCP server bridge | Function tools calling `context.call_tool` |
| Context management | SDK-internal | History replay (workflow owns) or `conversationId` |
| Auth | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` (or per-provider via `llm.connection`) |
| Models | Claude family only | OpenAI, Anthropic, 100+ via LiteLLM |
| Conversation state | `.claude/` session transcript | `result.to_input_list()` replay |
| Skills | Written to `.claude/skills/`, SDK discovers | Injected into system prompt via `load_skill` tool |

### What exists

| Component | File | Status |
|-----------|------|--------|
| `Executor` ABC | `runtime/executors/base.py` | `from_spec`, `run_turn`, lifecycle hooks |
| `ClaudeAgentsExecutor` | `runtime/executors/claude.py` | Reference for SDK-managed executor pattern |
| `_create_executor` dispatch | `runtime/workflow.py:167` | `"llm"`, `"claude_sdk"`, `"remote"` branches |
| `ExecutorSpec.type` | `spec/types.py:57` | Discriminator field |
| `executors/__init__.py` | `runtime/executors/__init__.py` | Public re-exports |
| Spec validator | `spec/validator.py:102` | Per-executor-type field validation |

### What's missing

1. `AgentsSdkExecutor` — new executor wrapping `openai-agents`
2. `"agents_sdk"` branch in `_create_executor`
3. `_validate_agents_sdk_executor` in `validator.py`
4. Example agent configs (basic + Codex-backed)

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

## Config YAML Specification

### `executor:` block

```yaml
executor:
  type: agents_sdk
  timeout: 3600            # task deadline in seconds (default: 3600)
  max_iterations: 1000     # max run_turn() calls (default: 1000)
```

**Supported fields:** `type`, `timeout`, `max_iterations`.

**Forbidden fields:**
- `endpoint` — remote executor only
- `request_timeout` — remote executor only

`max_iterations` maps to the workflow's outer loop limit (how many
times the workflow calls `run_turn()`). The SDK's internal
`max_turns` (how many LLM calls within a single `run_turn()`) is
set to a high value (200) because the workflow already enforces its
own iteration budget. If the SDK hits its internal `max_turns`, it
raises `MaxTurnsExceeded` which the executor maps to
`TurnComplete(text=partial)`.

### `llm:` block

```yaml
llm:
  model: gpt-5.4                    # required — any Agents SDK-supported model
  reasoning_effort: medium           # optional — low | medium | high
  max_completion_tokens: 4096        # optional
  temperature: 0.7                   # optional (not supported by reasoning models)
  request_timeout: 300               # optional — per-LLM-call timeout (default: 300)
  connection:                        # optional — per-provider overrides
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
  retry:                             # optional
    max_attempts: 3
    backoff_base: 2.0
```

**All `llm` fields are supported.** Unlike `claude_sdk` which
forbids `llm.connection`, this executor passes `connection` through
to configure the Agents SDK's underlying OpenAI client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=connection.get("api_key"),
    base_url=connection.get("base_url"),
)
agent = Agent(model=OpenAIResponsesModel(model=model, openai_client=client))
```

**`llm.extra` pass-through:** All non-standard keys in the `llm:`
block are collected into `LLMConfig.extra` and mapped to the Agents
SDK's `ModelSettings`:

| `llm:` YAML key | `ModelSettings` field |
|-----------------|----------------------|
| `reasoning_effort` | `reasoning=Reasoning(effort=...)` |
| `max_completion_tokens` | `max_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| (other keys) | `extra_body={...}` pass-through |

### `tools:` block

```yaml
tools:
  builtins:
    - web_search_openai              # hosted tool — WebSearchTool
    - codex:Shell                    # Codex MCP tool
    - codex:ApplyPatch               # Codex MCP tool
  agents:
    - researcher                     # sub-agent (unchanged)
    - reviewer
  timeout: 60                        # per-tool timeout (default: 60)
  retry:
    max_attempts: 2
```

**Built-in tool handling:**

| Builtin prefix | How it's registered | Executor type |
|---------------|-------------------|---------------|
| `web_search_openai` | `WebSearchTool()` hosted tool on the Agent | SDK-native |
| `codex:*` | Codex MCP server attached to Agent | MCP |
| (no prefix) | Function tool wrapping `context.call_tool()` | Agent-plane |

**Sub-agents:** Work identically to all other executors. The
spawn/collect tools are registered by the workflow's `ToolManager`,
not the executor. No executor-specific sub-agent handling needed.

### `compaction:` block

**Forbidden.** The Agents SDK manages its own context window via
truncation (`ModelSettings.truncation`) and server-side compaction.
The validator rejects `compaction:` for `agents_sdk`, same as
`claude_sdk`.

### Skills

```yaml
skills/
  deep-search/
    SKILL.md
```

Skills work via agent-plane's `load_skill` and `read_skill_file`
function tools — **not** via SDK-native discovery.

The Agents SDK has no native skill system. Skills are a **Responses
API** feature (uploaded via `/v1/skills`, attached to the shell tool
via `tools[].environment.skills`), and the Agents SDK does not expose
this surface. OpenAI's recommendation is to use Codex for skills
([openai/openai-agents-python#2361](https://github.com/openai/openai-agents-python/issues/2361)).

Three options considered:

| Approach | Pros | Cons |
|----------|------|------|
| Agent-plane `load_skill` tool | Works for all models, no Codex dependency | LLM must decide to call the tool |
| Codex-native skills (via MCP) | Codex discovers skills automatically | Only works when Codex MCP is attached |
| Responses API skills | Server-side, model sees skills natively | Requires skill upload to OpenAI; SDK doesn't expose it |

**Decision:** Use agent-plane's `load_skill` / `read_skill_file`
function tools — same mechanism as `DefaultExecutor`. This works
universally regardless of whether Codex is attached. When the LLM
calls `load_skill`, the workflow injects the skill content into the
conversation as a tool result. If Codex MCP is attached and the
operator also configures skills inside Codex's own directory, Codex
can discover those independently — but that's the operator's choice,
not the executor's concern.

### MCP servers from agent spec (`tools/mcp/*.yaml`)

Agent-spec MCP servers are handled by the workflow's `ToolManager`,
not the executor. The `ToolManager` connects to each MCP server,
discovers tools, and registers them. These tools reach the executor
as function tool schemas in the `tools` parameter of `run_turn()`,
where they become Agents SDK function tools that call
`context.call_tool()`.

The executor does **not** pass agent-spec MCP servers to the Agents
SDK's `mcp_servers` parameter. Only the Codex MCP server (if
`codex:` tools are declared) uses that parameter.

Rationale: agent-plane already handles MCP connection lifecycle,
retry, timeout, and tool dispatch. Duplicating this in the SDK would
create two competing MCP connection managers.

---

## Spec Validation Rules

New function `_validate_agents_sdk_executor` in `validator.py`:

```python
def _validate_agents_sdk_executor(
    spec: AgentSpec, result: ValidationResult,
) -> None:
    # Remote-only fields are invalid
    if spec.executor.endpoint is not None:
        result.add(
            "executor.endpoint",
            "not supported when executor.type is 'agents_sdk'",
        )
    if spec.executor.request_timeout is not None:
        result.add(
            "executor.request_timeout",
            "not supported when executor.type is 'agents_sdk'"
            " — use llm.request_timeout instead",
        )
    # SDK manages compaction
    if spec.compaction is not None:
        result.add(
            "compaction",
            "not supported when executor.type is 'agents_sdk'"
            " — SDK manages context internally",
        )
```

**Differences from `claude_sdk` validation:**
- `llm.connection` is **allowed** (SDK supports custom OpenAI clients)
- `compaction` is **forbidden** (same as `claude_sdk`)
- `executor.endpoint` and `executor.request_timeout` are **forbidden**
  (same as `claude_sdk` and `llm`)

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
| `run_item_stream_event` + `tool_search_output_created` | `NativeToolOutput(item=raw_dict)` | Web search results (hosted tool) |
| `run_item_stream_event` + `mcp_approval_requested` | (auto-approve via approval callback) | Codex MCP tool approvals |
| Run completion (async iterator ends) | `TurnComplete(text=final_output)` | `result.final_output` has the text |
| `MaxTurnsExceeded` exception | `TurnComplete(text=partial)` | SDK hit its internal turn limit |
| `ModelBehaviorError`, `UserError` | `ExecutorError(message, code)` | Unrecoverable failures |

### `NativeToolOutput` for hosted tools

`web_search_openai` is registered as a `WebSearchTool()` hosted tool
on the Agent. When the SDK streams search results, they appear as
`tool_search_output_created` events. The executor wraps the raw item
dict in `NativeToolOutput` — same as `DefaultExecutor` does for
provider-native tool outputs. The workflow persists and streams them
to the client as-is.

No new event types needed.

---

## Design Decisions

### SDK manages the agent loop (partially)

When `Runner.run_streamed()` is called, the SDK:
1. Calls the LLM with messages + tools
2. If the LLM requests tool calls, executes them
3. Re-calls the LLM with results
4. Repeats until the LLM produces a final response or hits `max_turns`

This is the same ownership model as `ClaudeAgentsExecutor` — the
executor's `run_turn()` maps to one full SDK run (potentially many
LLM calls internally). The workflow sees a single turn with streamed
events.

`max_context_tokens()` returns `None` — the SDK manages context
truncation. The workflow skips compaction and the `@step` wrapper.

### Conversation state via history replay

Each `run_turn()` call passes the full message history to the SDK
via `result.to_input_list()` format. The SDK replays it into the LLM
context. No persistent SDK-side state.

If the process restarts, the workflow reloads history from the
conversation store and passes it in. No `.claude/`-style session
recovery needed. This matches `DefaultExecutor`'s approach.

We do **not** use `conversationId` (server-managed state) in v1.
History replay is simpler, portable (works with non-OpenAI models),
and doesn't create an external state dependency. Token cost is
managed by the SDK's truncation.

### Tools as function tools (no MCP bridge needed)

The Agents SDK supports `@function_tool`-decorated Python functions.
Agent-plane's server-side and client-side tools are registered as
function tools whose implementations call `context.call_tool()`:

```python
from agents import function_tool

def _make_tool(name: str, schema: dict, context: ExecutorContext):
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
bridge — no in-process MCP server, no callback holder, no threading
for tool dispatch. The function tool is a plain async function that
blocks on `call_tool()`, which parks for client-side tools or
executes immediately for server-side tools.

### Codex coding tools via MCP (optional)

When the agent spec declares `codex:Shell` or `codex:ApplyPatch` in
`tools.builtins`, the executor launches `codex mcp-server` as an MCP
server and attaches it to the Agent:

```python
from agents.mcp import MCPServerStdio

codex_mcp = MCPServerStdio(
    name="Codex CLI",
    params={"command": "codex", "args": ["mcp-server"]},
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

### `web_search_openai` as a hosted tool

When `web_search_openai` is in `tools.builtins`, the executor adds
`WebSearchTool()` to the Agent's `tools` list. This is a hosted tool
— the OpenAI API executes it server-side and returns results. The
executor surfaces these as `NativeToolOutput` events, matching the
`DefaultExecutor`'s behavior.

Other builtin tools (`web_search_google`, `web_search_perplexity`)
are server-executed function tools managed by `ToolManager` — they
reach the executor as regular function tool schemas.

### Sandbox: no executor-level sandboxing

The Agents SDK does not sandbox tool execution — function tools run
in-process. Sandboxing is handled at two levels:

1. **Agent-plane local tools** (`tools/python/*.py`): Already
   sandboxed by the `ToolManager` via subprocess isolation
   (srt, Docker, uv). The executor doesn't interfere.

2. **Codex MCP tools**: Codex manages its own sandbox
   (`workspaceWrite` policy). Configured via Codex's own
   settings when launching the MCP server.

If the agent spec sets `sandbox.docker_image`, it applies to
agent-plane's local Python tool execution — not to the Agents SDK
or Codex. No conflict.

### Workspace / `storage_dir` usage

The executor sets `cwd` on the Codex MCP server (if active) to
`context.storage_dir / "workspace"`, scoping Codex file operations.

For non-Codex usage, `storage_dir` is managed by the workflow for
artifact store I/O. The executor itself does not write to
`storage_dir` — unlike the Claude executor which writes skills and
session transcripts there.

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
        skills: list[SkillSpec] | None = None,
    ) -> None: ...

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """Extract model, codex tools, connection, and skills."""

    def max_context_tokens(self) -> int | None:
        return None  # SDK manages truncation

    def on_task_start(self, context: ExecutorContext) -> None:
        """No-op. Codex MCP launched lazily on first run_turn."""

    def on_task_end(self, context: ExecutorContext) -> None:
        """Shut down Codex MCP server if running."""

    def run_turn(
        self, messages, tools, system_prompt,
        llm_config, context,
    ) -> Iterator[ExecutorEvent]:
        """Run SDK agent, bridge async stream to sync iterator."""
```

### Threading model

The workflow calls `run_turn()` synchronously (DBOS thread pool). The
Agents SDK is async. Bridge via the same event-queue pattern as the
Claude executor:

```python
def run_turn(self, messages, tools, system_prompt,
             llm_config, context) -> Iterator[ExecutorEvent]:
    event_queue: queue.Queue[ExecutorEvent | None] = queue.Queue(
        maxsize=256,
    )
    loop = _get_or_create_loop(context.conversation_id)
    asyncio.run_coroutine_threadsafe(
        _async_run(self, messages, tools, system_prompt,
                   llm_config, context, event_queue),
        loop,
    )
    while True:
        event = event_queue.get()
        if event is None:
            break
        yield event
```

**Per-conversation event loop.** Same pattern as `_ClientRegistry` in
the Claude executor. Each `conversation_id` gets a dedicated
`asyncio.AbstractEventLoop` on a background daemon thread. The loop
persists across tasks so the Codex MCP subprocess (if active) stays
alive between turns and shell state survives.

```python
class _LoopRegistry:
    """Per-conversation event loops. Mirrors _ClientRegistry."""
    _loops: dict[str, tuple[asyncio.AbstractEventLoop, threading.Thread]]
    _codex_mcps: dict[str, MCPServerStdio]  # optional, per conversation
    _ttl: float = 3600.0

    def get_or_create_loop(self, conv_id: str) -> asyncio.AbstractEventLoop: ...
    def register_codex_mcp(self, conv_id: str, mcp: MCPServerStdio) -> None: ...
    def evict_stale(self) -> None: ...
```

Stale event loops (idle > TTL) are evicted by stopping the loop,
shutting down any Codex MCP server, and joining the thread.

Fresh event loops are **not** created per `run_turn()` call — that
would be expensive and would kill the Codex MCP subprocess. The loop
lives for the lifetime of the conversation.

### LLM config mapping

```python
def _build_model_settings(llm_config: LLMConfig) -> ModelSettings:
    from agents import ModelSettings
    from openai.types.shared import Reasoning

    extra = dict(llm_config.extra)
    reasoning_effort = extra.pop("reasoning_effort", None)

    settings = ModelSettings(
        temperature=extra.pop("temperature", None),
        top_p=extra.pop("top_p", None),
        max_tokens=extra.pop("max_completion_tokens", None),
    )
    if reasoning_effort:
        settings.reasoning = Reasoning(
            effort=reasoning_effort,
            summary="detailed",
        )
    if extra:
        settings.extra_body = extra
    return settings


def _build_openai_client(
    connection: dict[str, str] | None,
) -> AsyncOpenAI | None:
    if connection is None:
        return None  # SDK uses OPENAI_API_KEY from env
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=connection.get("api_key"),
        base_url=connection.get("base_url"),
    )
```

### Timeout / retry mapping

| Agent-plane config | Agents SDK mapping |
|-------------------|-------------------|
| `executor.timeout` | Workflow-level task deadline (unchanged) |
| `executor.max_iterations` | Workflow's outer loop limit (unchanged) |
| `llm.request_timeout` | `AsyncOpenAI(timeout=...)` on the client |
| `llm.retry` | `AsyncOpenAI(max_retries=...)` on the client |

The SDK's `max_turns` parameter on `Runner.run_streamed()` is set to
`200` (high ceiling). The workflow's `max_iterations` is the
authoritative loop limit. If the SDK exhausts `max_turns` before
the workflow hits `max_iterations`, the executor catches
`MaxTurnsExceeded` and returns `TurnComplete(text=partial)`. The
workflow then calls `run_turn()` again with updated history (if
`max_iterations` budget remains).

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

    function_tools = _build_function_tools(tools, context)
    hosted_tools = _build_hosted_tools(executor)
    mcp_servers = _build_mcp_servers(executor, context)
    model_settings = _build_model_settings(llm_config)
    client = _build_openai_client(executor._connection)

    model = _build_model(llm_config.model, client)
    agent = Agent(
        name="agent",
        instructions=system_prompt,
        model=model,
        model_settings=model_settings,
        tools=[*function_tools, *hosted_tools],
        mcp_servers=mcp_servers,
    )

    input_items = _messages_to_input(messages)
    try:
        result = Runner.run_streamed(
            agent,
            input=input_items,
            max_turns=200,
        )
        async for event in result.stream_events():
            _map_event(event, event_queue)
        event_queue.put(TurnComplete(text=result.final_output))
    except MaxTurnsExceeded:
        event_queue.put(TurnComplete(text=None))
    except Exception as exc:
        event_queue.put(ExecutorError(
            message=f"Agents SDK error: {exc}",
            code=type(exc).__name__,
        ))
    finally:
        event_queue.put(None)


def _map_event(event: StreamEvent, eq: queue.Queue) -> None:
    from openai.types.responses import (
        ResponseTextDeltaEvent,
        ResponseReasoningSummaryTextDeltaEvent,
    )

    if event.type == "raw_response_event":
        data = event.data
        if isinstance(data, ResponseTextDeltaEvent):
            eq.put(TextChunk(text=data.delta))
        elif isinstance(data, ResponseReasoningSummaryTextDeltaEvent):
            eq.put(ReasoningChunk(
                delta=data.delta,
                event_type="reasoning_summary",
            ))
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
        elif event.name == "tool_search_output_created":
            eq.put(NativeToolOutput(item=event.item.raw_item))
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
| `runtime/executors/agents_sdk.py` | New file — `AgentsSdkExecutor`, `_LoopRegistry`, event mapping |
| `runtime/executors/__init__.py` | Export `AgentsSdkExecutor` |
| `runtime/workflow.py:_create_executor` | Add `"agents_sdk"` branch |
| `spec/types.py:ExecutorSpec` | Update docstring to include `"agents_sdk"` |
| `spec/validator.py` | Add `"agents_sdk"` to `_VALID_EXECUTOR_TYPES`, add `_validate_agents_sdk_executor` |
| `spec/AGENTSPEC.md` | Document `executor.type: agents_sdk`, `codex:` tool prefix, validation rules |
| `tests/runtime/test_executor.py` | Unit tests for event mapping, model settings mapping |
| `tests/spec/test_validator.py` | Validation tests for `agents_sdk` executor type |
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

tools:
  builtins:
    - web_search_openai

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
  timeout: 1800

llm:
  model: gpt-5.4
  reasoning_effort: medium
  max_completion_tokens: 8192
  connection:
    api_key: ${OPENAI_API_KEY}

tools:
  builtins:
    - codex:Shell
    - codex:ApplyPatch
    - web_search_openai
  agents:
    - reviewer

skills/
  code-review/
    SKILL.md

instructions: INSTRUCTIONS.md
```

**With sub-agents (showing full feature set):**

```yaml
# examples/agents/openai-supervisor/config.yaml
spec_version: 1

name: openai-supervisor
description: Supervisor that delegates to specialized sub-agents.

executor:
  type: agents_sdk
  max_iterations: 500

llm:
  model: gpt-5.4
  connection:
    api_key: ${OPENAI_API_KEY}

tools:
  agents:
    - researcher
    - coder
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

1. **Codex MCP lifecycle.** The `codex mcp-server` subprocess is
   started lazily on the first `run_turn()` with `codex:` tools and
   persists in the `_LoopRegistry` across tasks. Should the Codex
   thread ID be stored so `codex-reply` can continue conversations
   across turns? Current leaning: yes — the executor tracks the
   Codex `threadId` returned by the first `codex` tool call and
   injects it into subsequent calls as `codex-reply`.

2. **Non-OpenAI models.** The Agents SDK supports arbitrary providers
   via its LiteLLM integration. Should `executor.type: agents_sdk`
   be the recommended path for all non-Anthropic models, replacing
   `DefaultExecutor` over time? Or keep both — `DefaultExecutor`
   for simple single-call LLM turns, `AgentsSdkExecutor` for
   SDK-managed multi-call agent loops? Current leaning: keep both.
   `DefaultExecutor` is simpler (one LLM call per turn, workflow owns
   the loop), `AgentsSdkExecutor` is richer (SDK owns the loop, can
   do multi-step tool chains within a single turn).

3. **`call_tool` threading.** `context.call_tool()` is a blocking
   call that runs in the DBOS thread pool. The Agents SDK's function
   tool runs in the per-conversation async event loop. We need
   `await asyncio.to_thread(context.call_tool, req)` to avoid
   blocking the event loop. This is straightforward but needs
   testing under concurrent tool calls.

---

## Test Plan

Three test layers, mirroring the existing executor test structure:
unit tests (fast, no IO), server integration tests (real stores +
mock LLM), and e2e tests (real LLM + real server). Each test
documents what production breakage would cause it to fail.

### Unit Tests — `tests/runtime/test_agents_sdk_executor.py`

Fast, no network, no stores. Test event mapping, config mapping,
and helper functions in isolation. Pattern: monkeypatch SDK imports
and stream factories, assert on executor event output.

#### Event mapping

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_map_event_text_delta` | `ResponseTextDeltaEvent` → `TextChunk(text=delta)` | Streamed text not delivered to client |
| `test_map_event_reasoning_summary_delta` | `ResponseReasoningSummaryTextDeltaEvent` → `ReasoningChunk(delta, "reasoning_summary")` | Reasoning output lost for reasoning models |
| `test_map_event_tool_called` | `run_item_stream_event` + `tool_called` → `ToolCallObserved` with parsed arguments, result, status | Tool call history missing from persisted output |
| `test_map_event_tool_called_error_status` | Tool call with error output → `ToolCallObserved(status="error")` | Failed tool calls silently reported as success |
| `test_map_event_web_search_output` | `tool_search_output_created` → `NativeToolOutput(item=raw_dict)` | Web search results not surfaced to client |
| `test_map_event_ignores_unknown_raw_events` | Unknown `raw_response_event` subtypes → no event queued | Crash on new SDK event types |
| `test_map_event_ignores_unknown_run_item_events` | Unknown `run_item_stream_event` names → no event queued | Crash on new SDK item types |

#### Turn completion

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_turn_complete_with_final_output` | Async iterator ends → `TurnComplete(text=result.final_output)` | Turn never completes; workflow hangs |
| `test_turn_complete_on_max_turns_exceeded` | `MaxTurnsExceeded` caught → `TurnComplete(text=None)` | Unhandled exception kills the task |
| `test_executor_error_on_model_behavior_error` | `ModelBehaviorError` → `ExecutorError(message, code)` | Bad model output kills the task with no useful error |
| `test_executor_error_on_generic_exception` | Arbitrary exception → `ExecutorError` with class name as code | Unhandled crash with no error info in response |
| `test_sentinel_always_sent` | `None` sentinel pushed to queue in `finally` block, even after exceptions | Sync iterator in `run_turn` blocks forever |

#### LLM config mapping

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_build_model_settings_basic` | `model` and default `ModelSettings` | Agent fails to call the LLM |
| `test_build_model_settings_reasoning_effort` | `reasoning_effort: high` → `Reasoning(effort="high", summary="detailed")` | Reasoning models don't reason; summary events missing |
| `test_build_model_settings_temperature` | `temperature: 0.7` → `ModelSettings(temperature=0.7)` | Temperature not applied; output entropy wrong |
| `test_build_model_settings_max_completion_tokens` | `max_completion_tokens: 4096` → `ModelSettings(max_tokens=4096)` | Output truncated or unbounded |
| `test_build_model_settings_extra_passthrough` | Unknown keys → `extra_body={...}` | Provider-specific params silently dropped |
| `test_build_openai_client_with_connection` | `connection: {api_key, base_url}` → `AsyncOpenAI(...)` | Wrong API key or wrong endpoint |
| `test_build_openai_client_none_uses_env` | `connection: None` → returns `None` (SDK reads env) | Env-based auth broken |

#### `from_spec` construction

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_from_spec_extracts_model` | `spec.llm.model` → `executor._model` | Wrong model used |
| `test_from_spec_extracts_codex_tools` | `codex:Shell` in builtins → `executor._codex_tools == ["Shell"]` | Codex MCP not launched when expected |
| `test_from_spec_extracts_connection` | `spec.llm.connection` → `executor._connection` | Auth not passed to SDK client |
| `test_from_spec_no_llm_raises` | `spec.llm is None` → `AssertionError` | Crash later with cryptic error |
| `test_max_context_tokens_returns_none` | `executor.max_context_tokens() is None` | Workflow tries to compact; `@step` wrapper applied to SDK-managed turn |

#### Function tool wrappers

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_make_function_tool_calls_call_tool` | Wrapper calls `context.call_tool(ToolCallRequested(...))` and returns `result.content` | Tools silently no-op |
| `test_make_function_tool_preserves_name_and_desc` | `name_override` and `description_override` set correctly | LLM sees wrong tool name/description |
| `test_make_function_tool_passes_kwargs` | Tool arguments forwarded as `ToolCallRequested.arguments` | Tool receives empty/wrong arguments |
| `test_make_function_tool_error_result` | `call_tool` returns `status="error"` → still returns content (SDK handles retry) | Error swallowed or wrong error propagation |

#### Hosted tool construction

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_build_hosted_tools_web_search` | `web_search_openai` in builtins → `WebSearchTool()` in tools list | Web search not available to agent |
| `test_build_hosted_tools_empty` | No `web_search_openai` → empty list | Spurious tools registered |

#### Codex MCP construction

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_build_mcp_servers_with_codex_tools` | `codex_tools=["Shell"]` → `MCPServerStdio` with `codex mcp-server` | Codex not launched; Shell/ApplyPatch unavailable |
| `test_build_mcp_servers_empty` | `codex_tools=[]` → empty list | Codex launched unnecessarily |
| `test_build_mcp_servers_sets_cwd` | `cwd` set to `storage_dir/workspace` | Codex file ops target wrong directory |

#### `_LoopRegistry`

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_loop_registry_creates_loop_per_conversation` | Different `conv_id` → different loops | Cross-conversation interference |
| `test_loop_registry_reuses_existing_loop` | Same `conv_id` → same loop | Codex MCP killed between turns; shell state lost |
| `test_loop_registry_evicts_stale` | Idle > TTL → loop stopped, thread joined | Resource leak (threads + subprocesses accumulate) |

---

### Server Integration Tests — `tests/server/integration/test_agents_sdk_integration.py`

Real stores (SQLAlchemy), real DBOS workflow, mock LLM via a custom
`AgentsSdkExecutor` subclass that yields canned events. Tests the
full path: HTTP request → workflow → executor → event persistence →
HTTP response. Pattern: same as `test_executor_integration.py`.

#### Custom test executor

```python
class _CannedAgentsSdkExecutor(Executor):
    """
    Test executor that yields predetermined events.

    Simulates the Agents SDK without importing it.
    No MagicMock — real Executor subclass so isinstance() works.
    """
    def __init__(self, events: list[ExecutorEvent]): ...
    def from_spec(cls, spec): ...
    def max_context_tokens(self) -> int | None: return None
    def run_turn(self, ...) -> Iterator[ExecutorEvent]:
        yield from self._events
```

Monkeypatch `_create_executor` to return the canned executor for
`type == "agents_sdk"`.

#### Test scenarios

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_text_streaming` | `TextChunk` events → SSE `response.output_text.delta` events → final `response.completed` with assembled text | Text not streamed or not persisted |
| `test_agents_sdk_tool_observed_persisted` | `ToolCallObserved` → `function_call` + `function_call_output` items in GET output and conversation store | Tool call history lost; client can't see what tools ran |
| `test_agents_sdk_native_tool_output_persisted` | `NativeToolOutput` → preserved in GET output as native item | Web search results lost |
| `test_agents_sdk_reasoning_chunks_streamed` | `ReasoningChunk` → SSE reasoning events | Reasoning output not visible to client |
| `test_agents_sdk_client_tool_park_patch_resume` | `ToolCallRequested` → `action_required` on response → PATCH result → executor continues → `TurnComplete` | Client-side tool calls hang or results lost |
| `test_agents_sdk_error_produces_failed_response` | `ExecutorError` → response `status: "failed"` with error details | Errors silently swallowed; response stuck in `in_progress` |
| `test_agents_sdk_multi_turn_history_replay` | Turn 1 completes → Turn 2 with `previous_response_id` → executor receives full history in `messages` | Context lost between turns; agent has no memory |
| `test_agents_sdk_second_turn_after_tool_calls` | Turn 1 with `ToolCallObserved` → Turn 2 → no regression in history loading | Tool call items corrupt the message history on replay |
| `test_agents_sdk_background_and_foreground` | `background=True` returns immediately; `background=False` blocks | Request hangs or returns before completion |

#### Validation integration

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_rejects_compaction_config` | Upload bundle with `compaction:` block → validation error | Invalid config silently accepted; compaction logic conflicts with SDK |
| `test_agents_sdk_rejects_endpoint` | Upload bundle with `executor.endpoint` → validation error | Remote-only field accepted for wrong executor type |
| `test_agents_sdk_accepts_connection` | Upload bundle with `llm.connection` → success | Auth config rejected (unlike `claude_sdk` which forbids it) |

---

### E2E Tests — `tests/e2e/test_agents_sdk_*.py`

Real LLM (OpenAI API), real server subprocess, real agent bundles.
Pattern: same as `test_claude_coder_*.py`. Uses `--llm-api-key`
CLI option. LLM judge (via `mlflow.genai.judges.make_judge`)
evaluates response quality where string matching is insufficient.

#### Fixtures (in `tests/e2e/conftest.py`)

```python
# Agent bundle directories
_OPENAI_BASIC_DIR = _REPO_ROOT / "examples" / "agents" / "openai-basic"
_OPENAI_CODER_DIR = _REPO_ROOT / "examples" / "agents" / "openai-coder"

@pytest.fixture(scope="session")
def openai_basic_agent(http_client: httpx.Client) -> str:
    """Upload the openai-basic agent and return its name."""
    return _upload_agent(http_client, _OPENAI_BASIC_DIR)

@pytest.fixture(scope="session")
def openai_coder_agent(http_client: httpx.Client) -> str:
    """Upload the openai-coder agent (with Codex MCP) and return its name."""
    return _upload_agent(http_client, _OPENAI_CODER_DIR)
```

#### `test_agents_sdk_basic.py` — Core agent loop

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_single_turn_completes` | POST → poll → `status: "completed"` with non-empty text output | Agent loop doesn't run; SDK misconfigured |
| `test_agents_sdk_streaming_delivers_text_deltas` | SSE stream contains `response.output_text.delta` events that concatenate to the final text | Streaming broken; client sees nothing until completion |

**`test_agents_sdk_single_turn_completes` detail:**

```python
def test_agents_sdk_single_turn_completes(
    http_client: httpx.Client,
    openai_basic_agent: str,
) -> None:
    """
    Basic smoke test: the Agents SDK executor runs a single
    turn and produces a completed response.

    What breaks if wrong:
    - If the SDK import fails, from_spec raises ImportError.
    - If model_settings mapping is wrong, the LLM rejects params.
    - If the async/sync bridge deadlocks, poll times out.
    - If TurnComplete is never emitted, status stays in_progress.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": openai_basic_agent,
            "input": "What is 2 + 2? Reply with just the number.",
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=60)
    assert body["status"] == "completed"

    text = _extract_all_text(body)
    assert "4" in text, f"Expected '4' in response: {text}"
```

#### `test_agents_sdk_multi_turn.py` — Conversation continuity

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_remembers_prior_turn` | Turn 1 states a fact → Turn 2 asks about it → response references Turn 1 content | History replay broken; agent has amnesia |

**`test_agents_sdk_remembers_prior_turn` detail:**

Turn 1: "My name is Zephyr and I live in Portland." Turn 2:
"What's my name and where do I live?" LLM judge verifies the
response contains both "Zephyr" and "Portland". If history replay
(`_messages_to_input`) is broken, the agent can't recall Turn 1.

#### `test_agents_sdk_web_search.py` — Hosted tool (web search)

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_web_search_executes` | Ask a current-events question → response contains factual answer with search evidence | `WebSearchTool` not registered; hosted tool dispatch broken |

Uses `openai_basic_agent` (which declares `web_search_openai`).
Asks "What was the most recent Nobel Prize in Physics awarded for?"
LLM judge checks the response cites a real, recent award. If
`WebSearchTool()` is not added to the Agent's tools list, the model
either hallucinates or says it can't search.

#### `test_agents_sdk_client_tools.py` — Client-side tool parking

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_parks_client_tool_call` | Register `get_current_time` tool → agent calls it → `action_required` appears → PATCH result → agent completes referencing the time | Function tool wrapper doesn't call `context.call_tool`; parking broken; PATCH result not delivered |

Same pattern as `test_claude_coder_client_tools.py`:

1. POST with `tools: [{type: "function", function: {name: "get_current_time", ...}}]`
2. Poll for `action_required` function_call
3. PATCH with tool result
4. Poll until completed
5. Assert response references the provided time

#### `test_agents_sdk_codex.py` — Codex MCP integration

Requires `codex` binary on PATH. Tests are skipped if unavailable.

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_codex_shell_executes_command` | Ask to list files in a directory → response contains actual file listing | Codex MCP not launched; `codex mcp-server` subprocess fails; tool dispatch broken |
| `test_codex_creates_and_reads_file` | Ask to create a file then read it back → response contains the file content | ApplyPatch tool broken; Codex sandbox blocks writes; workspace cwd wrong |

Uses `openai_coder_agent` (which declares `codex:Shell` and
`codex:ApplyPatch`).

**`test_codex_shell_executes_command` detail:**

```python
@pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex binary not on PATH",
)
def test_codex_shell_executes_command(
    http_client: httpx.Client,
    openai_coder_agent: str,
    llm_api_key: str,
) -> None:
    """
    Codex MCP Shell tool executes a real command.

    What breaks if wrong:
    - If MCPServerStdio fails to start, Runner.run_streamed errors.
    - If codex mcp-server is not found, subprocess.Popen raises.
    - If cwd is wrong, the command runs in an unexpected directory.
    - If approval_policy isn't "never", the command hangs waiting.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": openai_coder_agent,
            "input": "Run `echo hello_from_codex` in the shell and tell me the output.",
            "background": True,
        },
    )
    resp.raise_for_status()
    body = poll_until_terminal(http_client, resp.json()["id"], timeout=120)
    assert body["status"] == "completed"

    text = _extract_all_text(body)
    assert "hello_from_codex" in text.lower(), (
        f"Expected 'hello_from_codex' in output: {text[:500]}"
    )
```

#### `test_agents_sdk_subagent.py` — Sub-agent spawn/collect

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_spawns_subagent` | Ask agent to spawn reviewer sub-agent → sub-agent runs → parent collects result | `spawn_sub_agents` function tool wrapper broken; sub-agent spec lookup fails; auto-collect fails |

Same structure as `test_claude_coder_subagent.py`: create a file
with code issues → ask the agent to delegate review to its
`reviewer` sub-agent → verify spawn_sub_agents was called → LLM
judge evaluates review quality.

Uses `openai_coder_agent` (which declares `agents: [reviewer]`).

#### `test_agents_sdk_skills.py` — Skill loading

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_loads_skill_via_tool` | Ask agent to load a skill → response contains skill content | `load_skill` function tool not registered; skill spec not passed to workflow |

Ask the agent to "list your skills and load the code-review skill."
LLM judge verifies the response mentions the skill name and shows
content from `SKILL.md` (e.g. the review format or priority ordering).

Structurally similar to `test_claude_coder_skills.py` but tests the
function-tool path (agent-plane's `load_skill`) rather than the
Claude SDK's native Skill tool.

#### `test_agents_sdk_reasoning.py` — Reasoning model output

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_reasoning_model_streams_summary` | Use a reasoning model (e.g. `o3`) → SSE stream contains reasoning summary events | `reasoning_effort` not mapped to `ModelSettings.reasoning`; summary events not emitted |

Uses a separate agent config with `reasoning_effort: medium` and a
reasoning-capable model. Sends a math problem and verifies the SSE
stream contains reasoning-related events (either via
`ReasoningChunk` executor events or raw model reasoning summaries).

#### `test_agents_sdk_error_handling.py` — Failure modes

| Test | What it verifies | What breaks if wrong |
|------|-----------------|---------------------|
| `test_agents_sdk_invalid_api_key_produces_failed` | Upload agent with bogus `api_key` → task fails with auth error | Error swallowed; response stuck in `in_progress` forever |

Upload a special agent bundle with `connection.api_key: "sk-invalid"`.
POST a response → poll → assert `status: "failed"` and error message
contains "auth" or "unauthorized" or "invalid".

---

### Test file summary

| File | Layer | Count | Dependencies |
|------|-------|-------|-------------|
| `tests/runtime/test_agents_sdk_executor.py` | Unit | ~30 | None (monkeypatch SDK) |
| `tests/spec/test_validator.py` (additions) | Unit | ~3 | None |
| `tests/server/integration/test_agents_sdk_integration.py` | Integration | ~12 | Real stores, mock executor |
| `tests/e2e/test_agents_sdk_basic.py` | E2E | 2 | Real LLM, real server |
| `tests/e2e/test_agents_sdk_multi_turn.py` | E2E | 1 | Real LLM, real server |
| `tests/e2e/test_agents_sdk_web_search.py` | E2E | 1 | Real LLM, real server |
| `tests/e2e/test_agents_sdk_client_tools.py` | E2E | 1 | Real LLM, real server |
| `tests/e2e/test_agents_sdk_codex.py` | E2E | 2 | Real LLM, real server, `codex` binary |
| `tests/e2e/test_agents_sdk_subagent.py` | E2E | 1 | Real LLM, real server |
| `tests/e2e/test_agents_sdk_skills.py` | E2E | 1 | Real LLM, real server |
| `tests/e2e/test_agents_sdk_reasoning.py` | E2E | 1 | Real LLM (reasoning model), real server |
| `tests/e2e/test_agents_sdk_error_handling.py` | E2E | 1 | Real server (intentionally bad key) |
| **Total** | | **~56** | |

### Running the tests

```bash
# Unit + integration (no API key needed)
pytest tests/runtime/test_agents_sdk_executor.py tests/server/integration/test_agents_sdk_integration.py -v

# E2E (requires real API key)
pytest tests/e2e/test_agents_sdk_*.py --llm-api-key $(cat /tmp/openai_key) -v

# E2E Codex tests only (requires codex binary)
pytest tests/e2e/test_agents_sdk_codex.py --llm-api-key $(cat /tmp/openai_key) -v
```
