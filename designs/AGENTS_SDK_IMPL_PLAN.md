# AgentsSdkExecutor Implementation Plan

## Context

The executor plugin system needs a fourth backend wrapping the OpenAI
Agents SDK (`openai-agents` on PyPI). The design doc
(`designs/AGENTS_SDK_EXECUTOR.md`) is complete and all prerequisite
async refactors are shipped (async `run_turn`, async `call_tool`, async
LLM client). This plan implements Layer 1 (MVP — no Codex) only.

## Implementation Order

### Step 1: Spec + validator changes (small, testable immediately)

**Files:**
- `agent_plane/spec/types.py` — update `ExecutorSpec.type` docstring
  to include `"agents_sdk"`
- `agent_plane/spec/validator.py` — add `"agents_sdk"` to
  `_VALID_EXECUTOR_TYPES`, add `_validate_agents_sdk_executor()`
  (forbids `endpoint`, `request_timeout`, `compaction`; allows
  `connection`)
- `tests/spec/test_validator.py` — 3 tests: rejects compaction,
  rejects endpoint, accepts connection

Pattern: mirror `_validate_claude_sdk_executor` but omit the
`llm.connection` check.

### Step 2: Core executor file

**New file:** `agent_plane/runtime/executors/agents_sdk.py`

Functions/classes in order:

1. `_ensure_sdk()` — lazy import of `agents` module with clear
   error message (pattern: `claude.py:60-75`)

2. `_extract_codex_tools(spec)` — extract `codex:`-prefixed tool
   names from builtins (pattern: `_extract_claude_tools` in
   `claude.py:1363-1378`)

3. `_has_web_search(builtins)` — check if `web_search_openai` is
   in builtins list

4. `_build_model_settings(llm_config)` — map `LLMConfig` to
   `ModelSettings`. Keys: `reasoning_effort` → `Reasoning(effort,
   summary="detailed")`, `temperature`, `top_p`,
   `max_completion_tokens` → `max_tokens`, rest → `extra_body`

5. `_build_openai_client(connection, timeout, max_retries)` —
   build `AsyncOpenAI` from connection params, or `None` for env
   defaults

6. `_build_model(model_name, client)` — wrap in
   `OpenAIResponsesModel` when custom client provided, else return
   string

7. `_make_function_tool(schema, context)` — wrap agent-plane tool
   schema as Agents SDK `FunctionTool`. Uses `context.call_tool`
   (async) directly. Key: tool parameters come from
   `schema["function"]["parameters"]` JSON schema.

8. `_messages_to_input(messages)` — pass through (already
   Responses API format). May need filtering if SDK rejects
   certain item types.

9. `_map_event(event, event_queue)` — map SDK `StreamEvent` to
   `ExecutorEvent`. Mappings per design doc:
   - `raw_response_event` + `ResponseTextDeltaEvent` → `TextChunk`
   - `raw_response_event` + reasoning summary → `ReasoningChunk`
   - `run_item_stream_event` + `tool_called` → `ToolCallObserved`
   - `run_item_stream_event` + `tool_search_output_created` →
     `NativeToolOutput`
   - Unknown events → silently ignored

10. `AgentsSdkExecutor(Executor)` — main class:
    - `__init__(model, codex_tools, builtins, connection,
      request_timeout, max_retries)`
    - `from_spec(spec)` — extract model, codex tools, builtins,
      connection, timeout, retries from AgentSpec
    - `max_context_tokens()` → `None`
    - `on_task_start()` — no-op (Layer 1)
    - `on_task_end()` — no-op (Layer 1)
    - `run_turn()` — async generator:
      1. Build function tools from `tools` param + `context`
      2. Build hosted tools (WebSearchTool if applicable)
      3. Build model settings from `llm_config`
      4. Build OpenAI client from connection
      5. Create `Agent(name, instructions, model, model_settings,
         tools)`
      6. `Runner.run_streamed(agent, input=messages, max_turns=200)`
      7. `async for event in result.stream_events()`: map + yield
      8. Yield `TurnComplete(text=result.final_output)`
      9. Catch `MaxTurnsExceeded` → `TurnComplete(text=None)`
      10. Catch exceptions → `ExecutorError`

**Key decision: NO queue bridge for Layer 1.** The Agents SDK is
async-native — `Runner.run_streamed()` runs in the caller's event
loop. Since `run_turn` is `async def`, we call the SDK directly.
No per-conversation event loop, no threading bridge. This is
simpler than the Claude executor and correct because the SDK has
no loop-binding constraint. Queue bridge deferred to Layer 2 when
Codex MCP needs a persistent event loop.

### Step 3: Wiring (small changes)

**Files:**
- `runtime/workflow.py:_create_executor` — add `"agents_sdk"`
  branch with lazy import (pattern: claude_sdk branch)
- `runtime/executors/__init__.py` — import + export
  `AgentsSdkExecutor`
- `pyproject.toml` — add `agents-sdk = ["openai-agents>=0.1,<1"]`
  to optional-dependencies

### Step 4: Example config

**New directory:** `examples/agents/openai-basic/`
- `config.yaml` — `executor.type: agents_sdk`, model `gpt-5.4`,
  `web_search_openai` builtin, `INSTRUCTIONS.md` reference
- `INSTRUCTIONS.md` — simple assistant prompt

### Step 5: Unit tests

**New file:** `tests/runtime/test_agents_sdk_executor.py`

Tests mock the `agents` module — no real SDK import needed. Use
`monkeypatch` to replace `_ensure_sdk()` return value with mock
objects.

Test categories (~30 tests per design doc):
- Event mapping (7): text delta, reasoning, tool called, error
  status, web search, unknown raw, unknown run item
- Turn completion (4): final output, max turns exceeded, model
  error, generic exception
- LLM config mapping (7): basic, reasoning_effort, temperature,
  max_completion_tokens, extra passthrough, client with
  connection, client None
- from_spec (5): model, codex tools, connection, no llm raises,
  max_context_tokens
- Function tool wrappers (4): calls call_tool, preserves name/
  desc, passes kwargs, error result
- Hosted tools (2): web search present, empty

### Step 6: Review subagent + test subagent

Run mandatory post-change review (CLAUDE.md checklist) and test
authoring review on all new/changed files.

### Step 7: Pre-commit + full test run

`pre-commit run --all-files` then
`python -m pytest tests/ --ignore=tests/e2e`

## Reusable patterns (with file paths)

| Pattern | Source | Line |
|---------|--------|------|
| Lazy SDK import | `claude.py` | 60-75 |
| Tool prefix extraction | `claude.py` | 1363-1378 |
| from_spec classmethod | `claude.py` | 415-430 |
| Validator per-type function | `validator.py` | 170-202 |
| Dispatch branch | `workflow.py` | 170-204 |
| __init__.py exports | `executors/__init__.py` | 27-69 |

## NOT in scope (Layer 2)

- `_CodexSessionRewriter`
- `_SessionAwareMcpServer`
- `_LoopRegistry` (per-conversation event loops)
- Codex MCP integration
- `examples/agents/openai-coder/`
- Codex-specific tests
- E2E tests (need real API key + installed SDK)

## Verification

1. `pre-commit run --all-files` — ruff format, ruff check, mypy
2. `python -m pytest tests/spec/test_validator.py -xvs`
3. `python -m pytest tests/runtime/test_agents_sdk_executor.py -xvs`
4. `python -m pytest tests/ --ignore=tests/e2e --timeout=120`
5. Review subagent on all changed files
6. Test authoring subagent on all test files
