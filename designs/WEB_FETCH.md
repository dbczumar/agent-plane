# `web_fetch` built-in tool

## Context

Agent plane agents need a way to look up web content without requiring
third-party API keys (Google, Perplexity) or being locked to OpenAI's
native web search. The onboarding assistant in particular needs to look
up docs, MCP registries, and debug resources during agent creation.

The existing `web_search` tool requires either an OpenAI model
(passthrough) or explicit search API credentials in the spec config.
`web_fetch` is a zero-config alternative that always works — it spawns
a built-in sub-agent with a code sandbox to search the web and fetch
page content using plain HTTP.

## Design

**`web_fetch` is a blocking function tool that internally spawns a
built-in sub-agent.** The LLM calls `web_fetch(query, url?)`, the tool
spawns a web researcher sub-agent with `code_sandbox`, waits for it to
complete, and returns the result. From the calling agent's perspective,
it's a synchronous tool call — the sub-agent mechanics are hidden.

### Executor support

`web_fetch` requires the `llm` executor (the default). The `claude_sdk`
and `agents_sdk` executors do not support sub-agents today. If an agent
using those executors declares `web_fetch`, the tool registers but
returns an error at invoke time explaining the limitation.

### Why a sub-agent, not a subprocess

A subprocess is one-shot: run a script, get output. A sub-agent can:

- **Iterate.** First search fails? Try a different query. Page is
  paywalled? Try the cached version. Need to follow links? Do it.
- **Reason.** The LLM inside the sub-agent decides what's relevant,
  extracts the right sections, and summarizes findings.
- **Use existing infrastructure.** Spawn, code sandbox, and auto-collect
  all exist. Zero new execution machinery needed.

### Tool schema

```
name: web_fetch
parameters:
  query: string (required) — what to look up
  url: string (optional) — a starting URL to fetch. The sub-agent
    will try this URL first, but if the content doesn't answer the
    query, it will search for and fetch other URLs. If omitted,
    the sub-agent searches the web directly.
```

### Behavior

1. Tool receives `query` (and optional `url`)
2. Spawns the built-in `__web_researcher` sub-agent via `_spawn_one()`
   with input constructed from the query and url
3. Polls `task_store` until the sub-agent reaches terminal state
4. Extracts text output from the sub-agent's response
5. Returns the text to the calling agent (blocking)

If the sub-agent fails or times out, returns an error string the
calling LLM can reason about.

### Comparison with `web_search`

| | `web_search` | `web_fetch` |
|---|---|---|
| **API keys** | Required (Google/Perplexity/OpenAI) | None |
| **Query support** | Yes | Yes |
| **URL fetch** | No | Yes |
| **Output** | Titles + URLs + snippets | Actual page content + summary |
| **Quality** | Higher (proper search API) | Good (LLM-guided scraping) |
| **Config needed** | `search_provider` + `api_key` | Nothing |
| **Executor** | Any | `llm` only |
| **Implementation** | Direct HTTP / passthrough | Sub-agent with code sandbox |

## Architecture

```
Parent agent (llm executor)
  └─ calls web_fetch(query="how to configure MCP servers")
       └─ WebFetchTool.invoke()
            ├─ _spawn_one(agent_name="__web_researcher", input=query)
            ├─ poll task_store until terminal
            └─ return sub-agent output text
                 │
                 └─ __web_researcher sub-agent (built-in)
                      ├─ has: code_sandbox
                      ├─ inherits parent's LLM model + credentials
                      ├─ instructions: search web, fetch pages,
                      │  extract relevant content, summarize
                      └─ uses urllib/curl in sandbox to fetch
```

### Model inheritance

Sub-agents in agent plane do NOT inherit the parent's model — each
needs its own explicit `llm` block. For `web_fetch`, this is solved
at registration time and invoke time:

1. `ToolManager` creates `WebFetchTool` and passes `self._spec`
   (the parent's full AgentSpec including LLM config)
2. `WebFetchTool.__init__` builds the web researcher's `AgentSpec`
   with the parent's `llm` config copied in
3. At registration time, the researcher spec is appended to the
   parent's `sub_agents` list so `_resolve_agent_spec_for_task` can
   find it when the spawned task runs.

This is a permanent append, not ephemeral, because tools execute
in parallel (`asyncio.ensure_future`) — concurrent `web_fetch`
calls would race on append/remove of the same list. A permanent
entry is safe: it's a small in-memory object, lives only as long
as the workflow, and no other tool will collide with the name
`__web_researcher`.

### The web researcher sub-agent

A built-in agent defined programmatically in `web_fetch.py` (no
separate directory needed — the spec is built in code):

**AgentSpec fields:**
- `name`: `"__web_researcher"`
- `llm`: copied from parent spec
- `executor`: default (`llm`)
- `tools.builtins`: `[BuiltinToolConfig(name="code_sandbox")]`
- `interaction.conversational`: `False` (one-shot)
- `instructions`: inline markdown (see below)

**Instructions (goal-oriented, not implementation-prescriptive):**
- Given a query (and optional URL), find and return relevant web content
- Use the code sandbox to write and run scripts that search the web,
  fetch pages, and extract text
- The sub-agent decides how to search — the instructions don't
  prescribe specific search engines, URLs, or libraries
- If a URL is provided but its content doesn't answer the query,
  the sub-agent is free to search for and fetch other URLs
- Summarize findings relevant to the query
- If first attempt fails, try alternative approaches

### Blocking execution (in thread)

Tools run in a thread pool via `_to_thread()` in the workflow, so
a sync poll loop with `time.sleep()` in `invoke()` won't block the
event loop. `web_fetch` uses this existing pattern:

```python
def invoke(self, arguments, ctx):
    # 1. Spawn sub-agent
    task_id = _spawn_one(...)
    
    # 2. Poll until done (time.sleep in thread — OK)
    result = _poll_until_terminal(task_id, timeout=60)
    
    # 3. Extract and return text
    return _extract_output_text(result)
```

## Files to create/modify

| File | Action |
|------|--------|
| `agent_plane/tools/builtins/web_fetch.py` | **Create** — tool + inline sub-agent spec |
| `agent_plane/tools/builtins/__init__.py` | **Modify** — register `web_fetch` |
| `agent_plane/tools/manager.py` | **Modify** — pass parent spec to WebFetchTool, inject sub-agent |
| `agent_plane/onboarding/agent/config.yaml` | **Modify** — add `web_fetch` to builtins |
| `agent_plane/spec/AGENTSPEC.md` | **Modify** — document `web_fetch` |
| `tests/tools/builtins/test_web_fetch.py` | **Create** — tests |

## Implementation details

### `web_fetch.py` structure

```python
class WebFetchTool(Tool):
    def __init__(self, parent_spec: AgentSpec):
        # Build __web_researcher AgentSpec with parent's LLM config
        self._researcher_spec = _build_researcher_spec(parent_spec)
    
    def name() -> str: return "web_fetch"
    
    def get_schema() -> dict:
        # function schema with query (required) + url (optional)
    
    def invoke(arguments, ctx) -> str:
        # 1. Parse query + url
        # 2. Build prompt for web researcher
        # 3. _spawn_one() to launch sub-agent
        # 4. Poll until terminal
        # 5. Extract and return output text

def _build_researcher_spec(parent_spec: AgentSpec) -> AgentSpec:
    # Create AgentSpec with:
    # - parent's LLM config
    # - code_sandbox builtin
    # - web research instructions

def _poll_until_terminal(task_id, timeout) -> dict:
    # Poll task_store.get() until completed/failed

def _extract_output_text(result) -> str:
    # Pull text from conversation items

_RESEARCHER_INSTRUCTIONS = """..."""  # inline AGENTS.md content
```

### ToolManager changes

In `_create_builtin()`, when name is `"web_fetch"`:
1. Create `WebFetchTool(parent_spec=self._spec)`
2. The constructor appends the researcher spec to
   `parent_spec.sub_agents` (permanent for the workflow lifetime)

### Tests

- `test_web_fetch_schema` — verify function schema (query required, url optional)
- `test_web_fetch_name` — verify tool name is "web_fetch"
- `test_web_fetch_non_llm_executor_returns_error` — verify error for claude_sdk/agents_sdk
- `test_web_fetch_builds_researcher_spec` — verify spec inherits parent model
- `test_web_fetch_researcher_has_code_sandbox` — verify code_sandbox in builtins

Integration/e2e tests require a running server with real LLM.

## Deferred

- **Async tool invoke** — add `async_invoke()` to the Tool base class so `web_fetch` can poll with `await asyncio.sleep()` instead of tying up a thread pool slot. Currently tools run in threads via `_to_thread()` which is fine, but an async path would be cleaner for long-running tools like `web_fetch`.
- **claude_sdk / agents_sdk executor support** — these executors don't support sub-agents today. When they do, `web_fetch` will work with them automatically.

## Verification

1. Unit tests: `pytest tests/tools/builtins/test_web_fetch.py -xvs`
2. Regression: `pytest tests/onboarding/ tests/tools/builtins/ -xvs`
3. E2E: verify the onboarding assistant can use `web_fetch` to look
   up documentation during agent creation
