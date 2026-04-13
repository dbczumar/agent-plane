# Agent Plane

A control *plane* and runtime for agents — and the *runway* that helps them take
*flight*. Define an agent as a directory — `config.yaml`, instructions in
markdown, skills, tools, sub-agents — and go from development to production
in a couple of commands.

Agent Plane handles the hard parts: the agent loop, durable execution,
multi-provider LLM routing, streaming, multimodal input, file management,
conversation management, context compaction, memory, tool isolation,
sub-agent orchestration, an openresponses-compatible API, and more.

You focus on the agent — its instructions, skills, and tools. Iterate on those,
and Agent Plane takes care of running your agent in production.

## Why

Getting an agent working locally is the easy part. Getting it into production
— with durability, streaming, multi-tenancy, tool isolation, and a real API —
is where the work explodes. Most frameworks couple agent logic to a specific
runtime, a specific LLM provider, and a specific deployment model. Changing
any of these means rewriting the agent.

Agent Plane decouples these concerns:

- **Spec** — defines what an agent is: its instructions, tools, skills, LLM
  config, and sub-agents. A portable, language-neutral artifact. This is what
  you iterate on.
- **Runtime** — executes the agent: prompt construction, LLM calls, tool
  dispatch, retry, timeout. A library, not a service.
- **Server** — hosts agents as a service: HTTP API, persistent storage,
  streaming, multi-tenancy. One deployment mode — not the only one.

Swap LLM providers without touching agent definitions. Run the same agent
locally or on a server. Test with the same code path production uses. The
agent definition is simple enough to edit in a text editor; the runtime
behind it is production-grade.

## Quick Start

The fastest way to get started is to run an example agent and then create
a custom agent of your own. The `examples/` directory has ready-to-run
agents, and `ap chat` starts a server, uploads the agent, and opens an
interactive chat.

https://github.com/user-attachments/assets/6dca96fb-6ee2-4e2d-afc4-6525ba1d5337


### Example 1: Archer — research assistant

Archer is a resourceful research assistant with skills for deep research and
explanation. It uses web search and accepts text, image, and file input.

```bash
ap chat examples/agents/archer/
```

This starts a temporary server, uploads the archer agent, and drops you into
a streaming chat. Ask it to research a topic, explain a concept, or
investigate a question.

### Example 2: Coder — coding assistant with client-side tools

Coder is a coding assistant that can read, write, and edit files, search
codebases, and run shell commands. These tools execute on your machine, not
the server — the agent's reasoning runs remotely, but its actions happen
locally. This is the same pattern as computer use or OpenClaw-style agents:
a hosted agent drives operations on your laptop through client-side tool
execution. The frontend receives `function_call` items from the server,
executes them locally, and sends results back.

```bash
ap chat examples/agents/coder/ --tools coding
```

The `--tools coding` flag loads client-side tool schemas (Read, Write,
Edit, Glob, Grep, Bash) and their local execution logic from
`agent_plane/client_tools/coding.py`.

### Example 3: Create a custom agent

Once you've tried the example agents, create your own with `ap create`.

```bash
ap create --allow-shell-access
```

This launches an onboarding assistant that helps you create a custom agent
through an interactive conversation. It asks what kind of agent you want,
helps you choose a model provider and model, and then generates the agent
for you.

`--allow-shell-access` gives the onboarding assistant access to read and
write files, run commands, and inspect your environment while creating the
agent. Without it, the assistant runs in a sandbox and exports the finished
agent to your chosen path.

After it finishes, run your new agent by passing the path to the generated
agent directory:

```bash
ap chat /path/to/your-agent
```

### Using the API directly

You can also interact with agents over HTTP. Start the server, upload a bundle,
and send requests:

```bash
# Start the server
ap server --port 8000

# Upload an agent
tar czf archer.tar.gz -C examples/agents/archer .
curl -X POST http://localhost:8000/api/agents \
  -F "bundle=@archer.tar.gz"

# Send a request
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "archer",
    "input": "Review this function for off-by-one errors",
    "stream": true
  }'

# Continue the conversation
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "archer",
    "input": "What about error handling?",
    "previous_response_id": "resp_abc123"
  }'
```

The `model` field is the agent's name. Responses stream back as SSE events,
or set `"stream": false` for a single JSON response.

## Agent Image Format

An agent image is a self-contained directory:

```
agent-name/
├── config.yaml             # LLM config, tools, interaction settings
├── AGENTS.md               # Instructions (personality, constraints, behavior)
├── skills/                 # On-demand instruction chunks
│   └── <skill-name>/
│       └── SKILL.md
├── tools/                  # Packaged tools
│   ├── python/             # Local Python tools (auto-discovered)
│   │   └── *.py
│   └── mcp/                # MCP server declarations
│       └── *.yaml
└── agents/                 # Sub-agents (recursive, same format)
    └── <agent-name>/
        ├── config.yaml
        └── ...
```

**config.yaml** is the only required file. Everything else is optional.

## Capabilities

### Working

- **Multi-provider LLM support** — OpenAI, Anthropic, Google Gemini, AWS
  Bedrock, Vertex AI, Databricks. Model strings use `provider/model-name`
  format (e.g. `anthropic/claude-sonnet-4-20250514`, `google/gemini-2.5-pro`).
  OpenAI-compatible endpoints (Groq, DeepSeek, xAI, OpenRouter, Ollama) work
  through the OpenAI adapter.

- **OpenAI-compatible inference API** — `POST /v1/responses` follows the
  [OpenResponses](https://openresponses.org) spec (superset of OpenAI
  Responses API). Streaming via SSE, blocking, and background execution modes.

- **Durable execution** — LLM calls and tool calls are checkpointed via DBOS.
  If the server crashes mid-execution, completed steps are not re-run on
  recovery.

- **Conversations** — multi-turn threading via `previous_response_id`.
  Conversations are created automatically. Forking (branching from a
  mid-conversation response) creates a new conversation with copied history.

- **Steering** — send input to an in-progress agent execution. The agent
  incorporates it at its next loop iteration. Useful for redirecting an agent
  that's going down the wrong path.

- **Context management** — compaction support for long conversations that
  exceed token limits. Configurable via `context_management` on the request.

- **Skills** — named instruction chunks the agent loads on demand. Each skill
  is a markdown file with frontmatter (`name`, `description`) and content
  injected into the system prompt when the agent calls `load_skill`.

- **MCP tools** — connect to MCP servers over HTTP/SSE. Declare servers in
  `tools/mcp/*.yaml` with URL, headers, and optional per-server
  timeout/retry config.

- **Built-in tools** — web search via OpenAI (passthrough), Google Custom
  Search, or Perplexity. Configured in `config.yaml` with API keys or
  environment variable references.

- **Local Python tools** — auto-discovered from `tools/python/*.py`. Each
  file exports a `SCHEMA` dict (OpenAI function format) and a `run(arguments)
  -> str` function. Executed in a subprocess for fault isolation — a crashed
  tool doesn't crash the server. Supports sandboxing.

- **Dependency management for tools** — local tool dependencies can be
  declared and installed per tool runtime instead of assuming they're already
  present on the host.

- **Sub-agents** — declare child agents under `agents/` and list them in
  `tools.agents`. The parent spawns sub-agents as independent workflows with
  isolated conversations and tool registries.

- **Subagent batch and parallel fan-out** — `agent.map` / `agent.spawn` for
  orchestrating sub-agents at scale.

- **Client-side tools** — tools whose execution happens on the caller's side,
  not the server. The server returns function call items; the client executes
  them and sends results back. Useful for file system access, IDE integration,
  and other client-local operations.

- **Multimodal input** — text, images, audio, files (PDF, docx, etc.).
  Declared per-agent in `interaction.modalities.input`.

- **File upload** — `POST /v1/files` for uploading files referenced in
  requests via `file_id`.

- **Agent management API** — CRUD for agent bundles (`/api/agents`). Upload
  validates the spec on the server side (structure, naming, constraints).

- **Conversation management API** — list, get, update, delete conversations.
  List items within a conversation with pagination.

- **Per-tool timeout and retry** — configurable at the global level
  (`tools.timeout`, `tools.retry`) and per-tool (MCP server config,
  `tools.local` block). Exponential backoff with jitter.

- **Executor support** — the default `llm` executor, Claude agents via
  `executor.type: claude_sdk`, and OpenAI Agents SDK agents via
  `executor.type: agents_sdk` are supported.

- **Security sandboxing** — when `srt` (Anthropic's Sandbox Runtime) is on
  PATH, `code_sandbox` and local Python tools run inside an OS-level sandbox
  with filesystem and network restrictions. Writes are restricted to the
  per-conversation workspace. Reads are denied outside the workspace and
  system directories. Network access is limited to package registries
  (pypi.org, npmjs.org). **Known limitation on macOS:** srt uses sandbox-exec
  seatbelt rules, which operate at the syscall level — `denyRead: ["/"]`
  blocks all `file-read*` operations including PATH resolution for shell
  commands. To preserve bash functionality, the macOS sandbox dynamically
  derives system directories from `$PATH` and re-allows them. On Linux, srt
  uses bubblewrap with a read-only root bind mount, so `denyRead: ["/"]`
  works without an allowlist — system binaries remain readable from the
  initial mount.

### Planned

- **Compatibility libraries** — adapters for running agents defined in
  other frameworks (OpenAI Agents SDK, Vercel AI SDK, LangGraph, CrewAI)
  on agent-plane, and vice versa. Use agent-plane's durable execution and
  multi-provider support with existing agent definitions.

- **Memory** — persistent memory across conversations. Agents will be able to
  store and retrieve information that survives individual sessions. Memory
  policy declarations (`memory:` block in config.yaml) for consent hints and
  scope. In v1, memory is purely a tool concern.

- **Structured I/O** — `interaction.schema` for declaring typed input/output
  contracts. The runtime would validate inputs and outputs against declared
  field types.

- **Frontend library** — a reusable client library for building frontends
  against the agent-plane API. SSE streaming, conversation management,
  client-side tool dispatch, and reconnection handling.

- **More example frontends** — web UI, desktop app, and other reference
  implementations beyond the terminal TUI.

- **Slack and Teams integrations** — agent-plane agents as chatbots in
  Slack and Microsoft Teams. Map conversations to channels/threads,
  deliver responses as messages.

- **Scheduled jobs** — run agents on a cron schedule. Periodic tasks like
  monitoring, reporting, data collection, and maintenance without manual
  invocation.

- **TypeScript tools** — local tool execution for `.ts` files via Node.js
  subprocess. The parser already discovers TypeScript files; the loader and
  runner are the missing pieces.

- **Built-in observability** — tracing, metrics, and logging for agent
  executions out of the box. LLM calls, tool calls, token usage, latency,
  and error rates — visible without wiring up external instrumentation.

- **Built-in feedback** — collect user feedback (thumbs up/down, corrections,
  ratings) on agent responses and tie it back to specific executions.
  First-class support for evaluation loops and quality tracking.


## Sandbox & Tool Isolation

Agent-plane isolates tool execution so agents can't access files or
resources outside their workspace.

### Default executor (LLM)

Local Python tools run in subprocesses. When `srt` is on PATH and
`RuntimeCaps.sandbox_enabled` is True (default), tools are wrapped
with OS-level sandboxing. Docker containers are supported via
`tools.sandbox.docker_image` in the agent spec.

### Claude SDK executor

Three isolation layers protect the host filesystem:

| Layer | What it covers | Mechanism |
|-------|---------------|-----------|
| **PreToolUse hooks** | Built-in tools (Read, Glob, Grep, Edit, Write) | Blocks file paths outside workspace. Cannot be bypassed. |
| **OS sandbox (writes)** | Bash write operations | Seatbelt (macOS) / bubblewrap (Linux) restricts writes to workspace. |
| **OS sandbox (reads)** | Bash read operations | `sandbox.filesystem.denyRead` configured but **not currently effective** — see known issues below. |

**Known issues with Bash read isolation:**

The `sandbox.filesystem.denyRead` setting does not block Bash reads
on macOS or Linux. This is a Claude Code platform bug, not an
agent-plane issue:

- [anthropics/claude-code#32226](https://github.com/anthropics/claude-code/issues/32226) — "denyRead seems ineffective" (primary tracker)
- [anthropics/claude-code#43043](https://github.com/anthropics/claude-code/issues/43043) — allowRead overrides denyRead
- [anthropics/claude-code#44379](https://github.com/anthropics/claude-code/issues/44379) — denyRead not enforced on bundled rg
- [anthropic-experimental/sandbox-runtime#193](https://github.com/anthropic-experimental/sandbox-runtime/issues/193) — denyRead inside allowRead directory

Agent-plane configures `denyRead` so that when these bugs are fixed
upstream, Bash reads will be restricted automatically. Until then,
built-in file tools are the primary read isolation mechanism (via
PreToolUse hooks).

### Configuration

Sandboxing is a **runtime** policy — agents cannot disable it:

```python
# RuntimeCaps (operator-controlled)
RuntimeCaps(sandbox_enabled=True)  # default: srt enabled when on PATH

# Agent spec (agent-controlled)
tools:
  sandbox:
    docker_image: python:3.12-slim  # optional: run tools in Docker
```


## LLM Providers

Model strings use `provider/model-name` format. Omitting the provider defaults
to OpenAI.

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai/` | `openai/gpt-4o`, `openai/o4-mini` |
| Anthropic | `anthropic/` | `anthropic/claude-sonnet-4-20250514` |
| Google Gemini | `google/` | `google/gemini-2.5-pro` |
| AWS Bedrock | `bedrock/` | `bedrock/anthropic.claude-v2` |
| Vertex AI | `vertex/` | `vertex/gemini-2.5-pro` |
| Databricks | `databricks/` | `databricks/databricks-meta-llama-3-70b` |
| Groq | `groq/` | `groq/llama-3.1-70b` |
| DeepSeek | `deepseek/` | `deepseek/deepseek-chat` |
| Ollama | `ollama/` | `ollama/llama3` |

OpenAI-compatible providers (Groq, DeepSeek, xAI, OpenRouter, Ollama) route
through the OpenAI adapter with the appropriate base URL.

## API Overview

Three namespaces:

**Agent Management** — `/api/agents`
- `POST /api/agents` — upload agent bundle (tarball)
- `GET /api/agents` — list agents
- `GET /api/agents/{id}` — get agent
- `DELETE /api/agents/{id}` — delete agent

**Inference** — `/v1/responses` (OpenResponses-compatible)
- `POST /v1/responses` — create response (streaming, blocking, or background)
- `GET /v1/responses/{id}` — get response
- `DELETE /v1/responses/{id}` — delete response
- `POST /v1/responses/{id}/cancel` — cancel in-progress response

**Conversations** — `/v1/conversations`
- `GET /v1/conversations` — list conversations
- `GET /v1/conversations/{id}` — get conversation
- `PATCH /v1/conversations/{id}` — update conversation (title)
- `DELETE /v1/conversations/{id}` — delete conversation
- `GET /v1/conversations/{id}/items` — list items in conversation

**Files** — `/v1/files`
- `POST /v1/files` — upload file
- `GET /v1/files` — list files
- `GET /v1/files/{id}` — get file metadata
- `GET /v1/files/{id}/content` — download file content
- `DELETE /v1/files/{id}` — delete file

## Architecture

```
┌──────────────────────────────────────────────┐
│  Frontend Layer                               │
│  Terminal TUI (ap chat), GUIs, and other    │
│  rich experiences                            │
└─────────────────────┬────────────────────────┘
                      │
┌─────────────────────▼────────────────────────┐
│  Server Layer (FastAPI)                       │
│  HTTP API, SSE streaming, agent bundle       │
│  management, persistent storage              │
└─────────────────────┬────────────────────────┘
                      │
┌─────────────────────▼────────────────────────┐
│  Runtime Layer (library)                     │
│  Agent loop, prompt construction, secure     │
│  tool dispatch, retry/timeout, skills, MCP   │
└──────────┬───────────────────┬───────────────┘
           │                   │
┌──────────▼──────────┐ ┌─────▼───────────────────┐
│  Spec Layer         │ │  LLM Layer              │
│  Agent image format,│ │  Multi-provider SDK,    │
│  config.yaml parse, │ │  Responses API interface│
│  validation         │ │  Chat Completions       │
└─────────────────────┘ │  lingua franca          │
                        └─────────────────────────┘
```

The runtime depends on both the spec layer (what to execute) and the LLM layer
(how to call models). The server depends on the runtime. The spec and LLM
layers are independent of each other.

The runtime is a library. The server is one host for it. The same runtime can
be embedded in other applications, used for local development, or invoked
directly from tests.

## Runtime Architecture

The runtime is where the interesting work happens. Here's what's under the
hood.

### Durable execution

Every agent execution is a [DBOS](https://docs.dbos.dev/) workflow. Each LLM
call and each tool call is a checkpointed step. If the server crashes
mid-execution, completed steps are not re-run — DBOS returns their cached
results on recovery. An LLM call that was mid-stream when the crash happened
re-executes from scratch (safe — LLM calls are idempotent). You don't lose a
10-minute agent execution because the server restarted.

### Dual-channel streaming

Token chunks are written to two places simultaneously: a DBOS durable stream
(for crash recovery and polling clients) and a live in-memory pub/sub channel
(for real-time SSE delivery). Clients that connect mid-execution get caught up
from the durable stream. Clients that disconnect can poll later. Background
mode lets the agent keep running after disconnect entirely.

### Tool isolation

Each workflow gets its own `ToolManager` via `contextvars` — concurrent
workflows in the same process don't share tool state. MCP server connections
are opened at workflow start and torn down in `finally`. Local Python tools
run in isolated subprocesses — a segfault, OOM, or infinite loop in a tool
kills the child process, not the server. Client-side tools never execute on
the server at all.

### Steering

Users can send messages to an agent while it's still working. Between LLM
turns, the workflow checks for new user input. If a redirect arrived, it's
folded into the next prompt. The inbox closes atomically when the agent
produces its final response — if a message arrives during that window, the
agent does one more loop iteration instead of dropping it.

### Sub-agent execution

Sub-agents are full workflows — same loop, same checkpointing, same tool
isolation. The parent spawns them as independent executions with their own
conversations and tool registries. No tool inheritance, no shared state.
Depth limit of 5 prevents runaway recursion.

### Agent cache

A two-tier cache (memory + disk) backed by pluggable artifact storage. First
request for an agent downloads, extracts, parses, and validates the bundle.
Subsequent requests hit memory. Server restarts hit disk. Only a full cache
miss goes to the artifact store.

### Store abstraction

All persistence — tasks, conversations, conversation items, agents, files,
artifacts — goes through abstract store interfaces. The default implementation
is SQLAlchemy (SQLite for dev, PostgreSQL for production). Stores are injected
at server startup, not imported. The runtime, the server routes, and the
workflows all operate against the abstractions.

## Project Structure

```
agent_plane/
├── cli.py             # CLI entry point (ap server)
├── entities/          # Domain models (Task, Conversation, Agent, etc.)
├── db/                # Database models and migrations
├── runtime/           # Execution engine (agent loop, tool dispatch)
├── server/            # HTTP layer (FastAPI routes, SSE, schemas)
├── spec/              # Agent image parsing and validation
├── stores/            # Abstract store interfaces + implementations
├── tools/             # Tool system (MCP, builtins, local, client-side)
└── client/            # CLI and typed client library
llms/                  # Multi-provider LLM SDK
├── adapters/          # Per-provider adapters
├── client.py          # Public Client class
└── types.py           # Response/streaming types
examples/
└── agents/            # Example agent images
```
