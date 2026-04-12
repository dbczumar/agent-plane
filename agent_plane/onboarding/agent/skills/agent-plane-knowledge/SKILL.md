---
name: agent-plane-knowledge
description: Deep reference on agent plane config format, executor types, skill/tool structure, and conventions. Load when you need to look up how the platform works.
---

# Agent Plane Knowledge Base

## What is agent plane?

Agent plane is a server that hosts, manages, and executes agents via an
OpenResponses-compatible API. Users create **agent directories** (also
called agent images) that contain configuration, instructions, skills,
and tools. The server loads these directories and serves them via HTTP.

## Agent Directory Layout

```
my-agent/
├── config.yaml          # REQUIRED — agent spec
├── AGENTS.md            # Recommended — instructions/personality
├── skills/              # Optional — load-on-demand skills
│   └── <skill-name>/
│       └── SKILL.md
├── tools/               # Optional — packaged tools
│   ├── python/          # Local Python tools (auto-discovered *.py)
│   ├── typescript/      # Local TypeScript tools (auto-discovered *.ts)
│   └── mcp/             # MCP server declarations (*.yaml)
└── agents/              # Optional — sub-agent directories (recursive)
    └── <agent-name>/
        ├── config.yaml
        └── ...
```

## config.yaml Reference

The only required file. All fields except `spec_version` are optional.

```yaml
spec_version: 1               # REQUIRED, must be 1

name: my-agent                # Display name
description: Does X and Y.    # One-line summary

# Instructions — path to a file or inline text.
# Default: looks for AGENTS.md in the agent directory.
instructions: AGENTS.md

llm:
  # Model in LiteLLM format: provider/model-name
  # Examples: openai/gpt-5.4, anthropic/claude-sonnet-4-20250514,
  #   gemini/gemini-2.5-pro, groq/llama-4-scout-17b-16e-instruct
  model: openai/gpt-5.4

  # Provider credentials. Use ${ENV_VAR} for secrets.
  connection:
    api_key: ${OPENAI_API_KEY}
    # base_url: https://custom-endpoint.example.com/v1  # for compatible APIs

  # Optional LLM parameters (passed through to the provider)
  reasoning_effort: medium     # low | medium | high
  max_completion_tokens: 4096  # caps total output including reasoning

executor:
  # Executor type determines how the agent runs.
  type: llm          # default — agent plane manages the LLM loop
  # type: claude_sdk   — Claude SDK agent (native)
  # type: agents_sdk   — OpenAI Agents SDK agent (native)
  # type: remote        — external agent behind HTTP endpoint

  timeout: 3600        # Task deadline in seconds (default: 3600)
  max_iterations: 1000 # Max LLM calls per task (default: 1000)
  # endpoint: http://localhost:5001  # required for type: remote

interaction:
  conversational: true   # Maintain turn history (default: true)
  modalities:
    input: [text, image, file]   # default: [text]
    output: [text]               # default: [text]

tools:
  # Sub-agents this agent can spawn (must match agents/ subdirectories)
  agents:
    - researcher
    - summarizer

  # Built-in tools — string name or dict with config
  builtins:
    - web_search                 # auto-detects backend based on model provider
    - code_sandbox
    - upload_file
    - search_conversations

  timeout: 60          # Default tool timeout in seconds

params:                # Arbitrary key-value (readable by skills/tools)
  max_results: 10
```

## Executor Types

| Type | When to use | How it works |
|------|------------|--------------|
| `llm` (default) | New agents, simple configs | Agent-plane manages the full LLM loop: prompt construction, tool calling, multi-turn |
| `claude_sdk` | Existing Claude SDK agent code | Agent-plane delegates to the Claude SDK, which manages its own loop |
| `agents_sdk` | Existing OpenAI Agents SDK code | Agent-plane delegates to the Agents SDK runner |
| `remote` | External agent behind HTTP | Agent-plane calls a remote endpoint implementing the executor protocol |

For **most new agents**, use the default `llm` executor — it's the simplest
and most capable option. Only use `claude_sdk` or `agents_sdk` if the user
has existing agent code built on those SDKs.

## AGENTS.md Format

Free-form markdown. This becomes the agent's system prompt. Best practices:

- Start with a clear identity statement ("You are a ...")
- List capabilities and constraints
- Reference skills by name ("You have a skill called deep-research")
- Reference sub-agents if any ("You can spawn the fact_checker agent")
- Keep it focused — the model reads this on every turn

## Skills Format

Each skill lives in `skills/<skill-name>/SKILL.md`:

```markdown
---
name: deep-research
description: Investigate a topic in depth using web search and source synthesis.
---

When researching a topic:

1. Search broadly first using web search...
2. Cross-reference multiple sources...
```

Rules:
- YAML frontmatter with `name` and `description` (both required)
- `name` must match the directory name, be lowercase, use `[a-z0-9-]+`
- Body is markdown instructions loaded on demand by the agent
- Referenced in AGENTS.md or config.yaml

## Tools

### Built-in tools
Available built-in tools include: `web_search`,
`code_sandbox`, `upload_file`, `search_conversations`. Reference by name in
`tools.builtins`.

### MCP tools
External MCP servers are declared in `tools/mcp/*.yaml`:

```yaml
transport: http
url: https://mcp-server.example.com/sse
headers:
  Authorization: Bearer ${MCP_TOKEN}
```

### Local tools
Python or TypeScript files in `tools/python/` or `tools/typescript/` are
auto-discovered and registered as tools.

## Example: Minimal Agent

```yaml
spec_version: 1
name: my-assistant
description: A helpful assistant.
llm:
  model: anthropic/claude-sonnet-4-20250514
  connection:
    api_key: ${ANTHROPIC_API_KEY}
instructions: |
  You are a helpful assistant. Answer questions clearly and concisely.
```

This is the simplest valid agent — just a name, model, and instructions.
No skills, no tools, no sub-agents.

## Example: Research Agent with Tools and Skills

```yaml
spec_version: 1
name: researcher
description: A research agent that searches the web and synthesizes findings.
llm:
  model: openai/gpt-5.4
  connection:
    api_key: ${OPENAI_API_KEY}
tools:
  builtins:
    - web_search
    - upload_file
interaction:
  modalities:
    input: [text, file]
    output: [text]
instructions: AGENTS.md
```

## Running an Agent

Once the agent directory is created:

```bash
# Start the server with the agent pre-registered
ap server --agent ./my-agent/

# Or deploy to a running server
ap deploy ./my-agent/ --server http://localhost:8000
```
