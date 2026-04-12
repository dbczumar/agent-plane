# Onboarding Experience

## Context

Agent-plane today requires manual authoring of an agent directory
(`config.yaml`, `AGENTS.md`, `skills/`, `tools/`) and CLI invocation
(`ap server --agent ./my-agent`) to get a running agent. There is no
guided path for new users. This creates friction in three common
situations:

1. **"I don't have an agent."** The user has never built an agent. They
   may or may not have Claude Code (or Codex, or another coding
   assistant) installed.

2. **"I have an agent built on framework X."** The user has an existing
   agent (LangChain, CrewAI, AutoGen, a custom Python script, etc.) and
   wants to plug it into agent-plane without rewriting it.

In both cases, the user needs an LLM API key to proceed — the
onboarding agent is itself an agent and needs a model to run. The
setup flow begins by having the user select a provider and supply an
API key. Each supported provider has a sensible default model so the
user doesn't need to know model identifiers upfront. This provider/key
selection then powers the onboarding agent and becomes the default
model configuration for the generated agent.

The onboarding system should handle both cases with minimal manual
steps, leveraging the user's existing tools (Claude Code, Codex, etc.)
where available and falling back to a CLI-driven flow otherwise.

### Current state

| Component | Status |
|-----------|--------|
| Agent spec format (`config.yaml`, `AGENTS.md`, skills, tools) | Stable |
| Executor types (`llm`, `claude_sdk`, `agents_sdk`, `remote`) | Stable |
| CLI (`ap server`) | Stable, no `create` command |
| Claude Code skills for agent-plane development | Exists in `.claude/skills/agent-plane-dev/` |
| Provider selection / model resolution | Not implemented |
| Coding assistant integration (Claude Code, Codex, Gemini) | Not implemented |
| Framework detection / scaffolding | Not implemented |

---

## Design Premise

The onboarding CLI (`ap create`) collects a model provider and API key,
then boots a temporary agent-plane server and drops the user into an
interactive shell connected to a built-in **onboarding agent**. This
dogfoods agent-plane itself: the onboarding agent runs on agent-plane
with client tools enabled (filesystem access, environment detection,
code scaffolding), and conversationally guides the user through setting
up a new agent or integrating an existing one.

### Guiding principles

- **Agent-first.** The primary onboarding path is conversational: an
  agent with filesystem tools guides the user. A non-interactive
  fallback exists for CI/scripted environments.
- **Detect, don't ask.** Probe the environment (installed tools, API
  keys, existing code) before asking the user questions.
- **Approve before mutate.** Every filesystem write or external action
  requires explicit user confirmation.
- **One correct path.** No dual-mode fallbacks or feature flags. Each
  user scenario maps to exactly one onboarding flow.

---

## Open Questions

The following questions must be resolved before implementation begins.
Each question includes the design context that motivates it and, where
applicable, a recommended default.

### Architecture & Scope

**Q1. Onboarding agent vs. CLI wizard — which is primary?** *(Resolved)*

**Hybrid.** The CLI handles two things before handing off:

1. **Provider selection** — user picks a model provider and supplies
   an API key (see "Provider & Model Selection" section).
2. **Server startup** — boots a temporary agent-plane server, loads
   the built-in onboarding agent using the selected model, and drops
   the user into an interactive shell.

From there the user is in a conversational shell talking to the
onboarding agent. The agent has **client tools enabled** — filesystem
access, environment detection, code generation — and drives the rest
of the setup: detecting existing agents or frameworks, scaffolding new
agent directories, installing IDE skills, and configuring `config.yaml`.
The user stays in this shell until setup is complete.

**Q2. Where does the onboarding agent live?** *(Resolved)*

`agent_plane/onboarding/agent/` — first-class built-in, shipped with
the package and referenced by `ap create`.

**Q3. MLflow assistant** *(Deferred — see Deferred section)*

---

### Framework Detection & Agent Generation

**Q4. "Supported frameworks" — detection semantics** *(Resolved)*

**(c)** — detect frameworks from the user's code and wire them to the
appropriate executor. If a native executor exists for the framework,
use it. Otherwise, scaffold a `remote` executor HTTP wrapper.

The executor set is expanding. LangGraph and DeepAgents will get native
executors soon, and others (LangChain, CrewAI, AutoGen) may follow.
The detection table below reflects current state; the onboarding agent
should be designed so that adding a new native executor only requires
updating this mapping, not changing the detection flow.

Detection signals (in priority order):

| Signal | Framework | Executor |
|--------|-----------|----------|
| `import anthropic` + Claude agent patterns | Claude SDK | `claude_sdk` |
| `import openai` + Agents SDK patterns | OpenAI Agents SDK | `agents_sdk` |
| `from langgraph` imports | LangGraph | Native (planned) |
| `from deepagents` imports | DeepAgents | Native (planned) |
| `from langchain` imports | LangChain | `remote` (native planned) |
| `from crewai` imports | CrewAI | `remote` (scaffold HTTP wrapper) |
| `from autogen` imports | AutoGen | `remote` (scaffold HTTP wrapper) |
| None of the above | Unknown | `remote` (scaffold HTTP wrapper) + file issue |

**Q5. Remote executor scaffolding** *(Partially resolved — details deferred)*

For unsupported frameworks, the onboarding agent generates a standalone
FastAPI app inside the user's agent directory that implements the remote
executor protocol (as defined in `EXECUTOR_CONTRACT.md`). Agent-plane
launches it as a subprocess, similar to the approach described in
[CLAUDE_SDK_PROGRAMMATIC_SUPPORT.md](CLAUDE_SDK_PROGRAMMATIC_SUPPORT.md)
for code-based program imports.

Remaining details (Dockerfile generation, subprocess lifecycle
management, etc.) to be resolved later.

**Q6. "File a GitHub issue" — automatic or manual?** *(Resolved)*

Ask the user if they want to request first-class support for their
framework. If yes, print a pre-filled GitHub issue URL they can click
to file. No automatic issue filing — keep it low-friction and
transparent.

---

### Provider & Model Selection *(Resolved)*

The onboarding agent is itself an agent — it needs an LLM to run.
Provider/model selection is therefore a **prerequisite step** that
happens before the onboarding agent starts, not a question the
onboarding agent answers.

**Flow:** In interactive mode, `ap create` presents the user with a
list of supported providers. The user picks one, supplies an API key,
and picks a model. The `--model provider/model_name` flag skips this
prompt entirely. In non-interactive mode, `--model` is required and
credentials come from environment variables.

Agent-plane's routing layer (`agent_plane/llms/routing.py`) already
supports the following providers. The onboarding flow must support all
of them, matching the breadth of providers available in the MLflow AI
Gateway.

**Data source:** The MLflow model catalog
(`mlflow/utils/model_catalog/`) contains 68 per-provider JSON files
with model metadata (pricing, context windows, capabilities,
deprecation dates). The provider registry
(`mlflow/utils/providers.py`) defines auth modes and credential fields
for each provider.

**Important:** The relevant code and data (catalog JSON files, provider
auth mode definitions, model listing logic) should be **copied** into
agent-plane, not imported from MLflow. MLflow must not be a dependency
of agent-plane. The copied code should be kept minimal — only what's
needed for provider selection, auth field prompting, and model listing.

The onboarding flow should use this copied catalog as the source
of truth for:

- **Provider list** — `get_all_providers()` returns all available
  providers, normalized and deduplicated.
- **Model list per provider** — `get_onboarding_models(provider)`
  returns text chat models with function calling support (the
  onboarding agent needs tools). Excludes audio, realtime, image,
  and embedding models. Sorted newest first (by version, then date).
- **Auth requirements** — `get_provider_config_response(provider)`
  returns the credential fields needed (API key, access keys, service
  account JSON, etc.) with support for multiple auth modes per
  provider (e.g. Bedrock supports API key, access keys, IAM role, and
  default credential chain).

The user explicitly picks a provider and supplies a key — no implicit
env var scanning. This means the provider table is not hardcoded in
agent-plane — it comes from the MLflow catalog and stays up to date
as providers and models are added there.

The selected provider and model are used for two things:
1. **Running the onboarding agent itself** during the setup session.
2. **Configuring the generated agent's `config.yaml`** as the default
   model.

---

---

### User Flow

**Q13. CLI entry point** *(Resolved)*

Command: `ap create`

**Interactive mode** (no positional argument):

```
ap create [--model PROVIDER/MODEL] [--allow-filesystem-access]
```

Launches the provider selection prompt (unless `--model` is given),
then drops the user into an interactive shell with the onboarding
agent. The agent drives everything from there.

**Non-interactive mode** (positional message argument):

```
ap create "create a research agent with web search" \
    --model anthropic/claude-sonnet-4-20250514 \
    --allow-filesystem-access
```

Like Claude Code's `-p` pattern: passing a message makes it
non-interactive. The onboarding agent runs with that message as
its initial prompt and executes without further user input.

- `--model`: Model in litellm format (`provider/model_name`).
  Required for non-interactive mode. In interactive mode, skips the
  provider selection prompt if provided.
- `--allow-filesystem-access`: Enable filesystem client-side tools
  (read/write files, detect frameworks from code). Off by default
  for safety — some users may not want the agent to have free reign
  of their filesystem. Recommended on for the best onboarding
  experience; without it the agent can only generate output to
  stdout and the user must copy files manually.

Auth credentials for non-interactive mode are read from environment
variables (matching the provider's expected env var, e.g.
`ANTHROPIC_API_KEY` for `anthropic/*` models). This keeps the
CLI simple and works naturally in CI/scripted environments.

The command starts a temporary agent-plane server, loads the
onboarding agent, and opens a session. When the session ends,
the server shuts down.

---

## Proposed User Flows

Pending resolution of the open questions above, the following flows are
sketched at high level.

### Flow 1: Interactive — "I don't have an agent"

```
$ ap create --allow-filesystem-access
Select a model provider (popular first, 68 total):
  1. openai           7. azure            13. ollama
  2. anthropic        8. xai              14. together_ai
  3. databricks       9. mistral          15. cohere
  4. bedrock         10. groq             16. fireworks_ai
  5. gemini          11. deepseek         17. ai21
  6. vertex_ai       12. openrouter       18. aleph_alpha
  ... (68 providers from MLflow model catalog)

Provider [1]: 2
ANTHROPIC_API_KEY: sk-ant-***  ✓ valid

Starting onboarding agent...

Agent: I'll help you create an agent. What would you like it to do?

User:  I want a research assistant that can search the web and
       summarize findings.

Agent: [detects environment, scaffolds agent directory, shows plan]
       ...
```

### Flow 2: Interactive — "I have a LangChain agent"

```
$ ap create --allow-filesystem-access
[provider selection...]

Agent: I detected a LangChain agent in the current directory.
       LangChain isn't natively supported as an executor yet, but
       I can create a remote executor wrapper that exposes your
       agent via HTTP and plugs it into agent-plane.
       ...
```

### Flow 3: Non-interactive — new agent

```
$ ANTHROPIC_API_KEY=sk-ant-*** ap create \
    "create a research agent with web search" \
    --model anthropic/claude-sonnet-4-20250514 \
    --allow-filesystem-access

Creating agent...
  research-agent/config.yaml
  research-agent/AGENTS.md
  research-agent/skills/deep-research/SKILL.md

Done. Run `ap server --agent ./research-agent/` to start.
```

### Flow 4: Non-interactive — existing agent

```
$ OPENAI_API_KEY=sk-*** ap create \
    "integrate the LangChain agent in ./my-agent into agent-plane" \
    --model openai/gpt-5.4 \
    --allow-filesystem-access

Detected LangChain framework in ./my-agent/agent.py
Generating remote executor wrapper...
  my-agent/ap_wrapper.py
  my-agent/config.yaml

Done. Run `ap server --agent ./my-agent/` to start.
```

---

## Implementation Sketch

Pending question resolution. The following is a rough structure.

### New files

```
agent_plane/
  onboarding/
    agent/
      config.yaml          # Onboarding agent spec
      AGENTS.md            # Onboarding agent instructions
      skills/
        agent-plane-knowledge/
          SKILL.md          # Deep knowledge of agent-plane: config.yaml
                            # format, AGENTS.md conventions, skill/tool
                            # structure, executor types. References
                            # existing example agents and design docs.
        detect-framework/
          SKILL.md          # Detect frameworks from user code (imports,
                            # dependencies) and map to executor types.
                            # Includes the detection signal table.
        generate-agent/
          SKILL.md          # Generate agent directory: config.yaml,
                            # AGENTS.md, skills/, tools/. Knows the
                            # spec format and produces valid output.
    providers/              # Copied from mlflow — NOT an mlflow dependency
      model_catalog/        # 68 per-provider JSON files
      __init__.py           # get_all_providers, get_models, etc.
    cli.py                  # `ap create` command implementation
```

### Onboarding agent skills

Each skill has a detailed SKILL.md with reference examples and file
paths so the onboarding agent has deep knowledge of agent-plane
conventions. The three core skills are:

- **agent-plane-knowledge** — the agent's understanding of agent-plane
  itself: config format, executor types, skill/tool structure, what
  makes a good AGENTS.md. References existing example agents and
  design docs as concrete examples.
- **detect-framework** — detect the user's framework from code imports
  and map it to the right executor type. Includes the full detection
  signal table and knows how to scaffold remote executor wrappers for
  unsupported frameworks.
- **generate-agent** — produce a valid agent directory (config.yaml,
  AGENTS.md, skills/, tools/) from the information gathered during
  the conversation.

Additional skills to add as the onboarding agent matures:

- **MCP discovery** — find and recommend relevant MCP servers for the
  agent being created (search registries, match to agent's purpose).
- **Dependency resolution** — detect missing packages and generate
  `requirements.txt` or `pyproject.toml` entries.

The onboarding agent should also have **web search** capability to
debug issues and find additional resources (MCP servers, tool
packages, example configs, etc.) during the creation flow. Strategy:

- If the selected model provider supports native web search (e.g.
  OpenAI), use it directly.
- Otherwise, fall back to bash tools (`curl`/`wget`) for web lookups.

### CLI entry point addition

```python
# In agent_plane/cli.py (Click group registered as `ap` in pyproject.toml),
# add alongside existing `server` and `deploy` commands:
@cli.command()
@click.argument("message", required=False, default=None)
@click.option("--model", default=None, help="Model in litellm format (provider/model_name).")
@click.option(
    "--allow-filesystem-access", is_flag=True, default=False,
    help="Enable filesystem client-side tools for the onboarding agent. "
    "Recommended for the best experience — lets the agent read your "
    "existing code and write the generated agent directory directly.",
)
def create(
    message: str | None,
    model: str | None,
    allow_filesystem_access: bool,
) -> None:
    """Create a new agent-plane agent.

    Interactive (no message): prompts for provider/key, then opens
    a shell with the onboarding agent.

    Non-interactive (message provided): requires --model; reads
    credentials from environment variables.
    """
    ...
```

### Provider selection and model resolution

```python
# Pseudocode — driven by MLflow model catalog, not hardcoded

# Copied from mlflow/utils/providers.py and mlflow/utils/model_catalog/
# into agent_plane/onboarding/. MLflow is NOT a dependency.
from agent_plane.onboarding.providers import (
    get_all_providers,
    get_models,
    get_provider_config_response,
)

def prompt_provider_selection() -> ProviderSelection:
    """
    Present providers, collect credentials, resolve model.

    1. List providers via get_all_providers() (sourced from
       mlflow/utils/model_catalog/*.json).
    2. User picks a provider. Prompt for credentials using the
       auth mode fields from get_provider_config_response(provider).
    3. List chat-capable models via get_models(provider), filtered
       to mode="chat". User picks one.

    :return: ProviderSelection with provider, model, and credentials.
    """
    ...
```

---

## Deferred

- **MLflow assistant on agent-plane** — reimplement the MLflow assistant UI as an agent-plane agent, runnable with any LLM API key. Separate design doc (`MLFLOW_ASSISTANT.md`).
- **Claude Code skills installation** — detect Claude Code, install `agent-plane-dev` skill into `.claude/skills/`.
- **Codex skills installation** — detect Codex, install agent-plane skills and configure `agents_sdk` executor with Codex tools.
- **Gemini IDE integration** — deferred until Gemini has a coding assistant with a skills/plugin system. Already supported as a model provider.
- **Approval granularity** — how the onboarding agent gets user approval for file writes (plan-level, per-file, or plan + diff review). Recommended: plan approval + final diff confirmation.
- **REPL frontend** — replace the TUI-based terminal frontend with a line-based REPL (like Claude Code's interface) for the `ap create` interactive session. The TUI works for now but a REPL is more natural for onboarding.
- **Remote executor scaffolding details** — Dockerfile generation, subprocess lifecycle management for the FastAPI wrapper.

---

## Related Documents

- [RUNTIME.md](RUNTIME.md) — Runtime initialization and store interfaces
- [EXECUTOR_SPEC.md](EXECUTOR_SPEC.md) — Executor type configuration
- [EXECUTOR_CONTRACT.md](EXECUTOR_CONTRACT.md) — Remote executor HTTP protocol
- [SKILLSDESIGN.md](SKILLSDESIGN.md) — Skills registry and discovery
- [CODE_TOOLS.md](CODE_TOOLS.md) — Tool system architecture
