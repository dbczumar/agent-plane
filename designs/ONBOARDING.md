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

3. **"I want a model but I don't know what's available."** The user
   doesn't know which LLM providers/models they can use or which API
   keys they have configured.

The onboarding system should handle all three cases with minimal manual
steps, leveraging the user's existing tools (Claude Code, Codex, etc.)
where available and falling back to a CLI-driven flow otherwise.

### Current state

| Component | Status |
|-----------|--------|
| Agent spec format (`config.yaml`, `AGENTS.md`, skills, tools) | Stable |
| Executor types (`llm`, `claude_sdk`, `agents_sdk`, `remote`) | Stable |
| CLI (`ap server`) | Stable, no `onboard`/`init` command |
| Claude Code skills for agent-plane development | Exists in `.claude/skills/agent-plane-dev/` |
| MLflow model config discovery | Not implemented |
| Coding assistant integration (Claude Code, Codex, Gemini) | Not implemented |
| Framework detection / scaffolding | Not implemented |

---

## Design Premise

The onboarding CLI (`ap onboard` or `ap init`) launches an **agent-plane
onboarding agent** — an agent-plane agent that guides the user through
setup. This dogfoods agent-plane itself: the onboarding agent runs on
agent-plane, has filesystem access (with user approval), and can
generate agent directories, install skills, and configure models.

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

**Q1. Onboarding agent vs. CLI wizard — which is primary?**

The premise says the CLI "just launches an agent-plane onboarding agent
with access to file system." Two interpretations:

- **(a) Agent-only.** `ap onboard` starts a temporary agent-plane server,
  loads the onboarding agent, and opens a conversational session. The
  agent does all the work (detect environment, generate files, install
  skills). No traditional CLI prompts.
- **(b) Hybrid.** A thin CLI collects a few mandatory inputs (target
  directory, approval scope), then hands off to the onboarding agent
  for the rest.

Recommended default: **(b) Hybrid** — the CLI handles server lifecycle
and approval gating; the agent handles discovery and generation.

**Q2. Where does the onboarding agent live?**

Options:

- **(a)** `examples/agents/onboarding/` — example agent, not shipped with
  the package.
- **(b)** `agent_plane/onboarding/agent/` — first-class built-in, shipped
  with the package and referenced by `ap onboard`.
- **(c)** Separate package/repo.

Recommended default: **(b)** — the onboarding agent is part of
agent-plane and `ap onboard` knows where to find it.

**Q3. "Implement MLflow assistant on agent plane" — scope and relationship**

Is the MLflow assistant:

- **(a)** A separate agent (`examples/agents/mlflow-assistant/`) that
  helps users interact with MLflow (query experiments, compare runs,
  deploy models to serving endpoints)?
- **(b)** A capability of the onboarding agent (it queries MLflow to
  discover available models)?
- **(c)** Both — the onboarding agent uses MLflow for model discovery,
  and there is also a standalone MLflow assistant agent?

This determines whether the MLflow assistant design belongs in this doc
or in a separate `MLFLOW_ASSISTANT.md`.

---

### Framework Detection & Agent Generation

**Q4. "Supported frameworks" — detection semantics**

Agent-plane supports four executor types. When the premise says "if it's
one of the supported frameworks," does this mean:

- **(a)** The user has a **Claude SDK** or **OpenAI Agents SDK** agent
  and we wire it to the corresponding executor (`claude_sdk`,
  `agents_sdk`).
- **(b)** The user has a **LangChain / CrewAI / AutoGen / etc.** agent
  and we detect the framework from their code, then generate a `remote`
  executor config that wraps it via an HTTP adapter.
- **(c)** Both (a) and (b).

Recommended default: **(c)** — detect Claude SDK and OpenAI Agents SDK
for native executors; detect other frameworks for remote executor
scaffolding.

Detection signals (in priority order):

| Signal | Framework | Executor |
|--------|-----------|----------|
| `import anthropic` + Claude agent patterns | Claude SDK | `claude_sdk` |
| `import openai` + Agents SDK patterns | OpenAI Agents SDK | `agents_sdk` |
| `from langchain` imports | LangChain | `remote` (scaffold HTTP wrapper) |
| `from crewai` imports | CrewAI | `remote` (scaffold HTTP wrapper) |
| `from autogen` imports | AutoGen | `remote` (scaffold HTTP wrapper) |
| None of the above | Unknown | `remote` (scaffold HTTP wrapper) + file issue |

**Q5. Remote executor scaffolding**

For unsupported frameworks, the onboarding agent generates a thin HTTP
wrapper implementing the remote executor protocol (as defined in
`EXECUTOR_CONTRACT.md`). Questions:

- Is the generated wrapper a standalone FastAPI app?
- Does it live inside the user's agent directory or beside it?
- Should the onboarding agent also generate a `Dockerfile`?

**Q6. "File a GitHub issue" — automatic or manual?**

For unsupported frameworks, should the onboarding flow:

- **(a)** Automatically file an issue on `dbczumar/agent-plane` requesting
  first-class support (requires GitHub auth).
- **(b)** Print a pre-filled issue URL that the user can click to file.
- **(c)** Ask the user whether they want to file an issue and do (a) or
  (b) based on their answer.

Recommended default: **(c)** — ask, then use whichever method the user
has auth for.

---

### Model Configuration & MLflow

**Q7. "Default model based on list of model confs from MLflow"**

Model configs are currently static in `config.yaml`. The onboarding
agent needs to pick a default. Options:

- **(a)** Query MLflow model serving endpoints (or Databricks serving
  endpoints) to discover models available to the user, then select one.
- **(b)** Probe for API keys in the environment (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `DATABRICKS_TOKEN`, etc.) and select the best
  available model from a hardcoded priority list.
- **(c)** Combine: check environment for API keys first, then optionally
  query MLflow/Databricks serving if credentials are present.

Implementation note: MLflow's model serving endpoint listing is
available via `GET /api/2.0/serving-endpoints` on Databricks workspaces.
The onboarding agent would need `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` to call this.

**Q8. Provider priority**

If multiple API keys are present, what is the default model selection
order? Proposed:

| Priority | Provider | Model | Env var |
|----------|----------|-------|---------|
| 1 | Anthropic | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| 2 | OpenAI | `openai/gpt-5.4` | `OPENAI_API_KEY` |
| 3 | Databricks | `databricks/<serving-endpoint>` | `DATABRICKS_TOKEN` |
| 4 | Google | `gemini/gemini-2.5-pro` | `GOOGLE_API_KEY` |

This also determines what model the **onboarding agent itself** uses.
The onboarding agent needs a model to run — it should use whatever is
available, with the priority above.

---

### Skills Installation & IDE Integration

**Q9. "Install Claude skills if user has Claude Code"**

Clarify what "Claude skills" means here:

- **(a)** Agent-plane skills (files in the generated agent's `skills/`
  directory) that teach the agent how to perform tasks.
- **(b)** Claude Code skills (files in `.claude/skills/`) that teach
  Claude Code how to work with agent-plane — e.g., the existing
  `agent-plane-dev` skill.
- **(c)** Both — install agent-plane dev skills into Claude Code AND
  generate agent skills for the new agent.

Recommended default: **(c)** — when Claude Code is detected, install
the `agent-plane-dev` skill into the user's `.claude/skills/` so their
Claude Code sessions understand agent-plane conventions. Also generate
starter skills for the new agent.

Detection: check for `~/.claude/` directory or `claude` on `$PATH`.

**Q10. Codex support**

What does Codex onboarding look like?

- **(a)** Detect Codex installation, configure the agent to use
  `agents_sdk` executor with `codex:Shell` and `codex:ApplyPatch`
  tools.
- **(b)** Install Codex-specific skills (analogous to Claude Code
  skills) that teach Codex how to work with agent-plane.
- **(c)** Both.

Detection: check for `codex` on `$PATH` or OpenAI Codex SDK imports.

**Q11. Gemini support**

Include Gemini as a supported IDE/assistant for skills installation, or
defer? The LLM layer already supports `gemini/` as a provider, but
there is no Gemini IDE integration today.

Recommended default: **Defer** — support Gemini as a model provider but
not as an IDE/assistant for skills installation. Revisit when Gemini has
a coding assistant with a skills/plugin system.

---

### User Flow & Approval

**Q12. Approval granularity**

The onboarding agent has filesystem access and needs user approval.
Options:

- **(a)** Approve the overall plan ("I'll create an agent at
  `./my-agent/` with config X, skills Y, tools Z"). Single approval,
  then the agent executes.
- **(b)** Approve each file write individually.
- **(c)** Approve the plan, then show a summary diff before final write.

Recommended default: **(c)** — plan approval + final diff confirmation.
Individual file approval is too noisy; plan-only approval doesn't let
the user review the output.

**Q13. CLI entry point**

Proposed command: `ap init`

```
ap init [--target DIR] [--model MODEL] [--framework FRAMEWORK] [--non-interactive]
```

- `--target`: Directory to create the agent in (default: current
  directory).
- `--model`: Override model selection (skip auto-detection).
- `--framework`: Override framework detection (skip auto-detection).
- `--non-interactive`: Skip the onboarding agent, use defaults +
  flags only. For CI/scripted use.

The command starts a temporary agent-plane server, loads the onboarding
agent, and opens a conversational session. When the session ends, the
server shuts down.

Alternative name: `ap onboard`. Preference?

---

## Proposed User Flows

Pending resolution of the open questions above, the following flows are
sketched at high level.

### Flow 1: "I don't have an agent"

```
$ ap init
Detecting environment...
  Claude Code: found (v2.1.0)
  API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY
  Default model: anthropic/claude-sonnet-4-20250514

Starting onboarding agent...

Agent: I see you have Claude Code and Anthropic API access. I'll help
       you create an agent. What would you like your agent to do?

User:  I want a research assistant that can search the web and
       summarize findings.

Agent: I'll create a research agent with:
       - Model: anthropic/claude-sonnet-4-20250514
       - Tools: web_search, summarize
       - Skills: deep-research
       - Executor: llm (default)

       Target directory: ./research-agent/

       [Approve plan? y/n]

User:  y

Agent: [generates files, shows diff summary]

       Created:
         research-agent/config.yaml
         research-agent/AGENTS.md
         research-agent/skills/deep-research/SKILL.md

       Also installed agent-plane-dev skill into .claude/skills/

       To start: ap server --agent ./research-agent/
       [Approve writes? y/n]

User:  y

Agent: Done! Run `ap server --agent ./research-agent/` to start.
```

### Flow 2: "I have a LangChain agent"

```
$ ap init --target ./my-langchain-agent/
Detecting environment...
  Framework detected: LangChain (from ./agent.py imports)
  Claude Code: not found
  API keys: OPENAI_API_KEY
  Default model: openai/gpt-5.4

Starting onboarding agent...

Agent: I detected a LangChain agent in your code. LangChain isn't
       natively supported as an executor, but I can create a remote
       executor wrapper that exposes your agent via HTTP and plugs it
       into agent-plane.

       I'll generate:
       - A FastAPI wrapper around your LangChain agent
       - An agent-plane config.yaml with executor type: remote
       - A Dockerfile for the wrapper

       Want me to also file a GitHub issue requesting first-class
       LangChain support? [y/n]

User:  y

Agent: [generates files, shows diff, files issue]
```

### Flow 3: "I have a Claude SDK agent"

```
$ ap init --target ./my-claude-agent/
Detecting environment...
  Framework detected: Claude SDK (from ./agent.py imports)
  Claude Code: found (v2.1.0)
  API keys: ANTHROPIC_API_KEY
  Default model: anthropic/claude-sonnet-4-20250514

Starting onboarding agent...

Agent: I detected a Claude SDK agent. I'll configure agent-plane to
       use the claude_sdk executor, which runs your agent natively.

       I'll generate:
       - config.yaml with executor type: claude_sdk
       - AGENTS.md with your agent's instructions
       - Skills ported from your existing prompts

       [Approve plan? y/n]
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
        detect-environment/
          SKILL.md          # Detect installed tools, API keys, frameworks
        generate-agent/
          SKILL.md          # Generate agent directory from template
        install-skills/
          SKILL.md          # Install IDE skills (Claude Code, Codex)
      tools/
        python/
          detect_env.py     # Environment detection tool
          scaffold.py       # File scaffolding tool
    cli.py                  # `ap init` command implementation
```

### CLI entry point addition

```python
# In agent_plane/cli.py, add:
@cli.command()
@click.option("--target", default=".", help="Directory to create the agent in.")
@click.option("--model", default=None, help="Override model selection.")
@click.option("--framework", default=None, help="Override framework detection.")
@click.option("--non-interactive", is_flag=True, help="Use defaults, no agent session.")
def init(target: str, model: str | None, framework: str | None, non_interactive: bool) -> None:
    """Initialize a new agent-plane agent with guided onboarding."""
    ...
```

### Model discovery

```python
# Pseudocode for model selection
def discover_default_model() -> str:
    """
    Discover the best available model based on environment.

    Probes for API keys in priority order, optionally queries
    MLflow/Databricks serving endpoints if credentials are present.

    :return: Model identifier string, e.g. "anthropic/claude-sonnet-4-20250514".
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-20250514"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-5.4"
    if os.environ.get("DATABRICKS_TOKEN"):
        # Optionally query serving endpoints
        return discover_databricks_model()
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-pro"
    raise ValueError("No API keys found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or another provider key.")
```

---

## Related Documents

- [RUNTIME.md](RUNTIME.md) — Runtime initialization and store interfaces
- [EXECUTOR_SPEC.md](EXECUTOR_SPEC.md) — Executor type configuration
- [EXECUTOR_CONTRACT.md](EXECUTOR_CONTRACT.md) — Remote executor HTTP protocol
- [SKILLSDESIGN.md](SKILLSDESIGN.md) — Skills registry and discovery
- [CODE_TOOLS.md](CODE_TOOLS.md) — Tool system architecture
