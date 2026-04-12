---
name: generate-agent
description: Patterns and templates for generating valid agent plane agent directories. Load when ready to create files.
---

# Agent Generation

Use these patterns to generate a valid agent directory. Always generate
the minimal set of files needed — don't over-engineer.

## Step 1: Choose a directory name

Use the agent name in kebab-case: `my-research-agent/`

## Step 2: Generate config.yaml

Always include:
- `spec_version: 1`
- `name` (lowercase, hyphens OK)
- `description` (one sentence)
- `llm.model` in litellm format (provider/model-name)
- `llm.connection.api_key` using `${ENV_VAR}` syntax

Include if needed:
- `tools.builtins` if the agent needs built-in tools
- `interaction.modalities` if the agent handles images or files
- `executor.type` if not using the default `llm` executor

## Step 3: Generate AGENTS.md

Write a focused system prompt:
- Identity: "You are a [role] that [does what]."
- Capabilities: what tools/skills are available
- Constraints: what NOT to do
- Style: how to communicate

Keep it under 500 words for a starter agent. The user can expand later.

## Step 4: Generate skills (optional)

Only generate skills if the agent has distinct modes of operation.
Each skill needs:

```
skills/<skill-name>/SKILL.md
```

With YAML frontmatter:
```markdown
---
name: skill-name
description: One-line description of what this skill does.
---

Detailed instructions for when this skill is loaded...
```

## Templates

### Minimal agent (no tools, no skills)

**config.yaml:**
```yaml
spec_version: 1
name: {agent_name}
description: {description}
llm:
  model: {provider}/{model}
  connection:
    api_key: ${{{env_var}}}
instructions: AGENTS.md
```

**AGENTS.md:**
```markdown
You are {agent_name}, {description}.

Answer questions clearly and concisely. If you don't know something,
say so rather than guessing.
```

### Agent with web search

**config.yaml:**
```yaml
spec_version: 1
name: {agent_name}
description: {description}
llm:
  model: {provider}/{model}
  connection:
    api_key: ${{{env_var}}}
tools:
  builtins:
    - web_search
interaction:
  modalities:
    input: [text]
    output: [text]
instructions: AGENTS.md
```

### Agent wrapping existing framework code (remote executor)

**config.yaml:**
```yaml
spec_version: 1
name: {agent_name}
description: {description}
llm:
  model: {provider}/{model}
  connection:
    api_key: ${{{env_var}}}
executor:
  type: remote
  endpoint: http://localhost:5001
instructions: AGENTS.md
```

## Environment variable naming conventions

Map providers to their standard env var names:
- `openai` → `OPENAI_API_KEY`
- `anthropic` → `ANTHROPIC_API_KEY`
- `gemini` → `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `groq` → `GROQ_API_KEY`
- `deepseek` → `DEEPSEEK_API_KEY`
- `xai` → `XAI_API_KEY`
- `mistral` → `MISTRAL_API_KEY`
- `databricks` → `DATABRICKS_TOKEN`

## Validation checklist

Before presenting the generated files to the user, verify:
- [ ] `spec_version: 1` is present
- [ ] `name` is set and uses lowercase + hyphens
- [ ] `llm.model` uses `provider/model-name` format
- [ ] `llm.connection.api_key` uses `${ENV_VAR}` syntax (never a real key)
- [ ] Any `tools.agents` entries have matching `agents/` subdirectories
- [ ] Skill names match their directory names and use `[a-z0-9-]+` pattern
- [ ] `instructions` field points to a file that exists or is inline text
