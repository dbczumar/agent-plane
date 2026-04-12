---
name: detect-framework
description: Detect Python agent frameworks from code imports and map them to agent plane executor types. Load when the user has existing agent code to integrate.
---

# Framework Detection

When the user has existing Python code they want to integrate into
agent plane, detect the framework from import statements and recommend
the appropriate executor type.

## Detection Procedure

1. Ask the user for the path to their agent code (or look for Python
   files in the current directory if filesystem access is enabled).

2. Scan Python files for import patterns. Check in this priority order:

| Import pattern | Framework | Executor type |
|---------------|-----------|---------------|
| `import anthropic` or `from anthropic` + agent patterns (e.g. `Agent`, `tool`, system prompt setup) | Claude SDK | `claude_sdk` |
| `import openai` or `from openai` + agents patterns (e.g. `Agent`, `Runner`, `function_tool`) | OpenAI Agents SDK | `agents_sdk` |
| `from langgraph` or `import langgraph` | LangGraph | Native executor (planned — use `remote` for now) |
| `from deepagents` or `import deepagents` | DeepAgents | Native executor (planned — use `remote` for now) |
| `from langchain` or `import langchain` | LangChain | `remote` (native planned) |
| `from crewai` or `import crewai` | CrewAI | `remote` |
| `from autogen` or `import autogen` | AutoGen | `remote` |
| None of the above | Unknown | `remote` |

3. Report what you found and recommend the executor type.

## What to generate for each executor type

### `llm` (default — no existing code)

Generate a standard agent directory:
```yaml
executor:
  type: llm  # or omit entirely (llm is the default)
```

### `claude_sdk`

The user's Claude SDK code runs directly. Generate config that points
to their entry module:
```yaml
executor:
  type: claude_sdk
```

### `agents_sdk`

The user's OpenAI Agents SDK code runs directly:
```yaml
executor:
  type: agents_sdk
```

### `remote`

For unsupported frameworks, the agent runs behind an HTTP endpoint.
Generate a FastAPI wrapper app that:
1. Imports the user's agent
2. Exposes it via the remote executor protocol
3. Lives inside the user's agent directory

```yaml
executor:
  type: remote
  endpoint: http://localhost:5001
```

## Asking about unsupported frameworks

If the user's framework is not natively supported, let them know:
- Explain that a remote executor wrapper will be generated
- Offer to show them a pre-filled GitHub issue URL requesting first-class
  support for their framework
- The issue URL format: `https://github.com/dbczumar/agent-plane/issues/new?title=...&body=...`
