# `introspect` built-in tool

## Context

Agents sometimes need to understand their own configuration — what
model they're using, what tools are available, what their instructions
say, what skills they have. This is useful for:

- **Self-debugging**: "Why can't I do X?" → introspect, see the tool
  isn't configured.
- **Self-description**: "What can you do?" → introspect, list tools
  and skills accurately instead of guessing.
- **Self-improvement**: The onboarding assistant can introspect to
  understand what tools it has in the current mode.

Currently agents have no way to examine their own spec. They rely on
hardcoded knowledge in their AGENTS.md, which can drift from the
actual config.

## Design

`introspect` is a function tool that lets an agent browse its own
`AgentSpec` progressively. With no arguments it returns a high-level
summary. With a `section` parameter it drills into specific parts —
skills, instructions, sub-agents, tools — without dumping everything
into context at once.

### Tool schema

```
name: introspect
parameters:
  section: string (optional) — path to a specific part of the spec.
    If omitted, returns a high-level summary.
```

### Section paths

| Section | Returns |
|---------|---------|
| *(empty)* | High-level summary: name, model, executor, tools list, skills list, sub-agents list |
| `instructions` | Full AGENTS.md content |
| `config` | Raw config.yaml content |
| `skills` | List of all skills with names and descriptions |
| `skills/<name>` | That skill's full SKILL.md content |
| `tools` | All tool details (builtins, MCP servers, local tools) |
| `sub_agents` | List of sub-agents with names and descriptions |
| `sub_agents/<name>` | That sub-agent's summary (name, model, tools) |
| `sub_agents/<name>/instructions` | That sub-agent's instructions |
| `sub_agents/<name>/skills` | That sub-agent's skills list |
| `sub_agents/<name>/skills/<skill>` | That sub-agent's skill content |

Recursive — sub-agents are full specs, so the same section paths
work at any nesting depth.

### Output examples

**`introspect()`** — summary:
```
Agent: archer
Model: openai/gpt-5.4
Executor: llm

Tools:
  builtins: web_search, web_fetch, code_sandbox, upload_file
  sub-agents: fact_checker, summarizer
  mcp: github (http://...)
  local: word_count

Skills: deep-research, explain

Interaction:
  conversational: true
  input: text, image, file
  output: text

Use introspect(section="...") to drill into any section.
```

**`introspect(section="skills/deep-research")`** — skill content:
```
Skill: deep-research
Description: Investigate a topic in depth...

---
When researching a topic:
1. Search broadly first...
...
```

**`introspect(section="sub_agents/fact_checker/instructions")`**:
```
You are a fact-checker. When given a claim or statement, search
the web to find evidence that supports or contradicts it...
```

### Computing the high-level summary

The summary is built directly from `AgentSpec` fields — no LLM call,
no external lookup, just field access:

```python
def _format_summary(spec: AgentSpec) -> str:
    lines = []
    lines.append(f"Agent: {spec.name or '(unnamed)'}")
    if spec.description:
        lines.append(f"Description: {spec.description}")
    if spec.llm:
        lines.append(f"Model: {spec.llm.model}")
    lines.append(f"Executor: {spec.executor.type or 'llm'}")
    lines.append("")

    # Tools section
    lines.append("Tools:")
    if spec.tools.builtins:
        names = [b.name for b in spec.tools.builtins]
        lines.append(f"  builtins: {', '.join(names)}")
    if spec.tools.agents:
        lines.append(f"  sub-agents: {', '.join(spec.tools.agents)}")
    if spec.mcp_servers:
        for mcp in spec.mcp_servers:
            lines.append(f"  mcp: {mcp.name} ({mcp.url})")
    if spec.local_tools:
        names = [t.name for t in spec.local_tools]
        lines.append(f"  local: {', '.join(names)}")
    lines.append("")

    # Skills
    if spec.skills:
        names = [s.name for s in spec.skills]
        lines.append(f"Skills: {', '.join(names)}")
        lines.append("")

    # Interaction
    lines.append("Interaction:")
    lines.append(f"  conversational: {spec.interaction.conversational}")
    lines.append(f"  input: {', '.join(spec.interaction.modalities.input)}")
    lines.append(f"  output: {', '.join(spec.interaction.modalities.output)}")

    lines.append("")
    lines.append('Use introspect(section="...") to drill into any section.')
    return "\n".join(lines)
```

All data comes from the already-parsed `AgentSpec` dataclass. No I/O,
no env vars, no external state.

### Implementation

The tool needs access to the `AgentSpec`. Same pattern as `web_fetch`
— `ToolManager` passes `self._spec` to the constructor:

```python
class IntrospectTool(Tool):
    def __init__(self, spec: AgentSpec):
        self._spec = spec

    def invoke(self, arguments, ctx) -> str:
        section = parsed.get("section")
        if not section:
            return _format_summary(self._spec)
        return _resolve_section(self._spec, section)
```

`_resolve_section` walks the spec tree based on the section path.
Unknown sections return a clear error listing valid options at that
level.

### Registration

Handled in `ToolManager._create_builtin` (needs `self._spec`), same
as `web_fetch`. Agents opt in via:

```yaml
tools:
  builtins:
    - introspect
```

## Files

| File | Action |
|------|--------|
| `agent_plane/tools/builtins/introspect.py` | Create |
| `agent_plane/tools/builtins/__init__.py` | Register |
| `agent_plane/tools/manager.py` | Pass spec in `_create_builtin` |
| `agent_plane/onboarding/agent/config.yaml` | Add to onboarding agent |
| `tests/tools/builtins/test_introspect.py` | Create |
