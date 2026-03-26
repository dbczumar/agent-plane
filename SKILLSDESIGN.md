# Skills Design

Design document for skill support in agent-plane, aligned with the
[Agent Skills specification](https://agentskills.io/specification).

## Current State

Skill support is implemented with a registry-based tool pattern:

| Component | File | Status |
|-----------|------|--------|
| `SkillSpec` dataclass | `agent_plane/spec/types.py` | name, description, content, skill_dir |
| Discovery from `skills/<name>/SKILL.md` | `agent_plane/spec/parser.py` | YAML frontmatter + markdown body |
| `Tool` ABC | `agent_plane/tools/base.py` | Abstract base for all tools |
| `LoadSkillTool` | `agent_plane/tools/builtins/load_skill.py` | Loads skill instructions |
| `ReadSkillFileTool` | `agent_plane/tools/builtins/read_skill_file.py` | Reads skill directory resources |
| `ToolManager` registry | `agent_plane/tools/manager.py` | `dict[str, Tool]` dispatch |
| `McpServerConnection` + `McpTool` | `agent_plane/tools/mcp.py` | MCP server connections + proxy tools |
| Skill hints in system prompt | `agent_plane/runtime/prompt.py` | Lists name + description |
| Validation (name format, uniqueness) | `agent_plane/spec/validator.py` | kebab-case, length limits |
| Tool dispatch via `_call_tool` | `agent_plane/runtime/workflow.py` | Routes through `ToolManager` |

### How it works today

1. **Parse time**: `_discover_skills()` reads `skills/<name>/SKILL.md`,
   extracts YAML frontmatter (`name`, `description`) and markdown body
   (`content`) into `SkillSpec`. `skill_dir` is set to the skill's
   directory path on disk.
2. **Prompt time**: `build_instructions()` appends a skill menu:
   `"Available skills (use the load_skill tool to load one):
   - code-review: Reviews code for quality..."`.
3. **Tool registration**: `ToolManager.__init__` creates `LoadSkillTool`
   and `ReadSkillFileTool` (if any skill has resources) and registers
   them in `self._tools: dict[str, Tool]`.
4. **Tool schema**: `get_tool_schemas()` returns schemas from all
   registered tools via `tool.get_schema()`.
5. **Runtime**: LLM calls `load_skill({"name": "code-review"})` ->
   `ToolManager.call_tool()` dispatches to `LoadSkillTool.invoke()` ->
   returns skill content with resource file listing.
6. **Resources**: LLM calls `read_skill_file({"skill_name":
   "code-review", "path": "references/style-guide.md"})` ->
   dispatches to `ReadSkillFileTool.invoke()` -> reads file with
   path traversal protection.

### Tool registry pattern

Tools implement the `Tool` ABC with three methods:
- `name` (property): unique identifier for dispatch
- `get_schema()`: OpenAI Chat Completions tool schema
- `invoke(arguments)`: execute the tool, return string result

`ToolManager` maintains a `dict[str, Tool]` registry. Dispatch is
`self._tools[name].invoke(arguments)` — no hardcoded if/elif chains.
New tool types (MCP, local) will implement `Tool` and register
themselves in `ToolManager`.

## Gaps vs. Official Spec

### Gap #1: Per-skill agent config (Not Yet)

The official spec allows `skills/<name>/agents/openai.yaml` to
override the LLM config (model, temperature) when a skill is active.
We don't parse or honor this.

Lower priority. When needed, add `agent_config: dict[str, Any] | None`
to `SkillSpec` and parse from `agents/openai.yaml` if present. The
workflow would swap LLM config when a skill is active.

### Gap #2: Local tool execution (Not Yet)

The agent spec supports `tools/python/*.py` and `tools/typescript/*.ts`
auto-discovered local tools (`LocalToolInfo` in `types.py`). The parser
discovers them and the validator checks for duplicate names, but the
runtime cannot execute them yet.

Industry patterns: decorated functions (`@tool`), schema inference from
type hints + docstrings, sandboxed subprocess execution. When
implemented, each local tool file will be loaded as a `Tool` subclass
and registered in the `ToolManager` alongside MCP and built-in tools.

Deferred until sandboxing design is settled (subprocess? container?
WASM?).

### Gap #3: Skill parameters / `params` injection (Not Yet)

The spec says skills can read agent-level `params` (defined in
`config.yaml`). `AgentSpec.params` exists but isn't surfaced to
skills at load time.

When needed: when `load_skill` returns content, append a
`## Parameters` section with the agent's `params` dict so the skill
instructions can reference them. Or expose a `get_params` built-in
tool.

## Implementation Order

1. ~~`allowed_tools` enforcement~~ (removed — runtime gating is the
   wrong approach; the LLM should follow skill instructions naturally)
2. ~~Skill directory path + `read_skill_file` tool~~ (done)
3. Per-skill agent config (deferred)
4. Params injection (deferred)
