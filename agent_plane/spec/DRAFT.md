# spec/ — What an agent IS

Depends on: nothing

Pure data types and parsing logic. No network, no database, no server concepts.
If you point it at an agent repo directory, it gives you back a typed object.

## Planned files

### types.py
Core dataclasses representing the agent specification:
- `AgentSpec` — top-level container for a fully parsed agent
- `LLMConfig` — model name, temperature, max_tokens, provider-specific params
- `SkillMetadata` — name, description, tools list, content body, reference paths, script paths
- `MCPServerConfig` — name, description, transport type, command/args/env or url/headers
- `LocalToolInfo` — name, description, language, file path relative to repo root
- `AgentConfig` — custom key-value config section from config.yaml

### parser.py
Parse an extracted agent repo directory into an `AgentSpec`:
- Read and parse `config.yaml` → `LLMConfig` + `AgentConfig`
- Read `AGENTS.md` → instructions string
- Discover `skills/*/SKILL.md` → list of `SkillMetadata`
- Discover `tools/mcp/*.yaml` → list of `MCPServerConfig`
- Discover `tools/local/<language>/*` → list of `LocalToolInfo`

### validator.py
Validate a parsed `AgentSpec`:
- Skill names: max 64 chars, lowercase + numbers + hyphens, matches directory name
- Skill descriptions: max 1024 chars
- config.yaml: LLM model field present
- MCP configs: transport type valid, required fields present per transport
- No duplicate skill/tool names

### tar_utils.py
Safe tarball extraction:
- Path traversal protection (all entries must resolve within extraction root)
- Symlink rejection (no symlinks pointing outside extraction root)
- Decompression bomb protection (max uncompressed size, max file count)
- Permission sanitization (strip setuid, apply safe defaults)

## Key design decisions
- All types are plain dataclasses (or Pydantic models), not ORM models
- Parser is purely filesystem-based: takes a `Path`, returns an `AgentSpec`
- No awareness of bundles, tarballs, or storage — that's server's job
- tar_utils lives here because extracting a bundle is a prerequisite to parsing
