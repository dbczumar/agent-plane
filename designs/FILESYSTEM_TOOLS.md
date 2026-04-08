# Built-in Filesystem Tools

## Problem

Agents need filesystem access (read, write, list, search) and a way to
publish files to clients. Today this requires either the Claude SDK
executor (which has its own built-in tools) or an external MCP server
(separate process, extra config). There's no built-in filesystem for
DefaultExecutor or RemoteExecutor agents.

The official Anthropic MCP filesystem server is Node.js-only, has had
two path traversal CVEs (CVE-2025-53109, CVE-2025-53110), and no
popular Python equivalent exists (largest is 22 GitHub stars).

---

## Decision

Implement filesystem tools as a Python built-in in agent-plane. ~150
lines. Same tool set as the Anthropic MCP server, plus `publish_file`
for uploading to the file store.

---

## Agent Spec

```yaml
tools:
  builtins:
    - name: filesystem
      root: /home/user/project   # allowed directory
      writable: true              # default: false (read-only)
```

`root` is required. Resolved to an absolute path at parse time. All
tool operations are confined to this directory.

`writable` controls whether `write_file` and `edit_file` are
registered. Read-only by default — agents must opt in to writes.

---

## Tools

| Tool | Args | Writable? | Description |
|------|------|-----------|-------------|
| `read_file` | `path` | no | Read text file contents |
| `write_file` | `path`, `content` | yes | Create or overwrite a file |
| `edit_file` | `path`, `old_text`, `new_text` | yes | Find-and-replace in a file |
| `list_directory` | `path` (default `"."`) | no | List entries with `/` suffix for dirs |
| `search_files` | `pattern`, `path` (default `"."`) | no | Glob search for files |
| `publish_file` | `path` | no | Upload to file store, return `file_id` |

`publish_file` is the bridge between "file on disk" and "downloadable
by the client." It reads the file at `path`, stores it via
`file_store.create()` + `artifact_store.put()`, and returns a JSON
result with `file_id`, `filename`, and `content_type`. The client
downloads via `GET /v1/files/{file_id}/content`.

---

## Path Validation

All paths are resolved and validated before any I/O:

```python
def _safe_resolve(path: str, root: Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes allowed directory: {path}")
    # Reject symlinks that point outside root
    if resolved.is_symlink():
        target = resolved.resolve(strict=True)
        if not target.is_relative_to(root):
            raise ValueError(f"Symlink escapes allowed directory: {path}")
    return resolved
```

This covers:
- `../` traversal (resolve + is_relative_to)
- Symlink escape (resolve target + is_relative_to)
- Absolute paths outside root (is_relative_to catches `/etc/passwd`)

---

## Implementation

### New file

**`tools/builtins/filesystem.py`** — `FilesystemTool` class:

- Constructor takes `root: str` and `writable: bool` from config
- Resolves `root` to absolute `Path` at init
- `name()` returns `"filesystem"` (but registers multiple tool
  schemas — one per operation)
- `invoke(operation, args, context)` dispatches to the right handler
- `publish_file` needs `file_store` and `artifact_store` — accessed
  via `ToolContext` (requires adding these to `ToolContext`)

### Changed files

**`tools/builtins/__init__.py`** — Register `FilesystemTool` in
`_BUILTIN_REGISTRY`.

**`tools/base.py`** — Add `file_store` and `artifact_store` to
`ToolContext` so `publish_file` can store files.

**`tools/manager.py`** — Pass `file_store` and `artifact_store`
when constructing `ToolContext`.

---

## Multi-Tool Registration

`FilesystemTool` exposes 5-6 tools from one builtin config entry.
Two approaches:

**A: One Tool class, multiple schemas.** `get_schema()` returns a
list instead of a single dict. `invoke()` dispatches on operation
name. Requires `ToolManager` to handle multi-schema tools.

**B: Factory pattern.** `get_builtin_tool("filesystem", config)`
returns a list of `Tool` instances — `ReadFileTool`,
`WriteFileTool`, `ListDirectoryTool`, etc. Each is a simple
single-operation tool. `ToolManager` registers each one.

**Decision: B (factory).** Each operation is its own `Tool` with
its own schema. Simpler — no dispatch, no multi-schema changes to
`ToolManager`. The factory function returns the appropriate set
based on `writable`.

```python
def create_filesystem_tools(
    config: dict[str, str],
) -> list[Tool]:
    root = Path(config["root"]).resolve()
    writable = config.get("writable", "false").lower() == "true"
    tools = [
        ReadFileTool(root),
        ListDirectoryTool(root),
        SearchFilesTool(root),
        PublishFileTool(root),  # needs file_store access
    ]
    if writable:
        tools.append(WriteFileTool(root))
        tools.append(EditFileTool(root))
    return tools
```

---

## `publish_file` and the Response

When the agent calls `publish_file(path="output/chart.png")`:

1. Tool reads bytes from `{root}/output/chart.png`
2. Stores via `file_store.create()` + `artifact_store.put()`
3. Returns: `{"file_id": "file_abc123", "filename": "chart.png",
   "content_type": "image/png"}`

The `file_id` appears in the tool result. The workflow can detect
`file_id` references in tool results and automatically add
`file_citation` annotations to the assistant message (per
OUTPUT_ATTACHMENTS.md). Or the agent can reference the file_id in
its text naturally.

---

## What This Enables

```
User: "Analyze sales.csv and make me a bar chart"

Agent (DefaultExecutor, filesystem root=/home/user/project):
  1. read_file(path="sales.csv")          → reads CSV content
  2. write_file(path="analysis.py", ...)  → writes Python script
  3. [Bash tool or code execution]        → runs script, generates chart
  4. publish_file(path="output/chart.png") → uploads to file store
  5. Agent says: "Here's the chart (file_id: file_abc123)"
  
Client downloads: GET /v1/files/file_abc123/content → chart.png
```

Works with DefaultExecutor (no Claude SDK needed), works with any
model, uses the existing file store infrastructure.
