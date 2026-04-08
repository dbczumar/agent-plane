# Sandboxed Local Tool Execution

## Context

Local Python tools (`tools/python/*.py`) currently execute **in-process** via
`asyncio.run(module.run(args))`. A crash, OOM, or segfault in a tool kills the
entire server. No agent framework sandboxes tool functions by default — this
would make agent-plane the first.

We want subprocess-based execution with four tiers that auto-detect
based on what's available:

1. **Plain subprocess** (default) — crash isolation, no external deps
2. **uv** (auto if tool has PEP 723 inline deps and `uv` on PATH) — auto-installs tool dependencies
3. **srt** (auto if `srt` on PATH) — OS-level filesystem/network sandboxing via Anthropic Sandbox Runtime
4. **Docker** (opt-in via `sandbox.docker_image` config) — full container isolation

Tiers compose: srt wraps uv which wraps python. Priority:
Docker > srt+uv > srt > uv > plain.

### Industry comparison

No framework sandboxes tool function calls:

| Framework | Tool execution | Sandboxing |
|-----------|---------------|------------|
| LangGraph/LangChain | In-process | None |
| CrewAI | In-process | None |
| AutoGen | In-process (tools), subprocess (code) | Docker optional for code only |
| smolagents | AST interpreter | E2B/Pyodide optional for code |
| Pydantic AI | In-process | Monty for code only |
| Claude Code | OS-level (bubblewrap/Seatbelt) | Yes — the outlier |
| **agent-plane (proposed)** | **Subprocess** | **srt/Docker optional** |

### srt (Anthropic Sandbox Runtime)

`srt` is a CLI (`npm install -g @anthropic-ai/sandbox-runtime`) that wraps
any command in OS-level sandboxing:

- **macOS**: Seatbelt profiles (Apple's native sandbox)
- **Linux**: bubblewrap + seccomp BPF + network namespace isolation
- Filesystem: deny-then-allow write restrictions
- Network: all traffic forced through a proxy with domain allowlisting
- Works on entire process trees (children inherit restrictions)

Usage: `srt python tool.py` — no code changes needed in the tool.

---

## Execution tiers

```
LLM emits function_call("my_tool", args)
  -> workflow._call_tool (async @step, thread pool)
    -> execute_tool_with_retry (timeout + retry)
      -> ToolManager.call_tool
        -> LocalPythonTool.invoke
          -> subprocess: [command prefix] python _runner.py
```

The command prefix varies by tier:

| Tier | Command | When |
|------|---------|------|
| Plain subprocess | `python _runner.py` | Default (always works) |
| uv + deps | `uv run --with dep1 --with dep2 -- python _runner.py` | Tool has PEP 723 `# /// script` metadata |
| srt sandbox | `srt python _runner.py` | `srt` on PATH and `sandbox.enabled: true` |
| srt + uv | `srt uv run --with ... -- python _runner.py` | Both conditions |
| Docker | `docker run --rm -i --network none ... python _runner.py` | `sandbox.docker_image` configured |

Priority: Docker > srt+uv > srt > uv > plain.

### PEP 723 inline dependency support

Tool authors declare deps directly in the file — no extra config files:

```python
# /// script
# dependencies = ["requests>=2.28", "beautifulsoup4"]
# ///

SCHEMA = { ... }

async def run(args):
    import requests
    return requests.get(args["url"]).text
```

`uv` auto-creates a cached venv on first run (~1-2s), reuses it on
subsequent runs (~50ms). Zero config for the tool author.

---

## Subprocess protocol: fd 3

The runner (`agent_plane/tools/_runner.py`) communicates via a dedicated
file descriptor so stdout/stderr remain free for tool debugging.

**Parent -> child** (stdin):
```json
{"module_path": "/abs/path/to/tool.py", "arguments": {"query": "test"}}
```

**Child -> parent** (fd 3):
```json
{"result": "search results..."}
```
or
```json
{"error": "TypeError: missing required argument 'query'"}
```

**Docker fallback**: fd 3 isn't available through `docker run -i`. When
`_AP_RESPONSE_MODE=stdout` env var is set, the runner writes the response
to stdout with a `__AP_RESPONSE__:` prefix instead.

### Timeout and cancellation

Current timeout uses `ThreadPoolExecutor.result(timeout=N)`, which leaks
the thread if the tool hangs. With subprocess execution:

1. `call_tool_with_timeout` gains a `tool: Tool | None` param
2. On timeout, calls `tool.cancel()` before raising
3. `LocalPythonTool.cancel()` calls `self._proc.kill()` (SIGKILL)
4. Subprocess dies immediately — no thread leak, no zombie process

---

## Files to create

### `agent_plane/tools/_runner.py`

Subprocess entry point. ~60 lines.

```python
_RESPONSE_FD = 3

def main() -> None:
    request = json.loads(sys.stdin.buffer.read())
    module = _load_module(request["module_path"])
    try:
        result = module.run(request["arguments"])
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except Exception as exc:
        _write_error(f"{type(exc).__name__}: {exc}")
        return
    _write_response({"result": str(result) if not isinstance(result, str) else result})

def _write_response(data: dict) -> None:
    fd = _get_output_fd()
    os.write(fd, json.dumps(data).encode())
    os.close(fd)

def _get_output_fd() -> int:
    if os.environ.get("_AP_RESPONSE_MODE") == "stdout":
        return sys.stdout.fileno()  # Docker mode
    return _RESPONSE_FD
```

### `agent_plane/tools/_pep723.py`

PEP 723 inline metadata parser. ~40 lines.

```python
@dataclass(frozen=True)
class InlineMetadata:
    dependencies: list[str]

def parse_inline_metadata(source: str) -> InlineMetadata | None:
    # Scan for # /// script ... # /// block
    # Extract dependencies = [...] line
    # Return InlineMetadata or None
```

## Files to modify

### `agent_plane/tools/local.py`

**`LocalPythonTool.__init__`** — stores `module_path: Path`, `workdir: Path`,
`sandbox_config`, `srt_available`, `uv_available` instead of the module.

**`LocalPythonTool.invoke()`** — subprocess execution via fd 3 protocol.

**`LocalPythonTool.cancel()`** — `self._proc.kill()`.

**`_build_command()`** — constructs command list based on tier.

**`load_local_python_tools()`** — gains sandbox/capability params. Scans for
PEP 723 metadata. Still loads module for SCHEMA validation, but doesn't
store it for execution.

### `agent_plane/tools/base.py`

Add `cancel()` default no-op to `Tool` ABC.

### `agent_plane/tools/manager.py`

Detect `srt` and `uv` at init via `shutil.which()`. Pass to
`load_local_python_tools()`. Add `get_tool(name) -> Tool | None`.

### `agent_plane/runtime/tool_retry.py`

`call_tool_with_timeout()` gains `tool: Tool | None = None`. Calls
`tool.cancel()` on timeout.

`execute_tool_with_retry()` gains `tool: Tool | None = None`, threads
it through.

### `agent_plane/runtime/workflow.py`

`_call_tool` step's `_blocking_call()` gets the tool instance via
`mgr.get_tool(tool_name)` and passes it to
`execute_tool_with_retry(tool=tool)`.

### `agent_plane/spec/types.py`

Add to `LocalToolInfo`: `has_inline_deps: bool`, `inline_deps: list[str] | None`.

Add `SandboxConfig` dataclass. Add `sandbox: SandboxConfig` to `ToolsConfig`.

### `agent_plane/spec/parser.py`

Add `_parse_sandbox_config()` for the `tools.sandbox` block.

---

## Config surface

```yaml
tools:
  timeout: 60
  sandbox: true                    # shorthand: enable srt when available
  # OR:
  sandbox:
    enabled: true                  # use srt if on PATH (default)
    docker_image: python:3.12-slim # Docker mode (overrides srt)
```

---

## Storage layout

Each (conversation, agent) pair gets a persistent directory:

```
storage_dir/                          ← per (conversation, agent)
  managed/                            ← agent-plane internal state
    .claude/                          ← Claude SDK session transcripts
    .claude/skills/                   ← skill files for SDK discovery
  workspace/                          ← user-visible files
    sales.csv
    output/chart.png
```

- **`managed/`** — internal state, invisible to filesystem tools and
  local tool subprocesses. The Claude SDK's `cwd` is set here so
  `.claude/` lands naturally. Agent-plane metadata lives here too.

- **`workspace/`** — user-visible filesystem. Filesystem tools resolve
  paths against this directory. Local tool subprocesses can access it
  via srt's allow-list. Not a cwd — a mounted path.

The whole `storage_dir/` is snapshotted to the artifact store for
durability across server restarts.

### How each tool type uses the layout

| Tool type | Execution | cwd | Filesystem access |
|-----------|-----------|-----|-------------------|
| Filesystem builtins | In-process, path-validated | N/A | Resolves paths against `workspace/` via `ToolContext.workspace` |
| Local Python tools | Sandboxed subprocess | `managed/` | `workspace/` mounted read-write by srt; `managed/` is cwd but srt denies writes outside `workspace/` |
| Claude SDK tools | SDK subprocess | `managed/` | SDK operates in `managed/`; separate from user files |

### Why cwd = managed/

The subprocess cwd is `managed/`, not `workspace/`. This prevents
local tools from accidentally reading/writing internal state via
relative paths. The tool accesses `workspace/` only through explicit
paths passed in its arguments (e.g. the agent passes
`{"file": "/workspace/sales.csv"}`), or through the srt mount.

With srt active, the sandbox mounts `workspace/` at a known path
inside the sandbox and restricts writes to it. Without srt, the
subprocess can technically reach anywhere — but cwd being `managed/`
still provides defense-in-depth against accidental relative-path
access to wrong directories.

### srt sandbox confinement

```bash
srt --allow-write {storage_dir}/workspace -- python _runner.py
```

On macOS: Seatbelt profile allows writes to `workspace/` only.
On Linux: bubblewrap bind-mounts `workspace/` as read-write within
a read-only root. `managed/` is not writable from the sandboxed
process.

### ToolContext gains workspace

`ToolContext` gets a `workspace: Path` field pointing to
`storage_dir/workspace/`. The `code_sandbox` builtin uses it as
`cwd`. The `upload_file` builtin resolves paths against it.
`LocalPythonTool` receives it at init for srt allowed paths.

See `designs/FILESYSTEM_TOOLS.md` for the `code_sandbox` and
`upload_file` builtin design.

---

## What does NOT change

- `Tool.invoke(arguments, ctx) -> str` signature
- `ToolManager.call_tool()` dispatch
- `execute_tool_with_retry` retry logic and SSE events
- MCP tools, builtins, client-side tools — unaffected
- SCHEMA validation, name consistency checks (already fixed)
- The async workflow and parallel tool execution via `asyncio_wait`

---

## Implementation sequence

**Phase A** (subprocess foundation):
1. `_runner.py` with fd 3 protocol
2. `cancel()` on Tool ABC
3. Rewrite `LocalPythonTool.invoke()` for subprocess
4. `call_tool_with_timeout` gains `tool.cancel()`
5. `get_tool()` on ToolManager, thread through workflow

**Phase B** (PEP 723):
6. `_pep723.py` parser
7. `has_inline_deps`/`inline_deps` on LocalToolInfo
8. `uv run --with` command construction

**Phase C** (sandboxing):
9. `SandboxConfig` type + parser
10. srt detection in ToolManager
11. srt command prefix in `_build_command`

**Phase D** (Docker):
12. `_build_docker_command` + `_AP_RESPONSE_MODE=stdout` fallback
13. Docker command construction

---

## Verification

1. **Unit tests** (`tests/tools/test_runner.py`): invoke `_runner.py` directly
   via subprocess, verify fd 3 JSON for success, import error, runtime error,
   async run, non-string coercion.

2. **Unit tests** (`tests/tools/test_pep723.py`): parse inline metadata from
   source strings with/without deps.

3. **Updated tests** (`tests/tools/test_local.py`): test subprocess invocation,
   cancel/kill on timeout, srt/uv/docker command construction (mock Popen).

4. **Updated tests** (`tests/runtime/test_tool_retry.py`): verify cancel()
   called on timeout, not called on success.

5. **Integration test**: Create a tool file that does `os._exit(1)`, invoke via
   the server, verify the server survives and returns an error response.

6. **Manual smoke test**: `python examples/frontends/terminal.py` with an agent
   that has a `tools/python/` tool, verify tool execution works end-to-end.
