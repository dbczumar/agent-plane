# Code-Based Tools (Local Tools)

## Context

The spec layer already supports local tools: `_discover_local_tools()` globs
`tools/python/*.py` and `tools/typescript/*.ts`, the parser produces
`LocalToolInfo` entries on `AgentSpec`, and `ToolManager._register_local_tools()`
adds them to the tool registry. Python tools are loaded and executable
end-to-end. What's missing is correctness and fault isolation — naming is
broken by design, schema validation is absent, TypeScript is silently ignored,
per-tool config is parsed but not wired, tools run in-process with no crash
isolation, and the spec documentation doesn't match the implementation.

The goal is a correct, documented, and fault-isolated local tool system for
Python in v1, with TypeScript explicitly deferred.

### What exists

| Component | File | Status |
|-----------|------|--------|
| `_discover_local_tools()` | `spec/parser.py:518` | Globs `.py` and `.ts` files, derives names |
| `LocalToolInfo` dataclass | `spec/types.py:268` | `name`, `path`, `language`, `timeout`, `retry` |
| `_validate_local_tools()` | `spec/validator.py` | Checks duplicate names across MCP + local |
| `load_local_python_tools()` | `tools/local.py:86` | Imports `.py` files, validates `SCHEMA` + `run` |
| `LocalPythonTool(Tool)` | `tools/local.py:31` | Wraps module with `invoke()` → `run()` dispatch |
| `ToolManager._register_local_tools()` | `tools/manager.py:130` | Adds to `_tools` registry with name validation |
| `execute_tool_with_retry()` | `runtime/tool_retry.py:87` | ThreadPoolExecutor timeout + retry loop |
| `_call_tool()` `@step` | `runtime/workflow.py:510` | DBOS-checkpointed tool dispatch |
| `resolve_tool_timeout()` / `resolve_tool_retry()` | `runtime/tool_retry.py:22` | Per-tool → global fallback resolution |

### What's missing

1. **Naming is broken** — underscore-to-dot derivation produces names that
   fail the OpenAI tool name regex (`^[a-zA-Z0-9_-]{1,64}$`). Every local
   tool with an underscore in its filename is silently dropped by
   `ToolManager._register_local_tools()`.
2. **No SCHEMA validation** — `_validate_module()` checks `hasattr(module,
   "SCHEMA")` but never validates structure (type, function, function.name).
3. **No name consistency check** — `SCHEMA.function.name` can differ from
   the filename-derived name. The LLM calls the schema name; dispatch uses
   the filename name. Silent failure.
4. **Per-tool timeout/retry not wired** — `LocalToolInfo.timeout` and
   `.retry` are parsed but `_execute_tools()` always uses the global
   `tools_config.timeout` / `tools_config.retry`.
5. **TypeScript silently ignored** — parser discovers `.ts` files, loader
   skips them without warning, no documentation of the limitation.
6. **`run()` return type not enforced** — non-string returns pass through
   silently, producing malformed `function_call_output`.
7. **`async def run()` not guarded** — returns a coroutine object as the
   tool "result" string.
8. **No fault isolation** — local tools execute in-process. A segfault, OOM,
   or `os._exit()` in a tool crashes the entire server, killing all
   concurrent workflows.
9. **AGENTSPEC.md inaccurate** — says "Schema is inferred from type hints
   and docstrings." The actual contract is explicit `SCHEMA` + `run()`.
10. **No authoring documentation** — no guide for tool authors.

---

## Design Decisions

### Tool naming: filename stem, no transformation

The current derivation replaces underscores with dots:
`arxiv_search.py` → `arxiv.search`. But `is_valid_tool_name("arxiv.search")`
returns `False` (dots are not in `[a-zA-Z0-9_-]`), so
`ToolManager._register_local_tools()` silently skips the tool. Any tool whose
filename contains an underscore is broken.

**Decision**: Tool name = filename stem. No transformation.
`arxiv_search.py` → `arxiv_search`. This is always valid, always predictable,
and matches how skill names use the directory name as truth.

**Migration**: Since the current behavior silently drops underscore-containing
tools, there are no working tools that depend on the dot-derived name. This is
a pure fix.

### Filename is the source of truth for tool names

The tool name is derived from the filename; `SCHEMA.function.name` must match.
If they differ, the LLM calls the schema name but dispatch looks up the
filename name — a silent failure. The loader must reject mismatches.

**Alternative considered**: Use `SCHEMA.function.name` as the source of truth.
This gives authors more control but breaks the "filesystem = truth" principle
used everywhere else in the spec (skill names = directory names, sub-agent
names = directory names). Consistency wins.

### Subprocess execution for fault isolation

The current implementation runs local tools in-process via
`ThreadPoolExecutor`. This is a fault isolation problem — even trusted code
has bugs. A segfault in a C extension, an accidental `os._exit(1)`, an OOM
from a runaway allocation, or an infinite loop that can't be interrupted — any
of these crashes the entire agent-plane server, killing all concurrent agent
executions.

**The in-process model's failure modes:**

| Failure | In-process impact | Recoverable? |
|---------|-------------------|--------------|
| Unhandled Python exception | Caught by `execute_tool_with_retry()` → error string to LLM | Yes |
| Segfault in C extension | Server process crashes | No — all concurrent workflows die. DBOS recovers on restart, but in-flight requests are lost. |
| `os._exit()` / `sys.exit()` | Server process exits | No — same as segfault |
| OOM (large allocation) | OS OOM-killer targets server process | No — same as segfault |
| Infinite loop (CPU-bound) | ThreadPoolExecutor timeout fires, but the thread keeps running. Python threads cannot be killed. | Partially — workflow continues, but the orphaned thread leaks CPU and memory until the process restarts. |
| `time.sleep(forever)` | ThreadPoolExecutor timeout fires, thread stays blocked | Partially — same as infinite loop |

The first row is fine. Every other row is a server-wide outage caused by a
single tool call. DBOS can recover workflows on restart, but in-flight
streaming responses are severed, and the restart window is a hard outage.

**Decision: subprocess execution.** Each local tool call runs in a separate
child process. The parent sends the request via stdin and reads the response
via **fd 3** (a dedicated pipe). The child's stdout/stderr remain normal —
tool authors can `print()` freely for debugging without corrupting the
protocol. The child's crash, OOM, or timeout is contained — the parent
process (and all other workflows) is unaffected.

```
LLM response
  → workflow._execute_tools()
    → workflow._call_tool() [@step — DBOS checkpoint]
      → tool_retry.execute_tool_with_retry()
        → call_tool_with_timeout(fn, timeout, tool)
          → ThreadPoolExecutor runs fn():
            → LocalPythonTool.invoke()
              → os.pipe() → (read_fd, write_fd)
              → subprocess.Popen(runner script, pass_fds=(write_fd,))
                → stdin: {"module_path": "...", "arguments": {...}}
                → child: import module, call run(), write result to fd 3
                → fd 3: {"result": "..."}
              → os.close(write_fd)  # parent closes write end
              → proc.communicate()  # no timeout — caller owns deadline
              → os.read(read_fd) → response JSON
              → on crash: non-zero exit code → RuntimeError
              → on success: parse fd 3 JSON → return result string
          → on timeout: tool.cancel() → proc.kill() (SIGKILL)
            → thread unblocks from proc.communicate(), future completes
```

**What subprocess execution provides:**

| Failure | Subprocess impact | Server impact |
|---------|-------------------|---------------|
| Unhandled Python exception | Child writes error JSON to fd 3, exits zero | Parent reads error from fd 3, returns error string to LLM |
| Segfault in C extension | Child process dies (signal 11) | Parent sees non-zero exit, returns error string. Server unaffected. |
| `os._exit()` | Child exits | Parent sees non-zero exit. Server unaffected. |
| OOM | OS OOM-killer targets the child process (smaller RSS) | Parent sees SIGKILL exit. Server unaffected. |
| Infinite loop | `call_tool_with_timeout()` deadline fires → `tool.cancel()` → `proc.kill()` (SIGKILL) | Hard kill. Thread unblocks. Server unaffected. |
| `time.sleep(forever)` | Same as infinite loop — SIGKILL via `cancel()` | Server unaffected. |

**Cost**: subprocess spawn overhead is ~50–100ms on Linux/macOS (fork + exec +
Python interpreter startup). For tool calls that typically take 100ms–minutes
(HTTP requests, file I/O, computation), this is negligible. The tradeoff —
50ms latency vs. server-wide crash risk — is overwhelmingly in favor of
subprocess isolation.

### Runner script: `tools/_runner.py`

A minimal script that the child process executes. Communication uses
**fd 3** (a pipe opened by the parent) for the JSON response, leaving
stdout/stderr free for tool authors to use `print()` for debugging.

```python
"""Subprocess runner for local Python tools.

Reads a JSON request from stdin, imports the tool module, calls
run(), and writes the JSON result to fd 3. Designed to be invoked
as: python -m agent_plane.tools._runner

Uses fd 3 (not stdout) for the response so that print() calls in
tool code don't corrupt the protocol.
"""

import importlib.util
import json
import os
import sys
import traceback

_RESPONSE_FD = 3  # parent opens a pipe and passes the write end as fd 3


def main() -> None:
    """
    Entry point for the tool runner subprocess.

    Reads ``{"module_path": str, "arguments": dict}`` from stdin.
    Writes ``{"result": str}`` or ``{"error": str}`` to fd 3.
    """
    request = json.loads(sys.stdin.buffer.read())
    module_path: str = request["module_path"]
    arguments: dict = request["arguments"]

    module = _load_module(module_path)
    if module is None:
        return

    try:
        result = module.run(arguments)
    except Exception as exc:
        traceback.print_exc()  # full traceback → stderr → captured by parent
        _write_error(f"{type(exc).__name__}: {exc}")
        return

    if not isinstance(result, str):
        result = str(result)

    _write_response({"result": result})


def _load_module(path):
    """Import a tool module from an absolute file path."""
    spec = importlib.util.spec_from_file_location("_tool", path)
    if spec is None or spec.loader is None:
        _write_error(f"Cannot load module from {path}")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        traceback.print_exc()  # full traceback → stderr → captured by parent
        _write_error(f"Import error: {type(exc).__name__}: {exc}")
        return None
    return module


def _write_response(data: dict) -> None:
    """Write a JSON response to fd 3 and close it."""
    os.write(_RESPONSE_FD, json.dumps(data).encode())
    os.close(_RESPONSE_FD)


def _write_error(message: str) -> None:
    """Write an error response to fd 3 and close it."""
    _write_response({"error": message})


if __name__ == "__main__":
    main()
```

### Parent-side invocation: `LocalPythonTool.invoke()`

```python
import os
import subprocess
import sys

def invoke(self, arguments: str) -> str:
    """
    Execute the tool in a subprocess for fault isolation.

    Opens a pipe (fd 3) for the response, spawns a child process
    running ``_runner.py``, passes the module path and parsed
    arguments via stdin JSON, reads the result from fd 3 JSON.

    Timeout is NOT enforced here — the caller
    (``call_tool_with_timeout``) owns the deadline and calls
    ``cancel()`` to kill the subprocess on expiry.

    :param arguments: JSON-encoded arguments string from the LLM.
    :returns: The tool's string result.
    :raises RuntimeError: If the subprocess crashes, returns an
        error, or produces malformed output.
    """
    parsed = json.loads(arguments) if arguments else {}
    request = json.dumps({
        "module_path": str(self._module_path),
        "arguments": parsed,
    })
    # fd 3 pipe: child writes JSON response, parent reads it.
    # stdout/stderr remain normal — tool code can print() freely.
    read_fd, write_fd = os.pipe()
    self._proc = subprocess.Popen(
        [sys.executable, "-m", "agent_plane.tools._runner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(write_fd,),
        cwd=str(self._workdir),  # tool sees agent image as cwd
    )
    os.close(write_fd)  # parent closes write end immediately
    _stdout, stderr = self._proc.communicate(
        input=request.encode(),
    )

    if self._proc.returncode != 0:
        os.close(read_fd)
        raise RuntimeError(
            f"Tool {self.name!r} process exited with code "
            f"{self._proc.returncode}: "
            f"{stderr.decode(errors='replace')[:500]}"
        )

    # Read response from fd 3. Loop until EOF (child closes fd 3
    # after writing). Cap at 1 MiB to prevent runaway memory usage.
    _MAX_RESPONSE = 1_048_576  # 1 MiB
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_RESPONSE:
            os.close(read_fd)
            raise RuntimeError(
                f"Tool {self.name!r}: response exceeded "
                f"1 MiB limit"
            )
        chunks.append(chunk)
    os.close(read_fd)
    raw = b"".join(chunks)
    try:
        response = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Tool {self.name!r}: malformed response from "
            f"runner: {exc}"
        ) from None

    if "error" in response:
        raise RuntimeError(
            f"Tool {self.name!r} error: {response['error']}"
        )
    return response["result"]

def cancel(self) -> None:
    """
    Kill the subprocess. Called by ``call_tool_with_timeout()``
    when the deadline expires.

    Guards against ``ProcessLookupError`` — if the timeout fires
    at the exact moment ``proc.communicate()`` returns, the
    process is already reaped and ``kill()`` would raise.
    """
    if self._proc is not None:
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass  # already exited
```

**Key details:**

- **fd 3 protocol** — the parent opens `os.pipe()`, passes the write end as
  fd 3 to the child via `pass_fds`. The child writes JSON to fd 3; the parent
  reads from the read end after `proc.communicate()` completes. stdout/stderr
  remain normal, so `print()` in tool code doesn't corrupt the protocol.
- **No timeout in `invoke()`** — the caller (`call_tool_with_timeout`)
  owns the deadline. On expiry, it calls `cancel()` → `proc.kill()` (SIGKILL).
  The thread waiting in `proc.communicate()` unblocks (child is dead), the
  future completes, and the retry loop handles the rest.
- **`cancel()` sends SIGKILL** — the child is hard-killed. No orphaned
  threads, no leaked resources. This is the critical difference from
  in-process tools, where Python threads cannot be killed. Guards
  `ProcessLookupError` for the race where the process exits between the
  timeout firing and `kill()` being called.
- **`self._proc` thread-safety** — `invoke()` sets `self._proc` on the
  ThreadPoolExecutor thread; `cancel()` reads it from the timeout thread.
  There is a theoretical window between `pool.submit()` and the `Popen()`
  assignment where `cancel()` would no-op. This is accepted because: the
  timeout is always orders of magnitude larger than the time to reach
  `Popen()`, and the worst case (cancel no-ops, timeout raises, thread
  leaks) is identical to the current in-process behavior — no regression.
- **`cwd=self._workdir`** — the subprocess runs with the agent image
  directory as its working directory. Tool authors can use relative paths.
- **Two error paths**: (1) non-zero exit = process crash (segfault, OOM,
  `os._exit`) — stderr is captured for diagnostics. (2) zero exit +
  `{"error": "..."}` on fd 3 = Python exception in `run()` — structured
  error from the runner. Full tracebacks are written to stderr by the
  runner (`traceback.print_exc()`), captured by the parent, and available
  for logging.
- **1 MiB response cap** — the parent reads fd 3 in a loop until EOF, with
  a 1 MiB hard cap. If a tool writes a larger response, `invoke()` raises
  `RuntimeError("response exceeded 1 MiB limit")` — an actionable error
  instead of a confusing `JSONDecodeError` from truncated JSON.
- **Malformed output guard** — `json.loads(raw)` is wrapped in try/except.
  If the runner crashes mid-write or fd 3 is empty, we get a clear
  `RuntimeError` instead of a confusing `JSONDecodeError`.
- **Module path, not module object** — the subprocess can't receive a Python
  module object. `LocalPythonTool` stores the absolute file path (from
  `workdir / info.path`) and passes it to the runner.

### Changes to `LocalPythonTool` construction

Since execution is now subprocess-based, the tool no longer needs to import
the module at registration time. However, **validation still happens at
load time** (not at call time) — we import the module in-process during
`load_local_python_tools()` to validate `SCHEMA`, `run()`, name consistency,
etc. The validated schema is cached on the `LocalPythonTool` instance. At
invoke time, only the subprocess runs the module.

```python
class LocalPythonTool(Tool):
    def __init__(
        self,
        info: LocalToolInfo,
        schema: dict[str, Any],
        module_path: Path,
        workdir: Path,
    ) -> None:
        """
        :param info: Discovered tool info with name, timeout, retry.
        :param schema: Validated SCHEMA dict from the module (cached
            at load time, not re-read at invoke time).
        :param module_path: Absolute path to the .py file, passed to
            the runner subprocess.
        :param workdir: Agent image directory, used as subprocess cwd
            so tool code can use relative paths.
        """
        self._info = info
        self._schema = schema
        self._module_path = module_path
        self._workdir = workdir
        self._proc: subprocess.Popen | None = None
```

### Timeout integration with subprocess execution

The existing `call_tool_with_timeout()` uses `ThreadPoolExecutor` to enforce
timeout. For in-process tools, this is the only mechanism — Python threads
can't be killed, so the timed-out thread leaks. For subprocess tools, the
caller can kill the child process from the outside, which is strictly better.

**The problem with in-process timeout for subprocesses:**

`ThreadPoolExecutor` timeout cancels the *future*, not the *thread*. The
thread sitting in `proc.communicate()` keeps running, and the subprocess
stays alive. Two rejected approaches:

- **Option A — Add `timeout: int` to `Tool.invoke()` ABC:** Pushes timeout
  into every tool implementation (7+ classes). Most ignore it. Only
  `LocalPythonTool` uses it for `proc.communicate(timeout=N)`.
- **Option B — `set_effective_timeout()` before `invoke()`:** Temporal
  coupling — must call set before invoke, or `proc.communicate(timeout=None)`
  blocks forever.

**Decision: external kill via `cancel()`.** The caller owns timeout
enforcement. `LocalPythonTool` exposes `cancel()` which kills the
subprocess. `call_tool_with_timeout()` calls it when the deadline expires.

```python
class Tool(abc.ABC):
    def cancel(self) -> None:
        """
        Cancel an in-progress invocation.

        Called by ``call_tool_with_timeout()`` when the deadline
        expires. Default is a no-op (in-process tools can't be
        cancelled — the thread leaks, same as today).
        Subprocess-based tools override to kill the child process.
        """
        pass  # default: no-op for in-process tools
```

```python
class LocalPythonTool(Tool):
    def __init__(self, ...):
        ...
        self._proc: subprocess.Popen | None = None

    def invoke(self, arguments: str) -> str:
        ...
        self._proc = subprocess.Popen(...)
        _stdout, stderr = self._proc.communicate()  # no timeout
        ...

    def cancel(self) -> None:
        """Kill the subprocess. Called by timeout enforcement."""
        if self._proc is not None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass  # already exited
```

```python
# call_tool_with_timeout() — updated
def call_tool_with_timeout(
    call_fn: Callable[[], str],
    timeout: int,
    tool: Tool,
) -> str:
    """
    Execute a tool call with a wall-clock timeout.

    Uses a thread pool to enforce the timeout. On expiry, calls
    ``tool.cancel()`` to kill any subprocess, then raises
    ``TimeoutError``.

    :param call_fn: Zero-argument callable that executes the tool.
    :param timeout: Timeout in seconds, e.g. ``60``.
    :param tool: The tool instance, for ``cancel()`` on timeout.
    :returns: The tool's string result.
    :raises TimeoutError: If the tool does not complete in time.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call_fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            tool.cancel()
            raise TimeoutError(
                f"Tool execution timed out after {timeout}s"
            ) from None
```

**Why this works:**

- No ABC signature change — `invoke()` stays `invoke(self, arguments: str) -> str`.
- No temporal coupling — `cancel()` is called *by the timeout path*, not
  pre-set by the caller.
- `cancel()` is a no-op for in-process tools (same leak as today — no
  regression). For subprocess tools, `proc.kill()` sends SIGKILL, the
  thread in `proc.communicate()` unblocks, the future completes.
- The `tool` parameter on `call_tool_with_timeout()` is the only call-site
  change. `_call_tool()` already has access to the `ToolManager`.

### Security sandboxing (not in v1)

Subprocess execution provides **fault isolation** — a crashed tool doesn't
crash the server. It does NOT provide **security sandboxing** — the child
process still has the same filesystem, network, and privilege access as the
parent.

Security sandboxing is only needed if agent-plane hosts untrusted third-party
agent images. At that point, additional measures are needed on top of
subprocess execution:

- **Filesystem sandboxing** — mount only the agent's `workdir`, read-only.
  Tempdir for scratch.
- **Resource limits** — cgroups or `ulimit` for memory, CPU, file
  descriptors.
- **Network isolation** — restrict outbound to declared MCP server URLs.
- **Import isolation** — clean Python environment with only declared
  dependencies.

This is out of scope for v1. The trust boundary is the agent image, and
the operator controls which images are deployed.

### Per-tool timeout/retry: properties on the Tool ABC

`LocalToolInfo.timeout` and `.retry` are parsed but `_execute_tools()` always
uses `tools_config.timeout` / `tools_config.retry`. The per-tool values are
ignored.

**Option A — Add `timeout`/`retry` properties to `Tool` ABC:**

```python
class Tool(abc.ABC):
    @property
    def timeout(self) -> int | None:
        """Per-tool timeout override. None = inherit global."""
        return None

    @property
    def retry(self) -> RetryConfig | None:
        """Per-tool retry override. None = inherit global."""
        return None
```

`LocalPythonTool` and `McpTool` override these from their config. Builtins
and client-side tools return `None` (inherit global).

**Option B — Lookup method on `ToolManager`:**

```python
def get_tool_timeout(self, name: str) -> int | None: ...
def get_tool_retry(self, name: str) -> RetryConfig | None: ...
```

**Decision: Option A.** The tool already knows its own config. Adding
default-returning properties to the ABC is minimal churn and keeps the
knowledge co-located.

### Per-tool config surface: `config.yaml`, not sidecars

Per-tool timeout/retry for MCP servers lives in `tools/mcp/<name>.yaml`.
Local tools need an analogous surface for overrides.

**Decision**: A `tools.local` block in `config.yaml`:

```yaml
tools:
  timeout: 60          # global default
  retry:
    max_attempts: 2

  local:               # per-tool overrides for local tools
    arxiv_search:
      timeout: 120
      retry:
        max_attempts: 3
```

**Alternative considered**: Per-file YAML sidecar (`arxiv_search.yaml`
alongside `arxiv_search.py`). More self-contained but adds filesystem
complexity and a new parsing path. `config.yaml` is already the single source
of config for everything else — consistency wins.

### TypeScript: defer explicitly, not silently

TypeScript execution requires a Node.js subprocess, `package.json` support,
and a TypeScript ↔ Python bridge. Substantial work, not needed for v1.

**Decision**: Keep parser discovery (existing agent images won't need
re-packaging when support lands), but emit a validation warning when
TypeScript tools are found. Move TypeScript to the "Not Yet" section in
AGENTSPEC.md.

### SCHEMA validation: structural, not deep

Validate the shape of `SCHEMA` at load time — must be a dict with
`type: "function"` and a `function` sub-dict containing a `name` string.
Do not validate `parameters` against JSON Schema spec (would require a JSON
Schema validator dependency). Do not validate `description` length or content.

The OpenAI API will reject truly malformed schemas at inference time. The
loader's job is to catch obvious authoring errors early (missing `function`
key, wrong `type`, no `name`) so the agent author gets feedback at deploy
time rather than at inference time.

### `run()` return type: coerce, don't crash

If `run()` returns a non-string, coerce with `str()` and log a warning.
The alternative (raising) would cause the tool call to fail, which is worse
for the user than a coerced result. The warning tells the author to fix it.

---

## File Details

### 1. `tools/base.py` — Add `timeout`/`retry` properties to Tool ABC

```python
class Tool(abc.ABC):
    # ... existing abstract methods ...

    @property
    def timeout(self) -> int | None:
        """
        Per-tool timeout override in seconds.

        ``None`` means inherit the global ``tools.timeout`` from the
        agent spec. Subclasses with per-tool config (MCP, local)
        override this to return their configured value.

        :returns: Timeout in seconds, or ``None`` to inherit global.
        """
        return None

    @property
    def retry(self) -> RetryConfig | None:
        """
        Per-tool retry policy override.

        ``None`` means inherit the global ``tools.retry`` from the
        agent spec. Subclasses with per-tool config (MCP, local)
        override this to return their configured value.

        :returns: Retry config, or ``None`` to inherit global.
        """
        return None
```

### 2. `tools/local.py` — Validation and invoke hardening

**`_validate_schema()`** — new function:

```python
def _validate_schema(tool_name: str, schema: dict[str, Any]) -> bool:
    """
    Validate that SCHEMA has the required OpenAI function structure.

    Checks for ``type: "function"`` and a ``function`` sub-dict with
    a ``name`` string. Does not validate ``parameters`` deeply — the
    LLM provider will reject truly malformed schemas at inference time.

    :param tool_name: Tool name for warning messages,
        e.g. ``"arxiv_search"``.
    :param schema: The module's ``SCHEMA`` attribute.
    :returns: ``True`` if the schema is structurally valid.
    """
    if not isinstance(schema, dict):
        _logger.warning(
            "Local tool %r: SCHEMA is not a dict — skipping",
            tool_name,
        )
        return False
    if schema.get("type") != "function":
        _logger.warning(
            "Local tool %r: SCHEMA.type must be 'function' — skipping",
            tool_name,
        )
        return False
    func = schema.get("function")
    if not isinstance(func, dict):
        _logger.warning(
            "Local tool %r: SCHEMA.function missing or not a dict — skipping",
            tool_name,
        )
        return False
    if not func.get("name"):
        _logger.warning(
            "Local tool %r: SCHEMA.function.name missing — skipping",
            tool_name,
        )
        return False
    return True
```

**`_validate_module()`** — extended:

```python
def _validate_module(
    tool_name: str,
    module: ModuleType,
    expected_name: str,
) -> bool:
    """
    Validate that a loaded module has the required exports.

    Checks: SCHEMA exists, SCHEMA structure is valid,
    SCHEMA.function.name matches the filename-derived name,
    run() exists, run() is callable, run() is not async.

    :param tool_name: Tool name for warning messages.
    :param module: The loaded Python module.
    :param expected_name: The filename-derived tool name that
        SCHEMA.function.name must match, e.g. ``"arxiv_search"``.
    :returns: ``True`` if the module is valid for use as a tool.
    """
    if not hasattr(module, "SCHEMA"):
        _logger.warning(
            "Local tool %r: module missing SCHEMA — skipping",
            tool_name,
        )
        return False
    if not _validate_schema(tool_name, module.SCHEMA):
        return False
    schema_name = module.SCHEMA["function"]["name"]
    if schema_name != expected_name:
        _logger.warning(
            "Local tool %r: SCHEMA.function.name is %r but "
            "filename-derived name is %r — these must match. Skipping.",
            tool_name,
            schema_name,
            expected_name,
        )
        return False
    if not hasattr(module, "run"):
        _logger.warning(
            "Local tool %r: module missing run() function — skipping",
            tool_name,
        )
        return False
    if not callable(module.run):
        _logger.warning(
            "Local tool %r: run is not callable — skipping",
            tool_name,
        )
        return False
    if asyncio.iscoroutinefunction(module.run):
        _logger.warning(
            "Local tool %r: run() is async — only synchronous "
            "run() is supported. Skipping.",
            tool_name,
        )
        return False
    return True
```

**`LocalPythonTool`** — subprocess-based execution:

The class changes substantially. It no longer holds a module reference —
validation happens at load time (in-process), but execution happens in a
subprocess. The validated `SCHEMA` dict and the absolute file path are
cached.

Full pseudocode is in the "Parent-side invocation" design decision above.
The File Details version matches it exactly — `invoke()` spawns the
subprocess with fd 3 pipe, `cancel()` kills it. No timeout in `invoke()`;
the caller owns the deadline.

**`load_local_python_tools()`** — updated construction:

```python
# After validation:
tools.append(LocalPythonTool(
    info=info,
    schema=module.SCHEMA,        # cached from validation
    module_path=tool_path,       # absolute path for subprocess
    workdir=workdir,             # agent image dir for subprocess cwd
))
```

### 3. `tools/_runner.py` — NEW — Subprocess entry point

The runner script that child processes execute. Reads a JSON request from
stdin, imports the tool module, calls `run()`, writes JSON result to **fd 3**
(a pipe opened by the parent). stdout/stderr remain normal — tool authors
can `print()` freely. Designed to be invoked as
`python -m agent_plane.tools._runner`.

Full pseudocode is in the "Runner script" design decision above. The File
Details version matches it exactly — one `main()`, one `_load_module()`,
one `_write_response()`, one `_write_error()`. All output goes to fd 3 via
`os.write(_RESPONSE_FD, ...)` + `os.close(_RESPONSE_FD)`.

### 4. `spec/parser.py` — Fix naming, add `tools.local` parsing

**`_discover_local_tools()`** — remove underscore-to-dot:

```python
# Before:
tool_name = stem.replace("_", ".")

# After:
tool_name = stem
```

**`_parse_tools_local_overrides()`** — new function. Called from
`_parse_tools()` after `_discover_local_tools()` returns and before
`AgentSpec` construction, when a `tools.local` key exists in the parsed
config dict:

```python
def _parse_tools_local_overrides(
    tools_local: dict[str, Any],
    local_tools: list[LocalToolInfo],
) -> None:
    """
    Apply per-tool timeout/retry overrides from ``tools.local``
    in config.yaml to the discovered ``LocalToolInfo`` entries.

    Modifies ``local_tools`` in place. Warns on overrides that
    don't match any discovered tool.

    Called from ``_parse_tools()`` after ``_discover_local_tools()``
    and before ``AgentSpec`` construction.

    :param tools_local: The ``tools.local`` dict from config.yaml,
        mapping tool name to override config, e.g.
        ``{"arxiv_search": {"timeout": 120}}``.
    :param local_tools: The discovered local tool entries to update.
    """
    by_name = {t.name: t for t in local_tools}
    for name, overrides in tools_local.items():
        tool = by_name.get(name)
        if tool is None:
            _logger.warning(
                "tools.local override for %r does not match any "
                "discovered tool — ignoring",
                name,
            )
            continue
        if "timeout" in overrides:
            tool.timeout = overrides["timeout"]
        if "retry" in overrides:
            tool.retry = _parse_retry_config(overrides["retry"])
```

### 5. `tools/mcp.py` — Add `timeout`/`retry` properties to `McpTool`

```python
@property
def timeout(self) -> int | None:
    """
    Per-tool timeout from the MCP server config.

    :returns: Timeout in seconds, or ``None`` to inherit global.
    """
    return self._connection.config.timeout

@property
def retry(self) -> RetryConfig | None:
    """
    Per-tool retry policy from the MCP server config.

    :returns: Retry config, or ``None`` to inherit global.
    """
    return self._connection.config.retry
```

### 6. `runtime/workflow.py` — Wire per-tool resolution

**`_execute_tools()`** — use per-tool config:

```python
# Before:
result = _call_tool(
    task_id,
    tc.name,
    tc.arguments,
    tools_config.timeout,
    tools_config.retry,
)

# After:
mgr = get_tool_manager()
tool = mgr.get_tool(tc.name)
effective_timeout = resolve_tool_timeout(
    tc.name, tools_config,
    tool.timeout if tool is not None else None,
)
effective_retry = resolve_tool_retry(
    tc.name, tools_config,
    tool.retry if tool is not None else None,
)
result = _call_tool(
    task_id,
    tc.name,
    tc.arguments,
    effective_timeout,
    effective_retry,
)
```

This requires two supporting changes:

**`ToolManager.get_tool()`** — new method:

```python
def get_tool(self, name: str) -> Tool | None:
    """
    Look up a registered tool by name.

    :param name: The tool function name, e.g. ``"arxiv_search"``.
    :returns: The tool instance, or ``None`` if not registered.
    """
    return self._tools.get(name)
```

**`_call_tool()`** — pass the tool instance to `call_tool_with_timeout` so
it can call `tool.cancel()` on timeout:

```python
# Before:
call_fn=lambda: mgr.call_tool(tool_name, arguments),

# After:
tool = mgr.get_tool(tool_name)
call_fn=lambda: mgr.call_tool(tool_name, arguments),
# ... pass tool to execute_tool_with_retry → call_tool_with_timeout
```

The full signature change to `call_tool_with_timeout()` and
`execute_tool_with_retry()` (adding `tool: Tool` parameter) is shown in
the "Timeout integration" design decision above.

### 7. `spec/validator.py` — TypeScript warning

```python
def _warn_typescript_tools(
    local_tools: list[LocalToolInfo],
    warnings: list[str],
) -> None:
    """
    Emit a warning if any TypeScript tools are discovered.

    :param local_tools: Discovered local tool entries.
    :param warnings: Mutable list to append warnings to.
    """
    ts_names = [t.name for t in local_tools if t.language == "typescript"]
    if ts_names:
        warnings.append(
            f"TypeScript tools are not yet supported and will be "
            f"skipped: {', '.join(ts_names)}"
        )
```

### 8. `spec/AGENTSPEC.md` and `spec/types.py` — Correct documentation

Update `LocalToolInfo` docstring and field comment in `types.py` to use
`"arxiv_search"` (underscore) instead of `"arxiv.search"` (dot).

Replace AGENTSPEC.md lines 249–258 with:

```markdown
## Local Tools — `tools/python/*.py`

Python files under `tools/python/` are auto-discovered. The tool name is the
filename stem: `arxiv_search.py` → `arxiv_search`.

Each file must export:

- `SCHEMA`: A dict in OpenAI function-calling format:
  ```python
  SCHEMA = {
      "type": "function",
      "function": {
          "name": "arxiv_search",          # must match filename stem
          "description": "Search arXiv.",
          "parameters": {
              "type": "object",
              "properties": {
                  "query": {"type": "string"},
              },
              "required": ["query"],
          },
      },
  }
  ```
- `run(arguments: dict) -> str`: A synchronous function that receives parsed
  arguments (dict) and returns a string result.

**Constraints:**
- `SCHEMA.function.name` must match the filename-derived tool name.
- `run()` must be synchronous (not `async def`).
- `run()` must return a string. Non-string returns are coerced with `str()`.
- Tool response size must not exceed 1 MiB. Larger responses are rejected.
- Tool names must match `^[a-zA-Z0-9_-]{1,64}$` (OpenAI function-calling
  constraint).
- `print()` is safe — stdout/stderr are separate from the response channel.

**Per-tool config:**

```yaml
tools:
  timeout: 60
  local:
    arxiv_search:
      timeout: 120
      retry:
        max_attempts: 3
```

Omitting a tool from `tools.local` inherits the global `tools.timeout` and
`tools.retry`.
```

---

## Implementation Plan

### Phase 1: Subprocess execution + validation fixes

These are a single phase because both touch `LocalPythonTool`,
`load_local_python_tools()`, and `_validate_module()` in `tools/local.py`.
Splitting them would require rewriting the same functions twice.

1. **Fix naming derivation** — remove `stem.replace("_", ".")` in
   `_discover_local_tools()`. Tool name = filename stem. Update
   `LocalToolInfo` docstring in `types.py`. Update parser tests.

2. **Add SCHEMA validation** — `_validate_schema()` in `tools/local.py`.
   Called from `_validate_module()` after `hasattr(module, "SCHEMA")`.

3. **Enforce name consistency** — `_validate_module()` checks
   `SCHEMA.function.name == expected_name`. Pass `info.name` as
   `expected_name` from `load_local_python_tools()`.

4. **Guard async run()** — `asyncio.iscoroutinefunction()` check in
   `_validate_module()`.

5. **Create `tools/_runner.py`** — subprocess entry point. Reads JSON from
   stdin, imports tool module, calls `run()`, writes JSON to fd 3. Handles
   import errors, runtime exceptions, and non-string return coercion.

6. **Rewrite `LocalPythonTool`** — store `module_path` (absolute `Path`),
   `workdir`, and validated `schema` dict instead of a module reference.
   `invoke()` spawns a subprocess via `_runner.py`, communicates via
   stdin (request) and fd 3 (response). No timeout in `invoke()` — the
   caller owns the deadline via `cancel()` → `proc.kill()`. Includes
   malformed output guard (`try/except` around `json.loads` of fd 3
   response).

7. **Update `load_local_python_tools()`** — validate module in-process
   (SCHEMA structure, name consistency, run signature), then construct
   `LocalPythonTool(info, schema, module_path, workdir)` with the file
   path and workdir, not the module.

8. **Add `cancel()` to Tool ABC** — default no-op. `LocalPythonTool`
   overrides to call `self._proc.kill()`. Update
   `call_tool_with_timeout()` to accept a `tool: Tool` parameter and
   call `tool.cancel()` on timeout expiry. Update
   `execute_tool_with_retry()` and `_call_tool()` to thread the tool
   instance through.

### Phase 2: Wire per-tool config

9. **Add `timeout`/`retry` properties to Tool ABC** — default `None`.
    Override in `LocalPythonTool` (from `LocalToolInfo`) and `McpTool`
    (from `MCPServerConfig`).

10. **Add `get_tool()` method to ToolManager** — simple
    `self._tools.get(name)` lookup.

11. **Wire per-tool resolution in `_execute_tools()`** — resolve per-tool
    timeout/retry via `resolve_tool_timeout()` / `resolve_tool_retry()`
    before calling `_call_tool()`.

12. **Add `tools.local` config block** — `_parse_tools_local_overrides()`
    in parser. Apply overrides to matching `LocalToolInfo` entries.

### Phase 3: Documentation and polish

13. **Fix AGENTSPEC.md** — replace local tools section with accurate
    contract. Update `LocalToolInfo` docstring in `types.py`. Move
    TypeScript to "Not Yet".

14. **TypeScript deferral warning** — `_warn_typescript_tools()` in
    validator.

15. **Create AUTHORING.md** — tool author guide covering: file placement,
    name derivation, SCHEMA format with example, `run()` contract,
    subprocess execution model (fd 3 protocol), timeout config, error
    handling, `print()` safety, limitations.

---

## Test Plan

Tests are organized by phase. Each phase's tests live in the existing test
file that mirrors the source module. Unit tests use function-based pytest
with fixtures (no class-based tests).

### Phase 1: Subprocess execution + validation fixes

**File**: `tests/tools/test_runner.py` (NEW)

| Test | Description |
|---|---|
| `test_runner_valid_tool` | Write a tool file that returns `"hello"`. Invoke `_runner.py` via `subprocess.run()` with fd 3 pipe. Assert fd 3 JSON has `{"result": "hello"}`. |
| `test_runner_import_error` | Write a tool file with `raise RuntimeError("broken")` at module level. Assert fd 3 JSON has `{"error": "Import error: RuntimeError: broken"}`. |
| `test_runner_runtime_error` | Write a tool file where `run()` raises `ValueError("bad")`. Assert fd 3 JSON has `{"error": "ValueError: bad"}`. |
| `test_runner_non_string_return_coerced` | Write a tool file where `run()` returns `42`. Assert result is `"42"`. |
| `test_runner_empty_arguments` | Send `{"arguments": {}}`. Assert the tool receives an empty dict. |
| `test_runner_print_does_not_corrupt_response` | Write a tool file that calls `print("debug")` inside `run()` then returns `"ok"`. Assert fd 3 JSON has `{"result": "ok"}` — stdout contains `"debug"` but does not affect the response. |

**File**: `tests/tools/test_local.py`

| Test | Description |
|---|---|
| `test_invoke_subprocess_success` | Write a valid tool file. Create `LocalPythonTool` with the file path. Call `invoke()`. Assert the result matches what `run()` returns. |
| `test_cancel_kills_subprocess` | Write a tool file with `time.sleep(60)` in `run()`. Start `invoke()` in a thread. Call `tool.cancel()` from the main thread. Assert the thread completes (subprocess killed, `proc.communicate()` unblocks). Assert `proc.returncode` is negative (SIGKILL). |
| `test_cancel_with_call_tool_with_timeout` | Write a tool file with `time.sleep(60)` in `run()`. Call `call_tool_with_timeout(fn, timeout=1, tool=tool)`. Assert `TimeoutError` raised. Assert subprocess is dead (no orphan). |
| `test_invoke_subprocess_crash` | Write a tool file with `os._exit(1)` in `run()`. Call `invoke()`. Assert `RuntimeError` is raised with non-zero exit code. Server process is unaffected. |
| `test_invoke_subprocess_segfault` | Write a tool file that triggers `ctypes` null pointer dereference. Call `invoke()`. Assert `RuntimeError` (signal 11). Server unaffected. |
| `test_invoke_subprocess_print_safe` | Write a tool file that prints to stdout inside `run()`. Call `invoke()`. Assert the result is correct — `print()` does not corrupt the fd 3 protocol. |
| `test_invoke_subprocess_malformed_response` | Write a tool file that writes garbage to fd 3 directly. Call `invoke()`. Assert `RuntimeError` with "malformed response" message. |
| `test_invoke_subprocess_cwd_is_workdir` | Write a tool file that returns `os.getcwd()`. Assert the returned cwd matches the `workdir` passed to `LocalPythonTool`. |

**File**: `tests/spec/test_parser.py`

| Test | Description |
|---|---|
| `test_discover_local_tool_name_is_filename_stem` | `arxiv_search.py` → `LocalToolInfo(name="arxiv_search")`. Asserts no dot transformation. |
| `test_discover_local_tool_single_word` | `search.py` → `LocalToolInfo(name="search")`. No underscores, no transformation. |

**File**: `tests/tools/test_local.py` (continued)

| Test | Description |
|---|---|
| `test_validate_schema_rejects_non_dict` | Module with `SCHEMA = "not a dict"`. Assert tool is skipped. |
| `test_validate_schema_rejects_wrong_type` | Module with `SCHEMA = {"type": "not_function", ...}`. Assert skipped. |
| `test_validate_schema_rejects_missing_function` | Module with `SCHEMA = {"type": "function"}` (no `function` key). Assert skipped. |
| `test_validate_schema_rejects_missing_function_name` | `SCHEMA.function` present but no `name` key. Assert skipped. |
| `test_validate_schema_accepts_valid` | Well-formed SCHEMA. Assert tool loads. |
| `test_name_mismatch_rejects_tool` | File `echo_tool.py` with `SCHEMA.function.name = "wrong_name"`. Assert skipped. |
| `test_name_match_accepts_tool` | File `echo_tool.py` with `SCHEMA.function.name = "echo_tool"`. Assert loads. |
| `test_async_run_rejected` | Module with `async def run(args): ...`. Assert skipped. |

### Phase 2: Wire per-tool config

**File**: `tests/tools/test_local.py`

| Test | Description |
|---|---|
| `test_timeout_property_returns_info_timeout` | `LocalToolInfo(timeout=120)`. Assert `tool.timeout == 120`. |
| `test_timeout_property_returns_none_when_unset` | `LocalToolInfo(timeout=None)`. Assert `tool.timeout is None`. |
| `test_retry_property_returns_info_retry` | `LocalToolInfo(retry=RetryConfig(max_attempts=5))`. Assert `tool.retry.max_attempts == 5`. |

**File**: `tests/tools/test_mcp.py`

| Test | Description |
|---|---|
| `test_mcp_tool_timeout_from_config` | `MCPServerConfig(timeout=120)`. Assert `mcp_tool.timeout == 120`. |
| `test_mcp_tool_retry_from_config` | `MCPServerConfig(retry=RetryConfig(...))`. Assert `mcp_tool.retry` matches. |

**File**: `tests/runtime/test_tool_retry.py` (or existing test file)

| Test | Description |
|---|---|
| `test_per_tool_timeout_overrides_global` | Tool with `timeout=120`, global `tools.timeout=60`. Assert 120s used. |
| `test_per_tool_timeout_none_inherits_global` | Tool with `timeout=None`, global `tools.timeout=60`. Assert 60s used. |
| `test_per_tool_retry_overrides_global` | Tool with `retry.max_attempts=5`, global `tools.retry.max_attempts=2`. Assert 5 used. |
| `test_call_tool_with_timeout_calls_cancel` | Pass a mock tool with `cancel()` to `call_tool_with_timeout()`. Assert `cancel()` is called on timeout. |
| `test_call_tool_with_timeout_no_cancel_on_success` | Pass a mock tool with `cancel()` to `call_tool_with_timeout()`. Tool succeeds within deadline. Assert `cancel()` is NOT called. |

**File**: `tests/spec/test_parser.py`

| Test | Description |
|---|---|
| `test_parse_tools_local_overrides_timeout` | `tools.local.arxiv_search.timeout: 120`. Assert `LocalToolInfo.timeout == 120`. |
| `test_parse_tools_local_overrides_retry` | `tools.local.arxiv_search.retry.max_attempts: 5`. Assert `LocalToolInfo.retry.max_attempts == 5`. |
| `test_parse_tools_local_unknown_name_warns` | `tools.local.nonexistent.timeout: 120`. Assert warning logged, no error. |

### Phase 3: Documentation and polish

**File**: `tests/spec/test_validator.py`

| Test | Description |
|---|---|
| `test_typescript_tools_emit_warning` | Agent with `LocalToolInfo(language="typescript")`. Assert validation produces a warning mentioning the tool name. |
| `test_no_warning_without_typescript` | Agent with only Python tools. Assert no TypeScript warning. |

---

## Not Yet

- **Security sandboxing** — subprocess execution provides **fault isolation**
  (crashed tool doesn't crash server) but not **security sandboxing** (child
  process has same filesystem, network, and privilege access as parent). Security
  sandboxing is only needed for untrusted third-party agent images, which are
  out of scope for v1. Would require: filesystem sandboxing (read-only workdir
  mount), resource limits (cgroups/`ulimit`), network isolation (restrict
  outbound to declared MCP URLs), import isolation (clean Python environment).
  See the "Security sandboxing" design decision for the full analysis.

- **TypeScript execution** — requires Node.js subprocess, `package.json`
  dependency resolution, and a TypeScript ↔ Python bridge (JSON over
  stdin/fd 3). The subprocess model designed here for Python translates
  directly — the runner script would be a Node.js equivalent of `_runner.py`.
  The parser already discovers `.ts` files; the loader and runtime are the
  missing pieces.

- **Dependency management** — `requirements.txt` for Python tools,
  `package.json` for TypeScript. The runtime currently assumes dependencies
  are pre-installed in the execution environment. Declared in AGENTSPEC.md
  "Not Yet" as "Tool environment declarations."

- **Async tool execution** — supporting `async def run()`. With subprocess
  execution, the child could run an event loop and await the async `run()`.
  Low demand — tools that need async (HTTP calls) can use `requests`
  synchronously.

- **Process pooling** — the current design spawns a fresh subprocess per tool
  call (~50–100ms overhead). A persistent worker pool (pre-forked processes
  that accept requests over a pipe) would amortize startup cost. Only worth
  it for tools called many times per second, which is unlikely in v1.

- **Dynamic tool registration** — tools are discovered at load time, not
  during execution. No mechanism for a tool to register new tools at runtime.

- **Schema inference from type hints** — inferring `SCHEMA` from `run()`
  type annotations and docstrings instead of requiring an explicit `SCHEMA`
  dict. This would be a convenience for simple tools but adds complexity,
  ambiguity (how to infer `description`, `required`, nested types), and a
  dependency on a schema inference library. Explicit `SCHEMA` gives authors
  full control and matches the MCP model where schemas are explicit.

- **Parallel local tool execution** — currently, multiple tool calls from a
  single LLM response are executed sequentially. Parallel execution would
  require a thread pool with multiple workers. Already tracked in
  AGENTLOOP.md "Not Yet" as "Parallel tool calls."
