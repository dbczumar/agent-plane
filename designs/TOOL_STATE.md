# Tool State — stateful `@tool` functions

## Goal

Let `@tool`-decorated custom tools persist state across calls, so
authors can write things like task queues, counters, caches, etc.
without running a separate process (MCP server) or hand-rolling
filesystem code.

Target DX:

```python
from agent_plane_client import tool, ToolState

@tool
def add_task(desc: str, state: ToolState) -> str:
    """Add a task to the queue."""
    queue = state.get("queue", default=[])
    queue.append({"id": len(queue), "desc": desc, "done": False})
    state.set("queue", queue)
    return f"added #{len(queue) - 1}"


@tool
def list_tasks(state: ToolState) -> list[dict]:
    """List all tasks in the queue."""
    return state.get("queue", default=[])


@tool
def complete_task(task_id: int, state: ToolState) -> str:
    """Mark a task as done — atomic read-modify-write."""
    with state.transaction("queue") as queue:
        queue[task_id]["done"] = True
    return "ok"
```

No paths. No JSON handling. No mkdir. No ctx. Authors see exactly
one new type (`ToolState`) and three methods.

## Scope (non-goals)

1. Cross-agent state sharing. Out of scope — use the file store.
2. Cross-conversation state. Out of scope — use the file store.
3. Values larger than ~1 MB/key. Soft cap; document, don't enforce
   in v1. Bigger payloads → `upload_file`.
4. Schema migration for evolved value shapes. Author's problem.
5. Cross-key transactions. Only one key at a time in `transaction()`.
6. Typed value handling (Pydantic models as first-class values).
   V1 is JSON only; authors call `.model_dump()` themselves.

## API — `ToolState`

Lives in `agent_plane_client.tools`. Re-exported at the package
top level so `from agent_plane_client import ToolState` works.

```python
class ToolState:
    """Per-agent, per-conversation key-value state for @tool functions.

    Values are JSON-encoded. Keys are arbitrary strings scoped by
    (conversation, agent) — all tools authored for the same agent
    in the same conversation see one keyspace.
    """

    def get(self, key: str, *, default: Any = None) -> Any:
        """Return the stored value, or ``default`` if absent."""

    def set(self, key: str, value: Any) -> None:
        """Replace (or create) the value at ``key``. JSON-serialized."""

    def delete(self, key: str) -> None:
        """Remove the key. No-op if absent."""

    def keys(self) -> list[str]:
        """Return all keys currently stored in this namespace."""

    @contextmanager
    def transaction(self, key: str) -> Iterator[Any]:
        """Atomic read-modify-write for one key.

            with state.transaction("queue") as queue:
                queue.append(...)
            # queue is written back on normal exit; no write on exception.

        Uses ``fcntl.flock`` to serialize concurrent writers racing
        on the same key. Prevents the classic last-writer-wins race
        when two ``@tool(synchronous=False)`` tools append to the
        same list in parallel.
        """
```

That's the whole surface.

## Storage layout

```
{workspace}/
  .tool_state/
    {agent_id}/
      queue.json
      counter.json
      ...
```

- Directory is created lazily on first `set`/`transaction`.
- Hidden by the dotfile convention so code_sandbox'd `ls` won't
  dump it into agent context (agents who really probe `.` will
  find it; that's fine, it's their state).
- `workspace` is the per-conversation directory already carried on
  `ToolContext.workspace`. Reusing it piggybacks on the existing
  workspace lifecycle (created with the conversation, cleaned when
  the conversation is deleted).
- `agent_id` is taken from `ToolContext.agent_id` — the registered
  agent that owns the tool call. Sub-agents share this with their
  parent (they run under the same `agent_id`); if we later need
  sub-agent isolation we can add a second namespace layer.

## Schema-builder change

The existing `_schema.build_function_schema` walks the function's
signature and includes every parameter in the JSON schema. For
`ToolState`-typed parameters we need to **skip** them — the LLM
must not see them.

One addition:

```python
# In _schema.py
_INJECTED_TYPES: set[type] = {ToolState}

# Inside the signature loop:
if annotation in _INJECTED_TYPES:
    # Framework injects; don't expose to the LLM.
    continue
```

Symmetric with the existing `self`/`cls` skip. Trivial.

The decorator also records which parameter was typed `ToolState`
(by name) in `ToolMetadata`, so the runner knows where to inject.

## Subprocess protocol

The runner request (fd-0 JSON → fd-3 JSON response) currently
carries `{module_path, tool_name, arguments}`. Add two fields:

```json
{
  "module_path": "...",
  "tool_name": "add_task",
  "arguments": {"desc": "write tests"},
  "state_root": "/workspace/conv_abc/.tool_state/ag_xyz",
  "state_param": "state"
}
```

- `state_root`: absolute path to the namespace directory. The
  parent (`LocalPythonTool.invoke`) resolves this from
  `ctx.workspace` + `ctx.agent_id` before spawning the subprocess.
  Parent creates the directory (so the subprocess doesn't race on
  mkdir).
- `state_param`: the name of the parameter to inject into. `null`
  if the tool doesn't take a `ToolState`. Runner uses this to
  construct kwargs: `fn(**arguments, **{state_param: ToolState(state_root)})`.

Both fields are optional; if absent, the runner behaves exactly
as today. Existing stateless `@tool` functions continue to work
unchanged.

## Concurrency — `transaction()`

```python
@contextmanager
def transaction(self, key: str) -> Iterator[Any]:
    path = self._path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open in r+ if it exists, else create empty
    if not path.exists():
        path.write_text("null")
    with path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            value = json.loads(f.read() or "null")
            yield value                      # caller mutates in place
            f.seek(0)
            f.truncate()
            json.dump(value, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
```

- `flock` is advisory but enforced by all `ToolState` callers, so
  effective.
- Cross-subprocess because `flock` on the same file descriptor
  path works across processes on the same host.
- `get` and `set` without a transaction are non-atomic; document
  that race-sensitive updates must go through `transaction()`.
- On `yield`-time exceptions we don't write back — the state is
  untouched, as callers expect.

## Testing

Unit tests in `tests/tools/test_state.py`:

- `get`/`set`/`delete`/`keys` round-trip
- `get` with default, with absent key
- `set` overwrites
- `transaction` commits on normal exit
- `transaction` does NOT commit on exception
- Two `transaction`s on the same key serialize (spawn two
  threads/processes, assert both increments land)
- Two `transaction`s on different keys do not block each other
- Directory is created lazily (not on `get` of absent key)
- `keys()` reflects only existing entries
- JSON round-trip of nested dicts/lists

Integration test in `tests/server/integration/test_tool_state.py`:

- End-to-end: bundle a task-queue agent, run two turns
  (`add_task`, `list_tasks`), assert the list survives across
  tool calls via the ToolState machinery (not via LLM memory).

Example + docs: `examples/clients/python/stateful_tool.py`
showing a working task queue.

## Migration

No existing users. Per rule #33, no compat shims needed. The
schema builder simply gains a `ToolState`-skip branch; the
subprocess protocol gains two optional fields. Stateless `@tool`
functions are unaffected.

## Open questions

None for v1 — the three shape decisions (per-agent namespace,
file locking via `transaction`, `.tool_state/` inside workspace)
were confirmed before writing this doc.
