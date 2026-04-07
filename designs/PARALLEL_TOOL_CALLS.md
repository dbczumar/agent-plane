# Parallel Tool Call Execution

## Problem

Tool calls in the default (LiteLLM) executor are executed sequentially in
`_execute_tools` in `agent_plane/runtime/workflow.py`. When the LLM emits a
batch of N tool calls, each waits for the previous to finish. For independent
tools (web searches, skill loads, sub-agent spawns) this is pure latency waste.

The Claude SDK executor delegates to the SDK, which handles parallelism
internally. This design covers only the default executor path.

---

## Concurrency Audit

All builtin tools were audited for safety under concurrent execution within a
single task/workflow invocation.

| Tool | Concurrent-safe? | Rationale |
|------|-----------------|-----------|
| `spawn_sub_agents` | ✅ Yes | Each spawn creates unique task/conversation IDs; no shared write conflicts |
| `check_sub_agents` | ✅ Yes | Read-only; queries separate task records |
| `cancel_sub_agent` | ✅ Yes | Each call targets a distinct task ID |
| `load_skill` | ✅ Yes | Read-only access to spec and filesystem |
| `read_skill_file` | ✅ Yes | Read-only filesystem access |
| `web_search_google` | ✅ Yes | External HTTP call; no shared local state |
| `web_search_perplexity` | ✅ Yes | External HTTP call; no shared local state |
| `web_search_openai` | ✅ Yes | Client-side passthrough; never invoked server-side |

### Infrastructure notes

- **ToolManager lookup** — `dict.get()` is atomic in CPython. Safe to call
  from concurrent threads.
- **ToolContext** — frozen dataclass, immutable, created fresh per call. No
  shared state.
- **EventLoopThread (MCP tools)** — uses `asyncio.run_coroutine_threadsafe()`,
  which is explicitly thread-safe for concurrent submitters.
- **`execute_tool_with_retry`** — no module-level mutable state; fresh
  `ThreadPoolExecutor(max_workers=1)` per call.
- **MCP discovery cache** — `TTLCache` is not explicitly locked, but is only
  written during `ToolManager.start()` (single-threaded initialization), so
  concurrent reads during workflow execution are safe.

**Conclusion:** all builtins are safe to run concurrently within a single task.
No tool depends on the result of another tool in the same batch; ordering
is only required for *persistence*, not execution.

---

## Design

### Execution: `asyncio.gather` over async steps

Make `_call_tool` an `async def` step. DBOS auto-detects async steps via
`inspect.iscoroutinefunction()` and checkpoints them the same way as sync
steps — each call gets its own durable record.

Run all calls in the batch concurrently using `DBOS.asyncio_wait()`, which is
a checkpoint-aware wrapper around `asyncio.wait()`. On replay, DBOS knows
which tasks already completed and skips re-executing them, preserving
per-call durability.

```
LLM emits [tool_A, tool_B, tool_C]
         │
         ▼
DBOS.asyncio_wait([_call_tool(A), _call_tool(B), _call_tool(C)])
         │
    ┌────┴────┐
  A runs   B runs   C runs   (concurrently)
    └────┬────┘
         │  all done
         ▼
persist results in original call order
```

### Persistence: original call order

The LLM expects `function_call_output` items in the same order as their
`function_call` items. Execution completes in arbitrary order; persistence
must re-sort to the original sequence before writing to `conv_store`.

Results are collected in a `dict[call_id → result]`, then iterated in
`tool_calls` order for persistence. SSE events are also emitted in call
order so the client sees a consistent stream.

### Workflow: async

The DBOS workflow function must be `async def` to use `await` and
`DBOS.asyncio_wait()`. DBOS supports async workflows; the decorator is the
same `@workflow()`.

---

## Changes

### 1. `_call_tool` — make async

```python
@step()
async def _call_tool(
    task_id: str,
    agent_id: str,
    tool_name: str,
    arguments: str,
    timeout: int,
    retry_config: RetryConfig,
) -> str:
    loop = asyncio.get_running_loop()
    # execute_tool_with_retry is sync (blocking); run in thread pool
    # so it doesn't block the DBOS event loop
    return await loop.run_in_executor(
        None,
        lambda: execute_tool_with_retry(
            tool_name=tool_name,
            call_fn=lambda: get_tool_manager().call_tool(
                tool_name, arguments, ToolContext(task_id=task_id, agent_id=agent_id)
            ),
            timeout=timeout,
            retry_config=retry_config,
            on_event=lambda event: _write_output(task_id, event),
        ),
    )
```

No change to `execute_tool_with_retry` or the tools themselves.

### 2. `_execute_tools` — parallelize with `DBOS.asyncio_wait()`

```python
async def _execute_tools(
    task_id: str,
    conversation_id: str,
    tool_calls: list[_ToolCall],
    tools_config: ToolsConfig,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
    conv_store: ConversationStore,
    agent_id: str,
) -> str:
    # Launch all tool calls concurrently
    tasks = [
        asyncio.ensure_future(
            _call_tool(
                task_id, agent_id, tc.name, tc.arguments,
                tools_config.timeout, tools_config.retry,
            )
        )
        for tc in tool_calls
    ]
    done, _ = await DBOS.asyncio_wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # Map completed tasks back to their tool calls by position
    # (tasks list and tool_calls list are in the same order)
    results: dict[str, str] = {
        tool_calls[i].call_id: task.result()
        for i, task in enumerate(tasks)
    }

    # Persist in original call order
    last_seen: str | None = None
    for tc in tool_calls:
        fco_items = _persist_and_stream(
            task_id,
            conv_store,
            conversation_id,
            [
                NewConversationItem(
                    type="function_call_output",
                    response_id=task_id,
                    data=FunctionCallOutputData(
                        call_id=tc.call_id,
                        output=results[tc.call_id],
                    ),
                ),
            ],
            output_items,
        )
        history.extend(fco_items)
        last_seen = fco_items[-1].id

    assert last_seen is not None
    return last_seen
```

### 3. Workflow function — make `async def`

The top-level DBOS workflow function (`_run_agent_loop` or equivalent) becomes
`async def`. All `await` callsites within it are updated accordingly.
`_execute_tools`, `_call_llm_streaming`, and any other internally awaited
helpers become async.

Callers of the workflow (DBOS dispatch, server routes) do not change — DBOS
handles async workflows transparently.

---

## What does NOT change

- `execute_tool_with_retry` — remains sync; called from thread pool.
- All tool implementations — no changes needed.
- `_split_tool_calls`, `_handle_tool_calls`, `_build_function_call_items` —
  unchanged; only `_execute_tools` changes.
- Client-side tool path — unaffected; client tools are never passed to
  `_execute_tools`.
- DBOS step durability — preserved per-call via async `@step()` +
  `DBOS.asyncio_wait()`.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DBOS async workflow limitations discovered at runtime | Medium | Spike with a minimal async workflow before full rollout; check DBOS version supports it |
| `run_in_executor` thread pool exhaustion under large batches | Low | LLM rarely emits >10 tool calls; default `ThreadPoolExecutor` size is sufficient |
| MCP tools acquire a shared resource (e.g. single HTTP connection) | Low | Audit MCP server setup; add tool-level serialization if needed |
| Out-of-order SSE confuses client | None | Persistence (and therefore SSE emission) is explicitly re-sorted to call order |

---

## Open questions

1. **Does the DBOS version in use support `async def` workflows?** Check
   `DBOS.asyncio_wait` is present in the installed version before
   implementing.

2. **`_write_output` thread safety** — currently called from within
   `execute_tool_with_retry` (running in a thread pool executor). If
   `_write_output` touches asyncio internals, it may need to be pushed back
   to the event loop via `loop.call_soon_threadsafe`. Audit before
   implementing.

3. **Single-tool batches** — when the LLM emits only one tool call, the
   parallel path adds overhead with no benefit. Consider a fast path: if
   `len(tool_calls) == 1`, fall back to the existing sync step. This avoids
   introducing async machinery on the common single-tool case.
