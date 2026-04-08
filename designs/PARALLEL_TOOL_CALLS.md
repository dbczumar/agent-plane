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
| `spawn_sub_agents` | Yes | Each spawn creates unique task/conversation IDs; no shared write conflicts |
| `check_sub_agents` | Yes | Read-only; queries separate task records |
| `cancel_sub_agent` | Yes | Each call targets a distinct task ID |
| `load_skill` | Yes | Read-only access to spec and filesystem |
| `read_skill_file` | Yes | Read-only filesystem access |
| `web_search_google` | Yes | External HTTP call; no shared local state |
| `web_search_perplexity` | Yes | External HTTP call; no shared local state |
| `web_search_openai` | Yes | Client-side passthrough; never invoked server-side |

### Infrastructure notes

- **ToolManager lookup** -- `dict.get()` is atomic in CPython. Safe to call
  from concurrent threads.
- **ToolContext** -- frozen dataclass, immutable, created fresh per call. No
  shared state.
- **EventLoopThread (MCP tools)** -- uses `asyncio.run_coroutine_threadsafe()`,
  which is explicitly thread-safe for concurrent submitters.
- **`execute_tool_with_retry`** -- no module-level mutable state; fresh
  `ThreadPoolExecutor(max_workers=1)` per call.
- **MCP discovery cache** -- `TTLCache` is not explicitly locked, but is only
  written during `ToolManager.start()` (single-threaded initialization), so
  concurrent reads during workflow execution are safe.

**Conclusion:** all builtins are safe to run concurrently within a single task.
No tool depends on the result of another tool in the same batch; ordering
is only required for *persistence*, not execution.

---

## DBOS Threading Model

### Why the workflow must be async

DBOS creates an unlimited `ThreadPoolExecutor` (`max_workers=sys.maxsize`) at
startup and sets it as the asyncio event loop's default executor. Sync workflows
each get their own thread -- no contention.

However, **sync `@step()` functions called from an async `@workflow()` run
inline on the event loop thread** -- they block the entire event loop and stall
all other concurrent workflows. The `asyncio.to_thread()` path is only used for
steps called outside a workflow context.

This means that if we make the workflow async (required for `DBOS.asyncio_wait`),
**all I/O-bound steps must also be async** and must `await run_in_executor()`
for their blocking work. Otherwise they block the event loop.

### Verified DBOS async APIs (installed version)

All required async primitives are present:

- `DBOS.asyncio_wait()` -- durable parallel task coordination
- `DBOS.recv_async()` -- async DBOS recv (for client tool parking)
- `DBOS.sleep_async()` -- async sleep (for polling loops)
- `DBOS.start_workflow_async()` -- async workflow dispatch
- `DBOS.write_stream_async()` -- async stream write
- `DBOS.close_stream_async()` -- async stream close

---

## Design

### Approach: async workflow + async steps with `run_in_executor`

The workflow function and all `@step()` functions become `async def`. Blocking
I/O (LLM calls, tool execution) is pushed to the DBOS thread pool via
`asyncio.get_running_loop().run_in_executor(None, ...)`. The `None` executor
uses DBOS's unlimited pool (set as the default executor at startup).

Fast operations (store queries, stream writes) stay as sync calls within async
functions. They briefly block the event loop (~1ms) but this is negligible
compared to LLM/tool call durations (seconds to minutes). Making them async
would require async store interfaces with no meaningful benefit.

### Parallel tool execution

```
LLM emits [tool_A, tool_B, tool_C]
         |
         v
DBOS.asyncio_wait([_call_tool(A), _call_tool(B), _call_tool(C)])
         |
    +----+----+
  A runs   B runs   C runs   (concurrently in thread pool)
    +----+----+
         |  all done
         v
persist results in original call order
```

### Persistence: original call order

The LLM expects `function_call_output` items in the same order as their
`function_call` items. Execution completes in arbitrary order; persistence
re-sorts to the original sequence before writing to `conv_store`. SSE events
are emitted in call order for a consistent client stream.

---

## Changes

### Group A: Workflow and control flow -> `async def`

These become `async def` for `await` propagation. No blocking I/O themselves.

| Function | Line | Why |
|----------|------|-----|
| `agent_execution_workflow` | 3690 | DBOS workflow entry; must be async for asyncio_wait |
| `_run_agent_loop` | 3320 | Core loop; calls async steps and helpers |
| `_executor_turn_with_compaction` | 980 | Coordinates compaction and executor turns |
| `_run_executor_turn` | 831 | Calls `_checkpointed_turn` step |
| `_handle_tool_calls` | 1844 | Calls `_execute_tools` |
| `_handle_final_response` | 1554 | Calls `_persist_and_stream`, `_check_steering_inbox` |
| `_persist_and_stream` | 1483 | Calls `write_stream` (fast, stays sync inside) |
| `_persist_observed_tool_calls` | 1431 | Calls `_persist_and_stream` |
| `_persist_text_before_auto_collect` | 2079 | Calls `_persist_and_stream`, `_auto_collect_sub_agents` |
| `_auto_collect_sub_agents` | 2134 | Calls polling function |
| `_poll_subagents_with_steering_check` | 2184 | Has `time.sleep` -> `await asyncio.sleep` |
| `_check_steering_inbox` | 1663 | Calls store ops |
| `_sync_history` | 2574 | Calls store |
| `_sync_steered_after_tools` | 2523 | Calls store |
| `_call_llm_maybe_compact` | 2830 | Calls `_invoke_llm_streaming` |
| `_invoke_llm_streaming` | 2615 | Calls `_call_llm_streaming` step |
| `_emit_native_tool_items` | 1352 | Calls `_write_output` |
| `_park_for_client_tools` | 2347 | Calls `_wait_for_pending_calls` |
| `_wait_for_pending_calls` | 2453 | Has `dbos_recv` -> `await dbos_recv_async` |
| `_complete_for_client_tools` | 2476 | Calls store ops |
| `_maybe_persist_compaction_item` | 2901 | Calls `_persist_and_stream` |
| `_load_initial_history` | 2679 | Calls store |
| `_get_or_restore_executor_storage` | 3031 | File/artifact I/O |
| `_persist_executor_storage` | 3079 | File/artifact I/O |
| `_consume_executor_live` | 878 | Calls executor iteration |

### Group B: `@step()` functions -> `async def @step()` with `run_in_executor`

These have long-running blocking I/O that must run in the thread pool.

```python
# Pattern for all four steps:
@step()
async def _call_xxx(...) -> R:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: <blocking_call>)
```

| Function | Line | Blocking call |
|----------|------|--------------|
| `_call_llm` | 434 | `client.responses.create(stream=False)` |
| `_call_llm_streaming` | 501 | `client.responses.create(stream=True)` + iterator |
| `_checkpointed_turn` | 655 | `executor.run_turn()` iterator |
| `_call_tool` | 1172 | `execute_tool_with_retry()` |

### Group C: Stays sync (no changes)

Pure computation, serialization, dict manipulation:

`_get_llm_client`, `_create_executor`, `_apply_request_reasoning`,
`_build_responses_args`, `_response_to_dict`, `_accumulate_stream`,
`_event_to_sse_dict`, `_observed_tool_call_sse_dicts`, `_prepare_messages`,
`_reactive_compact_from_overflow`, `_inject_client_tools`, `_item_to_output`,
`_has_tool_calls`, `_get_tool_calls`, `_get_text_content`,
`_build_observed_tool_items`, `_build_function_call_items`,
`_split_tool_calls`, `_track_spawn_collect`, `_recover_spawn_state`,
`_build_assistant_item`, `_inject_collect_results`, `_proactive_compact_if_needed`,
`_reactive_compact`, `_find_spec_by_name`, `_resolve_agent_spec_for_task`,
`_events_to_response_dict`, `_build_executor_context`, `_handle_execution_timeout`

### Group D: Upstream changes

**`agent_plane/runtime/durability.py`** -- export async APIs:

```python
# Add these exports:
dbos_recv_async = DBOS.recv_async
dbos_sleep_async = DBOS.sleep_async
write_stream_async = DBOS.write_stream_async
close_stream_async = DBOS.close_stream_async
start_workflow_async = DBOS.start_workflow_async
asyncio_wait = DBOS.asyncio_wait
```

**`agent_plane/runtime/workflow.py`** -- `_write_output` and `_close_output`
stay sync. They are called both from within `run_in_executor` lambdas (where
they run on a thread pool thread, not the event loop) and from async functions
(where the brief blocking is negligible). No change needed.

**`task_store.start()`** -- stays sync. `DBOS.start_workflow()` dispatches
async workflows correctly from sync contexts (it schedules them on the event
loop). No change needed.

---

## Key implementation: `_execute_tools`

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
    # Launch all tool calls concurrently as async tasks
    tasks = [
        asyncio.ensure_future(
            _call_tool(
                task_id, agent_id, tc.name, tc.arguments,
                tools_config.timeout, tools_config.retry,
            )
        )
        for tc in tool_calls
    ]
    done, _ = await asyncio_wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # Map results back by position (tasks and tool_calls are same order)
    results: dict[str, str] = {
        tool_calls[i].call_id: tasks[i].result()
        for i in range(len(tool_calls))
    }

    # Persist in original call order
    last_seen: str | None = None
    for tc in tool_calls:
        fco_items = _persist_and_stream(
            task_id, conv_store, conversation_id,
            [NewConversationItem(
                type="function_call_output",
                response_id=task_id,
                data=FunctionCallOutputData(
                    call_id=tc.call_id, output=results[tc.call_id],
                ),
            )],
            output_items,
        )
        history.extend(fco_items)
        last_seen = fco_items[-1].id

    assert last_seen is not None
    return last_seen
```

---

## What does NOT change

- `execute_tool_with_retry` -- remains sync; called from thread pool.
- All tool implementations -- no changes needed.
- Client-side tool path -- unaffected; client tools are never passed to
  `_execute_tools`.
- DBOS step durability -- preserved per-call via async `@step()` +
  `DBOS.asyncio_wait()`.
- `task_store.start()` -- stays sync; DBOS handles async workflow dispatch.
- Store interfaces -- all remain sync (fast operations, negligible blocking).

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Sync store ops briefly block event loop | Low impact | Sub-ms DB queries; negligible vs seconds-long LLM/tool calls |
| `run_in_executor` thread pool sizing | Low | DBOS default is unlimited (`sys.maxsize`); LLM rarely emits >10 parallel calls |
| MCP tools with shared resources | Low | Audit MCP server setup; `asyncio.run_coroutine_threadsafe` is thread-safe |
| Out-of-order SSE | None | Persistence re-sorts to original call order before emitting |
| DBOS async workflow crash recovery | Medium | Test recovery with a multi-step async workflow before rolling out |

---

## Open questions (resolved)

1. **DBOS async support?** -- Confirmed. All required APIs present in
   installed version.

2. **`_write_output` thread safety?** -- Safe. Called from `run_in_executor`
   threads where `write_stream` runs on the DBOS thread. `_live_publish`
   uses `asyncio.run_coroutine_threadsafe` which is explicitly thread-safe.

3. **Single-tool fast path?** -- Not needed for v1. The `asyncio_wait` overhead
   for a single task is negligible (~microseconds). Optimize later if profiling
   shows otherwise.
