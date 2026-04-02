# Executor Contract

## Context

Agent-plane's `_run_agent_loop` today has the LLM call, tool dispatch, compaction,
and steering all coupled together in one function. This makes it impossible to swap
in a different agent runtime (Claude SDK, OpenAI Agents SDK, future harnesses)
without forking the loop.

The goal is an `Executor` ABC that decouples the "call LLM + run tools" concern from
the "persist, stream, compact, steer" concerns. The outer loop stays exactly as it is;
the executor is the only thing that varies.

### What exists today in `_run_agent_loop` (`runtime/workflow.py:1999`)

```
for iteration in range(max_iterations):
    _sync_history(...)                          # pick up steered messages
    llm_resp = _call_llm_maybe_compact(...)     # LLM call + compaction
    _emit_native_tool_items(...)                # web_search_call etc.

    if not _has_tool_calls(llm_resp):
        _auto_collect_sub_agents(...)           # wait for uncollected spawned agents
        result = _handle_final_response(...)    # persist-first-then-check steering
    else:
        _handle_tool_calls(...)                 # split server/client, execute server tools
        _park_for_client_tools(...)  (or)       # sub-agent: park for client
        _complete_for_client_tools(...)         # top-level: return to client
        _sync_steered_after_tools(...)          # pick up steering between tool calls
```

Everything in the loop except `_call_llm_maybe_compact` stays. The executor replaces
that one call.

---

## Design Decisions

### Event semantics replace capability flags

OmniAgents has a `handles_tools_internally()` flag that tells the framework whether
the executor executed a tool or is asking the framework to execute it. This is a flag
doing work that event types should do.

Two events with distinct, unambiguous semantics:

- **`ToolCallRequested`** — the executor wants the workflow to execute this tool.
  It is waiting. The workflow MUST execute and re-invoke `run_turn()` with the result
  appended to `messages`.

- **`ToolCallObserved`** — the executor already executed this tool internally. It is
  informational. The workflow persists both sides and streams them; it does NOT execute
  the tool or re-invoke `run_turn()`.

No flag needed. The workflow reacts to which event type arrives:

```python
for event in _run_executor_turn(executor, ...):
    if isinstance(event, ToolCallRequested):
        result = _call_tool(...)   # @step — workflow executes
        messages.append(_tool_result_message(event.name, result, event.call_id))
        # re-invoke run_turn() with updated messages in next outer loop iteration

    elif isinstance(event, ToolCallObserved):
        _persist_and_stream(...)   # just persist both sides
```

### The full event set

```python
@dataclass
class TextChunk:
    """
    A streamed text token from the model.

    :param text: The incremental text fragment, e.g. ``"Hello"``.
    """
    text: str


@dataclass
class ToolCallRequested:
    """
    The executor wants the workflow to execute a tool.

    The executor is suspended until the workflow re-invokes run_turn()
    with the result appended to messages. The workflow must execute the
    tool via _call_tool() (@step) and append a tool_result message.

    :param call_id: Stable identifier for this call, e.g. ``"call_abc123"``.
        Used to correlate the function_call and function_call_output items.
    :param name: Tool name, e.g. ``"web_search"``.
    :param arguments: Parsed tool arguments dict.
    """
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallObserved:
    """
    The executor ran a tool internally. Workflow just persists and streams.

    Emitted by internal executors (Claude SDK, OpenAI Agents SDK) after
    each tool the harness executed autonomously. Both the call and the
    result are bundled — the workflow never executes the tool itself.

    :param call_id: Stable identifier, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"Bash"``.
    :param arguments: Parsed tool arguments dict.
    :param result: The tool's output string.
    :param status: ``"success"`` | ``"error"`` | ``"blocked"``.
    :param duration_ms: Wall-clock time the tool took, e.g. ``342.1``.
    """
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    status: str
    duration_ms: float


@dataclass
class TurnComplete:
    """
    The executor has finished its turn.

    For external executors (yielded ToolCallRequested events): text is None
    and the workflow re-invokes run_turn() with tool results.
    For internal executors or final text responses: text is the full
    assistant reply.

    :param text: The assistant's text response, or ``None`` if the turn
        ended with tool calls only.
    """
    text: str | None


@dataclass
class ContextWindowExceeded:
    """
    The executor hit a context window overflow.

    The workflow reacts by compacting messages and retrying run_turn().
    This replaces the ContextWindowExceededError exception path used
    today inside _call_llm_maybe_compact.

    :param max_tokens: The model's context window size, e.g. ``128000``.
        Used by the workflow to calibrate the compaction budget.
    :param actual_tokens: The prompt size that triggered the overflow,
        e.g. ``131072``. Used by the workflow to validate the tiktoken
        estimate before reactive compaction.
    """
    max_tokens: int
    actual_tokens: int


@dataclass
class ExecutorError:
    """
    An unrecoverable executor failure.

    The workflow emits a response.error SSE event and returns a failed
    _AgentLoopResult. No retry.

    :param message: Human-readable error description.
    :param code: Machine-readable error code, e.g. ``"auth_failed"``.
    """
    message: str
    code: str | None = None


ExecutorEvent: TypeAlias = (
    TextChunk
    | ToolCallRequested
    | ToolCallObserved
    | TurnComplete
    | ContextWindowExceeded
    | ExecutorError
)
```

### The ABC

```python
class Executor(abc.ABC):
    @abc.abstractmethod
    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
    ) -> Iterator[ExecutorEvent]:
        """
        Run one executor turn and yield events.

        For external executors: one LLM call. Yields TextChunks and
        ToolCallRequested events. The workflow executes requested tools,
        appends results to messages, and calls run_turn() again.

        For internal executors: the full agent loop until the model
        produces a final response. Yields TextChunks, ToolCallObserved
        pairs, and a terminal TurnComplete or ExecutorError.

        :param messages: Conversation history as Responses API input
            items (output of history_to_input_items or compact()).
            Already compacted by the workflow if needed.
        :param tools: OpenAI-format tool schemas for this turn.
        :param system_prompt: Assembled system instructions string.
        :param config: Model and sampling configuration.
        :param task_id: The current task's identifier. Used by executors
            that need to emit SSE events or key per-task state,
            e.g. ``"task_abc123"``.
        :param conversation_id: The conversation's identifier. Used by
            executors that maintain per-conversation subprocess state
            (temp dirs, artifact store keys),
            e.g. ``"conv_abc123"``.
        """
        ...

    def on_task_start(
        self, task_id: str, conversation_id: str, storage_dir: Path,
    ) -> None:
        """
        Called once at the start of a task, before the first run_turn().

        The workflow restores the scoped persistent directory from the
        artifact store before calling this hook. The executor can read
        prior session state from storage_dir immediately.

        :param task_id: The task identifier, e.g. ``"task_abc123"``.
        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param storage_dir: Scoped persistent directory for this
            conversation. Contents survive across tasks — the workflow
            manages artifact store I/O. The executor reads/writes
            freely within this directory.
        """

    def on_task_end(
        self, task_id: str, conversation_id: str, storage_dir: Path,
    ) -> None:
        """
        Called once after the task completes (or fails), in a finally block.

        The workflow persists storage_dir to the artifact store after
        this hook returns. Use for cleanup (e.g. disconnecting SDK
        client) while preserving any state written to storage_dir.

        :param task_id: The task identifier, e.g. ``"task_abc123"``.
        :param conversation_id: The conversation identifier,
            e.g. ``"conv_abc123"``.
        :param storage_dir: Same scoped directory from on_task_start.
        """

    def max_context_tokens(self) -> int | None:
        """
        The executor's context window limit, or None if unknown/managed
        internally.

        Used by the workflow to seed the compaction budget on the first
        turn, before any overflow has been observed. Executors that manage
        their own compaction (e.g. Claude SDK) return None.

        :returns: Token limit, e.g. ``128000``, or ``None``.
        """
        return None
```

`run_turn()` returns a sync `Iterator[ExecutorEvent]`. The workflow is sync
(DBOS `@workflow`); async executors bridge via a thread + queue (see
Implementation). The ABC is sync to match the caller.

### ExecutorConfig

```python
@dataclass
class ExecutorConfig:
    """
    Model and sampling parameters for one run_turn() call.

    :param model: The litellm model identifier, e.g.
        ``"databricks/databricks-claude-sonnet-4"``.
    :param temperature: Sampling temperature, e.g. ``0.0``.
    :param max_tokens: Max output tokens, e.g. ``4096``.
    :param timeout: Per-call timeout in seconds, e.g. ``120``.
    :param retry: Retry policy for transient failures.
    :param reasoning: Optional reasoning config dict, e.g.
        ``{"effort": "high"}``.
    :param connection: Optional provider connection override.
    """
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    retry: RetryConfig
    reasoning: dict[str, str] | None = None
    connection: str | None = None
```

Strongly typed. No `extra: dict[str, Any]` catch-all — provider-specific
parameters go in `connection` or are baked into the executor subclass.

### Compaction is the workflow's responsibility

The executor receives already-compacted messages. It does not compact.
Two signals:

<DOESNT THIS CONTRADICT THE DESIGN FOR CLAUDE CODE INTEGRATION WHERE WE DECIDED
THAT COMPACTION IS LIKELY MUCH BETTER IN CLAUDE THAN IN THE WORKFLOW SO ITS BETTER
TO DELEGATE IF POSSIBLE? I THINK THERE SHOULD BE AN OPTION>

1. **Proactive** — the workflow estimates tokens before calling `run_turn()`.
   If over threshold, it compacts messages and then calls `run_turn()`.
   This is exactly `_proactive_compact_if_needed()` today — it moves
   to just before the executor call.

2. **Reactive** — if the executor hits overflow, it yields
   `ContextWindowExceeded(max_tokens, actual_tokens)` instead of
   `TurnComplete`. The workflow catches it, validates via tiktoken,
   compacts, and retries `run_turn()` with compacted messages.

The tiktoken validation and divergence check from `_reactive_compact()`
today (`0.7 ≤ ratio ≤ 1.3`) stay in the workflow. The executor only
needs to surface the provider's reported token counts.

### Mapping of existing `_run_agent_loop` logic

This table shows every significant function in the current loop and where
its responsibility lands in the new design:

| Current function | Stays in workflow | Moves to executor |
|-----------------|-------------------|-------------------|
| `_load_initial_history()` | ✓ unchanged | |
| `_sync_history()` | ✓ unchanged | |
| `_proactive_compact_if_needed()` | ✓ called before executor | |
| `_reactive_compact()` | ✓ called on `ContextWindowExceeded` | |
| `_maybe_persist_compaction_item()` | ✓ in finally block | |
| `_invoke_llm_streaming()` | | ✓ inside `DefaultExecutor.run_turn()` |
| `_emit_native_tool_items()` | ✓ unchanged (native tools like web_search don't go through executor) | |
| `_handle_final_response()` | ✓ unchanged — triggered by `TurnComplete` | |
| `_handle_tool_calls()` | partially — split/persist/stream stays; `_call_tool()` @step stays | |
| `_split_tool_calls()` | ✓ unchanged for `ToolCallRequested` events | |
| `_call_tool()` @step | ✓ called for `ToolCallRequested` | |
| `_persist_and_stream()` | ✓ called for all events | |
| `_sync_steered_after_tools()` | ✓ unchanged | |
| `_park_for_client_tools()` | ✓ unchanged — triggered after `ToolCallRequested` for client tools | |
| `_complete_for_client_tools()` | ✓ unchanged | |
| `_track_spawn_collect()` | ✓ unchanged | |
| `_auto_collect_sub_agents()` | ✓ unchanged | |
| `_handle_execution_timeout()` | ✓ unchanged | |
| `_build_instructions()` | ✓ called before executor | |
| `history_to_input_items()` | ✓ called before executor | |

The executor absorbs exactly one function: `_invoke_llm_streaming()`. For SDK
executors, it also absorbs the inner tool loop (the LLM → tool → LLM iteration
that today would require multiple `_run_agent_loop` iterations).

### The new loop body

`_run_agent_loop` after the executor is introduced. Only the inner call site
changes — the loop structure, all helper calls, and all durability logic are
unchanged:

```python
def _run_agent_loop(
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    agent_name: str,
    agent_id: str,
    instructions: str | None,
    tool_mgr: ToolManager,
    executor: Executor,               # ← new parameter
    reasoning: dict[str, str] | None = None,
) -> _AgentLoopResult:
    ...
    storage_dir = _restore_executor_storage(conversation_id)
    executor.on_task_start(task_id, conversation_id, storage_dir)
    try:
        for iteration in range(max_iterations):
            # unchanged: timeout, _sync_history
            ...

            # Build messages — unchanged
            sys_instructions = build_instructions(spec, instructions, tool_schemas)
            messages = history_to_input_items(resolved_history)
            messages = _proactive_compact_if_needed(
                messages, history, sys_tokens, compaction_state, task_id
            )

            # Run executor — replaces _call_llm_maybe_compact
            tool_calls_this_turn: list[ToolCallRequested] = []
            turn_complete: TurnComplete | None = None

            for event in _run_executor_turn(executor, messages, tool_schemas,
                                            sys_instructions, config, task_id,
                                            conversation_id):
                if isinstance(event, TextChunk):
                    _write_output(task_id, {
                        "type": "response.output_text.delta",
                        "delta": event.text,
                    })

                elif isinstance(event, ToolCallRequested):
                    fc_item = _persist_and_stream(
                        task_id, conv_store, conversation_id,
                        [_build_function_call_item(task_id, agent_name, event)],
                        output_items,
                    )
                    history.extend(fc_item)
                    tool_calls_this_turn.append(event)

                elif isinstance(event, ToolCallObserved):
                    # Internal executor — persist both sides, never execute
                    _persist_and_stream(
                        task_id, conv_store, conversation_id,
                        [
                            _build_function_call_item(task_id, agent_name, event),
                            _build_function_call_output_item(task_id, event),
                        ],
                        output_items,
                    )
                    _track_spawn_collect(output_items, spawned_ids, collected_ids)

                elif isinstance(event, ContextWindowExceeded):
                    messages = _reactive_compact(
                        messages, history, sys_tokens,
                        event.max_tokens, event.actual_tokens,
                        compaction_state, task_id,
                    )
                    break  # retry outer iteration with compacted messages

                elif isinstance(event, TurnComplete):
                    turn_complete = event
                    break

                elif isinstance(event, ExecutorError):
                    _emit_llm_error_event(task_id, event)
                    return _AgentLoopResult(status="failed", ...)

            if turn_complete is None:
                continue   # compaction retry — restart iteration

            # From here: identical to today
            _emit_native_tool_items(task_id, llm_resp, output_items)

            if not tool_calls_this_turn:
                # No tools — auto-collect, steering handshake
                ...unchanged...
            else:
                # Tool calls — split client/server, execute server, park/complete for client
                ...unchanged (operates on tool_calls_this_turn instead of llm_resp)...

    finally:
        executor.on_task_end(task_id, conversation_id, storage_dir)
        _persist_executor_storage(conversation_id, storage_dir)
        if compaction_state.last_summary is not None:
            _maybe_persist_compaction_item(...)
```

The only substantive change to the loop: `_call_llm_maybe_compact()` is replaced
by a `for event in _run_executor_turn(...)` block. Everything else is identical.

### DefaultExecutor: existing litellm path

`DefaultExecutor` wraps `_invoke_llm_streaming()`. It is a thin shim — all the
logic that currently lives in `_call_llm_maybe_compact()` is split: compaction
stays in the workflow (see above); only the raw LLM call moves into the executor.

```python
class DefaultExecutor(Executor):
    """
    Executor backed by litellm. Does not handle tools internally.

    Yields ToolCallRequested events for each tool call in the LLM response.
    The workflow executes tools and re-invokes run_turn() with results.
    This is exactly the existing _invoke_llm_streaming() behavior,
    wrapped in the Executor interface.
    """

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
    ) -> Iterator[ExecutorEvent]:
        try:
            llm_resp = _invoke_llm_streaming(
                task_id, messages, system_prompt,
                config.model, tools, config.connection,
                config.timeout, config.retry,
            )
        except ContextWindowExceededError as exc:
            yield ContextWindowExceeded(
                max_tokens=exc.max_context_tokens,
                actual_tokens=exc.actual_tokens,
            )
            return
        except (RetryableLLMError, PermanentLLMError) as exc:
            yield ExecutorError(message=str(exc), code=exc.code)
            return

        # Yield text chunks (already streamed to SSE inside _invoke_llm_streaming;
        # here we yield the final accumulated text as a single TextChunk for the
        # workflow to build the assistant item from)
        text = _get_text_content(llm_resp)
        if text:
            yield TextChunk(text=text)

        # Yield tool call requests
        for tc in _get_tool_calls(llm_resp):
            yield ToolCallRequested(
                call_id=tc.call_id,
                name=tc.name,
                arguments=json.loads(tc.arguments),
            )

        yield TurnComplete(text=text if not _get_tool_calls(llm_resp) else None)
```

`ClaudeSDKExecutor` follows `CODINGAGENTS.md`. It yields `ToolCallObserved` pairs
for every tool the SDK runs internally and `TurnComplete` at the end. It uses
`on_task_start` / `on_task_end` for artifact store save/restore.

### Sync/async bridge

DBOS `@workflow` is sync. SDK executors are async. The bridge is a single
generic function wrapping any executor's async generator:

```python
def _run_executor_turn(
    executor: Executor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    config: ExecutorConfig,
    task_id: str,
    conversation_id: str,
) -> Iterator[ExecutorEvent]:
    """
    Bridge between async executor implementations and the sync DBOS workflow.

    Runs the executor's async generator in a daemon thread with its own
    asyncio event loop. Events are pushed into a bounded queue and yielded
    to the sync caller. A None sentinel signals completion.

    :param executor: The executor to run.
    :param messages: Pre-compacted Responses API input items.
    :param tools: OpenAI-format tool schemas.
    :param system_prompt: Assembled system instructions.
    :param config: Model and sampling config.
    :param task_id: Current task identifier.
    :param conversation_id: Current conversation identifier.
    """
    q: queue.Queue[ExecutorEvent | None] = queue.Queue(maxsize=256)

    def _run_async() -> None:
        async def _drain() -> None:
            async for event in executor.run_turn(
                messages, tools, system_prompt, config, task_id, conversation_id
            ):
                q.put(event)
        asyncio.run(_drain())
        q.put(None)

    thread = threading.Thread(target=_run_async, daemon=True)
    thread.start()
    while True:
        event = q.get()
        if event is None:
            break
        yield event
    thread.join()
```

`DefaultExecutor` is sync and skips the bridge entirely — it returns a sync
generator directly.

### Executor construction

`agent_execution_workflow` creates the executor once, before `_run_agent_loop`:

```python
def agent_execution_workflow(task_id, conversation_id, ...):
    ...
    executor = _create_executor(spec)
    result = _run_agent_loop(..., executor=executor)
```

```python
def _create_executor(spec: AgentSpec) -> Executor:
    """
    Construct the executor for the given agent spec.

    :param spec: The parsed AgentSpec with a non-None llm field.
    :returns: A concrete Executor instance.
    """
    assert spec.llm is not None
    config = ExecutorConfig(
        model=spec.llm.model,
        temperature=spec.llm.temperature,
        max_tokens=spec.llm.max_tokens,
        timeout=spec.llm.timeout,
        retry=spec.llm.retry,
        reasoning=None,
        connection=spec.llm.connection,
    )
    harness = spec.llm.executor
    if harness == "claude_sdk":
        return ClaudeSDKExecutor(config=config)
    return DefaultExecutor(config=config)
```

`LLMConfig` gains one new field: `executor: str | None = None`. All existing
specs continue to work — `None` maps to `DefaultExecutor`.

---

## Implementation

### New files

**`runtime/executor.py`** — Contains:
- All event dataclasses (`TextChunk`, `ToolCallRequested`, `ToolCallObserved`,
  `TurnComplete`, `ContextWindowExceeded`, `ExecutorError`, `ExecutorEvent` alias)
- `ExecutorConfig` dataclass
- `Executor` ABC
- `_run_executor_turn()` — sync/async bridge
- `DefaultExecutor` — wraps `_invoke_llm_streaming`
- `_create_executor()` — factory

**`runtime/claude_sdk_executor.py`** — `ClaudeSDKExecutor`. Described in
`CODINGAGENTS.md`. Uses `on_task_start` / `on_task_end` for artifact store.

### Changed files

**`spec/types.py`** — Add `executor: str | None = None` to `LLMConfig`.

**`runtime/workflow.py`** — Three changes only:
1. `agent_execution_workflow`: call `_create_executor(spec)`, pass to loop.
2. `_run_agent_loop`: add `executor: Executor` parameter; replace
   `_call_llm_maybe_compact()` call with `_run_executor_turn()` event loop.
3. `_reactive_compact()`: accept `max_tokens` / `actual_tokens` from
   `ContextWindowExceeded` event instead of from the exception.

All other functions in `workflow.py` are unchanged.

### Reused without changes

| Component | Why no changes needed |
|-----------|----------------------|
| `ConversationStore` | Same item types, same write patterns |
| `TaskStore` | Steering handshake unchanged |
| `_write_output` | Same SSE delivery, same event dict format |
| `_handle_final_response` | Triggered by `TurnComplete` — unchanged |
| `_handle_tool_calls` | Now operates on `ToolCallRequested` list instead of `llm_resp` |
| `_call_tool` @step | Called for `ToolCallRequested` events from external executors |
| `_persist_and_stream` | Called from the event consumer loop |
| `_park_for_client_tools` | Called when `ToolCallRequested` is a client-side tool |
| `_complete_for_client_tools` | Unchanged |
| `_proactive_compact_if_needed` | Called before `_run_executor_turn` |
| `_maybe_persist_compaction_item` | In `finally` block — unchanged |
| All API routes | `POST /v1/responses` contract unchanged |
| DBOS `@workflow` | Same `agent_execution_workflow` |

---

## Durability

The durability story is unchanged because all durable operations stay in the
workflow:

- **conv_store writes** happen in the event consumer loop — each `ToolCallRequested`
  and `ToolCallObserved` event triggers a `_persist_and_stream` call before the loop
  continues. Same write-ahead semantics as today.

- **`_call_tool()` @step** is still called from the workflow (not inside the executor)
  for `ToolCallRequested` events. DBOS caches the step output and skips re-execution
  on crash recovery — same guarantee as today.

- **`ContextWindowExceeded` vs exception**: the event-based approach is equivalent
  to the exception-based approach for recovery purposes — both trigger compaction and
  retry from the same loop iteration.

- **`on_task_end` in `finally`**: artifact store persistence for SDK executors runs
  in the same `finally` block as `_maybe_persist_compaction_item`. On crash, neither
  runs — DBOS re-invokes the full workflow and both run on recovery.

The one durability tradeoff is for `ToolCallObserved` (internal executors): the SDK's
inner tool loop is not @step-wrapped. On crash mid-SDK-turn, the last in-flight tool
call may re-execute. This is acceptable — Claude Code's built-in tools are idempotent
file operations. This tradeoff is described in `CODINGAGENTS.md`.

---

## Test Plan

1. **DefaultExecutor text response**: verify `TextChunk` events arrive, `TurnComplete`
   triggers `_handle_final_response`, assistant message persisted.

2. **DefaultExecutor tool call**: verify `ToolCallRequested` emitted, `_call_tool`
   @step called, result appended to messages, second `run_turn()` returns `TurnComplete`.

3. **DefaultExecutor context overflow**: configure model with tiny context. Verify
   `ContextWindowExceeded` emitted, `_reactive_compact` called, `run_turn()` retried
   with compacted messages.

4. **DefaultExecutor client-side tool**: verify `ToolCallRequested` for client tool
   triggers `_park_for_client_tools` / `_complete_for_client_tools` unchanged.

5. **ClaudeSDKExecutor internal tool**: verify `ToolCallObserved` pair persisted,
   `_call_tool` NOT called, conv_store has both function_call and function_call_output.

6. **`on_task_start` / `on_task_end`**: verify both called once per task, in the
   right order relative to `run_turn()`. Verify `on_task_end` runs even if executor
   raises.

7. **Executor construction**: `executor: "claude_sdk"` in spec → `ClaudeSDKExecutor`.
   `executor: null` or absent → `DefaultExecutor`. Existing agents unaffected.

8. **Steering handshake**: unchanged — `TurnComplete` triggers `_handle_final_response`
   which does persist-first-then-check. Verify steered messages still caught.

9. **Execution timeout**: verify timeout check fires before `_run_executor_turn` on
   each iteration. Timeout during executor call: verify thread terminates cleanly.

10. **Crash recovery (DefaultExecutor)**: kill workflow after first tool @step. Verify
    DBOS re-invokes, @step result cached, tool not re-executed, workflow completes.

11. **Crash recovery (ClaudeSDKExecutor)**: kill workflow mid-SDK-turn. Verify DBOS
    re-invokes, `on_task_start` restores artifact blob or falls back to conv_store
    reconstruction, workflow completes.

12. **Backward compatibility**: agent spec with no `executor` field runs on
    `DefaultExecutor` with full existing behavior.

---

## Concrete Implementations and Gap Analysis

This section writes out two concrete executor implementations against the ABC
defined above, identifies every friction point, and proposes contract changes.

### Implementation 1: ClaudeSDKExecutor (agent-plane)

The Claude Agent SDK runs its own agent loop. Tools execute inside the SDK
subprocess (or via `@tool` MCP handlers). Agent-plane observes and persists.

```python
class ClaudeSDKExecutor(Executor):
    """
    Executor that wraps the Claude Agent SDK.

    The SDK runs Claude Code's full agent loop internally. Built-in tools
    (Bash, Read, Edit, etc.) execute server-side inside the SDK subprocess.
    Client-side tools (registered by the API caller) are bridged via ``@tool``
    MCP handlers that park until the client delivers results.

    State: maintains a persistent ``ClaudeSDKClient`` per conversation_id
    across run_turn() calls (the SDK subprocess keeps context in memory).

    :param allowed_tools: Server-side tool names, e.g. ``["Bash", "Read", "Edit"]``.
    :param model: Optional model override, e.g. ``"claude-sonnet-4-20250514"``.
    """

    def __init__(
        self,
        *,
        allowed_tools: list[str],
        model: str | None = None,
    ) -> None:
        self._allowed_tools = allowed_tools
        self._model = model
        # Persistent SDK clients keyed by conversation_id
        self._clients: dict[str, ClaudeSDKClient] = {}
        # storage_dir per conversation_id (set by on_task_start)
        self._storage_dirs: dict[str, Path] = {}
        # Client-side tool parking callback (set by set_client_tool_callback)
        self._park_callback: ClientToolCallback | None = None

    # --- Lifecycle hooks ---

    def on_task_start(
        self, task_id: str, conversation_id: str, storage_dir: Path,
    ) -> None:
        """
        Record the storage dir for this conversation.

        The workflow already restored prior session state into
        storage_dir. The SDK can ``--continue`` if ``.claude/``
        exists inside it.

        :param task_id: Current task identifier, e.g. ``"task_abc123"``.
        :param conversation_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :param storage_dir: Scoped persistent directory (already restored).
        """
        self._storage_dirs[conversation_id] = storage_dir

    def on_task_end(
        self, task_id: str, conversation_id: str, storage_dir: Path,
    ) -> None:
        """
        Disconnect SDK client. Session state in storage_dir is
        persisted by the workflow after this returns.

        :param task_id: Current task identifier.
        :param conversation_id: Conversation identifier.
        :param storage_dir: Scoped persistent directory.
        """
        self._storage_dirs.pop(conversation_id, None)
        client = self._clients.pop(conversation_id, None)
        if client is not None:
            asyncio.run(client.disconnect())

    def max_context_tokens(self) -> int | None:
        """
        SDK manages its own context window — return None.

        :returns: None — workflow should not proactively compact.
        """
        return None  # GAP C: compaction is irrelevant

    # --- Core turn ---

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
    ) -> Iterator[ExecutorEvent]:
        """
        Run one SDK turn. Bridges async SDK into sync iterator.

        The SDK may execute many internal tool calls before producing
        a final response. Each observed tool call is yielded as
        ``ToolCallObserved``. Client-side tools are handled internally
        via ``pending_tool_calls`` parking — the workflow never sees
        ``ToolCallRequested`` from this executor.

        :param messages: Responses API input items. Used only for
            fresh sessions (to extract the user's message) or crash
            recovery (to build a history prompt). Ignored for continuing
            sessions where the SDK subprocess has context in memory.
        :param tools: Client-side tool schemas. Server-side tools are
            configured at construction time via ``allowed_tools``.
        :param system_prompt: System instructions string.
        :param config: Model and sampling config.
        :param task_id: Current task identifier.
        :param conversation_id: Current conversation identifier.
        """
        # Sync/async bridge via thread + queue
        q: queue.Queue[ExecutorEvent | None] = queue.Queue(maxsize=256)

        def _run():
            asyncio.run(self._async_run_turn(
                messages, tools, system_prompt, config,
                task_id, conversation_id, q,
            ))
            q.put(None)  # sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        while True:
            event = q.get()
            if event is None:
                break
            yield event
        thread.join()

    async def _async_run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
        q: queue.Queue[ExecutorEvent | None],
    ) -> None:
        """
        Async implementation of run_turn. Pushes events into the queue.

        :param messages: Responses API input items.
        :param tools: Client-side tool schemas.
        :param system_prompt: System instructions.
        :param config: Model and sampling config.
        :param task_id: Task identifier.
        :param conversation_id: Conversation identifier.
        :param q: Thread-safe queue for bridging events to sync caller.
        """
        sdk = _ensure_sdk()
        storage_dir = self._storage_dirs.get(conversation_id)
        is_continuing = conversation_id in self._clients

        # Build prompt
        if is_continuing:
            # Subprocess has context in memory — just send the new message
            prompt = _extract_latest_user_message(messages)
        elif storage_dir and _has_transcript(storage_dir):
            # Restored blob — CLI replays transcript via --continue
            prompt = _extract_latest_user_message(messages)
        else:
            # Fresh session or crash recovery without blob — full prompt
            prompt = _build_prompt_from_history(messages)

        # Build @tool handlers for client-side tools (GAP B: needs parking deps)
        client_tool_handlers = _build_client_tool_handlers(
            tools,
            task_id=task_id,
            task_store=self._task_store,
            write_output=self._write_output,
            root_task_id_fn=self._root_task_id_fn,
        )
        mcp_server = sdk.create_sdk_mcp_server(
            "client_tools", tools=client_tool_handlers,
        )

        # Get or create persistent client
        client = await self._get_or_create_client(
            sdk, conversation_id, system_prompt, mcp_server,
        )

        # Track pending tools for call_id correlation
        pending: dict[str, tuple[str, float]] = {}
        got_stream_events = False
        response_text = ""

        try:
            await client.query(prompt, session_id=conversation_id)
            async for message in client.receive_response():
                if isinstance(message, sdk.StreamEvent):
                    got_stream_events = True
                    evt = message.event
                    evt_type = evt.get("type", "")

                    if evt_type == "content_block_start":
                        block = evt.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "unknown")
                            pending[tool_id] = (tool_name, time.monotonic())

                    elif evt_type == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                response_text += text
                                q.put(TextChunk(text=text))

                elif isinstance(message, sdk.UserMessage):
                    # Tool results — match to pending request
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, sdk.ToolResultBlock):
                                name, start = pending.pop(
                                    block.tool_use_id,
                                    ("unknown", time.monotonic()),
                                )
                                duration_ms = (time.monotonic() - start) * 1000
                                q.put(ToolCallObserved(
                                    call_id=block.tool_use_id,
                                    name=name,
                                    arguments={},  # GAP D: SDK doesn't echo args
                                    result=_extract_text(block.content),
                                    status="error" if block.is_error else "success",
                                    duration_ms=duration_ms,
                                ))

                elif isinstance(message, sdk.AssistantMessage):
                    if not got_stream_events:
                        for block in message.content:
                            if isinstance(block, sdk.TextBlock):
                                response_text += block.text
                                q.put(TextChunk(text=block.text))
                            elif isinstance(block, sdk.ToolUseBlock):
                                pending[block.id] = (block.name, time.monotonic())

                elif isinstance(message, sdk.ResultMessage):
                    if not response_text and message.result:
                        response_text = message.result

            q.put(TurnComplete(text=response_text or None))

        except Exception as exc:
            # Crash — mark session as dead, clean up
            await self._close_client(conversation_id)
            q.put(ExecutorError(message=f"Claude SDK error: {exc}"))
```

#### Gaps exposed by ClaudeSDKExecutor

**GAP A: Executor needs persistent storage across tasks.** ✅ Resolved.

The workflow passes a scoped `storage_dir: Path` into `on_task_start` and
`on_task_end`. The executor reads/writes freely within that directory — it
persists across tasks automatically. The workflow manages all artifact store
I/O (tar/upload/download/restore) transparently. The executor never sees
the artifact store.

```python
def on_task_start(
    self, task_id: str, conversation_id: str, storage_dir: Path,
) -> None:
    """
    Called once at task start, after storage_dir has been restored.

    :param storage_dir: Scoped persistent directory for this conversation.
        Contents survive across tasks. The workflow manages persistence.
    """

def on_task_end(
    self, task_id: str, conversation_id: str, storage_dir: Path,
) -> None:
    """
    Called once at task end. Writes to storage_dir will be persisted.
    """
```

Workflow side:
1. Restore: `artifact_store.get(f"executor/{conversation_id}")` → extract to temp dir
2. Call `on_task_start(..., storage_dir=temp_dir)`
3. Task runs
4. Call `on_task_end(..., storage_dir=temp_dir)`
5. Persist: if dir non-empty → tar → `artifact_store.put(...)`

ClaudeSDKExecutor writes `.claude/` into `storage_dir`. DefaultExecutor ignores it.
No executor ever gets an `ArtifactStore` reference.

**GAP B: Client-side tool parking requires workflow-level dependencies.**
SDK @tool handlers for client-side tools need:
- `task_store` — to INSERT/poll `pending_tool_calls`
- `_write_output` — to emit `action_required` SSE events
- `root_task_id` — to tunnel to the root response

These are workflow concerns, not executor concerns. But the @tool handler runs
INSIDE the SDK's event loop — the executor can't yield a "please park this"
event and wait for the workflow to respond, because the SDK expects a synchronous
return from the handler.

Options:
1. **Inject dependencies at construction** (shown above). Works but couples the
   executor to agent-plane's task_store and SSE plumbing.
2. **Callback-based**: Pass a `park_for_client_tool(call_id, name, args) -> str`
   callback that blocks until the client delivers the result. The callback
   encapsulates task_store + SSE + polling. Executor calls it from the @tool
   handler. This is cleaner — executor depends on one opaque function, not
   three stores.
3. **New event type**: `ToolCallParkRequested`. Executor yields it, workflow
   parks. But this breaks the streaming model — the SDK's @tool handler is
   running synchronously inside the executor, so the executor can't yield and
   wait for the workflow simultaneously.

**Recommendation: Option 2.** Add a `client_tool_callback` to the contract:

```python
ClientToolCallback: TypeAlias = Callable[[str, str, dict[str, Any]], str]
# (call_id, tool_name, arguments) -> tool_result_string
# Blocks until the client delivers the result.

class Executor(abc.ABC):
    def set_client_tool_callback(
        self, callback: ClientToolCallback,
    ) -> None:
        """
        Register a callback for parking client-side tool calls.

        Called by the workflow before the first run_turn(). Executors
        that handle tools internally (e.g. ClaudeSDKExecutor) use this
        callback in their @tool handlers to park for client delivery.
        Executors that yield ToolCallRequested ignore it.

        :param callback: Blocks until client delivers the result.
            Signature: ``(call_id, tool_name, arguments) -> result_string``.
        """
```

The workflow creates this callback (encapsulating task_store, SSE, polling)
and calls `executor.set_client_tool_callback(cb)` before the loop.

**GAP C: Compaction is irrelevant for internal executors.**
The contract says "compaction is the workflow's responsibility" and the workflow
calls `_proactive_compact_if_needed()` before every `run_turn()`. But when
`max_context_tokens()` returns None:

- Proactive compaction has no budget → skipped (good).
- If the SDK hits overflow internally, it compacts internally → workflow never
  sees `ContextWindowExceeded` (good).
- The `messages` parameter is only used for prompt building, not as the SDK's
  context window (good — no stale compacted messages).

**No contract change needed.** `max_context_tokens() -> None` already signals
"don't compact for me." Document this explicitly: when `max_context_tokens()`
returns None, the workflow skips both proactive and reactive compaction for
that executor.

**GAP D: ToolCallObserved.arguments is empty for SDK-observed tools.**
The SDK stream gives us `content_block_start` with tool_use block containing
the tool name and ID, and later `content_block_delta` with `input_json_delta`
for arguments. The `UserMessage` with `ToolResultBlock` echoes back
`tool_use_id` but NOT the original arguments.

To populate `ToolCallObserved.arguments`, we'd need to buffer
`input_json_delta` events and parse them per tool_use_id. This is doable but
adds complexity.

**Recommendation**: Buffer `input_json_delta` in the executor's stream consumer.
No contract change needed — `arguments: dict[str, Any]` is already on
`ToolCallObserved`. The implementation just needs to accumulate them.

**GAP E: `_write_output` for streaming text.**
The executor yields `TextChunk` events, and the workflow writes them to SSE.
But the existing OmniAgents ClaudeSDKExecutor also calls `_write_output`
directly for streaming deltas. In the new model, the workflow's event consumer
loop handles this — the executor ONLY yields events, never calls SSE directly.

**No contract change needed.** This is a clean separation — executor yields,
workflow streams. Confirmed correct.

---

### Implementation 2a: OmniAgentsExecutorAdapter (LLM backend swap)

This wraps an OmniAgents Executor (not Session) as an agent-plane Executor.
Agent-plane owns history, tools, compaction. The OmniAgents Executor is just
the LLM backend.

```python
class OmniAgentsExecutorAdapter(Executor):
    """
    Wraps an OmniAgents Executor as an agent-plane Executor.

    This is the "LLM backend swap" use case: agent-plane owns all
    state (history, tools, compaction, steering). The OmniAgents
    Executor just makes LLM calls and returns events.

    Since OmniAgents executors that don't handle tools internally
    yield ``ToolCallRequest`` events, this adapter maps them to
    ``ToolCallRequested`` for the workflow to execute. For SDK-backed
    OmniAgents executors, ``ToolCallRequest``+``ToolCallComplete``
    pairs map to ``ToolCallObserved``.

    :param oa_executor: The OmniAgents Executor instance.
    """

    def __init__(self, oa_executor: oa_Executor) -> None:
        self._executor = oa_executor

    def max_context_tokens(self) -> int | None:
        """
        Delegate to the OmniAgents executor.

        :returns: Token limit or None.
        """
        return self._executor.max_context_tokens()

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
    ) -> Iterator[ExecutorEvent]:
        """
        Bridge OmniAgents async executor into agent-plane's sync contract.

        Maps OmniAgents event types to agent-plane event types:

        - ``oa.TextChunk`` → ``TextChunk``
        - ``oa.ToolCallRequest`` → ``ToolCallRequested`` (non-SDK executors)
        - ``oa.TurnComplete`` → ``TurnComplete``
        - ``oa.ExecutorError`` → ``ExecutorError``

        For SDK-backed OmniAgents executors (handles_tools_internally=True),
        ``ToolCallRequest``+``ToolCallComplete`` pairs are buffered and emitted
        as ``ToolCallObserved``.

        :param messages: Responses API input items (history).
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: System instructions.
        :param config: Model and sampling config.
        :param task_id: Task identifier.
        :param conversation_id: Conversation identifier.
        """
        handles_internal = self._executor.handles_tools_internally()

        # Map agent-plane config to OmniAgents config
        oa_config = oa_ExecutorConfig(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        q: queue.Queue[ExecutorEvent | None] = queue.Queue(maxsize=256)

        async def _drain() -> None:
            # Buffer for correlating ToolCallRequest → ToolCallComplete
            pending_calls: dict[str, oa_ToolCallRequest] = {}
            # OmniAgents ToolCallRequest has no call_id — use sequential counter
            call_counter = 0

            async for event in self._executor.run_turn(
                messages, tools, system_prompt, oa_config,
            ):
                if isinstance(event, oa_TextChunk):
                    q.put(TextChunk(text=event.text))

                elif isinstance(event, oa_ToolCallRequest):
                    if handles_internal:
                        # Buffer — wait for matching ToolCallComplete
                        synthetic_id = f"oa_call_{call_counter}"
                        call_counter += 1
                        pending_calls[synthetic_id] = event
                        # GAP F: no call_id on OmniAgents events
                    else:
                        # External — workflow executes
                        synthetic_id = f"oa_call_{call_counter}"
                        call_counter += 1
                        q.put(ToolCallRequested(
                            call_id=synthetic_id,
                            name=event.name,
                            arguments=event.args,
                        ))

                elif isinstance(event, oa_ToolCallComplete):
                    if handles_internal and pending_calls:
                        # Match to the oldest pending request (FIFO —
                        # OmniAgents executes tools sequentially)
                        syn_id, request = next(iter(pending_calls.items()))
                        del pending_calls[syn_id]
                        q.put(ToolCallObserved(
                            call_id=syn_id,
                            name=event.name,
                            arguments=request.args,
                            result=str(event.result) if event.result else "",
                            status=event.status.value,
                            duration_ms=event.duration_ms,
                        ))
                    # else: ToolCallComplete from Session (non-SDK) — skip,
                    # workflow already executed via ToolCallRequested

                elif isinstance(event, oa_TurnComplete):
                    q.put(TurnComplete(text=event.response or None))

                elif isinstance(event, oa_ExecutorError):
                    q.put(ExecutorError(message=event.message))

            if not any(isinstance(e, (TurnComplete, ExecutorError))
                       for e in list(q.queue)):
                q.put(TurnComplete(text=None))

        def _run():
            asyncio.run(_drain())
            q.put(None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        while True:
            event = q.get()
            if event is None:
                break
            yield event
        thread.join()
```

#### Gaps exposed by OmniAgentsExecutorAdapter

**GAP F: OmniAgents events have no call_id.**
`ToolCallRequest` has `name`, `args`, `metadata` but no `call_id`.
`ToolCallComplete` has `name`, `status`, `result`, `error`, `duration_ms` but
no `call_id`. Agent-plane requires `call_id` for correlating function_call
and function_call_output items in conv_store.

For non-SDK executors (workflow executes tools), synthetic call_ids work fine —
the workflow generates them and uses them consistently within one turn.

For SDK executors (handles_tools_internally), synthetic call_ids work for a
single turn but break on crash recovery: the re-invoked turn generates
different synthetic IDs, causing duplicate conv_store entries. Fix: idempotency
check in `_persist_and_stream` (same as `_maybe_persist_compaction_item`).

**No contract change needed** — this is an implementation concern in the
adapter and the workflow's persist logic.

**GAP G: OmniAgents config has fewer fields.**
`oa.ExecutorConfig` has `model`, `temperature`, `max_tokens`, `extra`.
Agent-plane's `ExecutorConfig` adds `timeout`, `retry`, `reasoning`,
`connection`. The adapter maps the common fields and drops the rest.

For most OmniAgents executors, `timeout` and `retry` are handled by the
executor internally (e.g. DatabricksExecutor has its own retry logic).
`reasoning` is Claude-specific. `connection` is agent-plane-specific.

**No contract change needed** — the adapter maps what it can and ignores the
rest. OmniAgents executors' `extra: dict` can carry overflow.

**GAP H: ContextWindowExceeded not in OmniAgents event set.**
OmniAgents executors that hit context overflow emit `ExecutorError`, not a
structured overflow event. Agent-plane's reactive compaction logic can't
activate because there are no `max_tokens`/`actual_tokens` values.

Options:
1. Parse the error message for token counts (fragile).
2. Rely on proactive compaction only — use `max_context_tokens()` to set
   a budget and compact before calling `run_turn()`. Never expect reactive.
3. Add `ContextWindowExceeded` to OmniAgents (requires upstream change).

**Recommendation: Option 2 for now.** Proactive compaction handles the common
case. If the executor reports `max_context_tokens()`, the workflow can keep
messages within budget. Reactive compaction is a safety net that won't fire
for OmniAgents executors — acceptable given that proactive should prevent
overflow.

---

### Implementation 2b: RemoteExecutor (remote agent service)

Instead of wrapping an OmniAgents Session in-process (which creates dual
history, crash recovery tension, and ignored parameters), the agent runs
as a standalone REST service. Agent-plane talks to it over HTTP using a
standardized event protocol. Any agent framework — OmniAgents, custom,
any language — can implement the endpoint.

#### The REST protocol

**Endpoint**: `POST /v1/turns`

**Normal request** (remote service has a live session):

```http
POST /v1/turns HTTP/1.1
Content-Type: application/json
Accept: text/event-stream

{
  "conversation_id": "conv_abc123",
  "new_messages": [
    {"role": "user", "content": "What files are in the repo?"}
  ]
}
```

**Response** (SSE stream of executor events):

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"type": "text_chunk", "text": "Let me "}
data: {"type": "text_chunk", "text": "check..."}
data: {"type": "tool_call_observed", "call_id": "call_1", "name": "Bash", "arguments": {"command": "ls"}, "result": "README.md\nsrc/", "status": "success", "duration_ms": 42.0}
data: {"type": "text_chunk", "text": "I found two entries."}
data: {"type": "turn_complete", "text": "I found two entries: README.md and src/."}
```

**Session not found** (remote service crashed/restarted, TTL expired):

```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{"error": "session_not_found", "conversation_id": "conv_abc123"}
```

**Recovery request** (agent-plane retries with full history):

```http
POST /v1/turns HTTP/1.1
Content-Type: application/json
Accept: text/event-stream

{
  "conversation_id": "conv_abc123",
  "new_messages": [
    {"role": "user", "content": "Now show me the README"}
  ],
  "history": [
    {"role": "user", "content": "What files are in the repo?"},
    {"role": "assistant", "content": "I found two entries: README.md and src/."},
    {"role": "assistant", "content": "[tool_call: Bash({\"command\": \"ls\"})]"},
    {"role": "tool", "content": "README.md\nsrc/", "name": "Bash"}
  ]
}
```

The remote service creates a new session from `history` and processes
`new_messages`. The response is the same SSE event stream.

#### Request schema

```
POST /v1/turns

{
  "conversation_id": string,       // Required. Keys the remote session.
  "new_messages": [                // Required. The new message(s) to process.
    {
      "role": "user" | "assistant" | "tool",
      "content": string,
      "name": string | null        // Tool name (for role=tool only)
    }
  ],
  "history": [                     // Optional. Full prior conversation.
    ...                            //   Sent only on recovery (after 404).
  ]                                //   Omitted on normal requests.
}
```

#### SSE event schema

Every SSE `data:` line is a JSON object with a `type` field. The event
types map 1:1 to the Python `ExecutorEvent` types:

```
{"type": "text_chunk",          "text": "Hello"}
{"type": "tool_call_requested", "call_id": "c1", "name": "search", "arguments": {"q": "test"}}
{"type": "tool_call_observed",  "call_id": "c1", "name": "Bash", "arguments": {...}, "result": "...", "status": "success", "duration_ms": 42.0}
{"type": "turn_complete",       "text": "Here is the answer." | null}
{"type": "context_exceeded",    "max_tokens": 128000, "actual_tokens": 131072}
{"type": "error",               "message": "...", "code": "auth_failed" | null}
```

**`tool_call_requested`**: The remote agent wants agent-plane to execute a
tool (e.g. a client-side tool). The turn ends. Agent-plane executes the tool
(or parks for client delivery), then calls `POST /v1/turns` again with the
tool result in `new_messages`:

```json
{
  "conversation_id": "conv_abc123",
  "new_messages": [
    {"role": "tool", "content": "file contents...", "name": "Read", "call_id": "c1"}
  ]
}
```

**`tool_call_observed`**: The remote agent already executed the tool. Agent-plane
persists both sides and streams to the client. No action needed.

**Terminal events**: `turn_complete`, `error`. Exactly one per SSE stream.
The connection closes after the terminal event.

#### Recovery handshake

```
Agent-plane                          Remote service
    |                                      |
    |  POST /v1/turns                      |
    |  {conversation_id, new_messages}     |
    |------------------------------------->|
    |                                      |
    |  (service has session)               |
    |  200 + SSE stream                    |
    |<-------------------------------------|
    |                                      |

    --- OR (service lost session) ---

    |  POST /v1/turns                      |
    |  {conversation_id, new_messages}     |
    |------------------------------------->|
    |                                      |
    |  404 session_not_found               |
    |<-------------------------------------|
    |                                      |
    |  POST /v1/turns                      |
    |  {conversation_id, new_messages,     |
    |   history: [...from conv_store...]}  |
    |------------------------------------->|
    |                                      |
    |  (service rebuilds from history)     |
    |  200 + SSE stream                    |
    |<-------------------------------------|
```

No dedup. The remote service always knows exactly what's new (`new_messages`)
and optionally gets full context (`history`) only when it needs it.

#### Python `RemoteExecutor`

```python
class RemoteExecutor(Executor):
    """
    Executor that delegates to a remote agent service over HTTP.

    The remote service manages its own agent loop, tools, prompt, and
    session state. Agent-plane sends messages and observes the event
    stream for persistence, SSE relay, and durability.

    Recovery: if the remote service returns 404 (session not found),
    the executor retries with full conversation history from conv_store
    so the remote service can rebuild.

    :param endpoint: URL of the remote ``POST /v1/turns`` endpoint,
        e.g. ``"https://my-agent:8000/v1/turns"``.
    :param timeout: HTTP request timeout in seconds, e.g. ``300``.
    :param headers: Optional extra HTTP headers (auth tokens, etc.),
        e.g. ``{"Authorization": "Bearer ..."}``.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        timeout: int = 300,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._timeout = timeout
        self._headers = headers or {}

    def max_context_tokens(self) -> int | None:
        """
        Remote service manages its own context.

        :returns: None.
        """
        return None

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
    ) -> Iterator[ExecutorEvent]:
        """
        POST to the remote service and yield events from the SSE stream.

        On 404 (session not found), retries with full history so the
        remote service can rebuild session state.

        :param messages: Full conversation history from conv_store.
            Used to extract new_messages (latest) and history (all
            prior) for the recovery handshake.
        :param tools: Ignored — remote service defines its own tools.
        :param system_prompt: Ignored — remote service defines its own prompt.
        :param config: Ignored — remote service defines its own config.
        :param task_id: Task identifier (not sent to remote service).
        :param conversation_id: Conversation identifier — keys the
            remote session.
        """
        new_messages = _extract_new_messages(messages)

        # First attempt: send only new messages
        body: dict[str, Any] = {
            "conversation_id": conversation_id,
            "new_messages": new_messages,
        }
        response = self._post(body)

        if response.status_code == 404:
            # Recovery: remote service lost session, send full history
            history = _messages_to_history(messages)
            body["history"] = history
            response = self._post(body)

        if response.status_code != 200:
            yield ExecutorError(
                message=f"Remote executor returned {response.status_code}",
                code="remote_error",
            )
            return

        # Consume SSE stream
        yield from self._consume_sse(response)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        """
        POST to the remote endpoint with SSE streaming.

        :param body: JSON request body.
        :returns: httpx.Response with streaming enabled.
        """
        return httpx.post(
            self._endpoint,
            json=body,
            headers={
                "Accept": "text/event-stream",
                **self._headers,
            },
            timeout=self._timeout,
            # stream=True for SSE consumption
        )

    def _consume_sse(
        self, response: httpx.Response,
    ) -> Iterator[ExecutorEvent]:
        """
        Parse SSE lines from the response and yield ExecutorEvents.

        :param response: Streaming HTTP response.
        :yields: ExecutorEvent instances parsed from SSE data lines.
        """
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            event_type = payload.get("type")

            if event_type == "text_chunk":
                yield TextChunk(text=payload["text"])

            elif event_type == "tool_call_requested":
                yield ToolCallRequested(
                    call_id=payload["call_id"],
                    name=payload["name"],
                    arguments=payload["arguments"],
                )

            elif event_type == "tool_call_observed":
                yield ToolCallObserved(
                    call_id=payload["call_id"],
                    name=payload["name"],
                    arguments=payload["arguments"],
                    result=payload["result"],
                    status=payload["status"],
                    duration_ms=payload["duration_ms"],
                )

            elif event_type == "turn_complete":
                yield TurnComplete(text=payload.get("text"))
                return  # terminal

            elif event_type == "context_exceeded":
                yield ContextWindowExceeded(
                    max_tokens=payload["max_tokens"],
                    actual_tokens=payload["actual_tokens"],
                )
                return  # terminal

            elif event_type == "error":
                yield ExecutorError(
                    message=payload["message"],
                    code=payload.get("code"),
                )
                return  # terminal
```

#### Agent spec for remote executors

```yaml
name: my-omni-agent
llm:
  executor: remote
  endpoint: https://my-agent-service:8000/v1/turns
  # Optional auth:
  # headers:
  #   Authorization: "Bearer ${AGENT_TOKEN}"
execution:
  timeout: 300
```

When `executor: remote`, `_create_executor()` builds a `RemoteExecutor`.
The `llm.model`, `instructions`, and `tools` fields are not needed — the
remote service defines all of these.

#### Why this replaces the in-process OmniAgentsSessionWrapper

The in-process approach (wrapping OmniAgents Session directly) had
fundamental gaps:

- **Gap I (tools ignored)**: Session defines its own tools. ✅ Not a problem
  for RemoteExecutor — the remote service owns its tools. Agent-plane doesn't
  pass them.
- **Gap J (system_prompt ignored)**: Session uses AgentDef.prompt. ✅ Not a
  problem — the remote service owns its prompt. Agent-plane doesn't pass it.
- **Gap K (config ignored)**: Session uses its own config. ✅ Same — remote
  service owns its config.
- **Gap L (dual history / crash recovery)**: Session History is in-memory,
  lost on crash. ✅ Solved by the recovery handshake: agent-plane sends
  `history` from conv_store on 404, remote service rebuilds. No dedup, no
  state injection, no serialization.
- **Gap M (steering)**: ✅ Works — steered messages become `new_messages`
  in the next `POST /v1/turns` request.
- **Gap N (compaction)**: ✅ Remote service manages its own context. If it
  hits overflow, it can emit `context_exceeded` and agent-plane compacts
  the `history` before retrying. Or the remote service handles it internally.

The in-process OmniAgentsSessionWrapper is retained in this doc as analysis
but is **not the recommended approach**. Use RemoteExecutor for deploying
external agents (OmniAgents or otherwise) on agent-plane.

#### What the remote service implementor provides

Any HTTP service that implements `POST /v1/turns` with the request/response
schema above. For OmniAgents, this is a thin adapter on the existing
`create_app()` server — add one endpoint that speaks this SSE protocol
instead of the Vercel AI SDK stream format.

For other frameworks: implement the endpoint in any language. The contract
is HTTP + SSE + a fixed JSON event schema. No Python dependency, no import
coupling, no shared types.

---

## Summary of gaps and proposed changes

| Gap | Severity | Proposed resolution |
|-----|----------|---------------------|
| **A: Persistent storage** | ✅ Resolved | `storage_dir: Path` passed to `on_task_start`/`on_task_end` — workflow manages artifact I/O |
| **B: Client-side tool parking** | ✅ Resolved | SDK connects to agent-plane's MCP bridge — parking handled by existing infra |
| **C: Compaction irrelevant for internal executors** | Low | Document: `max_context_tokens() -> None` means "skip compaction" |
| **D: Empty arguments in ToolCallObserved** | Low | Buffer `input_json_delta` in executor — no contract change |
| **E: SSE streaming** | None | Already clean — executor yields, workflow streams |
| **F: No call_id in OmniAgents events** | Medium | Synthetic IDs + idempotent persist — no contract change |
| **G: Config field mismatch** | Low | Adapter maps common fields, drops rest — no contract change |
| **H: No ContextWindowExceeded in OmniAgents** | Medium | Rely on proactive compaction for OmniAgents executors |
| **I: tools parameter ignored** | ✅ Resolved | RemoteExecutor doesn't pass tools — remote service owns them |
| **J: system_prompt ignored** | ✅ Resolved | RemoteExecutor doesn't pass prompt — remote service owns it |
| **K: config partially ignored** | ✅ Resolved | RemoteExecutor doesn't pass config — remote service owns it |
| **L: Dual history / crash recovery** | ✅ Resolved | Recovery handshake: 404 → retry with `history` from conv_store |
| **M: Steering** | ✅ Resolved | Steered messages sent as `new_messages` in next request |
| **N: No compaction for remote executors** | Low | Remote service manages own context; can emit `context_exceeded` |

### Proposed ABC changes

```python
class Executor(abc.ABC):
    @abc.abstractmethod
    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        task_id: str,
        conversation_id: str,
    ) -> Iterator[ExecutorEvent]:
        """Unchanged."""
        ...

    def on_task_start(
        self, task_id: str, conversation_id: str, storage_dir: Path,
    ) -> None:
        """
        Called once at task start, after storage_dir has been restored.

        The workflow restores the scoped persistent directory from the
        artifact store before calling this hook. The executor can read
        prior session state from storage_dir immediately.

        :param task_id: Task identifier, e.g. ``"task_abc123"``.
        :param conversation_id: Conversation identifier,
            e.g. ``"conv_abc123"``.
        :param storage_dir: Scoped persistent directory for this
            conversation. Contents survive across tasks — the workflow
            manages persistence. The executor reads/writes freely
            within this directory but has no access outside it.
        """

    def on_task_end(
        self, task_id: str, conversation_id: str, storage_dir: Path,
    ) -> None:
        """
        Called once at task end. Writes to storage_dir will be persisted.

        The workflow persists the directory to artifact store after this
        hook returns. Use for cleanup (e.g. disconnecting SDK client)
        while preserving any state written to storage_dir.

        :param task_id: Task identifier, e.g. ``"task_abc123"``.
        :param conversation_id: Conversation identifier,
            e.g. ``"conv_abc123"``.
        :param storage_dir: Same scoped directory from on_task_start.
        """

    def set_client_tool_callback(
        self,
        callback: Callable[[str, str, dict[str, Any]], str],
    ) -> None:
        """
        Register a callback for parking client-side tool calls.

        Called by the workflow before the first run_turn(). Internal
        executors (Claude SDK, OmniAgents Session) call this from
        their @tool handlers to park for client delivery. External
        executors (DefaultExecutor) ignore it.

        The callback signature: ``(call_id, tool_name, arguments) -> result``.
        It blocks until the client delivers the result via PATCH.

        :param callback: Blocking callable that parks and waits for
            client tool result delivery.
        """

    def max_context_tokens(self) -> int | None:
        """
        Context window limit, or None if managed internally.

        When None, the workflow skips both proactive and reactive
        compaction for this executor. The executor is responsible
        for its own context management.

        :returns: Token limit (e.g. ``128000``) or None.
        """
        return None
```

### Compaction ownership: resolving the contradiction

The inline comment flagged a real conflict:

- **EXECUTOR_CONTRACT.md**: "compaction is the workflow's responsibility"
- **CODINGAGENTS.md**: "Claude's compaction is better — delegate if possible"

Resolution: **both are correct, for different executors.**

- **External executors** (DefaultExecutor, OmniAgentsExecutorAdapter): compaction
  IS the workflow's responsibility. These executors make single LLM calls. The
  workflow compacts before calling `run_turn()`.

- **Internal executors** (ClaudeSDKExecutor, OmniAgentsSessionWrapper): compaction
  is the executor's (or Session's) responsibility. `max_context_tokens() -> None`
  tells the workflow to skip compaction entirely.

The contract supports both modes via `max_context_tokens()`. No code branching
needed — the workflow checks the return value and either compacts or doesn't.
Document this dual-path clearly:

> When `max_context_tokens()` returns an int, the workflow owns compaction
> (proactive via tiktoken estimate, reactive via `ContextWindowExceeded` event).
> When it returns None, the executor owns compaction internally and the workflow
> passes messages through without modification.
