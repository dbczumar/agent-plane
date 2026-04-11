# SDK Executor Failure Handling

## Context

The `ClaudeAgentsExecutor` and `AgentsSdkExecutor` have three gaps in failure
handling relative to the `DefaultExecutor`:

1. **No timeout propagation.** The agent spec declares `llm.request_timeout`,
   `llm.retry`, and `executor.timeout`, but SDK executors ignore them. The
   Claude executor passes only the model name; the OpenAI executor passes
   `request_timeout` and `max_retries` to the `AsyncOpenAI` client but
   ignores backoff, status codes, and the execution deadline.

2. **No error classification.** Both SDK executors catch `Exception` and
   emit a single `ExecutorError`. The workflow treats every `ExecutorError`
   as `PermanentLLMError` (workflow.py:1126). A transient failure — rate
   limit, subprocess OOM, network blip — kills the task permanently.

3. **No hang detection.** If the SDK subprocess hangs or the stream stalls,
   the workflow blocks forever. The execution timeout check
   (workflow.py:3884) only runs between loop iterations, so it never fires
   while waiting for a stuck SDK call.

### Current behavior by executor

| Aspect | Default | Claude SDK | OpenAI SDK |
|--------|---------|------------|------------|
| `llm.request_timeout` | Passed to httpx | **Ignored** | Passed to `AsyncOpenAI` |
| `llm.retry.max_attempts` | Used by `execute_with_retry_async` | **Ignored** (SDK retries internally, uncontrolled) | Passed as `max_retries` |
| `llm.retry.backoff_*` | Used | **Ignored** | **Ignored** |
| `llm.retry.status_codes` | Used | **Ignored** | **Ignored** |
| `executor.timeout` | Checked between iterations | **Not enforced during SDK call** | **Not enforced during SDK call** |
| Transient error → retry | Yes (classified by `classify_llm_error`) | No (all errors permanent) | No (all errors permanent) |
| SDK hangs | Partial (httpx timeout) | **Stuck forever** | **Stuck forever** |

---

## Design

Three changes, each independent and deployable separately:

### 1. Execution deadline enforcement via `asyncio.wait_for`

**Problem.** The execution timeout check at the top of each loop iteration
cannot fire while the current `_executor_turn_with_compaction()` call blocks
inside an SDK. If the SDK hangs, the task is stuck in `IN_PROGRESS` forever.

**Solution.** Wrap the executor turn call in `asyncio.wait_for()` with the
remaining execution budget as the timeout.

```
remaining = execution_timeout - (monotonic() - start_time)
```

If the budget is exhausted, `asyncio.wait_for` raises `asyncio.TimeoutError`.
The agent loop catches it and produces the same `INCOMPLETE` result with
`reason: "execution_timeout"` that the between-iteration check produces today.

#### Changes

**`workflow.py` — `_run_agent_loop`** (around line 3907):

```python
remaining = execution_timeout - (_monotonic() - start_time)
if remaining <= 0:
    return _handle_execution_timeout(task_id, output_items, execution_timeout)

try:
    llm_resp = await asyncio.wait_for(
        _executor_turn_with_compaction(
            task_id, executor, spec, llm_config, history,
            instructions, tool_schemas, compaction_state,
            executor_context, content_cache,
        ),
        timeout=remaining,
    )
except asyncio.TimeoutError:
    return _handle_execution_timeout(task_id, output_items, execution_timeout)
```

The existing between-iteration check (line 3884) stays as a fast path for
executors that return promptly — it avoids the `asyncio.wait_for` overhead
when the budget is already exhausted before the call starts.

**`executors/claude.py` — `run_turn`** (line 517):

The consumer loop `await asyncio.to_thread(event_queue.get)` also needs a
bounded wait. When `asyncio.wait_for` cancels the outer coroutine, the
`_async_turn` coroutine running in the per-conversation loop must be
cancelled too, and the `event_queue.get` must not block forever.

Add a timeout to the queue get:

```python
while True:
    try:
        event = await asyncio.to_thread(event_queue.get, timeout=5.0)
    except Empty:
        # Check if the SDK coroutine is still alive; if not, break.
        if future.done():
            break
        continue
    if event is None:
        break
    yield event
```

And capture the `Future` from `run_coroutine_threadsafe` so we can cancel
the SDK coroutine on timeout:

```python
future = asyncio.run_coroutine_threadsafe(_async_turn(...), loop)
```

When the outer `asyncio.wait_for` cancels `run_turn`, the `finally` block
cancels the future:

```python
try:
    while True:
        ...
finally:
    future.cancel()
```

#### Edge cases

- **Cancellation during tool parking.** If a client-side tool call is
  parked (waiting for client to deliver results via PATCH), the
  `asyncio.wait_for` cancellation propagates through `call_tool` →
  `dbos_recv_async`. The parked sub-agent's workflow is cancelled by DBOS.
  The pending tool call row remains in the DB but the task status becomes
  `INCOMPLETE`. The client sees the incomplete status on next poll.

- **Cancellation during streaming.** If the SDK is mid-stream (text deltas
  arriving), cancellation closes the async generator. Events already
  pushed to the queue are lost. The `finally` block in `_async_turn`
  still pushes `None` (sentinel), so the consumer exits cleanly.

- **Nested SDK process.** The Claude SDK runs as a subprocess. Cancelling
  the Python coroutine does not kill the subprocess. `_close_client_async`
  must be called to disconnect. Add `_close_client_async` to the
  cancellation/timeout cleanup path.

#### Why not a watchdog thread?

A watchdog thread that monitors elapsed time and forcibly cancels would
work but adds complexity (thread lifecycle, shutdown ordering). Since the
executor turn is an `await`-able coroutine, `asyncio.wait_for` is the
idiomatic and simpler mechanism. It composes with structured cancellation
and does not require additional threads.

---

### 2. Retryable executor errors

**Problem.** `_events_to_response_dict` (workflow.py:1125) raises
`PermanentLLMError` for every `ExecutorError`. A rate-limited SDK call or
a crashed subprocess that could succeed on retry kills the task.

**Solution.** Add a `retryable: bool` field to `ExecutorError`. SDK
executors classify errors before emitting. The workflow retries retryable
errors with backoff.

#### Event type change

**`executors/base.py`:**

```python
@dataclass
class ExecutorError:
    """
    An executor failure.

    :param message: Human-readable error description.
    :param code: Machine-readable error code, e.g. ``"auth_failed"``.
    :param retryable: Whether the workflow should retry the turn.
        ``True`` for transient failures (rate limit, timeout, subprocess
        crash). ``False`` for permanent failures (auth, config).
    """
    message: str
    code: str | None = None
    retryable: bool = False
```

Adding a field with a default value is backwards-compatible — existing
code that constructs `ExecutorError(message=..., code=...)` without
`retryable` gets `retryable=False` (same as today's behavior).

#### Error classification in SDK executors

**`executors/claude.py` — `_async_turn` (line 603):**

```python
except Exception as exc:
    await _close_client_async(conv_id)
    retryable = _is_retryable_claude_error(exc)
    event_queue.put(ExecutorError(
        message=f"Claude SDK error: {exc}",
        code=type(exc).__name__,
        retryable=retryable,
    ))
```

**`_is_retryable_claude_error`:** Classifies by exception type and message.

| Signal | Retryable? | Rationale |
|--------|-----------|-----------|
| `SystemMessage` with `api_retry`, status 429 | Yes | Rate limited. |
| `SystemMessage` with `api_retry`, status 529 | Yes | Overloaded. |
| `SystemMessage` with `api_retry`, status 500/502/503 | Yes | Transient server error. |
| `SystemMessage` with `api_retry`, status 401/403 | No | Auth failure. |
| `SystemMessage` with `api_retry`, status 404 | No | Model not found. |
| Subprocess crash (BrokenPipeError, ConnectionResetError) | Yes | OOM, segfault — retry may succeed. |
| `OSError` / `ConnectionError` | Yes | Network-level failure. |
| `TimeoutError` / `asyncio.TimeoutError` | Yes | SDK call timed out. |
| Other `Exception` | No | Unknown — fail loud. |

Note: today `_check_terminal_error` handles 401/403/404 from `api_retry`
system messages and yields `ExecutorError` mid-stream. That path stays —
it catches permanent errors early (before the stream ends). The new
classification handles exceptions thrown by the SDK itself, which surface
as `Exception` in the `except` block rather than as stream events.

**`executors/agents_sdk.py` — `_stream_sdk_turn` (line 820):**

```python
except Exception as exc:
    cls_name = type(exc).__name__
    if cls_name == "MaxTurnsExceeded":
        yield TurnComplete(text=None)
    else:
        retryable = _is_retryable_agents_sdk_error(exc)
        yield ExecutorError(
            message=f"Agents SDK error: {exc}",
            code=cls_name,
            retryable=retryable,
        )
```

**`_is_retryable_agents_sdk_error`:** Similar heuristic — check for
`httpx.TimeoutException`, HTTP status codes on `httpx.HTTPStatusError`,
connection errors. The OpenAI SDK wraps errors in its own types
(`APITimeoutError`, `RateLimitError`, `APIConnectionError`) — match on
class name since the import may not be available.

#### Workflow retry logic

**`workflow.py` — `_events_to_response_dict` (line 1125):**

```python
elif isinstance(event, ExecutorError):
    if event.retryable:
        raise RetryableLLMError(
            event.message,
            code=event.code or "executor_error",
            detail=LLMErrorDetail(),
        )
    raise PermanentLLMError(
        event.message,
        code=event.code or "executor_error",
        detail=LLMErrorDetail(),
    )
```

**`workflow.py` — `_run_agent_loop`** (around the executor turn call):

SDK executors (`max_context_tokens() is None`) currently go through
`_consume_executor_live`, which has no retry. Add a retry wrapper for
the executor-managed path:

```python
if executor.max_context_tokens() is None:
    llm_resp = await _executor_turn_with_retry(
        task_id, executor, messages, tools,
        system_prompt, llm_config, context,
        retry_config=llm_config.retry,
    )
else:
    # Existing path — retry is inside the @step.
    llm_resp = await _executor_turn_with_compaction(...)
```

`_executor_turn_with_retry` wraps `_consume_executor_live` with the same
`execute_with_retry_async` helper the default executor uses, respecting
`llm_config.retry` (max_attempts, backoff_base, backoff_max). Only
`RetryableLLMError` triggers a retry; `PermanentLLMError` propagates
immediately.

Between retries the Claude executor must call `_close_client_async` so the
broken subprocess is torn down. The next retry calls `_get_or_create_client`
which spins up a fresh subprocess.

#### Max retry budget vs. execution deadline

Retries must respect the execution deadline. If the remaining execution
budget is less than `backoff_base`, skip the retry and fail immediately.
The `asyncio.wait_for` wrapper from change 1 enforces this structurally —
if a retry attempt exceeds the deadline, it is cancelled.

---

### 3. Config propagation to SDK executors

**Problem.** SDK executors ignore `llm.request_timeout`, `llm.retry`,
and `executor.timeout` from config.yaml. Agent authors cannot control SDK
resilience behavior.

**Solution.** Pass config values into each SDK's client/runner construction.

#### Claude SDK executor

The Claude Agent SDK (`ClaudeAgentOptions`) does **not** expose
`request_timeout`, retry count, backoff, or status code configuration.
The full `ClaudeAgentOptions` dataclass (from `claude_agent_sdk/types.py`)
has execution-control fields (`max_turns`, `max_budget_usd`) but nothing
for per-API-call resilience. The SDK manages its own retry loop
internally with a hardcoded ~10-minute API timeout (`API_TIMEOUT_MS:
600000`), surfacing retries as `SystemMessage(subtype="api_retry")`
events. There are open issues about this being non-configurable
(anthropics/claude-agent-sdk-python#533, #701).

Available `ClaudeAgentOptions` fields relevant to execution limits:

| Field | Type | Effect |
|---|---|---|
| `max_turns` | `int \| None` | Cap on agentic turns (tool-use round trips). |
| `max_budget_usd` | `float \| None` | Spend ceiling in USD for the session. |
| `fallback_model` | `str \| None` | Automatic failover model. |

None of these control per-request timeout or retry behavior.

Since we cannot configure the SDK's internal retry/timeout behavior,
the approach is external enforcement:

- **`executor.timeout`**: Enforced by the `asyncio.wait_for` wrapper
  (change 1). No SDK-side config needed.
- **`llm.retry`**: Enforced by the workflow-level retry wrapper
  (change 2). When the SDK fails (exception or terminal `api_retry`
  system message), the workflow retries the entire `run_turn` call
  according to `llm.retry`.
- **`llm.request_timeout`**: Cannot be injected into the SDK today.
  The SDK's internal `API_TIMEOUT_MS` (600s / 10 minutes) is the
  de facto per-request timeout. Document this as a known limitation.
  The execution deadline (`executor.timeout`) provides an upper bound.
  If the Claude Agent SDK adds a `request_timeout` option in the
  future, pass `llm.request_timeout` through.
- **`max_turns`**: Wire `executor.max_iterations` to
  `ClaudeAgentOptions.max_turns` so the agent spec controls the
  SDK's internal iteration cap. Today it is unset (unlimited).

##### Interaction with SDK internal retries

The Claude SDK retries API errors internally before surfacing them as
`SystemMessage(subtype="api_retry")` or as exceptions. This creates a
potential double-retry: the SDK retries internally, exhausts its attempts,
raises an exception, and then the workflow retries the entire turn.

This is acceptable because:

1. The two retry layers operate at different granularities. The SDK retries
   a single API call. The workflow retries the entire turn, which includes
   rebuilding prompts and re-establishing subprocess state.
2. The execution deadline (change 1) caps the total time regardless of how
   many retries happen at either layer.
3. The alternative — disabling SDK internal retries — is not possible via
   the current SDK API.

#### OpenAI Agents SDK executor

The OpenAI SDK exposes timeout and retry on the `AsyncOpenAI` client,
which is already partially wired:

```python
# agents_sdk.py:163 — already passes timeout and max_retries
AsyncOpenAI(
    timeout=float(timeout),
    max_retries=max_retries,
)
```

Missing pieces:

- **`executor.timeout`**: Not enforced. Add the `asyncio.wait_for`
  wrapper (change 1) — same mechanism as Claude.
- **`llm.retry.backoff_*`**: The `AsyncOpenAI` client uses its own
  backoff defaults. The openai SDK does not expose backoff configuration
  on the client constructor. Document as a known limitation.
- **`llm.retry.status_codes`**: Same — the `AsyncOpenAI` client retries
  429/500/502/503/504 unconditionally. Not configurable. Document.
- **Workflow-level retry**: The same `_executor_turn_with_retry` wrapper
  (change 2) applies. When the Agents SDK raises (e.g., `MaxTurnsExceeded`
  is not retried, but `APIConnectionError` is), the workflow can retry the
  full turn.

#### Summary of what flows where

| Config field | Claude SDK | OpenAI SDK |
|---|---|---|
| `executor.timeout` | `asyncio.wait_for` (external) | `asyncio.wait_for` (external) |
| `executor.max_iterations` | `ClaudeAgentOptions.max_turns` (new) | Hardcoded `_SDK_MAX_TURNS=200` (wire to spec) |
| `llm.request_timeout` | Not injectable — SDK uses internal 600s `API_TIMEOUT_MS` | `AsyncOpenAI(timeout=...)` (already wired) |
| `llm.retry.max_attempts` | Workflow-level retry count | `AsyncOpenAI(max_retries=...)` + workflow-level retry |
| `llm.retry.backoff_*` | Workflow-level backoff | SDK internal (not configurable) + workflow-level backoff |
| `llm.retry.status_codes` | Workflow-level classification via `_is_retryable_claude_error` | SDK internal (not configurable) + workflow-level classification |

---

## Implementation Plan

### Phase 1: Execution deadline (change 1)

Files:
- `runtime/workflow.py` — wrap executor turn in `asyncio.wait_for`
- `runtime/executors/claude.py` — bounded `queue.get`, capture future,
  cancel + disconnect on timeout

Tests:
- Unit test: mock executor that hangs forever, verify task completes as
  `INCOMPLETE` after deadline
- Unit test: mock executor that returns normally, verify no interference
  from `asyncio.wait_for`
- Integration test: Claude executor with unreachable API endpoint, verify
  task does not hang

### Phase 2: Retryable errors (change 2)

Files:
- `runtime/executors/base.py` — add `retryable` field to `ExecutorError`
- `runtime/executors/claude.py` — add `_is_retryable_claude_error`,
  classify in `_async_turn` exception handler
- `runtime/executors/agents_sdk.py` — add `_is_retryable_agents_sdk_error`,
  classify in `_stream_sdk_turn`
- `runtime/workflow.py` — branch on `event.retryable` in
  `_events_to_response_dict`, add `_executor_turn_with_retry`

Tests:
- Unit test: `ExecutorError(retryable=True)` → `RetryableLLMError`
- Unit test: `ExecutorError(retryable=False)` → `PermanentLLMError`
- Unit test: `_is_retryable_claude_error` with various exception types
- Unit test: `_is_retryable_agents_sdk_error` with various exception types
- Integration test: mock SDK that fails twice then succeeds, verify
  workflow retries and completes

### Phase 3: Config propagation (change 3)

Files:
- `runtime/executors/claude.py` — pass `executor.max_iterations` as
  `ClaudeAgentOptions.max_turns`, document SDK limitation for
  `request_timeout`, wire workflow retry to `llm.retry`
- `runtime/executors/agents_sdk.py` — wire `executor.max_iterations`
  to `max_turns` (replacing hardcoded `_SDK_MAX_TURNS=200`), verify
  existing `timeout`/`max_retries` wiring, document backoff/status_codes
  limitations

Tests:
- Unit test: verify `llm.retry` config flows to
  `_executor_turn_with_retry` for SDK executors
- Unit test: verify `AsyncOpenAI` receives correct `timeout` and
  `max_retries` from spec
- Unit test: verify `max_turns` is set from `executor.max_iterations`
  for both Claude and OpenAI SDK executors

---

## Non-goals

- **Per-request timeout overrides on `POST /v1/responses`.** Out of scope.
  The existing TIMEOUTS.md design defers this.
- **Configuring SDK-internal retry behavior.** Neither the Claude Agent SDK
  nor the OpenAI Python SDK expose backoff or status code configuration.
  We accept the SDK defaults and layer workflow-level retry on top.
- **Health check endpoint for SDK clients.** Useful for observability but
  orthogonal to failure handling. A stuck SDK is detected by the deadline;
  a crashed SDK is detected by the exception.
- **Circuit breaker.** If every retry fails, the task fails. A circuit
  breaker that fast-fails subsequent tasks for the same provider is a
  separate concern (see OBSERVABILITY.md).
