# Timeouts and Retries

## Overview

Timeout and retry policies for LLM calls, tool calls, and overall agent execution. Policies are declared at three layers:

1. **Agent spec** — the agent author declares per-agent defaults in `config.yaml`.
2. **Per-request** — the caller can override on `POST /v1/responses`.
3. **Runtime caps** — the operator configures hard ceilings that clamp agent/request values.

Each layer narrows the window — a request cannot exceed the agent spec, and the agent spec's execution timeout cannot exceed the runtime cap. The runtime cap is a single value (execution timeout) that bounds total agent runtime.

---

## Retry — A Reusable Block

`retry` is a reusable configuration block that can appear under `llm`, under `tools` (as a global default for all tools), or on any individual tool. The shape is the same everywhere:

```yaml
retry:
  max_attempts: 3        # total attempts (1 = no retry)
  backoff_base: 2.0      # exponential backoff base (seconds)
  backoff_max: 30.0      # cap on backoff delay between retries (seconds)
  status_codes:          # HTTP status codes that trigger a retry
    - 429                # rate limit
    - 500                # server error
    - 502                # bad gateway
    - 503                # service unavailable
```

**Timeouts always trigger a retry** (if `max_attempts > 1`), regardless of `status_codes`. The `status_codes` field controls which HTTP error responses are retried on top of timeouts.

**`status_codes` applicability**: LLM calls and MCP server tools are HTTP-based, so `status_codes` applies. Local tools (Python/TypeScript) don't return HTTP status codes — for those, only timeout triggers a retry. Specifying `status_codes` on a local tool is ignored (not an error, just has no effect).

### Defaults

| Field | Default | Rationale |
|---|---|---|
| `max_attempts` | `3` | One original + two retries covers transient failures without excessive delay. |
| `backoff_base` | `2.0` | Standard exponential backoff starting point. |
| `backoff_max` | `30.0` | Prevents retries from spacing out too far. |
| `status_codes` | `[429, 500, 502, 503]` | Transient HTTP errors only. Auth/validation errors are never retried. |

---

## Agent Spec

### LLM

Single `timeout` (seconds) applies to both streaming and non-streaming calls. `retry` controls retry behavior for transient failures.

```yaml
llm:
  model: openai/gpt-5.4
  timeout: 300           # seconds — LLM call timeout (streaming and non-streaming)
  retry:
    max_attempts: 3
    backoff_base: 2.0
    backoff_max: 30.0
    status_codes: [429, 500, 502, 503]
```

### Tools

A global `timeout` and `retry` apply to all tool calls by default. Individual tools can override either or both by declaring them in their own config block.

```yaml
tools:
  timeout: 60            # seconds — default for all tool calls
  retry:
    max_attempts: 2      # default retry policy for all tools
    backoff_base: 1.0
    backoff_max: 10.0

  # Per-tool overrides — co-located with the tool's own config.
  # Omit timeout/retry to inherit the global tools default.
  mcp:
    github:
      transport: http
      url: https://mcp.example.com
      timeout: 120       # this MCP server gets 2 minutes
      retry:
        max_attempts: 5  # flaky server, retry more aggressively
        status_codes: [429, 500, 502, 503, 504]

  python:
    arxiv_search:
      timeout: 300       # slow external API
      # no retry block — inherits tools.retry

  agents:
    - name: summarizer
      timeout: 30        # sub-agent should be fast
      # no retry — inherits tools.retry
```

Resolution order for a given tool call:
1. Per-tool `timeout` / `retry` (if declared on the tool's config)
2. `tools.timeout` / `tools.retry` (global default)
3. (No runtime clamping — per-call timeouts are the agent author's concern)

### Execution

```yaml
execution:
  timeout: 3600          # seconds — wall-clock deadline for the entire agent loop
  max_iterations: 1000   # iteration cap (existing behavior, now configurable)
```

### Defaults

When the agent spec omits a field, these defaults apply:

| Field | Default | Rationale |
|---|---|---|
| `llm.timeout` | `300` | Generous for both streaming and non-streaming. Reasoning models can be slow. |
| `llm.retry` | `{max_attempts: 3, ...}` | See retry defaults table above. |
| `tools.timeout` | `60` | Tools should complete quickly. Long-running tools override per-tool. |
| `tools.retry` | `{max_attempts: 2, backoff_base: 1.0, backoff_max: 10.0}` | Conservative — tools fail faster than LLMs. |
| Per-tool `timeout` | Inherits `tools.timeout` | Co-located on the tool config. |
| Per-tool `retry` | Inherits `tools.retry` | Co-located on the tool config. |
| `execution.timeout` | `3600` | One hour wall-clock. Generous but finite. |
| `execution.max_iterations` | `1000` | Current hardcoded constant. |

### Spec Types

```python
@dataclass
class RetryConfig:
    """
    Retry policy. Reusable across LLM and tool configs.

    :param max_attempts: Total attempts including the first call,
        e.g. ``3`` means up to 2 retries.
    :param backoff_base: Base delay in seconds for exponential
        backoff, e.g. ``2.0``.
    :param backoff_max: Maximum delay between retries in seconds,
        e.g. ``30.0``.
    :param status_codes: HTTP status codes that trigger a retry.
        Timeouts always trigger a retry regardless of this list.
        Ignored for non-HTTP tools (local Python/TypeScript).
    """
    max_attempts: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 30.0
    status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503]
    )


# LLM retry defaults
_LLM_RETRY_DEFAULTS = RetryConfig(
    max_attempts=3, backoff_base=2.0, backoff_max=30.0,
)

# Tool retry defaults (more conservative)
_TOOL_RETRY_DEFAULTS = RetryConfig(
    max_attempts=2, backoff_base=1.0, backoff_max=10.0,
)


@dataclass
class ExecutionConfig:
    """
    Overall agent execution limits.

    :param timeout: Wall-clock deadline for the entire agent loop
        in seconds, e.g. ``3600``.
    :param max_iterations: Maximum agent loop iterations, e.g.
        ``1000``.
    """
    timeout: int = 3600
    max_iterations: int = 1000


# Updated LLMConfig
@dataclass
class LLMConfig:
    model: str
    extra: dict[str, Any] = field(default_factory=dict)
    connection: dict[str, str] | None = None
    timeout: int = 300
    retry: RetryConfig = field(default_factory=lambda: RetryConfig())


# Per-tool timeout and retry are co-located on the tool's own
# config. MCPServerConfig, LocalToolInfo, and sub-agent
# references each gain optional timeout and retry fields:

@dataclass
class MCPServerConfig:
    name: str
    transport: str
    # ... existing fields ...
    timeout: int | None = None     # None = inherit tools.timeout
    retry: RetryConfig | None = None  # None = inherit tools.retry


@dataclass
class LocalToolInfo:
    name: str
    path: str
    language: str
    timeout: int | None = None     # None = inherit tools.timeout
    retry: RetryConfig | None = None  # None = inherit tools.retry


@dataclass
class SubAgentRef:
    """
    A sub-agent reference with optional timeout and retry.

    :param name: Sub-agent name, e.g. ``"summarizer"``.
    :param timeout: Per-call timeout in seconds. ``None``
        means inherit ``tools.timeout``.
    :param retry: Per-tool retry policy. ``None`` means
        inherit ``tools.retry``.
    """
    name: str
    timeout: int | None = None
    retry: RetryConfig | None = None


# Updated ToolsConfig
@dataclass
class ToolsConfig:
    agents: list[SubAgentRef] = field(default_factory=list)
    timeout: int = 60
    retry: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_attempts=2, backoff_base=1.0, backoff_max=10.0,
        )
    )


# Updated AgentSpec (new field)
@dataclass
class AgentSpec:
    # ... existing fields ...
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
```

---

## Per-Request Overrides

Not in scope for the initial implementation. Future work: allow `POST /v1/responses` to accept a `timeouts` field that overrides the agent spec for that request, with execution timeout clamped by the runtime cap.

---

## Runtime Caps

The operator configures a single runtime-level ceiling: the maximum wall-clock time an agent can run. This is a hard upper bound — regardless of what the agent spec declares, the runtime clamps `execution.timeout` to this value. Per-call timeouts and retry policies are the agent author's concern, not the operator's.

### Configuration

```python
@dataclass
class RuntimeCaps:
    """
    Operator-configured hard ceiling for agent execution.

    :param execution_timeout: Max wall-clock time for the entire
        agent loop in seconds, e.g. ``7200``. Agent specs with
        a higher ``execution.timeout`` are clamped to this value.
    """
    execution_timeout: int = 7200
```

### Clamping

```python
def resolve_execution_timeout(
    spec: AgentSpec,
    caps: RuntimeCaps,
) -> int:
    """
    Return the effective execution timeout in seconds.

    Takes the minimum of the agent spec's declared timeout and
    the operator's cap.
    """
    return min(spec.execution.timeout, caps.execution_timeout)
```

LLM and tool timeout/retry configs are used as-is from the agent spec — the runtime does not cap them. The execution timeout is the single lever the operator pulls to bound total resource usage.

### How Caps Are Specified

Runtime caps can be configured through three surfaces: the YAML config file, `ap server` CLI flags, and the Python `init()` API. CLI flags take precedence over config file values, which take precedence over defaults.

#### 1. YAML Config File (`-c config.yaml`)

```yaml
# server-config.yaml
database_uri: sqlite:///agent_plane.db
artifact_location: ./artifacts

execution_timeout: 7200
```

#### 2. `ap server` CLI Flag

```bash
ap server --execution-timeout 7200
```

#### 3. Python API (`init()`)

For programmatic usage (embedding the runtime in a custom app):

```python
from agent_plane.runtime import init as init_runtime
from agent_plane.runtime.caps import RuntimeCaps

caps = RuntimeCaps(execution_timeout=7200)
init_runtime(
    conversation_store=conversation_store,
    task_store=task_store,
    agent_store=agent_store,
    agent_cache=agent_cache,
    caps=caps,  # None = use defaults
)
```

#### Resolution Order

1. CLI flag (if provided — not `None`)
2. Config file `execution_timeout` (if present)
3. `RuntimeCaps` default (`7200`)

In the CLI entrypoint:

```python
def server(host, port, database_uri, artifact_location, config_path,
           execution_timeout):
    cfg = _load_config(config_path)

    caps = RuntimeCaps(
        execution_timeout=execution_timeout or cfg.get("execution_timeout", 7200),
    )
    init_runtime(..., caps=caps)
```

### Default

| Cap | Default | Rationale |
|---|---|---|
| `execution_timeout` | `7200` (2 hrs) | Prevents runaway agents from consuming resources indefinitely. High enough that most agents never hit it. |

---

## Retry Behavior

### What Triggers a Retry

Timeouts **always** trigger a retry (if attempts remain). HTTP status codes listed in `status_codes` also trigger a retry. Everything else is a permanent failure.

| Error | Retry? | Rationale |
|---|---|---|
| Timeout (request/read) | Always | Transient network issue. |
| HTTP status in `status_codes` | Yes | Configured as transient by the agent author. |
| HTTP 401/403 (auth) | No (unless in `status_codes`) | Retrying won't fix credentials. |
| HTTP 400 (bad request) | No (unless in `status_codes`) | Retrying won't fix the payload. |
| Connection refused | No | Infrastructure is down, not transient. |

### Backoff Calculation

```python
delay = min(backoff_base ** attempt_index, backoff_max)
# attempt_index is 0-based (0 = first retry)
#
# With LLM defaults (base=2.0, max=30.0):
#   Retry 1: 2^0 = 1.0s
#   Retry 2: 2^1 = 2.0s
#   Retry 3: 2^2 = 4.0s  (if max_attempts > 3)
#   ...capped at 30s
#
# With tool defaults (base=1.0, max=10.0):
#   Retry 1: 1^0 = 1.0s
#   ...capped at 10s
```

Jitter is added: `delay = delay * uniform(0.5, 1.0)` to avoid thundering herd when multiple agents retry simultaneously.

### Retry and DBOS Checkpointing

LLM and tool calls are `@step`-decorated (checkpointed by DBOS). Retries within a single `@step` invocation are internal to that step — DBOS sees the step as one unit. If the step ultimately fails (all retries exhausted), DBOS records the failure.

This means:
- Retries do NOT cause duplicate checkpoints.
- A recovered workflow replays the final result of the step (success or failure), not each retry attempt.
- If a crash occurs mid-retry, recovery re-enters the step from scratch (retry count resets). This is acceptable — the alternative (checkpointing each retry) would complicate the step boundary.

### Tool Retry vs. LLM-Driven Retry

When a tool call has `max_attempts > 1`, the runtime retries automatically on timeout or matching status code. This is **infrastructure-level retry** — the LLM never sees the transient failure.

When all retries are exhausted, the error is returned to the LLM as the tool output (e.g. `"Error: tool execution timed out after 60s (3 attempts)"`). The LLM can then decide to try a different tool, adjust its approach, or respond to the user.

Both levels coexist:
- **Runtime retry**: handles transient infrastructure failures silently.
- **LLM retry**: handles semantic failures (wrong tool, bad arguments) intelligently.

---

## SSE Error and Retry Events

Retry attempts, timeouts, and terminal failures are streamed as SSE events so clients can surface progress during what would otherwise be silent hangs. All events follow the existing `response.<noun>.<verb>` naming convention.

### Event Types

#### `response.error`

Emitted when an LLM or tool call fails terminally (all retries exhausted, or a non-retryable error). This is the final failure — the workflow may still continue (tool errors are returned to the LLM) or may transition to `failed` status (LLM errors are fatal).

The `error.message` includes the **raw error detail** from the provider or tool — not a sanitized summary. This is the diagnostic event. The `error.detail` field carries structured provider-specific information when available (HTTP response body, exception type, traceback summary).

```json
{
  "type": "response.error",
  "source": "llm",
  "error": {
    "code": "429",
    "message": "OpenAI rate limit exceeded: Rate limit reached for gpt-5.4. Please retry after 2s. (3 attempts exhausted)",
    "detail": {
      "provider": "openai",
      "status_code": 429,
      "response_body": "{\"error\":{\"message\":\"Rate limit reached for gpt-5.4\",\"type\":\"rate_limit_error\"}}"
    }
  }
}
```

```json
{
  "type": "response.error",
  "source": "tool",
  "tool_name": "web.search",
  "error": {
    "code": "timeout",
    "message": "Tool 'web.search' timed out after 120s (2 attempts exhausted)",
    "detail": null
  }
}
```

```json
{
  "type": "response.error",
  "source": "tool",
  "tool_name": "github.create_issue",
  "error": {
    "code": "exception",
    "message": "Tool 'github.create_issue' raised ConnectionError: DNS resolution failed for api.github.com (2 attempts exhausted)",
    "detail": {
      "exception_type": "ConnectionError",
      "exception_message": "DNS resolution failed for api.github.com"
    }
  }
}
```

#### `response.retry`

Emitted when a retryable failure occurs and the runtime is about to retry. Includes the attempt number, delay before retry, and what went wrong. One event per retry attempt — clients can show "Retrying in 2s..." or a progress indicator.

Same `error` shape as `response.error` — includes raw diagnostic detail so the client can display it immediately without waiting for the terminal event.

```json
{
  "type": "response.retry",
  "source": "llm",
  "attempt": 2,
  "max_attempts": 3,
  "delay_seconds": 2.0,
  "error": {
    "code": "429",
    "message": "OpenAI rate limit exceeded: Rate limit reached for gpt-5.4. Please retry after 2s.",
    "detail": {
      "provider": "openai",
      "status_code": 429,
      "response_body": "{\"error\":{\"message\":\"Rate limit reached for gpt-5.4\",\"type\":\"rate_limit_error\"}}"
    }
  }
}
```

```json
{
  "type": "response.retry",
  "source": "tool",
  "tool_name": "github.search",
  "attempt": 2,
  "max_attempts": 5,
  "delay_seconds": 1.0,
  "error": {
    "code": "502",
    "message": "MCP server 'github' returned HTTP 502: Bad Gateway",
    "detail": {
      "status_code": 502,
      "response_body": "Bad Gateway"
    }
  }
}
```

### Error Codes

The `error.code` field uses a fixed set of string values:

| Code | Meaning |
|---|---|
| `"timeout"` | Request or execution timed out. |
| `"429"`, `"500"`, etc. | HTTP status code from the provider (as a string). |
| `"connection_error"` | Could not connect to the provider. |
| `"execution_timeout"` | Wall-clock execution deadline exceeded. |

### Event Flow Examples

**LLM call succeeds on second attempt (429 → retry → success):**
```
→ response.retry       {source: "llm", attempt: 2, delay: 1.0, error: {code: "429"}}
  ... 1s delay ...
→ response.output_text.delta  {delta: "Hello"}
→ response.output_text.delta  {delta: " world"}
→ response.output_item.done   {...}
```

**Tool times out, retries once, then fails (error returned to LLM):**
```
→ response.retry       {source: "tool", tool_name: "web.search", attempt: 2, delay: 1.0, error: {code: "timeout"}}
  ... 1s delay ...
→ response.error       {source: "tool", tool_name: "web.search", error: {code: "timeout", message: "... 2 attempts exhausted"}}
→ response.output_text.delta  {delta: "I wasn't able to search..."}
```

**LLM call fails permanently (401 — not retried):**
```
→ response.error       {source: "llm", error: {code: "401", message: "Invalid API key"}}
```

**Execution timeout:**
```
→ response.error       {source: "execution", error: {code: "execution_timeout", message: "Wall-clock deadline exceeded after 3600s"}}
```

### Implementation Notes

- Events are written via `_write_output(task_id, event)` — same path as text deltas.
- `response.retry` is emitted **before** the backoff sleep, so clients see it immediately.
- `response.error` for tools is emitted **before** the error string is returned to the LLM, so the client sees the error before the LLM's recovery response.
- These events are persisted in the DBOS stream (same as text deltas), so they survive crash recovery and are available to clients that reconnect mid-stream.

---

## Workflow Integration

### Where Timeouts and Retries are Applied

```
┌─────────────────────────────────────────────────────┐
│  _run_agent_loop                                    │
│  execution.timeout = wall-clock deadline            │
│  execution.max_iterations = iteration cap           │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  _call_llm (per iteration)                    │  │
│  │  llm.timeout                                  │  │
│  │  llm.retry (max_attempts + backoff)           │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  _call_tool (per tool invocation)             │  │
│  │  per-tool timeout → tools.timeout → cap       │  │
│  │  per-tool retry  → tools.retry  → cap         │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Execution Timeout

The wall-clock deadline is checked at the top of each agent loop iteration and before each LLM/tool call. When exceeded:

1. The current iteration completes (no mid-call interruption).
2. The agent loop exits with `status="incomplete"` and `incomplete_details={"reason": "execution_timeout"}`.
3. The task transitions to terminal state — the conversation is preserved, and the user can continue with a new request.

### LLM Call with Retries (Pseudocode)

```python
@step
def _call_llm_with_retry(
    task_id, input_items, instructions, model, tool_schemas,
    resolved: ResolvedLLMConfig,
    *, stream: bool,
) -> dict[str, Any]:
    last_error = None

    for attempt in range(resolved.max_attempts):
        try:
            return _do_llm_call(
                input_items, instructions, model, tool_schemas,
                timeout=resolved.timeout, stream=stream,
            )
        except RetryableError as exc:
            last_error = exc
            if attempt + 1 < resolved.max_attempts:
                delay = min(
                    resolved.backoff_base ** attempt,
                    resolved.backoff_max,
                )
                delay *= uniform(0.5, 1.0)  # jitter
                _write_output(task_id, {
                    "type": "response.retry",
                    "source": "llm",
                    "attempt": attempt + 2,
                    "max_attempts": resolved.max_attempts,
                    "delay_seconds": delay,
                    "error": {"code": exc.code, "message": str(exc)},
                })
                time.sleep(delay)

    # All retries exhausted — emit error event, then raise
    _write_output(task_id, {
        "type": "response.error",
        "source": "llm",
        "error": {
            "code": last_error.code,
            "message": f"{last_error} ({resolved.max_attempts} attempts exhausted)",
        },
    })
    raise last_error
```

### Tool Call with Retry (Pseudocode)

```python
@step
def _call_tool(
    task_id, name, arguments, *,
    resolved: ResolvedToolConfig,
) -> str:
    last_error = None

    for attempt in range(resolved.max_attempts):
        try:
            return tool_manager.call_tool(
                name, arguments, timeout=resolved.timeout,
            )
        except RetryableToolError as exc:
            last_error = exc
            if attempt + 1 < resolved.max_attempts:
                delay = min(
                    resolved.backoff_base ** attempt,
                    resolved.backoff_max,
                )
                delay *= uniform(0.5, 1.0)  # jitter
                _write_output(task_id, {
                    "type": "response.retry",
                    "source": "tool",
                    "tool_name": name,
                    "attempt": attempt + 2,
                    "max_attempts": resolved.max_attempts,
                    "delay_seconds": delay,
                    "error": {"code": exc.code, "message": str(exc)},
                })
                time.sleep(delay)

    # All retries exhausted — emit error event, return error to LLM
    _write_output(task_id, {
        "type": "response.error",
        "source": "tool",
        "tool_name": name,
        "error": {
            "code": last_error.code,
            "message": f"{last_error} ({resolved.max_attempts} attempts exhausted)",
        },
    })
    return f"Error: {last_error} ({resolved.max_attempts} attempts)"
```

---

## Implementation Plan

### Phase 1: Adapter-level timeouts (parameterize existing constants)

1. Add `timeout` parameter to adapter `create()` methods.
2. Replace hardcoded `_REQUEST_TIMEOUT` / `_STREAM_TIMEOUT` with the parameter.
3. Thread timeout from `ResolvedLLMConfig` through `_call_llm` → adapter.

### Phase 2: LLM retry with exponential backoff

1. Add `RetryConfig` to spec types.
2. Parse `retry` block from `config.yaml` in the spec parser.
3. Implement retry loop in `_call_llm` (inside the `@step` boundary).
4. Classify errors as retryable (timeout + configured status codes) vs. permanent.

### Phase 3: Tool execution timeout and retry

1. Add `timeout` and `retry` fields to `ToolsConfig`, `MCPServerConfig`, `LocalToolInfo`, `SubAgentRef`.
2. Implement tool config resolution — per-tool → global default fallback.
3. Wrap `tool_manager.call_tool()` with the resolved timeout and retry loop.
4. Return error string to LLM when all retries are exhausted.

### Phase 4: Execution timeout

1. Add `ExecutionConfig` to spec types.
2. Record `start_time` at top of `_run_agent_loop`.
3. Check deadline at each iteration boundary.
4. Exit with `status="incomplete"` on expiry.

### Phase 5: Runtime caps

1. Add `RuntimeCaps` dataclass (single `execution_timeout` field).
2. Add `caps` parameter to `init()`.
3. Clamp `execution.timeout` via `min(spec, cap)` in `_run_agent_loop`.
4. Add `--execution-timeout` CLI flag and config file support.

### Phase 6: Spec parser and validation

1. Parse `timeout`, `retry`, and `execution` blocks from YAML.
2. Validate ranges (positive integers, valid status codes).
3. Reject unknown fields in these blocks.

---

## Test Plan

Tests are organized by phase. Each phase's tests live in the existing test file that mirrors the source module. Unit tests use function-based pytest with fixtures (no class-based tests).

### Phase 1: Adapter-level timeouts

**File**: `tests/llms/test_openai_adapter.py` (and analogous files for other adapters)

| Test | Description |
|---|---|
| `test_create_uses_custom_timeout` | Pass `timeout=42` to `create()`, assert the underlying client is configured with that timeout (not the hardcoded default). |
| `test_create_uses_default_timeout_when_omitted` | Call `create()` without `timeout`, assert the default constant is used. |
| `test_timeout_propagated_to_streaming_call` | Mock the client, call `create()` with `stream=True` and a custom timeout, verify the HTTP client's timeout matches. |

### Phase 2: LLM retry with exponential backoff

**File**: `tests/runtime/test_workflow.py`

| Test | Description |
|---|---|
| `test_llm_retry_on_timeout` | Mock LLM to raise `TimeoutError` on first call and succeed on second. Assert the retry happens and the final result is the success response. |
| `test_llm_retry_on_configured_status_code` | Mock LLM to return HTTP 429 on first call, succeed on second. Assert retry triggers for status codes in `retry.status_codes`. |
| `test_llm_no_retry_on_non_retryable_status` | Mock LLM to return HTTP 401. Assert no retry — error propagates immediately. |
| `test_llm_retry_exhausted_raises` | Mock LLM to raise `TimeoutError` on every call. With `max_attempts=2`, assert the error is raised after 2 attempts. |
| `test_llm_backoff_delays` | Mock LLM to fail twice then succeed (`max_attempts=3`). Capture sleep durations, assert exponential backoff (`base ** attempt_index`) with jitter in `[0.5, 1.0]` range. |
| `test_llm_backoff_capped_at_max` | Set `backoff_base=10, backoff_max=5`. Assert delay never exceeds `backoff_max`. |
| `test_llm_retry_emits_sse_retry_event` | Mock LLM to fail once then succeed. Assert `response.retry` event is written via `_write_output` with correct `attempt`, `max_attempts`, `delay_seconds`, and `error` fields. |
| `test_llm_retry_exhausted_emits_sse_error_event` | Mock LLM to fail on all attempts. Assert `response.error` event is written with `source: "llm"` and the raw error message. |

**File**: `tests/spec/test_parser.py` (or existing spec test file)

| Test | Description |
|---|---|
| `test_parse_retry_config_all_fields` | YAML with all retry fields set. Assert `RetryConfig` has the correct values. |
| `test_parse_retry_config_defaults` | YAML with `retry: {}`. Assert defaults are applied. |
| `test_parse_retry_config_omitted` | YAML with no `retry` block. Assert LLM defaults (`_LLM_RETRY_DEFAULTS`) are used. |

### Phase 3: Tool execution timeout and retry

**File**: `tests/runtime/test_workflow.py`

| Test | Description |
|---|---|
| `test_tool_retry_on_timeout` | Mock tool to time out once then succeed. Assert retry happens and the tool output is the success result. |
| `test_tool_retry_on_mcp_status_code` | Mock MCP tool to return HTTP 502 once then succeed. Assert retry triggers. |
| `test_tool_status_codes_ignored_for_local_tool` | Mock local Python tool to raise an error with a status code. Assert no status-code retry (only timeout retries apply). |
| `test_tool_retry_exhausted_returns_error_to_llm` | Mock tool to fail on all attempts. Assert the error string (not exception) is returned as tool output to the LLM. |
| `test_tool_retry_emits_sse_events` | Mock tool to fail once then succeed. Assert `response.retry` event with `source: "tool"` and `tool_name`. |
| `test_tool_terminal_failure_emits_sse_error` | Mock tool to fail on all attempts. Assert `response.error` event with `source: "tool"`, `tool_name`, and raw error detail. |

**File**: `tests/runtime/test_workflow.py`

| Test | Description |
|---|---|
| `test_per_tool_timeout_overrides_global` | Agent spec with `tools.timeout=60` and a specific tool with `timeout=120`. Assert the tool call uses 120s. |
| `test_per_tool_retry_overrides_global` | Agent spec with `tools.retry.max_attempts=2` and a specific tool with `retry.max_attempts=5`. Assert the tool uses 5 attempts. |
| `test_per_tool_inherits_global_when_omitted` | Agent spec with `tools.timeout=60` and a tool with no `timeout`. Assert the tool call uses 60s. |

### Phase 4: Execution timeout

**File**: `tests/runtime/test_workflow.py`

| Test | Description |
|---|---|
| `test_execution_timeout_exits_incomplete` | Set `execution.timeout=1` (1 second). Mock LLM to sleep longer than the deadline. Assert the task finishes with `status="incomplete"` and `incomplete_details.reason == "execution_timeout"`. |
| `test_execution_timeout_preserves_conversation` | After execution timeout, assert conversation items from completed iterations are persisted (not lost). |
| `test_execution_timeout_emits_sse_error` | Assert `response.error` event with `source: "execution"` and `code: "execution_timeout"`. |
| `test_max_iterations_exits_incomplete` | Set `execution.max_iterations=1`. Mock LLM to request a tool call (which would require iteration 2). Assert exit with `status="incomplete"`. |

### Phase 5: Runtime caps

**File**: `tests/runtime/test_workflow.py`

| Test | Description |
|---|---|
| `test_execution_timeout_clamped_by_runtime_cap` | Agent spec `execution.timeout=7200`, runtime cap `execution_timeout=3600`. Assert the agent loop uses 3600. |
| `test_execution_timeout_under_cap_unchanged` | Agent spec `execution.timeout=1800`, cap `execution_timeout=7200`. Assert the agent loop uses 1800 (no clamping). |
| `test_runtime_caps_default_value` | Create `RuntimeCaps()` with no args. Assert `execution_timeout == 7200`. |

**File**: `tests/test_cli.py` (or existing CLI test file)

| Test | Description |
|---|---|
| `test_cli_execution_timeout_flag` | Run `ap server --execution-timeout 3600`. Assert `RuntimeCaps` is created with `execution_timeout=3600`. |
| `test_cli_execution_timeout_from_config` | Config file with `execution_timeout: 3600`. Assert `RuntimeCaps` picks it up. |
| `test_cli_flag_overrides_config` | Config file `execution_timeout: 7200`, CLI `--execution-timeout 3600`. Assert CLI wins. |

### Phase 6: Spec parser and validation

**File**: `tests/spec/test_parser.py` (or existing spec test file)

| Test | Description |
|---|---|
| `test_parse_llm_timeout_and_retry` | Full `llm` block with `timeout` and `retry`. Assert `LLMConfig` has correct values. |
| `test_parse_tools_global_timeout_and_retry` | `tools` block with global `timeout` and `retry`. Assert `ToolsConfig` populated. |
| `test_parse_per_tool_timeout_on_mcp` | MCP server config with `timeout: 120`. Assert `MCPServerConfig.timeout == 120`. |
| `test_parse_per_tool_retry_on_mcp` | MCP server config with `retry` block. Assert `MCPServerConfig.retry` is a `RetryConfig`. |
| `test_parse_per_tool_timeout_on_local` | Local tool with `timeout: 300`. Assert `LocalToolInfo.timeout == 300`. |
| `test_parse_execution_config` | `execution` block with `timeout` and `max_iterations`. Assert `ExecutionConfig` values. |
| `test_parse_execution_defaults` | No `execution` block. Assert defaults (3600, 1000). |
| `test_validation_negative_timeout_rejected` | `llm.timeout: -1`. Assert parse error. |
| `test_validation_zero_max_attempts_rejected` | `retry.max_attempts: 0`. Assert parse error. |
| `test_validation_non_integer_status_code_rejected` | `status_codes: ["429"]`. Assert parse error (must be integers). |
| `test_validation_unknown_retry_field_rejected` | `retry: {unknown_field: true}`. Assert parse error. |

### Integration Tests

**File**: `tests/server/integration/test_concurrency.py` (or `test_routes_responses.py`)

These tests use the full server stack (FastAPI → workflow → mock LLM) to verify end-to-end behavior visible through the SSE stream.

| Test | Description |
|---|---|
| `test_llm_retry_visible_in_sse_stream` | Configure agent with `llm.retry.max_attempts=2`. Mock LLM to fail once (429) then succeed. Consume SSE stream, assert `response.retry` event appears before the successful output. |
| `test_tool_retry_visible_in_sse_stream` | Configure agent with a tool that times out once then succeeds. Consume SSE stream, assert `response.retry` event with `source: "tool"`. |
| `test_execution_timeout_visible_in_sse_stream` | Configure agent with `execution.timeout=1`. Assert SSE stream contains `response.error` with `code: "execution_timeout"` and the task reaches terminal `incomplete` status. |
| `test_llm_terminal_failure_visible_in_sse_stream` | Configure agent with `llm.retry.max_attempts=1`. Mock LLM to return 401. Assert SSE stream contains `response.error` with `source: "llm"` and task reaches `failed` status. |
