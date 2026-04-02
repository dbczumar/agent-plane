# Executor Spec: YAML Config Changes

## Context

The executor contract (EXECUTOR_CONTRACT_FINAL.md) introduces three executor
types: `DefaultExecutor`, `ClaudeSDKExecutor`, `RemoteExecutor`. This design
defines how they're configured in `config.yaml` and how the validator
enforces per-type field validity.

---

## New top-level section: `executor:`

```yaml
executor:
  type: llm | claude_sdk | remote   # default: llm
  timeout: 3600                      # task deadline (seconds)
  max_iterations: 50                 # max run_turn() calls
  endpoint: https://...              # remote only
  request_timeout: 300               # per-call timeout (remote only)
```

`executor:` is optional. When omitted, `type` defaults to `llm`.

`executor.type` is the discriminator for the entire spec's validity. It
determines which other top-level sections and fields are valid. Invalid
fields are rejected by the validator — not silently ignored.

The `execution:` top-level section is removed. `timeout` and
`max_iterations` move under `executor:`.

---

## Existing section: `llm:`

```yaml
llm:
  model: openai/gpt-4o
  request_timeout: 120               # per-LLM-call timeout (seconds)
  connection:
    api_key: "..."
    base_url: "..."
  retry:
    max_attempts: 3
    backoff_base: 2.0
    backoff_max: 30.0
```

`llm:` is unchanged in shape. The only rename is `timeout` → `request_timeout`
to distinguish from `executor.timeout` (task deadline).

---

## Validation matrix

`executor.type` determines which fields are valid across the entire spec.
Fields marked **invalid** are rejected by the validator — the user gets a
clear error, e.g. "llm.connection is not supported for executor type
claude_sdk."

| Field | `type: llm` | `type: claude_sdk` | `type: remote` |
|---|---|---|---|
| **executor section** | | | |
| `executor.type` | optional (default) | required | required |
| `executor.timeout` | optional | optional | optional |
| `executor.max_iterations` | optional | optional | optional |
| `executor.endpoint` | **invalid** | **invalid** | required |
| `executor.request_timeout` | **invalid** | **invalid** | optional |
| **llm section** | | | |
| `llm.model` | required | optional | **invalid** |
| `llm.request_timeout` | optional | optional | **invalid** |
| `llm.connection` | optional | **invalid** | **invalid** |
| `llm.retry` | optional | **invalid** | **invalid** |
| **other sections** | | | |
| `tools:` | valid | `claude:` prefix only | **invalid** |
| `instructions:` | valid | valid | **invalid** |
| `compaction:` | valid | **invalid** | **invalid** |

---

## Canonical examples

### DefaultExecutor — `executor:` omitted (implicit `type: llm`)

```yaml
name: my-agent
description: A general assistant

llm:
  model: openai/gpt-4o
  request_timeout: 120
  connection:
    api_key: "${OPENAI_API_KEY}"

tools:
  - web_search
  - spawn

compaction:
  trigger_threshold: 0.8
  recent_window: 5
```

### ClaudeSDKExecutor

```yaml
name: my-coder
description: A coding agent backed by Claude Code

executor:
  type: claude_sdk
  timeout: 3600
  max_iterations: 10

llm:
  model: anthropic/claude-sonnet-4-20250514
  request_timeout: 600

tools:
  - claude:Bash
  - claude:Read
  - claude:Edit
  - claude:Write
  - claude:Glob
  - claude:Grep
```

### RemoteExecutor

```yaml
name: my-omni-agent
description: OmniAgents coding agent

executor:
  type: remote
  timeout: 3600
  max_iterations: 50
  endpoint: http://localhost:8000/v1/turns
  request_timeout: 300
```

No `llm:`, `tools:`, `instructions:`, or `compaction:` sections.
The remote service owns all of those.

---

## Implementation

### Changed types (`spec/types.py`)

**New: `ExecutorSpec`**

```python
@dataclass
class ExecutorSpec:
    """
    Top-level executor configuration.

    :param type: Executor type. ``"llm"`` (default), ``"claude_sdk"``,
        or ``"remote"``.
    :param timeout: Task deadline in seconds, e.g. ``3600``.
    :param max_iterations: Max ``run_turn()`` calls, e.g. ``50``.
    :param endpoint: URL for remote executor,
        e.g. ``"http://localhost:8000/v1/turns"``.
    :param request_timeout: Per-HTTP-call timeout for remote executor
        in seconds, e.g. ``300``.
    """
    type: str = "llm"
    timeout: int = 3600
    max_iterations: int = 1000
    endpoint: str | None = None
    request_timeout: int | None = None
```

**Changed: `LLMConfig`**

Rename `timeout` → `request_timeout`. Remove `executor` field (it moved
to `ExecutorSpec.type`).

**Removed: `ExecutionConfig`**

`timeout` and `max_iterations` moved to `ExecutorSpec`.

**Changed: `AgentSpec`**

```python
@dataclass
class AgentSpec:
    ...
    executor: ExecutorSpec    # replaces execution: ExecutionConfig
    llm: LLMConfig | None    # unchanged
    ...
```

### Changed parser (`spec/parser.py`)

- Parse `executor:` block into `ExecutorSpec`.
- Default `type` to `"llm"` when `executor:` is absent.
- Rename `llm.timeout` → `llm.request_timeout` during parsing.

### Changed validator (`spec/validator.py`)

After parsing, validate the full spec against `executor.type`:

```python
def _validate_executor_type(spec: AgentSpec) -> list[str]:
    """
    Validate that all spec fields are valid for the declared executor type.

    :param spec: The parsed agent spec.
    :returns: List of validation error messages (empty = valid).
    """
    errors = []
    etype = spec.executor.type

    if etype == "remote":
        if spec.executor.endpoint is None:
            errors.append("executor.endpoint is required for type: remote")
        if spec.llm is not None:
            errors.append("llm section is not supported for type: remote")
        if spec.instructions is not None:
            errors.append("instructions is not supported for type: remote")
        if spec.tools.agents or spec.tools.builtins:
            errors.append("tools section is not supported for type: remote")
        if spec.compaction is not None:
            errors.append("compaction is not supported for type: remote")

    elif etype == "claude_sdk":
        if spec.executor.endpoint is not None:
            errors.append("executor.endpoint is not supported for type: claude_sdk")
        if spec.llm and spec.llm.connection is not None:
            errors.append("llm.connection is not supported for type: claude_sdk")
        if spec.compaction is not None:
            errors.append("compaction is not supported for type: claude_sdk")

    elif etype == "llm":
        if spec.executor.endpoint is not None:
            errors.append("executor.endpoint is not supported for type: llm")
        if spec.executor.request_timeout is not None:
            errors.append(
                "executor.request_timeout is not supported for type: llm"
                " — use llm.request_timeout instead"
            )

    return errors
```

### Changed workflow (`runtime/workflow.py`)

Replace `spec.execution.timeout` → `spec.executor.timeout` and
`spec.execution.max_iterations` → `spec.executor.max_iterations`.

Replace `spec.llm.timeout` → `spec.llm.request_timeout`.

### Executor construction

`_create_executor(spec)` reads `spec.executor.type`:

```python
def _create_executor(spec: AgentSpec) -> Executor:
    etype = spec.executor.type
    if etype == "claude_sdk":
        return ClaudeSDKExecutor.from_spec(spec)
    if etype == "remote":
        return RemoteExecutor.from_spec(spec)
    return DefaultExecutor.from_spec(spec)
```
