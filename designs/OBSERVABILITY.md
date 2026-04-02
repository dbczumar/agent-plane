# Observability with OpenTelemetry

## Context

The agent-plane runtime has no structured observability. All logging uses
Python's standard `logging` module with unstructured text messages. There
are no traces, spans, or metrics. Debugging a multi-iteration agent
execution — especially one involving tool calls, retries, and sub-agents
— requires reading raw log lines and mentally reconstructing the
execution timeline.

OpenTelemetry (OTel) provides a vendor-neutral observability framework.
The **GenAI Semantic Conventions** (currently experimental, stabilizing
rapidly) define standard span names, attributes, and events for LLM
applications. Adopting both gives us structured, correlated observability
that works with any backend (Jaeger, Datadog, Grafana Tempo, OTLP
collectors, etc.).

---

## Goals

1. **Traces for every agent execution** — one trace per `POST /v1/responses`
   request, spanning the full workflow lifecycle.
2. **GenAI semantic convention spans** for LLM calls — standard attributes
   so any GenAI-aware backend can render them correctly.
3. **Tool call spans** — timing, tool name, success/failure for every
   tool invocation.
4. **Structured log correlation** — logs emitted during a span carry
   the trace ID and span ID automatically.
5. **Runtime configuration** — operators configure telemetry via CLI
   flags and environment variables, not agent specs. Agent authors
   should not need to think about observability.
6. **Zero overhead when disabled** — no performance impact when no
   exporter is configured.

---

## What We Capture

### Span Hierarchy

```
[HTTP] POST /v1/responses
  └─ [Workflow] agent_execution {agent.id, task.id, conversation.id}
       ├─ [GenAI] chat {gen_ai.system, gen_ai.request.model}        ← iteration 1
       │    ├─ event: gen_ai.user.message
       │    ├─ event: gen_ai.assistant.message
       │    └─ event: gen_ai.usage {input_tokens, output_tokens}
       ├─ [Tool] tool_call {tool.name, tool.call_id}                ← tool from iteration 1
       ├─ [GenAI] chat {gen_ai.system, gen_ai.request.model}        ← iteration 2
       │    ├─ event: gen_ai.tool.message                           ← tool result fed back
       │    ├─ event: gen_ai.assistant.message
       │    └─ event: gen_ai.usage {input_tokens, output_tokens}
       └─ [Workflow] agent_execution {agent.id, ...}                ← sub-agent (spawn)
            ├─ [GenAI] chat ...
            └─ [Tool] tool_call ...
```

### Operations and Their Spans

| Operation | Span Name | Kind | Key Attributes |
|---|---|---|---|
| HTTP request | `POST /v1/responses` | `SERVER` | Standard HTTP semconv (auto-instrumented) |
| Agent workflow | `agent_execution` | `INTERNAL` | `agent.id`, `task.id`, `conversation.id`, `agent.name` |
| Agent loop iteration | `agent_iteration` | `INTERNAL` | `agent.iteration.number` |
| LLM call | `chat {gen_ai.request.model}` | `CLIENT` | GenAI semconv (see below) |
| Tool invocation | `tool_call {tool.name}` | `INTERNAL` | `tool.name`, `tool.call_id`, `tool.type` |
| MCP server call | `mcp_call {server.name}` | `CLIENT` | `mcp.server.name`, `mcp.server.url` |
| Sub-agent spawn | `agent_execution` | `INTERNAL` | Same as workflow, nested under parent |
| LLM retry | (no new span) | — | Retry events on the `chat` span |
| Tool retry | (no new span) | — | Retry events on the `tool_call` span |

### GenAI Semantic Convention Attributes

Following the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

**Span attributes on `chat` spans:**

| Attribute | Source | Example |
|---|---|---|
| `gen_ai.system` | Parsed from model string prefix | `"openai"`, `"anthropic"` |
| `gen_ai.request.model` | `LLMConfig.model` (without provider prefix) | `"gpt-5.4"` |
| `gen_ai.response.model` | LLM response metadata | `"gpt-5.4-2025-03-01"` |
| `gen_ai.request.max_tokens` | From `LLMConfig.extra` if set | `4096` |
| `gen_ai.request.temperature` | From `LLMConfig.extra` if set | `0.7` |
| `gen_ai.operation.name` | Always `"chat"` | `"chat"` |
| `gen_ai.usage.input_tokens` | LLM response usage | `1523` |
| `gen_ai.usage.output_tokens` | LLM response usage | `847` |

**Span events on `chat` spans:**

| Event | When | Body |
|---|---|---|
| `gen_ai.user.message` | Before LLM call | User message content (if capture enabled) |
| `gen_ai.assistant.message` | After LLM call | Assistant response content (if capture enabled) |
| `gen_ai.tool.message` | Tool result fed back | Tool output (if capture enabled) |
| `gen_ai.choice` | After LLM call | `finish_reason`, `index` |

**Content capture is opt-in** — message bodies can contain sensitive data.
When disabled (default), events are still emitted but with content
redacted to `"[redacted]"`. This follows the GenAI semconv guidance on
sensitive data.

### Metrics

| Metric | Type | Unit | Description |
|---|---|---|---|
| `gen_ai.client.token.usage` | Histogram | `{token}` | Input/output tokens per LLM call |
| `gen_ai.client.operation.duration` | Histogram | `s` | LLM call duration |
| `agent.tool.duration` | Histogram | `s` | Tool call duration |
| `agent.iteration.count` | Counter | `{iteration}` | Iterations per agent execution |

---

## Configuration

Telemetry is **runtime configuration** — it belongs to the operator, not
the agent author. Agent specs (`config.yaml`) have no telemetry fields.

### CLI Flags

New flags on the `agent-plane serve` command:

```
--otel-exporter <protocol>       OTLP exporter protocol: "grpc", "http", or "none" (default: "none")
--otel-endpoint <url>            OTLP endpoint (default: "http://localhost:4317" for grpc,
                                 "http://localhost:4318" for http)
--otel-service-name <name>       Service name for resource attributes (default: "agent-plane")
--otel-capture-content            Enable capturing message content in span events (default: off)
--otel-headers <key=val,...>     Extra headers for the OTLP exporter (e.g., auth tokens)
```

### Environment Variables

Standard OTel environment variables are respected as fallbacks when CLI
flags are not provided. CLI flags take precedence.

| Env Var | Maps To |
|---|---|
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `--otel-exporter` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `--otel-endpoint` |
| `OTEL_SERVICE_NAME` | `--otel-service-name` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `--otel-headers` |
| `AGENT_PLANE_OTEL_CAPTURE_CONTENT` | `--otel-capture-content` |

### Configuration Dataclass

```python
@dataclass
class TelemetryConfig:
    """
    Operator-level telemetry configuration for the agent-plane runtime.

    :param exporter: OTLP exporter protocol. "grpc", "http", or "none".
    :param endpoint: OTLP collector endpoint URL.
    :param service_name: Service name reported in resource attributes.
    :param capture_content: Whether to include message content in span events.
    :param headers: Extra headers for the OTLP exporter (e.g. auth).
    """

    exporter: str = "none"
    endpoint: str | None = None
    service_name: str = "agent-plane"
    capture_content: bool = False
    headers: dict[str, str] | None = None
```

### Initialization Flow

```
cli.py: parse CLI flags + env vars → TelemetryConfig
  │
  ▼
telemetry.init(config: TelemetryConfig)
  ├─ If exporter == "none": install NoOp providers → return
  ├─ Create Resource(service.name=..., service.version=...)
  ├─ Create TracerProvider with OTLP span exporter
  ├─ Create MeterProvider with OTLP metric exporter
  ├─ Create LoggerProvider with OTLP log exporter
  ├─ Set global providers: trace.set_tracer_provider(...)
  ├─ Instrument FastAPI (auto-instrumentation for HTTP spans)
  └─ Store capture_content flag in module-level variable
```

When `exporter == "none"`, the NoOp providers are zero-cost — the OTel
SDK's no-op implementation skips all attribute recording and export. The
only overhead is the function-call boundary at span creation, which is
negligible.

---

## How We Capture: Instrumentation Approach

### Principle: Manual Spans at Semantic Boundaries

We use **manual instrumentation** with the OTel SDK, not auto-
instrumentation libraries for LLM calls. Reasons:

1. Our LLM client (`llms/client.py`) already abstracts provider
   differences — wrapping litellm with an auto-instrumentation library
   would double-layer the abstraction.
2. We need spans at our semantic boundaries (workflow, iteration, tool
   call), not at litellm's internal boundaries.
3. Manual spans give us full control over attribute population and
   content capture gating.

The one exception is **FastAPI auto-instrumentation** — the
`opentelemetry-instrumentation-fastapi` package instruments HTTP
request/response spans automatically. This is standard and low-risk.

### Tracer Module

New module: `agent_plane/runtime/telemetry.py`

```python
from opentelemetry import trace

_tracer = trace.get_tracer("agent_plane.runtime")
_capture_content: bool = False


def init(config: TelemetryConfig) -> None:
    """
    Initialize OpenTelemetry providers and exporters.

    :param config: Operator-level telemetry configuration.
    """
    global _capture_content
    _capture_content = config.capture_content

    if config.exporter == "none":
        return  # NoOp providers already installed by default

    # ... create and register providers, exporters


def workflow_span(agent_id: str, task_id: str, conversation_id: str, agent_name: str | None):
    """
    Create a span for an agent workflow execution.

    :param agent_id: The agent's unique identifier.
    :param task_id: The task/response identifier.
    :param conversation_id: The conversation identifier.
    :param agent_name: Human-readable agent name, if available.
    """
    return _tracer.start_as_current_span(
        "agent_execution",
        attributes={
            "agent.id": agent_id,
            "task.id": task_id,
            "conversation.id": conversation_id,
            "agent.name": agent_name or "",
        },
    )


def chat_span(model: str, system: str, operation: str = "chat"):
    """
    Create a GenAI semantic convention span for an LLM call.

    :param model: The model identifier without provider prefix, e.g. "gpt-5.4".
    :param system: The GenAI system/provider, e.g. "openai".
    :param operation: The GenAI operation name, e.g. "chat".
    """
    return _tracer.start_as_current_span(
        f"chat {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.system": system,
            "gen_ai.request.model": model,
            "gen_ai.operation.name": operation,
        },
    )


def record_llm_usage(span, usage: dict) -> None:
    """
    Record token usage attributes and event on a chat span.

    :param span: The active chat span.
    :param usage: Dict with "input_tokens" and "output_tokens" keys.
    """
    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))


def tool_span(tool_name: str, call_id: str, tool_type: str):
    """
    Create a span for a tool invocation.

    :param tool_name: The tool's registered name.
    :param call_id: The unique call identifier from the LLM response.
    :param tool_type: Tool category: "local", "mcp", "builtin", "client".
    """
    return _tracer.start_as_current_span(
        f"tool_call {tool_name}",
        attributes={
            "tool.name": tool_name,
            "tool.call_id": call_id,
            "tool.type": tool_type,
        },
    )
```

### Instrumentation Points

#### 1. Workflow (`workflow.py`)

```python
# In agent_execution_workflow():
with telemetry.workflow_span(agent_id, task_id, conversation_id, spec.name):
    result = _run_agent_loop(...)
```

#### 2. Agent Loop Iteration (`workflow.py`)

```python
# In _run_agent_loop(), inside the iteration loop:
with telemetry.iteration_span(iteration_number):
    response = _call_llm_for_iteration_with_error_handling(...)
    if response.tool_calls:
        _handle_tool_calls(...)
```

#### 3. LLM Calls (`workflow.py`)

```python
# In _call_llm() and _call_llm_streaming():
system, model = parse_model_string(full_model)  # "openai/gpt-5.4" → ("openai", "gpt-5.4")

with telemetry.chat_span(model, system) as span:
    response = client.responses.create(...)
    if response.usage:
        telemetry.record_llm_usage(span, response.usage)
    if telemetry.should_capture_content():
        span.add_event("gen_ai.assistant.message", {"content": response.text})
```

**Streaming**: The span stays open for the duration of the stream. Token
events are not recorded individually (too noisy). Usage is recorded when
the stream completes and the accumulated response is available.

**Retries**: Retry attempts are recorded as span events on the `chat`
span, not as separate spans. This keeps the retry visible without
inflating span count:

```python
span.add_event("gen_ai.retry", {
    "attempt": attempt,
    "max_attempts": max_attempts,
    "error.type": error_code,
    "error.message": error_message,
})
```

#### 4. Tool Calls (`workflow.py`)

```python
# In _call_tool():
with telemetry.tool_span(tool_name, call_id, tool.tool_type) as span:
    result = execute_tool_with_retry(...)
    span.set_attribute("tool.status", "success" if not is_error else "error")
```

#### 5. Sub-Agent Spans

Sub-agent workflows (`spawn` tool) create nested `agent_execution` spans.
OTel context propagation handles this automatically — when `spawn`
starts a new workflow on the same thread/task, the parent span context
is inherited. For async sub-agents, the spawn tool must explicitly
propagate the trace context to the child workflow.

#### 6. Log Correlation

The OTel Logs SDK bridges Python's `logging` module:

```python
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

# In telemetry.init():
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
handler = LoggingHandler(logger_provider=logger_provider)
logging.getLogger().addHandler(handler)
```

This means all existing `_logger.info(...)` / `_logger.error(...)` calls
automatically get `trace_id` and `span_id` fields when emitted inside an
active span. No changes to existing logging call sites needed.

---

## Dependencies

New packages added to `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20",
    "opentelemetry-exporter-otlp-proto-http>=1.20",
    "opentelemetry-instrumentation-fastapi>=0.44b0",
    "opentelemetry-semantic-conventions>=0.50b0",   # GenAI semconv constants
]
```

The `opentelemetry-semantic-conventions` package provides the GenAI
attribute constants (e.g., `GenAiOperationNameValues.CHAT`,
`GEN_AI_SYSTEM`). We use these constants instead of raw strings to track
upstream renames as the conventions stabilize.

---

## Implementation Plan

### Phase 1: Foundation

**Files:** New `agent_plane/runtime/telemetry.py`, modify `agent_plane/cli.py`

1. Add `TelemetryConfig` dataclass.
2. Implement `telemetry.init()` — provider creation, exporter setup,
   NoOp fallback.
3. Add CLI flags and env var fallbacks to `cli.py`.
4. Wire `telemetry.init()` into server startup (before `create_app()`).
5. Add FastAPI auto-instrumentation.

### Phase 2: Core Spans

**Files:** Modify `agent_plane/runtime/workflow.py`

1. Add `workflow_span` around `agent_execution_workflow()`.
2. Add `iteration_span` inside the agent loop.
3. Add `chat_span` around `_call_llm()` and `_call_llm_streaming()`.
4. Record `gen_ai.usage.*` attributes from LLM responses.
5. Add `tool_span` around `_call_tool()`.
6. Add retry events to chat and tool spans.

### Phase 3: Content Capture and Metrics

**Files:** Modify `agent_plane/runtime/telemetry.py`, `workflow.py`

1. Implement gated content capture (message events on chat spans).
2. Add `gen_ai.client.token.usage` histogram.
3. Add `gen_ai.client.operation.duration` histogram.
4. Add `agent.tool.duration` histogram.

### Phase 4: Log Correlation

**Files:** Modify `agent_plane/runtime/telemetry.py`

1. Bridge Python `logging` to OTel LoggerProvider.
2. Verify existing log call sites get trace/span IDs.

---

## Not Yet

- **Baggage propagation for cross-service tracing** — if the caller
  sends a `traceparent` header, we should join their trace. Deferred
  until we have a concrete cross-service use case.
- **Custom span processors** (e.g., sampling strategies beyond the OTel
  SDK defaults). Use `OTEL_TRACES_SAMPLER` env var for now.
- **Dashboard templates** — pre-built Grafana/Datadog dashboards for
  agent execution metrics. Useful but out of scope for the runtime.
- **DBOS span integration** — DBOS has its own internal tracing. We
  don't wrap DBOS operations in OTel spans; our spans are at the
  semantic level above DBOS (workflow, iteration, LLM call, tool call).
  Bridging the two is possible but not a priority.
- **Per-agent telemetry overrides** — e.g., an agent spec that says
  "always capture content for this agent." Telemetry is operator
  config, not agent config, for now.
- **Prompt/completion token cost estimation** — using model pricing
  tables to convert token counts to dollar amounts. Interesting for
  dashboards but not core to the observability layer.
