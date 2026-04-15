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
  └─ [Workflow] agent_execution {agent.id, task.id, session.id}
       ├─ [GenAI] chat {gen_ai.provider.name, gen_ai.request.model}   ← iteration 1
       │    attrs: gen_ai.input.messages, gen_ai.output.messages,
       │           gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
       ├─ [Tool] tool_call {tool.name, tool.call_id}                  ← tool from iteration 1
       ├─ [GenAI] chat {gen_ai.provider.name, gen_ai.request.model}   ← iteration 2
       │    attrs: gen_ai.input.messages (includes tool result),
       │           gen_ai.output.messages, gen_ai.usage.*
       └─ [Workflow] agent_execution {agent.id, ...}                  ← sub-agent (spawn)
            ├─ [GenAI] chat ...
            └─ [Tool] tool_call ...
```

### Operations and Their Spans

| Operation | Span Name | Kind | Key Attributes |
|---|---|---|---|
| HTTP request | `POST /v1/responses` | `SERVER` | Standard HTTP semconv (auto-instrumented) |
| Agent workflow | `agent_execution` | `INTERNAL` | `agent.id`, `task.id`, `conversation.id`, `session.id`, `agent.name`, `agent.executor.type`, `agent.background` |
| Agent loop iteration | `agent_iteration` | `INTERNAL` | `agent.iteration.number`, `agent.iteration.input_message_count`, `agent.iteration.tool_count` |
| LLM call | `chat {gen_ai.request.model}` | `CLIENT` | GenAI semconv (see below) |
| Tool invocation | `tool_call {tool.name}` | `INTERNAL` | `tool.name`, `tool.call_id`, `tool.type`, `tool.status` |
| MCP server call | `mcp_call {server.name}` | `CLIENT` | `mcp.server.name`, `mcp.server.url` |
| Sub-agent spawn | `agent_execution` | `INTERNAL` | Same as workflow, plus `agent.conversation.kind = "sub_agent"` |
| LLM retry | (no new span) | — | Retry events on the `chat` span |
| Tool retry | (no new span) | — | Retry events on the `tool_call` span |

### GenAI Semantic Convention Attributes

Following the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(**v1.37.0+** — the v1.37.0 release consolidated per-message span
events into structured span attributes; older per-message events like
`gen_ai.user.message`, `gen_ai.choice`, etc. are deprecated).

**Span attributes on `agent_execution` spans:**

| Attribute | Type | Source | Example |
|---|---|---|---|
| `agent.id` | `string` | Agent entity | `"ag_abc123"` |
| `agent.name` | `string` | Agent spec | `"my-coder"` |
| `task.id` | `string` | Task entity | `"resp_d8e9f0a1..."` |
| `conversation.id` | `string` | Conversation entity | `"conv_e4f5a6b7..."` |
| `session.id` | `string` | = conversation.id (OTel session grouping) | `"conv_e4f5a6b7..."` |
| `agent.executor.type` | `string` | Spec executor type | `"llm"`, `"claude_sdk"`, `"agents_sdk"`, `"remote"` |
| `agent.conversation.kind` | `string` | Conversation entity | `"default"`, `"sub_agent"` |
| `agent.background` | `bool` | Request param | `true`, `false` |
| `agent.executor.max_iterations` | `int` | Spec executor config | `25` |
| `agent.executor.timeout_seconds` | `int` | Spec executor config | `300` |
| `agent.modalities.input` | `string[]` | Spec modalities | `["text", "image", "file"]` |
| `agent.modalities.output` | `string[]` | Spec modalities | `["text"]` |
| `agent.previous_response_id` | `string` | Request param (if set) | `"resp_abc123"` |

**Span attributes on `agent_iteration` spans:**

| Attribute | Type | Source | Example |
|---|---|---|---|
| `agent.iteration.number` | `int` | Loop counter | `3` |
| `agent.iteration.input_message_count` | `int` | Length of messages list | `12` |
| `agent.iteration.tool_count` | `int` | Number of tools available | `8` |
| `agent.iteration.has_images` | `bool` | Input contains image content | `true` |
| `agent.iteration.has_files` | `bool` | Input contains file attachments | `true` |

**Span attributes on `chat` spans:**

| Attribute | Type | Req. Level | Source | Example |
|---|---|---|---|---|
| `gen_ai.provider.name` | `string` | Required | Parsed from model string prefix | `"openai"`, `"anthropic"` |
| `gen_ai.operation.name` | `string` | Required | Always `"chat"` | `"chat"` |
| `gen_ai.request.model` | `string` | Required | `LLMConfig.model` (without provider prefix) | `"gpt-5.4"` |
| `gen_ai.response.model` | `string` | Recommended | LLM response metadata | `"gpt-5.4-2025-03-01"` |
| `gen_ai.response.id` | `string` | Recommended | LLM response ID | `"chatcmpl-abc123"` |
| `gen_ai.response.finish_reasons` | `string[]` | Recommended | LLM response | `["stop"]`, `["tool_call"]` |
| `gen_ai.usage.input_tokens` | `int` | Recommended | LLM response usage | `1523` |
| `gen_ai.usage.output_tokens` | `int` | Recommended | LLM response usage | `847` |
| `gen_ai.usage.cache_read.input_tokens` | `int` | Recommended | LLM response usage | `1200` |
| `gen_ai.usage.cache_creation.input_tokens` | `int` | Recommended | LLM response usage | `500` |
| `gen_ai.request.max_tokens` | `int` | Recommended | From `LLMConfig.extra` `max_completion_tokens` | `4096` |
| `gen_ai.request.temperature` | `double` | Recommended | From `LLMConfig.extra` if set | `0.7` |
| `gen_ai.request.top_p` | `double` | Recommended | From `LLMConfig.extra` if set | `0.9` |
| `openai.request.reasoning_effort` | `string` | Recommended | From `LLMConfig.extra` `reasoning_effort` | `"high"`, `"medium"`, `"low"` |
| `gen_ai.input.messages` | `any` | Opt-In | Serialized input messages (if capture enabled) | JSON array |
| `gen_ai.output.messages` | `any` | Opt-In | Serialized output messages (if capture enabled) | JSON array |
| `gen_ai.system_instructions` | `any` | Opt-In | System prompt (if capture enabled) | JSON array |
| `gen_ai.tool.definitions` | `any` | Opt-In | Tool schemas (if capture enabled) | JSON array |

Any additional kwargs from `LLMConfig.extra` that are not mapped above
(e.g., provider-specific params like `frequency_penalty`,
`presence_penalty`, `stop_sequences`, or arbitrary keys) are recorded
as `gen_ai.request.<key>` when they match a known semconv attribute,
or omitted otherwise. The default executor passes all `extra` keys
through to the LLM client, so the set of params is open-ended.

**Images, files, and multimodal content in traces.**

When content capture is enabled, `gen_ai.input.messages` and
`gen_ai.output.messages` follow the GenAI semconv input/output
message JSON schemas, which define three part types for non-text
content:

| Part type | When to use | What's in the span |
|---|---|---|
| `uri` | Image/file passed as a URL | `{"type": "uri", "uri": "https://...", "modality": "image", "mime_type": "image/png"}` — lightweight reference, no bytes |
| `file` | File uploaded via agent-plane's file store | `{"type": "file", "file_id": "file_abc123", "modality": "image", "mime_type": "image/png"}` — reference by ID, retrievable via file API |
| `blob` | Inline base64 image/audio (small content only) | `{"type": "blob", "content": "<base64>", "modality": "image", "mime_type": "image/png"}` — actual bytes in the span |

**Content capture strategy:**

- **URL-referenced content** (e.g., `image_url` in OpenAI format):
  Record as a `uri` part. The URL is a lightweight reference.
- **File-store content** (uploaded via `POST /v1/files`): Record as
  a `file` part with the agent-plane file ID. The operator can
  retrieve the file contents separately via the file API.
- **Inline base64 content**: Record as a `blob` part **only if** the
  encoded size is under 64KB. Larger content is downgraded to a
  placeholder: `{"type": "blob", "modality": "image",
  "mime_type": "image/png", "content": "[truncated, 2.3MB]"}`.
  This prevents blowing past OTel backend attribute size limits.
- **Output images** (e.g., from `image_generation_call`): Same rules
  as input — prefer `uri` or `file` references, truncate large blobs.

When content capture is **disabled** (default), these parts are
omitted entirely along with all other message content — the
`gen_ai.input.messages` and `gen_ai.output.messages` attributes are
not set.

**Span attributes on `tool_call` spans:**

| Attribute | Type | Source | Example |
|---|---|---|---|
| `tool.name` | `string` | Tool schema | `"get_weather"` |
| `tool.call_id` | `string` | LLM response | `"call_abc123"` |
| `tool.type` | `string` | Tool registry | `"local"`, `"mcp"`, `"builtin"`, `"client"` |
| `tool.status` | `string` | Execution result | `"success"`, `"error"` |

**Span events on `chat` spans:**

| Event | When | Attributes |
|---|---|---|
| `gen_ai.retry` | On LLM retry attempt | `attempt`, `max_attempts`, `error.type`, `error.message`, `backoff_seconds` |

**Span events on `agent_iteration` spans:**

| Event | When | Attributes |
|---|---|---|
| `context_window_compaction` | After reactive compaction | `compaction.layer` (1=clear, 2=summarize, 3=truncate), `pre_compaction.message_count`, `post_compaction.message_count` |
| `native_tool_call` | When a provider-native tool is invoked | `tool.name` (e.g. `"web_search_call"`, `"code_interpreter_call"`) |

**Attribute namespace notes:**

- **Cache token attributes** are part of the standard GenAI semconv
  (both `openai.md` and `anthropic.md` list them as Recommended).
  Anthropic's `input_tokens` excludes cached tokens, so
  `gen_ai.usage.input_tokens` must be computed as
  `input_tokens + cache_read + cache_creation`. Record these when
  the LLM client returns cache breakdown fields.
- **`openai.request.reasoning_effort`** uses the `openai.*` provider
  namespace following the GenAI semconv pattern for provider-specific
  attributes (see `openai.request.service_tier` as precedent).
  Reasoning effort originates from the OpenAI Responses API. If the
  semconv promotes it to `gen_ai.request.*` as a cross-provider
  attribute, we rename to match.
| `gen_ai.input.messages` | `any` | Opt-In | Serialized input messages (if capture enabled) | JSON array of `ChatMessage` |
| `gen_ai.output.messages` | `any` | Opt-In | Serialized output messages (if capture enabled) | JSON array of `OutputMessage` |
| `gen_ai.system_instructions` | `any` | Opt-In | System prompt (if capture enabled) | JSON array of parts |

**Message content attributes are opt-in** — they can contain sensitive
data. When content capture is disabled (default), `gen_ai.input.messages`,
`gen_ai.output.messages`, and `gen_ai.system_instructions` are omitted
entirely from the span. This follows the GenAI semconv guidance on
sensitive data.

When the OTel SDK does not support structured (complex) span
attributes, message content is serialized as a JSON string on the
span attribute and recorded in structured form on the
`gen_ai.client.inference.operation.details` event.

**Note on renamed/deprecated attributes:**

| Deprecated (pre-v1.37.0) | Replaced By |
|---|---|
| `gen_ai.system` | `gen_ai.provider.name` |
| `gen_ai.user.message` (event) | `gen_ai.input.messages` (attribute) |
| `gen_ai.assistant.message` (event) | `gen_ai.output.messages` (attribute) |
| `gen_ai.tool.message` (event) | `gen_ai.input.messages` (attribute, tool role) |
| `gen_ai.choice` (event) | `gen_ai.output.messages` + `gen_ai.response.finish_reasons` |

### Metrics

| Metric | Type | Unit | Description |
|---|---|---|---|
| `gen_ai.client.token.usage` | Histogram | `{token}` | Input/output tokens per LLM call |
| `gen_ai.client.operation.duration` | Histogram | `s` | LLM call duration |
| `agent.tool.duration` | Histogram | `s` | Tool call duration |
| `agent.iteration.count` | Counter | `{iteration}` | Iterations per agent execution |

---

## Trace Identity and Session Grouping

### Problem

Operators need two capabilities the basic span hierarchy doesn't
provide:

1. **Search by response ID** — "show me the trace for `resp_d8e9f0a1...`"
   without maintaining a separate response-ID-to-trace-ID mapping.
2. **Group traces by conversation** — "show me all traces in this
   multi-turn conversation" to debug a session end-to-end.

### Response ID as Trace ID

Agent-plane response IDs have the format `resp_<32-char hex>`, where
the hex suffix is a UUID4 hex string — exactly 128 bits, which is
exactly the OTel trace ID format. We exploit this:

**The trace ID for a response's trace is the hex suffix of its
response ID.**

| Response ID | Trace ID |
|---|---|
| `resp_d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3` | `d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3` |

This means an operator can go from a response ID to its trace by
stripping the `resp_` prefix and pasting the hex into Jaeger, Tempo,
or any trace backend. No lookup table, no log correlation, no extra
query.

#### Implementation

The OTel Python SDK supports custom trace IDs via a custom
`IdGenerator` or by constructing a `SpanContext` manually. We use
the latter — when creating the root span for a response, we inject
the response ID's hex as the trace ID:

```python
from opentelemetry import trace
from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan

def _trace_id_from_response_id(response_id: str) -> int:
    """
    Extract the OTel trace ID from an agent-plane response ID.

    Response IDs have the format ``resp_<32-char hex>``.
    The hex suffix is a valid 128-bit OTel trace ID.

    :param response_id: The response/task ID,
        e.g. ``"resp_d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3"``.
    :returns: The trace ID as a 128-bit integer.
    :raises ValueError: If the response ID format is invalid.
    """
    prefix = "resp_"
    if not response_id.startswith(prefix):
        raise ValueError(f"Expected resp_ prefix: {response_id!r}")
    hex_part = response_id[len(prefix):]
    if len(hex_part) != 32:
        raise ValueError(
            f"Expected 32 hex chars after prefix: {response_id!r}"
        )
    return int(hex_part, 16)
```

In `workflow.py`, the root `agent_execution` span is started with
this trace ID:

```python
trace_id = _trace_id_from_response_id(task_id)
span_context = SpanContext(
    trace_id=trace_id,
    span_id=trace.get_tracer_provider()
        .id_generator.generate_span_id(),
    is_remote=False,
    trace_flags=TraceFlags(TraceFlags.SAMPLED),
)
parent_ctx = trace.set_span_in_context(
    NonRecordingSpan(span_context)
)
with _tracer.start_as_current_span(
    "agent_execution", context=parent_ctx, ...
):
    ...
```

All child spans (iterations, LLM calls, tool calls, sub-agents)
inherit this trace ID automatically via OTel context propagation.

#### Compatibility

- **OTel**: Valid. Trace IDs must be 32 lowercase hex chars and
  non-zero. UUID4 hex satisfies both constraints (the probability
  of an all-zero UUID4 is effectively zero).
- **OpenResponses**: Compatible. The response ID format is
  `resp_<hex>`, which we already generate from `uuid.uuid4().hex`.
  No change to ID generation is needed — we are simply reusing the
  hex portion as the trace ID rather than generating a separate one.

### Session Grouping via `session.id`

The OTel semantic conventions define `session.id` (experimental) as
a span attribute for grouping related traces. The GenAI semantic
conventions recommend it for multi-turn conversation correlation.

**We set `session.id` = conversation ID on all root spans.**

| Conversation ID | `session.id` attribute |
|---|---|
| `conv_e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9` | `conv_e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9` |

We use the full conversation ID (including the `conv_` prefix) as
`session.id`, not just the hex suffix. Unlike trace IDs, `session.id`
is a free-form string with no format constraints. Keeping the prefix
makes it immediately recognizable and copy-pasteable between the
trace backend and agent-plane's conversation API.

#### What this enables

In any OTel-compatible backend:

- **Jaeger**: Filter by tag `session.id = conv_e4f5a6b7...` to see
  all traces (turns) in a conversation.
- **Grafana Tempo**: TraceQL query
  `{ span.session.id = "conv_e4f5a6b7..." }` returns all traces in
  the session.
- **Datadog**: Filter by `@session.id` tag.

Combined with response-ID-as-trace-ID, an operator can:
1. Find a specific response's trace by its ID (strip `resp_` prefix).
2. Find all traces in the same conversation via `session.id`.
3. See the full span tree within each trace (LLM calls, tools,
   sub-agents).

#### Implementation

In the `workflow_span` helper:

```python
def workflow_span(
    agent_id: str,
    task_id: str,
    conversation_id: str,
    agent_name: str | None,
):
    """..."""
    return _tracer.start_as_current_span(
        "agent_execution",
        attributes={
            "agent.id": agent_id,
            "task.id": task_id,
            "conversation.id": conversation_id,
            "agent.name": agent_name or "",
            "session.id": conversation_id,  # OTel session grouping
        },
    )
```

The `session.id` attribute is set on the root span only. Child spans
do not need it — trace backends group by root span attributes, and
the trace ID already links all spans in a single response.

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
--otel-capture-content            Enable capturing message content in span attributes (default: off)
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
    :param capture_content: Whether to include message content in span attributes.
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
   differences — adding a separate auto-instrumentation library
   would double-layer the abstraction.
2. We need spans at our semantic boundaries (workflow, iteration, tool
   call), not at the LLM client's internal boundaries.
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


def chat_span(model: str, provider: str, operation: str = "chat"):
    """
    Create a GenAI semantic convention span for an LLM call.

    :param model: The model identifier without provider prefix, e.g. "gpt-5.4".
    :param provider: The GenAI provider name, e.g. "openai", "anthropic".
    :param operation: The GenAI operation name, e.g. "chat".
    """
    return _tracer.start_as_current_span(
        f"chat {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "gen_ai.operation.name": operation,
        },
    )


def record_llm_usage(span, usage: dict) -> None:
    """
    Record token usage attributes on a chat span.

    :param span: The active chat span.
    :param usage: Dict with "input_tokens" and "output_tokens" keys.
    """
    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
    # Cache breakdown — record when the LLM client returns them.
    # Anthropic's input_tokens excludes cached tokens; the semconv
    # requires input_tokens = base + cache_read + cache_creation.
    if "cache_read_input_tokens" in usage:
        span.set_attribute(
            "gen_ai.usage.cache_read.input_tokens",
            usage["cache_read_input_tokens"],
        )
    if "cache_creation_input_tokens" in usage:
        span.set_attribute(
            "gen_ai.usage.cache_creation.input_tokens",
            usage["cache_creation_input_tokens"],
        )


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
provider, model = parse_model_string(full_model)  # "openai/gpt-5.4" → ("openai", "gpt-5.4")

with telemetry.chat_span(model, provider) as span:
    response = client.responses.create(...)
    if response.usage:
        telemetry.record_llm_usage(span, response.usage)
    span.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
    if telemetry.should_capture_content():
        # Structured attributes (v1.37.0+). Serialized as JSON
        # strings if the SDK does not support complex attributes.
        span.set_attribute("gen_ai.input.messages", json.dumps(input_messages))
        span.set_attribute("gen_ai.output.messages", json.dumps(output_messages))
```

**Streaming**: The span stays open for the duration of the stream.
Message content attributes are recorded when the stream completes and
the accumulated response is available.

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

#### 6. Failure, Recovery, and Non-LLM Retries

The DBOS durability layer introduces failure modes that don't exist in
a stateless server. The observability layer must surface these without
exposing DBOS internals.

**Workflow failure.** When the top-level exception handler in
`agent_execution_workflow()` catches an unrecoverable error, the
`agent_execution` span must record the failure:

```python
except Exception as exc:
    span = trace.get_current_span()
    span.set_status(trace.StatusCode.ERROR, str(exc))
    span.set_attribute("error.type", type(exc).__name__)
    span.record_exception(exc)
    raise
```

This sets the OTel span status to ERROR, which trace backends render
as a red/failed span. The `error.type` attribute enables filtering
traces by exception class (e.g. "show me all PermanentLLMError
failures").

**Crash recovery.** When the server crashes and restarts, the
durability layer replays pending workflows from their last checkpoint.
Because we derive trace ID from response ID (see "Response ID as
Trace ID"), the recovered workflow produces spans with the **same
trace ID** as the original execution — an operator sees one trace per
response, with both pre-crash and post-recovery spans.

Replayed `@step` calls (LLM calls, tool calls) return cached results
**without executing the function body**. Since our span creation
lives inside those function bodies, no phantom spans are emitted for
replayed steps. Only steps that actually re-execute produce spans.
This is correct by construction — no special handling needed.

**Pre-crash span loss.** Spans emitted before a crash may not have
been exported (the `BatchSpanProcessor` flushes periodically, not on
every span). The trace may have a gap between pre-crash spans and
post-recovery spans. This is inherent to crash recovery — operators
should expect partial pre-crash data.

**Recovery exhaustion.** When the durability layer exhausts recovery
attempts, the workflow transitions to a terminal failed status. If it
never re-enters our instrumented code, no span is emitted for the
final failure. The task store records the status, and the operator
can correlate via response ID.

**Context window overflow and compaction.** When the LLM returns a
context-window-exceeded error, the workflow performs reactive
compaction (summarize history, retry). This should be visible in the
trace as a span event on the iteration span:

```python
span.add_event("context_window_compaction", {
    "pre_compaction.message_count": pre_count,
    "post_compaction.message_count": post_count,
    "compaction.reason": "context_window_exceeded",
})
```

The subsequent retry LLM call gets its own `chat` span as normal.

**Steering retries.** When a late steering message arrives after
`close_inbox()`, the agent loop retries with the new input. This
produces a new iteration span (with a new `agent.iteration.number`),
which is the correct representation — it's genuinely a new iteration,
not a retry of the previous one. No special handling needed.

#### 7. Log Correlation

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
`GEN_AI_PROVIDER_NAME`). We use these constants instead of raw strings
to track upstream renames as the conventions stabilize. Note: the
package must be >= v0.50b0 to include the v1.37.0+ attributes
(`gen_ai.provider.name`, `gen_ai.input.messages`, etc.).

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

1. Implement gated content capture (message content as span attributes).
2. Add `gen_ai.client.token.usage` histogram.
3. Add `gen_ai.client.operation.duration` histogram.
4. Add `agent.tool.duration` histogram.

### Phase 4: Log Correlation

**Files:** Modify `agent_plane/runtime/telemetry.py`

1. Bridge Python `logging` to OTel LoggerProvider.
2. Verify existing log call sites get trace/span IDs.

---

## Executor Trace Capture

The core spans (Phases 1–4 above) cover the **native executor**
(`DefaultExecutor`), where agent-plane itself calls the LLM and runs
tool invocations in-process. But the **Claude Agent SDK** and **OpenAI
Agents SDK** executors run in separate processes with their own LLM
clients and tool loops. Without explicit instrumentation, their
internal work is invisible — the `agent_execution` span shows a single
opaque block with no child spans for the executor's LLM calls, tool
invocations, or sub-agent spawns.

### Goal

Executor subprocess spans nest under the parent `agent_execution` span
in the same trace. An operator viewing a trace sees:

```
[Workflow] agent_execution {agent.id, task.id}
  └─ [Executor] claude_sdk / agents_sdk
       ├─ [GenAI] chat {model}          ← executor's LLM call
       ├─ [Tool] tool_call {name}       ← executor's tool call
       └─ [GenAI] chat {model}          ← second iteration
```

### Trace Context Propagation

Both executor types launch subprocesses. Agent-plane must inject the
W3C `traceparent` header as an environment variable so the child
process joins the parent trace.

```python
# In telemetry.py:
from opentelemetry import context, trace
from opentelemetry.trace.propagation import TraceContextTextMapPropagator

def get_traceparent_env() -> dict[str, str]:
    """
    Serialize the current trace context into env vars for subprocess
    inheritance.

    :returns: Dict with ``TRACEPARENT`` (and optionally ``TRACESTATE``)
        keys, or empty dict if no active span.
    """
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    # Rename to env-var convention (uppercase).
    env = {}
    if "traceparent" in carrier:
        env["TRACEPARENT"] = carrier["traceparent"]
    if "tracestate" in carrier:
        env["TRACESTATE"] = carrier["tracestate"]
    return env
```

### Per-Executor Activation

Each SDK has a different mechanism for enabling OTel tracing within
the subprocess. Agent-plane injects the necessary env vars alongside
`TRACEPARENT` when telemetry is enabled.

#### Claude Agent SDK (`claude_sdk` executor)

The Claude Agent SDK has **native OTel support** inherited from the
Claude Code runtime. It is activated entirely via environment variables.

**Env vars to inject** (in `executors/claude.py`, merged into the
`env` dict passed to `ClaudeAgentOptions`):

```python
# Only when agent-plane telemetry is enabled (exporter != "none"):
otel_env = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",  # required for traces
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": config.exporter,       # "grpc" or "http"
    "OTEL_EXPORTER_OTLP_ENDPOINT": config.endpoint,
    **telemetry.get_traceparent_env(),                     # TRACEPARENT
}
if config.capture_content:
    otel_env["OTEL_LOG_USER_PROMPTS"] = "1"
    otel_env["OTEL_LOG_TOOL_DETAILS"] = "1"
    otel_env["OTEL_LOG_TOOL_CONTENT"] = "1"
if config.headers:
    otel_env["OTEL_EXPORTER_OTLP_HEADERS"] = ",".join(
        f"{k}={v}" for k, v in config.headers.items()
    )
```

**What it emits**: LLM call spans, tool execution spans, sub-agent
spans, token usage metrics, structured events — all following the
Claude Agent SDK's built-in OTel schema. The `TRACEPARENT` env var
causes these spans to nest under agent-plane's `agent_execution` span.

**No additional dependencies** — the Claude Agent SDK bundles its own
OTel instrumentation.

**Limitation**: Trace support in the Claude Agent SDK is currently
beta (requires `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`). Monitor
upstream releases for GA promotion and flag removal.

#### OpenAI Agents SDK (`agents_sdk` executor)

The OpenAI Agents SDK has its **own proprietary tracing system** (not
OTel). An official OTel bridge is maintained in the
`opentelemetry-python-contrib` repository.

**Dependency** (added to `pyproject.toml`):

```toml
dependencies = [
    # ... existing deps ...
    "opentelemetry-instrumentation-openai-agents-v2>=0.1",
]
```

**Activation** — the executor entrypoint must call `instrument()`
once before any agent runs. In `executors/agents_sdk.py`, during
executor initialization:

```python
# Only when agent-plane telemetry is enabled:
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

OpenAIAgentsInstrumentor().instrument(
    tracer_provider=trace.get_tracer_provider(),
)
```

Because the OpenAI Agents SDK runs **in-process** (not a subprocess),
it inherits the `TracerProvider` already configured by
`telemetry.init()`. No `TRACEPARENT` injection is needed — the OTel
context propagates naturally through Python's context vars. The Codex
MCP subprocess is a child process, but its spans are captured by the
SDK's instrumentation layer before they cross the process boundary.

**Env vars for content capture** (set on the process, not subprocess):

```python
if config.capture_content:
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "span_and_event"
```

**What it emits**: Agent spans, generation (LLM) spans, function/tool
spans, guardrail spans, handoff spans — following OTel GenAI semantic
conventions. Attributes include model, token counts, prompts (when
content capture is enabled).

**Limitation**: Setting `OPENAI_AGENTS_DISABLE_TRACING=1` disables
both the SDK's native tracing and the OTel instrumentation layer.
Agent-plane must not set this env var.

#### Remote Executor

The `RemoteExecutor` communicates over HTTP. If the remote service
supports W3C trace context, agent-plane should inject `traceparent`
as an HTTP header on the SSE request. This is standard `httpx` +
OTel integration — `opentelemetry-instrumentation-httpx` handles it
automatically if installed.

**Optional dependency** (added to `pyproject.toml`):

```toml
dependencies = [
    # ... existing deps ...
    "opentelemetry-instrumentation-httpx>=0.44b0",   # auto-injects traceparent on httpx calls
]
```

### Implementation Plan

#### Phase 5: Executor Trace Capture

**Files:** Modify `agent_plane/runtime/telemetry.py`,
`agent_plane/runtime/executors/claude.py`,
`agent_plane/runtime/executors/agents_sdk.py`

1. Add `get_traceparent_env()` to `telemetry.py`.
2. In `ClaudeAgentsExecutor`: merge OTel env vars into the `env`
   dict passed to `ClaudeAgentOptions` when telemetry is enabled.
3. In `AgentsSdkExecutor`: call `OpenAIAgentsInstrumentor().instrument()`
   during executor init when telemetry is enabled.
4. Add `opentelemetry-instrumentation-openai-agents-v2` to
   `pyproject.toml` dependencies.
5. Optionally add `opentelemetry-instrumentation-httpx` for remote
   executor trace propagation.
6. Verify end-to-end: start a collector (e.g. Jaeger), run each
   executor type, confirm child spans nest under `agent_execution`.

---

## Not Yet

- **Inbound trace context** — if the caller of `POST /v1/responses`
  sends a `traceparent` header, we should join their trace rather
  than starting a new root. Separate from executor trace capture
  (which is outbound context propagation).
- **Custom span processors** (e.g., sampling strategies beyond the OTel
  SDK defaults). Use `OTEL_TRACES_SAMPLER` env var for now.
- **Dashboard templates** — pre-built Grafana/Datadog dashboards for
  agent execution metrics. Useful but out of scope for the runtime.
- **Durability layer span bridging** — the durability layer has its
  own internal tracing for checkpoint/replay mechanics. We surface
  failure and recovery at our semantic boundaries (workflow failure
  via span status, compaction events) but don't expose internal
  checkpoint spans. Bridging the two is possible but not a priority.
- **Synthetic span for recovery exhaustion** — when the durability
  layer gives up without re-entering our code, no span is emitted.
  A future enhancement could emit a synthetic error span or metric.
- **Per-agent telemetry overrides** — e.g., an agent spec that says
  "always capture content for this agent." Telemetry is operator
  config, not agent config, for now.
- **Prompt/completion token cost estimation** — using model pricing
  tables to convert token counts to dollar amounts. Interesting for
  dashboards but not core to the observability layer.
