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
5. **Runtime configuration** — operators configure telemetry via
   environment variables, not agent specs. Agent authors should not
   need to think about observability.
6. **Zero overhead when disabled** — no performance impact when no
   exporter is configured.

---

## What We Capture

### Span Hierarchy

```
[HTTP] POST /v1/responses
  └─ [Agent] invoke_agent my-coder {gen_ai.agent.id, session.id}
       ├─ [GenAI] chat gpt-5.4 {gen_ai.provider.name, gen_ai.conversation.id}
       │    attrs: gen_ai.input.messages, gen_ai.output.messages,
       │           gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
       ├─ [Tool] execute_tool get_weather {tool.name, tool.type}
       │    └─ [MCP] tools/call get_weather {mcp.method.name}         ← if MCP tool
       ├─ [GenAI] chat gpt-5.4 {gen_ai.provider.name, gen_ai.conversation.id}
       │    attrs: gen_ai.input.messages (includes tool result),
       │           gen_ai.output.messages, gen_ai.usage.*
       └─ [Agent] invoke_agent sub-coder {gen_ai.agent.id, ...}      ← sub-agent
            ├─ [GenAI] chat ...
            └─ [Tool] execute_tool ...
```

### Operations and Their Spans

| Operation | Span Name | Kind | Key Attributes |
|---|---|---|---|
| HTTP request | `POST /v1/responses` | `SERVER` | Standard HTTP semconv (auto-instrumented) |
| Agent invocation | `invoke_agent {gen_ai.agent.name}` | `INTERNAL` | `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.conversation.id`, `session.id` |
| Agent loop iteration | `agent_iteration` | `INTERNAL` | `agent.iteration.number`, `agent.iteration.input_message_count`, `agent.iteration.tool_count` |
| LLM call | `chat {gen_ai.request.model}` | `CLIENT` | GenAI semconv (see below) |
| Tool execution | `execute_tool {tool.name}` | `INTERNAL` | `gen_ai.operation.name = "execute_tool"`, `tool.name`, `tool.call_id` |
| MCP tool call | `tools/call {tool.name}` | `CLIENT` | `mcp.method.name = "tools/call"`, `mcp.session.id`, `server.address` |
| Sub-agent spawn | `invoke_agent {gen_ai.agent.name}` | `INTERNAL` | Same as agent invocation, nested under parent |
| LLM retry | (no new span) | — | Retry events on the `chat` span |
| Tool retry | (no new span) | — | Retry events on the `execute_tool` span |

### GenAI Semantic Convention Attributes

Following the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(**v1.37.0+** — the v1.37.0 release consolidated per-message span
events into structured span attributes; older per-message events like
`gen_ai.user.message`, `gen_ai.choice`, etc. are deprecated).

**Span attributes on `invoke_agent` spans:**

These follow the [GenAI Agent Spans semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/).
Standard attributes use `gen_ai.*` names; agent-plane-specific
attributes (no semconv equivalent) use `agent.*`.

| Attribute | Type | Semconv? | Source | Example |
|---|---|---|---|---|
| `gen_ai.operation.name` | `string` | Yes | Always `"invoke_agent"` | `"invoke_agent"` |
| `gen_ai.provider.name` | `string` | Yes | Parsed from agent spec LLM model prefix | `"openai"`, `"anthropic"` |
| `gen_ai.agent.id` | `string` | Yes | Agent entity | `"ag_abc123"` |
| `gen_ai.agent.name` | `string` | Yes | Agent spec | `"my-coder"` |
| `gen_ai.agent.description` | `string` | Yes | Agent spec (if set) | `"A coding assistant"` |
| `gen_ai.agent.version` | `string` | Yes | Agent spec (if set) | `"1.0.0"` |
| `gen_ai.conversation.id` | `string` | Yes | Conversation entity | `"conv_e4f5a6b7..."` |
| `gen_ai.request.model` | `string` | Yes | Agent spec LLM config | `"gpt-5.4"` |
| `task.id` | `string` | No | Response/task ID | `"resp_d8e9f0a1..."` |
| `session.id` | `string` | Yes | = conversation.id (OTel session grouping) | `"conv_e4f5a6b7..."` |
| `agent.executor.type` | `string` | No | Spec executor type | `"llm"`, `"claude_sdk"`, `"agents_sdk"`, `"remote"` |
| `agent.conversation.kind` | `string` | No | Conversation entity | `"default"`, `"sub_agent"` |
| `agent.background` | `bool` | No | Request param | `true`, `false` |
| `agent.executor.max_iterations` | `int` | No | Spec executor config | `25` |
| `agent.executor.timeout_seconds` | `int` | No | Spec executor config | `300` |
| `agent.modalities.input` | `string[]` | No | Spec modalities | `["text", "image", "file"]` |
| `agent.modalities.output` | `string[]` | No | Spec modalities | `["text"]` |
| `agent.previous_response_id` | `string` | No | Request param (if set) | `"resp_abc123"` |

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
| `gen_ai.conversation.id` | `string` | Cond. Required | Conversation entity | `"conv_e4f5a6b7..."` |
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

OTel has no native concept of span attachments or binary content.
The GenAI semconv's `blob` part type puts base64 inline in span
attributes, which doesn't scale (most backends cap attributes at
64KB–1MB). Instead, we use **external references** — the span
contains a lightweight pointer, and the operator retrieves the
actual content from agent-plane's file store.

Agent-plane already has a file store (`POST /v1/files`,
`GET /v1/files/{id}/content`). Images and files uploaded to an
agent are stored there with stable IDs (`file_<hex>`). This is the
natural retrieval layer — no new storage, no OTel backend
limitations, no proprietary attachment protocol.

**Content capture strategy** (when content capture is enabled):

When building `gen_ai.input.messages` and `gen_ai.output.messages`,
non-text content is represented as follows:

- **File-store content** (uploaded via `POST /v1/files`): Reference
  by file ID. The operator retrieves the content via the file API.
  ```json
  {"type": "file", "file_id": "file_abc123",
   "modality": "image", "mime_type": "image/png"}
  ```

- **URL-referenced content** (e.g., `image_url` in OpenAI format):
  Record the URL directly — no bytes in the span.
  ```json
  {"type": "uri", "uri": "https://example.com/photo.png",
   "modality": "image", "mime_type": "image/png"}
  ```

- **Inline base64 content** (not uploaded as a file attachment):
  Recorded as-is using the GenAI semconv `blob` part type. The
  base64 data stays in the span attribute. This can be large, but
  we do **not** truncate or persist to the file store —
  observability must not have side effects on storage. Operators
  should be aware that enabling content capture on agents with
  heavy inline image usage will produce large span attributes.
  ```json
  {"type": "blob", "content": "<base64>",
   "modality": "image", "mime_type": "image/png"}
  ```

- **Output images** (e.g., from `image_generation_call`): Same rules
  — file store references if the output was persisted as a file,
  inline `blob` otherwise.

**Retrieval workflow for operators:**

1. Find the trace in the OTel backend (Jaeger, Tempo, etc.).
2. Inspect `gen_ai.input.messages` — see `file` or `blob` parts.
3. For `file` parts: call `GET /v1/files/{file_id}/content` to
   download the image/file.
4. For `blob` parts: decode the base64 content directly.

**Span attribute size limits.** OTel backends may cap attribute
values (Jaeger defaults to 256 bytes). However, MLflow's OTLP
receiver stores span attributes at full fidelity with no size cap
(it only truncates trace-level metadata previews to 250 chars for
the table UI). For non-MLflow backends, operators should configure
`OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT` appropriately — or accept
that very long conversations will be truncated by the backend. We do
not truncate on the agent-plane side because silently losing
conversation content from traces would undermine their diagnostic
value.

When content capture is **disabled** (default), these parts are
omitted entirely — `gen_ai.input.messages` and
`gen_ai.output.messages` are not set.

**Span attributes on `execute_tool` spans:**

| Attribute | Type | Semconv? | Source | Example |
|---|---|---|---|---|
| `gen_ai.operation.name` | `string` | Yes | Always `"execute_tool"` | `"execute_tool"` |
| `tool.name` | `string` | Yes | Tool schema | `"get_weather"` |
| `tool.call_id` | `string` | No | LLM response | `"call_abc123"` |
| `tool.type` | `string` | No | Tool registry | `"local"`, `"mcp"`, `"builtin"`, `"client"` |
| `tool.status` | `string` | No | Execution result | `"success"`, `"error"` |

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
- **`gen_ai.conversation.id` vs `session.id`** — both carry the
  conversation ID but serve different purposes. `session.id` is a
  general OTel attribute on the **root span** used by generic backends
  (Jaeger, Tempo) to group traces by session. `gen_ai.conversation.id`
  is a GenAI semconv attribute on **chat spans** used by GenAI-aware
  backends (MLflow) to correlate LLM calls across turns and compute
  per-conversation token totals. Same value, different consumers,
  different span levels. We set both.

**Note on renamed/deprecated attributes (v1.37.0):**

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

In `workflow.py`, the root `invoke_agent` span is started with this
trace ID:

```python
trace_id = _trace_id_from_response_id(response_id)
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
    f"invoke_agent {agent_name}", context=parent_ctx, ...
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

In the `invoke_agent_span` helper (see Tracer Module), `session.id`
is set alongside the standard `gen_ai.conversation.id`:

```python
attributes={
    "gen_ai.conversation.id": conversation_id,
    "session.id": conversation_id,  # OTel session grouping
    ...
}
```

The `session.id` attribute is set on the root span only. Child spans
do not need it — trace backends group by root span attributes, and
the trace ID already links all spans in a single response.

---

## Configuration

Telemetry is **runtime configuration** — it belongs to the operator, not
the agent author. Agent specs (`config.yaml`) have no telemetry fields.

### Environment Variables

All OTel configuration uses **standard OTel environment variables**.
Agent-plane does not introduce its own CLI flags for exporter config —
operators already know these env vars, and duplicating them adds
unnecessary complexity.

**Standard OTel env vars** (handled by the OTel SDK automatically):

| Env Var | Purpose | Example |
|---|---|---|
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol | `"grpc"`, `"http/protobuf"` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint | `"http://localhost:4317"` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Extra headers (auth, routing) | `"x-mlflow-experiment-id=0,Authorization=Bearer tok"` |
| `OTEL_SERVICE_NAME` | Service name in resource attributes | `"agent-plane"` |
| `OTEL_TRACES_SAMPLER` | Sampling strategy | `"always_on"`, `"parentbased_traceidratio"` |
| `OTEL_TRACES_SAMPLER_ARG` | Sampler argument (e.g., ratio) | `"0.1"` |

**Agent-plane-specific env var** (the only custom one):

| Env Var | Purpose | Default |
|---|---|---|
| `AGENT_PLANE_OTEL_CAPTURE_CONTENT` | Include message content in span attributes | `"false"` |

**Telemetry is enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.**
When unset, the OTel SDK installs NoOp providers — zero overhead, no
spans exported. No explicit "enable/disable" flag is needed.

### Examples

```bash
# Jaeger (gRPC)
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=agent-plane

# MLflow (HTTP, requires experiment ID header)
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5000
export OTEL_EXPORTER_OTLP_HEADERS="x-mlflow-experiment-id=0"
export OTEL_SERVICE_NAME=agent-plane

# With content capture and auth
export AGENT_PLANE_OTEL_CAPTURE_CONTENT=true
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer mytoken"
```

### Initialization Flow

```
cli.py: read env vars
  │
  ▼
telemetry.init()
  ├─ If OTEL_EXPORTER_OTLP_ENDPOINT not set: NoOp providers → return
  ├─ Create Resource(service.name from OTEL_SERVICE_NAME or "agent-plane")
  ├─ Create TracerProvider (OTel SDK reads OTEL_EXPORTER_* env vars)
  ├─ Create MeterProvider (same)
  ├─ Create LoggerProvider (same)
  ├─ Set global providers: trace.set_tracer_provider(...)
  ├─ Instrument FastAPI (auto-instrumentation for HTTP spans)
  └─ Read AGENT_PLANE_OTEL_CAPTURE_CONTENT into module-level flag
```

The OTel SDK handles exporter creation, endpoint configuration,
header injection, and protocol selection from env vars automatically.
Agent-plane's `telemetry.init()` just creates providers and registers
them — it does not parse or proxy OTel env vars.

### Trace Backend Configuration

Agent-plane exports standard OTLP. The operator chooses the backend
by setting `OTEL_EXPORTER_OTLP_ENDPOINT`. No agent-plane code changes
are needed per backend.

#### Generic OTLP collector (Jaeger, Grafana Tempo, Datadog)

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

The collector receives traces, metrics, and logs via OTLP. Searching
by response ID works by pasting the trace ID (hex suffix of `resp_*`)
into the backend's trace lookup UI.

#### MLflow Tracking Server

The MLflow tracking server exposes an OTLP receiver at `/v1/traces`.
It accepts standard OTLP/HTTP protobuf payloads but requires one
additional header: `x-mlflow-experiment-id`, which specifies which
MLflow experiment the traces belong to. This header is **required** —
the endpoint rejects requests without it.

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5000
export OTEL_EXPORTER_OTLP_HEADERS="x-mlflow-experiment-id=0"
```

The `OTEL_EXPORTER_OTLP_HEADERS` env var passes through to the OTLP
exporter's request headers. The experiment ID is a string (e.g.,
`"0"` for the default experiment, or a numeric/named experiment ID).

MLflow receives the OTLP spans and translates them using its GenAI
semantic convention translator — `gen_ai.operation.name` maps to
MLflow span types (`CHAT_MODEL`, `TOOL`, `AGENT`, etc.), token
usage is aggregated automatically, and traces appear in the MLflow
Traces UI.

**MLflow-specific features that work automatically:**

- **Token aggregation**: MLflow sums `gen_ai.usage.input_tokens` and
  `gen_ai.usage.output_tokens` across all spans in a trace.
- **Span type mapping**: `gen_ai.operation.name = "chat"` →
  `CHAT_MODEL`, tool spans → `TOOL`, workflow spans → `AGENT`.
- **Search**: `mlflow.search_traces(filter_string='tags.task.id =
  "resp_abc123"')` finds traces by response ID via span attributes
  promoted to tags.

**What MLflow does NOT provide when used as a pure OTLP receiver:**

- File/image attachments (those require using the MLflow Tracing SDK
  directly, not raw OTLP). Our file-store reference strategy still
  works — the operator retrieves files via the agent-plane file API.
- `client_request_id` (only set via MLflow SDK, not inferred from
  OTLP spans). Operators search by `task.id` tag instead.

#### Dual export (multiple backends)

To send traces to both MLflow and a generic collector, deploy an
[OTel Collector](https://opentelemetry.io/docs/collector/) as a
fan-out proxy:

```
agent-plane → OTel Collector → MLflow (/v1/traces)
                             → Jaeger (OTLP gRPC)
                             → Grafana Tempo (OTLP HTTP)
```

The OTel Collector's `exporters` config routes traces to multiple
backends. Agent-plane still exports to a single endpoint (the
collector). This is standard OTel infrastructure — no agent-plane
code involved.

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


def init() -> None:
    """
    Initialize OpenTelemetry providers and exporters.

    Reads all configuration from standard OTel env vars
    (OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME, etc.)
    and the agent-plane-specific AGENT_PLANE_OTEL_CAPTURE_CONTENT.
    """
    global _capture_content
    _capture_content = os.environ.get(
        "AGENT_PLANE_OTEL_CAPTURE_CONTENT", ""
    ).lower() in ("true", "1")

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return  # NoOp providers already installed by default

    # OTel SDK reads OTEL_EXPORTER_OTLP_* env vars automatically
    # when creating exporters — no manual config parsing needed.
    # ... create and register providers, exporters


def invoke_agent_span(
    agent_id: str,
    response_id: str,
    conversation_id: str,
    agent_name: str | None,
    executor_type: str,
    conversation_kind: str,
    background: bool,
):
    """
    Create an invoke_agent span following GenAI Agent Spans semconv.

    :param agent_id: The agent's unique identifier, e.g. ``"ag_abc123"``.
    :param response_id: The response identifier, e.g. ``"resp_d8e9f0a1..."``.
    :param conversation_id: The conversation identifier, e.g. ``"conv_e4f5..."``.
    :param agent_name: Human-readable agent name, if available.
    :param executor_type: Executor type, e.g. ``"llm"``, ``"claude_sdk"``.
    :param conversation_kind: ``"default"`` or ``"sub_agent"``.
    :param background: Whether this is a background task.
    """
    span_name = f"invoke_agent {agent_name}" if agent_name else "invoke_agent"
    return _tracer.start_as_current_span(
        span_name,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": provider,
            "gen_ai.agent.id": agent_id,
            "gen_ai.agent.name": agent_name or "",
            "gen_ai.conversation.id": conversation_id,
            "task.id": response_id,
            "session.id": conversation_id,
            "agent.executor.type": executor_type,
            "agent.conversation.kind": conversation_kind,
            "agent.background": background,
        },
    )


def iteration_span(iteration_number: int, input_message_count: int, tool_count: int):
    """
    Create a span for one iteration of the agent loop.

    :param iteration_number: 1-based iteration index.
    :param input_message_count: Number of messages in the LLM prompt.
    :param tool_count: Number of tools available to the LLM.
    """
    return _tracer.start_as_current_span(
        "agent_iteration",
        attributes={
            "agent.iteration.number": iteration_number,
            "agent.iteration.input_message_count": input_message_count,
            "agent.iteration.tool_count": tool_count,
        },
    )


def chat_span(
    model: str,
    provider: str,
    conversation_id: str,
    operation: str = "chat",
):
    """
    Create a GenAI semantic convention span for an LLM call.

    :param model: The model identifier without provider prefix, e.g. "gpt-5.4".
    :param provider: The GenAI provider name, e.g. "openai", "anthropic".
    :param conversation_id: The conversation ID for cross-turn correlation.
    :param operation: The GenAI operation name, e.g. "chat".
    """
    return _tracer.start_as_current_span(
        f"chat {model}",
        kind=trace.SpanKind.CLIENT,
        attributes={
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "gen_ai.operation.name": operation,
            "gen_ai.conversation.id": conversation_id,
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


def execute_tool_span(tool_name: str, call_id: str, tool_type: str):
    """
    Create an execute_tool span following GenAI Agent Spans semconv.

    :param tool_name: The tool's registered name.
    :param call_id: The unique call identifier from the LLM response.
    :param tool_type: Tool category: "local", "mcp", "builtin", "client".
    """
    return _tracer.start_as_current_span(
        f"execute_tool {tool_name}",
        attributes={
            "gen_ai.operation.name": "execute_tool",
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
with telemetry.invoke_agent_span(agent_id, response_id, conversation_id, spec.name, ...):
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

with telemetry.chat_span(model, provider, conversation_id) as span:
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
with telemetry.execute_tool_span(tool_name, call_id, tool.tool_type) as span:
    result = execute_tool_with_retry(...)
    span.set_attribute("tool.status", "success" if not is_error else "error")
```

#### 5. Sub-Agent Spans

Sub-agent workflows (`spawn` tool) create nested `agent_execution`
spans **within the root response's trace**. The trace ID is always
derived from the root response ID, not the sub-agent's own response
ID. This means the entire execution tree — root agent, all
sub-agents, their sub-agents — appears as one trace.

The sub-agent's own response ID is recorded as `task.id` on its
`agent_execution` span, so operators can still identify which
sub-agent produced which span. But trace lookup by response ID only
works for the root response — sub-agent response IDs are span
attributes, not trace IDs.

For synchronous sub-agents (same thread), OTel context propagation
handles nesting automatically. For async sub-agents, the spawn tool
must explicitly propagate the trace context to the child workflow
(pass the parent span context when starting the child).

#### 6. MCP Server Calls

MCP tool calls go through the tool manager, which routes them to the
appropriate MCP server via the MCP client. The `execute_tool` span
(section 4) covers the tool invocation from agent-plane's
perspective. Inside that span, the actual MCP server call gets its
own child span following the [OTel MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/).

```python
with telemetry.mcp_client_span(tool_name, server_address) as span:
    result = await mcp_client.call_tool(tool_name, arguments)
```

The `mcp_client_span` helper:

```python
def mcp_client_span(
    tool_name: str,
    server_address: str,
    session_id: str | None = None,
):
    """
    Create an MCP client span following OTel MCP semconv.

    :param tool_name: The MCP tool being called, e.g. ``"get_weather"``.
    :param server_address: The MCP server's address, e.g. ``"localhost"``.
    :param session_id: The MCP session ID, if available.
    """
    attrs: dict[str, Any] = {
        "mcp.method.name": "tools/call",
        "gen_ai.operation.name": "execute_tool",
        "server.address": server_address,
    }
    if session_id:
        attrs["mcp.session.id"] = session_id
    return _tracer.start_as_current_span(
        f"tools/call {tool_name}",
        kind=trace.SpanKind.CLIENT,
        attributes=attrs,
    )
```

This produces a two-level span for MCP tools:
```
[Tool] execute_tool {tool.name="get_weather", tool.type="mcp"}
  └─ [MCP] tools/call get_weather {mcp.method.name, server.address}
```

Non-MCP tools (local, builtin, client) have only the `execute_tool`
span with no child.

#### 7. Failure, Recovery, and Retries

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

**Cancellation.** When a task is cancelled (via `POST
/v1/responses/{id}/cancel`), the workflow is interrupted. Any open
spans — `agent_execution`, `agent_iteration`, `chat` — must be
ended cleanly. OTel has no `CANCELLED` status code (only `OK`,
`ERROR`, and `UNSET`), so we use `ERROR` with a distinguishing
attribute:

```python
span.set_status(trace.StatusCode.ERROR, "cancelled")
span.set_attribute("error.type", "cancelled")
```

The cancellation handler must walk up the span stack and end all
open spans. In practice, the workflow's `finally` block handles
this — when the workflow coroutine is cancelled, Python unwinds the
`with` statement context managers, which end the spans. The
`agent_execution` span's `finally` block sets the error status
before the span closes.

Operators can filter cancelled traces with
`error.type = "cancelled"` to distinguish them from real failures.

#### 8. Log Correlation

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

#### 9. Span Flushing

The `BatchSpanProcessor` buffers spans and flushes them periodically
(default: every 5 seconds) or when the buffer is full. Without
explicit flushing, spans can be lost in two scenarios:

**Clean server shutdown.** `telemetry.init()` registers an `atexit`
handler that calls `tracer_provider.shutdown()`. This flushes all
buffered spans and shuts down exporters cleanly. The OTel SDK also
registers its own `atexit` handler, but calling shutdown explicitly
ensures it happens before other cleanup (e.g., database connections
closing).

**Per-request failure.** When a workflow fails with an unhandled
exception, the `agent_execution` span is ended with ERROR status
(see section 6). The span is added to the batch processor's buffer.
If the failure is catastrophic (e.g., OOM, segfault), the buffer
may not flush before the process dies. For non-catastrophic failures,
the span is flushed on the next periodic export cycle — no explicit
per-request flush is needed.

To minimize span loss on crashes, configure a shorter export interval:

```bash
export OTEL_BSP_SCHEDULE_DELAY=1000  # flush every 1s instead of 5s
```

The tradeoff is more frequent network calls to the collector. For
most deployments, the default 5s interval is fine — crash-induced
span loss is an edge case, and the pre-crash span loss section
(see "Crash recovery") already documents this limitation.

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
    "opentelemetry-instrumentation-httpx>=0.44b0",          # Remote executor trace propagation
]

[project.optional-dependencies]
agents-sdk = [
    "opentelemetry-instrumentation-openai-agents-v2>=0.1",  # OpenAI Agents SDK OTel bridge
]
```

The `opentelemetry-instrumentation-openai-agents-v2` package is only
needed when using the `agents_sdk` executor. It is an optional extra
to avoid pulling in the OpenAI Agents SDK dependency chain for
deployments that don't use it. `opentelemetry-instrumentation-httpx`
is a core dependency because it's lightweight and benefits both the
remote executor and any other httpx-based outbound calls.

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

1. Implement `telemetry.init()` — reads standard OTel env vars,
   creates providers and exporters, NoOp fallback when endpoint unset.
2. Wire `telemetry.init()` into server startup (before `create_app()`).
3. Add FastAPI auto-instrumentation.

### Phase 2: Core Spans

**Files:** Modify `agent_plane/runtime/workflow.py`

1. Add `invoke_agent_span` around `agent_execution_workflow()`.
2. Add `iteration_span` inside the agent loop.
3. Add `chat_span` around `_call_llm()` and `_call_llm_streaming()`.
4. Record `gen_ai.usage.*` attributes from LLM responses.
5. Add `execute_tool_span` around `_call_tool()`.
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
[Agent] invoke_agent my-coder {gen_ai.agent.id, task.id}
  └─ [Executor] claude_sdk / agents_sdk
       ├─ [GenAI] chat {model}          ← executor's LLM call
       ├─ [Tool] execute_tool {name}    ← executor's tool call
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
# Only when agent-plane telemetry is enabled (OTEL_EXPORTER_OTLP_ENDPOINT set):
otel_env = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",  # required for traces
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    # Forward the operator's OTel config to the subprocess.
    "OTEL_EXPORTER_OTLP_PROTOCOL": os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", ""),
    "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
    **telemetry.get_traceparent_env(),                     # TRACEPARENT
}
# Forward headers (auth, MLflow experiment ID, etc.).
if headers := os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"):
    otel_env["OTEL_EXPORTER_OTLP_HEADERS"] = headers
# Forward content capture setting.
if os.environ.get("AGENT_PLANE_OTEL_CAPTURE_CONTENT", "").lower() in ("true", "1"):
    otel_env["OTEL_LOG_USER_PROMPTS"] = "1"
    otel_env["OTEL_LOG_TOOL_DETAILS"] = "1"
    otel_env["OTEL_LOG_TOOL_CONTENT"] = "1"
```

**What it emits**: LLM call spans, tool execution spans, sub-agent
spans, token usage metrics, structured events — all following the
Claude Agent SDK's built-in OTel schema. The `TRACEPARENT` env var
causes these spans to nest under agent-plane's `invoke_agent` span.

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
if os.environ.get("AGENT_PLANE_OTEL_CAPTURE_CONTENT", "").lower() in ("true", "1"):
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

## Test Plan

### Unit Tests (`tests/runtime/test_telemetry.py`)

Test the telemetry helpers in isolation using OTel's
`InMemorySpanExporter` — no external backend needed.

1. **`_trace_id_from_response_id`** — correct extraction for valid
   IDs, ValueError for malformed IDs (wrong prefix, wrong length).
2. **`invoke_agent_span`** — creates span with correct name
   (`invoke_agent {name}`), all standard attributes
   (`gen_ai.operation.name`, `gen_ai.agent.id`, `gen_ai.agent.name`,
   `gen_ai.conversation.id`, `session.id`, `task.id`), and custom
   attributes (`agent.executor.type`, `agent.background`, etc.).
3. **`chat_span`** — creates CLIENT span, sets `gen_ai.provider.name`,
   `gen_ai.request.model`, `gen_ai.conversation.id`.
4. **`record_llm_usage`** — sets token attributes; includes cache
   breakdown when present, omits when absent.
5. **`execute_tool_span`** — sets `gen_ai.operation.name =
   "execute_tool"`, `tool.name`, `tool.call_id`, `tool.type`.
6. **`mcp_client_span`** — sets `mcp.method.name = "tools/call"`,
   `server.address`; includes `mcp.session.id` when provided.
7. **`get_traceparent_env`** — returns `TRACEPARENT` when inside an
   active span, empty dict when no span.
8. **`init()` with no endpoint** — NoOp providers, zero spans exported.
9. **`init()` with endpoint** — providers created, exporter registered.
10. **Trace ID derivation** — response ID hex suffix becomes the trace
    ID on the root span.
11. **Span hierarchy** — child spans (iteration, chat, tool) are
    parented under the root invoke_agent span.

### Integration Tests (`tests/runtime/test_telemetry_integration.py`)

Use `InMemorySpanExporter` with the real workflow machinery (mock LLM,
real stores, real tool manager). These verify that spans are emitted
at the correct points in the actual execution flow.

1. **Single-turn happy path** — mock LLM returns text, no tool calls.
   Assert: one `invoke_agent` span, one `agent_iteration` span, one
   `chat` span. Verify token usage attributes, finish_reasons, model.
2. **Tool call round-trip** — mock LLM returns a tool call, then a
   final text response after receiving the tool result. Assert:
   `invoke_agent` → `agent_iteration` → `chat` → `execute_tool` →
   `chat`. Tool span has `tool.name`, `tool.status = "success"`.
3. **MCP tool call** — same as above but the tool is an MCP tool.
   Assert: `execute_tool` span has child `tools/call` span with
   `mcp.method.name` and `server.address`.
4. **Multi-iteration agent** — mock LLM returns tool calls for 3
   iterations before final text. Assert: 3 `agent_iteration` spans
   with correct `agent.iteration.number` (1, 2, 3).
5. **Sub-agent spawn** — mock LLM calls the spawn tool, sub-agent
   completes. Assert: nested `invoke_agent` span under the parent's
   `execute_tool` span, with `agent.conversation.kind = "sub_agent"`.
   Same trace ID as parent (root response ID).
6. **Workflow failure** — mock LLM raises PermanentLLMError. Assert:
   `invoke_agent` span has `status = ERROR`, `error.type =
   "PermanentLLMError"`, recorded exception.
7. **LLM retry** — mock LLM fails once (retryable), succeeds on
   second attempt. Assert: `chat` span has a `gen_ai.retry` event
   with `attempt = 1`, `error.type`.
8. **Cancellation** — start a background task, cancel it mid-execution
   (mock LLM blocks). Assert: `invoke_agent` span has `status =
   ERROR`, `error.type = "cancelled"`.
9. **Content capture disabled** — default config. Assert:
   `gen_ai.input.messages` and `gen_ai.output.messages` are NOT set
   on chat spans.
10. **Content capture enabled** — set
    `AGENT_PLANE_OTEL_CAPTURE_CONTENT=true`. Assert: chat spans have
    `gen_ai.input.messages` and `gen_ai.output.messages` attributes
    with correct JSON content.
11. **Multimodal content** — input includes an image from the file
    store. Assert: `gen_ai.input.messages` contains a `file` part
    with the file ID (not inline base64).
12. **Context window compaction** — mock LLM raises
    ContextWindowExceeded, then succeeds after compaction. Assert:
    `agent_iteration` span has `context_window_compaction` event.
13. **Telemetry disabled** — no `OTEL_EXPORTER_OTLP_ENDPOINT` set.
    Assert: zero spans exported (NoOp provider), workflow completes
    normally with no overhead.
14. **Session grouping** — two turns in the same conversation (second
    uses `previous_response_id`). Assert: both root spans have the
    same `session.id` but different trace IDs (derived from their
    respective response IDs).

### E2E Tests (`tests/e2e/test_telemetry_e2e.py`)

These use a **real MLflow tracking server** as the trace backend and
verify the full pipeline: agent-plane emits OTLP → MLflow receives
and stores → MLflow SDK can fetch and inspect the trace.

**Prerequisites:**
- `LLM_API_KEY` env var (real LLM for realistic traces)
- MLflow tracking server running (started by test fixture)
- `OTEL_EXPORTER_OTLP_ENDPOINT` pointed at the MLflow server
- `OTEL_EXPORTER_OTLP_HEADERS` with `x-mlflow-experiment-id`

**Test fixture setup:**

```python
@pytest.fixture(scope="module")
def mlflow_server():
    """Start an MLflow tracking server with SQLite backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_uri = f"sqlite:///{tmpdir}/mlflow.db"
        artifacts = f"{tmpdir}/artifacts"
        proc = subprocess.Popen([
            "mlflow", "server",
            "--backend-store-uri", db_uri,
            "--default-artifact-root", artifacts,
            "--host", "127.0.0.1",
            "--port", "5555",
        ])
        _wait_for_server("http://127.0.0.1:5555/health")
        yield "http://127.0.0.1:5555"
        proc.terminate()


@pytest.fixture
def agent_plane_server(mlflow_server):
    """Start agent-plane with OTel export to MLflow."""
    env = {
        **os.environ,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_ENDPOINT": mlflow_server,
        "OTEL_EXPORTER_OTLP_HEADERS": "x-mlflow-experiment-id=0",
        "OTEL_SERVICE_NAME": "agent-plane-test",
        "AGENT_PLANE_OTEL_CAPTURE_CONTENT": "true",
    }
    # Start agent-plane server with test agents deployed
    ...
```

**Test cases:**

1. **Simple agent → MLflow trace roundtrip.** Deploy a single-tool
   agent, send one request, wait for completion. Fetch the trace from
   MLflow using `mlflow.get_trace(trace_id)` where trace_id = hex
   suffix of the response ID. Assert:
   - Trace exists and has the expected trace ID.
   - Root span name starts with `invoke_agent`.
   - Root span has `gen_ai.agent.id`, `gen_ai.agent.name`, `task.id`.
   - At least one `chat` child span with `gen_ai.usage.input_tokens > 0`.
   - At least one `execute_tool` child span with `tool.name`.

2. **Multi-turn conversation → session grouping.** Send two requests
   in the same conversation (second uses `previous_response_id`).
   Fetch both traces. Assert:
   - Two distinct traces (different trace IDs).
   - Both have the same `session.id` = conversation ID.
   - `mlflow.search_traces(filter_string='tags."session.id" =
     "<conv_id>"')` returns exactly 2 traces.

3. **Sub-agent workflow → nested trace.** Deploy a root agent that
   spawns a sub-agent (via the `spawn` tool). Send one request, wait
   for the root to complete. Fetch the trace. Assert:
   - Single trace (root response ID).
   - Two `invoke_agent` spans: root and sub-agent (nested).
   - Sub-agent span has `agent.conversation.kind = "sub_agent"`.
   - Sub-agent's own response ID appears as `task.id` on its span.
   - Both invoke_agent spans have chat and tool children.

4. **Tool calls with MCP → span nesting.** Deploy an agent with an
   MCP tool (e.g., a test MCP server). Send a request that triggers
   the MCP tool. Fetch the trace. Assert:
   - `execute_tool` span has a child `tools/call` span.
   - `tools/call` span has `mcp.method.name = "tools/call"` and
     `server.address`.

5. **Cancellation → error status.** Send a background request, cancel
   it while the LLM is processing. Fetch the trace. Assert:
   - Root `invoke_agent` span has status ERROR.
   - `error.type` attribute is `"cancelled"`.
   - The trace is still complete (all open spans were closed).

6. **Workflow failure → error recording.** Deploy an agent with an
   invalid model (will fail on LLM call). Send a request, wait for
   failure. Fetch the trace. Assert:
   - Root span has status ERROR.
   - `error.type` is the exception class name.
   - Span has a recorded exception event.

7. **Content capture → messages in spans.** With
   `AGENT_PLANE_OTEL_CAPTURE_CONTENT=true`, send a request. Fetch
   the trace and inspect the chat span. Assert:
   - `gen_ai.input.messages` attribute exists and contains valid JSON.
   - JSON contains at least one message with `role = "user"`.
   - `gen_ai.output.messages` contains at least one message with
     `role = "assistant"` and non-empty text.
   - `gen_ai.system_instructions` is present if agent has instructions.

8. **Content capture with file-store images → file ref roundtrip.**
   Upload an image via `POST /v1/files`, then send a request that
   includes the file. Fetch the trace from MLflow and extract all
   `file` parts from `gen_ai.input.messages`. For each file
   reference, download the content via
   `GET /v1/files/{file_id}/content`. Assert:
   - `gen_ai.input.messages` contains a part with
     `"type": "file", "file_id": "file_..."`.
   - Every `file_id` in the trace resolves to a 200 response from
     the file API.
   - Downloaded bytes match the original uploaded image
     (content equality or hash match).
   - `mime_type` in the trace part matches the Content-Type header
     from the file download response.

   Then send a second request with an inline base64 image (not
   uploaded via file store). Fetch the trace and assert:
   - `gen_ai.input.messages` contains a part with
     `"type": "blob", "content": "<base64>"`.
   - The base64 decodes to valid image bytes matching the original.

9. **Token usage aggregation in MLflow.** Send a multi-iteration
   request (multiple LLM calls). Fetch the trace from MLflow. Assert:
   - Each `chat` span has `gen_ai.usage.input_tokens` and
     `gen_ai.usage.output_tokens`.
   - MLflow's trace-level token aggregation sums them correctly
     (check `trace.info.token_usage`).

10. **Complex sub-agent + tools + cancellation.** Deploy a root agent
    that spawns 2 sub-agents, each with different tools (one MCP, one
    local). Start the request, let sub-agent 1 complete, cancel while
    sub-agent 2 is running. Fetch the trace. Assert:
    - Single trace with root + 2 sub-agent invoke_agent spans.
    - Sub-agent 1's subtree is complete (all spans have status OK).
    - Sub-agent 2's subtree has status ERROR with
      `error.type = "cancelled"`.
    - MCP tool spans are present under the appropriate sub-agent.
    - Total span count is reasonable (no phantom or duplicate spans).

### Running the Tests

```bash
# Unit + integration (no external dependencies)
python -m pytest tests/runtime/test_telemetry.py -xvs
python -m pytest tests/runtime/test_telemetry_integration.py -xvs

# E2E (requires LLM API key, starts MLflow server automatically)
pytest tests/e2e/test_telemetry_e2e.py \
  --llm-api-key $LLM_API_KEY -v
```

The e2e tests are excluded from the default `pytest` run (same as the
existing e2e tests — no API key needed for CI). They must be run
manually before committing changes to telemetry code.

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
