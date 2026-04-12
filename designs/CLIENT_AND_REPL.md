# Client Library and REPL Shell

## Problem

The only way to interact with an agent-plane server today is by hand-rolling
raw HTTP requests. The TUI in `examples/frontends/terminal.py` (2,226 lines)
does this — and it's the only working reference for how to talk to the server.
But the protocol logic (SSE parsing, conversation threading, tool execution
loops, file upload, agent bundle management) is inextricable from the Textual
widget code. Anyone building a new frontend — a CLI, a web UI, a Slack bot, a
programmatic test harness — has to reverse-engineer and re-implement the
protocol from scratch.

The `agent_plane/client/` package exists but is empty. The `DRAFT.md` there
covers agent CRUD only — it doesn't address the hard parts: streaming, the tool
execution loop, conversation continuation, or steering.

### What's wrong with the current terminal

The Textual TUI conflates three concerns in one file:

1. **Protocol**: SSE frame parsing, event dispatch, request body construction,
   `previous_response_id` threading, the "stream → collect tool calls →
   execute → send results → repeat" loop, tunneled sub-agent tool calls via
   PATCH, steering delivery, file upload and content block assembly.

2. **Server lifecycle**: spawning a local subprocess, health-check polling,
   remote server auth, agent bundle tar/upload.

3. **Rendering**: Textual widget tree, DOM pruning, collapsibles, live widget
   updates, keyboard navigation, clipboard paste handling.

Because these are entangled, none can be reused. The polling client in
`examples/agents/coder/client.py` reimplements the tool loop and tunneling
from scratch (blocking/sync instead of streaming, ~200 lines of duplicated
logic).

---

## Design

Two layers, each a standalone package:

```
agent_plane/client/       ← Layer 1: typed Python SDK for the server API
examples/frontends/repl/  ← Layer 2: REPL shell built on top of the client
```

The client library lives inside `agent_plane/` because it's a first-class
part of the project — any program that talks to agent-plane should use it.
The REPL is an example frontend, same as the current TUI.

---

## Layer 1: Client Library (`agent_plane.client`)

### Surface area

The client covers every endpoint in `server/API.md`:

```python
from agent_plane.client import AgentPlaneClient

client = AgentPlaneClient(base_url="http://localhost:8080")

# ── Agents ──────────────────────────────────────────────
agent = await client.agents.create(bundle_path="./my-agent/")
agents = await client.agents.list()
agent = await client.agents.get(agent.id)
await client.agents.delete(agent.id)

# ── Files ───────────────────────────────────────────────
file = await client.files.upload("./data.csv")
files = await client.files.list()
content = await client.files.get_content(file.id)
await client.files.delete(file.id)

# ── Conversations ───────────────────────────────────────
convos = await client.conversations.list()
convo = await client.conversations.get(convo_id)
items = await client.conversations.list_items(convo_id)
await client.conversations.update(convo_id, title="Weather chat")
await client.conversations.delete(convo_id)

# ── Responses (streaming) ──────────────────────────────
async for event in client.responses.stream(
    model="archer",
    input="explain this code",
):
    match event:
        case TextDelta(delta=text):
            print(text, end="")
        case ToolCall(name=name, arguments=args):
            print(f"calling {name}...")
        case ToolResult(output=output):
            print(f"result: {output[:100]}")
        case ReasoningDelta(delta=text):
            pass  # thinking...
        case ResponseCompleted(response=resp):
            print("done")

# ── Responses (blocking) ───────────────────────────────
response = await client.responses.create(
    model="archer",
    input="hello",
)

# ── Responses (background + polling) ───────────────────
response = await client.responses.create(
    model="coder",
    input="fix the bug",
    background=True,
)
# poll:
response = await client.responses.get(response.id)

# ── Steering ────────────────────────────────────────────
await client.responses.steer(response.id, "try a different approach")

# ── Cancel ──────────────────────────────────────────────
await client.responses.cancel(response.id)

# ── Delete ──────────────────────────────────────────────
await client.responses.delete(response.id)
```

### Tool execution: client owns the loop

When the agent calls a client-side tool during streaming, the client
handles the full loop: (1) detect the tool call in the event stream,
(2) execute the tool locally via the consumer's callback, (3) send
the result back to the server, (4) resume streaming the next response,
(5) repeat until no more tool calls. The consumer sees one flat stream
of events — tool iterations are transparent.

This also covers tunneled sub-agent tool calls (PATCH-based), polling
mode tool calls, and all the `previous_response_id` threading. Without
this, every consumer would reimplement ~50 lines of subtle protocol
logic (and get the edge cases wrong).

Consumer control is provided through a **hooks system** on
`ToolHandler` — async callbacks at key points in the tool execution
lifecycle. The execute function itself is the primary hook, and it
receives a rich context object:

```python
@dataclass
class ToolCallInfo:
    """Context passed to the tool handler's execute callback."""
    name: str
    arguments: dict[str, object]
    call_id: str
    agent_name: str          # "coder" or "coder.researcher" for sub-agents
    # (removed — use agent_name to identify sub-agent calls)        # True if from a sub-agent (PATCH path)
    response_id: str         # Current response ID
    iteration: int           # Tool loop iteration count

handler = ToolHandler(schemas=[...], execute=my_execute)
```

The execute function can be async and can do anything before returning
a result — approval prompts, logging, argument modification, denial:

```python
async def my_execute(call: ToolCallInfo) -> str:
    # Approval gate:
    if call.name == "Bash":
        approved = await prompt_user(
            f"Run `{call.arguments['command']}`?"
        )
        if not approved:
            return "User denied tool execution."

    # Logging:
    log.info(f"[{call.agent_name}] {call.name}({call.arguments})")

    # Normal execution:
    return run_tool(call.name, call.arguments)
```

Lifecycle hooks (approval gates, logging, display, error handling)
are on a separate `StreamHooks` object — see the Hooks section below.

`tool_handler` is only needed when the agent declares **client-side
tools** — tools that execute on the caller's machine (Read, Write,
Bash, etc.), like the coder agent. Many agents only use server-side
tools (MCP servers, web search, Python local tools defined in the
agent spec). For those agents, no `tool_handler` is needed — the
server executes everything and the client just watches events go by:

```python
# Agent "archer" uses only server-side tools — no tool_handler.
session = client.session(model="archer")
async for event in session.send("find recent papers on RLHF"):
    match event:
        case TextDelta(delta=t):
            print(t, end="")
        case ToolCall(name=n):
            print(f"▸ {n}...")  # informational, already executed
```

Without a `tool_handler`, `stream()` yields raw events for a single
server response. If the agent calls a client-side tool and no handler
is registered, the `ToolCall` event is yielded with
`status="action_required"` and the stream ends — the consumer is
responsible for executing and continuing manually. This is the
low-level escape hatch for consumers who genuinely need to control
the outer loop.

---

### Protocol nuances the client handles

The client absorbs all the following protocol details so consumers
don't have to think about them. These are derived from the server
implementation in `routes/responses.py`, the runtime workflow in
`runtime/workflow.py`, and the existing terminal frontend.

#### SSE stream lifecycle

The server emits events in this order:

```
response.created              (sequence 0, always)
response.queued               (sequence 1, only if background=true)
response.in_progress          (sequence 1 or 2)
  ... live stream events (any order, interleaved) ...
  response.output_text.delta  (text tokens)
  response.reasoning.started  (reasoning block opens)
  response.reasoning_text.delta  (reasoning tokens)
  response.reasoning_summary_text.delta  (summary tokens)
  response.output_item.done   (item finished — message, function_call,
                                function_call_output, native tool, reasoning,
                                compaction)
  response.output_file.done   (file artifact produced)
  response.retry              (retryable failure, will retry)
  response.error              (error during execution)
  response.compaction.in_progress  (context being compacted)
  ... more live stream events ...
response.completed|failed|incomplete|cancelled  (terminal)
[DONE]                        (sentinel — stream ends)
```

Every event includes `type` (string) and `sequence_number` (monotonic
integer). The client validates the sequence is contiguous (no gaps).

The server registers the live stream subscriber BEFORE starting the
workflow — this is a structural guarantee that no early events are
lost. The client does not need to handle missed events.

Note: several OpenResponses spec events are NOT emitted by the
server: `output_item.added`, `output_text.done`, `content_part.*`,
`function_call.arguments.*`, `refusal.*`. The client should skip
unknown event types gracefully for forward-compatibility.

#### Steering detection

When the consumer calls `session.steer()` (or `client.responses.steer()`),
the client POSTs with `previous_response_id` pointing to the in-progress
response plus `background: true, stream: false`. Three outcomes:

1. **Delivered**: Server returns the existing response object (same ID,
   `status: "in_progress"`). The steering message is in the agent's
   inbox. The current SSE stream continues — the agent incorporates the
   message at its next loop iteration.

2. **Inbox closed**: The agent is finishing. Server waits for completion,
   then creates a new response. The client detects this (different
   response ID returned) and reports it to the consumer.

3. **Already terminal**: The previous response is completed/failed/
   cancelled. Server creates a new response. Same detection as (2).

The client tracks `_response_terminal` synchronously from
`response.completed`/`response.failed` events — not from async worker
state — to avoid races between steering and completion.

#### Conversation threading and forking

The client tracks `previous_response_id` automatically in the session
helper. On each `ResponseCompleted` event, the session updates its
stored ID to the completed response's ID.

Fork detection is server-side: if you POST with a
`previous_response_id` that isn't the latest in its conversation, the
server creates a new conversation (copying items up to the fork point).
The client doesn't need to detect forks — it just follows the
`conversation.id` in the response.

The client validates the `conversation` field if explicitly provided:
it must match the conversation that `previous_response_id` belongs to,
and the referenced response must be the latest. Otherwise the server
returns 400.

#### Client-side tool execution loop

When `tool_handler` is provided, the client runs this loop internally:

```
1. POST /v1/responses {model, input, tools: handler.schemas, stream: true}
2. Read SSE stream, yield events to consumer
3. Collect function_call items from output_item.done events
4. Filter out server-side calls (those with a matching
   function_call_output already in the stream)
5. For remaining client-side calls:
   a. Call handler.execute(ToolCallInfo(...)) for each
   b. Yield ToolCall + ToolResult events to consumer
6. Call hooks.on_tool_results_ready(results) if set
7. POST /v1/responses {input: [function_call_output items],
   previous_response_id: current_id, tools: handler.schemas}
8. Go to step 2
9. Exit when no pending tool calls remain after a stream
```

The completed_call_ids filtering (step 4) is critical: when the
server executes a tool server-side, both `function_call` and
`function_call_output` appear in the stream. The client must not
re-execute these.

#### Tunneled sub-agent tool calls

Sub-agents that hit client-side tools publish their tool calls to the
**root response's output** with `status: "action_required"`. These
arrive during the parent's SSE stream as `output_item.done` events.

The client handles these differently from regular client-side tools:

1. **Detection**: `function_call` item with `status == "action_required"`
2. **Execution**: Immediate — fired in background, not batched for
   post-stream. The SSE stream continues while the tool executes.
3. **Result delivery**: `PATCH /v1/responses/{root_response_id}` with
   `{"tool_results": [{"call_id": "...", "output": "..."}]}`
4. **Unblocking**: The PATCH wakes the parked sub-agent via DBOS
   messaging. The sub-agent resumes and continues its loop.

The consumer sees `ToolCall` and `ToolResult` events with
`agent_name="parent.child"` — the dotted name tells them it's from a
sub-agent. They can render it differently but don't need to handle
the PATCH themselves.

**PATCH idempotency**: Re-PATCHing the same `call_id` is safe (first
writer wins). The client can safely retry on network errors.

**PATCH error handling**:
- 404: `call_id` not found — sub-agent may have been cleaned up
- 409: Sub-agent already terminal (timed out or failed independently)
- Both are logged but not raised to the consumer — the sub-agent
  handles its own failure.

#### Polling mode (background responses)

For `background=True, stream=False`, the server returns immediately
with `status: "queued"`. The client provides a polling helper:

```python
response = await client.responses.create(
    model="coder", input="...", background=True
)
# Poll until terminal:
response = await client.responses.poll(response.id)
```

`poll()` calls `GET /v1/responses/{id}` repeatedly until the status
is terminal. Between polls, it checks for tunneled tool calls:
`function_call` items with `status: "action_required"` in the output.
If a `tool_handler` is set, it executes them and PATCHes results back.

This replicates the logic in `examples/agents/coder/client.py` but
without the consumer needing to implement it.

#### File upload and content blocks

The client builds the correct `input` format:

```python
# Plain text (no files):
input = "explain this code"  # → server receives string directly

# With file attachments:
file = await client.files.upload("./data.csv")
input = [
    {"type": "input_text", "text": "summarize this"},
    {"type": "input_file", "file_id": file.id, "filename": "data.csv"},
]

# Images use input_image:
img = await client.files.upload("./screenshot.png")
input = [
    {"type": "input_text", "text": "what's wrong here?"},
    {"type": "input_image", "file_id": img.id},
]
```

The session helper provides a convenience method:

```python
async for event in session.send(
    "summarize this",
    files=["./data.csv", "./screenshot.png"],
):
    ...
```

This uploads each file, detects images vs documents by MIME type,
and builds the content block list automatically.

#### Cancellation

`client.responses.cancel(response_id)` POSTs to
`/v1/responses/{id}/cancel`. The server sets the DBOS workflow status
to cancelled — the workflow observes this at its next checkpoint and
winds down.

Cancellation preserves the response record. Partial output (whatever
the agent produced before the checkpoint) is persisted. The cancelled
response can be used as `previous_response_id` to continue the
conversation.

If the response is already terminal, cancel is a no-op (idempotent).

#### Error handling and retries

The server handles LLM and tool retries internally (configurable per
agent spec — `max_attempts`, `backoff_base`, `backoff_max`,
`status_codes`). The client sees `response.retry` SSE events during
retries (informational — includes attempt count and delay).

The client handles its own transport errors:

- **Connection errors** (server down, network): Raised as
  `ConnectionError`. Consumer decides whether to retry.
- **HTTP 4xx** (bad request, not found, conflict): Raised as typed
  exceptions (`AgentNotFoundError`, `InvalidInputError`, etc.).
  These are not retryable.
- **HTTP 5xx** (server error): Raised as `ServerError`. The
  `on_transport_error` hook can return True to retry.
- **Read timeout** (SSE stream stalls): The client uses a long read
  timeout (600s) since tool execution can pause the stream for
  minutes. Raised as `TimeoutError` if exceeded.
- **Stream disconnect** (for non-background streams): The server
  cancels the workflow when the SSE connection drops. The client
  should not silently reconnect — the response is cancelled.
- **Stream disconnect** (for background streams): The server
  continues execution. The client can reconnect via
  `GET /v1/responses/{id}` to poll for the final result. Partial
  output is not available via GET — only the final completed output.

#### Response status lifecycle

```
queued → in_progress → completed
                     → failed        (error field populated)
                     → incomplete    (incomplete_details.reason populated)
                     → cancelled     (via cancel endpoint)
```

Terminal statuses and their meanings:
- `completed`: Agent finished successfully. `output` has full results.
- `failed`: Unrecoverable error. `error.code` + `error.message` set.
  Common codes: `"server_error"`, `"model_error"`.
- `incomplete`: Agent made progress but stopped early.
  `incomplete_details.reason` is one of: `"max_iterations"`,
  `"execution_timeout"`, `"context_overflow"`,
  `"max_output_tokens"`, `"content_filter"`.
- `cancelled`: User or system cancelled. Partial output preserved.

All terminal responses can be used as `previous_response_id` to
continue the conversation.

#### Compaction

When the conversation exceeds the token budget, the server auto-
compacts: tool result bodies are cleared, then old messages are
summarized. The client sees a `response.compaction.in_progress` SSE
event (informational). The compaction result is a `compaction` item
in the conversation — an opaque summary the server uses on the next
turn. The client doesn't need to do anything with it.

Compaction is invisible at the API level. Conversation items are
never deleted — the full history is always browsable via
`GET /v1/conversations/{id}/items`. The compacted summary lives
alongside the original items.

#### Agent name denormalization

The response object's `model` field is the agent name at task creation
time. If the agent is later renamed or deleted, existing responses
still show the original name. The client should use the `model` field
from the response, not from a separate agent lookup.

For tunneled sub-agent tool calls, the `model` field uses dotted
notation: `"parent.child"`. Agent names themselves never contain dots
(validated by the spec). This lets the client attribute tool calls to
the correct agent in a multi-agent execution.

---

### Event types

Typed dataclasses for every SSE event the server emits. The client
parses raw SSE frames into these — consumers never see `event: ` /
`data: ` strings or raw JSON.

The complete set of events, derived from an audit of the server
implementation (`routes/responses.py`, `runtime/workflow.py`,
`runtime/compaction.py`, `runtime/llm_retry.py`,
`runtime/tool_retry.py`):

#### Response lifecycle events

Emitted by the route handler at stream start and end:

```python
@dataclass
class ResponseCreated:
    """response.created — always first, sequence 0."""
    response: Response

@dataclass
class ResponseQueued:
    """response.queued — only when background=True."""
    response: Response

@dataclass
class ResponseInProgress:
    """response.in_progress — execution started."""
    response: Response

@dataclass
class ResponseCompleted:
    """response.completed — agent finished successfully."""
    response: Response

@dataclass
class ResponseFailed:
    """response.failed — unrecoverable error."""
    response: Response

@dataclass
class ResponseIncomplete:
    """response.incomplete — stopped early (max_iterations,
    execution_timeout, context_overflow, etc.)."""
    response: Response
    reason: str

@dataclass
class ResponseCancelled:
    """response.cancelled — cancelled via POST /cancel."""
    response: Response
```

#### Text streaming events

Emitted by the workflow as the LLM generates text:

```python
@dataclass
class TextDelta:
    """response.output_text.delta — incremental text token."""
    delta: str
```

Note: the server does NOT emit `response.output_text.done` or
`response.output_item.added` or `response.content_part.added/done`.
Only `response.output_text.delta` and `response.output_item.done`.

#### Reasoning events

Emitted when the model uses extended thinking (gated by
`reasoning_effort` in the agent's LLM config):

```python
@dataclass
class ReasoningStarted:
    """response.reasoning.started — reasoning block opened."""
    pass

@dataclass
class ReasoningDelta:
    """response.reasoning_text.delta — reasoning token."""
    delta: str

@dataclass
class ReasoningSummaryDelta:
    """response.reasoning_summary_text.delta — summary token.
    For models like o4-mini, the summary IS the only reasoning
    content available."""
    delta: str
```

Note: the server does NOT emit `response.reasoning.delta`,
`response.reasoning.done`, `response.reasoning_summary.delta`,
`response.reasoning_summary.done`, or the `_part.added/done`
variants. Only the three events above.

#### Output item events

Emitted when a complete output item finishes. This is the main
event for tool calls, messages, native tools, and compaction:

The raw `response.output_item.done` SSE event carries a generic dict.
The client parses it internally and yields higher-level typed events
to the consumer — consumers never see the raw dict:

```python
@dataclass
class ToolCall:
    """Parsed from output_item.done with type "function_call"."""
    name: str
    arguments: dict[str, object]  # Parsed from JSON
    call_id: str
    status: str          # "completed", "action_required", "incomplete"
    agent_name: str      # "coder" or "coder.researcher" (dotted for sub-agents)
    # (removed — use agent_name to identify sub-agent calls)    # True if status == "action_required" (from sub-agent)

@dataclass
class ToolResult:
    """Parsed from output_item.done with type "function_call_output"."""
    call_id: str
    output: str

@dataclass
class NativeToolCall:
    """Parsed from output_item.done with a native tool type.

    Native tool types (provider-executed, not client-side):
    - web_search_call
    - file_search_call
    - code_interpreter_call
    - computer_call
    - image_generation_call
    - mcp_call
    - mcp_list_tools
    """
    tool_type: str       # e.g. "web_search_call"
    data: dict           # Full item payload (action, status, results, etc.)

@dataclass
class MessageDone:
    """Parsed from output_item.done with type "message".
    The final assistant message for a turn."""
    content: list[dict]  # Content blocks (output_text, etc.)
```

#### File output events

Emitted when the agent produces file artifacts (e.g., generated
images, code output files) referenced by annotations:

```python
@dataclass
class OutputFileDone:
    """response.output_file.done — file artifact produced."""
    file_id: str
    filename: str | None
    content_type: str | None
```

#### Error and retry events

Emitted during execution when the server retries LLM calls or
tool calls, or when an error occurs:

```python
@dataclass
class RetryEvent:
    """response.retry — a retryable failure, will retry.

    Source is "llm" or "tool". For tool retries, tool_name
    identifies which tool failed.
    """
    source: str          # "llm" or "tool"
    tool_name: str | None  # Only for source="tool"
    attempt: int         # Current attempt (2 = first retry)
    max_attempts: int
    delay_seconds: float # Backoff delay before retry
    error: ErrorInfo

@dataclass
class ErrorEvent:
    """response.error — an error during execution.

    May arrive before the terminal event. Source is "llm" or
    "tool". For tool errors, tool_name identifies which tool.
    """
    source: str          # "llm" or "tool"
    tool_name: str | None
    error: ErrorInfo
```

#### Compaction events

```python
@dataclass
class CompactionInProgress:
    """response.compaction.in_progress — server is compacting
    conversation history. Informational — may take time due to
    the summarization LLM call."""
    pass
```

#### Summary of server event types vs OpenResponses spec

Events the server emits (client must handle):

| SSE event type | Client dataclass |
|---|---|
| `response.created` | `ResponseCreated` |
| `response.queued` | `ResponseQueued` |
| `response.in_progress` | `ResponseInProgress` |
| `response.output_text.delta` | `TextDelta` |
| `response.reasoning.started` | `ReasoningStarted` |
| `response.reasoning_text.delta` | `ReasoningDelta` |
| `response.reasoning_summary_text.delta` | `ReasoningSummaryDelta` |
| `response.output_item.done` | Parsed internally → `ToolCall`, `ToolResult`, `NativeToolCall`, `MessageDone` |
| `response.output_file.done` | `OutputFileDone` |
| `response.retry` | `RetryEvent` |
| `response.error` | `ErrorEvent` |
| `response.compaction.in_progress` | `CompactionInProgress` |
| `response.completed` | `ResponseCompleted` |
| `response.failed` | `ResponseFailed` |
| `response.incomplete` | `ResponseIncomplete` |
| `response.cancelled` | `ResponseCancelled` |

Events in the OpenResponses spec that the server does NOT emit
(client should ignore if encountered for forward-compatibility):

| Spec event type | Status |
|---|---|
| `response.output_item.added` | Not emitted |
| `response.output_text.done` | Not emitted |
| `response.content_part.added` / `.done` | Not emitted |
| `response.output_text_annotation.added` | Not emitted |
| `response.refusal.delta` / `.done` | Not emitted |
| `response.function_call.arguments.delta` / `.done` | Not emitted |
| `response.reasoning.delta` / `.done` | Not emitted (only `.started` and `_text.delta`) |
| `response.reasoning_summary.delta` / `.done` | Not emitted (only `_text.delta`) |
| `response.reasoning_summary_part.added` / `.done` | Not emitted |
| `error` (generic) | Not emitted (always `response.error` with source) |

The client should log and skip unknown event types rather than
crashing — this allows the server to add new event types without
breaking existing clients.

### Response and related types

```python
@dataclass
class Response:
    id: str
    status: str
    model: str
    output: list[OutputItem]
    created_at: int
    completed_at: int | None
    previous_response_id: str | None
    conversation: ConversationRef | None
    usage: Usage | None
    error: ErrorInfo | None
    incomplete_details: IncompleteDetails | None
    background: bool
    instructions: str | None

@dataclass
class ConversationRef:
    id: str

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class ErrorInfo:
    code: str
    message: str

@dataclass
class IncompleteDetails:
    reason: str

@dataclass
class Agent:
    id: str
    name: str
    description: str | None
    created_at: int

@dataclass
class File:
    id: str
    filename: str
    bytes: int
    created_at: int

@dataclass
class Conversation:
    id: str
    title: str | None
    created_at: int
```

### Hooks

Every class of event in the stream has a corresponding hook — a
callback the consumer can register to inject behavior at that point
in the lifecycle. Hooks are optional. Unset hooks are no-ops.

The hooks live on a `StreamHooks` object passed to `stream()`:

```python
async for event in client.responses.stream(
    model="coder",
    input="find all Python files",
    tool_handler=tool_handler,
    hooks=StreamHooks(
        on_tool_call_start=my_approval_gate,
        on_transport_error=my_error_handler,
        ...
    ),
):
    ...
```

`tool_handler` and `hooks` are separate concerns: `tool_handler`
provides tool schemas and the execute function (what to run).
`hooks` provides lifecycle callbacks (when to intervene). Both are
optional and independent.

```python
@dataclass
class ToolHandler:
    """Client-side tool execution configuration.

    Passed to stream() to enable automatic tool execution.
    The client runs the full loop: stream → detect tool calls
    → call execute() → send results → continue streaming.
    """
    schemas: list[dict]

    execute: Callable[[ToolCallInfo], Awaitable[str] | str]
    """Called for each client-side tool call. Can be sync or
    async. Receives full context via ToolCallInfo. Return the
    tool's output string.

    To deny a tool call, return an error string — the agent
    sees it as the tool's output and adapts.
    """

@dataclass
class ToolCallInfo:
    """Context passed to ToolHandler.execute."""
    name: str
    arguments: dict[str, object]
    call_id: str
    agent_name: str      # "coder" or "coder.researcher"
    # (removed — use agent_name to identify sub-agent calls)    # True if from a sub-agent
    response_id: str     # Current response ID
    iteration: int       # Tool loop iteration (0-based)
```

#### Hook definitions

All hooks can be sync or async. All receive a context dataclass
with relevant fields for that event. All are optional (default: None,
no-op).

```python
@dataclass
class StreamHooks:
    """Lifecycle hooks for stream events.

    Every class of event has a start/end hook pair where
    applicable. Hooks are called synchronously in the event
    processing path — the stream pauses until the hook returns.
    """

    # ── Tool calls (server-side and client-side) ─────────

    on_tool_call_start: Callable[[ToolCallStartCtx], Awaitable[None] | None] | None = None
    """Called when a function_call item is detected in the
    stream, BEFORE execution. Fires for both server-side tools
    (informational — already executed) and client-side tools
    (about to be executed via tool_handler).

    Use for: approval gates, logging, display. For client-side
    tools, raise ToolCallDenied to skip execution and return
    an error string to the agent."""

    on_tool_call_end: Callable[[ToolCallEndCtx], Awaitable[None] | None] | None = None
    """Called after a tool call completes (result available).
    Fires for both server-side and client-side tools.

    Use for: logging results, display, auditing."""

    # ── Native tool calls (provider-executed) ────────────

    on_native_tool_call: Callable[[NativeToolCallCtx], Awaitable[None] | None] | None = None
    """Called when a provider-native tool output appears
    (web_search_call, mcp_call, code_interpreter_call, etc.).
    These are executed server-side by the LLM provider — the
    client cannot intervene.

    Use for: display, logging."""

    # ── Tool loop iteration ──────────────────────────────

    on_tool_results_ready: Callable[[ToolResultsReadyCtx], Awaitable[None] | None] | None = None
    """Called after all tool calls in a batch are executed,
    BEFORE POSTing results back to the server. Receives the
    list of results. Can modify outputs in place (e.g.,
    truncation).

    Use for: result filtering, truncation, logging.
    Raise to abort the tool loop."""

    # ── Reasoning ────────────────────────────────────────

    on_reasoning_start: Callable[[ReasoningStartCtx], Awaitable[None] | None] | None = None
    """Called when reasoning.started arrives — the model has
    entered extended thinking."""

    on_reasoning_end: Callable[[ReasoningEndCtx], Awaitable[None] | None] | None = None
    """Called when the reasoning block ends (next non-reasoning
    event arrives, or message finalizes). Receives the full
    accumulated reasoning and summary text."""

    # ── Compaction ───────────────────────────────────────

    on_compaction_start: Callable[[CompactionStartCtx], Awaitable[None] | None] | None = None
    """Called when compaction.in_progress arrives — the server
    is summarizing conversation history.

    Use for: display ("compacting..."), logging."""

    on_compaction_end: Callable[[CompactionEndCtx], Awaitable[None] | None] | None = None
    """Called when the compaction item appears in
    output_item.done (type "compaction"). The next LLM turn
    will use the compacted context.

    Use for: display, logging."""

    # ── Message (assistant response) ─────────────────────

    on_message_start: Callable[[MessageStartCtx], Awaitable[None] | None] | None = None
    """Called when the first text delta arrives for a new
    assistant message."""

    on_message_end: Callable[[MessageEndCtx], Awaitable[None] | None] | None = None
    """Called when a message output_item.done arrives. Receives
    the full message content."""

    # ── File output ──────────────────────────────────────

    on_file_output: Callable[[FileOutputCtx], Awaitable[None] | None] | None = None
    """Called when output_file.done arrives — the agent produced
    a file artifact. Receives file_id, filename, content_type.

    Use for: auto-download, display."""

    # ── Retry and error ──────────────────────────────────

    on_retry: Callable[[RetryCtx], Awaitable[None] | None] | None = None
    """Called when a response.retry event arrives — a retryable
    failure occurred and the server will retry.

    Use for: display ("retrying in 2s..."), logging."""

    on_server_error: Callable[[ServerErrorCtx], Awaitable[None] | None] | None = None
    """Called when a response.error event arrives from the server
    (LLM or tool error). Informational — the server handles
    retries internally. The client cannot intervene.

    Use for: display, logging."""

    on_transport_error: Callable[[TransportErrorCtx], Awaitable[bool] | bool] | None = None
    """Called on transport errors (connection, timeout, HTTP 5xx)
    during streaming. Return True to retry the current stream,
    False to propagate the exception to the consumer."""

    # ── Sub-agent lifecycle ──────────────────────────────

    on_sub_agent_spawned: Callable[[SubAgentSpawnedCtx], Awaitable[None] | None] | None = None
    """Called when spawn_sub_agents tool result is detected.
    Receives the spawned response IDs and agent names.

    Use for: display, starting background stream subscriptions."""

    on_sub_agent_completed: Callable[[SubAgentCompletedCtx], Awaitable[None] | None] | None = None
    """Called when a sub-agent reaches a terminal status
    (detected from collect_sub_agents result or polling).

    Use for: display, cleanup."""

    # ── Response lifecycle ───────────────────────────────

    on_response_start: Callable[[ResponseStartCtx], Awaitable[None] | None] | None = None
    """Called when response.created arrives. Receives the
    response object with ID, model, conversation, etc."""

    on_response_end: Callable[[ResponseEndCtx], Awaitable[None] | None] | None = None
    """Called when any terminal event arrives (completed,
    failed, incomplete, cancelled). Receives the final
    response object.

    Use for: cleanup, final logging, usage reporting."""
```

#### Hook context types

Each hook receives a focused context dataclass — not the raw SSE
payload, but parsed and enriched:

```python
@dataclass
class ToolCallStartCtx:
    name: str
    arguments: dict[str, object]
    call_id: str
    agent_name: str       # "coder" or "coder.researcher"
    executed_by: str      # "client" or "server"

@dataclass
class ToolCallEndCtx:
    name: str
    call_id: str
    agent_name: str       # "coder" or "coder.researcher"
    output: str           # Tool result

@dataclass
class NativeToolCallCtx:
    tool_type: str        # "web_search_call", "mcp_call", etc.
    data: dict            # Full item payload

@dataclass
class ToolResultsReadyCtx:
    results: list[ToolResultInfo]  # Mutable — can modify in place
    iteration: int

@dataclass
class ToolResultInfo:
    call_id: str
    name: str
    output: str
    agent_name: str       # "coder" or "coder.researcher"

@dataclass
class ReasoningStartCtx:
    pass

@dataclass
class ReasoningEndCtx:
    reasoning_text: str   # Full accumulated reasoning
    summary_text: str     # Full accumulated summary

@dataclass
class CompactionStartCtx:
    pass

@dataclass
class CompactionEndCtx:
    item: dict            # The compaction output item

@dataclass
class MessageStartCtx:
    response_id: str

@dataclass
class MessageEndCtx:
    content: list[dict]   # Content blocks (output_text, etc.)

@dataclass
class FileOutputCtx:
    file_id: str
    filename: str | None
    content_type: str | None

@dataclass
class RetryCtx:
    source: str           # "llm" or "tool"
    tool_name: str | None
    attempt: int
    max_attempts: int
    delay_seconds: float
    error: ErrorInfo

@dataclass
class ServerErrorCtx:
    source: str           # "llm" or "tool"
    tool_name: str | None
    error: ErrorInfo

@dataclass
class TransportErrorCtx:
    error: Exception      # httpx.ConnectError, TimeoutError, etc.

@dataclass
class SubAgentSpawnedCtx:
    parent_response_id: str
    sub_agents: list[SubAgentInfo]  # IDs + names of spawned agents

@dataclass
class SubAgentInfo:
    response_id: str
    agent_name: str       # "researcher", "critic", etc.

@dataclass
class SubAgentCompletedCtx:
    response_id: str
    agent_name: str
    status: str           # "completed", "failed", "incomplete", "cancelled"
    output_summary: str | None  # First 500 chars if completed

@dataclass
class ResponseStartCtx:
    response: Response

@dataclass
class ResponseEndCtx:
    response: Response
    status: str           # "completed", "failed", "incomplete", "cancelled"
```

#### Example: REPL hooks

The REPL wires hooks to Rich rendering:

```python
hooks = StreamHooks(
    on_tool_call_start=lambda ctx: console.print(
        f"[green]▸ {ctx.name}({truncate(ctx.arguments)})[/]"
    ),
    on_tool_call_end=lambda ctx: render_collapsible(
        f"result: {ctx.output[:80]}", ctx.output
    ),
    on_native_tool_call=lambda ctx: console.print(
        f"[cyan]▸ {format_native_tool(ctx)}[/]"
    ),
    on_reasoning_start=lambda ctx: console.print(
        "[dim cyan]thinking...[/]"
    ),
    on_compaction_start=lambda ctx: console.print(
        "[dim magenta]compacting conversation...[/]"
    ),
    on_retry=lambda ctx: console.print(
        f"[dim yellow]retrying {ctx.source} "
        f"(attempt {ctx.attempt}/{ctx.max_attempts})...[/]"
    ),
    on_response_end=lambda ctx: console.print(
        f"[dim]usage: {ctx.response.usage}[/]"
        if ctx.status == "completed" else
        f"[red]{ctx.status}[/]"
    ),
)
```

#### ToolCallDenied exception

To deny a client-side tool call from `on_tool_call_start`:

```python
from agent_plane.client import ToolCallDenied

async def my_approval_gate(ctx: ToolCallStartCtx) -> None:
    if ctx.executed_by != "client":
        return  # server-side — can't intervene
    if ctx.name == "Bash":
        approved = await prompt_user(
            f"Run `{ctx.arguments['command']}`?"
        )
        if not approved:
            raise ToolCallDenied("User denied execution.")
```

When `ToolCallDenied` is raised, the client sends the exception's
message as the tool output (so the agent knows it was denied and
can adapt). The tool loop continues normally.

### Conversation session helper

For interactive use cases (REPL, chat UI), the client provides a
session helper that tracks `previous_response_id` automatically:

```python
session = client.session(model="archer")

# First message — no previous_response_id needed.
async for event in session.send("what's the weather?"):
    ...

# Follow-up — session tracks the chain automatically.
async for event in session.send("and tomorrow?"):
    ...

# Send while agent is running — auto-steers.
await session.send("focus on SF")

# With file attachments:
async for event in session.send(
    "summarize this",
    files=["./data.csv"],
):
    ...

# Cancel current response.
await session.cancel()
```

This is a thin wrapper — it holds the model name, last response ID,
and optional tool_handler. All behavior delegates to the underlying
client methods.

The session also tracks whether the current response is terminal
(from `ResponseCompleted`/`ResponseFailed`/etc. events) to correctly
route input. If the user calls `session.send()` while a response is
in progress, it automatically steers. If the response is already
terminal, it starts a new turn. The developer doesn't need to decide
which operation to use — `send()` always does the right thing.

### Agent bundle helper

```python
# From a directory (tars it automatically):
agent = await client.agents.create(bundle_path="./my-agent/")

# From an existing tarball:
agent = await client.agents.create(bundle_path="./my-agent.tar.gz")

# With replace semantics (delete + re-create if name exists):
agent = await client.agents.create(bundle_path="./my-agent/", replace=True)
```

### Auth

```python
client = AgentPlaneClient(
    base_url="https://my-app.databricks.com",
    headers={"Authorization": "Bearer ..."},
)
```

Auth headers are sent on every request — SSE streams, POSTs, PATCHes,
GETs. For Databricks Apps, we may provide a separate OAuth helper
(extracted from terminal.py's `auth` module).

### Server lifecycle (optional)

For local development, the client can start and manage a server:

```python
from agent_plane.client import LocalServer

async with LocalServer(agent_path="./my-agent/") as server:
    client = server.client  # pre-configured AgentPlaneClient
    async for event in client.responses.stream(...):
        ...
# Server stopped and cleaned up on exit.
```

This extracts `_start_server` / `wait_for_server` from terminal.py.
Uses a temporary SQLite database and artifact directory. Sends SIGINT
on exit, falls back to SIGKILL after 10 seconds.

### Timeouts

The client uses these defaults (matching the existing terminal):

| Operation | Timeout | Reason |
|-----------|---------|--------|
| SSE connect | 30s | Initial handshake |
| SSE read | 600s | Tool execution can pause the stream for minutes |
| Steering POST | 120s | Server may be executing a long tool (e.g. npm install) |
| Cancel POST | 10s | Simple DB update |
| PATCH (tool results) | 60s | Wake parked sub-agent |
| File upload | 30s | Network transfer |
| Other HTTP calls | 30s | Standard operations |

All timeouts are configurable via `AgentPlaneClient(timeouts=...)`.

### Implementation details

- **HTTP client**: `httpx.AsyncClient` internally. The client manages
  its own connection pool and lifecycle. Supports `async with` for
  explicit cleanup.
- **SSE parsing**: One internal parser that converts raw byte chunks
  into typed event objects. Handles the `event: ` / `data: ` /
  `[DONE]` framing. Buffers partial chunks across read boundaries.
- **Sync wrapper**: The primary API is async. A thin sync wrapper
  (`SyncAgentPlaneClient`) is provided for scripts that don't want
  to manage an event loop.
- **Error handling**: HTTP errors become typed exceptions
  (`AgentNotFoundError`, `ResponseNotFoundError`, `BundleInvalidError`,
  `InvalidInputError`, `ConflictError`, `ServerError`) with the
  server's error message and code attached.
- **No dependency on server/ or runtime/**: The client is a standalone
  package. It depends only on `httpx` and its own type definitions.
  It does NOT import from `agent_plane.server`, `agent_plane.runtime`,
  `agent_plane.stores`, or `agent_plane.spec`.

### File layout

```
agent_plane/client/
    __init__.py          # Public API: AgentPlaneClient, event types, etc.
    _client.py           # AgentPlaneClient class
    _agents.py           # agents namespace (create, list, get, delete)
    _files.py            # files namespace (upload, list, get_content, delete)
    _conversations.py    # conversations namespace
    _responses.py        # responses namespace (create, stream, get, cancel, etc.)
    _events.py           # Event dataclasses (TextDelta, ToolCall, etc.)
    _types.py            # Response, Agent, File, Conversation dataclasses
    _sse.py              # SSE frame parser
    _session.py          # Conversation session helper
    _server.py           # LocalServer context manager
    _errors.py           # Typed exception classes
    _tool_handler.py     # ToolHandler, ToolCallInfo, hooks
```

---

## Layer 2: REPL Shell (`examples/frontends/repl/`)

A print-based interactive shell built on Rich (terminal formatting) and
prompt_toolkit (line editing). No widget tree, no event loop complexity,
no DOM management. Just: read input → send via client → print formatted
output → repeat.

### Rich

Rich is a Python library for writing formatted terminal output. It provides:

- **Syntax-highlighted code blocks** (any language Pygments supports)
- **Markdown rendering** in the terminal
- **Tables, panels, trees** with box-drawing
- **Styled text** via markup (`[bold red]error[/]`)
- **Live displays** (spinners, status lines)

It's print-based — `console.print(...)` writes to stdout. No widget
lifecycle, no async rendering, no DOM. That's why it fits a REPL: each
event from the stream maps to a `console.print()` call.

### prompt_toolkit

prompt_toolkit handles the input side — readline-like line editing,
persistent history, multi-line input, tab completion. Together with
Rich it gives a polished REPL without TUI complexity.

### UX

```
$ ap repl ./my-agent/
Starting server... ready.
Agent "archer" deployed.

archer> explain the authentication flow in this codebase

  thinking...

  ▸ Read(file_path="/home/user/project/auth/middleware.py")
  ▸ Read(file_path="/home/user/project/auth/tokens.py")

  ┌─ result: middleware.py ─────────────────────────────┐
  │ 1  from .tokens import verify_jwt                   │
  │ 2  ...                                              │
  │ (collapsed — press Enter to expand)                 │
  └─────────────────────────────────────────────────────┘

  The authentication flow works as follows:

  1. Requests hit `AuthMiddleware` in `auth/middleware.py`
  2. The middleware extracts the JWT from the `Authorization`
     header and calls `verify_jwt()` from `auth/tokens.py`

  ```python
  # auth/middleware.py:15-22
  async def __call__(self, request, call_next):
      token = request.headers.get("Authorization", "").removeprefix("Bearer ")
      payload = verify_jwt(token)
      request.state.user = payload["sub"]
      return await call_next(request)
  ```

archer> /attach ./screenshot.png
  📎 attached: screenshot.png

archer> what's wrong in this screenshot?
  The button is misaligned because...

archer>
```

Key rendering behaviors:

- **Code blocks**: Detected in markdown output, rendered with Rich
  `Syntax` (syntax highlighting, line numbers, panel border).
- **Markdown**: Full markdown rendering via Rich `Markdown` — headers,
  lists, bold/italic, links, inline code.
- **Tool calls**: Displayed inline as `▸ ToolName(args...)` in dim
  green. Tool results are shown as collapsed panels — first line
  visible, full content on demand.
- **Reasoning**: Streamed as dim text below a "thinking..." label.
  Collapsed after the response completes.
- **Streaming text**: Printed token-by-token to stdout. The cursor
  stays at the end of the current line as tokens arrive. Rich's
  `Live` display handles the redraw.
- **Errors**: Red panels with the error message.
- **Sub-agent tool calls**: Attributed by `agent_name` with
  `[researcher]` prefix in the display.

### REPL commands

Special commands prefixed with `/`:

| Command | Action |
|---------|--------|
| `/attach <path>` | Upload and attach a file to the next message |
| `/new` | Start a new conversation (clear session) |
| `/conversations` | List past conversations |
| `/resume <id>` | Resume a past conversation |
| `/cancel` | Cancel the current in-progress response |
| `/model <name>` | Switch to a different agent |
| `/agents` | List available agents |
| `/history` | Show conversation history |
| `/quit` or Ctrl+D | Exit |

### Steering

While the agent is streaming, the user can type and press Enter to
send a steering message. The REPL detects that a response is in
progress (via the session's terminal flag) and routes the input to
`session.steer()` instead of starting a new turn. A `(steering)`
label is shown next to the message.

### Multi-line input

Shift+Enter or `\` at end-of-line enters multi-line mode
(prompt_toolkit handles this). The prompt changes to `...` for
continuation lines. Enter on an empty line submits.

### Architecture

The REPL is thin because the client does the hard work:

```
┌──────────────────────────────────────────────────────┐
│  repl.py (~300-500 lines)                            │
│                                                      │
│  prompt_toolkit  →  read user input                  │
│       ↓                                              │
│  client.session.send(input)  →  stream events        │
│       ↓                                              │
│  for event in stream:                                │
│      render(event)  →  Rich console.print()          │
│                                                      │
│  No SSE parsing. No HTTP calls. No tool loops.       │
│  No conversation threading. No widget lifecycle.     │
└──────────────────────────────────────────────────────┘
```

The client's `session` helper tracks `previous_response_id`. The
client's `tool_handler` runs the tool execution loop. The client's
`LocalServer` manages the server subprocess. The REPL just renders
events and reads input.

### File layout

```
examples/frontends/repl/
    __init__.py
    repl.py              # Main REPL loop
    renderer.py          # Rich-based event rendering
    commands.py          # /slash command handlers
```

Entry point: `python -m examples.frontends.repl ./my-agent/`
(or eventually `ap repl ./my-agent/` via the CLI).

---

## Sub-agent observability

### The gap today

When a parent agent spawns sub-agents, the client has very limited
visibility into what they're doing:

- The `spawn_sub_agents` tool returns `{response_ids: [...]}` to the
  parent LLM, but this is buried inside a `function_call_output` item
  in the parent's stream — the client would have to parse tool result
  JSON to extract the IDs.
- `GET /v1/responses/{sub_agent_id}` works if you have the ID, but
  returns empty output until the sub-agent completes.
- The task store has `list_tasks(root_task_id=...)` but no HTTP
  endpoint exposes it. A client cannot discover sub-agents.
- Sub-agents have their own SSE streams, but there's no way for a
  client to subscribe to them mid-flight (SSE is only on POST, not
  GET).
- `check_sub_agents` is an LLM-facing tool, not an API endpoint —
  it returns truncated activity (5 items, 2000 chars) designed for
  context budgets, not for display.

The result: a client watching the parent's stream sees tunneled tool
calls from sub-agents (`action_required` items) but has no idea what
the sub-agent is thinking or doing between those calls.

### Server API: TBD

The server needs new API surface to enable sub-agent observability.
Two capabilities are needed:

1. **Discovery**: a way for the client to find out which sub-agents
   exist for a given response and their current status.
2. **Live streaming**: a way to subscribe to a sub-agent's event
   stream after it's already running.

The exact API design (endpoints, shapes, streaming mechanism) is TBD.
Building blocks that exist internally today:
- `task_store.list_tasks(root_task_id=...)` — returns all sub-agent
  tasks for a parent, but not exposed via HTTP.
- `live_stream.subscribe(task_id)` — subscribes to a task's live
  events, but only used by the POST streaming path today.
- Sub-agents have their own task IDs (returned by `spawn_sub_agents`
  to the parent LLM as `{response_ids: [...]}`).

### Client library and hooks

Regardless of the API shape, the client library will need:

- A way to discover and list sub-agents
- A way to subscribe to a sub-agent's live stream
- Hooks for sub-agent lifecycle events

The hooks are defined on `StreamHooks` (see Hooks section):
`on_sub_agent_spawned` and `on_sub_agent_completed`. These fire by
parsing the `spawn_sub_agents` / `collect_sub_agents` tool results
from the parent's stream — they work today without any new server
API. The streaming subscription requires the server work above.

### REPL rendering

In the REPL, sub-agent activity would be shown inline under the
parent's output:

```
archer> research RLHF and critique the findings

  ▸ spawn_sub_agents(researcher, critic)

  ┌ researcher (in progress) ─────────────────────────┐
  │ Searching for RLHF papers...                      │
  │ ▸ web_search("RLHF reinforcement learning")       │
  │ Found 3 relevant papers. Reading...               │
  └───────────────────────────────────────────────────┘

  ┌ critic (in progress) ─────────────────────────────┐
  │ Waiting for research results...                   │
  └───────────────────────────────────────────────────┘

  ▸ collect_sub_agents(researcher, critic)

  The research shows...
```

### Implementation order

Sub-agent observability is not required for the initial client library
and REPL. It can be added after the core streaming and tool loop work.
The hooks (`on_sub_agent_spawned`, `on_sub_agent_completed`) can ship
first since they only require parsing parent-stream tool results. Live
sub-agent streaming requires the server API work (TBD).

---

## What this replaces

The REPL does not replace the Textual TUI immediately. Both can coexist:

- `examples/frontends/terminal.py` — existing TUI (unchanged for now)
- `examples/frontends/repl/` — new REPL built on the client lib

Over time the TUI should be refactored to use the client library too,
eliminating its raw HTTP/SSE code. But that's a separate effort.

The polling client in `examples/agents/coder/client.py` should be
rewritten to use `agent_plane.client` once the library exists — it
currently duplicates the tool loop and tunneling logic.

---

## Implementation order

1. **Event types and response types** (`_events.py`, `_types.py`) —
   the data model that everything else depends on.

2. **SSE parser** (`_sse.py`) — extracted from terminal.py's
   `_run_sse_stream`, but producing typed events instead of calling
   widget methods.

3. **Responses namespace** (`_responses.py`) — `create()`, `stream()`,
   `get()`, `cancel()`, `steer()`, `delete()`. This is the core.

4. **Tool handler integration** (`_tool_handler.py`, wired into
   `stream()`) — the automatic tool execution loop with hooks.

5. **Agents, Files, Conversations namespaces** — straightforward
   CRUD wrappers.

6. **Session helper** (`_session.py`) — thin wrapper for REPL use.

7. **LocalServer** (`_server.py`) — extracted from terminal.py.

8. **Client entry point** (`_client.py`, `__init__.py`) — ties the
   namespaces together.

9. **REPL renderer** (`renderer.py`) — Rich-based event rendering.

10. **REPL core** (`repl.py`) — input loop, slash commands, session
    management.

Steps 1-8 are the client library. Steps 9-10 are the REPL. The
client should be usable and tested before the REPL is started.

---

## Dependencies

**Client library** (added to agent-plane's dependencies):
- `httpx` (already a dependency)

**REPL** (example frontend, not a core dependency):
- `rich`
- `prompt_toolkit`

---

## Not yet (future)

- Sync wrapper for the client (for scripts that don't use async)
- WebSocket transport (alternative to SSE for lower-latency streaming)
- Token usage display in the REPL
- REPL configuration file (default model, color theme, etc.)
- Tab completion for file paths in `/attach`
- Databricks OAuth helper (extracted from terminal.py's `auth` module)
- Stream resumption on reconnect (would require server-side support
  for sequence_number-based catch-up — not available today)
