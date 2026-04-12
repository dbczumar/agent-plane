# Agent Plane Server API

Three namespaces: agent management (`/api/agents`), inference (`/v1/responses`),
and files (`/v1/files`).

## Compatibility Reference

| Namespace | Compatible with | Reference implementation |
|---|---|---|
| Agent Management (`/api/agents`) | Agent Plane (ours) | No external reference — this is our own API. |
| Conversations (`/v1/conversations`) | Agent Plane (ours) | No external reference. OpenResponses defines a minimal `conversation` object on responses but no conversation management endpoints. |
| Inference (`/v1/responses`) | [OpenResponses spec](https://openresponses.org) | OpenAI Responses API is the reference implementation. Test with: `curl https://api.openai.com/v1/responses -H "Authorization: Bearer $KEY" ...` |
| Cancel Response (`POST /v1/responses/{id}/cancel`) | OpenAI (undocumented) | Not in OpenResponses spec. Discovered empirically on OpenAI. Test with: `curl -X POST https://api.openai.com/v1/responses/{id}/cancel -H "Authorization: Bearer $KEY"` |
| Files (`/v1/files`) | OpenAI Files API | Test with: `curl https://api.openai.com/v1/files -H "Authorization: Bearer $KEY"` |

**How to verify consistency**: When implementing or modifying these APIs, test the
corresponding endpoint on the reference implementation using an OpenAI API key
(stored at `/tmp/mykey`). Compare request/response shapes, field names, status codes,
and edge case behavior. Our APIs should accept a superset of the reference input
(additional fields like `conversation`) and return a superset of the reference output
(additional fields like `conversation`, `response_id` on items).

**Our extensions beyond OpenResponses / OpenAI**:
- `conversation` field on request and response objects
- `response_id` and `model` fields on conversation items
- Conversation management endpoints (list, get, items, delete)
- Agent management endpoints (CRUD for agent bundles)
- Steering via `previous_response_id` pointing to an in-progress response
- Fork detection and automatic conversation creation

---

## Agent Management

### Create Agent

```
POST /api/agents
Content-Type: multipart/form-data

Parts:
  bundle: <tarball>       required — must contain config.yaml with a unique
                          name and optional description. The name becomes
                          the "model" for inference requests.

The server validates the bundle on upload: extracts it to a temporary
directory, parses config.yaml, and runs the spec validator. Name and
description are derived from the spec — no separate form fields.

201 Created
{
  "id": "ag_abc123",
  "object": "agent",
  "name": "my-agent",
  "description": "...",
  "created_at": 1774118382
}

409 Conflict — name already exists
400 Bad Request — invalid bundle (corrupt tarball, missing config.yaml,
    spec validation failure, missing name, path traversal, etc.)
```

### List Agents

```
GET /api/agents

Query parameters:
  limit (integer, optional, default: 20, max: 100)
    Number of agents to return.

  after (string, optional)
    Cursor for forward pagination. Pass the `last_id` from a previous response
    to get the next page.

  before (string, optional)
    Cursor for backward pagination. Pass the `first_id` from a previous response
    to get the previous page.

  order (string, optional, default: "desc")
    Sort order by `created_at`. Either "asc" or "desc".

200 OK
{
  "object": "list",
  "data": [
    {"id": "ag_abc123", "object": "agent", "name": "my-agent", ...},
    {"id": "ag_def456", "object": "agent", "name": "other-agent", ...}
  ],
  "first_id": "ag_abc123",
  "last_id": "ag_def456",
  "has_more": false
}
```

Items in `data` have the same shape as the create/get response.

### Get Agent

```
GET /api/agents/{id}

200 OK — same shape as create response
404 Not Found
```

### Delete Agent

```
DELETE /api/agents/{id}

200 OK
{"id": "ag_abc123", "object": "agent.deleted", "deleted": true}

404 Not Found
```

Cancels all in-flight responses for this agent before deleting.

---

## Files

Upload files that can be referenced by `file_id` in `input_image` and `input_file`
content types. Files are immutable once uploaded.

### Upload File

```
POST /v1/files
Content-Type: multipart/form-data

Parts:
  file: <binary>        required

201 Created
{
  "id": "file_abc123",
  "object": "file",
  "filename": "report.pdf",
  "bytes": 214961,
  "created_at": 1774118382
}

400 Bad Request — missing file
```

### List Files

```
GET /v1/files

Query parameters:
  limit (integer, optional, default: 20, max: 100)
  after (string, optional)
  before (string, optional)
  order (string, optional, default: "desc")
    Sort order by `created_at`. Either "asc" or "desc".

200 OK
{
  "object": "list",
  "data": [
    {"id": "file_abc123", "object": "file", "filename": "report.pdf", ...},
    {"id": "file_def456", "object": "file", "filename": "photo.jpg", ...}
  ],
  "first_id": "file_abc123",
  "last_id": "file_def456",
  "has_more": false
}
```

### Get File

```
GET /v1/files/{id}

200 OK — same shape as upload response
404 Not Found
```

### Delete File

```
DELETE /v1/files/{id}

200 OK
{"id": "file_abc123", "object": "file", "deleted": true}

404 Not Found
```

### Get File Content

```
GET /v1/files/{id}/content

200 OK
Content-Type: <original media type>
<binary content>

404 Not Found
```

---

## Conversations

Conversations are created automatically. When a response has no `previous_response_id`,
the server creates a new conversation and assigns the response to it. When a response
has a `previous_response_id` pointing to the **latest** response in a conversation,
it joins that conversation.

When `previous_response_id` points to a **non-latest** response (a fork), the server
creates a new conversation. Items up to and including the fork point are copied into
the new conversation with new response IDs, and the new response is added there.
The original conversation is unchanged. Each conversation is always a linear thread
— no branching. Response IDs are globally unique, so `previous_response_id` is
never ambiguous across conversations.

Clients may optionally pass a conversation ID when creating responses (must be
paired with `previous_response_id`). Conversation APIs are primarily for
**retrieval** — listing past conversations, loading message history, and finding
the latest response ID to continue from.

### List Conversations

```
GET /v1/conversations

Query parameters:
  limit (integer, optional, default: 20, max: 100)
    Number of conversations to return.

  after (string, optional)
    Cursor for forward pagination. Pass the `last_id` from a previous response.

  before (string, optional)
    Cursor for backward pagination. Pass the `first_id` from a previous response.

  order (string, optional, default: "desc")
    Sort order. Either "asc" or "desc".

  sort_by (string, optional, default: "created_at")
    Column to sort on. Either "created_at" or "updated_at".

200 OK
{
  "object": "list",
  "data": [
    {"id": "conv_abc123", "object": "conversation", "title": null, "created_at": ..., "updated_at": ...},
    {"id": "conv_def456", "object": "conversation", "title": "Weather chat", "created_at": ..., "updated_at": ...}
  ],
  "first_id": "conv_abc123",
  "last_id": "conv_def456",
  "has_more": false
}
```

Results ordered by `sort_by` column descending (newest first) by default.

### Get Conversation

```
GET /v1/conversations/{id}

200 OK
{
  "id": "conv_abc123",
  "object": "conversation",
  "title": null,
  "created_at": 1774118382,
  "updated_at": 1774118400
}

404 Not Found
```

### List Conversation Items

```
GET /v1/conversations/{id}/items

Query parameters:
  limit (integer, optional, default: 20, max: 100)
  after (string, optional)
  before (string, optional)
  order (string, optional, default: "asc")
    Sort order by position in conversation. Either "asc" (chronological) or "desc".

200 OK
{
  "object": "list",
  "data": [
    {"id": "msg_aaa", "response_id": "resp_001", "type": "message",
     "role": "user", "status": "completed",
     "content": [{"type": "input_text", "text": "What's the weather?"}]},
    {"id": "msg_bbb", "response_id": "resp_001", "model": "my-agent", "type": "message",
     "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "It's sunny in SF.", "annotations": []}]},
    {"id": "msg_ccc", "response_id": "resp_002", "type": "message",
     "role": "user", "status": "completed",
     "content": [{"type": "input_text", "text": "And tomorrow?"}]},
    {"id": "fc_ddd", "response_id": "resp_002", "model": "my-agent", "type": "function_call",
     "status": "completed", "name": "get_weather",
     "arguments": "{\"location\": \"SF\", \"date\": \"tomorrow\"}", "call_id": "call_001"},
    {"id": "fco_eee", "response_id": "resp_002", "type": "function_call_output",
     "status": "completed",
     "call_id": "call_001", "output": "{\"forecast\": \"rain\", \"high\": 58}"},
    {"id": "msg_fff", "response_id": "resp_002", "model": "my-agent", "type": "message",
     "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "Rain expected, high of 58°F.", "annotations": []}]}
  ],
  "first_id": "msg_aaa",
  "last_id": "msg_fff",
  "has_more": false
}

404 Not Found — conversation doesn't exist
```

Items include all input and output messages, function calls, and function call
outputs accumulated across all responses in this conversation. Each item carries
a `response_id` linking it to the response that produced it. Model-produced items
(assistant messages, function calls, reasoning) include a `model` field identifying
the agent. User messages and function call outputs do not have `model` — the agent
is always recoverable from `response_id` if needed. To continue a conversation,
pass the `response_id` from the last item as `previous_response_id`.

### Update Conversation

```
PATCH /v1/conversations/{id}
Content-Type: application/json

{"title": "Weather chat"}

200 OK
{
  "id": "conv_abc123",
  "object": "conversation",
  "title": "Weather chat",
  "created_at": 1774118382,
  "updated_at": 1774118400
}

404 Not Found
400 Bad Request — invalid field
```

Currently only `title` (string | null) is updatable.

### Delete Conversation

```
DELETE /v1/conversations/{id}

200 OK
{"id": "conv_abc123", "object": "conversation.deleted", "deleted": true}

404 Not Found
```

Deletes the conversation and all associated responses. Cancels any in-flight
responses in the conversation before deleting.

### Typical Chat UI Flow

```
1. User opens app
   → GET /v1/conversations              → show list of past chats

2. User clicks a conversation
   → GET /v1/conversations/{id}/items   → render message history
   → take response_id from the last item (e.g. "resp_002")

3. User sends a new message
   → POST /v1/responses {model: "my-agent", input: "...", previous_response_id: "resp_xyz"}
   → response is automatically added to the same conversation

4. User starts a brand new chat
   → POST /v1/responses {model: "my-agent", input: "..."}
   → no previous_response_id, so server creates a new conversation

5. User clicks "regenerate" on the last response
   → POST /v1/responses/{id}/cancel           → stop and preserve the response
   → POST /v1/responses {model: "my-agent", input: "...", previous_response_id: "resp_prev"}
   → same previous_response_id as the cancelled response used, creating a fork
   → new conversation is created with copied history + the new response

6. User steers a running agent
   → POST /v1/responses {model: "my-agent", input: "Focus on X instead",
       previous_response_id: "resp_in_progress"}
   → input delivered to running agent's inbox; no new response created
```

---

## Inference (OpenResponses-compatible)

The `model` field is the bridge between the two namespaces: the caller passes the
agent's `name` as `model` in the request, and the response echoes it back. To go
from a response back to an agent, use the `model` field in the response.

### Create Response

```
POST /v1/responses
Content-Type: application/json

{
  "model": "my-agent",
  "input": "What's the weather in SF?",
  "stream": true,
  "instructions": "Respond in French",
  "previous_response_id": "resp_xyz",
  "conversation": {"id": "conv_abc123"}
}

Accepted request fields:

  model (string, required)
    Agent name to invoke. Maps to the `name` given at agent creation time.

  input (string | array, required)
    The user's input. Either a plain string (shorthand for a single user message)
    or an array of message items following the OpenResponses item schema:
      - {type: "message", role: "user", content: [{type: "input_text", text: "..."}]}
      - {type: "message", role: "user", content: [
          {type: "input_text", text: "Summarize this"},
          {type: "input_image", image_url: "https://...", detail: "auto"},
          {type: "input_file", file_data: "<base64>", filename: "report.pdf"}
        ]}
      - {type: "message", role: "developer", content: "..."}
      - {type: "item_reference", id: "msg_xxx"}

    Content types in user messages:
      input_text:  {type, text}
      input_image: {type, image_url?, file_id?, detail?: "low"|"high"|"auto"}
      input_file:  {type, file_id?, file_data?, file_url?, filename?}

  stream (boolean, optional, default: false)
    If true, return a text/event-stream SSE connection with incremental events.
    If false, block until completion and return a single JSON response.

  background (boolean, optional, default: false)
    If true, execution is durable — the agent continues running even if the client
    disconnects. If false, execution is tied to the connection. Disconnect stops
    execution.

  store (boolean, optional, default: true)
    Must be true. If false, returns 400. Ephemeral responses are not supported.
    All responses are persisted and retrievable via GET /v1/responses/{id}.

  instructions (string | null, optional, default: null)
    Per-request instructions layered on top of the agent's built-in instructions
    (from AGENTS.md). Use for per-request steering like "respond in French."

  previous_response_id (string | null, optional, default: null)
    ID of a prior response to continue a multi-turn conversation. The server uses
    the prior response's context as conversation history.
    If the referenced response is still in progress, the input is delivered as
    a steering message to the running agent — the agent incorporates it at its
    next loop iteration. No new response is created; the server returns the
    existing in-progress response. Multiple steering messages queue up and are
    delivered in order. If the agent's inbox has closed (it is about to
    complete), a new response is created instead (normal multi-turn flow).

  conversation (object | null, optional, default: null)
    Explicitly associate the response with an existing conversation.
    Pass {"id": "conv_xxx"}. Requires previous_response_id to also be set;
    returns 400 if conversation is provided without previous_response_id.
    The referenced response must belong to this conversation; returns 400
    if they don't match. Returns 400 if the previous_response_id is not the
    latest response in the conversation (fork + explicit conversation is not
    allowed — forks always auto-create a new conversation).
    If omitted: auto-creates a new conversation (when no previous_response_id
    or when forking), or joins the conversation of the referenced response.

  context_management (array | null, optional, default: null)
    Strategies for managing conversation context that exceeds token limits.
    Currently supports one strategy:
      [{"type": "compaction", "compact_threshold": 50000}]
    When the conversation chain exceeds `compact_threshold` tokens, the server
    auto-compacts: a `compaction` item with encrypted content appears in the
    response output alongside the normal response. Subsequent requests via
    `previous_response_id` automatically use the compressed context. The
    conversation stays intact — no fork, no new conversation.

Ignored fields (agent controls these — silently dropped if provided):
  temperature, top_p, tools, tool_choice, reasoning,
  max_output_tokens, frequency_penalty, presence_penalty,
  parallel_tool_calls, max_tool_calls, top_logprobs

404 Not Found — unknown model (no agent with that name)
400 Bad Request — invalid previous_response_id (not found)
400 Bad Request — store: false is not supported
400 Bad Request — conversation provided without previous_response_id
400 Bad Request — previous_response_id does not belong to the specified conversation
400 Bad Request — conversation provided with a fork (previous_response_id is not latest)
400 Bad Request — invalid input format
```

**Non-streaming response** (`stream: false`):

```
200 OK
Content-Type: application/json

{
  "id": "resp_abc123",
  "object": "response",
  "status": "completed",
  "model": "my-agent",
  "created_at": 1774118382,
  "completed_at": 1774118390,
  "output": [
    {
      "id": "rs_abc789",
      "type": "reasoning",
      "summary": [{"type": "summary_text", "text": "Considered current SF weather data..."}],
      "content": null,
      "encrypted_content": null
    },
    {
      "id": "msg_def456",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "La météo à SF est...", "annotations": []}
      ]
    }
  ],
  "background": false,
  "store": true,
  "usage": {
    "input_tokens": 42,
    "output_tokens": 108,
    "output_tokens_details": {"reasoning_tokens": 30},
    "total_tokens": 150
  },
  "previous_response_id": null,
  "conversation": {"id": "conv_abc123"},
  "instructions": "Respond in French",
  "error": null,
  "incomplete_details": null
}
```

**Background, non-streaming response** (`background: true, stream: false`):

Returns immediately without waiting for completion. Client polls via GET.

```
200 OK
Content-Type: application/json

{
  "id": "resp_abc123",
  "object": "response",
  "status": "queued",
  "model": "my-agent",
  "created_at": 1774118382,
  "completed_at": null,
  "output": [],
  "background": true,
  "store": true,
  "usage": null,
  "previous_response_id": null,
  "conversation": {"id": "conv_def456"},
  "instructions": null,
  "error": null,
  "incomplete_details": null
}
```

**Streaming response** (`stream: true`):

```
200 OK
Content-Type: text/event-stream

event: response.created
data: {"type":"response.created","response":{...},"sequence_number":0}

event: response.in_progress
data: {"type":"response.in_progress","response":{...},"sequence_number":1}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{...},"sequence_number":2}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"La météo","sequence_number":3}

...

event: response.completed
data: {"type":"response.completed","response":{...},"sequence_number":N}

data: [DONE]
```

If `background: true` and the client disconnects, the agent continues executing.
If `background: false` and the client disconnects, execution stops.

**Supported streaming event types** (all from OpenResponses spec):

Response lifecycle:
  `response.created`              — response object created
  `response.queued`               — response queued for execution (background)
  `response.in_progress`          — execution started
  `response.completed`            — execution finished successfully
  `response.failed`               — execution failed
  `response.incomplete`           — execution stopped early

Output items:
  `response.output_item.added`    — new output item (message, function_call, reasoning, etc.)
  `response.output_item.done`     — output item finished

Text output:
  `response.content_part.added`   — content part added to a message
  `response.content_part.done`    — content part finished
  `response.output_text.delta`    — incremental text chunk
  `response.output_text.done`     — text content complete
  `response.output_text_annotation.added` — annotation added to text

Refusal:
  `response.refusal.delta`        — incremental refusal text
  `response.refusal.done`         — refusal complete

Function calls:
  `response.function_call.arguments.delta` — incremental function arguments
  `response.function_call.arguments.done`  — function arguments complete

Reasoning:
  `response.reasoning.delta`               — incremental reasoning content
  `response.reasoning.done`                — reasoning complete
  `response.reasoning_summary.delta`       — incremental reasoning summary
  `response.reasoning_summary.done`        — reasoning summary complete
  `response.reasoning_summary_part.added`  — reasoning summary part added
  `response.reasoning_summary_part.done`   — reasoning summary part complete

Error:
  `error`                         — error during streaming

Every event includes `type` (string) and `sequence_number` (integer, incrementing from 0).
The stream ends with `data: [DONE]`.

### Retrieve Response

```
GET /v1/responses/{response_id}

200 OK — same shape as non-streaming response above
         returns current state: completed, in_progress, failed, or incomplete
404 Not Found
```

Always returns a JSON snapshot, not a stream.
If still in progress, `output` is empty — partial output is not available via GET.
Returns 404 for unknown IDs and deleted responses.

### Cancel Response

```
POST /v1/responses/{response_id}/cancel

200 OK — same shape as non-streaming response above, with status: "cancelled"
404 Not Found
```

Stops execution if in progress. The response is preserved and still retrievable
via GET. Can be referenced as `previous_response_id` to continue or redirect
the conversation.

### Delete Response

```
DELETE /v1/responses/{response_id}

200 OK
{"id": "resp_abc123", "object": "response.deleted", "deleted": true}

404 Not Found
```

Removes the response entirely. Works on any stored response regardless of status.
Returns 404 for unknown IDs. Subsequent GET returns 404.

Deleting a mid-chain response does not cascade. Downstream responses survive with
a dangling `previous_response_id`. Chaining forward from a surviving downstream
response still works, but conversation history is truncated at the gap (the server
stops resolving context at the missing link). Attempting to create a new response
with `previous_response_id` pointing to a deleted response returns 400.


---

## Status Lifecycle

```
queued → in_progress → completed
                     → failed
                     → incomplete
                     → cancelled
```

- **completed**: agent finished successfully.
- **failed**: an error prevented execution. `error` is populated:
  `{"code": "server_error", "message": "An internal error occurred."}`
- **incomplete**: agent made progress but stopped early. `incomplete_details` is populated:
  `{"reason": "max_output_tokens"}` (other reasons: content filter, turn limit).
- **cancelled**: execution was stopped via `POST /v1/responses/{id}/cancel`.
  Response is preserved and referenceable as `previous_response_id`.

---

## Background × Stream Behavior Matrix

Tested empirically against the OpenAI Responses API (the reference OpenResponses implementation).

| `background` | `stream` | Behavior |
|---|---|---|
| `false` | `false` | Blocking HTTP call. Server holds connection until completion, returns full JSON response. If connection drops, execution stops. |
| `false` | `true` | Streaming SSE connection. Events flow incrementally. If connection drops, execution stops. |
| `true` | `false` | Returns immediately with `status: "queued"` and empty `output`. Execution continues server-side. Client polls via `GET /v1/responses/{id}` to check status and retrieve results. |
| `true` | `true` | Streams events (queued → in_progress → deltas → completed) AND execution is durable. If connection drops mid-stream, execution continues. Client reconnects via `GET /v1/responses/{id}` — output is empty until completed. |

### Key observations from OpenAI's implementation

**background: true returns immediately with queued status:**
```json
{"id": "resp_xxx", "status": "queued", "output": [], "completed_at": null}
```

**GET retrieves current state at any point:**
```json
{"id": "resp_xxx", "status": "completed", "output": [{...full output...}]}
```

**background + stream is not contradictory:**
It means "start durable execution AND stream me events while I'm connected."
The stream shows `response.created` → `response.queued` → `response.in_progress` →
text deltas → `response.completed`, same as non-background streaming. The difference
is that the server keeps going if the client disconnects.

**Streaming events include sequence_number:**
Each event has an incrementing `sequence_number` (0, 1, 2, ...). However, OpenAI
does not support stream resumption — there is no way to reconnect and say "give me
events from sequence N onward." The reconnection path is always a full GET snapshot.

**DELETE cancels and removes:**
Returns `{"id": "resp_xxx", "object": "response.deleted", "deleted": true}`.
The response is fully removed — subsequent GET returns 404.

### Laptop-closing scenario (the primary use case for agents)

1. Chat UI sends `POST /v1/responses` with `background: true, stream: true`
2. Events stream to the UI while the user is watching
3. User closes laptop — SSE connection drops
4. Agent continues executing on the server (because `background: true`)
5. User reopens laptop — chat UI calls `GET /v1/responses/{id}`
6. If completed: full response with output is returned
7. If still in progress: `output` is empty, UI polls again until completed

---

## Not Yet

- `store: false` (ephemeral responses) — responses that aren't persisted, don't create or update
  conversations, and aren't retrievable via GET. Currently returns 400. OpenResponses spec supports
  this for stateless one-shot queries where the caller doesn't need server-side persistence.
- `PUT /api/agents/{id}` — update agent (new bundle)
- Stream resumption on GET (sequence_number-based reconnection)
- Authentication
- Rate limiting
- User filtering on conversation list (by metadata, dedicated user field, or auth identity)
- Conversation update metadata (beyond title)
- Search across conversations (full-text search over message content)
- `GET /v1/responses` — list responses. OpenAI has this (browser-session-only). Lower priority
  because conversation items endpoint covers the main browsing use case.
- Multi-user identity (`user` field on requests/items to attribute messages in shared conversations)
- `logprobs` on `output_text` content blocks (optional in OpenResponses spec, used with `top_logprobs`)
- Request params from OpenResponses spec: `user` (end-user identifier for abuse monitoring),
  `include` (controls extra fields in response), `stream_options` (streaming behavior),
  `service_tier` (request tier), `prompt_cache_key` / `prompt_cache_retention` (prompt caching)
- `metadata` on responses — caller-attached key-value pairs (max 16 keys, keys ≤64 chars, values
  ≤512 chars). Stored with the response, returned on retrieval. OpenAI supports this on their
  Responses API. Would need: field on request/response models, Task dataclass, TaskStore.create().
- `purpose` field on file uploads (e.g. `"input"`, `"fine-tune"`) — tag indicating how the file
  will be used. OpenAI requires this; we may add it later if needed.
- Audio input (`input_audio` content type)
- `GET /v1/responses/{id}/input_items` — returns the reconstructed input context (new input +
  resolved conversation history from `previous_response_id` chain). OpenAI-specific (not in
  OpenResponses spec). Lower priority because `GET /v1/conversations/{id}/items` already exposes
  full conversation history. Main value is debugging (see exactly what the model received,
  especially after compaction).
- `input` field on response object — OpenResponses spec echoes back the input items on the
  response. Useful for record-keeping but clients already know what they sent.
- `next_response_ids` on response object — optional field in OpenResponses spec. Array of
  response IDs created as follow-ups (forward pointers). Less critical given our fork model
  creates new conversations; conversation items endpoint covers forward traversal.
- Additional output item types: `image_generation_call`, `web_search_call`, `file_search_call`,
  `code_interpreter_call`, `mcp_tool_call`, `computer_call`, `local_shell_call`, `apply_patch_call`,
  `compaction` (all defined in OpenResponses spec)
