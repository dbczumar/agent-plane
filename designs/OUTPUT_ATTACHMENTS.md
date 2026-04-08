# Output Attachments

## Problem

Agents can receive images and files as input (via `input_image` and
`input_file` content blocks) but cannot return them in output. All
assistant output is text-only: `output_text` blocks inside `message`
items. Three categories of output attachments are unsupported:

1. **LLM-generated images** — OpenAI's `image_generation_call` returns
   base64 image data inline. Today this flows through as a
   `NativeToolOutput` opaque dict — it reaches the SSE stream but is
   not persisted, not downloadable via the files API, and not included
   in conversation history.

2. **LLM-generated files** — OpenAI's `code_interpreter_call` produces
   files referenced via `container_file_citation` annotations. Same
   problem: opaque pass-through, not persisted.

3. **Agent-produced files** — A Claude SDK executor writes files to
   disk via Edit/Write tools. The files exist in `storage_dir` but
   there is no mechanism for the agent to say "here is a file I
   created" in its response output, and no way for the client to
   download it.

All three categories need the same thing: a way to include file
references in agent output that clients can download via the existing
`GET /v1/files/{file_id}/content` endpoint.

---

## Design Decisions

### One mechanism: `output_file` content blocks with `file_id`

Instead of three different output types (one per category), all
attachments use the same content block type inside assistant messages:

```json
{
  "type": "output_file",
  "file_id": "file_abc123",
  "filename": "chart.png",
  "content_type": "image/png"
}
```

The client downloads the content via the existing endpoint:
`GET /v1/files/{file_id}/content`.

**Why one type, not separate `output_image` / `output_file`:**

- The download mechanism is identical regardless of content type.
- The client knows it's an image from `content_type: "image/png"` — no
  need for a type-level distinction.
- OpenAI splits these because `image_generation_call` is inline base64
  while `code_interpreter_call` uses container references. We always
  use `file_id` references — the split is unnecessary.
- One content block type means one code path for persistence, SSE,
  compaction, and prompt construction.

### Files are stored via the existing file infrastructure

Generated files are stored the same way uploaded files are:

- Metadata in `FileStore` (filename, content_type, size)
- Binary content in `ArtifactStore` (keyed by `file_id`)
- Downloaded via `GET /v1/files/{file_id}/content`

No new stores, no new endpoints, no new DB tables.

### Native tool outputs become first-class when they contain files

Today, `NativeToolOutput` items (e.g. `image_generation_call`,
`code_interpreter_call`) are opaque dicts that flow through SSE but
are NOT persisted to the conversation store. With this change:

1. The workflow inspects native tool outputs for file content.
2. If an `image_generation_call` contains a base64 `result`, the
   workflow stores it in the file infrastructure and emits an
   `output_file` block on the assistant message.
3. If a `code_interpreter_call` contains `container_file_citation`
   annotations, the workflow downloads the file from OpenAI's
   container API, stores it locally, and emits `output_file` blocks.

The native tool output item ALSO flows through SSE for clients that
want the raw provider-specific data. The `output_file` block is the
canonical, provider-agnostic representation.

### Agent-produced files use the same path

When a Claude SDK executor (or any internal executor) produces files,
the executor is responsible for storing them via the file
infrastructure and including `output_file` blocks in the response.

For the Claude SDK executor specifically:

- The SDK's `ToolResultBlock` for file-producing tools (Write, Edit)
  contains the file path in `storage_dir`.
- The executor reads the file content, calls `file_store.create()` +
  `artifact_store.put()`, and includes the `file_id` in a
  `ToolCallObserved` event's result.
- At turn end, the workflow builds the assistant message with
  `output_file` blocks referencing the stored files.

For the `DefaultExecutor`, file attachments come from provider-native
tools (image generation, code interpreter) — handled by the native
tool output extraction described above.

For the `RemoteExecutor`, the remote service can include `output_file`
blocks directly in its SSE stream. The workflow stores the referenced
files if they arrive as inline base64, or passes through `file_id`
references if the remote service pre-stored them.

---

## Content Block Schema

### `output_file` (new)

```json
{
  "type": "output_file",
  "file_id": "file_abc123",
  "filename": "chart.png",
  "content_type": "image/png"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | Always `"output_file"` |
| `file_id` | string | yes | Reference to a file in the file store. Client downloads via `GET /v1/files/{file_id}/content`. |
| `filename` | string | yes | Original or generated filename, e.g. `"chart.png"`. |
| `content_type` | string | yes | MIME type, e.g. `"image/png"`, `"text/csv"`, `"application/pdf"`. |

This block appears inside assistant message `content` arrays alongside
`output_text` blocks:

```json
{
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "output_text", "text": "Here's the chart you requested:"},
    {
      "type": "output_file",
      "file_id": "file_abc123",
      "filename": "chart.png",
      "content_type": "image/png"
    }
  ]
}
```

### `output_text` annotations (new, optional)

For inline references within text (e.g. "see [chart.png]"), the
existing `output_text` block gains an optional `annotations` array
matching OpenAI's pattern:

```json
{
  "type": "output_text",
  "text": "Here's the analysis [1]...",
  "annotations": [
    {
      "type": "file_citation",
      "file_id": "file_abc123",
      "filename": "report.pdf",
      "index": 0
    }
  ]
}
```

Annotations are informational — the canonical attachment is the
`output_file` block. Annotations provide inline context for where in
the text a file is referenced.

---

## SSE Events

### New: `response.output_file.done`

Emitted when an output file is ready for download:

```json
{
  "type": "response.output_file.done",
  "file_id": "file_abc123",
  "filename": "chart.png",
  "content_type": "image/png",
  "output_index": 2,
  "sequence_number": 5
}
```

Clients that want to show a thumbnail or download link can react to
this event immediately. The file is already stored and downloadable
by the time this event is emitted.

### Existing: `response.output_item.done`

The completed assistant message item (with `output_file` blocks in
its content array) is emitted as normal via `response.output_item.done`.
The `output_file.done` event is a convenience for eager rendering —
clients that only consume `output_item.done` still get the full picture.

---

## Conversation Storage

`output_file` blocks are stored as part of the assistant message's
`content` array in `MessageData`. No new item type needed — files are
content blocks within messages, not standalone items.

```python
MessageData(
    role="assistant",
    content=[
        {"type": "output_text", "text": "Here's the chart:"},
        {
            "type": "output_file",
            "file_id": "file_abc123",
            "filename": "chart.png",
            "content_type": "image/png",
        },
    ],
    agent="my-agent",
)
```

`MessageData.content` is already `list[dict[str, Any]]` — no schema
change needed. The heterogeneous content block pattern is the same as
input (`input_text`, `input_image`, `input_file`).

---

## Prompt Construction

`history_to_input_items()` must handle `output_file` blocks when
replaying history. On subsequent turns, the LLM should know what files
it previously produced. Two options:

**Option A: Convert to text reference.** Replace `output_file` blocks
with a text description: `"[Attached file: chart.png (image/png)]"`.
The LLM knows it produced a file without seeing the binary content.
Simple, works with all providers.

**Option B: Convert to `input_file` block.** Re-inject the file as an
`input_file` block so the LLM can see the content on subsequent turns.
Expensive (re-encodes the file every turn) and not always useful (the
LLM already produced the file).

**Decision: Option A.** Convert to text reference. If the LLM needs to
re-examine a file it produced, the user can re-attach it as input. This
avoids bloating the context with file content the LLM already knows
about.

---

## Compaction

`output_file` blocks are treated like `input_image` / `input_file`
blocks during compaction:

- **Layer 1:** Clear binary content (replace `output_file` blocks
  outside the recent window with
  `"[output file cleared — file_id: file_abc123]"`). The `file_id`
  is preserved so the client can still download the file via the API.
- **Layer 2:** The text reference from Layer 1 is included in the
  summarization input. The summary captures "the agent produced
  chart.png" without the binary content.

---

## Implementation Plan

### Phase 1: `output_file` content blocks (core)

**Changed files:**

1. **`runtime/workflow.py`** — `_build_assistant_item()`: include
   `output_file` blocks in the assistant message content array when
   file attachments are present.

2. **`runtime/prompt.py`** — `history_to_input_items()`: convert
   `output_file` blocks to text references when replaying history.

3. **`runtime/compaction.py`** — `_clear_binary_content()`: handle
   `output_file` blocks the same as `input_image` / `input_file`.

4. **`server/routes/responses.py`** — No changes to the response
   builder. `output_file` blocks are already inside `MessageData.content`
   which is serialized as-is.

### Phase 2: Image generation extraction

**New file:**

5. **`runtime/file_extraction.py`** — Extract files from native tool
   outputs:
   - `image_generation_call`: decode base64 `result`, store via
     file infrastructure, return `output_file` block.
   - `code_interpreter_call`: detect `container_file_citation`
     annotations, download from OpenAI container API, store locally,
     return `output_file` blocks.

**Changed files:**

6. **`runtime/workflow.py`** — `_emit_native_tool_items()`: after
   emitting the raw native tool output, also extract files and
   append `output_file` blocks to the assistant message.

### Phase 3: Claude SDK executor file extraction

**Changed files:**

7. **`runtime/claude_agents_executor.py`** — When the SDK writes
   files via Edit/Write tools, detect file creation in
   `ToolCallObserved` events, store via file infrastructure, and
   include `output_file` references. Requires access to
   `file_store` and `artifact_store` — passed via `ExecutorContext`.

### Phase 4: Remote executor file support

8. **`runtime/executor.py`** — `RemoteExecutor`: if the remote
   service SSE stream includes `output_file` events with inline
   base64, store them locally and rewrite to `file_id` references.
   If the remote service already provides `file_id` references,
   pass them through (assumes the remote service's file store is
   accessible to the client, or a proxy endpoint is added).

---

## What Changes vs. What Stays the Same

| Component | Changes? | Notes |
|-----------|----------|-------|
| `FileStore` | No | Same upload/download API |
| `ArtifactStore` | No | Same put/get for binary content |
| `GET /v1/files/{id}/content` | No | Already returns file bytes |
| `POST /v1/files` | No | Client uploads still work |
| `MessageData` | No | `content: list[dict[str, Any]]` already flexible |
| `ConversationStore` | No | Stores `MessageData` as-is |
| Conversation items API | No | Returns `MessageData` content as-is |
| `history_to_input_items()` | Yes | Convert `output_file` to text reference |
| `_clear_binary_content()` | Yes | Handle `output_file` blocks in compaction |
| `_build_assistant_item()` | Yes | Include `output_file` blocks in content |
| `_emit_native_tool_items()` | Yes | Extract files from native tool outputs |
| SSE events | Yes | Add `response.output_file.done` |
| `ExecutorContext` | Yes | Add file_store + artifact_store for Phase 3 |

---

## Examples

### Image generation → output_file

User asks: "Generate a chart of Q3 revenue."

LLM response includes `image_generation_call` with base64 result.

Workflow:
1. Decodes base64 → stores as `file_abc123` (image/png)
2. Emits SSE: `response.output_file.done` with `file_id`
3. Builds assistant message:
   ```json
   {
     "content": [
       {"type": "output_text", "text": "Here's the Q3 revenue chart:"},
       {"type": "output_file", "file_id": "file_abc123",
        "filename": "q3_revenue.png", "content_type": "image/png"}
     ]
   }
   ```
4. Client downloads: `GET /v1/files/file_abc123/content`

### Claude SDK executor → output_file

Claude Code writes `solution.py` via the Edit tool.

Executor:
1. Detects file write in `ToolCallObserved` for Edit tool
2. Reads file from `storage_dir/solution.py`
3. Stores as `file_xyz789` (text/x-python)
4. Includes reference in turn output

Workflow builds assistant message:
```json
{
  "content": [
    {"type": "output_text", "text": "I created solution.py with the implementation."},
    {"type": "output_file", "file_id": "file_xyz789",
     "filename": "solution.py", "content_type": "text/x-python"}
  ]
}
```

### Subsequent turn — history replay

On the next turn, `history_to_input_items()` converts:
```json
{"type": "output_file", "file_id": "file_abc123",
 "filename": "q3_revenue.png", "content_type": "image/png"}
```
to:
```json
{"type": "output_text",
 "text": "[Attached file: q3_revenue.png (image/png, file_id=file_abc123)]"}
```

The LLM knows it produced the file without re-seeing the binary
content.
