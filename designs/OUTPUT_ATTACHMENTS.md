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

### Use `output_text` annotations with `file_id` references

No new content block types. File attachments are represented as
`file_citation` annotations on `output_text` blocks:

```json
{
  "type": "output_text",
  "text": "I created solution.py with the implementation.",
  "annotations": [
    {
      "type": "file_citation",
      "file_id": "file_abc123",
      "filename": "solution.py",
      "content_type": "text/x-python"
    }
  ]
}
```

The text is the human-readable description. The `file_id` is the
download reference. The client fetches file content via the existing
`GET /v1/files/{file_id}/content` endpoint. No content is inlined
in the response — no bloated SSE streams.

**Why annotations, not a new content block type:**

- **OpenResponses compatible.** The spec defines `output_text` with
  an `annotations` array. `file_citation` is an existing OpenAI
  annotation type. No spec extensions needed.
- **Backward compatible.** Clients that don't understand annotations
  still see the text description. The file reference is additive.
- **One mechanism.** Same pattern for images, code files, PDFs,
  CSVs — the `content_type` field tells the client what it is,
  the `file_id` tells it where to download.
- **Aligns with OpenResponses Issue #66** (MIME Types for Content
  Parts) which proposes `content_type` metadata on content parts.

**Why not inline file content in the response:**

- SSE streams bloat with base64 images or full code files.
- The file store already exists — files are uploaded via
  `POST /v1/files` and downloaded via `GET /v1/files/{id}/content`.
  Output files use the same infrastructure.
- Clients that want the content fetch it on demand. Clients that
  only need the filename and type use the annotation metadata.

### Annotation schema: `file_citation`

```json
{
  "type": "file_citation",
  "file_id": "file_abc123",
  "filename": "chart.png",
  "content_type": "image/png"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"file_citation"` |
| `file_id` | string | yes | File store reference. Download via `GET /v1/files/{file_id}/content`. |
| `filename` | string | yes | Original or generated filename. |
| `content_type` | string | yes | MIME type, e.g. `"image/png"`, `"text/x-python"`. |

OpenAI's `file_citation` has `file_id`, `filename`, and `index`.
We add `content_type` (per Issue #66's direction) and omit `index`
(agent-plane doesn't insert citations at character offsets — the
annotation applies to the whole text block).

### Files are stored via the existing file infrastructure

Generated and agent-produced files go through the same pipeline as
client-uploaded files:

- Metadata in `FileStore` (filename, content_type, size)
- Binary content in `ArtifactStore` (keyed by `file_id`)
- Downloaded via `GET /v1/files/{file_id}/content`

No new stores, no new endpoints, no new DB tables.

---

## How Each Category Works

### 1. LLM-generated images (`image_generation_call`)

Today: `image_generation_call` flows through as a `NativeToolOutput`
with base64 in the `result` field. Not persisted, not downloadable.

After:

1. Workflow inspects native tool output for `image_generation_call`.
2. Decodes base64 `result` → stores via `file_store.create()` +
   `artifact_store.put()` → gets `file_id`.
3. Builds `file_citation` annotation on the assistant message's
   `output_text` block.
4. The raw `image_generation_call` item ALSO flows through SSE for
   clients that want the provider-specific data.

### 2. LLM-generated files (`code_interpreter_call`)

Today: `code_interpreter_call` flows through with
`container_file_citation` annotations. Not stored locally.

After:

1. Workflow detects `container_file_citation` annotations.
2. Downloads file from OpenAI's container API via
   `GET /containers/{container_id}/files/{file_id}/content`.
3. Stores locally → gets agent-plane `file_id`.
4. Builds `file_citation` annotation with the local `file_id`.

### 3. Agent-produced files (Claude SDK executor)

Today: Files exist in `storage_dir` but aren't referenced in output.

After:

1. Executor detects file-producing tool calls (Write, Edit) in
   `ToolCallObserved` events.
2. Reads file from `storage_dir`, stores via file infrastructure.
3. Includes `file_citation` annotation on the tool result or
   assistant message.

Requires `file_store` and `artifact_store` access in the executor.
These can be added to `ExecutorContext`.

---

## Conversation Storage

Annotations live inside `output_text` blocks within
`MessageData.content`. No new item type needed:

```python
MessageData(
    role="assistant",
    content=[
        {
            "type": "output_text",
            "text": "Here's the chart you requested:",
            "annotations": [
                {
                    "type": "file_citation",
                    "file_id": "file_abc123",
                    "filename": "chart.png",
                    "content_type": "image/png",
                }
            ],
        },
    ],
    agent="my-agent",
)
```

`MessageData.content` is `list[dict[str, Any]]` — annotations are
just a nested list within the dict. No schema change needed.

---

## Prompt Construction

On subsequent turns, `history_to_input_items()` encounters
`output_text` blocks with `file_citation` annotations. The text
description is already human-readable ("Here's the chart you
requested:"). The annotation metadata is stripped or preserved
depending on provider support:

- **For providers that support annotations** (OpenAI): pass through.
- **For providers that don't**: strip annotations, keep text only.
  The LLM still sees "Here's the chart you requested:" and knows
  it produced a file.

No binary content enters the context window. The file exists in
the file store for the client to download — the LLM doesn't need
to re-see it.

---

## Compaction

`file_citation` annotations are metadata on text blocks, not binary
content. They survive compaction naturally:

- **Layer 1:** No change — there's no binary content to clear.
  The annotation is just `{type, file_id, filename, content_type}`.
- **Layer 2:** The text ("Here's the chart you requested:") is
  included in summarization input. The summary captures that the
  agent produced a file. Annotations are stripped before
  summarization (they're metadata, not content for the LLM).
- **Layer 3:** If the message is truncated, the annotation goes
  with it. The file still exists in the file store.

---

## SSE Events

### New: `response.output_file.done`

Emitted when an output file is stored and ready for download:

```json
{
  "type": "response.output_file.done",
  "file_id": "file_abc123",
  "filename": "chart.png",
  "content_type": "image/png",
  "sequence_number": 5
}
```

Clients that want to show a thumbnail or download link can react
immediately. The file is already stored and downloadable by the time
this event is emitted.

### Existing: `response.output_item.done`

The completed assistant message (with annotations) is emitted as
normal via `response.output_item.done`. Clients that only consume
`output_item.done` still get the full picture via annotations.

---

## Implementation Plan

### Phase 1: Annotation support in assistant messages

1. **`runtime/workflow.py`** — `_build_assistant_item()`: propagate
   annotations from `output_text` blocks to the persisted message.
   Today annotations are silently dropped.

2. **`runtime/prompt.py`** — `history_to_input_items()`: strip
   `annotations` from `output_text` blocks when building input
   (annotations are output metadata, not input content).

3. **`runtime/compaction.py`** — `_clear_binary_content()`: strip
   annotations before passing to summarization LLM.

### Phase 2: Image generation extraction

4. **`runtime/file_extraction.py`** (new) — Extract files from
   native tool outputs:
   - `image_generation_call`: decode base64, store, return annotation.
   - `code_interpreter_call`: detect container citations, download,
     store, return annotations.

5. **`runtime/workflow.py`** — `_emit_native_tool_items()`: after
   emitting raw native tool output, extract files and add
   annotations to the assistant message.

### Phase 3: Claude SDK executor file extraction

6. **`runtime/claude_agents_executor.py`** — Detect file-producing
   tool calls, store files, include annotations. Requires
   `file_store` and `artifact_store` on `ExecutorContext`.

### Phase 4: Remote executor file support

7. **`runtime/executor.py`** — `RemoteExecutor`: if the remote
   service SSE stream includes file annotations, verify files
   exist in the local file store or proxy them.

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
| Items API | No | Returns `MessageData` content with annotations |
| `history_to_input_items()` | Yes | Strip annotations from output blocks |
| `_clear_binary_content()` | Yes | Strip annotations before summarization |
| `_build_assistant_item()` | Yes | Propagate annotations to persisted message |
| `_emit_native_tool_items()` | Yes | Extract files from native tool outputs |
| SSE events | Yes | Add `response.output_file.done` |
| `ExecutorContext` | Yes | Add file_store + artifact_store for Phase 3 |

---

## Examples

### Image generation → file_citation annotation

User: "Generate a chart of Q3 revenue."

LLM response includes `image_generation_call` with base64 result.

Workflow:
1. Decodes base64 → stores as `file_abc123` (image/png)
2. Emits SSE: `response.output_file.done`
3. Builds assistant message:
   ```json
   {
     "type": "message",
     "role": "assistant",
     "content": [{
       "type": "output_text",
       "text": "Here's the Q3 revenue chart:",
       "annotations": [{
         "type": "file_citation",
         "file_id": "file_abc123",
         "filename": "q3_revenue.png",
         "content_type": "image/png"
       }]
     }]
   }
   ```
4. Client downloads: `GET /v1/files/file_abc123/content`

### Claude SDK executor → file_citation annotation

Claude Code writes `solution.py` via Edit tool.

Executor:
1. Detects file write in `ToolCallObserved`
2. Reads from `storage_dir/solution.py`, stores as `file_xyz789`
3. Includes annotation in turn output

Assistant message:
```json
{
  "type": "output_text",
  "text": "I created solution.py with the implementation.",
  "annotations": [{
    "type": "file_citation",
    "file_id": "file_xyz789",
    "filename": "solution.py",
    "content_type": "text/x-python"
  }]
}
```

### Subsequent turn — history replay

On the next turn, `history_to_input_items()` strips annotations:

```json
{"type": "output_text", "text": "I created solution.py with the implementation."}
```

The LLM sees the text description. The file exists in the file
store. No binary content in the context window.
