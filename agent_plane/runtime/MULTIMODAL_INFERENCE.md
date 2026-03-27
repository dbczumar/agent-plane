# Multimodal Inference

## Overview

Multimodal input (images and files) flows through three layers: upload, conversation storage, and the agent loop. Today, the first two layers work — files can be uploaded and referenced in messages, and content blocks are stored faithfully in conversation items. The third layer is broken: `history_to_input_items()` discards all non-text blocks, so the LLM never sees images or files.

This document traces the full lifecycle of an image and a file from the client to the LLM, identifies every component involved, and specifies what needs to change.

---

## Content Block Types

The API accepts three content block types inside user messages (per API.md):

```
input_text:   {type, text}
input_image:  {type, image_url?, file_id?, detail?}
input_file:   {type, file_id?, file_data?, file_url?, filename?}
```

`input_text` works end-to-end. `input_image` and `input_file` are stored but never reach the LLM.

**`detail` field on `input_image`:** This is a pass-through from the OpenAI Responses API. It controls the resolution at which the model processes the image (`"low"`, `"high"`, or `"auto"`). We do not validate, default, or interpret this field today — it is accepted as-is, stored in the conversation item, and then dropped along with the rest of the `input_image` block by `_extract_text()`. Once multimodal blocks reach the LLM, `detail` should be forwarded to the provider unchanged. Validation (if any) is the provider's responsibility.

---

## Walkthrough: Image via file_id

The client wants to send an image for the agent to analyze. The image is not yet uploaded.

### Step 1 — Upload the image

```
POST /v1/files
Content-Type: multipart/form-data

file: <photo.png binary>
```

**What happens** (`server/routes/files.py:48-71`):

1. `await file.read()` reads the binary content.
2. `mimetypes.guess_type("photo.png")` → `"image/png"`.
3. `file_store.create(filename="photo.png", bytes=len(content), content_type="image/png")` persists metadata to the `files` table. Generates a unique ID like `file_abc123`.
4. `artifact_store.put("file_abc123", content)` writes the binary to the local filesystem at `{storage_root}/file_abc123`.

**Response:**

```json
{
  "id": "file_abc123",
  "filename": "photo.png",
  "bytes": 204800,
  "created_at": 1711500000
}
```

**State after step 1:**
- `files` table: row with `id=file_abc123`, `content_type=image/png`
- Artifact store: binary blob at key `file_abc123`

### Step 2 — Send a message referencing the image

```
POST /v1/responses
{
  "model": "my-agent",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "file_id": "file_abc123", "detail": "auto"}
      ]
    }
  ]
}
```

**What happens** (`server/routes/responses.py`):

1. `_normalize_input()` receives the array and passes it through unchanged — it only transforms plain strings.
2. The handler creates a `NewConversationItem` with `type="message"` and `data=MessageData(role="user", content=[...both blocks...])`.
3. `conversation_store.append(conv_id, [item])` persists the item. The content list is serialized as JSON — both the `input_text` and `input_image` blocks survive.

**State after step 2:**
- `conversation_items` table: row with `data.content` containing both blocks, including `{"type": "input_image", "file_id": "file_abc123", "detail": "auto"}`

### Step 3 — Agent loop reconstructs history (THE GAP)

The workflow calls `history_to_input_items()` (`runtime/prompt.py:77-125`) to build the LLM's input. For each message item:

```python
text = _extract_text(item.data.content)
result.append({"role": item.data.role, "content": text})
```

`_extract_text()` (`runtime/prompt.py:16-34`) iterates through content blocks and only picks up `input_text`, `output_text`, and `text` types. The `input_image` block is silently dropped.

**Result:**

```python
[{"role": "user", "content": "What's in this image?"}]
```

The image reference is gone. The LLM sees a question about an image it was never shown.

### Step 4 — LLM call

The text-only input is passed to `client.responses.create()`. The LLM responds with something like "I don't see an image" or hallucinates.

### What step 3 should do instead

`history_to_input_items()` should preserve the full content block array for user messages instead of collapsing to text. But content blocks with `file_id` references need resolution first — the LLM doesn't have access to our file store. The `file_id` must be resolved to actual content the LLM can consume:

1. Look up `file_abc123` in the file store to get `content_type=image/png`.
2. Fetch the binary from the artifact store.
3. Base64-encode the bytes.
4. Emit the content block in a format the LLM provider understands.

For the **Responses API path** (OpenAI models), content blocks can be passed through directly — OpenAI handles `file_id` natively via their own file API. But our `file_id` is local, so we must inline the content regardless.

For the **Chat Completions path** (all other providers), the translation layer (`_responses_to_chat.py`) must convert `input_image` blocks to the Chat Completions vision format:

```python
# Responses API format (what we store)
{"type": "input_image", "file_id": "file_abc123", "detail": "auto"}

# Chat Completions format (what the LLM needs)
{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR...", "detail": "auto"}}
```

---

## Walkthrough: PDF file via file_id

The client wants the agent to analyze a PDF. Same upload-then-reference pattern.

### Step 1 — Upload the PDF

```
POST /v1/files
Content-Type: multipart/form-data

file: <report.pdf binary>
```

**What happens:** Same as the image flow. `file_store.create()` → `artifact_store.put()`. Returns `file_xyz789` with `content_type=application/pdf`.

### Step 2 — Send a message referencing the PDF

```
POST /v1/responses
{
  "model": "my-agent",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Summarize the key findings"},
        {"type": "input_file", "file_id": "file_xyz789", "filename": "report.pdf"}
      ]
    }
  ]
}
```

**What happens:** Same as image — both blocks are stored in the conversation item.

### Step 3 — Agent loop (THE GAP)

Same problem. `_extract_text()` drops the `input_file` block. The LLM sees "Summarize the key findings" with no file attached.

### What step 3 should do instead

File handling is more complex than images because provider support varies:

**Provider capabilities for file input:**

| Provider | Images (base64) | Images (URL) | PDF | Other files | Notes |
|---|---|---|---|---|---|
| OpenAI | Yes | Yes | Yes (via file search or base64 in Responses API) | Limited | Has native Responses API path — bypasses Chat Completions translation entirely |
| Anthropic | Yes (base64) | Yes (URL) | Yes (base64 as document type) | Limited | |
| Gemini | Yes (inline data) | No (must inline) | Yes (inline data) | Limited | |
| Bedrock | Yes (base64) | No (must inline) | Provider-dependent | Limited | |
| Vertex | Inherits Gemini | Inherits Gemini | Inherits Gemini | Inherits Gemini | Subclass of `GeminiAdapter` — gets multimodal for free |
| Databricks | Inherits OpenAI | Inherits OpenAI | Inherits OpenAI | Inherits OpenAI | Subclass of `OpenAICompatibleAdapter` |

Resolution for PDF files:
1. Look up `file_xyz789` in file store → `content_type=application/pdf`.
2. Fetch binary from artifact store.
3. Base64-encode.
4. Emit in provider-appropriate format.

For Chat Completions providers that don't natively support file content, the content may need to be extracted as text (e.g., PDF → text extraction) before being passed as a context block. This is a provider-specific concern.

---

## Walkthrough: Image via URL (no upload)

The client references an image by URL, skipping the upload step entirely:

```
POST /v1/responses
{
  "model": "my-agent",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this"},
        {"type": "input_image", "image_url": "https://example.com/photo.png", "detail": "high"}
      ]
    }
  ]
}
```

No upload step. The URL is stored in the conversation item as-is. At prompt construction time, the URL is passed through to providers that support URL-based images (OpenAI, Anthropic). For providers that don't support URLs (Gemini, Bedrock), the request is rejected with a clear error directing the client to upload via `POST /v1/files` instead.

**We never fetch user-provided URLs server-side.** Fetching arbitrary URLs is an SSRF vector — a client could point `image_url` at internal services or cloud metadata endpoints. Only `file_id` references are resolved, and only against our own file store.

**Currently broken at the same point:** `_extract_text()` drops the `input_image` block.

---

## Walkthrough: File via inline base64 (no upload)

The client sends file content directly:

```json
{
  "type": "input_file",
  "file_data": "JVBERi0xLjQK...",
  "filename": "report.pdf"
}
```

No upload step. The base64 data is stored in the conversation item. At prompt construction time, it's already in the right format — just needs to be routed to the provider's expected structure.

**Currently broken at the same point:** `_extract_text()` drops the `input_file` block.

---

## Component Map

Every component involved, what it does today, and what needs to change.

### Working — no changes needed

| Component | File | Role |
|---|---|---|
| File upload | `server/routes/files.py` | Upload, store, retrieve file content |
| File store | `stores/file_store/` | File metadata CRUD |
| Artifact store | `stores/artifact_store/` | Binary blob storage |
| API input acceptance | `server/routes/responses.py` | Accepts all content block types |
| Conversation storage | `stores/conversation_store/` | Persists content blocks as-is |

### Broken — changes required

| Component | File | Current Behavior | Required Behavior |
|---|---|---|---|
| History → input | `runtime/prompt.py` | `_extract_text()` drops non-text blocks | Preserve full content block arrays; resolve `file_id` references to inline content |
| Responses → Chat translation | `llms/_responses_to_chat.py` | Passes `content` as string only | Handle content block arrays; convert `input_image`/`input_file` to Chat Completions vision format |
| Provider adapters | `llms/adapters/*.py` | String-only `content` in messages | Accept content block arrays; convert to provider-native multimodal format |

*`file_id` validation and content caching are addressed in Phase 4. Full-text search (`extract_search_text()` in `db/utils.py`) correctly ignores multimodal blocks — image bytes are not searchable content, and this requires no changes.*

---

## Data Flow Summary

```
Client
  │
  ├─ POST /v1/files ─────────► FileStore + ArtifactStore
  │                                 │
  │                                 │ file_id
  │                                 ▼
  ├─ POST /v1/responses ─────► _normalize_input()
  │   (content blocks               │
  │    with file_id refs)            ▼
  │                            ConversationStore.append()
  │                                 │
  │                                 │ (persisted as-is, file_id refs intact)
  │                                 ▼
  │                            resolve_content_references()  ◄── NEW (Phase 1)
  │                                 │
  │                                 │ file_id → file_store lookup
  │                                 │ → artifact_store fetch
  │                                 │ → base64 encode → inline
  │                                 │ returns copies (originals unchanged)
  │                                 ▼
  │                            history_to_input_items()  ◄── CHANGE (Phase 1)
  │                                 │
  │                                 │ preserve content block arrays
  │                                 │ (all file_id refs already resolved)
  │                                 ▼
  │                      ┌──────────┴──────────┐
  │                      │                     │
  │                OpenAI native          Non-OpenAI
  │                (Responses API)        providers
  │                      │                     │
  │                      │          responses_input_to_chat_messages()
  │                      │            ◄── CHANGE (Phase 2)
  │                      │                     │
  │                      │                     │ convert input_image → image_url
  │                      │                     │ convert input_file → provider fmt
  │                      │                     ▼
  │                      │             Provider adapter
  │                      │               ◄── CHANGE (Phase 3)
  │                      │                     │
  │                      ▼                     ▼
  │                 OpenAI /v1/responses   Provider API
```

---

## Implementation Plan

### Scope

This plan covers **input-side multimodal only**: getting images and files from the client into the LLM's context. Output-side multimodal (image generation, audio) is out of scope.

### Phase 1 — Resolve content blocks in prompt construction

**File:** `runtime/prompt.py`

**Change:** Replace `_extract_text()` with a function that preserves the full content block array for user messages. For each block:

- `input_text` / `output_text` / `text` → pass through unchanged. These are the text block types that `_extract_text()` handles today. The new function must continue to handle all three — `output_text` appears in assistant messages and `text` is a legacy format.
- `input_image` with `file_id` → look up file in `file_store` to get `content_type`, fetch binary from `artifact_store`, base64-encode, replace `file_id` with `image_url` containing a `data:` URI.
- `input_image` with `image_url` → pass through unchanged. The URL is forwarded to the provider. If the provider doesn't support URL-based images (Gemini, Bedrock), reject with an error directing the client to upload via `POST /v1/files`. **Never fetch the URL server-side** (SSRF risk).
- `input_file` with `file_id` → look up file in `file_store`, fetch binary from `artifact_store`, base64-encode, replace `file_id` with `file_data`.
- `input_file` with `file_data` → pass through unchanged.
- `input_file` with `file_url` → same rule as `image_url`: pass through to providers that support it, reject for providers that don't. **Never fetch the URL server-side.**

**Precedence when multiple sources are present:** If a client sends both `file_id` and `image_url` on the same `input_image` block (or `file_id` + `file_data` on `input_file`), resolve in this order: `file_id` takes precedence over `image_url`/`file_data`/`file_url`. Rationale: `file_id` is the most explicit reference — the client uploaded a specific file and is pointing to it. If `file_id` resolution fails, do not fall back to the URL — fail loud (Phase 4 error handling).

**Design decision — resolver as a pre-processing step:** `history_to_input_items()` is synchronous and currently takes only `items: list[ConversationItem]`. Resolving `file_id` references requires I/O (file store lookup + artifact store fetch), which may be async depending on the store implementation. Rather than making `history_to_input_items()` async (which ripples to its caller in `workflow.py:1219`), introduce a separate **`resolve_content_references()`** function that runs before `history_to_input_items()`. This function:

1. Scans all conversation items for content blocks with `file_id` references — **regardless of block type**. This includes `input_image`, `input_file`, and any future types (e.g., `input_audio`). The resolution logic is the same: look up file metadata, fetch binary, base64-encode, inline. The `content_type` comes from the file store, not the block type.
2. Fetches file metadata and binary content from our own stores (batch if possible).
3. Replaces `file_id` references with inline content (base64 data URIs for images, `file_data` for files).
4. Passes through unrecognized block types unchanged (after resolving any `file_id` they carry). The provider decides whether it supports the block type — we are a routing layer, not a content type validator.
5. Returns **copies** of the items with all references resolved. The originals in the conversation store remain unchanged (still contain `file_id` references). This avoids keeping base64 data in memory for the lifetime of the conversation — resolved copies are discarded after each LLM call.

This function **only resolves `file_id` references against our own file store**. It never fetches external URLs (`image_url`, `file_url`) — those are passed through unchanged and forwarded to the provider or rejected downstream if the provider doesn't support them.

The caller in `workflow.py` calls `resolve_content_references()` first, then passes the resolved copies to `history_to_input_items()`. This keeps the prompt builder pure (no I/O) and the resolution step independently testable.

**Output format:** After this phase, `history_to_input_items()` returns messages where `content` is a `list[dict]` (content block array) instead of a `str` when multimodal blocks are present. Downstream layers must handle both.

### Phase 2 — Translate content blocks to Chat Completions format

**File:** `llms/_responses_to_chat.py`

**Change:** `responses_input_to_chat_messages()` currently does `"content": item.get("content")` assuming a string. When `content` is a list, iterate the blocks and convert:

- `input_text` → `{"type": "text", "text": "..."}` (Chat Completions text block).
- `input_image` with `image_url` → `{"type": "image_url", "image_url": {"url": "...", "detail": "..."}}`.
- `input_file` → provider-specific (see Phase 3).

When `content` is a string (text-only messages), pass through unchanged for backward compatibility.

**OpenAI exception:** The OpenAI adapter has a native `responses_create()` path (`adapters/openai.py:377-436`) that bypasses `_responses_to_chat.py` entirely and calls `/v1/responses` directly. This means Phase 2 changes **do not affect OpenAI** — content block arrays are passed through natively. Phase 2 only matters for non-OpenAI providers that go through the Chat Completions translation layer.

**Response-direction functions are unaffected:** `chat_response_to_response()` and `chat_stream_to_response_events()` in the same file assume `content` is a string, but this is correct — LLM output is text-only today. These functions are on the output path (LLM → client), not the input path. If output-side multimodal is added later, these will need changes, but that is out of scope.

### Phase 3 — Provider adapter multimodal support

**Files:** `llms/adapters/*.py`

Each adapter must translate the Chat Completions content block array to the provider's native format:

| Provider | Image format | File/PDF format | Notes |
|---|---|---|---|
| OpenAI | Native Responses API — no adapter changes needed | Native Responses API — no adapter changes needed | Bypasses Chat Completions; content blocks pass through |
| Anthropic | `{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}` | `{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}` | |
| Gemini | `{"inline_data": {"mime_type": "...", "data": "..."}}` | Same inline_data format for PDFs | |
| Bedrock | `{"image": {"format": "png", "source": {"bytes": "..."}}}` | Provider-dependent | |
| Vertex | Inherits Gemini | Inherits Gemini | Subclass of `GeminiAdapter` — gets changes for free |
| Databricks | Inherits OpenAI | Inherits OpenAI | Subclass of `OpenAICompatibleAdapter` |

**URL rejection for non-supporting providers:** When an adapter receives a content block with `image_url` or `file_url` and the provider doesn't support URL-based input (Gemini, Bedrock), the adapter must raise a clear error: e.g., "Provider X does not support URL-based images. Upload the file via POST /v1/files and reference it by file_id." This check lives in the adapter because only the adapter knows the provider's capabilities.

**Incremental approach:** OpenAI needs no adapter changes (native Responses API path). Start with Anthropic or Gemini. Vertex and Databricks inherit from their parents — no separate work needed. Each adapter change is independently testable.

### Phase 4 — Validation and resilience

After the core path works:

1. **`file_id` validation at request time** — When `POST /v1/responses` receives an `input_image` or `input_file` with a `file_id`, check that the file exists in the file store before persisting the conversation item. Return 400 if not found. This prevents silent failures at prompt construction time.
2. **Content caching** — Fetching and base64-encoding the same file on every prompt rebuild (every agent loop iteration in a multi-turn conversation) is wasteful. Add an in-memory cache scoped to the workflow run, keyed by `file_id`.
3. **Error handling** — If a `file_id` reference cannot be resolved at prompt construction time (file deleted between request and agent loop), emit a clear error rather than silently dropping the block.

### Ordering and dependencies

```
Phase 1 (prompt.py)
  │
  ▼
Phase 2 (_responses_to_chat.py)  ── depends on Phase 1 output format
  │
  ▼
Phase 3 (adapters)               ── depends on Phase 2 output format
  │                                  can be done one adapter at a time
  ▼
Phase 4 (validation/caching)     ── independent, can start after Phase 1
```

Phases 1–3 are sequential — each layer's output feeds the next. Phase 4 is independent and can be done in parallel with Phase 3.

---

## Not Yet

These are real concerns but do not block Phases 1–3. Each can be addressed independently after the core multimodal path works.

- **Token counting for multimodal content**: Images consume tokens that vary by resolution and `detail` level (e.g., OpenAI charges ~85 tokens for a low-detail image, up to ~1105 for high-detail). Today we don't do client-side token counting for text either — we send what we have and let the provider reject if it exceeds the limit. Images don't change that dynamic. If we add client-side token budgeting later, it must account for multimodal content.
- **Compaction of multimodal turns**: The compaction system (`COMPACTION.md`) only handles text. If a compaction summary replaces a turn that contained images, those images are lost. The simplest correct behavior: preserve multimodal blocks as-is during compaction and only compact text portions. This needs a design decision but doesn't block the inference path.
- **Multi-turn payload bloat**: In a 10-turn conversation where turn 1 had a 5MB image, that image is base64-encoded (~6.7MB) and re-sent in every subsequent LLM call. Provider APIs are stateless — there's no way to say "use the image I sent last time." This means multimodal conversations grow faster than text-only ones. Mitigation options: drop images from older turns after N iterations, let compaction handle it, or accept the cost. Not a blocker for initial implementation but will matter for long conversations with large files.
- **Output-side multimodal**: Image generation, audio output, etc. The current output types are text and function calls only.
- **Streaming multimodal**: Streaming image tokens or progressive rendering.
- **File search / RAG**: Using uploaded files as a retrieval corpus rather than inline context. This is a separate tool, not part of the inference path.
- **Content extraction fallback**: For providers that can't handle raw PDFs, extracting text and passing it as `input_text` instead. Requires a text extraction pipeline.
- **Per-provider capability detection**: Automatically routing multimodal content based on what the target provider supports. Currently the caller must know their model supports vision.
