# Compaction Design

## Problem

The agent loop sends the full conversation history with every LLM call. Two growth dimensions will eventually break this:

1. **Within a single execution:** The in-memory `history` list grows each iteration (LLM response + tool calls + tool results). Long-horizon tasks — hundreds of iterations, verbose tool output — will exceed the model's context window and the LLM call will fail or degrade.

2. **Across turns:** Each new turn loads the full conversation via `search_items(conversation_id)`. A 500-turn conversation will have thousands of items. Even if each execution manages its own context, the initial history load becomes the bottleneck.

Both problems are identified in RUNTIME.md § "Context management for long-horizon tasks."

---

## Survey of Existing Approaches

### Truncation (drop oldest messages)

**Who does it:** LangChain `trim_messages`, OpenAI Agents SDK `truncation: "auto"`, AutoGen `MessageHistoryLimiter`, Semantic Kernel `ChatHistoryTruncationReducer`.

**How it works:** Keep the most recent N messages (or N tokens). Oldest messages are discarded. Some implementations preserve the system message regardless. OpenAI's variant keeps the first item (system/developer message) plus the last N items that fit within `max_input_tokens`.

**LangChain specifics:** `trim_messages(messages, max_tokens=N, strategy="last", token_counter=model, include_system=True)`. Supports `start_on` / `end_on` constraints (e.g. always start on a human message). `allow_partial=True` lets it split a single message to fit the budget. `strategy="first"` keeps oldest instead.

**Semantic Kernel specifics:** `ChatHistoryTruncationReducer(target_count, threshold_count)`. The `threshold_count` is a buffer — reduction triggers when history exceeds `target_count + threshold_count`, leaving room for recent tool call pairs to avoid orphaning.

**Trade-offs:**
- Simple, fast, no LLM cost.
- Loses context permanently — the agent forgets early instructions, prior decisions, and context that informed current state.
- Tool call pair orphaning: truncating a `function_call` without its `function_call_output` (or vice versa) produces an invalid message sequence that most LLMs reject. Only Semantic Kernel handles this explicitly (via threshold buffer). LangChain documents this as a known issue (#29637).

### Summarization (LLM-generated summary replaces old messages)

**Who does it:** LangGraph (summarization node pattern), Letta/MemGPT (recursive summarization), CrewAI (`respect_context_window` auto-summarize), Semantic Kernel `ChatHistorySummarizationReducer`, Anthropic (extended thinking compaction).

**How it works:** When history exceeds a threshold, an LLM call summarizes the oldest messages into a compact summary message. The original messages are replaced by the summary. Some implementations retain the summary as a system message; others as a user or assistant message.

**LangGraph specifics:** Implemented as a graph node, not built-in middleware. The pattern: check if history exceeds a threshold → call LLM with "summarize this conversation" prompt → emit a `RemoveMessage` for each summarized message → store the summary as a new message. The developer controls the trigger condition, summary prompt, and storage.

**Letta/MemGPT specifics:** Recursive summarization — when context exceeds ~70% of the limit, the agent itself triggers `conversation_search` and `archival_memory_insert` to move facts to long-term storage, then summarizes the evicted portion. At ~100%, forced eviction summarizes and removes the oldest FIFO messages. Summaries are recursive: a summary of a summary is possible for very long conversations.

**Semantic Kernel specifics:** `ChatHistorySummarizationReducer(service, target_count, threshold_count)`. Uses the same `threshold_count` buffer as truncation to protect tool call pairs.

**Trade-offs:**
- Preserves semantic content — the agent retains awareness of prior decisions and context.
- Costs an extra LLM call (latency + tokens). For long histories, the summarization call itself may be expensive.
- Lossy — the summary is the LLM's interpretation, not the original content. Nuance, exact values, and specific instructions can be lost.
- Summary quality degrades recursively (summary of a summary of a summary).

### Hybrid (recent messages raw + older messages summarized)

**Who does it:** Letta/MemGPT (FIFO queue + recursive summary), LangChain legacy `ConversationSummaryBufferMemory`, Anthropic (layered compaction).

**How it works:** A sliding window of recent messages is kept verbatim. Everything older than the window is summarized. The prompt contains: system message + summary of old context + recent raw messages.

**Anthropic specifics:** Layered approach applied in order: (1) clear tool result contents (keep call/result structure, replace bodies with "[content cleared]"), (2) clear thinking blocks, (3) summarize oldest messages, (4) if still too long, create structured notes instead of a summary. Each layer is tried before the next, minimizing information loss.

**Trade-offs:**
- Best of both: recent context is exact, older context is preserved in summary form.
- More complex to implement — need to manage the boundary between raw and summarized.
- Still requires an LLM call for summarization.

### Compression (token-level reduction)

**Who does it:** AutoGen `TextMessageCompressor` (via LLMLingua).

**How it works:** A specialized compression model (not the main LLM) removes redundant tokens from messages while preserving semantic meaning. Works at the token level, not the message level.

**Trade-offs:**
- Preserves more information than summarization (token-level, not semantic-level).
- Requires an additional model/library (LLMLingua).
- Less mature, less predictable behavior.
- Not widely adopted — only AutoGen offers this.

### Opaque compaction (provider-managed)

**Who does it:** OpenAI Agents SDK.

**How it works:** The provider returns an encrypted, opaque "compaction item" that replaces part of the history. The client stores and re-sends it, but cannot read or modify it. The provider decrypts and expands it server-side.

**Trade-offs:**
- Zero implementation cost for the application.
- Vendor lock-in — the compaction item only works with that provider's API.
- No transparency — the application cannot inspect what was summarized or how.
- Not applicable to agent-plane: we target multiple LLM providers and need provider-agnostic compaction.

### Retrieval-based (memory as a tool)

**Who does it:** Letta/MemGPT (archival memory), Haystack (memory-as-tool via RAG).

**How it works:** Instead of keeping all history in the prompt, older context is stored in a searchable store (vector DB, keyword index). The agent has a tool that searches this store when it needs historical context. The prompt contains only recent messages plus whatever the agent explicitly retrieves.

**Trade-offs:**
- Scales to unlimited history — storage cost is decoupled from prompt length.
- The agent must know when to search (and what to search for). If it doesn't search, it has no context.
- Adds round trips (tool call → search → tool result → LLM call).
- Complex infrastructure (vector store, embeddings, indexing).
- Out of scope for MVP — requires memory store infrastructure that doesn't exist yet.

### ChatGPT (product-level context management)

**Who does it:** OpenAI's ChatGPT product.

**How it works:** Sliding-window truncation — when a conversation exceeds the model's context window (128K tokens for GPT-4o), oldest messages are dropped from the front. No summarization of dropped context. Tool results (Code Interpreter output, browsing results) are included as regular messages and subject to the same truncation. Large Code Interpreter outputs are pre-truncated (last N lines of stdout) before inclusion.

**Memory feature (separate from compaction):** ChatGPT has a persistent memory layer (launched Feb 2024) that stores short factual statements ("User prefers Python", "User works at Acme Corp") across conversations. Memory entries are injected into the system prompt at conversation start — they are NOT conversation summaries. Memory is cross-session but limited (~100-200 entries). It does not replace or supplement truncated conversation history — if old messages are truncated, they are gone.

**Trade-offs:**
- Simple, no extra LLM cost for compaction.
- Context loss is abrupt — no graceful degradation. The agent forgets everything beyond the window.
- Memory feature provides cross-session persistence of key facts but is coarse-grained (individual facts, not conversation context).
- No special handling for tool call pair integrity during truncation — no public documentation of orphan prevention.
- Opaque — no user control over what gets truncated within a conversation.

### Claude Code (layered compaction with surgical clearing)

**Who does it:** Anthropic's Claude Code CLI agent.

**How it works:** Two-phase compaction triggered when context approaches the limit. Phase 1: clear older tool outputs (file read results, bash command output) while preserving conversation structure. Phase 2: if still over budget, generate an LLM summary of the conversation history. The summary preserves: user requests, key code snippets, code patterns, file states, and key decisions. What gets lost: detailed instructions from early in the conversation, verbose tool outputs, redundant messages.

**Anthropic Compaction API (server-side):** The underlying `compact-2026-01-12` beta API triggers at a configurable threshold (default 150K input tokens, minimum 50K). The API generates a `compaction` block containing a summary. All messages prior to the compaction block are dropped on subsequent requests. Default summarization prompt asks for state, next steps, and learnings wrapped in `<summary>` tags. Custom summarization instructions can replace the default prompt entirely. A `pause_after_compaction` option lets the caller inject content after summary generation but before the response continues.

**CLAUDE.md survives compaction:** After compaction, Claude Code re-reads CLAUDE.md files from disk and re-injects them fresh. This means persistent instructions survive compaction even though they were part of the original context.

**Subagents for context isolation:** Claude Code recommends subagents for verbose operations — each subagent runs in its own context window and returns only a summary to the main conversation, keeping the primary context clean. This is a structural approach to context management that complements compaction.

**Claude.ai (web product) is different:** Uses FIFO truncation (oldest messages dropped), similar to ChatGPT. No summarization.

**Trade-offs:**
- Layered approach minimizes information loss — surgical clearing before summarization.
- CLAUDE.md re-injection ensures persistent instructions survive compaction.
- Subagent delegation is an architectural pattern for managing context, not just a compaction strategy — prevents large tool results from entering the main context at all.
- Custom summarization instructions give the user control over what gets preserved.
- Summarization still costs an LLM call and is lossy — detailed instructions given only in conversation (not in CLAUDE.md) can be lost.
- The compaction API is provider-specific (Anthropic models only).

### Surgical clearing (targeted content removal)

**Who does it:** Anthropic (as a technique within Claude Code's layered approach).

**How it works:** Instead of removing entire messages, clear specific content within messages. Primary target: tool result bodies, which are often the largest content (file contents, command output, search results). Replace the body with a marker like `[content cleared]` while preserving the call/result structure. This maintains valid message sequences (no orphaned tool calls) while recovering significant token budget.

**Trade-offs:**
- High token savings with minimal information loss — the LLM retains that a tool was called with specific arguments and that it returned a result, just not the full result body.
- Preserves message sequence validity — no tool call pair orphaning.
- Simple to implement — string replacement, no LLM call.
- The cleared content is gone — if the agent needs it later, it must re-call the tool.

---

## Design Constraints from agent-plane

1. **DBOS durability:** The agent loop runs inside a `@workflow`. LLM calls and tool calls are `@step` functions whose outputs are checkpointed. Compaction must work correctly with DBOS replay — a recovered workflow must produce the same compacted history as the original.

2. **Steering:** Messages can arrive via `try_deliver` at any point during execution. Compaction must not summarize away a steered message that hasn't been processed yet. The steering inbox is checked each iteration via `_sync_history`.

3. **Conversation store is the source of truth:** Output items are persisted to the conversation store at the end of execution. The full history is always recoverable from the store — compaction summaries are additive (appended as new items), never destructive (original items are never deleted or modified).

4. **Multi-provider:** Compaction must work with any LLM provider (OpenAI, Anthropic, Gemini, etc. via our multi-provider `llms.Client`). Provider-specific compaction (OpenAI opaque items) is not usable.

5. **Tool call pair integrity:** The compacted message sequence must never contain a `function_call` without its `function_call_output` or vice versa. LLMs reject orphaned tool call pairs.

6. **Per-agent configurability:** Different agents have different context needs. A code agent with verbose tool output needs aggressive compaction; a simple Q&A agent may need none. Compaction strategy should be configurable in the agent spec.

---

## Recommended Approach: Layered Compaction

A layered strategy, applied in order from least-lossy to most-lossy — directly informed by Claude Code's proven two-phase approach (surgical clearing → summarization) and Anthropic's context engineering guidance. Each layer is tried before the next. Compaction triggers when the estimated token count of the prompt exceeds a configurable threshold (e.g. 80% of the model's context window).

### Layer 1: Surgical clearing of large content

**When:** Token count exceeds threshold.

**What:** Two sub-operations, applied to messages older than the recent window:

1. **Tool result bodies:** Replace `function_call_output` bodies with `[Previous tool result cleared — re-call tool if needed]`. Preserve the `function_call` (name + arguments) and the `function_call_output` (with cleared body). This maintains the conversation structure and lets the LLM know what tools were called and when.

2. **Binary content blocks:** In any message type (user, assistant), replace the payload of image/file content blocks with a marker that preserves the file ID:

   ```
   # Before
   {"type": "image", "file_id": "file_abc123", "data": "<base64 ...>"}

   # After
   {"type": "image", "file_id": "file_abc123", "data": "[binary content removed for context management — use file_id to retrieve]"}
   ```

   Text content blocks within the same message are left untouched. The structure (block type, file ID, position in the message) is preserved — only the binary payload is stripped. This recovers significant token budget (a single base64 image can consume 25K+ tokens) while keeping a reference the agent can use to re-fetch the content if needed.

**Why first:** Tool results and binary content blocks are typically the largest items (file contents, command output, base64 images). Clearing them recovers significant token budget with minimal semantic loss — the LLM retains the fact that content was present and has the file ID to retrieve it.

**Protect:** Messages within the recent window (last N iterations) are never cleared — tool results, images, and files all stay intact. Steered messages are never cleared.

**Definition of "iteration":** One iteration = one LLM call and everything it produces. Concretely: one assistant message (or one set of function calls + their function call outputs) + any steered user messages that arrived during that iteration. The recent window of N iterations means: find the Nth-most-recent LLM response in the history and protect everything from that point onward. This is determined by counting backward through items of type `message` with `role=assistant` or `function_call` (both indicate an LLM response boundary).

### Layer 2: Summarization of old messages

**When:** Layer 1 was not sufficient — token count still exceeds threshold after clearing tool results.

**What:** Summarize all messages older than the recent window into a single summary message. The summary is generated by an LLM call (using the agent's same model). The prompt becomes: system message + summary + recent raw messages.

**Multimodal content in summarization input:** By the time Layer 2 runs, Layer 1 has already cleared binary content blocks (images, files) from messages outside the recent window. The summarization LLM receives text content only — it sees the file ID markers (e.g. `[binary content removed for context management — use file_id to retrieve]`) and, critically, the assistant messages that originally described what was in those images/files ("I can see a dashboard showing revenue of $1.2M..."). The summary captures the descriptions, not the raw binary data. The summary itself is always text-only.

**Summarization prompt:** The `summarize_history` step sends the messages to be summarized as conversation context, with a system prompt instructing the LLM to produce a continuation-oriented summary. The prompt draws on three patterns: LangChain's progressive summarization ("add onto the previous summary"), Semantic Kernel's grounding constraint ("do not incorporate other general knowledge"), and a task-continuation framing (the summary exists so the agent can pick up where it left off).

```
Summarize the conversation above so that a future assistant can continue
the work without access to the original messages.

Include: the user's goals, key decisions and why they were made, tool
results that matter going forward (paths, values, errors), and any
outstanding commitments or next steps.

Exclude: verbose tool output, redundant exchanges, and intermediate
reasoning that led to a final decision — keep the decision, not the path.

Do not incorporate knowledge from outside this conversation. Do not
invent facts. Write in plain text with no markup.
```

For recursive summarization (when the conversation begins with a prior summary), the prompt prepends: `"The conversation starts with a summary of earlier context. Incorporate it into your new summary — do not discard it."` This is LangChain's progressive summarization pattern: each summary builds on the last rather than replacing it.

**Implementation:** This is itself a `@step` function, so the summary is checkpointed by DBOS. On recovery, the cached summary is reused — no re-summarization.

**Protect:** The recent window (last N iterations) is never summarized. Steered messages within the recent window are preserved verbatim.

**Summary input overflow:** If the history to be summarized is itself too large for the model's context window (using the cached window size discovered from a prior overflow error, or estimated via tiktoken), apply Layer 1 (surgical clearing) to the summarization input first — clear tool result bodies before passing to the LLM. If still too large, truncate the oldest messages from the summarization input (the summary will be partial, covering only the most recent portion of the old history). The resulting summary is still better than no summary — it captures the most recent context that would otherwise be lost.

**Summarization failure:** If the `summarize_history` LLM call fails (model down, rate limit, timeout), fall back to Layer 3 (truncation) for this iteration rather than failing the workflow. The agent continues with a truncated context. No compaction item is persisted (there is no summary to persist). The next iteration that exceeds the threshold will retry summarization. This is logged as a warning.

### Layer 3: Truncation (emergency fallback)

**When:** Layers 1 and 2 were not sufficient — the recent window alone exceeds the threshold (unlikely but possible with very large tool results in recent messages).

**What:** Truncate the oldest messages in the recent window, preserving tool call pair integrity. Use Semantic Kernel's approach: maintain a buffer zone where tool call pairs are kept together.

**Why last:** Truncation loses information permanently with no summary. It's the emergency valve.

---

## Configuration

Compaction parameters in the agent spec (`agent.yaml`):

```yaml
llm:
  model: openai/gpt-4o
  max_completion_tokens: 16384

compaction:
  # Trigger threshold as fraction of model's context window (default: 0.8)
  trigger_threshold: 0.8

  # Number of recent iterations to protect from compaction (default: 5)
  recent_window: 5
```

When `compaction` is omitted, defaults apply (0.8 threshold, 5-iteration window). Summarization uses the agent's main `llm.model`.

---

## Token Counting and Context Window Discovery

Compaction requires two pieces of information: (1) how many tokens the current prompt will consume, and (2) the model's context window size. Rather than trying to know the context window upfront (hardcoded registry, provider APIs), we discover it reactively from the first overflow error and validate it with a local token estimate.

### Approach: reactive discovery + tiktoken validation

**No proactive threshold check on the first overflow.** We do not maintain a model registry or attempt to predict the context window before the first LLM call. Instead:

1. **Call the LLM.** If it succeeds, no compaction is needed.

2. **If the call fails with a context overflow error**, parse the error message to extract:
   - The model's max context window (e.g. `128000`)
   - The actual token count the provider saw (e.g. `142000`)

   All major providers include these values in their error messages:
   - **OpenAI**: `"This model's maximum context length is 128000 tokens. However, you requested 142000 tokens"`
   - **Anthropic**: `"197202 + 21333 > 200000"`
   - **Gemini**: `"exceeds the maximum number of tokens allowed (1048576)"`

3. **Validate with tiktoken before compacting.** Use tiktoken to estimate the prompt's token count locally. Compare our estimate against the token count reported in the error message. If they are in the same ballpark (within ~30%), we have high confidence this is a genuine context overflow. If the estimates diverge wildly, the error may be something else misidentified as an overflow — log a warning and propagate the original error rather than entering a pointless compact-retry loop.

4. **Compact and retry.** Apply the layered compaction strategy (Layer 1 → Layer 2 → Layer 3) targeting the discovered context window, then retry the LLM call.

5. **Cache the discovered window size in-memory** for the rest of this execution. Subsequent iterations use tiktoken + the cached window to check proactively (using the `trigger_threshold` fraction), avoiding further overflow errors. The cache is per-execution — it does not persist across executions.

### Why this approach

- **No registry to maintain.** No hardcoded dict of model context windows that goes stale when providers launch new models or change limits.
- **Works with any model.** Custom fine-tunes, self-hosted models, new providers — all work automatically.
- **One wasted API call per execution, at most.** After the first overflow, the cached window enables proactive checks for all subsequent iterations.
- **tiktoken as sanity check, not oracle.** tiktoken is ~85-95% accurate for non-OpenAI models. We don't rely on it for threshold decisions — we use it only to validate that the error message's token count is plausible. Add `tiktoken>=0.7` to `pyproject.toml`.
- **CrewAI validates this pattern.** CrewAI uses reactive catch-compact-retry in production.

### Error classification: `ContextWindowExceededError`

Context overflow errors are HTTP 400 — currently classified as `PermanentLLMError` by `_classify_error()` in `llms/client.py`. But unlike other permanent errors (auth failure, malformed request), a context overflow is recoverable after compaction. A new subclass carries the parsed token counts so the workflow can compact and retry without touching raw error strings.

```python
# llms/errors.py

class ContextWindowExceededError(PermanentLLMError):
    """
    The LLM rejected the request because the prompt exceeded
    the model's context window. Carries the parsed token
    counts so the caller can compact and retry.

    Subclass of ``PermanentLLMError`` so existing catch blocks
    that handle ``PermanentLLMError`` still work — if the
    workflow does not specifically catch this subclass, the
    error propagates as fatal (safe default).

    :param max_context_tokens: The model's context window size
        as reported by the provider, e.g. ``128000``.
    :param actual_tokens: The token count the provider measured
        for the rejected request, e.g. ``142000``.
    """
    max_context_tokens: int
    actual_tokens: int
```

**Where classification happens:** In `_classify_error()` within `llms/client.py`, not in the workflow. The existing flow is:

```
HTTP 400 from provider
  → _classify_error() inspects status + response body
  → if response body matches a context overflow pattern:
      parse max_context_tokens and actual_tokens from message
      raise ContextWindowExceededError(
          code=str(status),
          detail=LLMErrorDetail(...),
          max_context_tokens=max_context_tokens,
          actual_tokens=actual_tokens,
      )
  → otherwise:
      raise PermanentLLMError (existing behavior)
```

The workflow catches `ContextWindowExceededError` specifically, compacts, retries. All other `PermanentLLMError` subclasses remain fatal.

**Provider-specific patterns** matched by `_classify_error()`:

| Provider | Match condition | Extraction |
|----------|----------------|------------|
| **OpenAI** | status=400, `error.code == "context_length_exceeded"` | `"maximum context length is {max} tokens"`, `"you requested {actual} tokens"` |
| **Anthropic** | status=400, message matches `"{input} + {max_tokens} > {limit}"` or `"prompt is too long: {actual} tokens > {limit} maximum"` | Integers from the inequality or the `>` comparison |
| **Gemini** | status=400, message matches `"input token count ({actual}) exceeds the maximum number of tokens allowed ({limit})"` | Parenthesized integers |

The patterns match conservatively — only well-known error shapes trigger `ContextWindowExceededError`. Unknown 400 errors remain `PermanentLLMError`. If a provider changes their error format, the worst case is that we fail to detect the overflow and propagate the error as-is (safe default, not silent misbehavior).

**tiktoken validation** happens in the workflow after catching `ContextWindowExceededError`, before compacting. The workflow compares its local tiktoken estimate against `error.actual_tokens`. If they diverge by more than ~30%, the error may be misclassified — log a warning and re-raise as `PermanentLLMError` rather than entering a compact-retry loop.

### Interaction with `trigger_threshold`

After the first overflow discovers the context window, subsequent iterations check proactively:

```
tiktoken_estimate = tiktoken.count(messages)
budget = cached_context_window * trigger_threshold  # e.g. 128000 * 0.8 = 102400
if tiktoken_estimate > budget:
    compact(messages)
```

The `trigger_threshold` (default 0.8) provides headroom so compaction fires before hitting the hard limit. This is conservative — tiktoken may under-estimate by 10%, but the 20% margin absorbs that.

**Multimodal token estimation:** tiktoken only counts text tokens. Image and file content blocks consume tokens according to provider-specific rules (e.g. OpenAI charges based on image resolution), but tiktoken has no way to estimate these. This means tiktoken will under-count when images are present. The 20% headroom from `trigger_threshold` absorbs moderate image content. For image-heavy conversations, the reactive path (catch `ContextWindowExceededError` from the provider, which counts images correctly) serves as the safety net. After Layer 1 clears binary content blocks outside the recent window, subsequent tiktoken estimates become more accurate because the cleared blocks are text markers, not base64 payloads.

---

## Persisted Compaction Items

### The problem with purely in-memory compaction

If compaction is only an in-memory transformation, every new execution must load the **full** conversation history from the store to reconstruct context. For a 500-turn conversation, this means loading thousands of items into memory just to summarize them down. The compaction work is thrown away at the end of each execution and repeated on the next turn.

The fix: persist the compaction summary as a conversation item. On the next execution, `load_history` loads only the summary + items after it, skipping everything the summary covers. The original items remain in the store for the user to browse via pagination — they are never deleted or modified.

### New item type: `compaction`

A new conversation item type alongside the existing `message`, `function_call`, `function_call_output`, and `reasoning` types.

```python
# entities/conversation.py

class CompactionData(BaseModel):
    """
    Data payload for a compaction summary item.

    :param summary: The LLM-generated summary text covering
        all conversation items from the start of the conversation
        (or the previous compaction item) up through the item
        identified by ``last_item_id``.
    :param last_item_id: The item ID (inclusive) of the last
        conversation item covered by this summary, e.g.
        ``"msg_abc123"``. Items at positions <= this item's
        position are summarized and do not need to be loaded
        for prompt construction. Used as the cursor for
        ``load_history`` — items after this ID are loaded
        as recent raw context.
    :param model: The model used to generate the summary,
        e.g. ``"openai/gpt-4o-mini"``.
    :param token_count: Approximate token count of the summary
        text, for budget tracking.
    """
    summary: str
    last_item_id: str
    model: str
    token_count: int
```

**Item ID prefix:** `"cmp_"` (e.g. `"cmp_a1b2c3"`).

**Appended like any other item:** The compaction item is appended to the conversation via `conv_store.append(conversation_id, [new_compaction_item])`. It gets a `position`, `created_at`, and `id` like every other item. It is part of the conversation's linear item sequence.

**Example conversation item sequence after compaction:**

```
position  id          type                  content
────────  ──────────  ────────────────────  ────────────────────────────────────
0         msg_001     message (user)        "Analyze this dataset"
1         msg_002     message (assistant)   "I'll start by loading the file..."
2         fc_003      function_call         grep(pattern="revenue", path="data.csv")
3         fco_004     function_call_output  "row1: revenue=100\nrow2: revenue=..."
4         msg_005     message (assistant)   "Found 50 revenue entries. Let me..."
...
247       fco_248     function_call_output  "Chart saved to output/chart.png"
248       msg_249     message (assistant)   "Here's the trend analysis..."
249       cmp_250     compaction            summary="User asked to analyze a dataset.
                                            Agent loaded data.csv, found 50 revenue
                                            entries, computed statistics, generated
                                            a trend chart..." last_item_id="msg_249"
250       msg_251     message (user)        "Now compare it with last quarter"
251       msg_252     message (assistant)   "Loading last quarter's data..."
...
```

Note: the compaction item's `last_item_id` points to `msg_249` (position 248), not to the compaction item itself (position 249). This is critical — `load_history` uses `last_item_id` as the cursor, loading items after position 248. This ensures items appended between the summary generation point and the compaction item's own position are not lost. See "Crash Safety" section for details.

### When compaction items are produced

Compaction items are produced at the **end of an execution** when Layer 2 (summarization) was triggered during the run. The workflow already has the summary in memory (generated by the `summarize_history` step) — persisting it is a single `append` call before the workflow returns.

**Precise trigger:** At the end of `_run_agent_loop`, after persisting output items and before returning:

```
if layer_2_summary was generated during this execution:
    compaction_item = NewConversationItem(
        type="compaction",
        response_id=task_id,
        data=CompactionData(
            summary=layer_2_summary.text,
            last_item_id=layer_2_summary.last_item_id,
            model=layer_2_summary.model,
            token_count=layer_2_summary.token_count,
        ),
    )
    conv_store.append(conversation_id, [compaction_item])
```

`last_item_id` is the ID of the last history item that was included in the summary input. This is captured at the time `summarize_history` runs — the last item in the history list at that moment. Note: in recursive compaction scenarios, the first item in the history list is a synthetic summary message whose ID is the previous compaction item's ID (a `cmp_`-prefixed ID). If Layer 2 triggers again during this execution, `last_item_id` will point to a content item (not the synthetic summary), because the summary is at position 0 in history and the summary covers items *older* than the recent window — the last item in the summarization input is always a real content item from the recent window boundary.

**Why at the end, not mid-execution:** Within a single execution, the in-memory `history` list is always available — there is no need to re-load from the store. The compaction item benefits the *next* execution, not the current one. Appending at the end also avoids mid-execution store writes that could complicate DBOS replay.

**Only Layer 2 produces compaction items.** Layer 1 (surgical clearing) is stateless — it clears tool result bodies in-memory each iteration based on the recent window. There is nothing to persist. Layer 3 (truncation) is an emergency fallback that drops messages in-memory. Only the summarization step produces a reusable artifact worth persisting.

### How `load_history` works with compaction items

Today, `_run_agent_loop` loads history with:

```python
# workflow.py line 1170 — current implementation
history = fetch_all_items(conv_store, conversation_id)
```

This loads every item from position 0. With compaction items, the load changes to:

**Step 1: Find the latest compaction item.**

Extend the existing `list_items` method with an optional `type` filter parameter. The SQL change is a single `WHERE type = :type` clause on the existing query. No new method needed.

```python
page = conv_store.list_items(
    conversation_id,
    type="compaction",
    order="desc",
    limit=1,
)
compaction_item = page.data[0] if page.data else None
```

**Step 2: Load history starting from the summary's coverage boundary.**

```python
# Updated load in _run_agent_loop
compaction_item = find_latest_compaction(conv_store, conversation_id)

if compaction_item is not None:
    # Load items AFTER the last item the summary covers —
    # NOT after the compaction item itself. The compaction item
    # may have been appended AFTER additional output items that
    # the summary does not cover (because the summary was
    # generated mid-execution, then more iterations ran before
    # the compaction item was persisted at the end).
    up_to_id = compaction_item.data.last_item_id
    recent_items = fetch_all_items(
        conv_store,
        conversation_id,
        after=up_to_id,
    )
    # Filter out compaction items — they are metadata, not
    # conversation content the LLM should see
    recent_items = [
        item for item in recent_items
        if item.type != "compaction"
    ]
    # Build history: synthetic summary pair + recent items
    summary_pair = compaction_to_history_items(compaction_item)
    history = summary_pair + recent_items
else:
    # No compaction yet — load everything (existing behavior)
    history = fetch_all_items(conv_store, conversation_id)
```

**Why `last_item_id`, not the compaction item's own ID:** The summary is generated mid-execution (when Layer 2 triggers). The execution may continue for several more iterations after that, appending output items to the conversation store. The compaction item is appended last, after all output items. If we used the compaction item's own ID as the cursor, items between the summary boundary and the compaction item would be skipped — neither in the summary nor loaded as recent items. Using `last_item_id` ensures those post-summary items are loaded.

**Step 3: Convert compaction item to a history entry the LLM can consume.**

The compaction item's summary becomes a message at the front of the history. The prompt builder sees: system message + summary message + recent raw items.

```python
def compaction_to_history_items(
    compaction_item: ConversationItem,
) -> list[ConversationItem]:
    """
    Convert a compaction item into a synthetic user+assistant
    message pair that the prompt builder can include as
    conversation context.

    The pair preserves natural turn-taking structure:
    a synthetic user message asks for a summary, and a
    synthetic assistant message provides it. This avoids
    attribution confusion — the LLM knows it produced a
    summary (not a real prior response), and the user
    message framing makes the context explicit.

    :param compaction_item: The compaction item from the store,
        e.g. with ``type="compaction"`` and
        ``data.summary="User asked to analyze..."``.
    :returns: Two ``ConversationItem`` instances: a
        ``role=user`` message requesting the summary and a
        ``role=assistant`` message containing it.
    """
```

The synthetic pair:

```
{"role": "user", "content": "[This is an automatically generated summary of the prior conversation context. The original messages are available but not included in this prompt for brevity.]\n\nPlease provide a summary of our conversation so far."}

{"role": "assistant", "content": "{summary text}"}
```

**Why a synthetic pair, not a single message:** There is no industry consensus on the role for summary messages. LangChain and LangGraph use `system`. Letta/MemGPT and CrewAI use `user`. Semantic Kernel and Anthropic use `assistant`. OpenAI Agents SDK uses a synthetic `user`+`assistant` pair.

We use the pair approach (matching OpenAI Agents SDK) because:
- A bare `assistant` message makes the LLM think it spontaneously said the summary in a prior turn, leading to "As I mentioned earlier..." artifacts. The pair avoids this — the LLM knows it was asked to summarize and it did, so it would say "As I summarized earlier..." which is accurate.
- A bare `user` message is a fiction — the user never wrote the summary. It can also confuse attribution when the summary describes what the assistant did ("I analyzed the data...").
- `system` is cleanest semantically but not all providers support system messages mid-conversation. Anthropic, Google Gemini, and AWS Bedrock only support system messages as a top-level parameter — our LLM client would need to extract mid-conversation system messages and merge them into the top-level system prompt, losing positional context entirely.
- The pair preserves natural turn-taking structure that all providers and models expect. It costs one extra message (~20 tokens for the synthetic user prompt) which is negligible.
- OpenAI chose this approach for their production Agents SDK, suggesting it was tested at scale.

### Recursive compaction (summary of a summary)

When a conversation grows long enough that a second compaction is needed, the new summary covers everything from the previous compaction item's summary through the items that follow it. The new compaction item's `last_item_id` will point to a later item than the previous one's.

On load, only the **latest** compaction item matters — `find_latest_compaction` returns the most recent one, and `fetch_all_items(after=compaction_item.data.last_item_id)` loads only items after the summary boundary. Previous compaction items are still in the conversation (they are never deleted), but they are filtered out during load (type != "compaction") and are simply part of the historical items that the latest summary covers.

The summary prompt for recursive compaction includes the previous summary as input:

```
Previous summary: {previous_compaction.data.summary}

New messages since previous summary:
{items between previous compaction and current compaction point}

Write an updated summary covering all of the above.
```

This is Letta/MemGPT's recursive summarization pattern. Summary quality degrades with each recursion, but each recursion covers a larger span, so the frequency of recursion decreases over time.

### What the user sees vs. what the LLM sees

| Viewer | What they see |
|--------|---------------|
| **User (API client)** | Full conversation history via `list_items` with pagination. All original items at positions 0–N are present and browsable. Compaction items appear in the sequence as a visible record that compaction occurred. |
| **LLM (prompt)** | Summary message (from latest compaction item) + recent raw items after the compaction point. The LLM never sees the original items that were summarized. |

The conversation store never deletes or modifies original items. Compaction items are additive — they are new items appended to the conversation. The user can always scroll through the full history. The LLM gets a bounded, compacted view.

### Storage growth

The conversation store grows unboundedly — every item is kept forever. This matches the industry standard (ChatGPT, Claude.ai, Semantic Kernel, AutoGen all keep everything). Storage cost grows linearly with conversation length, but:

- Individual item reads are O(1) by ID
- Paginated listing is O(page_size) regardless of total conversation length
- `load_history` is O(items_after_compaction), not O(total_items) — bounded by the compaction point
- Database size is a storage cost problem, not a performance problem

If storage size becomes a concern in the future, a TTL-based cleanup job (like Google Gemini's 18-month auto-delete or LangGraph's checkpoint TTL) can delete items older than a threshold. This is independent of compaction and does not require design changes.

---

## Interaction with DBOS

Compaction is **not** applied on every iteration. It triggers only when the estimated token count of the prompt exceeds the configured threshold. Most iterations — especially early ones — skip compaction entirely and call the LLM with the full, unmodified history.

```
history (full) → build messages → estimate tokens (system + tools + messages)
                                    │
                          under threshold? → call_llm(messages)
                                    │
                          over threshold? → compact(messages) → call_llm(compacted)
```

The token estimate includes the **full prompt**: system instructions + tool schemas + conversation messages. An agent with many tools may consume a large fraction of the context window before any history is added. The effective history budget is `(threshold × context_window) - system_tokens - tool_schema_tokens`. `compact()` receives the system and tool token counts so it knows the actual budget available for history.

The in-memory `history` list always retains the full, uncompacted data for the current execution. Compaction only transforms the `messages` copy passed to the LLM.

There are two distinct functions involved:

1. **`compact()`** — a plain function (no decorator). It takes the already-converted `messages` list (output of `history_to_input_items` — a list of dicts, not `ConversationItem` objects) plus the system/tool token budget. It applies Layer 1 (surgical clearing) on this dict list, checks if the result fits, and if not, calls `summarize_history()`. It returns a `CompactionResult` dataclass:

```python
@dataclasses.dataclass
class CompactionResult:
    """
    Result of running ``compact()`` on a messages list.

    :param messages: The compacted messages list, ready to pass
        to the LLM.
    :param summary_metadata: Present only when Layer 2
        (summarization) was triggered. Contains the summary
        text and the ``last_item_id`` of the last item
        covered by the summary — used by the workflow to
        persist a compaction item at end of execution.
        ``None`` when only Layer 1 or Layer 3 applied
        (or when compaction was not needed).
    """
    messages: list[dict[str, object]]
    summary_metadata: SummaryMetadata | None
```

```python
@dataclasses.dataclass
class SummaryMetadata:
    """
    Metadata from a Layer 2 summarization, carried from
    ``compact()`` to the workflow's end-of-execution
    persistence step.

    :param text: The LLM-generated summary text.
    :param last_item_id: The ID of the last conversation
        item covered by this summary, e.g. ``"msg_abc123"``.
    :param model: The model used for summarization, e.g.
        ``"openai/gpt-4o"``.
    :param token_count: Approximate token count of the
        summary text.
    """
    text: str
    last_item_id: str
    model: str
    token_count: int
```

The workflow checks `result.summary_metadata is not None` at end of execution to decide whether to persist a compaction item. The original `history` list of `ConversationItem` objects is never read or mutated by `compact()`.

2. **`summarize_history()`** — a separate `@step`-decorated function. It makes the LLM call to generate a summary. Because it is a `@step`, DBOS checkpoints its output. On recovery, the cached summary is reused — no re-summarization.

Layer 1 (surgical clearing) happens inside `compact()` as pure data transformation — deterministic, no side effects. Layer 2 (summarization) is delegated to the separate `summarize_history` step only when Layer 1 was not sufficient.

The compaction item is persisted to the conversation store at the end of the workflow, outside of a `@step`. This raises a crash-safety question: if the process crashes after the `append` but before the workflow marks complete, DBOS recovery will re-execute the tail of the workflow and the `append` runs again — producing a duplicate compaction item.

**Solution: idempotent append via `response_id` dedup.** Every compaction item carries a `response_id` (the task ID), which is unique per execution. Before appending, check whether a compaction item with that `response_id` already exists in the conversation. If so, skip the append. This is a simple `list_items(conversation_id, type="compaction")` scan filtered by `response_id` — or a dedicated `EXISTS` query. No duplicate is ever created, regardless of how many times recovery replays the workflow tail.

---

## Crash Safety

### Duplicate compaction item on crash recovery

**Scenario:** Process crashes after the compaction item `append` but before the workflow marks complete. DBOS recovery replays the workflow tail, and the `append` runs again.

**Solution:** Idempotent append via `response_id` dedup (described above). Check-before-write using the task ID — unique per execution, so duplicates are impossible.

### Summary generated mid-execution, more items appended after

**Scenario:** Layer 2 triggers at iteration 5 and summarizes history up to item `msg_100`. Iterations 6–10 continue, appending items at positions 101–120. The compaction item is appended at position 121.

**Risk without mitigation:** If `load_history` used the compaction item's own position as the cursor, it would load items after position 121 — skipping items 101–120, which are neither in the summary nor loaded as recent items.

**Solution:** `CompactionData.last_item_id` stores the ID of the last item the summary covers (e.g. `msg_100`), not the compaction item's own ID. `load_history` uses `last_item_id` as the cursor: `fetch_all_items(after="msg_100")` loads items at positions 101+, including the post-summary output items.

### Crash before compaction item is persisted

**Scenario:** Layer 2 runs mid-execution (summary is cached as a `@step`). The execution continues, appends output items. Process crashes before the compaction item is written at the end.

**Effect:** No compaction item exists in the store. The next execution loads full history (existing behavior). The summary work is wasted but there is no inconsistency. The `summarize_history` step result is cached in DBOS, but that only helps if the *same* workflow recovers — a new execution starts a fresh workflow.

**This is acceptable:** The compaction item is an optimization for the next turn. Missing it means one extra full load, not data loss or corruption.

### Concurrent execution on the same conversation

**Scenario:** Could two executions run simultaneously on the same conversation and both write compaction items?

**Not possible:** The task store / inbox mechanism enforces at most one active execution per conversation. Steering delivers messages to the running execution's inbox — it does not start a new execution. This invariant is enforced before compaction is involved.

### Stale compaction item after conversation modification

**Scenario:** A compaction item references `last_item_id="msg_100"`. Could items at positions <= 100 be modified or deleted, making the summary inaccurate?

**Not possible:** Conversation items are append-only and immutable after creation. The store has no update or delete API for individual items (only `delete_conversation` which removes everything). The summary's coverage is stable once written. Note: `last_item_id` is only valid within the conversation that contains the compaction item — it is not a cross-conversation reference.

---

## Interaction with Steering

Steered messages arrive via `try_deliver` and are discovered by `_sync_history` each iteration. Compaction must handle steered messages correctly:

1. **Never compact away unprocessed steered messages.** The recent window protects the last N iterations. Steered messages within that window are preserved verbatim.

2. **Summarized steered messages are acceptable.** Once a steered message has been processed (the LLM has seen it and responded), it can be included in a summary like any other message. The summary should capture the steering content.

3. **Clearing tool results does not affect steering.** Steered messages are user-role messages, not tool results. Layer 1 never touches them.

4. **Compaction items do not interfere with steering cursors.** The `last_seen` cursor tracks the latest item the agent has processed. A compaction item appended at the end of an execution has a position after all output items. The next execution's `_sync_history` uses `after=last_seen` which starts after the compaction item, correctly skipping it.

5. **`_sync_history` must filter out compaction items.** `_sync_history` calls `fetch_all_items(after=last_seen)` and extends `history` in-place. If a compaction item from a previous execution has a position after `last_seen`, it would be added to `history`. The prompt builder (`history_to_input_items`) would silently skip it (unhandled type), but the history list should not contain items the prompt builder drops. `_sync_history` must filter: `new_items = [item for item in new_items if item.type != "compaction"]`.

---

## Interaction with Sub-Agents

Sub-agents (spawned via `SpawnTool`) create isolated conversations (`kind="sub_agent"`) and run in separate DBOS workflows. The compaction design applies uniformly — no special sub-agent behavior.

### Same compaction strategy

Sub-agents execute via `_run_agent_loop()` — the same code path as top-level tasks. All three compaction layers (surgical clearing, summarization, truncation) apply identically. The sub-agent's own `compaction` config (from its spec) controls thresholds and recent window size. No framework in the industry treats sub-agent compaction differently from main agent compaction — LangGraph, Google ADK, OpenAI Agents SDK, CrewAI, AutoGen, Letta, and Semantic Kernel all apply the same rules uniformly.

### Sub-agent compaction does not block the parent

Sub-agents run in their own DBOS workflows (W2, W3, etc.), not inside the parent's workflow (W1). The parent only blocks when it explicitly calls `collect_sub_agents()`, which waits via `handle.get_result()`. If a sub-agent triggers Layer 2 summarization, that LLM call happens in the sub-agent's workflow thread — it extends the sub-agent's total execution time but does not hold a lock or resource in the parent. From the parent's perspective, the sub-agent simply takes longer to complete. This is no different from the sub-agent making an extra tool call.

### Compaction items are persisted for sub-agent conversations

Sub-agent conversations are single-use — each `SpawnTool` invocation creates a new conversation, and no future execution loads from it. Despite this, compaction items are persisted to the conversation store for auditability, matching the industry norm:

- **Google ADK** writes compaction events to the Session even after the sub-agent is done.
- **OpenAI Agents SDK** persists compaction items to Sessions so they appear in browsable history.
- **LangGraph** checkpoints include compacted state for all subgraphs.

The cost is one DB write per sub-agent execution that triggers Layer 2. The value is observability: when debugging why a sub-agent produced poor output, seeing "compaction happened at item 40" in `GET /v1/conversations/{conv_id}/items` is useful diagnostic context. Without it, the only signal would be log lines.

### Sub-agents as a complementary context management strategy

Sub-agents are themselves a form of context management. Verbose operations (file analysis, multi-step research, large code generation) run in a sub-agent's isolated context window. Only the extracted final text (~100–2000 tokens) returns to the parent via `collect_sub_agents`. This is the pattern recommended by Anthropic's context engineering guidance and used by Claude Code — prevent large content from entering the main context rather than compacting it after the fact.

Compaction and sub-agent delegation are complementary:
- **Sub-agent delegation** prevents context growth proactively (verbose work never enters the parent's history).
- **Compaction** handles context growth reactively (when history grows despite best efforts).

Both mechanisms coexist without interference. A parent agent may compact its own history (which includes the short sub-agent output strings) while sub-agents independently compact theirs.

---

## Client Status Reporting

Compaction — especially Layer 2 summarization — involves an extra LLM call that adds seconds of latency. Without a signal, the client cannot distinguish "agent is thinking" from "something is stuck."

### Event: `response.compaction.in_progress`

Emitted once, immediately before the `summarize_history` `@step` call. No corresponding `.completed` event — the next `response.output_text.delta` or `response.output_item.done` implicitly signals compaction finished.

```json
{
  "type": "response.compaction.in_progress",
  "sequence_number": 12
}
```

**Naming rationale:** Follows the OpenAI Responses API convention for internal operations that take time: `response.{operation}.in_progress` (e.g. `response.file_search_call.in_progress`, `response.web_search_call.in_progress`, `response.mcp_call.in_progress`). OpenAI does not define a compaction event (their compaction is opaque), and no other platform (Anthropic, Google, LangGraph, Vercel, AG-UI) emits compaction or summarization events. This is a new event type specific to agent-plane.

**When emitted:** Only when Layer 2 (summarization) triggers. Layer 1 (clearing tool result bodies) and Layer 3 (truncation) are fast in-memory operations — no status event needed.

**Not emitted when:** Compaction is not needed (token estimate below threshold), or only Layer 1/Layer 3 apply.

**Client behavior:** Clients can use this to show a transient indicator (e.g. "Summarizing conversation..."). Clients that don't recognize the event can safely ignore it — SSE consumers skip unknown event types by convention.

**Implementation:** In `workflow.py`, call `_write_output(task_id, {"type": "response.compaction.in_progress"})` immediately before the `summarize_history()` step.

---

## Test Plan

### Unit tests: `compact()` (pure function, no mocks needed)

1. **Under threshold — no-op.** Build a history with a few small messages. Call `compact()` with a high threshold. Assert the returned messages are identical to the input (no clearing, no summarization).

2. **Layer 1 triggers — tool results cleared.** Build a history with several large `function_call_output` items outside the recent window and a few small items inside it. Set threshold low enough that the full history exceeds it but cleared history does not. Assert: tool result bodies outside the recent window are replaced with the clearing marker. Tool result bodies inside the recent window are untouched. `function_call` items (name + arguments) are never modified. Non-tool messages are never modified.

3. **Layer 1 preserves tool call pairs.** Build a history with `function_call` + `function_call_output` pairs. After clearing, assert every `function_call` still has its corresponding `function_call_output` (with cleared body). No orphans.

4. **Layer 1 never touches steered messages.** Build a history with user messages (including one injected via steering) interleaved with tool results outside the recent window. Assert: text content in user messages is untouched after clearing.

5. **Layer 1 clears binary content blocks in user/assistant messages.** Build a history with user messages containing image content blocks (with `file_id` and base64 `data`) outside the recent window. Assert: after clearing, the image block's `file_id` is preserved, `data` is replaced with `[binary content removed for context management — use file_id to retrieve]`, and text content blocks in the same message are untouched. Assert: image blocks inside the recent window are not modified.

6. **Recent window boundary.** Parameterize `recent_window` (e.g. 3, 5, 10). Build history with exactly that many recent iterations. Assert items inside the window are never modified. Items just outside the window are cleared.

7. **Layer 2 triggers when Layer 1 insufficient.** Build a history where even after clearing all tool results, the token count still exceeds threshold (e.g. many long assistant messages). Mock `summarize_history` to return a canned summary. Assert: `compact()` called `summarize_history` and returned the summary + recent window items.

8. **Layer 1 feeds into Layer 2 — summarization input has cleared content.** Build a history with large tool results AND image content blocks outside the recent window, plus long assistant messages (so Layer 1 alone is insufficient). Mock `summarize_history` and capture the messages it receives. Assert: tool result bodies and binary content blocks in the summarization input are already cleared (Layer 1 was applied before passing to Layer 2). Assert: file IDs are preserved in the cleared image blocks. This proves the layers compose correctly — the summary is generated from the cleared messages, not the original full-size messages.

9. **Layer 3 triggers when Layer 2 insufficient.** Build a history where the recent window alone exceeds threshold (e.g. one massive tool result in the most recent iteration). Assert: oldest items in the recent window are truncated, tool call pairs are kept together, and the result fits under threshold.

### Unit tests: `summarize_history()` step

10. **Produces summary with correct metadata.** Mock the LLM call. Assert: returned summary includes `last_item_id` matching the last item in the input, `model` matching the agent's `llm.model`, and `token_count` > 0.

11. **Recursive summarization includes previous summary.** Provide a history that starts with a synthetic summary message (from a prior compaction). Assert: the prompt sent to the LLM includes the previous summary text.

12. **Summarization failure falls back to Layer 3.** Mock `summarize_history` to raise an exception (e.g. `RetryableLLMError`). Call `compact()` with a history that requires Layer 2 (Layer 1 alone insufficient). Assert: `compact()` does not raise — it falls back to Layer 3 (truncation). Assert: returned messages are truncated (oldest items removed, tool call pairs intact). Assert: `summary_metadata` is `None` (no compaction item should be persisted). Assert: a warning is logged indicating summarization failure and fallback.

### Unit tests: `ContextWindowExceededError` classification

13. **OpenAI overflow detected.** Feed `_classify_error()` an HTTP 400 response with `error.code == "context_length_exceeded"` and message `"This model's maximum context length is 128000 tokens. However, you requested 142000 tokens"`. Assert: raises `ContextWindowExceededError` with `max_context_tokens=128000` and `actual_tokens=142000`.

14. **Anthropic overflow detected.** Feed `_classify_error()` an HTTP 400 response with message matching `"197202 + 21333 > 200000"`. Assert: raises `ContextWindowExceededError` with `max_context_tokens=200000` and `actual_tokens=197202`. Test a second pattern: `"prompt is too long: 210000 tokens > 200000 maximum"`. Assert: same extraction logic works.

15. **Gemini overflow detected.** Feed `_classify_error()` an HTTP 400 response with message `"input token count (1100000) exceeds the maximum number of tokens allowed (1048576)"`. Assert: raises `ContextWindowExceededError` with `max_context_tokens=1048576` and `actual_tokens=1100000`.

16. **Unrecognized 400 error is not misclassified.** Feed `_classify_error()` an HTTP 400 response with a generic message (e.g. `"invalid request: missing 'model' field"`). Assert: raises `PermanentLLMError`, NOT `ContextWindowExceededError`.

### Unit tests: tiktoken validation gate

17. **Plausible overflow proceeds with compaction.** Catch a `ContextWindowExceededError` with `actual_tokens=142000`. Monkeypatch tiktoken to estimate 140000 tokens for the same messages. Assert: the workflow proceeds to compact (divergence is ~1.4%, well within 30%).

18. **Implausible overflow re-raises as PermanentLLMError.** Catch a `ContextWindowExceededError` with `actual_tokens=142000`. Monkeypatch tiktoken to estimate 50000 tokens for the same messages (divergence ~65%). Assert: the error is re-raised as `PermanentLLMError` (not compacted). Assert: a warning is logged indicating the token count divergence.

### Unit tests: `compaction_to_history_item()` output format

19. **Produces valid synthetic pair.** Pass a compaction item with a known summary. Assert: returns a list of two items. First item has `type="message"`, `data.role="user"`, content includes the summary marker prefix. Second item has `type="message"`, `data.role="assistant"`, content equals the summary text. Assert: both items can be processed by `history_to_input_items` without error and produce valid LLM input dicts.

### Unit tests: `list_items` type filter

20. **Filter by type returns only matching items.** Append a mix of `message`, `function_call`, `function_call_output`, and `compaction` items. Call `list_items(type="compaction")`. Assert: only compaction items returned. Call `list_items(type="message")`. Assert: only message items returned. Call `list_items()` (no filter). Assert: all items returned.

21. **Type filter with order and limit.** Append multiple compaction items. Call `list_items(type="compaction", order="desc", limit=1)`. Assert: returns only the latest compaction item.

### Unit tests: `load_history` with compaction items

22. **No compaction item — loads everything.** Append 50 items with no compaction item. Assert: `load_history` returns all 50 items.

23. **Single compaction item — loads summary + recent items.** Append 50 items, then a compaction item with `last_item_id` pointing to item 40. Append 10 more items after the compaction item. Assert: `load_history` returns a synthetic summary message + the 10 items after item 40 (excluding the compaction item itself). The 40 original items are not loaded.

24. **Post-summary items are not lost.** Append 50 items. Append a compaction item with `last_item_id` pointing to item 30 (simulating summary generated mid-execution at iteration covering items 0–30, then iterations continued appending items 31–50). Assert: `load_history` returns summary + items 31–50 (excluding the compaction item). Items 31–50 are NOT covered by the summary and must be present. This is the gap scenario from crash safety.

25. **Multiple compaction items — uses latest.** Append items, then compaction item C1 (covering items 0–20), then more items, then compaction item C2 (covering items 0–40 via recursive summary). Assert: `load_history` uses C2, loads summary + items after item 40. C1 is filtered out.

26. **Compaction items filtered from history.** Assert: the list returned by `load_history` contains no items with `type="compaction"`. They are metadata, not conversation content.

### Unit tests: compaction item persistence

27. **Idempotent append — no duplicate on retry.** Append a compaction item with `response_id="task_1"`. Attempt to append another compaction item with the same `response_id`. Assert: only one compaction item exists in the conversation.

28. **Different executions produce separate compaction items.** Append a compaction item with `response_id="task_1"`, then another with `response_id="task_2"`. Assert: both exist in the conversation.

### Integration tests: compaction in the agent loop

All integration tests run against a real server: real FastAPI app, real SQLAlchemy stores, real DBOS, `httpx.AsyncClient` via `ASGITransport`. The LLM is replaced with `ControllableMockClient`. Token counts are controlled by monkeypatching `token_counter` to return inflated values so compaction triggers with a small number of messages. Tests use the same `build_server` / `destroy_dbos` pattern as the durability tests.

29. **Compaction triggers and persists during long execution.** Build server. Register agent. Monkeypatch `token_counter` to return inflated counts so the threshold is exceeded after a few tool-call iterations. Queue mock LLM calls that produce tool calls, then a final text response. `POST /v1/responses` and poll until complete. Assert: `GET /v1/responses/{id}` shows completed status. Assert: `GET /v1/conversations/{conv_id}/items` (paginated, unfiltered) includes a `compaction` item. Assert: the compaction item's `last_item_id` points to an item that exists in the conversation. Assert: the mock LLM's call history shows it received compacted messages (tool result bodies cleared or summary present) on later iterations.

30. **Next execution loads from compaction item.** Continuing from test 29's database (same `db_uri`), rebuild the server. `POST /v1/responses` with `previous_response_id` pointing to the previous response. Poll until complete. Assert: the response completes successfully. Assert: `GET /v1/conversations/{conv_id}/items` shows the new response's items appended after the compaction item. Assert: the mock LLM received a summary message as the first history entry (not the original items from positions 0–N). This proves `load_history` used the compaction item rather than loading everything.

31. **Compaction item survives crash and recovery.** Build server. Register agent. Monkeypatch `token_counter` for inflated counts. Queue mock LLM calls: tool-call response (triggers compaction via `summarize_history` step), then a blocking call (simulates crash mid-LLM). `POST /v1/responses`, wait for the blocking call to be entered. Tear down server + DBOS (crash). Rebuild server on the same `db_uri`. Queue a recovery mock call. Let DBOS recover. Assert: the `summarize_history` step replays from DBOS cache (mock LLM call count proves it — the summary LLM call is NOT repeated). Assert: `GET /v1/conversations/{conv_id}/items` includes exactly one compaction item (idempotent dedup works).

32. **Steering messages survive compaction.** Build server. Register agent. Monkeypatch `token_counter` for inflated counts. Queue mock LLM calls: first call blocks (to create a window for steering). `POST /v1/responses` to start execution. Wait for the first LLM call to be entered. `POST /v1/responses` with `previous_response_id` to deliver a steered message. Release the blocking call. Queue remaining mock calls (tool call + final text). Poll until complete. Assert: `GET /v1/conversations/{conv_id}/items` includes both the steered message and the compaction item. Assert: the steered message's position is after the compaction item's `last_item_id` (it was in the recent window, not summarized away).

33. **User can browse full history after compaction.** After test 29 completes, paginate through the full conversation: `GET /v1/conversations/{conv_id}/items?order=asc` repeatedly using `after` cursors until `has_more` is false. Assert: all original items from position 0 onward are present (user messages, assistant messages, function calls, function call outputs). Assert: the compaction item appears in sequence at the expected position. Assert: no items have been deleted or modified — the total count matches the number of items appended during the execution. This proves compaction is additive and the user's scrollable history is unaffected.

34. **`response.compaction.in_progress` event emitted when Layer 2 triggers.** Build server. Register agent. Monkeypatch `token_counter` for inflated counts so compaction triggers. Stream the response via `POST /v1/responses` (streaming mode). Collect all SSE events. Assert: a `response.compaction.in_progress` event appears in the stream. Assert: it appears before any `response.output_text.delta` events from the post-compaction LLM call. Assert: no `response.compaction.completed` event exists (we only emit `.in_progress`).

35. **No compaction event when threshold not exceeded.** Build server. Register agent. Use default `token_counter` (no monkeypatch — history is small). Stream the response. Collect all SSE events. Assert: no `response.compaction.in_progress` event appears.

---

## Implementation Checklist

Adding the `compaction` item type requires changes across multiple files. This is the full list:

1. **`agent_plane/entities/conversation.py`**
   - Add `CompactionData(BaseModel)` class.
   - Add `"compaction": CompactionData` to `ITEM_TYPE_TO_DATA_CLS` dict. Without this, `parse_item_data` raises `ValueError("unknown item type: 'compaction'")` when reading compaction items from the DB.

2. **`agent_plane/db/utils.py`**
   - Add `"compaction": "cmp_"` to `_ITEM_TYPE_PREFIX` dict. Without this, `generate_item_id` fails when appending compaction items.

3. **`agent_plane/db/utils.py`** (search text) **+** **`agent_plane/stores/conversation_store/sqlalchemy_store.py`** (type filter)
   - In `db/utils.py`: add a `"compaction"` case to `extract_search_text`. The summary text should be searchable — return `data.summary`.
   - In `sqlalchemy_store.py`: add optional `type` filter parameter to `list_items` (and the abstract base class). Single `WHERE type = :type` clause.

4. **`agent_plane/stores/conversation_store/__init__.py`**
   - Add `type: str | None = None` parameter to the abstract `list_items` method signature.

5. **`agent_plane/runtime/prompt.py`**
   - Add explicit skip for `"compaction"` in `history_to_input_items`, with a comment like the existing `reasoning` skip. Today unknown types are silently dropped — make this intentional and visible.

6. **`agent_plane/runtime/workflow.py`**
   - Update `_run_agent_loop`: add compaction item lookup at start, conditional history load via `last_item_id`, filter out `type="compaction"` from loaded items.
   - Update `_run_agent_loop` end: persist compaction item (with `response_id` dedup check) if Layer 2 was triggered.
   - Update `_sync_history`: filter out `type="compaction"` from fetched items before extending history (compaction items appended by a previous execution should not enter the in-memory history).
   - Add `compact()` function and `summarize_history()` `@step` function.
   - Emit `response.compaction.in_progress` event via `_write_output()` immediately before calling `summarize_history()`.

7. **`agent_plane/runtime/prompt.py` or new `agent_plane/runtime/compaction.py`**
   - `compact()`: pure function taking messages list + config, returning compacted messages.
   - `compaction_to_history_item()`: convert compaction item to synthetic message.

8. **`agent_plane/spec/` (agent spec parsing)**
   - Add `compaction` config section to the agent spec schema (parser + validator). Fields: `trigger_threshold`, `recent_window`.

9. **`agent_plane/server/` (API layer)**
   - Verify the API response schema for `list_items` accepts `type="compaction"`. If the schema validates against a fixed set of types, add `"compaction"` to the allowed values.

10. **`llms/errors.py`** + **`llms/client.py`** (overflow error classification)
   - Add `ContextWindowExceededError(PermanentLLMError)` with `max_context_tokens` and `actual_tokens` fields.
   - Update `_classify_error()` to detect context overflow patterns in HTTP 400 responses (OpenAI `context_length_exceeded`, Anthropic token limit messages, Gemini `INVALID_ARGUMENT` with token limit) and raise `ContextWindowExceededError` instead of `PermanentLLMError`.

11. **`pyproject.toml`**
   - Add `tiktoken>=0.7` dependency for local token estimation and overflow validation.

12. **Tests** — see Test Plan section.

---

## What This Design Does NOT Cover (Future Work)

- **Memory/retrieval:** Letta-style archival memory (move facts to a searchable store, retrieve on demand) requires memory store infrastructure that doesn't exist yet. This is complementary to compaction — compaction manages the prompt, memory manages long-term knowledge.

- **Compaction-aware steering:** If a steered message references context that was summarized away, the agent may not understand the reference. This is an inherent trade-off of summarization. A future enhancement could detect this and re-expand relevant history.

- **Separate summary model:** Summarization currently uses the agent's main `llm.model`. A future `summary_model` config option could allow using a cheaper/faster model (e.g. `gpt-4o-mini`) for summarization. This requires checking the summary model's context window for input overflow (not the main model's) and adds a new config field to the agent spec.

- **Storage TTL:** If database size becomes a concern, a background job can delete conversation items older than a retention threshold. This is independent of the compaction design and requires no changes to the item schema or load logic.
