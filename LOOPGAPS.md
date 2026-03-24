# Layer 2 Implementation Gaps

Gaps discovered during agent loop implementation. Review before committing.

## Addressed During Implementation

_(Gaps that were found and fixed inline)_

## Open Gaps

### Streaming LLM calls
AGENTLOOP.md calls for `litellm.completion(stream=True)` with token-by-token SSE deltas.
Current implementation uses `stream=False` — the client gets the complete response at once
(still via SSE lifecycle events, just no `response.output_text.delta` events).
**Reason**: Need to verify `write_stream()` works inside a DBOS `@step` first. The design
doc flags this as an open question.

### MCP tool support
`ToolManager.start()` does not connect to MCP servers. Only the `load_skill` built-in is
implemented. Agents with `mcp_servers` in their spec will have tools listed but MCP calls
will fail at runtime.

### Output item `_to_api_item` duplication
`_item_to_output()` in `workflow.py` duplicates `_to_api_item()` in
`server/routes/conversations.py`. Should be a shared function in `entities/`.

### `agent_name` threaded through workflow
The workflow needs the agent's registered name (for the `model` field on output items).
Currently resolved by looking up the agent from `get_agent_store()`. A cleaner approach
is to pass `agent_name` as a workflow parameter (it's already on the task row).

### Reasoning items not emitted
If the LLM returns reasoning tokens (e.g. OpenAI o-series models), they are not extracted
or persisted as `reasoning` ConversationItems. Only text content and tool calls are handled.

### Token usage tracking
`litellm` responses include `usage.prompt_tokens` and `usage.completion_tokens` but the
workflow does not extract or propagate these to the task result.

### `completed_at` timestamp
The workflow result does not include a `completed_at` timestamp.

### Parallel tool calls
Tool calls from a single LLM response are executed sequentially, not in parallel.

### Cancellation propagation
Workflow cancellation via DBOS does not interrupt an in-flight LLM call or tool execution.

### Local tool execution
`LocalToolInfo` tools in the agent spec are not executable. `ToolManager.call_tool()` returns
an error for any tool that isn't `load_skill`.
