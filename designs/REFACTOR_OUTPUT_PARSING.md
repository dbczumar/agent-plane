# Refactor: Typed Output Items

## Problem

Output items lose type safety too early. `_item_to_output()` flattens
`ConversationItem` → `dict[str, Any]` inside the workflow, so everything
downstream (`Task.output`, `ResponseObject.output`, `_extract_output_text`)
operates on untyped dicts with string-matching (`item.get("type") == "output_text"`).

The typed models already exist: `MessageData`, `FunctionCallData`,
`FunctionCallOutputData`, `ReasoningData`, plus the `ItemData` union.

## Proposed Fix

**Stop returning output items from the workflow.** The items are already
durably persisted in the conversation store by `_persist_and_stream_items()`.
The workflow returning them in its result dict is redundant.

### Changes

1. **Workflow return**: Remove `output` from `_AgentLoopResult`. The workflow
   returns only `status`, `error`, `incomplete_details`, `usage`, `completed_at`.

2. **`_apply_workflow_status`**: Stop extracting `result["output"]`. The
   `Task.output` field is no longer populated from the workflow result.

3. **`Task.output`**: Change type from `list[dict[str, Any]]` to
   `list[ConversationItem]`. Populated by querying the conversation store
   (items are already there).

4. **Enrichment** (`get`, `get_sync`): After applying workflow status, query
   `conversation_store.get_items(task.conversation_id, response_id=task.id)`
   to populate `task.output` with typed `ConversationItem` objects.

5. **API serialization**: `_build_response_object` calls `_item_to_output()`
   (or equivalent) at the boundary, converting `ConversationItem` → dict
   for JSON serialization. Single place, single conversion.

6. **Consumers** (spawn.py `_extract_output_text`): Operate on typed
   `ConversationItem` objects. Replace string matching with field access.

7. **Delete** `_item_to_output` from workflow.py. Consolidate with
   `_to_api_item` in conversations.py into a shared function in `entities/`.

### Migration

- Remove `output` from workflow return dict
- Existing DBOS workflow results with `output` key are harmless —
  `_apply_workflow_status` simply stops reading it
- No DB migration needed (output was never in the tasks table)
