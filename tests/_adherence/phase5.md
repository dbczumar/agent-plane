# Phase 5 Adherence Checklist

Source: `/home/ubuntu/session_model_notes.md` §8 Phase 5.

Phase 5 = Async client-side tools. The LLM can call a
client-defined function whose body runs OUTSIDE the server (in
the user's frontend / SDK process), and the framework treats
it identically to async server-side `@tool` work — same
unified `async_work_complete` drain, same handle JSON,
auto-delivery on completion.

---

## Core deliverables

- [ ] **D1**: ClientSideToolSpec schema extension — recognize
      `synchronous: false` field on function definitions in the
      `tools:` list of `POST /responses`. Default `True`
      preserves Phase 1 client-tool behavior.
- [ ] **D2**: New task kind `"client_tool"` in `task_store`
      lifecycle. Eligible for unified drain (already in
      `_DRAIN_KINDS` from Phase 2/3 — verify includes
      "client_tool").
- [ ] **D3**: Workflow dispatch — when LLM calls a
      `synchronous=false` client tool, do NOT park; create a
      kind="client_tool" task row, emit `function_call` SSE
      with `task_id` field, return
      `{task_id, kind: "client_tool"}` handle to the LLM
      inline. The async drain delivers the eventual result
      from the client's PATCH.
- [ ] **D4**: PATCH `/v1/responses/{id}` extended schema —
      new optional `async_tool_results: list[{task_id, status,
      output?, error?}]` alongside existing `tool_results`.
      Handler updates `task_store` (idempotent) and signals
      `DBOS.send(parent_task_id, payload,
      topic="async_work_complete")` so the parent's drain
      auto-delivers.
- [ ] **D5**: New SSE event `response.client_task.cancel`
      `{task_id}`. Emitted by `cancel_task` (and parent-cancel
      propagation) when the target has `kind="client_tool"`.
      Lets the client-side dispatcher cancel the local asyncio
      task and PATCH back `status: "cancelled"`.
- [ ] **D6**: Client library — extend `agent_plane_client` (or
      whichever SDK package owns frontends) to scan for
      `@tool`-decorated functions, dispatch on incoming
      `function_call` events, spawn background asyncio task for
      `synchronous=False`, track by server-issued `task_id`,
      PATCH `async_tool_results` on completion, handle
      `response.client_task.cancel`. (Deferrable if the SDK
      package layout is mid-refactor.)
- [ ] **D7**: Migrate `agent_plane/client_tools/coding.py` to
      `@tool`-decorated functions. All 7 (Read, Write, Edit,
      Glob, Grep, Bash, LSP) remain synchronous.

## G-decisions applicable to Phase 5

- [ ] **G3**: Client cancel race — first-write-wins on terminal
      status in `task_store`; late PATCH after server-side
      cancel is accepted but doesn't change the cancelled
      status away.

## Tests

### Unit — `tests/server/test_schemas.py` (extend)

- [ ] `test_client_tool_accepts_synchronous_false_field`.
- [ ] `test_patch_request_accepts_async_tool_results`.
- [ ] `test_patch_request_async_tool_result_requires_task_id`.

### Server integration — `tests/server/integration/test_async_client_tool_integration.py` (new)

- [ ] `test_async_client_tool_returns_handle`.
- [ ] `test_async_client_tool_completion_delivered_via_patch`.
- [ ] `test_async_client_tool_mixed_patch_body`.
- [ ] `test_async_client_tool_failure_surfaces_error`.
- [ ] `test_async_client_tool_auto_collect_at_turn_end`.
- [ ] `test_cancel_client_tool_emits_sse_cancel_event` —
      concurrency.
- [ ] `test_parent_cancel_propagates_to_client_tools` —
      concurrency / non-blocking.
- [ ] `test_async_client_tool_idempotent_patch`.
- [ ] `test_unknown_task_id_in_async_patch_returns_404`.
- [ ] `test_patch_for_already_cancelled_task_respects_cancellation`.
- [ ] `test_check_task_on_client_tool_returns_status_no_activity`.
- [ ] `test_list_tasks_includes_client_tools`.

### E2E — `tests/e2e/test_async_client_tool_e2e.py` (new)

- [ ] `test_async_client_tool_decorator_long_running_e2e`.
- [ ] `test_async_client_tool_decorator_cancel_e2e`.
- [ ] `test_async_client_tool_decorator_parallel_e2e`.
- [ ] `test_async_client_tool_decorator_mixed_sync_and_async_e2e`.
- [ ] `test_async_client_tool_failure_e2e`.
- [ ] `test_full_stack_kitchen_sink_e2e` — sub-agent + server
      async tool + client async tool in parallel.

## Closing checks

- [ ] All Phase 5 unit tests pass.
- [ ] All Phase 5 server integration tests pass.
- [ ] Phase 2/3/4 test suites still green (no regressions).
- [ ] E2E suite (where the client-library wiring lands) green.
- [ ] Post-change review subagent invoked per part.
