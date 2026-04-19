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

- [x] **D1**: ClientSideToolSpec schema extension — recognize
      `synchronous: false` field on function definitions in the
      `tools:` list of `POST /responses`. Default `True`
      preserves Phase 1 client-tool behavior.
      *Landed in part A (76deb55).*
- [x] **D2**: New task kind `"client_tool"` in `task_store`
      lifecycle. Eligible for unified drain (already in
      `_DRAIN_KINDS` from Phase 2/3 — verify includes
      "client_tool"). *Landed in part A (76deb55).*
- [x] **D3**: Workflow dispatch — when LLM calls a
      `synchronous=false` client tool, do NOT park; create a
      kind="client_tool" task row, emit `function_call` SSE
      with `task_id` field, return
      `{task_id, kind: "client_tool"}` handle to the LLM
      inline. The async drain delivers the eventual result
      from the client's PATCH. *Landed in part A (76deb55).*
- [x] **D4**: PATCH `/v1/responses/{id}` extended schema —
      new optional `async_tool_results: list[{task_id, status,
      output?, error?}]` alongside existing `tool_results`.
      Handler updates `task_store` (idempotent) and signals
      `DBOS.send(parent_task_id, payload,
      topic="async_work_complete")` so the parent's drain
      auto-delivers. *Landed in part B (eb0951f).*
- [x] **D5**: New SSE event `response.client_task.cancel`
      `{task_id}`. Emitted by `cancel_task` (and parent-cancel
      propagation) when the target has `kind="client_tool"`.
      Lets the client-side dispatcher cancel the local asyncio
      task and PATCH back `status: "cancelled"`.
      *Server-side: part B (eb0951f). SDK parser: part D.*
- [~] **D6**: Client library — extend `agent_plane_client` (or
      whichever SDK package owns frontends) to scan for
      `@tool`-decorated functions, dispatch on incoming
      `function_call` events, spawn background asyncio task for
      `synchronous=False`, track by server-issued `task_id`,
      PATCH `async_tool_results` on completion, handle
      `response.client_task.cancel`.
      **Partial (part D, this commit):**
      - `build_tool_handler` propagates `synchronous: false`
        flag onto the wire schema so the server takes the
        async-dispatch path (covered by
        `tests/frontends/sdk/test_async_client_tool_sdk.py`).
      - `ClientTaskCancel` event class + SSE parser entry
        (covered by same test file).
      **Deferred to follow-up work:**
      - Background asyncio task tracking (`call_id` →
        `asyncio.Task`).
      - Auto-PATCH `async_tool_results` when the asyncio task
        completes (success / failure / cancellation).
      - On `ClientTaskCancel`, cancel the matching local
        `asyncio.Task` and PATCH back
        `status: "cancelled"`.
      - 1-hour max lifetime cap via `asyncio.wait_for`.
      - Stream loop changes to fire-and-forget rather than
        block on async tools (current loop assumes synchronous
        execution model).
      The deferred lifecycle work requires a redesign of the
      `stream()` execution model that exceeds the scope safely
      validatable in one commit; the wire-protocol pieces above
      are sufficient for hand-rolled async-tool clients to
      interoperate today.
- [ ] **D7**: Migrate `agent_plane/client_tools/coding.py` to
      `@tool`-decorated functions. All 7 (Read, Write, Edit,
      Glob, Grep, Bash, LSP) remain synchronous.
      **Deferred** — depends on D6 lifecycle work to validate
      the migrated tools against the SDK; sequencing kept with
      D6.

## G-decisions applicable to Phase 5

- [ ] **G3**: Client cancel race — first-write-wins on terminal
      status in `task_store`; late PATCH after server-side
      cancel is accepted but doesn't change the cancelled
      status away.

## Tests

### Server integration — `tests/server/integration/test_async_client_tool_integration.py`

Landed in part C (f38f20e); strengthened in review fixes (bea3f1c).
Coverage of the spec'd test cases (named-equivalent in our file):

- [x] `test_async_client_tool_returns_handle`.
- [x] `test_async_client_tool_completion_delivered_via_patch`.
- [x] `test_mixed_tool_results_and_async_tool_results_in_one_patch`.
- [x] `test_async_client_tool_failure_surfaces_error`.
- [x] `test_async_client_tool_auto_delivers_completion`
      (auto-collect-at-turn-end equivalent).
- [x] `test_parent_cancel_emits_response_client_task_cancel_sse`
      (parent-cancel + cancel-SSE combined).
- [x] `test_async_patch_after_cancel_is_noop`
      (idempotent-patch + first-write-wins combined).
- [x] `test_async_patch_unknown_task_id_returns_404`.
- [x] `test_async_patch_rejects_non_client_tool_kind`.
- [x] `test_list_tasks_includes_client_tool_kind`.

Total: 9 server-integration tests, all green in 40s.

### SDK unit — `tests/frontends/sdk/test_async_client_tool_sdk.py`

Landed in part D (this commit). Covers the wire-protocol slice:

- [x] `test_build_tool_handler_omits_synchronous_for_sync_tools`.
- [x] `test_build_tool_handler_emits_synchronous_false_for_async_tools`.
- [x] `test_build_tool_handler_mixes_sync_and_async`.
- [x] `test_sse_parser_emits_client_task_cancel`.
- [x] `test_sse_parser_drops_client_task_cancel_without_task_id`.
- [x] `test_sse_parser_unchanged_for_function_call_output`.

Total: 6 SDK unit tests, all green in 0.1s.

### E2E — `tests/e2e/test_async_client_tool_e2e.py`

**Deferred** with D6 lifecycle work — these tests require
the full SDK-side asyncio task tracking + auto-PATCH machinery
to be in place. The server-side guarantees they would exercise
are already covered by the integration suite above; the
remaining gap is end-to-end SDK use with a real LLM.

- [ ] `test_async_client_tool_decorator_long_running_e2e`.
- [ ] `test_async_client_tool_decorator_cancel_e2e`.
- [ ] `test_async_client_tool_decorator_parallel_e2e`.
- [ ] `test_async_client_tool_decorator_mixed_sync_and_async_e2e`.
- [ ] `test_async_client_tool_failure_e2e`.
- [ ] `test_full_stack_kitchen_sink_e2e` — sub-agent + server
      async tool + client async tool in parallel.

## Closing checks

- [x] All Phase 5 server integration tests pass.
- [x] Phase 2/3/4 test suites still green (no regressions
      attributable to Phase 5; pre-existing MCP-test bundle
      failures and known cross-file ordering pollution
      reproduce both with and without Phase 5).
- [ ] E2E suite — deferred with D6/D7.
- [x] Post-change review subagent invoked per part.
