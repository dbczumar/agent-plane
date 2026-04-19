# Phase 2 Adherence Checklist

Source: `/home/ubuntu/session_model_notes.md` §8 Phase 2.

Phase 2 = Async custom tools + shared task lifecycle. Mark `[x]`
when the corresponding code + test has landed and the test passes.

---

## Core deliverables

- [ ] **D1**: Alembic migration adds `kind: str NOT NULL DEFAULT
      'agent_task'` column to the `tasks` table. Existing rows
      backfill to `'agent_task'`.
- [ ] **D2**: `@tool(synchronous=False)` extends the decorator.
      When `False`, the framework dispatches the tool via a DBOS
      background workflow rather than calling it inline.
- [ ] **D3**: New DBOS workflow `background_tool_workflow(tool_name,
      args, parent_task_id)` runs a single `@step` (subprocess
      invocation via `_runner.py`) then signals the parent via
      `DBOS.send(parent_task_id, payload, topic="async_work_complete")`.
- [ ] **D4**: Parent workflow gains a between-iteration drain:
      before every LLM call, drain queued
      `async_work_complete` signals and inject each as a system
      message before the next call.
- [ ] **D5**: Parent workflow gains end-of-turn auto-collect: when
      LLM produces a final response with `pending_tasks`
      non-empty, wait via `dbos_recv_async` for each remaining
      completion, inject, and give the LLM another iteration.
      Turn ends only when LLM done AND `pending_tasks` empty.
- [ ] **D6**: New builtin tools — `check_task`, `cancel_task`,
      `list_tasks` — implemented per §4.3 schemas.
- [ ] **D7**: Unified task handle shape:
      `{task_id, kind, status, message, ...}`. The `message`
      field carries self-explanatory instructions to the LLM
      (G12).
- [ ] **D8**: Result size cap (B5 / G44): truncate to ~10,000
      Python `len(str)` characters at the LLM boundary; full
      value in conversation/task store.
- [ ] **D9**: Parent cancel propagates to all pending child
      task_ids non-blocking (B3).
- [ ] **D10**: Frontend SDK changes — `examples/frontends/terminal.py`
      subscribes to `response.heartbeat` (no-op handler);
      renders `[System: task ... completed]` / `[System: task ...
      failed]` user messages distinctly (dim, ⤵ prefix).

## G-decisions applicable to Phase 2

- [ ] **G3**: Cancel/completion race — first-write-wins on
      terminal status in `task_store`; late signal is a no-op.
- [ ] **G7**: Auto-collect under cancel — strict cancel: no
      further LLM call; already-arrived completions persisted.
- [ ] **G10**: `task_store.kind` migration. Forward + reversible.
- [ ] **G12**: `message` field on async tool handles with
      self-explanatory text including specific task_id.
- [ ] **G13**: TUI rendering (in D10).
- [ ] **G18**: Auto-delivered system message persistence shape:
      `role="user"` with `[System: ...]`-prefixed `input_text`
      block. Reuses existing convention from
      `agent_plane/runtime/workflow.py`.
- [ ] **G19**: Drain protocol — `dbos_recv_async(timeout_seconds=0)`
      loop for the between-iteration drain; `dbos_recv_async`
      with no timeout for end-of-turn auto-collect. Both filter
      against `task_store` terminal status.
- [ ] **G20**: SSE keepalive — emit `response.heartbeat` every
      15s during any wait state.
- [ ] **G23**: Task access control — `check_task`, `cancel_task`,
      `list_tasks` scoped to caller's conversation tree.
- [ ] **G31**: Verify DBOS supports long-running steps before
      committing to the single-step background-tool design; if
      not, split into two steps.
- [ ] **G33**: Item ordering — auto-delivered completion message
      appended at completion time, not backdated to spawn time.
- [ ] **G35**: Test infrastructure — blocking mock tool helper,
      multi-agent MockLLM scripting (verify exists), SSE event
      capture helpers.
- [ ] **G44**: Result truncation unit — Python `len(str)` on
      Unicode code points. Bytes/grapheme clusters NOT used.
- [ ] **G50**: `recent_activity` count — 5 items / 2000 chars
      per item (matches existing `_ACTIVITY_TAIL` /
      `_ACTIVITY_MAX_CHARS`).
- [ ] **G54**: Server graceful shutdown — DBOS handles workflow
      durability; subprocess children may double-execute on
      restart for non-idempotent tools (documented in `@tool`
      decorator's docstring).
- [ ] **G55**: `pending_tasks` reconstruction — in-memory; on
      workflow start (after replay), explicit re-query via
      `task_store.list_active_tasks()` filtered to immediate
      child conversations.
- [ ] **G56**: `task_id` ≡ `workflow_id` — documented; no
      separate field.
- [ ] **G57**: `list_tasks` filters out `kind="agent_task"` by
      default. `check_task` / `cancel_task` reject
      `agent_task` task_ids with `task_not_found`.
- [ ] **G61**: Steering during auto-collect — wakes the wait;
      already-drained signals stay; LLM gets steered message +
      drained completions on next iteration.
- [ ] **G71**: Cooperative cancellation — async bodies use
      `asyncio.sleep(0)` checkpoints; sync bodies non-preemptable
      (documented).
- [ ] **G74**: `kind` set explicitly at every task-row creation
      site (Phase 2 covers `kind="tool"` in `background_tool_workflow`;
      `kind="agent_task"` set on user-turn task creation).
- [ ] **G79**: Per-kind timeout policy — sync `@tool` keeps
      existing `tools.timeout`; async tools have no default
      timeout (use `cancel_task` for control).
- [ ] **G86**: Cancel emits `async_work_complete` signal so the
      parent's drain wakes and removes the task from
      `pending_tasks`.
- [ ] **G91**: Use `created_at`, not `started_at`. (Phase 1
      already addressed this in the doc; verify the new code
      follows the same convention.)

## Tests

### Unit — `tests/db/test_migration_task_kind.py` (new)

- [ ] `test_migration_adds_kind_column_with_default` — after
      migration, all pre-existing task rows have
      `kind="agent_task"`.
- [ ] `test_migration_is_reversible` — downgrade removes the
      column cleanly.

### Unit — `tests/tools/test_tool_decorator.py` (extend Phase 1's file)

- [ ] `test_decorator_synchronous_false_marks_async`.
- [ ] `test_decorator_synchronous_false_return_type_documented` —
      asserts the LLM-facing schema's `message` field includes
      a return description indicating a task handle.

### Unit — `tests/runtime/test_background_tool_workflow.py` (new)

- [ ] `test_background_workflow_calls_step_and_signals` — spawn
      workflow with mock step; assert `DBOS.send` called once
      with the correct payload shape.
- [ ] `test_background_workflow_signals_failure_with_truncated_trace` —
      step raises; assert signaled payload has
      `status: "failed"`, `error_message`, and a `traceback`
      field ≤ 30 frames.
- [ ] `test_background_workflow_truncates_result_over_10k` —
      step returns 20k-char string; assert signaled
      `result.output` is ≤ 10k chars + truncation marker; assert
      full value stored in task output.

### Unit — `tests/tools/builtins/test_task_lifecycle.py` (new)

- [ ] `test_check_task_running_returns_status_and_activity` —
      running task; assert `recent_activity` contains the last N
      items as real items (not MagicMocks).
- [ ] `test_check_task_completed_returns_result` — terminal
      task; assert `result` field has the real output.
- [ ] `test_check_task_failed_returns_truncated_trace`.
- [ ] `test_cancel_task_sets_state_and_sends_signal`.
- [ ] `test_list_tasks_filters_by_status` — parametrized over
      `"running"`, `"completed"`, `"all"`.
- [ ] `test_list_tasks_excludes_agent_task_kind` (G57).
- [ ] `test_check_task_rejects_agent_task` (G57).
- [ ] `test_check_task_rejects_cross_conversation_task` (G23).

### Server integration — `tests/server/integration/test_async_tool_integration.py` (new)

All concurrency tests use blocked mock tool + sync gate +
deterministic release per the testing skill.

- [ ] `test_async_tool_returns_handle_immediately`.
- [ ] `test_async_tool_auto_delivers_between_iterations`.
- [ ] `test_async_tool_auto_collect_at_turn_end`.
- [ ] `test_async_tool_error_surfaces_truncated_trace`.
- [ ] `test_async_tool_result_truncated_to_10k_chars`.
- [ ] `test_check_task_via_tool_call`.
- [ ] `test_cancel_task_during_execution` — concurrency.
- [ ] `test_list_tasks_filter_running_only_returns_active`.
- [ ] `test_parent_cancel_propagates_non_blocking` —
      concurrency: cancel returns < some threshold while child
      is blocked, then children transition.
- [ ] `test_parallel_async_tool_spawns` — 3 parallel
      `@tool(synchronous=False)` calls; all three children
      complete; all three results auto-deliver.
- [ ] `test_crash_during_auto_collect_recovers` — kill parent
      mid-`dbos_recv_async`; on restart, replays and resolves.

### E2E — `tests/e2e/test_async_tool_e2e.py` (new)

Per the design doc's robust E2E plan:

- [ ] Test fixture agent at
      `tests/_fixtures/agents/async-test/` with two
      `@tool`-decorated functions (`delayed_echo` async and
      `count_files` sync) plus a `boom` failing tool.
- [ ] `test_multiple_async_tools_different_durations_e2e`.
- [ ] `test_mixed_sync_and_async_tools_e2e`.
- [ ] `test_async_tool_failure_surfaces_e2e`.
- [ ] `test_async_tool_user_cancel_propagates_e2e`.

## Test infrastructure (new helpers built in this phase)

- [ ] **MockTool** — `tests/server/integration/mock_tool.py` with
      `add_call(block=True)` / `call.call_event.wait()` /
      `call.release()` API mirroring `mock_llm.add_call`. Required
      for every concurrency test in Phase 2+.
- [ ] **CapturedEvent + collector** — `tests/server/integration/helpers.py`
      gets a `CapturedEvent` dataclass + an SSE event collector
      helper (per testing skill rule #20).

## Migration / housekeeping

- [ ] **M1**: All Phase 1 sync `@tool` invocations continue to
      work unchanged (regression coverage via existing tests).
- [ ] **M2**: `agent_plane/runtime/workflow.py` updated for
      drain + auto-collect rewrite.
- [ ] **M3**: Old auto-collect path (lines around 4091–4122 in
      `workflow.py`) replaced/extended for the new lifecycle —
      no dual code path.

## Closing checks

- [ ] All Phase 2 unit tests pass.
- [ ] All Phase 2 server integration tests pass (real DBOS, real
      task_store, real signals — no pure-mock DBOS).
- [ ] No new failures introduced into the rest of the test
      suite (verified vs. Phase-1 baseline).
- [ ] TUI manual verification documented (heartbeat events fire
      during long waits; auto-delivered messages render distinctly).
- [ ] Post-change review subagent invoked with the augmented
      Phase 2 prompt (read this checklist, verify each item is
      honored in the diff, report ✅/❌ per item).
