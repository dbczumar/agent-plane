# Phase 3 Adherence Checklist

Source: `/home/ubuntu/session_model_notes.md` §8 Phase 3.

Phase 3 = Async sub-agents (no persistence yet) + unification of
the sub-agent lifecycle with Phase 2's drain channel. Mark `[x]`
when the corresponding code + test has landed and the test passes.

---

## Core deliverables

- [ ] **D1**: New builtin `spawn_sub_agent(type, input)` —
      singular (one call per sub-agent), no `name` param.
      Creates a fresh anonymous child conversation
      (`title=None`, `parent_conversation_id=None` — Phase 3
      has no persistence). Starts a DBOS sub-agent workflow
      pinned to the new task_id (G56). Returns the same
      `_AsyncToolHandle` JSON shape as Phase 2:
      `{task_id, kind: "sub_agent", type, status, message}`.

- [ ] **D2**: Sub-agent workflow signals completion via the
      unified `async_work_complete` topic. Replaces Phase 2's
      polling-based `_auto_collect_sub_agents` path. Handler
      lives at the parent's existing
      `_drain_async_completions` so the per-iteration drain
      and end-of-turn wait pick up sub-agent results
      identically to `kind="tool"` results.

- [ ] **D3**: Delete `spawn_sub_agents` (batch),
      `check_sub_agents`, `cancel_sub_agent` from
      `agent_plane/tools/builtins/spawn.py`. Delete the
      polling auto-collect branch in
      `agent_plane/runtime/workflow.py` (the
      `_auto_collect_sub_agents` block around the
      `uncollected = spawned_ids - collected_ids` check).
      No dual code path (M3).

- [ ] **D4**: Migrate every `examples/agents/*/AGENTS.md` and
      `examples/agents/*/config.yaml` that references the
      removed tool names. Loud failure at agent load if any
      stale reference remains. Inventory:
      ```
      grep -r "spawn_sub_agents\|check_sub_agents\|cancel_sub_agent" \\
          examples/agents/*/AGENTS.md examples/agents/*/config.yaml
      ```

- [ ] **D5**: `check_task` / `cancel_task` / `list_tasks`
      builtins (Phase 2) work on `kind="sub_agent"` handles
      with no further code change. Verify by integration test
      — already aligned by Phase 2's kind discriminator.

## G-decisions applicable to Phase 3

- [ ] **G3**: Cancel/completion race for sub-agents — same
      first-write-wins on terminal status as Phase 2's tool
      tasks (sub-agent workflow's BaseException handler must
      `await _send_payload` before re-raising, mirroring
      `background_tool_workflow`).
- [ ] **G7**: Auto-collect under cancel — strict cancel: no
      further LLM call; already-arrived completions persisted.
- [ ] **G12**: `message` field on sub-agent handles names the
      task_id and points the LLM at `check_task` /
      `cancel_task`.
- [ ] **G19**: Drain protocol — same `dbos_recv_async` loop
      consumes both `kind="tool"` and `kind="sub_agent"`
      payloads. No new topic.
- [ ] **G23**: Task access control — sub-agent task lookups
      scoped to the caller's conversation tree.
- [ ] **G55**: `pending_tasks` reconstruction — sub-agent
      handles recovered from history alongside async-tool
      handles via `task_store.list_tasks(root_task_id=...)`.
- [ ] **G56**: `task_id` ≡ `workflow_id` for sub-agents too.
- [ ] **G57**: `list_tasks` excludes `kind="agent_task"`
      (top-level user turns) but includes both
      `kind="sub_agent"` and `kind="tool"`.
- [ ] **G74**: `kind="sub_agent"` set explicitly by
      `spawn_sub_agent` at task-row creation
      (already landed in Phase 2 part 9; verify still wired).
- [ ] **G86**: Cancel emits `async_work_complete` with
      `status="cancelled"` so the parent's drain wakes and
      removes the sub-agent from `pending_tasks`.

## Tests

### Unit — `tests/spec/test_subagent_cycle_detection.py` (new)

- [ ] `test_self_declaration_rejected` — agent A's
      `agents: [A]` fails to load with a cycle error naming A.
- [ ] `test_two_node_cycle_rejected` — A↔B cycle.
- [ ] `test_three_node_cycle_rejected` — A→B→C→A.
- [ ] `test_dag_no_cycle_loads` — diamond, no cycle.
- [ ] `test_disjoint_subgraphs_no_cycle` — two isolated trees.

### Unit — `tests/tools/builtins/test_spawn_sub_agent.py` (replaces `test_spawn.py`)

- [ ] `test_spawn_creates_child_conversation_anonymous`.
- [ ] `test_spawn_starts_dbos_workflow_with_correct_args`.
- [ ] `test_spawn_returns_handle_shape`.
- [ ] `test_spawn_unknown_type_raises_unknown_subagent`.

### Unit — `tests/runtime/test_sub_agent_signaling.py` (new)

- [ ] `test_sub_agent_workflow_signals_parent_on_success`.
- [ ] `test_sub_agent_workflow_signals_parent_on_failure_with_trace`.
- [ ] `test_sub_agent_workflow_signals_on_cancel`.

### Server integration — `tests/server/integration/test_sub_agent_integration.py` (replaces old)

- [ ] `test_spawn_sub_agent_auto_delivers_result`.
- [ ] `test_spawn_sub_agent_auto_collect_at_turn_end`.
- [ ] `test_multiple_parallel_sub_agents`.
- [ ] `test_sub_agent_error_surfaces_to_parent`.
- [ ] `test_cancel_sub_agent_via_cancel_task` — concurrency.
- [ ] `test_parent_cancel_propagates_to_sub_agents_non_blocking` — concurrency.
- [ ] `test_check_task_on_sub_agent_returns_recent_items`.
- [ ] `test_old_spawn_sub_agents_tool_removed`.
- [ ] `test_check_sub_agents_tool_removed`.
- [ ] `test_cancel_sub_agent_tool_removed`.

### E2E — `tests/e2e/test_sub_agent_phase3_e2e.py` (new)

- [ ] `test_parallel_sub_agents_e2e`.
- [ ] `test_sub_agent_failure_surfaces_to_parent_e2e`.
- [ ] `test_sub_agent_uses_its_own_tools_e2e`.
- [ ] `test_mixed_sub_agent_and_async_tool_e2e`.
- [ ] `test_sub_agent_user_cancel_propagates_e2e`.

### E2E — update existing

- [ ] `tests/e2e/test_coder_subagent.py` — replace batch
      `spawn_sub_agents` calls with singular
      `spawn_sub_agent` in the test agent's prompt.
- [ ] `tests/e2e/test_claude_coder_subagent.py` — same.

## Migration / housekeeping

- [ ] **M1**: Phase 2 `@tool(synchronous=False)` paths
      continue to work unchanged (regression coverage via
      `tests/server/integration/test_async_tool_integration.py`
      and `tests/e2e/test_async_tools_e2e.py`).
- [ ] **M2**: `agent_plane/runtime/workflow.py` — old
      `uncollected = spawned_ids - collected_ids` block and
      its dependencies (`_auto_collect_sub_agents`,
      `_persist_text_before_auto_collect`,
      `_recover_spawn_ids_from_history`,
      `_inject_collect_results`,
      `_AutoCollectResult`, `_check_terminal`) deleted.
- [ ] **M3**: Old `agent_plane/tools/builtins/spawn.py`
      file deleted (or contents replaced) — no backwards-
      compat shim per CLAUDE.md.
- [ ] **M4**: Every `examples/agents/*/AGENTS.md` and
      `config.yaml` migrated. Grep clean for the deleted
      tool names.

## Closing checks

- [ ] All Phase 3 unit tests pass.
- [ ] All Phase 3 server integration tests pass.
- [ ] Phase 2 test suite still green (no regressions).
- [ ] E2E suite (`pytest tests/e2e/ --llm-api-key
      $LLM_API_KEY -v`) fully green, including the migrated
      coder/claude-coder sub-agent tests.
- [ ] TUI manual verification documented for each updated
      agent (`archer`, `coder`, `claude-coder`).
- [ ] Post-change review subagent invoked with the
      Phase 3 prompt.
