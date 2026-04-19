# Phase 4 Adherence Checklist

Source: `/home/ubuntu/session_model_notes.md` §8 Phase 4.

Phase 4 = Sub-agent persistence (named, continuation). Every
spawn now requires a name; later turns reuse the named child
via `send_to_sub_agent`; LLMs see an ambient hint of open
sub-agents + running tasks. Mark `[x]` when the corresponding
code + test has landed and the test passes.

---

## Core deliverables

- [x] **D1**: Alembic migration adds `parent_conversation_id`
      nullable column to `conversations` (FK with ON DELETE
      CASCADE) plus a partial unique index on
      `(parent_conversation_id, title)` WHERE
      `parent_conversation_id IS NOT NULL`. Folded into the
      original initial migration since no users yet.
- [x] **D2**: `ConversationStore.create_conversation` gains
      `title=` and `parent_conversation_id=` kwargs.
      `list_conversations` gains a `parent_conversation_id=`
      filter (cleaner than a dedicated `list_children` per the
      anti-pattern about specialized list+action methods).
      `NameAlreadyExistsError` translates the partial unique
      index violation.
- [x] **D3**: `spawn_sub_agent(type, name, input)` — `name`
      now required. Child conversation gets
      `title="<type>:<name>"` + `parent_conversation_id=<caller's
      conversation>`. Returns
      `{task_id, kind: "sub_agent", type, name, status, message}`
      handle JSON or `{"error": "name_already_exists", ...}`.
- [x] **D4**: New builtin `send_to_sub_agent(type, name, input)`
      — strict continue-only. Looks up child by `(parent, title)`,
      appends user message, creates new task on existing
      conversation. Errors `sub_agent_not_found` or
      `sub_agent_busy`.
- [x] **D5**: New builtin `list_sub_agents()` — returns
      `{sub_agents: [{type, name}, ...]}`.
- [x] **D6**: Ambient hint at turn start — every iteration's
      LLM call's system instructions get a compact "Open
      sub-agents:" + "Running tasks:" block appended.
- [ ] **D7**: Clean up Phase 3's anonymous-spawn path. Phase 4
      makes name required so the anonymous code path is
      already gone via the schema requirement; verify no
      remaining call site treats `name` as optional. *(Done in
      D3 — schema requires it; checked via integration tests.)*

## G-decisions applicable to Phase 4

- [x] **G36**: Parallel-spawn race — partial unique index on
      `(parent_conversation_id, title)` rejects duplicate
      `(type, name)` at the DB layer. Translated to
      `NameAlreadyExistsError` → `name_already_exists` tool
      result.
- [ ] **G55**: `pending_tasks` reconstruction on workflow start
      — Phase 4 doesn't change this; sub-agents flow through
      Phase 2/3 drain.

## Tests

### Unit — `tests/db/` (existing migration tests)

- [x] Migration tests still pass (9 tests, 1.4s).

### Unit — `tests/stores/test_conversation_store.py` (extended in part 2)

- [x] `test_create_conversation_with_parent_pointer_and_title` —
      round-trip parent + title.
- [x] `test_create_duplicate_title_under_same_parent_raises` —
      G36 partial unique index rejection.
- [x] `test_create_same_title_under_different_parents_succeeds` —
      per-parent uniqueness.
- [x] `test_create_null_parent_allows_duplicate_titles` — top-
      level conversations exempt.
- [x] `test_list_conversations_filtered_by_parent_returns_children_only`.
- [x] `test_cascade_delete_removes_descendants`.

### Unit — `tests/tools/builtins/test_spawn_sub_agent.py` (new)

- [ ] `test_spawn_returns_handle_with_type_name_taskid`.
- [ ] `test_spawn_missing_name_returns_error`.
- [ ] `test_spawn_unknown_type_returns_error`.
- [ ] `test_handle_message_names_send_to_sub_agent`.

### Unit — `tests/tools/builtins/test_send_to_sub_agent.py` (new)

- [ ] `test_send_missing_name_returns_error`.
- [ ] `test_send_unknown_type_returns_error`.

### Unit — `tests/tools/builtins/test_list_sub_agents.py` (new)

- [ ] `test_schema_takes_no_parameters`.
- [ ] `test_invoke_with_no_children_returns_empty_list`.

### Server integration — `tests/server/integration/test_sub_agent_persistence.py` (new)

- [ ] `test_persistence_across_turns` — turn 1 spawns
      `coder:auth`; turn 2 sends to same name; child's loaded
      history contains both messages.
- [ ] `test_spawn_duplicate_name_fails_loud_to_llm` — second
      spawn with same `(type, name)` returns
      `name_already_exists`.
- [ ] `test_send_to_missing_sub_agent_fails_loud_to_llm` —
      send to a name that was never spawned returns
      `sub_agent_not_found`.
- [ ] `test_send_to_busy_sub_agent_errors` — concurrency:
      blocked mock child LLM, second send to same name returns
      `sub_agent_busy`.
- [ ] `test_list_sub_agents_returns_children` — spawn 2 named
      sub-agents; list returns both.
- [ ] `test_ambient_hint_injected_at_turn_start` — verify the
      first LLM call's input contains the hint block.
- [ ] `test_cross_parent_name_isolation` — two conversations,
      both spawn `(coder, auth)` — both succeed (per-parent
      uniqueness).

### E2E — `tests/e2e/test_named_sub_agent_persistence.py` (new)

- [ ] `test_named_collaboration_e2e` — turn 1 spawns
      `coder:auth`; turn 2 sends to that name with continuation
      task; child's response cites prior turn's content.
- [ ] `test_parallel_named_sub_agents_e2e` — two distinct
      named sub-agents in one turn, both markers in final.
- [ ] `test_ambient_hint_usage_e2e` — turn 2 user prompt is
      neutral; LLM picks up the named sub-agent from the
      ambient hint and emits send_to_sub_agent.
- [ ] `test_cross_parent_named_isolation_e2e` — same name in
      two top-level conversations, no leakage.

### Migration / housekeeping

- [ ] **M1**: Phase 2 + Phase 3 paths continue to work
      (regression coverage via existing test suites).
- [ ] **M2**: `web_fetch.py` updated for the new `_spawn_one`
      signature (done — see commit ed2f8b9).

## Closing checks

- [ ] All Phase 4 unit tests pass.
- [ ] All Phase 4 server integration tests pass.
- [ ] Phase 2 + Phase 3 test suites still green.
- [ ] E2E suite (`pytest tests/e2e/ --llm-api-key
      $LLM_API_KEY -v`) fully green.
- [ ] TUI manual verification — deferred (no interactive
      session available in autonomous mode).
- [ ] Post-change review subagent invoked per part.
