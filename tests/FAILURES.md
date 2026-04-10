# Test Failures (31 on main)

## Root Cause 1: Ghost assistant messages before tool calls (6 failures)

The workflow now persists an empty assistant message before `function_call`
items (fix for 400 error when replaying `web_search_call` + `function_call`).
Tests expect the old item count without this extra message.

- `test_routes_conversations.py::test_tool_call_items_position_order`
- `test_routes_conversations.py::test_multiple_tool_call_rounds_ordering`
- `test_routes_conversations.py::test_steering_position_among_tool_items`
- `test_routes_conversations.py::test_client_side_tool_completes_with_function_call`
- `test_mcp_integration.py::test_mcp_list_directory_returns_real_files`
- `test_mcp_integration.py::test_mcp_read_file_returns_real_content` / `test_mcp_multi_tool_round_trip`

**Fix:** Update expected item counts and index offsets in these tests.

## Root Cause 2: `_check_steering_inbox` gained `extra_own_ids` kwarg (1 failure)

- `test_concurrency.py::test_steering_between_persist_and_close_inbox`

**Fix:** Update the test's mock `check_then_inject` to accept `extra_own_ids`.

## Root Cause 3: `call_tool_with_timeout` signature changed (3 failures)

- `test_tool_retry.py::test_execute_tool_with_retry_retries_on_timeout`
- `test_tool_retry.py::test_tool_retry_event_has_correct_fields`
- `test_tool_retry.py::test_tool_retry_non_timeout_exception_no_retry`

**Fix:** Update test lambdas to accept 3 args `(call_fn, timeout, cancel_fn)`.

## Root Cause 4: `_has_session_transcript` path changed (1 failure)

- `test_claude.py::test_build_prompt_with_transcript_returns_latest_message`

**Fix:** Create `.claude/` under `tmp_path / "workspace"` instead of `tmp_path`.

## Root Cause 5: `SandboxConfig.enabled` removed (3 failures)

- `test_local.py::test_build_command_with_srt`
- `test_local.py::test_build_command_srt_disabled`
- `test_local.py::test_build_command_uv_outside_srt_inside`

**Fix:** Remove `enabled=True/False` from `SandboxConfig()` construction.
Use the runtime `sandbox_enabled` parameter instead.

## Root Cause 6: Tool name max length 64 → 256 (3 failures)

- `test_manager.py::test_mcp_tool_invalid_name_skipped[too_long]`
- `test_manager.py::test_client_tool_invalid_name_raises[too_long]`
- `test_client_specified.py::test_parse_rejects_invalid_tool_name[too_long]`

**Fix:** Change test data from `"a" * 65` to `"a" * 257`.

## Root Cause 7: `_build_await_tool_output` now async (1 failure)

- `test_workflow.py::test_build_await_tool_output_end_to_end`

**Fix:** Await the callback in an async context or use `asyncio.run()`.

## Root Causes 8/9: DBOS async workflow + mock LLM interaction (6 remaining failures)

- `test_durability.py::test_workflow_recovers_after_server_restart`
- `test_durability.py::test_incomplete_step_reexecutes_after_crash`
- `test_durability.py::test_steered_messages_survive_crash`

The mock LLM client is correctly patched in both `workflow._get_llm_client`
and `executors.default._get_llm_client`. The mock's `create()` is async
and returns proper async generators. But when DBOS runs the async workflow
via `_execute_workflow_async`, the mock's async generator encounters
`object list_iterator can't be used in 'await' expression` — suggesting
DBOS's internal async scheduling doesn't fully support async generators
returned from mocked coroutines.

The durability tests additionally hit DBOS lifecycle errors (`No DBOS was
created yet`) when destroying and recreating DBOS between crash simulations.

- `test_durability.py::test_workflow_recovers_after_server_restart`
- `test_durability.py::test_incomplete_step_reexecutes_after_crash`
- `test_durability.py::test_steered_messages_survive_crash`
- `test_task_store.py::test_stream_closed_on_workflow_exception`
- `test_task_store.py::test_persist_first_prevents_ghost_tokens`

**Fix:** Needs investigation of DBOS async workflow thread pool interaction
with mock async generators. May require the mock to return a different
async iterable type, or DBOS configuration changes.

## Root Cause 10: MCP schema normalization changed (2 failures)

- `test_mcp.py::test_normalize_warns_on_ref`
- `test_mcp.py::test_normalize_warns_on_oneof`

**Fix:** Update assertions to match new normalization behavior.

## Root Cause 11: MCP class structure changed (2 failures)

- `test_mcp.py::test_circuit_breaker_trip_log_includes_server_name`
- `test_mcp.py::test_event_loop_thread_stop_logs_warning_on_join_timeout`

**Fix:** Update mocks to match current MCP class attributes.

## Root Cause 12: Spawn recovery hangs (1 failure) ⚠️ CRITICAL

- `test_routes_responses.py::test_spawn_recovery_across_client_tool_boundary`

Async refactor created a deadlock in the spawn recovery + client-tool
polling path. Test hangs past 120s timeout.

**Fix:** Debug the async deadlock in the spawn recovery path.
