# POLICIES Integration Guide

Step-by-step recipe for wiring the already-shipped policy
system into `_run_agent_loop`. Phase 6 of the implementation
plan. This is a design document for the implementer — every
code snippet below has a passing test in
`tests/runtime/policies/test_four_phase_contract.py` or
`tests/runtime/policies/test_ask_cycle_e2e.py` that pins the
expected shape.

## Scope

**What this guide covers:**
- Building a `PolicyEngine` at the top of `_run_agent_loop`.
- Wiring `_enforce_policy` at each of the 4 enforcement sites.
- Wiring `_await_policy_approval` on ASK results.
- Handling DENY / ALLOW / ASK branching at each site.

**Out of scope** (separate phases):
- Phase 9 — connecting `PromptPolicy._default_classifier` to
  the real LLM executor.
- Phase 9 — Claude SDK hooks + AgentsSdk MCP subclass.
- Phase 10 — client-side reserved-name handling.
- Phase 11 — observability spans.

## Step 1 — Build the engine

At the top of `_run_agent_loop`, alongside `tool_mgr`,
`executor`, `compaction_state`:

```python
from agent_plane.runtime.policies import build_policy_engine

policy_engine = build_policy_engine(
    spec=spec,
    conversation_id=conversation_id,
    conversation_store=conv_store,
)
```

Pinned by: `tests/runtime/policies/test_builder.py`.

The engine is a plain local, not a ContextVar (POLICIES.md §4).
Pass it explicitly to any helper that needs it.

## Step 2 — INPUT phase

After `_sync_history` surfaces new user messages, BEFORE
`_executor_turn_with_compaction`:

```python
from agent_plane.runtime.policies import _enforce_policy
from agent_plane.spec.types import (
    EvaluationContext,
    Phase,
    PolicyAction,
)

for new_user_message in newly_surfaced_messages:
    result = await _enforce_policy(
        policy_engine,
        EvaluationContext(
            phase=Phase.INPUT,
            content=new_user_message.content_text,
            tool_name=None,
        ),
    )
    if result.action == PolicyAction.DENY:
        # Replace the user message with a sentinel; the LLM
        # sees the sentinel in history on the next turn.
        _replace_with_sentinel(new_user_message, result.reason)
    elif result.action == PolicyAction.ASK:
        # Park for approval (Step 5 below).
        ...
    # PolicyAction.ALLOW — no-op, proceed normally.
```

Pinned by: `test_four_phase_contract.py::test_input_phase_*`
+ `test_enforcement_integration.py::test_policies_demo_*`.

## Step 3 — TOOL_CALL phase

Inside `_call_tool`, BEFORE dispatch:

```python
result = await _enforce_policy(
    policy_engine,
    EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"tool": tool_name, "args": arguments},
        tool_name=tool_name,
    ),
)
if result.action == PolicyAction.DENY:
    # Short-circuit — return the blocked sentinel as the
    # tool's output so the LLM sees it in history.
    return {"blocked": True, "reason": result.reason}
elif result.action == PolicyAction.ASK:
    approved = await _handle_ask(
        policy_engine, result, Phase.TOOL_CALL, _preview(arguments),
    )
    if not approved:
        return {"blocked": True, "reason": result.reason}
# ALLOW — dispatch normally.
```

Pinned by: `test_four_phase_contract.py::test_tool_call_*`
+ `test_enforcement_integration.py::test_*_allow_*` /
`test_*_deny_*`.

## Step 4 — TOOL_RESULT phase

Inside `_call_tool`, AFTER dispatch, BEFORE the result is
surfaced to the executor or persisted:

```python
result = await _enforce_policy(
    policy_engine,
    EvaluationContext(
        phase=Phase.TOOL_RESULT,
        content={"output": tool_output_string},
        tool_name=tool_name,  # same name as the tool_call step
    ),
)
if result.action == PolicyAction.DENY:
    return {"blocked": True, "reason": result.reason}
elif result.action == PolicyAction.ASK:
    approved = await _handle_ask(
        policy_engine, result, Phase.TOOL_RESULT,
        _preview(tool_output_string),
    )
    if not approved:
        return {"blocked": True, "reason": result.reason}
# ALLOW — surface / persist the result normally.
```

The `tool_name` MUST match what was passed at the TOOL_CALL
step — LabelPolicy taint-on-tool-result cases rely on this
(e.g. `taint_web_search` on `tool_call:web_search`). The
caller already has the name in scope from the dispatch.

Pinned by:
`test_four_phase_contract.py::test_tool_result_*`.

## Step 5 — OUTPUT phase

In `_handle_final_response`, AFTER the executor emits the
final assistant text, BEFORE persistence:

```python
result = await _enforce_policy(
    policy_engine,
    EvaluationContext(
        phase=Phase.OUTPUT,
        content=response_text,
        tool_name=None,
    ),
)
if result.action == PolicyAction.DENY:
    # Replace the response BEFORE persist — the raw text
    # never hits conversation_items (POLICIES.md §11.4).
    response_text = f"[DENIED by policy: {result.reason}]"
elif result.action == PolicyAction.ASK:
    approved = await _handle_ask(
        policy_engine, result, Phase.OUTPUT, _preview(response_text),
    )
    if not approved:
        response_text = f"[DENIED by user]"
# Persist `response_text` as the final assistant message.
```

**Load-bearing pre-persistence ordering.** POLICIES.md §11.4:
the raw assistant text must never reach `conversation_items`
when OUTPUT policy DENYs — otherwise compaction could resurface
blocked content. Pinned by
`test_four_phase_contract.py::test_output_phase_deny`.

## Step 6 — ASK handling

The `_handle_ask` helper wraps `_await_policy_approval` with
real DBOS + SSE callbacks:

```python
from agent_plane.runtime.policies import _await_policy_approval
from agent_plane.spec.types import Phase, PolicyResult

async def _handle_ask(
    engine: PolicyEngine,
    result: PolicyResult,
    phase: Phase,
    content_preview: str,
) -> bool:
    """Park, wait for PATCH verdict, return True on approve."""

    def _register(call_id: str, task_id_inner: str, args_json: str) -> None:
        # Insert into pending_tool_calls via task_store.
        task_store.create_pending_tool_call(
            call_id=call_id,
            root_task_id=root_task_id or task_id,
            task_id=task_id_inner,
            tool_name="request_approval",
            arguments=args_json,
        )

    def _emit(event: dict[str, Any]) -> None:
        # Publish on the root task's SSE stream — the client
        # with reserved-name handling renders an approval UI.
        _write_output(root_task_id or task_id, event)

    async def _park(call_id: str, timeout_s: int) -> str | None:
        # Park on the existing tool_result DBOS topic —
        # same machinery as client-side tool tunneling.
        try:
            await dbos_recv_async(
                topic="tool_result", timeout_seconds=timeout_s,
            )
        except TimeoutError:
            raise
        # Fetch the completed row and return its result
        # string.
        pending = task_store.list_pending_tool_calls(
            call_id=call_id, status="completed",
        )
        if not pending:
            return None  # cancelled or missing
        return pending[0].result

    return await _await_policy_approval(
        task_id=task_id,
        root_task_id=root_task_id or task_id,
        result=result,
        phase=phase,
        content_preview=content_preview,
        policy_engine=engine,
        register=_register,
        emit=_emit,
        park=_park,
    )
```

Pinned by
`tests/runtime/policies/test_ask_cycle_e2e.py` and
`test_approval.py::test_approval_*`.

## Step 7 — PATCH route handler (cancel during ASK)

When the user cancels a response while the workflow is
parked:

```python
# Inside POST /v1/responses/{id}/cancel handler:
pending = task_store.list_pending_tool_calls(
    task_id=task_id, status="action_required",
)
for row in pending:
    if row.tool_name == "request_approval":
        # Mark as cancelled and wake the parked workflow
        # with a None verdict → _parse_verdict returns False
        # → helper returns False → caller sees DENY.
        task_store.update_pending_tool_call_status(
            row.call_id, "cancelled",
        )
        DBOS.send(task_id, None, topic="tool_result")
```

Pinned by
`test_approval.py::test_approval_missing_verdict_row_denies`.

## Invariants the wiring MUST preserve

Every test in `tests/runtime/policies/` is a contract the
wiring must honor. The load-bearing ones:

1. **Label writes apply on approve ONLY**
   (`test_ask_cycle_refuse_drops_labels`): a denied ASK
   leaves no trace in the store.
2. **OUTPUT DENY is pre-persistence**
   (`test_output_phase_deny`): raw blocked text never hits
   `conversation_items`.
3. **tool_name flows through to TOOL_RESULT**
   (`test_tool_result_*`): the caller must pass the same
   name at both sites so selectors match.
4. **Cross-conversation isolation**
   (`test_conversation_isolation.py`): every engine is
   scoped to one `conversation_id`.
5. **Per-policy ask_timeout wins over spec-level**
   (`test_per_policy_ask_timeout_override_wins`): the park
   callback must receive the right timeout for the
   deciding policy.
6. **Labels survive workflow restart**
   (`test_taint_persists_across_workflow_restarts`):
   rebuilds read the store snapshot, don't reset state.
7. **Monotonic constraints enforced post-approve**
   (`test_ask_approve_respects_monotonic_drop`):
   schema-valid-filter runs on ASK-approved writes too.

## Verification

After wiring, run:

```bash
python -m pytest tests/runtime/policies/ -q
```

**If this suite goes red**, stop. The wiring broke a
contract. Diagnose via the failing test's docstring — each
one names the specific invariant it guards.

Then run the e2e suite via the terminal TUI (per
`CLAUDE.md` Mandatory TUI Verification) against the three
fixture agents:

```bash
python examples/frontends/terminal.py tests/_fixtures/agents/secure-research/
python examples/frontends/terminal.py tests/_fixtures/agents/rate-limited-search/
python examples/frontends/terminal.py tests/_fixtures/agents/policies-demo/
```

Confirm each agent's policy-gated tool calls behave as the
fixture YAML declares.
