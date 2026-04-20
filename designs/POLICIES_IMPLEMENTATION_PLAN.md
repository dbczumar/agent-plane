# POLICIES Implementation Plan

Phased plan for implementing `designs/POLICIES.md`. Each phase
is self-contained, independently testable, and leaves the
codebase shippable. Phases may be merged individually; the
system is only feature-complete after Phase 11.

## Implementation progress

| Phase | Status | Branch | Tests added |
|---|---|---|---|
| 0 — Spec types + parser | ✅ landed | `policies-phase-0-spec-types` | 59 |
| 1 — conversation_labels + store API | ✅ landed | `policies-phase-0-spec-types` (continued) | 15 |
| 2 — PolicyEngine skeleton + unified builtin registry | ✅ landed | `policies-phase-2-engine-skeleton` | 20 + 8 + 2 validator |
| 3 — LabelPolicy runtime + engine composition | ✅ landed | `policies-phase-3-label-policy` | 19 |
| 3+ — LabelDef schema validation (values + monotonic) | ✅ landed | `policies-phase-8-ask-flow` | 9 |
| 4 — FunctionPolicy + engine safety net | ✅ landed | `policies-phase-4-function-policy` | 19 |
| 5 — `_enforce_policy` API + 3 fixture agents | ✅ landed | `policies-phase-5-input-output-enforcement` | 14 |
| 5b — Combined-policies integration tests | ✅ landed | `policies-phase-5b-combined-integration` | 9 |
| 6 — Workflow enforcement-site wiring | ⏳ pending | — | — |
| 7 — PromptPolicy runtime | ✅ landed | `policies-phase-7-prompt-policy` | 20 |
| 7b — PromptPolicy integration tests | ✅ landed | `policies-phase-5b-combined-integration` | 6 |
| 8 — ASK flow helper (`_await_policy_approval`) | ✅ landed | `policies-phase-8-ask-flow` | 25 |
| 8b — ASK-cycle e2e composition | ✅ landed | `policies-phase-8-ask-flow` | 8 |
| 8c — Multi-turn engine-rebuild e2e | ✅ landed | `policies-phase-8-ask-flow` | 5 |
| 9 — Executor integration (Claude SDK, AgentsSdk, LLM wiring) | ⏳ pending | — | — |
| 10 — Client-side (REPL + Python UI SDK) | ⏳ pending | — | — |
| 11 — Observability spans + polish | ⏳ pending | — | — |

**Total new tests so far: ~240+** covering spec parsing +
validation, store persistence, engine composition, all three
policy subclasses, schema-validated label writes, ASK-flow
round-trip, multi-turn engine rebuild, and multi-type
integration through 4 ported / built fixtures (agent_with_policies,
rate_limited_search, secure_research, combined-policies,
prompt-policy-demo). Full non-e2e suite: 850+ passing.

**The policy system is feature-complete at the engine +
persistence + approval-helper level.** Real YAML specs flow
through parse → validate → build → evaluate → persist →
(optional approval round-trip) against real SQLAlchemy stores
with schema-validated label writes, strict verdict parsing,
and per-policy ask_timeout overrides. Every ported omniagents
example agent executes to the asserted behavior.

What remains to unlock a LIVE e2e run with a real agent
workflow:

- **Phase 6** — wire `_enforce_policy` into `_run_agent_loop`'s
  4 sites + wire `_await_policy_approval`'s register / emit /
  park seams to the real DBOS + SSE stack.
- **Phase 9** — wire `PromptPolicy._default_classifier` to the
  real LLM executor; wire Claude SDK `PreToolUse` / `PostToolUse`
  hooks + AgentsSdk MCP subclass to call `_enforce_policy`.
- **Phase 10** — REPL + Python UI SDK handle the
  `request_approval` reserved name.
- **Phase 11** — observability spans (policy_phase /
  policy_eval) + span events for dropped label writes.

Phases 6-11 are all additive — no changes needed to the
shipped engine, approval helper, spec types, or store layer.

---


## Omniagents test + example parity — load-bearing assurance

Agent-plane's policy system is a port of omniagents'. The
omniagents test suite already verifies every invariant this
design inherits. **Every agent-plane phase must include the
ported versions of the relevant omniagents tests**, adapted
to agent-plane's spec shape (YAML keys renamed per the port,
DBOS-durable workflow instead of in-memory Session, etc.)
but preserving the original assertion semantics.

### Reference test files

| omniagents file | lines | agent-plane phase |
|---|---|---|
| `tests/test_policies.py` | 320 | Phase 3 (LabelPolicy), Phase 4 (FunctionPolicy), Phase 7 (PromptPolicy) |
| `tests/test_labels_and_policies.py` | 1051 | Phase 2 (engine), Phase 3, Phase 4, Phase 7, Phase 8 (ASK flow) |
| `tests/test_label_examples.py` | 848 | Phase 8 + e2e (multi-dimensional IFC scenarios) |
| `tests/test_label_propagation.py` | 292 | deferred — v2 / G5 sub-agent propagation |

### Reference example YAMLs — ported as e2e fixtures

| omniagents example | agent-plane e2e fixture | exercises |
|---|---|---|
| `examples/agent_with_policies.yaml` | `tests/_fixtures/agents/policies-demo/` | FunctionPolicy + PromptPolicy (input + output phases) |
| `examples/rate_limited_search_agent.yaml` | `tests/_fixtures/agents/rate-limited-search/` | Stateful FunctionPolicy + ASK timeout |
| `examples/secure_research_agent.yaml` | `tests/_fixtures/agents/secure-research/` | Full IFC scenario — labels + label policies + enforcement |

Each phase's test-coverage list below calls out which
omniagents test cases it ports. When a test name appears in
both the omniagents file and the agent-plane phase, the
**assertion semantics must match**; only framework-specific
plumbing (store setup, spec loading) differs.

### E2E priority

Per user direction: **e2e tests take priority over unit
tests** when coverage trade-offs appear. A phase ships with
at least one e2e test exercising the ported omniagents
example YAML against a real agent-plane workflow, even if
some unit-level coverage is still pending. The e2e agents
live under `tests/_fixtures/agents/` (alongside existing
`sub-agent-test`, etc.) and run under `tests/e2e/` with the
`--llm-api-key` flag, matching agent-plane's existing e2e
pattern.

### Phase-closure check

Before closing a phase: `diff` the ported test list against
the corresponding omniagents test file. Any omniagents test
NOT in the ported list either (a) must be added, or (b)
needs a written justification for its exclusion (e.g. "tests
a v2 feature we defer").

---

## Design-adherence assurance — applies to every phase

Before closing any phase:

1. **Grep the design for every rule the phase touches.** Any
   invariant stated in POLICIES.md that the phase's code is
   responsible for enforcing must have a test that fails when
   the code doesn't enforce it.
2. **Run both review subagents** (anti-pattern + test
   authoring) per `CLAUDE.md`. Block on any BLOCKING finding.
3. **Run the mandatory TUI / REPL verification** (per
   `CLAUDE.md`) once Phase 5 lands; re-run at Phase 8 and
   Phase 10.
4. **Trust-model audit** (§2.5): every untrusted input
   introduced by the phase must cross exactly one validation
   step. Enumerate them in the phase's PR description.
5. **Trace a concrete YAML example end-to-end** — pick one
   from §3.1 covering the subset the phase implements, write
   an integration test that loads it, runs through the
   enforcement path, and asserts the design-specified
   outcome.

Adherence is verified at three checkpoints per phase: before
starting (does the plan match the design?), during (does the
code?), after (does the review confirm it?).

---

## Phase 0 — Foundation: spec types + parser

**Scope.** Pure data types and YAML loading. No runtime
behavior. No DB schema yet.

**Files** (✨ new, 📝 modified):
- ✨ `agent_plane/spec/types.py` (extend): `Phase`,
  `EvaluationContext`, `PolicyResult`, `PolicyAction`,
  `LabelDef`, `PhaseSelector`, `FunctionRef`, `PolicySpec`
  (base + 3 subclasses), `GuardrailsSpec`.
- 📝 `agent_plane/spec/parser.py`: `_parse_on`,
  `_parse_condition`, `_parse_action_list`,
  `_parse_writable_labels`, `_parse_function`,
  `_parse_label_defs`, `_parse_policy_spec`,
  `_parse_guardrails`. Wire `guardrails` into `AgentSpec`.
- 📝 `agent_plane/spec/validator.py`: cross-field rules
  (§13 spec-load rejections) — empty `on: []`, `condition: {}`,
  `ask_timeout <= 0`, unknown phases, bad action values,
  `LabelDef` sanity, unresolvable `function.path`.

**Design-adherence invariants (POLICIES.md refs):**
- §3.2 shapes exactly reproduced (field names, optionality).
- §3.3 parser behavior — every YAML shape in §3.1 loads
  without error; every rejection in §13 fails loud.
- §14 condition-value coercion: scalars and list elements
  cast to `str`.

**Test coverage.**
- `tests/spec/test_policy_parser.py` — one test per YAML
  shape in §3.1 (short-form label, long-form label,
  schema'd label, function short, function dict with
  arguments, prompt, label-policy with condition).
- `tests/spec/test_policy_validator.py` — one test per
  rejection rule in §13: empty `on`, `condition: {}`,
  `ask_timeout <= 0`, `PromptPolicySpec.action` invalid
  value, dict-form function missing `path`, `LabelDef.initial`
  not in `values`.
- `tests/spec/test_condition_coercion.py` —
  `condition: {integrity: 0}` coerces to `"0"`;
  `condition: {roles: [admin, ops]}` preserves list with
  string elements.

**Adherence check.** Diff `spec/types.py` against the
dataclass blocks in §3.2 field-by-field. No extras, no
missing fields. Run parser tests; every §3.1 example
round-trips.

---

## Phase 1 — Storage: conversation_labels table + store API

**Scope.** DB schema, migration, store methods. No policy
runtime wired yet.

**Files:**
- ✨ `agent_plane/db/migrations/versions/<ts>_add_conversation_labels.py`:
  `CREATE TABLE conversation_labels (conversation_id,
  key, value, updated_at)`. PK on
  `(conversation_id, key)`. ON DELETE CASCADE to
  conversations.
- 📝 `agent_plane/entities/conversation.py`: add
  `labels: dict[str, str]` field.
- 📝 `agent_plane/stores/conversation_store/__init__.py`:
  abstract `set_labels(conv_id, updates, updated_at)`.
- 📝 `agent_plane/stores/conversation_store/sqlalchemy_store.py`:
  implementation with batched UPSERT;
  `INSERT ... ON CONFLICT DO NOTHING` helper for seeding;
  JOIN to populate `Conversation.labels` in `get()`.

**Design-adherence invariants:**
- §6.1 schema exactly.
- §6.3 compaction-surviving property: `conversation_items`
  writes/deletes do not touch `conversation_labels`.
- §10 Storage race-safe seeding (`INSERT ... ON CONFLICT`).

**Test coverage.**
- `tests/stores/conversation_store/test_set_labels.py`:
  CRUD; idempotent re-UPSERT; batched multi-key write in
  one transaction; JOIN loads labels on `get()`.
- `tests/stores/conversation_store/test_seed_labels.py`:
  concurrent seeding (two threads call the seeding helper,
  only one INSERT wins; both read same value).
- `tests/stores/conversation_store/test_labels_survive_compaction.py`:
  write labels, delete all conversation_items, verify
  labels persist.

**Adherence check.** Inspect migration SQL — column names and
types match §6.1 table exactly. Grep the store impl for
`UPDATE conversation_items` touching labels (should be zero
matches).

---

## Phase 2 — `PolicyEngine` skeleton + unified builtin registry

**Scope.** Engine class and build helper, WITHOUT any policy
subclasses yet. Unified builtin registry. All enforcement
sites remain absent.

**Files:**
- ✨ `agent_plane/runtime/policies/__init__.py`: public
  re-exports.
- ✨ `agent_plane/runtime/policies/engine.py`:
  `PolicyEngine` class per §4: `__init__`, `evaluate`
  (loops over `self.policies`, currently empty — still
  returns a composed `PolicyResult(ALLOW)`),
  `apply_label_writes`, `spec_for`, `_context`.
- ✨ `agent_plane/runtime/policies/builder.py`:
  `_build_policy_engine(spec, conversation_id)` — seeds
  initial labels via ON CONFLICT DO NOTHING; returns a
  no-op engine when `spec.guardrails is None`.
- 📝 `agent_plane/tools/builtins/__init__.py`: refactor to
  single `_BUILTIN_REGISTRY` dict with `"request_approval":
  None`; `BUILTIN_NAMES` and `INSTANTIABLE_BUILTINS`
  derived. Update `get_builtin_tool` to skip `None`
  factories.
- 📝 `agent_plane/tools/manager.py`: replace the
  `web_fetch` / `introspect` special cases with registry
  dispatch; reserve reject on `request_approval`.
- 📝 `agent_plane/spec/validator.py`: reject user-defined
  tool names that collide with `BUILTIN_NAMES`.
- 📝 `agent_plane/onboarding/agent/tools/python/list_builtin_tools.py`:
  derive from `INSTANTIABLE_BUILTINS`.

**Design-adherence invariants:**
- §4: engine is a plain class, not a ContextVar;
  `evaluate(ctx)` returns a `PolicyResult`.
- §15.8: single registry, `request_approval` has `None`
  factory, `ToolManager` rejects attempts to enable it.

**Test coverage.**
- `tests/runtime/policies/test_engine_empty.py`:
  construct engine with no policies → `evaluate(ctx)`
  returns ALLOW with empty `set_labels`;
  `deciding_policy=None`.
- `tests/runtime/policies/test_builder_seeding.py`:
  seeds declared initials on first run; no-ops when rows
  already exist.
- `tests/runtime/policies/test_builder_no_guardrails.py`:
  `spec.guardrails = None` → engine builds with empty
  policy list.
- `tests/tools/builtins/test_registry_unified.py`: single
  registry source; `BUILTIN_NAMES` matches registry keys;
  `request_approval` factory is `None`;
  `get_builtin_tool("request_approval")` → `None`.
- `tests/spec/test_validator_reserved_names.py`: attempt
  to declare a tool with a builtin name → fail-loud at
  spec load.

**Adherence check.** Grep `_BUILTIN_REGISTRY` — exactly one
assignment. Grep `web_fetch_special` / `introspect_special`
— zero matches. Run the existing onboarding list-tools test
with `request_approval` in scope — it must not appear in
the output.

---

## Phase 3 — `LabelPolicy` runtime (simplest)

**Scope.** LabelPolicy is pure YAML — no Python, no LLM.
Simplest policy type; exercises condition gating and
`set_labels` without external dependencies.

**Files:**
- ✨ `agent_plane/runtime/policies/base.py`: `Policy` ABC.
- ✨ `agent_plane/runtime/policies/label.py`: `LabelPolicy`
  per §9.3.
- 📝 `builder.py`: dispatch `LabelPolicySpec` → `LabelPolicy`.

**Design-adherence invariants:**
- §9.3: `LabelPolicy.evaluate` emits the declared action
  and `set_labels`, no more.
- §4 Step 2: condition gate runs before dispatch; non-
  matching policies emit no action.
- §10: label validation (values / monotonic) runs in
  `apply_label_writes`, not in the policy.

**Test coverage.**
- `tests/runtime/policies/test_label_policy.py` — ports the
  omniagents `test_labels_and_policies.py` cases that cover
  label policies specifically:
  `test_load_label_policy_produces_function_policy`,
  `test_label_policy_with_set_labels`,
  `test_label_set_by_policy_on_tool_call`,
  `test_condition_matches`, `test_condition_no_match`,
  `test_list_condition_or`, `test_multi_key_and`,
  `test_match_tools_filter`,
  `test_match_tools_ignored_for_non_tool_call`,
  `test_labels_change_future_policy_decisions`,
  `test_engine_enforces_root_label_schema_monotonicity`,
  `test_invalid_initial_label_value_rejected_by_schema`.
- `tests/runtime/policies/test_label_policy_composition.py`:
  DENY short-circuits; earlier ALLOWing writes still land
  on DENY; ASK accumulates and withholds labels (verify
  nothing wrote to the store until explicit apply).
  Ports `test_deny_short_circuits`,
  `test_ask_continues_evaluation`,
  `test_all_policies_evaluated`, `test_phase_filtering`
  from omniagents `test_policies.py`.

**Adherence check.** Read §4 engine loop line-by-line with
the LabelPolicy test in hand — every step in the pseudocode
has an assertion covering it.

---

## Phase 4 — `FunctionPolicy` runtime (sync + async callables)

**Scope.** Python-callable policies. No LLM yet.

**Files:**
- ✨ `agent_plane/runtime/policies/function.py`:
  `FunctionPolicy` per §9.1. Resolve `FunctionRef.path`
  once at build time; detect sync vs async with
  `inspect.iscoroutinefunction`; wrap sync via
  `asyncio.to_thread`.
- 📝 `builder.py`: dispatch `FunctionPolicySpec` →
  `FunctionPolicy`. Short form (no `arguments`) uses the
  resolved callable directly; dict form calls
  `factory(**arguments)` at build time.

**Design-adherence invariants:**
- §9.1: both YAML shapes (bare path, dict with arguments)
  produce working runtime policies.
- §4 Step 4: action whitelist validation kicks in with
  both carve-outs (DENY-allowed → fail-closed DENY,
  classifier-only → substituted ALLOW).
- §9.1 sync + async: both callable styles work.
- §4.1 "per-workflow instances": stateful (closure)
  policies keep state across evaluations in the same run;
  new instance per workflow.

**Test coverage.** Ports these omniagents
`test_policies.py` cases:
`test_allow_by_default`, `test_sync_callable_block`,
`test_sync_callable_allow`, `test_async_callable`,
`test_callable_returns_dict`, `test_deny_action_from_dict`,
`test_tool_call_rate_limit`, `test_reset_turn` (renamed —
agent-plane's per-workflow instance replaces
`reset_turn`), plus the `test_labels_and_policies.py`
FunctionPolicy-context cases:
`test_two_arg_callable_still_works` (now `test_single_arg_ctx`
since agent-plane uses `EvaluationContext`),
`test_three_arg_callable_receives_context`,
`test_three_arg_callable_reads_labels_for_decision`,
`test_three_arg_async_callable`,
`test_dict_return_set_labels`,
`test_no_session_gives_empty_labels` (renamed —
agent-plane's `_context()` always returns a dict),
`test_rate_limit_counter_isolated`,
`test_zero_arg_factory_copy_creates_fresh_state`.

- `tests/runtime/policies/test_function_policy_sync.py`:
  bare `def` callable; action whitelist filtering;
  set_labels filtering.
- `tests/runtime/policies/test_function_policy_async.py`:
  `async def` callable works identically.
- `tests/runtime/policies/test_function_policy_factory.py`:
  factory with arguments; closure rate-limit example
  counts correctly across 5 evaluations in one run.
- `tests/runtime/policies/test_function_policy_isolation.py`:
  two sequential workflow runs → fresh closure state in
  each (§4 invariant).
- `tests/runtime/policies/test_function_policy_failure.py`:
  callable raises → DENY with exception message;
  callable raises when `action: [allow]` declared →
  substituted ALLOW (carve-out).

**Adherence check.** Point-check the `rate_limit_search`
example from §9.1 is runnable verbatim against this code.
Copy omniagents `examples/search_rate_limit_policy.py`
into `tests/_fixtures/agents/rate-limited-search/tools/python/`
and confirm the factory-with-arguments form produces the
same closure behavior.

---

## Phase 5 — Enforcement: INPUT and OUTPUT phases

**Scope.** Wire the two non-tool enforcement sites. Tool
phases wait for Phase 6. This phase lets us exercise
real policies end-to-end on a live workflow.

**Files:**
- ✨ `agent_plane/runtime/policies/enforcement.py`:
  `_enforce_policy(engine, ctx)` (thin wrapper; does NOT
  apply labels — engine / await_approval handle it per §5).
- 📝 `agent_plane/runtime/workflow.py`: build engine in
  `_run_agent_loop`; call `_enforce_policy` at INPUT
  (after `_sync_history`) and OUTPUT (in
  `_handle_final_response`). DENY → sentinel; ALLOW →
  continue; ASK raises NotImplementedError (wired in
  Phase 8).

**Design-adherence invariants:**
- §5.1, §5.4: enforcement site signatures and branching.
- §11.4: OUTPUT enforcement is pre-persistence — only
  sentinel lands if DENY.
- §5 "Three lines": `_enforce_policy` body is
  `return await engine.evaluate(ctx)`; no label-apply
  call.

**Test coverage.**
- `tests/runtime/policies/test_workflow_input_allow.py`:
  end-to-end workflow with a LabelPolicy that ALLOWs →
  user message persists normally.
- `tests/runtime/policies/test_workflow_input_deny.py`:
  LabelPolicy denies → sentinel replaces user content;
  denial event streams.
- `tests/runtime/policies/test_workflow_output_deny.py`:
  FunctionPolicy denies on output → `[DENIED by policy:
  ...]` sentinel is persisted; original text is NOT in
  `conversation_items` (§11.4 pre-persistence invariant).
- `tests/runtime/policies/test_enforce_no_label_leak.py`:
  call `_enforce_policy` with a policy that ASKs and
  writes labels → store has no new label rows (ASK
  withholds writes).

**Adherence check.** Run TUI verification per `CLAUDE.md`:
spin up an agent with an input-policy blocking "canada";
verify DENY message renders in TUI.

---

## Phase 6 — Enforcement: TOOL_CALL and TOOL_RESULT (DefaultExecutor)

**Scope.** Tool-phase enforcement at the `_call_tool`
chokepoint, DefaultExecutor only. SDK executors wait for
Phase 9.

**Files:**
- 📝 `agent_plane/runtime/workflow.py`: insert
  `_enforce_policy` calls at the start and end of
  `_call_tool` (`workflow.py:1369`). DENY returns
  `{"blocked": true, "reason": ...}` sentinel as tool
  result.

**Design-adherence invariants:**
- §5.2, §5.3: tool-phase call sites build
  `EvaluationContext` with `tool_name` populated.
- §5.5 coverage matrix: DefaultExecutor gets full 4-phase
  coverage.

**Test coverage.**
- `tests/runtime/policies/test_tool_call_deny.py`:
  policy blocks `web_search` → tool returns blocked
  sentinel; downstream LLM turn sees the sentinel.
- `tests/runtime/policies/test_tool_result_label_write.py`:
  LabelPolicy on `tool_result:read_doc` writes
  `sensitivity=confidential`; post-tool-call label state
  reflects the write.
- `tests/runtime/policies/test_tool_name_resolution.py`:
  on TOOL_RESULT phase, `ctx.tool_name` matches the
  earlier TOOL_CALL dispatch (no `call_id → name` map
  needed inside engine).

**Adherence check.** Grep workflow.py for `_enforce_policy`
— exactly 4 call sites (input, output, pre-tool, post-
tool). Each constructs `EvaluationContext` with phase and
tool_name as §5 specifies.

---

## Phase 7 — `PromptPolicy` runtime

**Scope.** LLM-backed classifier policies. Most complex
single runtime component.

**Files:**
- ✨ `agent_plane/runtime/policies/prompt.py`:
  `PromptPolicy` per §9.2. Constructor accepts optional
  `classifier: Callable | None = None` override (testability
  hook §9.4). Production wires to `_call_llm_step`.
- Integration: framework-generated prompt template per §9.2;
  JSON-schema response parsing; 30 s default timeout (§9.2);
  classifier-only carve-out handling surfaces as either
  fail-closed DENY or substituted ALLOW per declared
  `action`.

**Design-adherence invariants:**
- §9.2: prompt template is framework-generated, not
  author-supplied.
- §9.2: 30 s default timeout; overrideable via
  `policy.llm.request_timeout`.
- §13: LLM timeout / unparseable JSON / unexpected tool
  calls all hit the fail-closed branch inside the engine
  evaluate loop (carve-out applies if no DENY declared).
- §9.4: testability hook works — no live LLM required for
  unit tests.

**Test coverage.** Ports these omniagents
`test_policies.py` PromptPolicy cases:
`test_prompt_policy_input_is_json_envelope`,
`test_prompt_policy_allows_from_json`,
`test_prompt_policy_denies_content`,
`test_prompt_policy_can_set_labels_when_enabled`,
`test_prompt_policy_ignores_set_labels_when_disabled`,
`test_prompt_policy_uses_configured_executor_spec`
(renamed — agent-plane uses `llm:` override, not
`executor:`), `test_prompt_policy_loader_fields`,
`test_prompt_policy_invalid_json_blocks`.

- `tests/runtime/policies/test_prompt_policy_happy.py`:
  stub classifier returns ALLOW, policy result ALLOWs;
  writes match declared whitelist.
- `tests/runtime/policies/test_prompt_policy_deny.py`:
  stub returns DENY; reason propagates.
- `tests/runtime/policies/test_prompt_policy_invalid_action.py`:
  stub returns action not in declared list → fail-closed
  DENY (declared `[allow, deny]`); fail-closed ALLOW
  (declared `[allow]`, carve-out fires).
- `tests/runtime/policies/test_prompt_policy_unparseable.py`:
  stub returns non-JSON → same as invalid_action.
- `tests/runtime/policies/test_prompt_policy_timeout.py`:
  stub raises `asyncio.TimeoutError` → DENY (or ALLOW
  carve-out). Default timeout value is 30 s.
- `tests/runtime/policies/test_prompt_policy_label_filter.py`:
  stub emits set_labels including a key outside the
  whitelist → filtered out; `label_write_dropped` span
  event (when Phase 11 spans ship).

**Adherence check.** Read §9.2 + §13 + §9.4 with the test
list open — every fail-closed rule has a test; every
carve-out has a test; every whitelist filter has a test.

---

## Phase 8 — ASK flow (synthetic `request_approval`)

**Scope.** End-to-end approval round-trip. Server +
workflow only; client changes in Phase 10.

**Files:**
- ✨ `agent_plane/runtime/policies/approval.py`:
  `_await_policy_approval` per §7.2. Strict verdict
  parsing per §13. Per-policy timeout via
  `engine.spec_for(result.deciding_policy)`. Applies
  `set_labels` on approve only.
- 📝 `agent_plane/runtime/workflow.py`: the 4
  enforcement sites' ASK branches now invoke
  `_await_policy_approval`.
- 📝 `agent_plane/server/routes/responses.py`: extend
  `POST /v1/responses/{id}/cancel` handler to mark any
  pending `request_approval` rows as `cancelled` and
  send a wake-up (see §12 Cancel during ASK).

**Design-adherence invariants:**
- §7.1: synthetic function_call uses reserved name
  `request_approval`; rides existing PATCH endpoint
  unchanged; no new SSE event.
- §7.2: labels accumulated during ASK apply ONLY on
  approve.
- §13: malformed verdict → DENY (strict
  `{"approved": true}` only).
- §13: cancel during ASK → pending row `cancelled`,
  wake delivers DENY.
- §8 "Shared topic": `tool_result` is the park/wake
  channel, same as client-tool tunneling.

**Test coverage.** Ports these omniagents
`test_labels_and_policies.py` ASK-flow cases:
`test_label_policy_ask_approve`,
`test_label_policy_ask_handler_receives_tool_args`
(renamed — agent-plane's approval payload carries
`content_preview` instead of raw tool_args),
`test_ask_handler_internal_type_error_is_not_silently_retried`,
`test_label_policy_ask_deny`,
`test_ask_timeout`,
`test_ask_user_denies_not_timeout_message`,
`test_no_handler_denies` (agent-plane equivalent:
timeout on no PATCH → DENY).

- `tests/server/integration/test_policy_approval_happy.py`:
  policy ASKs → PATCH approves → workflow resumes →
  label writes land.
- `tests/server/integration/test_policy_approval_refuse.py`:
  PATCH refuses → DENY path → no label writes.
- `tests/server/integration/test_policy_approval_timeout.py`:
  no PATCH before `ask_timeout` → DENY; per-policy
  override honored.
- `tests/server/integration/test_policy_approval_malformed.py`:
  PATCH with `{"output": "garbage"}` → workflow sees
  DENY; route still returns 200.
- `tests/server/integration/test_policy_approval_cancel.py`:
  cancel while parked → pending row `cancelled`;
  workflow resumes with DENY.
- `tests/runtime/policies/test_ask_combined_labels.py`:
  three policies all ASK with overlapping `set_labels`;
  single approval → all three policies' writes land
  (last-writer-wins across YAML order).

**E2E coverage — mandatory before closing Phase 8.**
`tests/e2e/test_policies_approval_e2e.py` runs the
ported `agent_with_policies.yaml` fixture end-to-end
with a live LLM and real PATCH round-trip. Validates
that (a) blocked tool calls return the sentinel the LLM
reacts to, (b) blocked input surfaces the DENY in the
response, (c) an ASK policy parks, a client PATCHes
approval, and the workflow resumes.
`tests/e2e/test_secure_research_e2e.py` runs the full
IFC scenario from `secure_research_agent.yaml` — web
search taints integrity, doc read taints confidentiality,
shell combo → DENY, solo → ASK.

**Adherence check.** End-to-end: trace one approval from
policy ASK through PATCH through workflow wake, hitting
each §7 subsection in order. Also: confirm the outer
phase span records `deciding_policy`.

---

## Phase 9 — Executor integration (ClaudeAgents, AgentsSdk)

**Scope.** SDK-specific hook wiring. RemoteExecutor is
INPUT/OUTPUT only (already covered by Phase 5).

**Files:**
- 📝 `agent_plane/runtime/executors/claude.py`:
  `_build_policy_hooks(policy_engine, ...)` returns
  `PreToolUse` / `PostToolUse` callbacks; merged into the
  existing `HookMatcher` list.
- 📝 `agent_plane/runtime/executors/agents_sdk.py`: extend
  `_SessionAware.call_tool` to wrap `super().call_tool`
  with `_enforce_policy` pre + post;
  `_make_session_aware_mcp_server` accepts
  `policy_engine` parameter.

**Design-adherence invariants:**
- §5.5.1: Claude hooks use `{"decision": "block", ...}`
  return shape.
- §5.5.2: MCP subclass uses `_blocked_mcp_result` for
  denied calls.
- §5.5: coverage matrix — Claude gates SDK-internals
  (Bash / Read / Edit / etc); AgentsSdk gates outer MCP
  calls only, not inside the Codex subprocess.

**Test coverage.**
- `tests/runtime/executors/test_claude_policy_hook.py`:
  policy denies `Bash` → SDK receives block decision.
- `tests/runtime/executors/test_agents_sdk_policy_mcp.py`:
  policy denies `codex(prompt=...)` call → blocked MCP
  result returned.
- `tests/runtime/executors/test_claude_policy_ask.py`:
  ASK on Claude SDK hook → approval round-trip works.

**Adherence check.** Run the TUI with a Claude-executor
agent + a policy that denies `Edit` → verify the agent
cannot edit files; verify approval flow on `Bash`.

---

## Phase 10 — Client-side (REPL + Python UI SDK)

**Scope.** Client rendering of `request_approval` function
calls. TUI is NOT in scope (being deprecated).

**Files:**
- ✨ `agent_plane/repl/approval_widget.py`: renders
  `reason`, `content_preview`, `phase`, `policy_name`;
  takes y/n; PATCHes verdict.
- 📝 REPL main loop: dispatch on `item.name == "request_approval"`.
- 📝 `frontends/sdks/python/agent_plane_ui_sdk/<session handler>`:
  recognize `request_approval`; expose
  `on_approval(context) -> bool` callback; PATCH the
  verdict.

**Design-adherence invariants:**
- §7.3: reserved-name dispatch is one branch; verdict
  format is `{"approved": true|false}` exactly.
- §17 Client Requirements: autonomous clients default-deny
  via timeout; human clients prompt.

**Test coverage.**
- `tests/frontends/repl/test_approval_widget.py`: widget
  renders from sample arguments dict; y/n produces
  correct PATCH.
- `tests/frontends/sdks/test_session_handler_approval.py`:
  session handler dispatches to `on_approval` callback;
  returning `False` → PATCH with `{"approved": false}`.
- `tests/e2e/test_policy_approval_repl.py`: full e2e
  through REPL with a live agent.

**Adherence check.** Run TUI verification per `CLAUDE.md`
section "Mandatory TUI Verification" — but substituting
REPL. Trace one full approval round-trip end-to-end.

---

## Phase 11 — Observability + polish

**Scope.** Spans, span events, telemetry attributes. Final
design-adherence sweep.

**Files:**
- 📝 `agent_plane/runtime/telemetry.py`: `GUARDRAIL` span
  type; outer `policy_phase:<phase>` + inner
  `policy_eval:<name>` wiring per §11.5.
- 📝 `engine.py`, `enforcement.py`: emit
  `label_write_dropped` span events on filtered writes;
  `policy_failure_substituted_allow` event on carve-out.

**Design-adherence invariants:**
- §11.5: two-level span structure matches exactly.
- §13: every drop path emits a telemetry event (so
  "silent" is silent in the engine but not in
  observability).
- §2.5 Trust model: every untrusted input is observable
  through at least one span / event in the validation
  step that gates it.

**Test coverage.**
- `tests/runtime/test_policy_telemetry.py`: one policy
  per phase → outer span attrs (phase, tool_name,
  n_policies, composed_action, deciding_policy,
  duration_ms) match expected.
- `tests/runtime/test_policy_span_events.py`: LabelPolicy
  with a monotonic violation emits
  `label_write_dropped {reason: monotonic}`;
  PromptPolicy with invalid action under `[allow]` emits
  `policy_failure_substituted_allow`.
- `tests/runtime/test_policy_deciding_policy.py`:
  compositions — first DENYer drives attribute; first
  ASKer in YAML order on pure ASK; null on pure ALLOW.

**Adherence check.** Final full-design pass. Walk
POLICIES.md section-by-section; for every specified
behavior, confirm there is a test that would fail if the
code stopped honoring it. Grep the doc for every
`(§N.M)` cross-reference and confirm the cited code
exists.

---

## Out of v1

Tracked in Open Questions (POLICIES.md §16), not part of
this plan:

- Concurrent workflows per conversation (Q6) — requires
  hot-cache invalidation.
- Label schema evolution (new `LabelDef.values` values
  added between deploys) — needs read-side coercion
  strategy.
- Recursive sub-agent depth policies — each level builds
  its own engine from its own spec; no depth limit
  enforced. If a limit is needed, add in v2.
- Policy verdict payload versioning — keep strict
  `{"approved": bool}` until first extension demands it.

---

## Cutting a PR per phase

One PR per phase. Each PR's description must include:

```
## Phase N — <name>

Implements designs/POLICIES.md <sections covered>.

### Design-adherence checklist
- [ ] Every invariant listed in the phase's plan entry
      has a test that would fail if violated.
- [ ] Mandatory review subagents ran; no BLOCKING
      findings.
- [ ] Mandatory test-authoring review ran; no BLOCKING
      findings.
- [ ] (Phase 5+, 8, 10) TUI / REPL verification completed
      per CLAUDE.md.
- [ ] Trust-model audit: all untrusted inputs introduced
      by this phase enumerated and validated exactly once.

### Tests added
<list>

### Tests run
<paste output>
```

This structure makes it trivial for a reviewer (or a
future audit) to confirm each phase shipped with the
design-adherence guarantees it claimed.
