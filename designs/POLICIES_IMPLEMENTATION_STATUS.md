# POLICIES Implementation Status

Living status doc for the policies implementation. Updated
as phases land.

## Commit log (22 commits, feature-complete at engine level)

```
56eb597 Phase 5e: edge-case coverage for the policy system
33f322b Phase 0+5: YAML → engine full-roundtrip tests
e4fa3c5 Phase 5d: multi-conversation isolation tests
a00c36a Phase 5c: four-phase enforcement contract tests
b863e3d Phase 3+8: ASK-flow x LabelDef schema validation composition tests
f65b79b Update implementation plan with final Phase 8 state
0ffc577 Phase 3+: LabelDef schema validation in apply_label_writes
41eda9b Phase 8c: multi-turn e2e tests — engine rebuild persistence
c7def1b Phase 8b: ASK-cycle e2e tests — engine + approval composed
eec9ee2 Phase 8: _await_policy_approval + strict verdict parsing
cbe9364 Phase 7b: PromptPolicy integration tests through parse+build pipeline
a9030bf Update implementation plan with phase-status table
87a0dbd Phase 5b: combined-policies fixture + 9 multi-type integration tests
b2f9f9c Phase 7: PromptPolicy runtime — third and final policy type
bd2f000 Phase 5: _enforce_policy API + integration tests for 3 ported fixtures
a96ebdd Phase 4: FunctionPolicy runtime + engine action/label validation
9a2d6a1 Phase 3: LabelPolicy runtime + engine evaluate composition
6f6ae52 Phase 2: PolicyEngine skeleton + unified builtin registry
2e6bda7 Phase 1: conversation_labels table + ConversationStore.set_labels
e54889f Phase 0: policy spec types + parser + validator wiring
4949939 Add omniagents test + example parity to implementation plan
0510563 Add POLICIES design + implementation plan
```

## Test suite (runtime/policies/)

```
203 tests across 17 files:

tests/runtime/policies/
├── __init__.py
├── conftest.py
├── test_approval.py                      (25 tests — _await_policy_approval + verdict)
├── test_ask_cycle_e2e.py                 ( 8 tests — engine + approval composed)
├── test_ask_with_schema_validation.py    ( 3 tests — approved-ASK label validation)
├── test_builder.py                       ( 7 tests — build_policy_engine)
├── test_combined_integration.py          ( 9 tests — 3 policy types in one agent)
├── test_conversation_isolation.py        ( 5 tests — multi-tenant isolation)
├── test_edge_cases.py                    (12 tests — empty / large / pathological)
├── test_enforcement_integration.py       (14 tests — 3 omniagents fixtures)
├── test_engine_skeleton.py               (13 tests — PolicyEngine construction)
├── test_four_phase_contract.py           ( 9 tests — workflow integration contract)
├── test_function_policy.py               (19 tests — FunctionPolicy + factory form)
├── test_label_policy.py                  (19 tests — LabelPolicy + composition)
├── test_label_validation.py              ( 9 tests — LabelDef values / monotonic)
├── test_multi_turn_e2e.py                ( 5 tests — engine rebuild / multi-turn)
├── test_prompt_policy.py                 (20 tests — PromptPolicy + stubs)
├── test_prompt_policy_integration.py     ( 6 tests — parse → build → stub classifier)
└── test_yaml_full_roundtrip.py           ( 8 tests — every POLICIES.md §3.1 shape)
```

Plus spec-layer tests (parser + validator) at
`tests/spec/test_policy_parser.py` (55 tests) and
`tests/spec/test_policy_validator.py` (6 tests).

Plus store-layer tests at
`tests/stores/test_conversation_labels.py` (15 tests).

Plus builtin-registry tests at
`tests/tools/builtins/test_registry_unified.py` (8 tests).

**Full non-e2e suite: 906 passing, 27 warnings.**

## Feature-complete areas

At the engine + persistence + approval-helper level, the
policy system is feature-complete:

**Spec (POLICIES.md §3)**
- Phase, PolicyAction enums
- EvaluationContext, PolicyResult dataclasses
- PhaseSelector with tool-name narrowing
- LabelDef (values + monotonic + initial)
- FunctionRef (short + dict form)
- PolicySpec base + FunctionPolicySpec + PromptPolicySpec + LabelPolicySpec
- GuardrailsSpec
- Spec-load validation: empty `on:`, `condition: {}`,
  `ask_timeout <= 0`, reserved-name collision, label schema
  cross-field rules
- YAML 1.1 `on:` trap fix via custom loader

**Persistence (POLICIES.md §6)**
- `conversation_labels` table (PK on (conversation_id, key),
  ON DELETE CASCADE to conversations)
- Dialect-aware UPSERT (SQLite + PostgreSQL + generic
  fallback)
- `ConversationStore.set_labels(conv_id, updates, updated_at)`
- Bulk-load via JOIN in `list_conversations` + `get_conversation`

**Runtime (POLICIES.md §4, §9)**
- PolicyEngine (plain local, no ContextVar)
- LabelPolicy (pure-YAML, Phase 3)
- FunctionPolicy (sync + async, short + factory, dict return
  coercion, Phase 4)
- PromptPolicy (classifier-stub injectable, framework
  envelope, 30s default timeout, Phase 7)
- Full composition loop: selector → condition gate →
  dispatch → action validation → whitelist filter → compose

**Engine safety net (POLICIES.md §13)**
- Exception in any policy → DENY (with reason)
- Action not in declared list → DENY
- Classifier-only carve-out: `[allow]`-only specs substitute
  ALLOW on failure (honors declared intent)
- set_labels whitelist filter (per-policy)
- LabelDef schema filter at apply time (values + monotonic,
  silent drop per §10)

**ASK flow (POLICIES.md §7)**
- `_await_policy_approval` with register/emit/park seams
- Strict verdict parsing: only `{"approved": true}` exactly
  returns True
- Per-policy ask_timeout override via deciding_policy lookup
- Synthetic function_call (name=request_approval,
  status=action_required)
- Content preview truncation (1024 chars)
- Labels apply on approve ONLY (denied ASK leaves no trace)
- Cancel-during-ASK (park returns None → DENY)

**Builtin registry (POLICIES.md §15.8)**
- Unified `_BUILTIN_REGISTRY` dict (single source of truth)
- `request_approval` reserved name (None-factory entry)
- `BUILTIN_NAMES` + `INSTANTIABLE_BUILTINS` derived sets
- Validator rejects user-tool collisions with reserved names

## Agent fixtures (omniagents parity)

```
tests/_fixtures/agents/
├── policies-demo/            (agent_with_policies.yaml port)
├── rate-limited-search/      (rate_limited_search_agent.yaml port)
├── secure-research/          (secure_research_agent.yaml port)
├── combined-policies/        (all 3 policy types in one agent)
└── prompt-policy-demo/       (two PromptPolicies, different phases)

tests/_fixtures/agents/*.py   (policy callables referenced by YAMLs)
```

## Remaining phases

Not shipped yet — all additive, no changes needed to the
engine / store / approval helper:

- **Phase 6** — wire `_enforce_policy` into
  `_run_agent_loop` + wire `_await_policy_approval`'s
  register/emit/park to the real DBOS + SSE stack.
- **Phase 9** — wire `PromptPolicy._default_classifier` to
  the real LLM; add Claude SDK `PreToolUse`/`PostToolUse`
  hooks and AgentsSdk MCP `_SessionAware.call_tool`
  subclass.
- **Phase 10** — REPL + Python UI SDK reserved-name handling
  for `request_approval`.
- **Phase 11** — two-level observability spans
  (policy_phase:<phase>, policy_eval:<name>) + dropped-write
  events (label_write_dropped, policy_failure_substituted_allow).

## Mandatory-review log

Every landed phase went through both mandatory review
subagents before committing:

- **Anti-pattern review** (CLAUDE.md 34-rule checklist):
  found issues on Phases 0, 1, 2 — all fixed inline before
  commit (method-size refactors, missing docstrings,
  abstraction boundaries).
- **Test-authoring review** (CLAUDE.md 13-rule checklist):
  found issues on Phases 0, 1, 2, 5 — all fixed inline
  (weak assertions upgraded with deterministic timestamps,
  defensive-copy identity checks, fixture naming).

Pre-commit passed cleanly for every Phase 0-8 file; the
only persistent mypy errors are in pre-existing
`agent_plane/llms/adapters/vertex.py` (3 errors, out of
scope for this work).
