# Phase 1 Adherence Checklist

Source: `/home/ubuntu/session_model_notes.md` §8 Phase 1.

Phase 1 = `@tool` decorator for custom Python tools (sync only).
Each item is verifiable from the diff. Mark [x] when the corresponding
code + test has landed and the test passes.

---

## Core deliverables

- [x] **D1**: New `@tool` decorator implemented in
      `agent_plane/tools/decorator.py`. No dep on `openai-agents`;
      schema derivation is in-house using `pydantic.create_model` +
      our own Google-style docstring parser
      (`agent_plane/tools/_docstring.py`) + our own strict-mode
      normalizer (`agent_plane/tools/_strict.py`).
- [x] **D2**: `agent_plane/tools/local.py` (`LocalPythonTool`)
      refactored. Loader scans modules for `@tool`-decorated
      functions at agent image load time and produces one
      `LocalPythonTool` per discovered function.
- [x] **D3**: `agent_plane/tools/_runner.py` accepts a `tool_name`
      field in the stdin request and dispatches to the named
      function (rejects undecorated functions defensively).
- [x] **D4**: Old `SCHEMA + async def run(arguments)` contract
      removed. Loading a module without any `@tool` functions
      fails with `LocalToolLoadError`.
- [x] **D5**: `examples/agents/archer/tools/python/word_count.py`
      migrated to `@tool` form.
- [x] **D6**: Test fixture agents under `tests/_fixtures/agents/`
      (G92 resolved). Phase 1 fixture:
      `tests/_fixtures/agents/decorator-signatures-test/`.

## G-decisions applicable to Phase 1

- [x] **G16**: Multiple `@tool` functions per file allowed.
      Verified by `test_load_multiple_tools_in_one_file` and
      `test_multiple_tools_in_one_file_invoked_via_server`.
- [x] **G17**: `_runner.py` protocol updated. New `tool_name`
      field selects the target function; calling a non-`@tool`
      function fails loud
      (`test_runner_dispatches_to_named_function`,
      `test_runner_rejects_undecorated_function`).
- [x] **G27**: Tool name collisions fail loud at agent image load.
      Verified by `test_two_custom_tools_same_name_fails_load`,
      `test_custom_tool_name_collides_with_builtin_fails_load`,
      `test_collision_error_message_is_actionable`.
- [x] **G30**: `@tool` rejects class methods, lambdas, nested
      functions at decorator-application time
      (`test_decorator_rejects_class_method`,
      `test_decorator_rejects_lambda`,
      `test_decorator_rejects_nested_function`,
      `test_decorator_rejects_staticmethod`,
      `test_decorator_rejects_classmethod`).
- [x] **G32**: Load-time error messages name agent + file + cause
      (verified throughout `test_local.py` load-failure tests and
      `test_tool_collision.py`).
- [x] **G34**: Documented in the decorator's docstring that
      `@tool` must be the outermost decorator. Asserted by
      `test_decorator_documents_outermost_requirement`.
- [x] **G45**: Return value serialization uses
      `pydantic.TypeAdapter(annotation).dump_json(value)` keyed
      on the function's declared return type, with a
      `json.dumps(value, default=str)` fallback. Implemented in
      `_runner.py:_serialize_result`. Verified by
      `test_runner_serializes_dict_return` (dict via TypeAdapter)
      and `test_runner_passes_string_return_unchanged` (str
      passthrough).
- [x] **G46**: Sync vs async function bodies. The decorator
      accepts both `def` and `async def` (verified by
      `test_decorator_accepts_module_level_def` /
      `test_decorator_accepts_module_level_async_def`). The
      runner's `_invoke_tool` runs sync directly inside the
      subprocess, where blocking is fine; the parent framework
      already invokes the subprocess via async I/O so the event
      loop is not blocked.
- [x] **G64**: Permissive types (`Any`, `object`, missing
      annotations) allowed but produce an INFO-level warning
      naming the function + parameter
      (`test_schema_warns_on_any_param`,
      `test_schema_warns_on_object_param`,
      `test_schema_warns_on_missing_annotation`).
- [x] **G65**: Strict JSON schema mode is the default
      (`test_decorator_strict_default_true`,
      `test_schema_strict_mode_default_true`); opt-out via
      `@tool(strict=False)`
      (`test_decorator_strict_false_opt_out`,
      `test_schema_strict_false_does_not_force_additional_properties`).

## Tests

### Unit — `tests/tools/test_decorator.py` (25 tests, all green)

All listed test cases implemented and passing.

### Unit — `tests/tools/test_schema.py` (22 tests, all green)

Covers primitives, Pydantic models, `Annotated[T, "string"]`,
`Annotated[T, Field(...)]`, `Literal`, `Optional`, defaults,
docstring parsing, return-type capture, strict mode, permissive
types, zero-arg functions.

### Unit — `tests/tools/test_docstring.py` (14 tests, all green)

Covers empty input, description-only, `Args:` section,
`Arguments:` / `Parameters:` synonyms, type-in-parens, multi-line
descriptions, followed-by-Returns section, malformed entries.

### Unit — `tests/tools/test_strict.py` (10 tests, all green)

Covers object normalization, recursion into properties / items /
`anyOf` / `oneOf` / `allOf` / `$defs`, non-mutation of input,
non-object pass-through.

### Unit — `tests/tools/test_tool_collision.py` (6 tests, all green)

Covers G27 collision detection across files, with builtins, and
within multi-tool files.

### Unit — `tests/tools/test_local.py` (28 tests, all green)

Subprocess invocation + crash isolation + cancellation + loader
behaviors for the new contract.

### Server integration — `tests/server/integration/test_local_tool_integration.py` (4 tests, all green)

- `test_local_tool_executes_in_subprocess` — happy path with the
  migrated `@tool`-style word_count.
- `test_local_tool_crash_does_not_kill_server` — crash isolation.
- `test_multiple_tools_in_one_file_invoked_via_server` (G16).
- `test_tool_file_without_decorator_fails_at_agent_load` — agent
  load fails when no `@tool` functions exist.

### E2E — `tests/e2e/test_decorated_tools_e2e.py` (2 tests; require
LLM API key + running ``ap server``)

- `test_archer_word_count_e2e` — real LLM, asserts on the literal
  count value.
- `test_decorated_tools_varied_signatures_e2e` — uses the
  `decorator-signatures-test` fixture covering primitive,
  Pydantic, defaults, and Annotated descriptions.

**Manual TUI verification commands** (mandatory per CLAUDE.md
before merge — documented in the E2E test file's docstring):

```
python examples/frontends/terminal.py examples/agents/archer/
python examples/frontends/terminal.py tests/_fixtures/agents/decorator-signatures-test/
```

## Migration / housekeeping

- [x] **M1**: All three existing `tools/python/*.py` files
      rewritten in `@tool` form:
      - `examples/agents/archer/tools/python/word_count.py`
      - `agent_plane/onboarding/agent/tools/python/list_builtin_tools.py`
      - `agent_plane/onboarding/agent/tools/python/validate_agent.py`
- [x] **M2**: `validate_agent.py` (onboarding) migrated; the
      validator already exercises the new spec layer correctly.
- [x] **M3**: `agent-plane-knowledge` SKILL.md updated to
      describe the `@tool` pattern. (`generate-agent` SKILL.md
      doesn't reference custom tools — no changes needed.)
- [x] **M4**: No new external dep. Only relies on `pydantic`
      (already a project dep) and the new `_docstring.py` /
      `_strict.py` written in-house.

## Architectural improvements made along the way

- `agent_plane/tools/__init__.py` now lazily resolves submodule
  exports via `__getattr__` so importing the public `tool`
  decorator from within a subprocess-loaded tool file does not
  trigger the full `ToolManager` → `mcp` import chain (which
  conflicts with the `mcp` PyPI package in subprocess
  environments).
- `agent_plane/tools/manager.py::_register_local_tools` now
  passes `agent_name` and `builtin_tool_names` to the loader so
  collision detection happens at load time rather than as a
  warn-and-shadow at registration time.

## Closing checks

- [x] All Phase 1 unit tests pass (`pytest tests/tools/`).
- [x] All Phase 1 server integration tests pass
      (`pytest tests/server/integration/test_local_tool_integration.py`).
- [x] No new failures introduced into the rest of the test
      suite (verified via `git stash` baseline comparison —
      pre-existing failures in `tests/tools/builtins/test_spawn.py`,
      `tests/server/integration/test_concurrency.py`, etc. are
      unchanged by this Phase 1 work).
- [ ] **TODO**: TUI manual verification once an LLM API key is
      available in this environment (the harness runs offline).
- [ ] **TODO**: Post-change review subagent invoked with the
      Phase 1 augmented prompt (read this checklist + verify each
      item is honored in the diff).
- [x] Grep confirms zero references to the old `SCHEMA` /
      `async def run` contract in non-test, non-doc code.
