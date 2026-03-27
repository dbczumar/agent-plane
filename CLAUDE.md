# Agent-Plane Project Instructions

## Mandatory Post-Change Review

**🚨 EXTREMELY IMPORTANT: After completing ANY set of code changes (before committing), you MUST spawn a review subagent. This is the SINGLE MOST IMPORTANT step in the development workflow. NEVER skip it. NEVER forget it. Run it after EVERY code change, no matter how small. Failure to run the review subagent is a BLOCKING issue.**

### How to run the review

Spawn an `Explore` subagent with the following prompt structure:

```
Review the following changed files for anti-patterns and spec deviations:
[list the files you changed]

Check each file against this checklist:

1. EMPTY STRING DEFAULTS: No `field: str = ""` or `field: int = 0` as
   placeholder/sentinel values. If a field is optional, use `Optional[str] = None`.
   If it's required, make it required.

2. UNUSED PARAMETERS: No function parameters accepted and silently ignored.
   If a parameter isn't used, remove it from the signature.

3. NAMES VS IDS: Store stable IDs as primary references, not names. Names
   can change; IDs are durable.

4. CLARIFYING COMMENTS: Non-obvious function arguments (framework workarounds,
   magic numbers, hardcoded status strings, empty-string placeholders) must
   have inline comments explaining WHY.

5. RESPONSE MODEL CONSISTENCY: FastAPI routes returning union types
   (e.g. Pydantic model | StreamingResponse) must use response_model=None
   with a comment explaining why.

6. SPEC COMPLIANCE: Changes to API routes must match server/API.md. Changes
   to store interfaces must match runtime/RUNTIME.md. Check field names,
   types, required/optional, and behavior.

7. TYPE HINTS: Use specific types, not Any, object, dict, list, Callable,
   or other overly generic types. Prefer concrete types (e.g.
   `dict[str, AgentSpec]` not `dict`, `Connection` not `object`,
   `Callable[[str], bool]` not `Callable`). If a generic type is truly
   unavoidable, add a comment explaining why.

8. PYDANTIC AT BOUNDARIES: Pydantic models for API/external data, dataclasses
   for internal entities. Don't mix these up.

9. STORE INTERFACE CONTRACTS: Abstract store methods must have clear docstrings.
   Implementations must honor the contract (e.g. KeyError on missing get,
   no-op on missing delete).

10. KEY VALIDATION: Artifact keys must be validated against traversal attacks.
    Use PurePosixPath for parsing, is_relative_to() for containment.

11. NO CLASS-BASED TESTS: Test files must use function-based tests with
    fixtures, not class-based (no `class TestFoo`). Test file structure
    must mirror the source directory (e.g. tests/stores/test_agent_store.py).

12. FIXTURE LOCALITY: Fixtures belong in the test file that uses them.
    Only promote to conftest.py when shared across multiple test files.
    Shared non-fixture helpers go in a helpers.py module, not conftest.

13. NO SLEEP IN TESTS: No `time.sleep()` in test code. Use `await wait()`,
    event-driven checks, or restructure the test. If truly unavoidable,
    flag to reviewer with a comment explaining why.

14. NO INTERNAL METHOD CALLS IN TESTS: Tests must not call `_`-prefixed
    (private/internal) methods of production code. If a test needs a
    private method, the public API is likely incomplete — flag to reviewer.

15. NO INVENTED DEFAULTS: No `.get("key", "fallback")` or `value or "unknown"`
    where the default was not explicitly designed. Required values must fail
    loud (KeyError, ValueError) rather than silently substituting a made-up
    value. If a default is genuinely needed, it must have a comment explaining
    why that specific value is correct.

16. NO OVERLY DEFENSIVE ERROR HANDLING: Don't wrap code in try/except
    "just in case". Catch only specific exceptions at system boundaries
    (user input, external APIs). Internal code should fail loud. Bare
    `except Exception` is almost always wrong. Never swallow errors.

17. DB COLUMN DEFAULT CORRELATION: When mapping DB rows to entities,
    correlate Python-side defaults with the column schema. Nullable
    columns must not have hardcoded fallbacks (masks NULL data).
    Non-nullable columns with server defaults must not have redundant
    Python-side fallbacks. Flag both cases to a human for review.

18. NO TUPLE RETURNS OR ARGUMENTS: Never return tuples from functions or
    pass tuples as arguments. Use lightweight dataclasses with named
    fields. Tuples are positionally fragile and not extensible.

19. NO EMPTY STRING SENTINELS: Never use empty strings (`""`) as
    sentinel values, default initializers, or "not yet set" placeholders.
    Use `None` (with `Optional` / `| None` type) for absence. Never
    assign `""` to a variable that will later be checked — use `None`
    and `is not None`. Never pass `""` as an argument to mean "no value".
    Empty strings hide bugs because they are falsy but not `None`,
    causing subtle `if x` vs `if x is not None` mismatches.

20. NO LARGE METHODS: Functions/methods must be <= 40 lines. If a
    method exceeds this, split it into named helper functions. Common
    splits: (a) extract validation into a helper that returns validated
    data or raises, (b) extract each major code path (streaming,
    blocking wait, background) into its own function, (c) extract
    complex setup/teardown into context managers or helpers. The goal
    is that each function does ONE thing and its logic fits on a
    screen. Nested `async def` closures count toward the enclosing
    function's length — extract them to module-level async functions
    that accept explicit arguments instead.

21. ABSTRACTION VIOLATIONS: Code must respect abstraction boundaries.
    This includes but is not limited to:
    - Importing from a package's internal submodules when it exposes a
      public API (e.g. import from `agent_plane.spec` not
      `agent_plane.spec.parser`). Exception: same-package siblings
      and unit tests for a specific submodule.
    - Manually orchestrating a multi-step pipeline that a higher-level
      function should encapsulate (e.g. calling parse() then validate()
      separately instead of a single load()).
    - Duplicating logic that belongs in another layer (e.g. route-level
      code reimplementing store-level validation).
    - Exposing internal state or implementation details through a
      public interface.
    If a needed abstraction doesn't exist, create it rather than
    working around the gap.

22. NO COUNTER VARIABLES FOR FIXED SEQUENCES: When values are
    deterministic constants (e.g. SSE sequence numbers 0, 1, 2 for a
    known set of events), hardcode the literals instead of
    incrementing a counter variable. Counter variables imply the
    count is dynamic; use them only when iterating over data of
    unknown length. A `seq = 0; seq += 1; seq += 1` sequence where
    the values are always 0, 1, 2 should just be 0, 1, 2.

23. NO RACE CONDITIONS: Zero tolerance. When bridging sync/async or
    cross-thread boundaries, ordering guarantees must be enforced
    structurally. "Register before start" not "register and hope."
    No "negligible window" or "in practice" handwaving. If a
    theoretical race exists, it must be eliminated by design.

24. NO DUAL MODES / FALLBACK PATHS: Don't support two ways of doing
    the same thing. If a new mechanism replaces an old one, make it
    the only path. No `if use_X ... else use_Y` branching for the
    same operation. Test infrastructure must use the same path as
    production code.

25. **COMPREHENSIVE DOCSTRINGS (CRITICAL)**: Every function, method,
    class, and dataclass MUST have a docstring. Docstrings MUST
    include `:param name: description` for EVERY parameter. For
    parameters whose values are not obvious from the name/type,
    include an example value (e.g. ``:param model: The litellm model
    identifier, e.g. ``"openai/gpt-4o"````). Dataclass fields must
    be documented either with field-level comments or in the class
    docstring with `:param:` entries. Missing or incomplete
    docstrings are a blocking issue.

26. MOCK INTEGRITY: MagicMock must NEVER be used when a real type
    exists (SDK types, Pydantic models, dataclasses). MagicMock
    silently returns MagicMock for any attribute access, making
    broken code pass green. Use real types from the same module
    the production code imports. MagicMock is ONLY acceptable for
    client/interface stubs where production calls methods but never
    checks types or accesses nested attributes. After any
    import-path refactor, grep tests for stale type references.
    Stale mock *targets* (monkeypatch of renamed function) raise
    errors; stale mock *types* silently degrade.

27. ASSERTION DEPTH: Test assertions must verify actual content values,
    not just structural properties. `assert len(x) >= 1` and
    `assert x[0]["role"] == "assistant"` pass even when the payload
    is None/empty. Always assert on the value that proves the mock
    data traversed the full pipeline.

28. NO RANDOM/INVENTED ENV VAR DEFAULTS: Never use
    `os.environ.get("REQUIRED_VAR", "some-default")` where the
    default is an invented value like `""`, `"us-central1"`, or
    `"localhost"`. Required env vars must use
    `os.environ.get("VAR")` + explicit `if var is None: raise
    ValueError(...)`. Invented defaults mask missing configuration
    and cause silent, hard-to-debug failures in production. If a
    genuine default exists (documented in provider docs), it must
    have a comment citing the source.

29. NO REINVENTING PRIMITIVES: Never hand-roll complex data structures
    or algorithms that have well-known, battle-tested implementations
    in the standard library or popular packages (e.g. custom LRU
    caches, TTL caches, retry logic, rate limiters, thread pools,
    connection pools). Custom implementations are a stability risk —
    they lack the edge-case coverage, testing, and maintenance of
    established libraries. Use `cachetools`, `tenacity`, `stdlib
    collections`, etc. If a custom implementation is truly needed
    (e.g. domain-specific invariants that no library enforces), add
    a comment explaining why no existing library suffices.

Report each finding as:
  [FILE:LINE] ISSUE — description of the problem and suggested fix

If no issues found, say "No issues found."
```

### When to run

- After implementing a feature or fix (before committing)
- After refactoring or renaming
- After adding new store implementations or route handlers
- NOT needed for: documentation-only changes, memory updates, test-only changes

## Testing: Mock Integrity and Assertion Depth

### Never let mocks silently degrade

When mocking an external boundary (LLM client, MCP server, etc.):

1. **Use real types, not MagicMock, for data objects.** If the production
   code does `isinstance(event, SomeType)`, a `MagicMock()` will silently
   fail the check and fall through to a default path. Use the real
   dataclass/type from the same module the production code imports. If
   the import path changes (e.g. `openai.types` → `llms.types`), the
   mock must change too — a stale mock won't error, it will just stop
   matching.

2. **Assert on content, not just structure.** `assert len(result) >= 1`
   and `assert result[0]["role"] == "assistant"` pass even when text is
   `None`. Always assert the actual value that proves the mock's data
   made it through the full pipeline.

3. **After any refactor that changes imports or type paths, grep test
   files for the old names.** Stale mock targets (`monkeypatch.setattr`
   of a renamed function) raise `AttributeError` — those are easy to
   catch. Stale *type* references in mock construction are silent.

### Test invariants at the right layer

Workflow integration tests (mock LLM → run full workflow → check result)
are good for "does the happy path complete?" They are **bad** for testing
ordering invariants, cursor arithmetic, or concurrency handshakes.

When a function has position-ordering or state-machine invariants
(e.g. `close_inbox` cursor advancement, `last_seen` tracking):

- Write a **focused unit test** that calls the function directly with
  controlled store state. Set up specific position orderings by hand,
  call the function, assert the exact cursor value and store mutations.
- The workflow integration test should verify the *outcome* (both
  responses persisted, inbox closed) but should NOT be the only test
  covering the ordering logic.

## Development Conventions

See `.claude/skills/agent-plane-dev/SKILL.md` for full project conventions,
architecture guide, and anti-patterns list.

## Git Workflow

- Always commit AND push after changes
- Run `pre-commit run --all-files` before committing
- Push to remote `corey`: `git push corey main`
