# Agent-Plane Project Instructions

## Mandatory Post-Change Review

**After completing ANY set of code changes (before committing), you MUST spawn a review subagent.** This is non-negotiable.

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

7. TYPE HINTS: Use specific types, not Any. If Any is unavoidable, add a
   comment explaining why.

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

Report each finding as:
  [FILE:LINE] ISSUE — description of the problem and suggested fix

If no issues found, say "No issues found."
```

### When to run

- After implementing a feature or fix (before committing)
- After refactoring or renaming
- After adding new store implementations or route handlers
- NOT needed for: documentation-only changes, memory updates, test-only changes

## Development Conventions

See `.claude/skills/agent-plane-dev/SKILL.md` for full project conventions,
architecture guide, and anti-patterns list.

## Git Workflow

- Always commit AND push after changes
- Run `pre-commit run --all-files` before committing
- Push to remote `corey`: `git push corey main`
