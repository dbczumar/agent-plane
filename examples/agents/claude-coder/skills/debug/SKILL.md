---
name: debug
description: Systematically diagnose and fix bugs using logs, stack traces, and targeted investigation.
---

When debugging:

1. **Reproduce first**: Run the failing case and capture the exact error. Don't guess — observe.
2. **Read the stack trace bottom-up**: The root cause is usually near the bottom. Work upward to understand the call chain.
3. **Form a hypothesis, then test it**: Don't shotgun-debug. Identify one likely cause, verify it, then move on if wrong.
4. **Check the boundaries**: Most bugs live at integration points — API calls, file I/O, type conversions, serialization. Check inputs and outputs at each boundary.
5. **Verify the fix**: Run the original failing case again. Then check that you didn't break the happy path.

Approach:

- Start with `grep` or `glob` to find the relevant code
- Read the error logs or stack trace carefully
- Add targeted debug output if needed (remove it after)
- Fix the root cause, not the symptom
- Run tests to confirm
