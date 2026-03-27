---
name: code-review
description: Review code for bugs, security issues, and style problems.
---

When reviewing code, follow this checklist:

1. **Correctness**: Look for logic errors, off-by-one mistakes, null/undefined handling, and incorrect assumptions.
2. **Security**: Check for injection vulnerabilities (SQL, XSS, command), hardcoded secrets, and improper input validation.
3. **Performance**: Flag unnecessary allocations, N+1 queries, missing indexes, and O(n^2) where O(n) is possible.
4. **Style**: Note inconsistent naming, overly complex expressions, missing error handling, and dead code.
5. **Testing**: Suggest what tests are missing or would catch the issues found.

Format your review as:

```
## Summary
One-sentence overall assessment.

## Issues
- [severity] file:line — description of the issue and suggested fix

## Suggestions
- Optional improvements that aren't bugs.
```

Severity levels: `critical` (will break), `warning` (likely bug), `style` (readability).
