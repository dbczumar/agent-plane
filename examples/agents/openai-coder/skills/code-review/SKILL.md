---
name: code-review
description: Review code for bugs, security issues, and style problems. Provide actionable feedback.
---

When reviewing code:

1. **Read the full context first**: Understand what the code does before critiquing it. Read related files if needed.
2. **Prioritize by impact**: Security issues > correctness bugs > performance > style. Don't bury a real bug under formatting nits.
3. **Be specific**: Point to exact lines. Show the problem and the fix, not just "this could be improved."
4. **Check edge cases**: Empty inputs, None values, concurrent access, error paths. These are where bugs hide.
5. **Verify tests**: Are the important behaviors tested? Are the tests actually asserting the right things?

Format your review as:

```
## Critical
- [file:line] Issue description — suggested fix

## Improvements
- [file:line] Issue description — suggested fix

## Looks Good
- Brief note on what's well done (if anything stands out).
```

Skip the "Looks Good" section if nothing is notable. Don't pad the review with praise to soften criticism.
