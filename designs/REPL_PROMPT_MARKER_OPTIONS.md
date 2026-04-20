# REPL prompt marker backspace bug — options

## Problem

In the terminal REPL, the orange prompt marker (`❯`) can appear to be deleted by backspace while the user is typing. The likely root issue is that the prompt chrome is rendered as inline `FormattedText` in `prompt_toolkit`, using a multi-line prompt with a wide Unicode glyph and surrounding spaces, which can produce redraw / cursor-width artifacts in some terminals.

## Goal

Make the prompt marker visually stable and clearly non-editable without regressing:

- pinned prompt UX
- attachment display
- bottom toolbar/status line
- terminal compatibility
- implementation simplicity

---

## Option 1 — Minimal mitigation: switch to ASCII marker

Change the prompt marker from `❯` to `>` (or another single-column ASCII glyph) and keep the rest of the prompt structure as-is.

### Pros
- Tiny patch
- Low risk
- May fix the issue immediately if the bug is mainly Unicode-width related

### Cons
- Does not address the architectural issue
- Multi-line prompt chrome is still embedded in the editable prompt
- Could still exhibit related redraw artifacts in some terminals

### When to choose
If we want the fastest low-risk experiment first.

---

## Option 2 — Keep inline prompt, but simplify it

Keep using `build_prompt()` with inline prompt text, but remove most decorative chrome from the prompt itself:

- no top bar in prompt
- no attachment lines in prompt
- marker becomes something like `❯ ` or `> ` only
- move bars/attachments elsewhere if needed

### Pros
- Smaller refactor than a full layout change
- Likely to reduce the bug substantially
- Preserves the current `PromptSession.prompt_async(prompt, ...)` flow

### Cons
- Still relies on inline prompt rendering
- Still may be vulnerable to terminal/prompt_toolkit quirks
- Not a fully robust separation of UI chrome vs editable input

### When to choose
If we want a pragmatic middle ground with limited churn.

---

## Option 3 — Proper fix: separate prompt chrome from editable input

Use prompt_toolkit layout primitives so the decorative marker/chrome is rendered outside the editable input buffer.

Possible approach:
- keep the actual prompt text minimal or empty
- render the marker in a dedicated left window / margin / fixed container
- render attachment list in a separate non-editable area
- keep status in bottom toolbar

### Pros
- Architecturally correct
- Best chance of eliminating the bug entirely
- Cleaner separation of concerns

### Cons
- Larger implementation
- Requires more prompt_toolkit layout work
- May need re-testing around resize, streaming, attachment handling, and focus behavior

### When to choose
If we want the most robust long-term solution.

---

## Option 4 — Prompt-toolkit custom buffer / container layout

Build a more explicit REPL layout instead of depending mainly on `PromptSession.prompt_async(...)`:

- dedicated output area
- dedicated input buffer
- dedicated prompt marker container
- dedicated attachment/status regions

This is effectively a fuller custom terminal UI within prompt_toolkit.

### Pros
- Maximum control
- Cleanest model if the REPL grows more sophisticated
- Easier to make prompt truly non-editable

### Cons
- Most work
- Highest maintenance cost
- Risks overlap with functionality already covered by the current host abstraction

### When to choose
If we expect substantial REPL UI expansion beyond this bugfix.

---

## Option 5 — Redraw hack / cosmetic workaround

Keep current prompt design and try to mask the issue with explicit invalidation/redraw logic after backspace or on each keypress.

### Pros
- Could be small if it works

### Cons
- Fragile
- Symptom treatment, not root-cause fix
- Likely terminal-specific and hard to trust

### When to choose
Probably never, unless we need an emergency temporary workaround.

---

## Recommendation

Recommended order:

1. **Option 1** as a quick experiment
2. **Option 2** if Option 1 is insufficient
3. **Option 3** as the best real fix if we want to solve it properly

My current bias: **Option 3 is the best design**, while **Option 2 is the best practical compromise**.

---

## Suggested validation checklist

Whichever option we pick, verify:

1. Backspace never visually erases the prompt marker
2. Prompt survives terminal resize
3. Prompt survives streaming output
4. Prompt survives cancellation (`Esc`)
5. Prompt survives attachment drag/drop flows
6. Marker alignment is correct in common terminals
   - iTerm2
   - macOS Terminal
   - Kitty
   - GNOME Terminal / xterm-compatible terminals
7. No regressions in history behavior or focus behavior

---

## Open questions

1. Is the bug reproducible in all terminals or only some?
2. Is the issue specific to `❯`, or to any Unicode-width marker?
3. Does the top-bar + multi-line prompt structure contribute more than the glyph itself?
4. Do attachments need to remain visually near the input line, or can they live only in the toolbar?
