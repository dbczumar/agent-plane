# Agent-Plane Design Principles

These principles govern all design and implementation decisions in
agent-plane. They are not guidelines — they are hard constraints.
Every feature, tool, and configuration mechanism must satisfy all of
them. When principles conflict, resolve in the order listed (earlier
principles take precedence).

---

## 1. Spec Self-Containment

**The agent spec is the single source of truth for agent behavior.**

An agent must behave identically regardless of which server it runs
on. All configuration — API keys, backend selection, tool parameters,
feature flags — must come from the spec (config.yaml + bundled files).
`${ENV_VAR}` references are resolved at **deploy time by the client**,
not at runtime by the server.

**Violations:**
- Reading `os.environ` at runtime to select backends or provide
  credential fallbacks.
- "If env var X is set on the server, use backend Y."
- Behavior that changes based on what's installed on the server
  (except for explicit runtime capabilities like sandbox availability,
  which are declared in `RuntimeCaps`).

**Why:** Reproducibility. If the same spec produces different behavior
on different servers, debugging is impossible and testing is
meaningless. The spec is a contract — it must be honored literally.

---

## 2. One Correct Path

**No dual-mode fallbacks or feature flags for the same operation.**

Each operation has exactly one implementation path. Never support two
ways of doing the same thing with `if use_X ... else use_Y` branching.
When a new mechanism replaces an old one, the old one is deleted in the
same change.

**Violations:**
- "Try backend A, fall back to backend B based on available keys."
- Feature flags that enable/disable code paths.
- Backwards-compatibility shims (`OldName = NewName`).

**Why:** Two paths means two things to test, two things to debug, and
an implicit priority order that someone has to remember. One path is
always simpler and more reliable.

---

## 3. Detect, Don't Ask (Within Spec Boundaries)

**Probe the spec and bundled files before asking the user questions.**

The onboarding agent should examine the user's code and the spec
config before asking what to do. But detection must stay within spec
boundaries — never probe the server environment (see Principle 1).

---

## 4. Approve Before Mutate

**Every filesystem write or external action requires explicit user
confirmation.** The agent explains what it will do, the user approves,
then the agent acts. No silent writes.

---

## 5. Fail Loud

**Required values must fail with clear errors, never silently
substitute defaults.**

No `.get("key", "fallback")` where the fallback is invented. No empty
string sentinels. No swallowed exceptions. If something is missing,
the error message must say exactly what's missing and how to fix it.

**Why:** Silent defaults mask configuration errors. A clear error at
startup is always better than mysterious behavior at runtime.

---

## 6. Minimal Surface Area

**Don't add features, abstractions, or configurability beyond what
the task requires.**

No speculative abstractions. No "in case we need it later." No helper
functions for one-time operations. Three similar lines of code is
better than a premature abstraction. When in doubt, don't add it.

---

## Related Documents

- [AGENTSPEC.md](../agent_plane/spec/AGENTSPEC.md) — Agent spec format
- [RUNTIME.md](RUNTIME.md) — Runtime initialization and store interfaces
- [EXECUTOR_CONTRACT.md](EXECUTOR_CONTRACT.md) — Remote executor protocol
