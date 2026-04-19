# Label Policies — Deep Dive

Companion to `POLICIES_OMNIAGENTS_NOTES.md`. Zooms in on the
`type: label` policy specifically — what it is, what it's for,
how it differs from the other two policy types, and where it
falls short.

Source code: `omniagents/policies.py:566` (`make_label_policy_callable`)
and `omniagents/loader.py:321` (`_parse_policy` for `type: label`).

---

## Glossary

Terms used throughout this doc. Grouped for learning order;
within each group, listed as they first appear.

### Runtime objects

- **Agent** — The static configuration: system prompt, tools,
  policies, initial labels, model, etc. Loaded from a YAML
  file. Code type: `AgentDef` in `omniagents/datamodel.py`.
  An Agent on its own doesn't *do* anything; it's
  instantiated into a Session to run.

- **Session** — A live, running instance of an Agent in
  conversation with a user. A Python object that owns: the
  message history, current label values, policy engine,
  memory, OS environments, tool registry, bidirectional
  Connection(s), any async tasks, and any named child
  sessions. Created when a user starts chatting; lives until
  the conversation ends. All "session-scoped" state (labels,
  memory, policies' closure state) lives on this object. In
  OmniAgents this state is in memory and lost on crash;
  agent-plane's closest analogue is a Conversation + its DBOS
  workflow + SQLAlchemy-persisted stores. Code type:
  `Session` in `omniagents/session.py`.

- **Turn** — One round of user↔agent exchange: user sends a
  message, the agent responds (possibly making many LLM calls
  and tool calls internally), then the agent is idle again
  waiting for the next user message. Stateful policies' per-
  turn counters reset between turns via `reset_turn()`.

- **Tool** — A function the agent can invoke during a response
  (e.g. `web_search`, `shell`, `read_internal_doc`). In
  OmniAgents, tools are registered Python callables — some
  wrap MCP servers, some wrap OS operations, some are sub-
  agents. Policies can intercept tool invocations at the
  `tool_call` and `tool_result` phases.

- **Sub-agent** — An Agent exposed to another Agent as a Tool.
  Spawning a sub-agent creates a child Session under the
  parent. Labels from the parent propagate into the child via
  `merged_with_child` rules in the label schema.

- **Connection** — A bidirectional asyncio Queue pair used to
  communicate with a Session. The `primary_connection` links
  the Session to the user's CLI/web client — specifically,
  ASK approval prompts are sent outbound on it and the user's
  yes/no comes back inbound.

- **PolicyEngine** — One object per Session (`Session._policy_engine`).
  Holds all policies for that Session and exposes
  `evaluate(content, phase)` which runs every policy that
  applies to the given phase, combining their results via max-
  action semantics (DENY > ASK > ALLOW).

### Policy primitives

- **Policy** — Base class for all policies. Declares a `name`,
  an `on` list of phases it cares about, and an `async
  evaluate(content, phase)` that returns a PolicyResult. Code:
  `omniagents/policies.py:70`.

- **FunctionPolicy** — A Policy whose `evaluate()` delegates
  to a user-supplied Python callable `fn(content, phase)` (or
  `fn(content, phase, context)`). Can be stateful via Python
  closures. The "escape hatch" when declarative YAML isn't
  expressive enough.

- **PromptPolicy** — A Policy whose `evaluate()` sends the
  content plus a policy-author-written instruction prompt to
  a small LLM and parses a JSON `{action, reason, set_labels}`
  decision from the response. Fails closed on any error (any
  exception → DENY).

- **Label policy** (`type: label` in YAML) — NOT a dedicated
  Python class. YAML-level convenience for declaring a
  FunctionPolicy whose callable checks session labels and
  emits an action. Compiled into a FunctionPolicy at load
  time via `make_label_policy_callable`. This whole doc is
  about this form.

- **PolicyResult** — The dataclass returned by every
  `evaluate()`. Three fields: `action` (the PolicyAction),
  `reason` (explanation string, shown in denial messages),
  `set_labels` (dict of label updates to apply after this
  phase's evaluation completes).

- **PolicyAction** — Enum of possible decisions: `ALLOW`
  (proceed), `ASK` (pause and prompt the user for approval),
  `DENY` (reject the content; replace with a sentinel).

- **Phase** — One of four hardcoded points in the Session
  loop where policies get evaluated: `input` (user message
  arriving), `tool_call` (before a tool runs), `tool_result`
  (after a tool's output, before injection into history),
  `output` (after the LLM produces a response). A policy's
  `on` field lists which phases it opts into.

### Labels

- **Label** — A string-valued tag on the Session
  (`session.labels: dict[str, str]`). Persists across phases
  and across turns within a Session. Any policy can read them
  (via `context["labels"]`) or write them (via `set_labels`
  on a PolicyResult). Lost when the Session ends. String
  values only — no ints, no bools.

- **`label_schema`** — Top-level YAML field declaring, for
  each label, the valid values and a monotonicity constraint.
  Updates that violate the schema are silently rejected.
  Code type: `LabelSchemaRule` in `omniagents/datamodel.py`.

- **Monotonic / monotonicity** — Constraint on how a label's
  value may change over time. `max` = only increases allowed
  (position in `values` list can only go up). `min` = only
  decreases. `none` = any transition between declared values
  allowed. Used to express "once tainted, stay tainted."

- **`set_labels`** — Two meanings. (1) A field on
  PolicyResult: a dict of label updates the engine applies
  after the policy runs. (2) A YAML field on label policies:
  the updates the generated closure emits when its condition
  matches. Accumulate across policies in one phase; silently
  drop when they violate the schema.

- **`condition`** — YAML field on label policies. Dict of
  labels that must match (equality or membership in a list)
  for the policy to fire. Empty dict = always match.

- **`match_tools`** — YAML field on label policies. Narrows
  the policy to fire only when one of the named tools is
  being called. Only meaningful on the `tool_call` phase
  (there's no tool name on other phases).

- **`merged_with_child`** — Method on `LabelSchemaRule` that
  computes how parent and child Session labels combine when a
  Session spawns a sub-agent. Uses monotonicity to ensure
  child can't "launder" the parent's taint (or vice versa).

### Security / information-flow concepts

- **Taint** — A label value indicating the Session has been
  exposed to a particular category of data (untrusted input,
  confidential data). Not a special concept in the code —
  just a convention for how label values are used.

- **Taint propagation** — The pattern of one tool
  automatically setting a taint label so later policies can
  restrict downstream tools. Mechanically: setter policy +
  gate policy sharing the same label key.

- **Source** — A tool that brings data INTO the Session
  (`web_search` brings in untrusted text; `read_internal_doc`
  brings in confidential data). In the label-policy pattern,
  sources are the things that *set* taint labels.

- **Sink** — A tool that sends data OUT of the Session
  (`shell`, `email_send`, `web_post`). Sinks are the things
  that *check* taint labels before accepting data.

- **Information-flow control (IFC)** — A security discipline
  where every piece of data carries a label and every
  operation is constrained by what data it's exposed to.
  OmniAgents' label system is a lightweight, session-scope
  IFC.

### YAML fields / config

- **`on`** — List of phases a policy should run on. Defaults
  to `[input, output]` if omitted.

- **`action`** — YAML field on label policies. The
  PolicyAction (`allow` / `ask` / `deny`) to emit when the
  policy's condition matches.

- **`reason`** — YAML field (and PolicyResult field). Human-
  readable explanation shown in denial messages and ASK
  prompts.

- **`ask_timeout`** — Top-level YAML field (agent-level, not
  per-policy). Seconds to wait for a user's response to an
  ASK prompt before auto-denying.

- **`allow_set_labels`** — YAML field on PromptPolicy only.
  Whether the classifier LLM is permitted to emit labels in
  its JSON decision. Defaults to false.

### Other

- **ASK flow** — The interactive approval process triggered
  when a policy returns `ASK`. The Session posts an approval
  request on the `primary_connection`, waits up to
  `ask_timeout`, and resumes with `ALLOW` if the user
  approves or treats it as `DENY` on refusal or timeout.

- **Max-action semantics** — The PolicyEngine's composition
  rule when multiple policies fire on one phase. Severity
  ordering: `DENY > ASK > ALLOW`. First `DENY` short-circuits.
  First `ASK` is remembered but evaluation continues so a
  later `DENY` can still escalate.

- **Fails closed** — When a policy errors (exception,
  unparseable LLM output, etc.), the result is treated as
  `DENY` rather than `ALLOW`. Prevents a broken policy from
  becoming a security bypass.

- **YAML sugar** — A YAML pattern that looks like a first-
  class feature but is compiled into a lower-level primitive
  at load time. Label policies are sugar for FunctionPolicy.

- **Closure** (Python term) — A function that captures
  variables from its enclosing scope. Used in FunctionPolicy
  to implement stateful policies (rate limiters, counters)
  without needing a class.

---

## 1. What label policies actually are

**Labels are session-scoped tags.** A `dict[str, str]` on the
Session object. They persist across every policy evaluation
within a session — they're the only cross-phase state the
policy system has.

Any policy can:
- **Read labels** — to decide what action to take now.
- **Write labels** — to record something for later policies to
  see via the `set_labels` field on `PolicyResult`.

Label policies (`type: label`) bake both behaviors into YAML
declaratively. Mechanically they are **sugar** — the loader
compiles them into a `FunctionPolicy` with a generated closure.
Anything a label policy does, you could write as a function
policy. The sugar just makes the common patterns readable
without Python.

No `LabelPolicy` class exists in the codebase. At runtime a
label policy is indistinguishable from a function policy.

---

## 2. Two common uses, same YAML shape

### Use 1 — setter ("when this happens, tag the session")

```yaml
taint_web_search:
  type: label
  on: [tool_call]
  condition: {}                # empty = always match
  match_tools: [web_search]    # narrow to this tool
  action: allow                # don't block, just tag
  set_labels:
    integrity: "0"             # mark "untrusted input has entered"
```

Reads as: "every time `web_search` is about to run, allow it —
but set `integrity=0` on the session." Does nothing to the tool
call itself; just annotates session state for other policies
(or future evaluations of this policy) to see.

### Use 2 — gate ("if the session is tagged X, block Y")

```yaml
deny_contaminated_shell:
  type: label
  on: [tool_call]
  condition:
    integrity: "0"             # trigger only if this label is set
  match_tools: [shell, bash]
  action: deny
  reason: "Cannot run shell on tainted session."
```

Reads as: "before `shell` or `bash` runs, check the session's
labels — if `integrity=0`, DENY with that reason." If the
condition doesn't match, the policy falls through with ALLOW
and no labels set.

Same YAML shape. Difference is just:
- **Setter**: `condition` is loose/empty; `set_labels` is populated.
- **Gate**: `condition` is specific; `action` is `deny` or `ask`.

---

## 3. The interesting part — composition

Any single label policy is simple. The power comes from
**multiple label policies that don't know about each other but
share the same label vocabulary.**

`examples/secure_research_agent.yaml` is the canonical example.
Four independent policies compose into information-flow
control:

```yaml
labels:             # initial session state
  confidentiality: "0"
  integrity: "1"

label_schema:
  confidentiality: {values: ["0","1"], monotonic: max}  # only goes UP
  integrity:       {values: ["0","1"], monotonic: min}  # only goes DOWN

policies:
  # --- Setters: taint on untrusted input / confidential data ---
  taint_web_search:
    type: label
    on: [tool_call]
    match_tools: [web_search]
    action: allow
    set_labels: {integrity: "0"}

  taint_confidential_read:
    type: label
    on: [tool_call]
    match_tools: [read_internal_doc]
    action: allow
    set_labels: {confidentiality: "1"}

  # --- Gates: block high-risk actions based on accumulated taint ---
  deny_shell_after_web:
    type: label
    on: [tool_call]
    condition: {integrity: "0"}
    match_tools: [shell]
    action: deny
    reason: "Don't let web-fetched commands execute."

  deny_exfil_after_internal:
    type: label
    on: [tool_call]
    condition: {confidentiality: "1"}
    match_tools: [web_post, email_send]
    action: deny
    reason: "Don't exfiltrate confidential data."
```

Observable behavior across a session:

| Session state             | `web_search` | `read_internal_doc` | `shell` | `email_send` |
| ---                       | ---          | ---                 | ---     | ---          |
| Initial                   | allow        | allow               | allow   | allow        |
| After one `web_search`    | allow        | allow               | **DENY**| allow        |
| After one `read_internal_doc` | allow    | allow               | allow   | **DENY**     |
| After both                | allow        | allow               | **DENY**| **DENY**     |

**None of the four policies know the others exist.** They all
just read and write the shared `labels` dict. Adding a fifth
rule ("also block shell if `confidentiality=1`") is one more
YAML block — no code changes, no refactoring.

This is a small [information-flow control](https://en.wikipedia.org/wiki/Information_flow_(information_theory))
system:
- **Sources** (web_search, read_internal_doc) stamp taint
  labels onto the session.
- **Sinks** (shell, email_send) check taint before accepting
  actions.
- **Monotonicity** (`max` / `min`) guarantees taint can only
  get worse — no policy can accidentally clear a taint label to
  let bad data flow through.

---

## 4. When to reach for a label policy

- **Taint propagation**: "once this tool runs, this other tool
  must be restricted." The canonical pattern.
- **Session-wide classification**: "if the session is marked
  `tier=free`, only these tools are allowed." Set once at
  session start, many policies reference it.
- **Quick approval gates**: "the first DB-write tool call needs
  user approval; once approved, allow silently." Uses a label
  as a one-shot memory. (You'd need two policies: one gate with
  `condition: {}` `action: ask` `set_labels: {db_approved: "1"}`,
  one gate with `condition: {db_approved: "1"}` `action: allow`.)

## When you shouldn't

Label policies look clean but have real limits. Use a
`FunctionPolicy` instead when:

- **You need arithmetic or counting.** Labels are strings. You
  can encode `"3"` as text but incrementing requires read /
  parse / write — much cleaner as a stateful Python callable
  with a closure counter.
- **You need time windows** ("rate limit: 3 per minute"). Labels
  have no clock.
- **Your condition is anything other than equality or `in`-list**.
  No `>`, no `!=`, no regex, no substring, no boolean combinators
  beyond implicit AND across keys.
- **You need to branch on the content being evaluated**, not just
  pre-existing labels. Label policies only look at the content
  via `match_tools` — and only on the `tool_call` phase.

---

## 5. Full list of limitations

1. **String values only.** No int, no bool, no timestamps, no
   lists. `"1"`/`"0"` is a convention, not a type. Numeric
   comparison requires manual parsing in a FunctionPolicy.

2. **Conditions are equality or membership only.**
   `condition: {key: "val"}` or `condition: {key: ["v1", "v2"]}`.
   No `>`, no `!=`, no regex. AND is implicit across keys; OR
   across values on one key. OR across keys requires duplicating
   the policy.

3. **`match_tools` is only meaningful on `tool_call` phase.**
   On input / output / tool_result phases the field is ignored
   — there's no tool name to match against.

4. **Per-session only.** Labels don't persist across session
   restarts. If you want durable policy state (e.g. per-user
   budgets, cross-session taint), labels are the wrong layer.

5. **No label history.** Current values only. Can't ask "what
   was it 3 turns ago" or "when did this change." Each
   evaluation sees only the present.

6. **Schema enforces only monotonicity.** `max` / `min` / `none`.
   Can't express "must reset every N turns," "allowed values
   depend on another label's value," or numeric bounds.

7. **Setter + gate in one policy is awkward.** You *can*
   combine them — set labels on DENY, for instance — but the
   YAML gets confusing. Examples convention: separate setter
   policies from gate policies.

8. **No way to clear a label except violating the schema.** If
   your schema says monotonic, the label is sticky for the
   whole session. Useful for taint; bad for transient state
   that should reset. There's no explicit "clear" operation.

9. **Sub-agent label propagation is a separate, subtle
   mechanism.** When a parent spawns a sub-agent, labels merge
   via `merged_with_child` rules (defined in `datamodel.py`
   `LabelSchemaRule`). Worth understanding but outside the
   label-policy YAML surface.

10. **YAML ordering matters but is invisible at runtime.** If
    two policies set the same key in the same phase, the *last*
    one to run wins (because `set_labels` updates accumulate
    via `dict.update`). This ordering is implicit in YAML
    insertion order with no visual indicator that order
    matters.

11. **Silent rejection on schema violations.** If a policy
    tries to set a label to an invalid value or a non-monotonic
    update, the change is dropped silently (see
    `_apply_root_label_update` in `session.py`). No error, no
    warning, no log — the policy just doesn't do what the YAML
    says.

12. **No observability into evaluation.** A label policy that
    didn't fire because `condition` didn't match looks
    identical to one that did fire with `action: allow` — both
    return `ALLOW`. Debugging requires reading the condition
    logic back to YAML in your head.

---

## 6. Under the hood

`make_label_policy_callable` (policies.py:566) returns this
closure (abridged):

```python
def _evaluate(content, phase, context) -> PolicyResult:
    # Tool-name filter (tool_call phase only)
    if phase == "tool_call" and match_tools is not None:
        tool_name = content.get("tool", "") if isinstance(content, dict) else ""
        if tool_name not in match_tools:
            return PolicyResult(action=ALLOW)

    # Condition check against session labels
    if condition:
        session_labels = context.get("labels", {})
        for key, required in condition.items():
            actual = session_labels.get(key)
            if actual is None:
                return PolicyResult(action=ALLOW)
            if isinstance(required, list):
                if actual not in required:
                    return PolicyResult(action=ALLOW)
            else:
                if actual != required:
                    return PolicyResult(action=ALLOW)

    # Matched → emit the configured action + set_labels
    return PolicyResult(
        action=action,
        reason=reason or "Label policy triggered",
        set_labels=dict(set_labels_on_match),
    )
```

That's the entire implementation. Three branches:
- Tool-name filter failed → ALLOW (policy doesn't apply).
- Condition failed → ALLOW (policy doesn't apply).
- Matched → return the configured action + labels.

Whatever ALLOW gets returned in the "doesn't apply" branches
has no effect — the engine treats ALLOW as a no-op unless
`set_labels` is non-empty. So a label policy that fails its
condition contributes nothing to the evaluation.

---

## 7. Mental model

A label policy is syntax for saying:

> **when** the session looks like `condition`,
> **do** `action` (allow / ask / deny),
> **and optionally update** the session to look like `set_labels`.

Labels are the shared variables multiple policies can read and
write without knowing about each other. The whole system is
intentionally narrow — if the expressiveness isn't enough, the
escape hatch is `FunctionPolicy`, where you have full Python.
