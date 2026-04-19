# Rewriters — Content Transformation Middleware

Related: `PORTING_FROM_OMNIAGENTS.md` G1 (guardrails port),
`POLICIES_OMNIAGENTS_NOTES.md` (policy system notes),
`LABEL_POLICIES_NOTES.md` (label policy deep-dive).

---

## 1. Overview

Policies in agent-plane's guardrails port (inherited from
omniagents) make **decisions**: `ALLOW`, `ASK`, or `DENY`. They
cannot transform content. The omniagents design doc
(`DESIGN_POLICIES.md:439`) explicitly flags this as unresolved
future work:

> **`modify` action**: A policy action that rewrites content
> rather than blocking it (e.g. redacting sensitive fields
> from output). Current workaround is to deny and let the
> agent retry with different content.

"Deny-and-retry" is not a solution for redaction, PII
normalization, or prompt-injection stripping — the agent has
already produced the content; it's the guardrail that needs to
sanitize, not block.

This design adds a **separate rewriter layer** parallel to
policies, running at the same four phase boundaries (`input`,
`tool_call`, `tool_result`, `output`). Rewriters transform
content; policies decide. Keeping them separate sidesteps the
composition problem that prevented omniagents from shipping a
`modify` action in the first place: rewrites compose trivially
via sequential ordering, decisions compose via max-action
semantics, and neither has to bend to accommodate the other.

Prior art (NeMo Guardrails, Guardrails AI) confirms this split
is the industry pattern — surveyed in §11.

---

## 2. Goals / Non-goals

**Goals**

- Transform content at any of the four guardrail phases
  (`input`, `tool_call`, `tool_result`, `output`).
- Sequential composition: rewriters run in YAML order; each
  sees the previous rewriter's output.
- Rewriters run **before** policies at each phase, so policies
  see the post-rewrite content and decide based on what the
  agent will actually emit or consume.
- Same authoring model as policies: `type: function` (Python
  callable) or `type: prompt` (LLM-evaluated).
- Fail-closed: a rewriter that errors or times out treats the
  content as denied.

**Non-goals**

- **Per-chunk streaming rewrite.** Universally unsolved across
  mature frameworks. Force non-streaming mode for
  rewriter-enabled phases, or buffer to completion. See §7.
- **Label writes from rewriters.** Rewriters are pure transforms.
  If a rewrite's outcome should tag the session, a downstream
  policy evaluates the rewritten content and emits the label.
- **Parallel rewriter execution.** NeMo docs explicitly warn
  that rail mutations under parallel execution cause race
  conditions. Rewriters are strictly sequential.
- **A fourth `PolicyAction` called `REWRITE`.** Explicitly the
  wrong shape per prior-art synthesis — rewriters are a
  separate layer, not a policy extension.

---

## 3. User-Facing API

New top-level `rewriters:` section in the agent spec, parallel
to `policies:`:

```yaml
rewriters:
  redact_pii:
    type: function
    on: [output, tool_result]
    callable: myorg.rewriters.redact_pii

  strip_secrets:
    type: prompt
    on: [output]
    executor:
      model: claude-haiku-4-5
    prompt: |
      Rewrite the text to replace any API keys, passwords, or
      tokens with "<REDACTED>". Preserve all other content
      verbatim. Return only the rewritten text.

policies:
  # ... existing policies; evaluated AFTER rewriters at each phase
```

### 3.1 Function rewriter

A Python callable. Signature either
`fn(content, phase) -> content` or
`fn(content, phase, context) -> content`. Returns the
transformed content.

```python
import re
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

def redact_pii(content: str, phase: str) -> str:
    return _EMAIL_RE.sub("<EMAIL>", content)
```

Factory pattern mirrors `FunctionPolicy` — stateful rewriters
via closures or `factory_params`.

### 3.2 Prompt rewriter

An LLM-evaluated rewriter. The configured executor receives the
rewriter's `prompt` plus the current content; the executor's
response text IS the rewritten content.

The prompt must instruct the LLM to emit only the rewritten
payload (no commentary, no JSON wrapping). Fail-closed if the
LLM refuses, errors, or emits tool calls.

Same executor inheritance as `PromptPolicy`: a prompt rewriter
either declares its own executor or inherits the session's.

### 3.3 Common fields

| Field | Required | Notes |
| --- | --- | --- |
| `type` | yes | `function` or `prompt` |
| `on` | no | Defaults to `[output]` |
| `callable` | type=function | Dotted Python path, resolved at load |
| `factory_params` | no | type=function only |
| `prompt` | type=prompt | Non-empty string |
| `executor` | no | type=prompt only; defaults to session executor |

---

## 4. Evaluation Points and Content Shapes

Rewriters plug into the same four enforcement points as
policies. What each phase sees:

| Phase | Content type | What rewriter receives |
| --- | --- | --- |
| `input` | `str` | The user's message text |
| `tool_call` | `dict` | `{"tool": str, "args": dict}` |
| `tool_result` | `dict` | Tool's output payload |
| `output` | `str` | LLM response text |

Rewriter returns the same type it received (see §9 safety
constraints). For `tool_call`, the rewriter may modify `args`
but must leave `tool` unchanged.

---

## 5. Pipeline per Phase

At each of the four phases, the evaluation site does:

```
1. Collect rewriters where phase in rewriter.on, in YAML order
2. For each rewriter in sequence:
     content = await rewriter.rewrite(content, phase, context)
3. Run policy engine on the (now fully rewritten) content
4. Act on the policy decision (ALLOW / ASK / DENY) using the
   rewritten content
```

Concretely, the existing `_apply_input_policy_to_message`
becomes:

```python
async def _apply_input_policy_to_message(self, msg):
    # NEW: rewriter chain runs first
    content = msg.content
    for rw in self._rewriter_chain.for_phase("input"):
        try:
            content = await rw.rewrite(
                content, "input", self._rewriter_context(),
            )
        except Exception as exc:
            # Fail closed — treat as DENY with the rewriter
            # name in the reason
            return None, f"[DENIED by rewriter '{rw.name}': {exc}]"

    msg = replace(msg, content=content)

    # EXISTING: policies now see the rewritten content
    result = await self._policy_engine.evaluate(content, "input")
    if result.action == PolicyAction.DENY:
        return None, f"[DENIED by policy: {result.reason}]"
    if result.action == PolicyAction.ASK:
        approved, timed_out = await self._handle_ask(...)
        if not approved:
            return None, ...
    return msg, None
```

Same pattern repeats for `tool_call`, `tool_result`, and
`output` sites. Only the content type differs.

---

## 6. Composition Model

**Sequential, YAML-order, no priority.** Rewriters for a phase
run in the order they appear in the spec's `rewriters:` section.
Each rewriter receives the previous rewriter's output. No
parallel execution, no priority fields, no merge semantics.

Why only sequential:

1. **Predictability.** The YAML is the execution order; no
   hidden priority to discover.
2. **Debuggability.** Telemetry logs show the chain
   (rewriter_1 output → rewriter_2 input → rewriter_2 output →
   …). Trace is linear.
3. **Handles "rewrite then deny" for free.** Because policies
   run after the full rewriter chain, a later block policy
   sees the rewritten content and decides based on that. The
   composition problem that blocked omniagents' `modify`
   action is dissolved by construction.
4. **Prior art agrees.** NeMo's `rails.<stage>.flows` list is
   declaration-ordered and defaults to sequential; parallel
   mode is documented as unsafe with mutation. Guardrails AI's
   `Guard.use()` is declaration-ordered. Every mature rewrite-
   capable framework lands here.

**No merge of concurrent rewrites.** Unlike Guardrails AI's
`fix_value` merging for structured output, this design does not
attempt to reconcile two rewriters producing different
modifications of the same content. If two rewriters need to
interact, they run sequentially and the second sees the first's
output — full stop.

---

## 7. Streaming

**Rewriter-enabled phases force non-streaming mode** for that
phase. If any rewriter in the spec declares `output` in its
`on` list, output streaming is disabled for that agent — the
full LLM response is buffered server-side, the rewriter chain
runs end-to-end, and the final content is delivered as a single
non-streamed payload.

Rationale: per-chunk rewrite is unsolved in every surveyed
framework. NeMo's streaming rails block/pass only — no
documented chunk-level rewrite. Guardrails AI explicitly warns
"not all `on_fail` types are supported with streaming."
Introducing agent-plane's own invented chunk-rewrite protocol
is too far outside the comfort zone of proven patterns.

Agents that need streaming AND rewriting must pick:

- Accept the latency hit (response streams as one chunk after
  rewriter chain completes).
- Move the rewriting logic into a tool the agent explicitly
  calls (`rewrite_response` tool called before the agent's
  final output). No longer a transparent guardrail — the
  agent sees its own rewrite — but preserves streaming.

Later work (Open Question §12.4) may add an opt-in
`output_rewrite_buffer: true` that enables streaming but
delays delivery until the rewriter chain completes. Not in
MVP.

---

## 8. Spec Loading

New `_parse_rewriter` helper in `loader.py`, parallel to
`_parse_policy`:

```python
def _parse_rewriter(name: str, data: dict) -> Rewriter:
    rewriter_type = data.get("type", "function")
    on = data.get("on", ["output"])

    if rewriter_type == "function":
        callable_path = data["callable"]  # required, fail loud
        fn = _resolve_callable(callable_path)
        # factory_params pattern identical to FunctionPolicy
        ...
        return FunctionRewriter(name=name, on=on, callable=fn, ...)

    if rewriter_type == "prompt":
        prompt = data["prompt"]  # required, fail loud, non-empty
        executor = _parse_executor_spec(data.get("executor"))
        return PromptRewriter(
            name=name, on=on, prompt=prompt, executor=executor,
        )

    raise ValueError(f"Unknown rewriter type: {rewriter_type!r}")
```

Validation rules (enforced at load time, fail loud):

- Every rewriter must declare at least one phase in `on`.
- `type: function` rewriters must resolve the callable at load
  time.
- `type: prompt` rewriters must have a non-empty `prompt`.
- If any `output`-phase rewriter exists AND the agent spec
  declares output streaming, either fail loud or explicitly
  disable streaming (single-path decision — no silent
  auto-disable).

---

## 9. Safety Constraints

- **Tool name is immutable.** A `tool_call`-phase rewriter
  may modify `args` but must leave `tool` unchanged. Rewriting
  the tool name reroutes the LLM's intent in a way the model
  never authorized. The `FunctionRewriter` for `tool_call`
  accepts `(tool, args, phase, context)` and returns new `args`
  only — the engine reconstructs the `{"tool": ..., "args": ...}`
  dict with the original tool name.

- **Content type preservation.** A rewriter must return the
  same type it received. `str` in → `str` out; `dict` in →
  `dict` out. Type mismatch is a fail-closed error.

- **Rewriter identity in errors.** Denial sentinels include the
  offending rewriter's name
  (`[DENIED by rewriter '<name>': <reason>]`). Easy to trace in
  logs and in the user-facing denial message.

- **Timeout inheritance.** Each rewriter call inherits the
  agent's `tools.timeout`. A rewriter exceeding the timeout is
  a fail-closed DENY.

- **Labels are read-only.** Rewriters receive `context["labels"]`
  but cannot emit `set_labels`. If a rewrite's outcome should
  tag the session, add a policy that inspects the rewritten
  content.

---

## 10. Observability

Each rewriter execution emits one telemetry event:

```
event:              "rewriter.evaluated"
rewriter_name:      str
phase:              str
content_before_hash: str   # SHA-256, not raw content
content_after_hash:  str
changed:            bool
duration_ms:        int
error:              str | None
```

Hashes by default because rewriters often process sensitive
data — raw logging defeats the point of redaction. A dev-mode
spec flag `rewriters.debug_log_raw: true` enables full
content logging for local debugging, off by default.

MLflow tracing (G13): new `REWRITER` span type parallel to
`GUARDRAIL`, attached to the enclosing `AGENT` span.

---

## 11. Prior Art

### NeMo Guardrails (NVIDIA)

Rewrites are Colang subflows that reassign a well-known
variable:

```colang
define subflow mask sensitive data on output
  $bot_message = execute mask_sensitive_data(source="output",
                                             text=$bot_message)
```

`rails.<stage>.flows` is an ordered list, sequential by
default. `parallel: True` exists but docs explicitly warn it's
unsafe with mutation. Streaming rails block/pass only.
[docs/configure-rails/yaml-schema/guardrails-configuration/](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/docs/configure-rails/yaml-schema/guardrails-configuration).

**This design copies:** mutation-as-return, sequential
ordering, rewrite-before-decision placement.

### Guardrails AI

```python
Guard().use(DetectPII(pii_entities=["EMAIL_ADDRESS"], on_fail="fix"))
```

`on_fail="fix"` makes a validator return
`FailResult(fix_value=...)`; Guard substitutes the fix.
Validators stack via `.use()` ordering; multiple fixes on the
same structured field get merged.
[docs/concepts/validator_on_fail_actions](https://www.guardrailsai.com/docs/concepts/validator_on_fail_actions).

**This design copies:** per-rewriter declaration order,
fail-closed on errors (Guardrails' `exception` mode).

**This design does NOT copy:** the merge-multiple-fixes
semantic. Brittle for free text; NeMo's "each rewriter operates
on the previous one's output" is cleaner.

### Llama Guard / OpenAI Moderation / Claude streaming refusals

Classification only. No framework-level rewrite primitive. Not
direct prior art for this design.

---

## 12. Open Questions

1. **Atomic chain on failure?** If rewriter 3 of 5 raises, do
   we DENY (discard rewriter 1's and 2's outputs too) or
   commit rewriters 1–2's results and DENY the residual?
   Current proposal: atomic — the whole phase DENYs on any
   rewriter failure. Simpler; matches fail-closed principle;
   avoids persisting partial transforms that might have been
   invalidated by the rewriter that failed.

2. **Shadow / dry-run mode.** A per-rewriter `shadow: true`
   flag would log what would have changed but pass through
   original content. Useful for rolling out a new rewriter
   without affecting behavior in production. Defer until a
   real rollout needs it.

3. **Per-rewriter result caching.** A deterministic
   `redact_pii` rewriter applied to the same input across
   multiple turns wastes work. Key-cache on
   `(rewriter_name, input_hash)`. Defer; most rewriters are
   fast and caching correctness around labels/context is
   fiddly.

4. **Streaming with buffered rewrite
   (`output_rewrite_buffer: true`).** Opt-in: tokens stream,
   but delivery waits until the rewriter chain completes.
   Gives streaming UX with rewrite semantics. Trade-off: full
   response latency, no incremental display. Defer until
   demand exists.

5. **Session history access for rewriters?** Current proposal:
   no — rewriters see only current content + read-only labels.
   Adding history access would enable context-aware rewriting
   (e.g. "redact this entity because it was flagged earlier")
   but vastly increases the surface area. If needed, introduce
   a `PromptRewriter` variant with configurable history
   windowing.

6. **Rewriter-initiated ASK?** Currently forbidden — rewriters
   transform, policies decide. A rewriter that can't safely
   rewrite raises (→ DENY), it doesn't ASK. If real use cases
   push on this, the answer is probably "write a policy that
   inspects the content and returns ASK." Open question
   whether this separation holds up in practice.

7. **Can rewriters add conversation items?** E.g. a rewriter
   that redacts PII but wants to add a system message
   explaining what was removed. Current proposal: no — a
   rewriter returns only the transformed content; side effects
   on the conversation belong to a dedicated mechanism
   (possibly a future `message_injector` layer). Keep the
   rewriter boundary narrow.

8. **Composition with sub-agents.** When a sub-agent spawns,
   does it inherit the parent's rewriters? Options:
   (a) inherit unless overridden, (b) fresh chain per agent
   spec, (c) union of parent and child declared rewriters.
   Mirrors label propagation decisions; punt until the port
   of sub-agent semantics is designed.
