"""
Tests for ``_parse_guardrails`` and helpers — spec-load
behavior for the policy system (POLICIES.md §3, §14).

Covers every YAML shape from §3.1 of the design doc. The
validator-layer rejections (empty lists, typo guards, unknown
types) live in ``test_policy_validator.py``; this file only
covers the happy-path parser behavior and the string-coercion
semantics. Invalid shapes that the parser raises on directly
(malformed dicts, missing required fields) live here because
they're parse-time errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_plane.errors import AgentPlaneError
from agent_plane.spec.parser import (
    _ConfigYamlLoader,
    _parse_condition,
    _parse_guardrails,
    parse,
)
from agent_plane.spec.types import (
    DEFAULT_ASK_TIMEOUT,
    FunctionPolicySpec,
    LabelPolicySpec,
    Phase,
    PolicyAction,
    PromptPolicySpec,
)


def _yaml(text: str) -> Any:
    """
    Parse YAML text using the spec's custom loader.

    Needed so ``on:``, ``off:``, ``yes:``, ``no:`` stay strings
    (the loader strips the YAML 1.1 bool aliases — see
    :class:`_ConfigYamlLoader`). Without this, ``on: [input]``
    in test YAML would parse to ``{True: [...]}``.
    """
    return yaml.load(text, Loader=_ConfigYamlLoader)


# ── Top-level parse ────────────────────────────────────────


def test_parse_guardrails_none_returns_none() -> None:
    """Absent `guardrails:` block → ``None`` (no-op engine)."""
    assert _parse_guardrails(None) is None


def test_parse_guardrails_empty_dict_returns_empty_spec() -> None:
    """Empty guardrails block parses to a GuardrailsSpec
    with default ask_timeout and no labels / policies."""
    spec = _parse_guardrails({})
    assert spec is not None
    assert spec.ask_timeout == DEFAULT_ASK_TIMEOUT
    assert spec.labels is None
    assert spec.policies is None


def test_parse_guardrails_rejects_non_mapping() -> None:
    """`guardrails: [...]` or other non-dict → clear error."""
    with pytest.raises(AgentPlaneError, match="guardrails: must be a mapping"):
        _parse_guardrails([1, 2, 3])  # type: ignore[arg-type]


def test_parse_guardrails_ask_timeout_override() -> None:
    """Author-supplied `ask_timeout:` overrides the default."""
    spec = _parse_guardrails({"ask_timeout": 60})
    assert spec is not None
    assert spec.ask_timeout == 60


def test_parse_guardrails_ask_timeout_zero_rejected() -> None:
    """`ask_timeout: 0` → fail-loud at spec load (POLICIES.md §13).

    A zero timeout is ambiguous (instant-DENY vs. wait-forever).
    If this test starts passing silently with ``ask_timeout=0``
    on the result, the §13 rejection was removed — that would
    let broken configs reach runtime.
    """
    with pytest.raises(AgentPlaneError, match="ask_timeout must be > 0"):
        _parse_guardrails({"ask_timeout": 0})


def test_parse_guardrails_ask_timeout_negative_rejected() -> None:
    """Negative ask_timeout → same §13 rejection."""
    with pytest.raises(AgentPlaneError, match="ask_timeout must be > 0"):
        _parse_guardrails({"ask_timeout": -5})


def test_parse_guardrails_ask_timeout_non_integer_rejected() -> None:
    """Non-integer ask_timeout → loud error (no silent coercion)."""
    with pytest.raises(AgentPlaneError, match="ask_timeout must be an integer"):
        _parse_guardrails({"ask_timeout": "soon"})


# ── Label definitions ──────────────────────────────────────


def test_parse_labels_bare_string_shorthand() -> None:
    """`integrity: "1"` → LabelDef(initial="1", values=None)."""
    spec = _parse_guardrails(_yaml('labels: {integrity: "1"}'))
    assert spec is not None and spec.labels is not None
    d = spec.labels["integrity"]
    # Bare-string shorthand sets only `initial`; no schema declared.
    assert d.initial == "1"
    assert d.values is None
    assert d.monotonic is None


def test_parse_labels_schema_with_initial() -> None:
    """Full-schema dict: initial + values + monotonic all land."""
    spec = _parse_guardrails(
        _yaml("""
labels:
  sensitivity:
    initial: public
    values: [public, internal, confidential]
    monotonic: increasing
""")
    )
    assert spec is not None and spec.labels is not None
    d = spec.labels["sensitivity"]
    assert d.initial == "public"
    assert d.values == ["public", "internal", "confidential"]
    assert d.monotonic == "increasing"


def test_parse_labels_schema_without_initial() -> None:
    """`{values: [...], monotonic: ...}` without initial —
    label is unset until a policy writes it (§10)."""
    spec = _parse_guardrails(
        _yaml("""
labels:
  role:
    values: [admin, user]
""")
    )
    assert spec is not None and spec.labels is not None
    d = spec.labels["role"]
    assert d.initial is None
    assert d.values == ["admin", "user"]


def test_parse_labels_empty_dict_rejected() -> None:
    """`integrity: {}` → typo guard (POLICIES.md §13).

    An empty dict declaring neither `initial`, `values`, nor
    `monotonic` is almost always an unfinished edit.
    """
    with pytest.raises(AgentPlaneError, match="empty dict"):
        _parse_guardrails(_yaml("labels: {integrity: {}}"))


def test_parse_labels_monotonic_without_values_rejected() -> None:
    """`monotonic` without `values` has no positions to order."""
    with pytest.raises(AgentPlaneError, match="monotonic.*requires a .values. list"):
        _parse_guardrails(
            _yaml("""
labels:
  bad:
    monotonic: increasing
""")
        )


def test_parse_labels_initial_not_in_values_rejected() -> None:
    """`initial: "5"` with `values: ["1", "2"]` → fail at load."""
    with pytest.raises(AgentPlaneError, match="initial.*not in declared .values."):
        _parse_guardrails(
            _yaml("""
labels:
  level:
    initial: "5"
    values: ["1", "2"]
""")
        )


def test_parse_labels_monotonic_unknown_direction_rejected() -> None:
    """`monotonic: up` → clear error listing the two valid values."""
    with pytest.raises(AgentPlaneError, match="must be 'increasing' or 'decreasing'"):
        _parse_guardrails(
            _yaml("""
labels:
  x:
    values: ["0", "1"]
    monotonic: up
""")
        )


def test_parse_labels_values_non_list_rejected() -> None:
    """`values: 1` → clear error (must be a list)."""
    with pytest.raises(AgentPlaneError, match="values. must be a list"):
        _parse_guardrails(
            _yaml("""
labels:
  x:
    values: 1
""")
        )


# ── Policy types — FunctionPolicy ──────────────────────────


def test_parse_function_policy_short_form() -> None:
    """Bare-string `function:` path → FunctionRef with no arguments."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  simple:
    type: function
    on: [input]
    function: myorg.policies.check
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, FunctionPolicySpec)
    assert p.name == "simple"
    assert p.function is not None
    assert p.function.path == "myorg.policies.check"
    assert p.function.arguments is None


def test_parse_function_policy_dict_form_with_arguments() -> None:
    """`function: {path, arguments}` → factory form."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  rate:
    type: function
    on: [tool_call]
    function:
      path: myorg.policies.rate_limit
      arguments:
        limit: 10
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, FunctionPolicySpec)
    assert p.function is not None
    assert p.function.path == "myorg.policies.rate_limit"
    # arguments are stored as-is for the factory to consume.
    assert p.function.arguments == {"limit": 10}


def test_parse_function_policy_missing_function_rejected() -> None:
    """Function policy without `function:` field → loud error."""
    with pytest.raises(AgentPlaneError, match="function. policies require"):
        _parse_guardrails(
            _yaml("""
policies:
  broken:
    type: function
    on: [input]
""")
        )


def test_parse_function_policy_dict_missing_path_rejected() -> None:
    """`function: {arguments: {...}}` (no path) → loud error."""
    with pytest.raises(AgentPlaneError, match="function.path. must be a"):
        _parse_guardrails(
            _yaml("""
policies:
  broken:
    type: function
    on: [input]
    function:
      arguments: {limit: 1}
""")
        )


def test_parse_function_policy_arguments_non_dict_rejected() -> None:
    """`function.arguments: [1, 2]` → loud error."""
    with pytest.raises(AgentPlaneError, match="function.arguments. must be a mapping"):
        _parse_guardrails(
            _yaml("""
policies:
  broken:
    type: function
    on: [input]
    function:
      path: myorg.x
      arguments: [1, 2]
""")
        )


def test_parse_function_policy_optional_action_list() -> None:
    """`action: [allow, deny]` declares the whitelist."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: function
    on: [tool_call]
    function: myorg.p.check
    action: [allow, deny]
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, FunctionPolicySpec)
    assert p.action == [PolicyAction.ALLOW, PolicyAction.DENY]


def test_parse_function_policy_action_omitted_is_none() -> None:
    """No `action:` → `None` (accept any action)."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: function
    on: [input]
    function: myorg.p.check
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, FunctionPolicySpec)
    assert p.action is None


# ── Policy types — PromptPolicy ────────────────────────────


def test_parse_prompt_policy_minimal() -> None:
    """Default action list is [allow, deny] per §9.2 default."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  canada:
    type: prompt
    on: [input]
    prompt: Deny if user mentions Canada.
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, PromptPolicySpec)
    assert p.prompt is not None and "Canada" in p.prompt
    # Default action list per PromptPolicySpec default_factory.
    assert p.action == [PolicyAction.ALLOW, PolicyAction.DENY]
    assert p.llm is None
    assert p.set_labels is None


def test_parse_prompt_policy_classifier_only() -> None:
    """`action: [allow]` declares a classifier that never blocks
    (carve-out in §13 — LLM failures substitute ALLOW)."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  classify:
    type: prompt
    on: [tool_result:read_doc]
    action: [allow]
    set_labels: [sensitivity]
    prompt: Classify document sensitivity.
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, PromptPolicySpec)
    assert p.action == [PolicyAction.ALLOW]
    assert p.set_labels == ["sensitivity"]


def test_parse_prompt_policy_with_llm_override() -> None:
    """Per-policy `llm:` override parses into nested LLMConfig."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: prompt
    on: [input]
    prompt: Deny bad stuff.
    llm:
      model: anthropic/claude-haiku-4-5
      request_timeout: 10
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, PromptPolicySpec)
    assert p.llm is not None
    assert p.llm.model == "anthropic/claude-haiku-4-5"
    # Per-policy request_timeout override lands on the nested config.
    assert p.llm.request_timeout == 10


def test_parse_prompt_policy_missing_prompt_rejected() -> None:
    """PromptPolicy without a prompt is meaningless → loud error."""
    with pytest.raises(AgentPlaneError, match="prompt. policies require a non-empty"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: prompt
    on: [input]
""")
        )


def test_parse_prompt_policy_empty_prompt_rejected() -> None:
    """Whitespace-only prompt also rejected."""
    with pytest.raises(AgentPlaneError, match="non-empty"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: prompt
    on: [input]
    prompt: "   "
""")
        )


# ── Policy types — LabelPolicy ─────────────────────────────


def test_parse_label_policy_minimal_deny() -> None:
    """Label policy with action=deny + reason."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  block_shell:
    type: label
    on: [tool_call:run_shell]
    condition:
      integrity: "0"
    action: deny
    reason: Untrusted content plus shell is disallowed.
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, LabelPolicySpec)
    assert p.action == PolicyAction.DENY
    assert p.reason == "Untrusted content plus shell is disallowed."
    assert p.condition == {"integrity": "0"}


def test_parse_label_policy_sets_labels() -> None:
    """`type: label` with `set_labels:` — the canonical
    taint pattern from §9.3."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  taint_web:
    type: label
    on: [tool_call:web_search]
    action: allow
    set_labels:
      integrity: "0"
""")
    )
    assert spec is not None and spec.policies is not None
    p = spec.policies[0]
    assert isinstance(p, LabelPolicySpec)
    # set_labels is a dict[str, str] for label policies —
    # distinct from the list form on function / prompt.
    assert p.set_labels == {"integrity": "0"}


def test_parse_label_policy_missing_action_rejected() -> None:
    """Label policies require an explicit `action:` field per
    POLICIES.md §9.3 — it's the core of what they declare.
    Without this rejection, a label policy would silently
    never fire."""
    with pytest.raises(AgentPlaneError, match="label. policies require an .action"):
        _parse_guardrails(
            _yaml("""
policies:
  bad:
    type: label
    on: [input]
""")
        )


def test_parse_label_policy_invalid_action_rejected() -> None:
    """`action: bogus` must fail loud at spec load. Silent
    coercion (defaulting to ALLOW, etc.) would make typos
    invisible and let broken specs reach runtime."""
    with pytest.raises(AgentPlaneError, match="must be one of"):
        _parse_guardrails(
            _yaml("""
policies:
  bad:
    type: label
    on: [input]
    action: bogus
""")
        )


def test_parse_label_policy_set_labels_non_mapping_rejected() -> None:
    """Label policy `set_labels` must be a dict, not a list."""
    with pytest.raises(AgentPlaneError, match="set_labels. must be a mapping"):
        _parse_guardrails(
            _yaml("""
policies:
  bad:
    type: label
    on: [input]
    action: allow
    set_labels: [integrity]
""")
        )


# ── `on:` selector parsing ─────────────────────────────────


def test_parse_on_wildcard() -> None:
    """`on: [tool_call]` → wildcard for the phase."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: label
    on: [tool_call]
    action: allow
""")
    )
    p = spec.policies[0]
    assert len(p.on) == 1
    # Wildcard selector — no tool_name narrows.
    assert p.on[0].phase == Phase.TOOL_CALL
    assert p.on[0].tool_name is None


def test_parse_on_narrowed_by_tool_name() -> None:
    """`tool_call:web_search` → narrowed selector."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: label
    on: [tool_call:web_search]
    action: allow
""")
    )
    sel = spec.policies[0].on[0]
    assert sel.phase == Phase.TOOL_CALL
    assert sel.tool_name == "web_search"


def test_parse_on_multi_phase() -> None:
    """Multiple entries produce one selector each."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: label
    on: [input, output]
    action: allow
""")
    )
    phases = [s.phase for s in spec.policies[0].on]
    assert phases == [Phase.INPUT, Phase.OUTPUT]


def test_parse_on_empty_list_rejected() -> None:
    """`on: []` → rejected at spec load (§13 — never fires)."""
    with pytest.raises(AgentPlaneError, match="must contain at least one"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: label
    on: []
    action: allow
""")
        )


def test_parse_on_unknown_phase_rejected() -> None:
    """Unknown phase name → clear error naming the offender."""
    with pytest.raises(AgentPlaneError, match="unknown phase"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: label
    on: [middle]
    action: allow
""")
        )


def test_parse_on_tool_narrowing_on_input_rejected() -> None:
    """`input:foo` is meaningless — tool filters only on tool_call / tool_result."""
    with pytest.raises(AgentPlaneError, match="cannot be narrowed by tool name"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: label
    on: ["input:foo"]
    action: allow
""")
        )


def test_parse_on_empty_tool_name_rejected() -> None:
    """`tool_call:` (trailing colon, no name) → clear error."""
    with pytest.raises(AgentPlaneError, match="empty tool name"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: label
    on: ["tool_call:"]
    action: allow
""")
        )


def test_parse_on_non_list_rejected() -> None:
    """`on: tool_call` (bare string, not list) → loud error."""
    with pytest.raises(AgentPlaneError, match="must be a list"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: label
    on: tool_call
    action: allow
""")
        )


# ── Policy defaults + ordering ─────────────────────────────


def test_parse_policy_default_on_is_input_and_output() -> None:
    """When `on:` is omitted, parser defaults to input + output
    (POLICIES.md §3.1)."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  p:
    type: prompt
    prompt: Test.
""")
    )
    phases = [s.phase for s in spec.policies[0].on]
    # §3.1 declared default — classifier phases.
    assert phases == [Phase.INPUT, Phase.OUTPUT]


def test_parse_policies_preserve_yaml_order() -> None:
    """Policies land in the list in their YAML declaration
    order — the engine iterates in this order per §4. If
    this breaks, DENY short-circuiting and ASK ordering
    would both silently reorder."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  first:
    type: label
    on: [input]
    action: allow
  second:
    type: label
    on: [input]
    action: allow
  third:
    type: label
    on: [input]
    action: allow
""")
    )
    names = [p.name for p in spec.policies]
    assert names == ["first", "second", "third"]


def test_parse_policy_unknown_type_rejected() -> None:
    """`type: weird` → clear error listing the three accepted values."""
    with pytest.raises(AgentPlaneError, match="must be 'function', 'prompt', or 'label'"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: weird
    on: [input]
""")
        )


def test_parse_policy_missing_type_rejected() -> None:
    """Every policy must declare `type:` — the dispatcher
    uses it to pick the concrete `PolicySpec` subclass. A
    missing type is an unfinished edit, not a default."""
    with pytest.raises(AgentPlaneError, match="missing required field .type"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    on: [input]
""")
        )


# ── Per-policy `ask_timeout` override ──────────────────────


def test_parse_per_policy_ask_timeout() -> None:
    """A policy may override the spec-wide `ask_timeout:` via
    its own field — validates the per-policy override wiring
    lands on `PolicySpec.ask_timeout`. Needed so `_await_policy_approval`
    can read the override off the deciding policy's spec."""
    spec = _parse_guardrails(
        _yaml("""
policies:
  long_review:
    type: label
    on: [output]
    action: ask
    ask_timeout: 300
""")
    )
    assert spec.policies[0].ask_timeout == 300


def test_parse_per_policy_ask_timeout_zero_rejected() -> None:
    """Per-policy `ask_timeout: 0` → same §13 rejection as
    spec-level: the zero-is-ambiguous rule applies at both
    layers. If only the spec-level check existed, authors
    could smuggle a broken `0` through a per-policy override."""
    with pytest.raises(AgentPlaneError, match="ask_timeout. must be > 0"):
        _parse_guardrails(
            _yaml("""
policies:
  p:
    type: label
    on: [input]
    action: allow
    ask_timeout: 0
""")
        )


# ── Integration with top-level parse() ─────────────────────


@pytest.fixture()
def agent_dir_with_guardrails(tmp_path: Path) -> Path:
    """Write a full config.yaml that exercises the
    guardrails block alongside other top-level sections.

    This is the closest thing to an integration test for
    Phase 0 — verifies the full parse() path wires
    guardrails into AgentSpec without breaking other
    sections.
    """
    (tmp_path / "config.yaml").write_text("""
spec_version: 1
name: guardrails-demo
llm:
  model: openai/gpt-4o
guardrails:
  labels:
    integrity: "1"
  policies:
    taint_web:
      type: label
      on: [tool_call:web_search]
      action: allow
      set_labels:
        integrity: "0"
  ask_timeout: 45
""")
    return tmp_path


def test_full_parse_loads_guardrails(agent_dir_with_guardrails: Path) -> None:
    """Top-level parse() populates AgentSpec.guardrails."""
    spec = parse(agent_dir_with_guardrails)
    assert spec.guardrails is not None
    assert spec.guardrails.ask_timeout == 45
    assert spec.guardrails.labels is not None
    assert spec.guardrails.labels["integrity"].initial == "1"
    assert len(spec.guardrails.policies) == 1
    p = spec.guardrails.policies[0]
    assert isinstance(p, LabelPolicySpec)
    assert p.name == "taint_web"
    # on: [tool_call:web_search] — tool-narrowed selector.
    assert p.on[0].phase == Phase.TOOL_CALL
    assert p.on[0].tool_name == "web_search"


def test_full_parse_without_guardrails(tmp_path: Path) -> None:
    """AgentSpec.guardrails is None when the block is absent —
    runtime builds a no-op engine (§10 zero-policy case)."""
    (tmp_path / "config.yaml").write_text("""
spec_version: 1
name: no-guardrails
""")
    spec = parse(tmp_path)
    assert spec.guardrails is None


# ── YAML 1.1 `on:` trap regression guard ───────────────────


def test_yaml_on_key_stays_string() -> None:
    """The custom ``_ConfigYamlLoader`` must NOT convert
    ``on:`` into a boolean key. If this regresses, every
    policy in a real config.yaml would silently lose its
    phase selectors (``on: [input]`` → ``True: [...]``).

    This is the single most load-bearing invariant of the
    policy spec parser — document explicitly so any future
    loader change breaks loudly here.
    """
    raw = _yaml("""
policies:
  p:
    type: label
    on: [input, output]
    action: allow
""")
    # `on` MUST be a string key. If this assertion fails, the
    # loader reverted to PyYAML's default YAML 1.1 bool
    # aliases — ``on`` / ``off`` / ``yes`` / ``no`` would
    # be coerced to booleans.
    assert "on" in raw["policies"]["p"]
    assert True not in raw["policies"]["p"]


def test_yaml_true_false_still_booleans() -> None:
    """The narrowing must not break ``true`` / ``false`` —
    other parts of the spec rely on YAML booleans parsing
    as Python booleans."""
    raw = _yaml("enabled: true\ndisabled: false")
    # These must remain booleans (the rest of the spec
    # consumes them via ``bool(...)``).
    assert raw["enabled"] is True
    assert raw["disabled"] is False


# ── `_parse_condition` string coercion ─────────────────────


def test_parse_condition_none() -> None:
    """Omitted condition → None (always-match)."""
    assert _parse_condition(None, policy_name="p") is None


def test_parse_condition_empty_dict_rejected() -> None:
    """`condition: {}` → §13 typo guard."""
    with pytest.raises(AgentPlaneError, match="rejected"):
        _parse_condition({}, policy_name="p")


def test_parse_condition_scalar_values_coerced_to_string() -> None:
    """Unquoted YAML ints / bools coerce to strings — labels
    are always string-valued in storage, so a condition
    written as ``{integrity: 0}`` would otherwise silently
    mismatch the stored ``"0"``. See POLICIES.md §14."""
    out = _parse_condition({"integrity": 0}, policy_name="p")
    # Key stays string (already is); value coerced from int
    # 0 to "0" to match stored label format.
    assert out == {"integrity": "0"}


def test_parse_condition_list_values_coerced_to_strings() -> None:
    """List-of-values condition → every element coerced."""
    out = _parse_condition({"role": ["admin", 1, True]}, policy_name="p")
    # Every element becomes its str() form — covers mixed
    # YAML shapes like ``[admin, 1]`` where `admin` is already
    # a string but `1` is an int.
    assert out == {"role": ["admin", "1", "True"]}


def test_parse_condition_non_mapping_rejected() -> None:
    """`condition: [foo, bar]` → loud rejection. Only a dict
    makes sense for a label-gate (key = label name, value =
    expected value or whitelist)."""
    with pytest.raises(AgentPlaneError, match="must be a mapping"):
        _parse_condition(["integrity", "0"], policy_name="p")  # type: ignore[arg-type]
