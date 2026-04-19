"""
Tests for :class:`FunctionPolicy` (Phase 4).

Ports and extends these omniagents cases:

From ``test_policies.py``:
- ``test_allow_by_default`` — empty FunctionPolicy → ALLOW
- ``test_sync_callable_block`` — sync DENY via callable
- ``test_sync_callable_allow`` — sync ALLOW via lambda
- ``test_async_callable`` — async def evaluator
- ``test_callable_returns_dict`` — dict return parses
- ``test_deny_action_from_dict`` — string 'deny' in dict
- ``test_tool_call_rate_limit`` — closure rate-limit policy

From ``test_labels_and_policies.py`` (FunctionPolicy-context):
- ``test_three_arg_callable_receives_context``
- ``test_three_arg_callable_reads_labels_for_decision``
- ``test_three_arg_async_callable``
- ``test_rate_limit_counter_isolated``
- ``test_zero_arg_factory_copy_creates_fresh_state``

Plus Phase 4 carve-outs:
- Exception → DENY (fail-closed)
- Exception with classifier-only action list → ALLOW substituted
- Action whitelist validation
- set_labels whitelist filtering
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from agent_plane.runtime.policies.engine import PolicyEngine
from agent_plane.runtime.policies.function import (
    FunctionPolicy,
    resolve_function_policy,
)
from agent_plane.runtime.policies.label import LabelPolicy
from agent_plane.spec.types import (
    EvaluationContext,
    FunctionPolicySpec,
    FunctionRef,
    LabelPolicySpec,
    Phase,
    PhaseSelector,
    PolicyAction,
    PolicyResult,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _install_module(tmp_path: Path, module_name: str, source: str) -> None:
    """
    Write a Python module into a tmp dir and make it importable.

    Used by tests that need to exercise
    ``resolve_function_policy`` — the real code path that
    production YAMLs go through.
    """
    pkg_dir = tmp_path / "test_fn_policy_pkg"
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / f"{module_name}.py").write_text(textwrap.dedent(source))
    sys.path.insert(0, str(tmp_path))


@pytest.fixture(autouse=True)
def _cleanup_sys_path(tmp_path: Path) -> None:
    """
    Remove any tmp-path entries we inserted after each test.

    Without this, successive tests could pick up a stale
    module with the same name from a previous test's tmp_path.
    """
    yield
    path_str = str(tmp_path)
    while path_str in sys.path:
        sys.path.remove(path_str)
    # Drop the cached package so re-use of the name in
    # another test (with different source) is a clean import.
    for mod_name in list(sys.modules):
        if mod_name.startswith("test_fn_policy_pkg"):
            del sys.modules[mod_name]


def _spec(
    *,
    name: str = "p",
    phase: Phase = Phase.INPUT,
    tool_name: str | None = None,
    function: FunctionRef | None = None,
    action: list[PolicyAction] | None = None,
    set_labels: list[str] | None = None,
) -> FunctionPolicySpec:
    """Build a FunctionPolicySpec with sensible defaults."""
    return FunctionPolicySpec(
        name=name,
        on=[PhaseSelector(phase=phase, tool_name=tool_name)],
        function=function or FunctionRef(path="test_fn_policy_pkg.probe.noop"),
        action=action,
        set_labels=set_labels,
    )


def _build_engine(
    store: SqlAlchemyConversationStore,
    policies: list,
    *,
    initial_labels: dict[str, str] | None = None,
) -> PolicyEngine:
    """Build a PolicyEngine + fresh conversation for tests."""
    conv = store.create_conversation()
    return PolicyEngine(
        policies=policies,
        label_defs={},
        ask_timeout=30,
        conversation_id=conv.id,
        initial_labels=initial_labels or {},
        conversation_store=store,
    )


# ── Direct FunctionPolicy (no dotted-path resolution) ──


@pytest.mark.asyncio
async def test_sync_callable_allow() -> None:
    """Ports omniagents ``test_sync_callable_allow``. A sync
    lambda that returns PolicyResult(ALLOW) produces ALLOW."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        return PolicyResult(action=PolicyAction.ALLOW)

    policy = FunctionPolicy(_spec(), fn)
    result = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="hi"),
        {"labels": {}, "conversation_id": "c"},
    )
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_sync_callable_block() -> None:
    """Ports omniagents ``test_sync_callable_block``. A sync
    function that returns DENY blocks."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        if isinstance(ctx.content, str) and "badword" in ctx.content:
            return PolicyResult(action=PolicyAction.DENY, reason="Profanity")
        return PolicyResult(action=PolicyAction.ALLOW)

    policy = FunctionPolicy(_spec(), fn)
    result = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="has badword here"),
        {"labels": {}, "conversation_id": "c"},
    )
    assert result.action == PolicyAction.DENY
    assert result.reason == "Profanity"


@pytest.mark.asyncio
async def test_async_callable() -> None:
    """Ports omniagents ``test_async_callable``. An async
    def evaluator works identically to sync."""

    async def fn(ctx: EvaluationContext) -> PolicyResult:
        return PolicyResult(action=PolicyAction.ALLOW)

    policy = FunctionPolicy(_spec(), fn)
    result = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {}, "conversation_id": "c"},
    )
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_callable_returns_dict_allow() -> None:
    """Ports omniagents ``test_callable_returns_dict``. A
    dict return with string action parses into PolicyResult."""
    policy = FunctionPolicy(
        _spec(),
        lambda ctx: {"action": "allow"},
    )
    result = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {}, "conversation_id": "c"},
    )
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_callable_returns_dict_deny_with_reason() -> None:
    """Ports omniagents ``test_deny_action_from_dict``. A
    dict return with explicit deny."""
    policy = FunctionPolicy(
        _spec(),
        lambda ctx: {"action": "deny", "reason": "policy says no"},
    )
    result = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {}, "conversation_id": "c"},
    )
    assert result.action == PolicyAction.DENY
    assert result.reason == "policy says no"


@pytest.mark.asyncio
async def test_callable_returns_dict_with_set_labels() -> None:
    """A dict return may carry set_labels. Verifies the dict
    coercion path doesn't drop the label writes."""
    policy = FunctionPolicy(
        _spec(),
        lambda ctx: {
            "action": "allow",
            "set_labels": {"integrity": "0"},
        },
    )
    result = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {}, "conversation_id": "c"},
    )
    assert result.action == PolicyAction.ALLOW
    assert result.set_labels == {"integrity": "0"}


@pytest.mark.asyncio
async def test_two_arg_callable_receives_context() -> None:
    """Ports omniagents
    ``test_three_arg_callable_receives_context`` (ours is 2-arg
    because we fold content+phase into EvaluationContext)."""
    captured: dict[str, Any] = {}

    def fn(ctx: EvaluationContext, context: dict[str, Any]) -> PolicyResult:
        captured.update(context)
        return PolicyResult(action=PolicyAction.ALLOW)

    policy = FunctionPolicy(_spec(), fn)
    await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {"integrity": "1"}, "conversation_id": "conv_42"},
    )
    # The callable observed the engine context bundle exactly.
    assert captured == {"labels": {"integrity": "1"}, "conversation_id": "conv_42"}


@pytest.mark.asyncio
async def test_two_arg_callable_reads_labels_for_decision() -> None:
    """Ports omniagents ``test_three_arg_callable_reads_labels_for_decision``.
    A policy whose decision depends on the current label state
    reads it through the context bundle."""

    def fn(ctx: EvaluationContext, context: dict[str, Any]) -> PolicyResult:
        labels = context["labels"]
        if labels.get("integrity") == "0":
            return PolicyResult(action=PolicyAction.DENY, reason="tainted")
        return PolicyResult(action=PolicyAction.ALLOW)

    policy = FunctionPolicy(_spec(), fn)
    # Pass tainted labels.
    tainted = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {"integrity": "0"}, "conversation_id": "c"},
    )
    assert tainted.action == PolicyAction.DENY

    # Pass clean labels.
    clean = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {"integrity": "1"}, "conversation_id": "c"},
    )
    assert clean.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_async_two_arg_callable() -> None:
    """Ports omniagents ``test_three_arg_async_callable``.
    Async callables also receive context."""

    async def fn(ctx: EvaluationContext, context: dict[str, Any]) -> PolicyResult:
        labels = context["labels"]
        if labels.get("blocked") == "1":
            return PolicyResult(action=PolicyAction.DENY, reason="blocked")
        return PolicyResult(action=PolicyAction.ALLOW)

    policy = FunctionPolicy(_spec(), fn)
    r = await policy.evaluate(
        EvaluationContext(phase=Phase.INPUT, content="x"),
        {"labels": {"blocked": "1"}, "conversation_id": "c"},
    )
    assert r.action == PolicyAction.DENY


# ── Rate-limit closure (the load-bearing §9.1 example) ─


@pytest.mark.asyncio
async def test_rate_limit_closure_counts() -> None:
    """Ports omniagents ``test_tool_call_rate_limit``. A
    closure counter ticks across evaluations in the same
    workflow. Without this, stateful FunctionPolicies are
    useless."""

    def rate_limit_search(limit: int = 3) -> Any:
        calls = 0

        def _eval(ctx: EvaluationContext) -> PolicyResult:
            nonlocal calls
            calls += 1
            if calls > limit:
                return PolicyResult(
                    action=PolicyAction.DENY,
                    reason=f"Rate limit {limit} exceeded",
                )
            return PolicyResult(action=PolicyAction.ALLOW)

        return _eval

    policy = FunctionPolicy(
        _spec(phase=Phase.TOOL_CALL, tool_name="web_search"),
        rate_limit_search(limit=3),
    )
    # First 3 calls ALLOW.
    for _ in range(3):
        r = await policy.evaluate(
            EvaluationContext(
                phase=Phase.TOOL_CALL,
                content={"tool": "web"},
                tool_name="web_search",
            ),
            {"labels": {}, "conversation_id": "c"},
        )
        assert r.action == PolicyAction.ALLOW
    # 4th denies.
    r = await policy.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_CALL,
            content={"tool": "web"},
            tool_name="web_search",
        ),
        {"labels": {}, "conversation_id": "c"},
    )
    assert r.action == PolicyAction.DENY


# ── Factory resolution (the dict-form YAML path) ───────


def test_resolve_function_policy_short_form(tmp_path: Path) -> None:
    """Short-form: `function: module.attr` → the attr IS
    the evaluator."""
    _install_module(
        tmp_path,
        "probe",
        """
        from agent_plane.spec.types import PolicyAction, PolicyResult

        def noop(ctx):
            return PolicyResult(action=PolicyAction.ALLOW)
        """,
    )
    spec = FunctionPolicySpec(
        name="p",
        on=[PhaseSelector(phase=Phase.INPUT)],
        function=FunctionRef(path="test_fn_policy_pkg.probe.noop"),
    )
    policy = resolve_function_policy(spec)
    # The policy is an instance, spec bound, callable ready.
    assert isinstance(policy, FunctionPolicy)
    assert policy.spec is spec


def test_resolve_function_policy_factory_form(tmp_path: Path) -> None:
    """Dict-form: `function: {path, arguments}` → path is a
    factory. The factory runs once at build time, returning
    the evaluator. Closure state is per-workflow."""
    _install_module(
        tmp_path,
        "probe_factory",
        """
        from agent_plane.spec.types import PolicyAction, PolicyResult

        def make(limit):
            calls = 0
            def _eval(ctx):
                nonlocal calls
                calls += 1
                if calls > limit:
                    return PolicyResult(action=PolicyAction.DENY)
                return PolicyResult(action=PolicyAction.ALLOW)
            return _eval
        """,
    )
    spec = FunctionPolicySpec(
        name="p",
        on=[PhaseSelector(phase=Phase.INPUT)],
        function=FunctionRef(
            path="test_fn_policy_pkg.probe_factory.make",
            arguments={"limit": 2},
        ),
    )
    policy = resolve_function_policy(spec)
    assert isinstance(policy, FunctionPolicy)


@pytest.mark.asyncio
async def test_factory_closure_counter_isolated_per_build(
    tmp_path: Path,
) -> None:
    """Ports omniagents ``test_rate_limit_counter_isolated``.
    Two separate FunctionPolicy builds from the same factory
    have independent closure state — if this regresses,
    rate limits for different agents (or different workflows
    of the same agent) would pool into one counter."""
    _install_module(
        tmp_path,
        "probe_iso",
        """
        from agent_plane.spec.types import PolicyAction, PolicyResult

        def make(limit):
            calls = 0
            def _eval(ctx):
                nonlocal calls
                calls += 1
                if calls > limit:
                    return PolicyResult(action=PolicyAction.DENY)
                return PolicyResult(action=PolicyAction.ALLOW)
            return _eval
        """,
    )
    spec = FunctionPolicySpec(
        name="p",
        on=[PhaseSelector(phase=Phase.INPUT)],
        function=FunctionRef(
            path="test_fn_policy_pkg.probe_iso.make",
            arguments={"limit": 1},
        ),
    )
    policy_a = resolve_function_policy(spec)
    policy_b = resolve_function_policy(spec)

    ctx = EvaluationContext(phase=Phase.INPUT, content="x")
    context = {"labels": {}, "conversation_id": "c"}

    # A: 1 ALLOW then DENY (limit=1).
    assert (await policy_a.evaluate(ctx, context)).action == PolicyAction.ALLOW
    assert (await policy_a.evaluate(ctx, context)).action == PolicyAction.DENY
    # B starts fresh — its first call is ALLOW even though
    # A has already exhausted its counter.
    assert (await policy_b.evaluate(ctx, context)).action == PolicyAction.ALLOW


# ── Engine-level FunctionPolicy dispatch ──────────────


@pytest.mark.asyncio
async def test_function_policy_exception_fails_closed_to_deny(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """A callable that raises → engine coerces to DENY with
    the exception message in reason. Critical safety property
    — a broken callable must not silently ALLOW."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        raise RuntimeError("crashed")

    policy = FunctionPolicy(_spec(), fn)
    engine = _build_engine(conversation_store, [policy])
    result = await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    assert result.action == PolicyAction.DENY
    # Reason contains both the policy name and the exception.
    assert "crashed" in result.reason
    assert "p" in result.reason  # policy name


@pytest.mark.asyncio
async def test_function_policy_exception_with_classifier_only_substitutes_allow(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """POLICIES.md §13 classifier-only carve-out: when the
    spec's action list contains no DENY, a raising callable
    becomes ALLOW instead of DENY. Honors the author's
    declared 'this policy never blocks' intent."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        raise RuntimeError("crashed")

    policy = FunctionPolicy(_spec(action=[PolicyAction.ALLOW]), fn)
    engine = _build_engine(conversation_store, [policy])
    result = await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    # Engine substituted ALLOW because DENY is not in the
    # declared action list.
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_function_policy_returns_action_outside_whitelist_fails_closed(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Callable returns ASK, but the spec declared only
    [allow, deny] — engine fail-closes to DENY."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        return PolicyResult(action=PolicyAction.ASK, reason="uncertain")

    policy = FunctionPolicy(
        _spec(action=[PolicyAction.ALLOW, PolicyAction.DENY]),
        fn,
    )
    engine = _build_engine(conversation_store, [policy])
    result = await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    assert result.action == PolicyAction.DENY
    # Reason names the violation explicitly so operators can
    # debug the misbehaving callable.
    assert "not in its declared action list" in result.reason


@pytest.mark.asyncio
async def test_function_policy_set_labels_whitelist_drops_extras(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Spec declares `set_labels: [integrity]`; callable
    returns extra keys → engine filters them out silently
    (POLICIES.md §9.2 on the prompt-policy path but applies
    uniformly here per §4 step 5)."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        return PolicyResult(
            action=PolicyAction.ALLOW,
            set_labels={"integrity": "0", "stealthy_key": "bad"},
        )

    policy = FunctionPolicy(_spec(set_labels=["integrity"]), fn)
    engine = _build_engine(conversation_store, [policy])
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    # Hot cache reflects only the whitelisted key.
    assert engine.labels == {"integrity": "0"}
    # Persisted state matches — the stealthy_key never
    # touched the store.
    conv = conversation_store.get_conversation(engine.conversation_id)
    assert conv is not None
    assert conv.labels == {"integrity": "0"}


@pytest.mark.asyncio
async def test_function_policy_without_whitelist_writes_freely(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """When the spec does NOT declare `set_labels`, every
    key the callable writes lands (schemaless semantics,
    matches omniagents parity)."""

    def fn(ctx: EvaluationContext) -> PolicyResult:
        return PolicyResult(
            action=PolicyAction.ALLOW,
            set_labels={"any": "value", "other": "thing"},
        )

    policy = FunctionPolicy(_spec(set_labels=None), fn)
    engine = _build_engine(conversation_store, [policy])
    await engine.evaluate(EvaluationContext(phase=Phase.INPUT, content="x"))
    assert engine.labels == {"any": "value", "other": "thing"}


# ── Composition: FunctionPolicy + LabelPolicy together ─


@pytest.mark.asyncio
async def test_function_and_label_policies_compose(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Mix a LabelPolicy (taint) and a FunctionPolicy
    (shell guard) across two evaluate() calls. Verifies:
    - LabelPolicy.set_labels persists (turn 1)
    - FunctionPolicy reads labels via context (turn 2)
    - DENY from FunctionPolicy names the deciding policy

    This is the same IFC pattern the secure_research_agent
    example uses — the Phase 4 e2e proxy."""
    taint = LabelPolicy(
        LabelPolicySpec(
            name="taint_web",
            on=[PhaseSelector(phase=Phase.TOOL_CALL, tool_name="web_search")],
            action=PolicyAction.ALLOW,
            set_labels={"integrity": "0"},
        )
    )

    def shell_guard(
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> PolicyResult:
        if context["labels"].get("integrity") == "0":
            return PolicyResult(
                action=PolicyAction.DENY,
                reason="tainted state; shell disallowed",
            )
        return PolicyResult(action=PolicyAction.ALLOW)

    shell = FunctionPolicy(
        _spec(
            name="shell_guard",
            phase=Phase.TOOL_CALL,
            tool_name="run_shell",
        ),
        shell_guard,
    )
    engine = _build_engine(conversation_store, [taint, shell])

    # Turn 1: web_search taints integrity to 0.
    r1 = await engine.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_CALL,
            content={"tool": "web"},
            tool_name="web_search",
        ),
    )
    assert r1.action == PolicyAction.ALLOW
    assert engine.labels["integrity"] == "0"

    # Turn 2: run_shell → shell_guard reads the tainted
    # label and DENIES.
    r2 = await engine.evaluate(
        EvaluationContext(
            phase=Phase.TOOL_CALL,
            content={"tool": "sh"},
            tool_name="run_shell",
        ),
    )
    assert r2.action == PolicyAction.DENY
    assert r2.deciding_policy == "shell_guard"
    assert "tainted" in r2.reason
