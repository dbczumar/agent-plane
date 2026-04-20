"""
:class:`FunctionPolicy` — policy backed by a Python callable.

The callable may be sync or async, may take one or two
arguments (``fn(ctx)`` or ``fn(ctx, context)``), and may
return either a :class:`PolicyResult` directly or a dict
that parses into one. See POLICIES.md §9.1 for the contract.

Two YAML shapes parse into a :class:`FunctionRef`:

- ``function: myorg.policies.rate_limit`` — the resolved
  dotted path IS the evaluator, called directly.
- ``function: {path: ..., arguments: {...}}`` — the resolved
  path is a factory, called once at build time with
  ``**arguments`` and returning the evaluator. Enables
  closure-state policies (rate limits, budgets).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable
from typing import Any

from agent_plane.policies.base import Policy
from agent_plane.policies.types import EvaluationContext, PolicyResult
from agent_plane.spec.types import FunctionPolicySpec, PolicyAction

# Type alias for what a resolved FunctionPolicy callable can be.
# Distinguishing form handled at call time — the adapter wraps
# each variant into a uniform async call.
_PolicyCallable = Callable[..., Any]


class FunctionPolicy(Policy):
    """
    A policy driven by a Python callable (POLICIES.md §9.1).

    The callable signature is discovered once at build time
    via :mod:`inspect` so each ``evaluate`` call is cheap.
    Sync callables are dispatched to a thread via
    :func:`asyncio.to_thread` so async policies (e.g. a
    PromptPolicy running alongside) are not blocked.

    :param spec: The :class:`FunctionPolicySpec` this policy
        was built from.
    :param callable_obj: The resolved callable. Either the
        evaluator directly (short-form spec) or the result of
        calling the factory with ``spec.function.arguments``
        (dict-form spec).
    """

    spec: FunctionPolicySpec

    def __init__(
        self,
        spec: FunctionPolicySpec,
        callable_obj: _PolicyCallable,
    ) -> None:
        """
        Wrap a resolved callable in the Policy contract.

        :param spec: The spec declaration.
        :param callable_obj: The evaluator callable (already
            unwrapped from any factory).
        """
        self.spec = spec
        self._callable = callable_obj
        self._is_async = inspect.iscoroutinefunction(callable_obj)
        self._arity = _callable_arity(callable_obj)

    async def evaluate(
        self,
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> PolicyResult:
        """
        Invoke the underlying callable and coerce the return.

        The engine is responsible for selector + condition
        gating + action whitelist validation +
        set_labels filtering; this method only:

        1. Dispatches sync vs async correctly.
        2. Passes ``ctx`` alone or ``ctx, context`` based on
           signature.
        3. Coerces dict returns into :class:`PolicyResult`.
        4. Lets any exception bubble up — the engine wraps it
           in fail-closed DENY (or substituted ALLOW under the
           classifier-only carve-out).

        :param ctx: Current evaluation context.
        :param context: Read-only engine context bundle.
        :returns: Normalized :class:`PolicyResult` with
            ``deciding_policy=None`` (engine sets it).
        :raises Exception: Propagates any callable-raised
            exception; the engine converts it.
        """
        raw = await self._call(ctx, context)
        return _coerce_to_policy_result(raw, spec_name=self.spec.name)

    async def _call(
        self,
        ctx: EvaluationContext,
        context: dict[str, Any],
    ) -> Any:
        """
        Dispatch the callable, adapting sync vs async + arity.

        :param ctx: The evaluation context.
        :param context: The engine context bundle.
        :returns: The raw callable return value (could be
            ``PolicyResult``, ``dict``, or anything the
            coercion helper rejects).
        """
        args: tuple[Any, ...]
        if self._arity >= 2:
            args = (ctx, context)
        else:
            args = (ctx,)
        if self._is_async:
            return await self._callable(*args)
        return await asyncio.to_thread(self._callable, *args)


def resolve_function_policy(spec: FunctionPolicySpec) -> FunctionPolicy:
    """
    Build a :class:`FunctionPolicy` from its spec.

    Resolves ``spec.function.path`` via :mod:`importlib`;
    when the spec supplies ``arguments``, treats the
    resolved path as a factory and calls it with those
    kwargs. The factory's return value is the evaluator.

    :param spec: Parsed :class:`FunctionPolicySpec` from the
        YAML policies block.
    :returns: A :class:`FunctionPolicy` ready to evaluate.
    :raises ImportError: If the dotted path cannot be
        imported.
    :raises AttributeError: If the target attribute is not
        present on the resolved module.
    :raises ValueError: If ``spec.function`` is absent (the
        parser should have rejected this — fail loud here
        rather than silently build a broken policy).
    """
    func_ref = spec.function
    if func_ref is None:
        raise ValueError(
            f"FunctionPolicy {spec.name!r} has no function reference; "
            f"parser should have rejected this at spec load.",
        )
    target = _resolve_dotted_path(func_ref.path)
    callable_obj = target(**func_ref.arguments) if func_ref.arguments else target
    if not callable(callable_obj):
        raise ValueError(
            f"FunctionPolicy {spec.name!r}: resolved object at "
            f"{func_ref.path!r} is not callable (got "
            f"{type(callable_obj).__name__})",
        )
    return FunctionPolicy(spec, callable_obj)


def _resolve_dotted_path(path: str) -> Any:
    """
    Resolve a ``module.sub.attr`` style path to its attribute.

    Splits on the last dot: everything before is the module
    path, the trailing component is the attribute name. A
    single-segment path is treated as a module-level import
    with no attribute — not useful in practice, so we reject.

    :param path: Dotted import path, e.g.
        ``"myorg.policies.search_rate_limit"``.
    :returns: The attribute found at ``module.attr``.
    :raises ValueError: On single-segment paths.
    :raises ImportError: If the module does not import.
    :raises AttributeError: If the attribute is missing.
    """
    if "." not in path:
        raise ValueError(
            f"function path {path!r} must be a dotted module.attribute "
            f"reference (e.g. 'myorg.policies.rate_limit')",
        )
    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _callable_arity(fn: _PolicyCallable) -> int:
    """
    Count the positional parameters a callable accepts.

    Used to decide whether to pass just ``ctx`` or
    ``ctx, context`` at dispatch time. ``*args`` / ``**kwargs``
    count as 0 here — policies that want them must declare
    explicit ``ctx`` / ``context`` parameters.

    Returns 1 on signature-introspection failure so the
    single-arg call path is attempted first (the caller
    surfaces errors from the actual call, not a brittle
    signature-parse).

    :param fn: The callable to inspect.
    :returns: Count of positional parameters (``POSITIONAL_ONLY``
        + ``POSITIONAL_OR_KEYWORD``).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return sum(1 for p in sig.parameters.values() if p.kind in positional_kinds)


def _coerce_to_policy_result(raw: Any, *, spec_name: str) -> PolicyResult:
    """
    Normalize a FunctionPolicy callable's return value.

    Accepts three shapes:

    - :class:`PolicyResult` — returned as-is.
    - ``dict`` — parsed into PolicyResult via
      :func:`_policy_result_from_dict`.
    - Anything else → :class:`TypeError` with a clear
      message. The engine catches it and fails closed (or
      substitutes ALLOW under the carve-out).

    :param raw: The raw return value.
    :param spec_name: Policy name for the error message.
    :returns: A :class:`PolicyResult`.
    :raises TypeError: On unrecognized return shape.
    """
    if isinstance(raw, PolicyResult):
        return raw
    if isinstance(raw, dict):
        return _policy_result_from_dict(raw, spec_name=spec_name)
    raise TypeError(
        f"FunctionPolicy {spec_name!r} returned unsupported type "
        f"{type(raw).__name__}; expected PolicyResult or dict",
    )


def _policy_result_from_dict(
    raw: dict[str, Any],
    *,
    spec_name: str,
) -> PolicyResult:
    """
    Parse a ``{"action": ..., "reason": ..., "set_labels": ...}``
    dict into a :class:`PolicyResult`.

    Accepts either enum-form or string-form action values so
    callables ported from omniagents (which use string action
    values) work unchanged.

    :param raw: The callable's dict return.
    :param spec_name: Policy name for error messages.
    :returns: A :class:`PolicyResult` with the corresponding
        action / reason / set_labels.
    :raises ValueError: If ``action`` is missing or not a
        valid :class:`PolicyAction` value.
    """
    action_raw = raw.get("action")
    if action_raw is None:
        raise ValueError(
            f"FunctionPolicy {spec_name!r} dict return missing 'action'",
        )
    try:
        action = (
            action_raw if isinstance(action_raw, PolicyAction) else PolicyAction(str(action_raw))
        )
    except ValueError:
        raise ValueError(
            f"FunctionPolicy {spec_name!r} returned invalid action "
            f"{action_raw!r}; must be one of 'allow', 'ask', 'deny'",
        )
    set_labels_raw = raw.get("set_labels")
    if set_labels_raw is not None and not isinstance(set_labels_raw, dict):
        raise ValueError(
            f"FunctionPolicy {spec_name!r} returned invalid set_labels "
            f"{set_labels_raw!r}; must be a mapping",
        )
    return PolicyResult(
        action=action,
        reason=raw.get("reason"),
        set_labels=dict(set_labels_raw) if set_labels_raw else None,
    )
