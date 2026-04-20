"""
Tests for the Phase 10 client-side approval wiring.

Covers:

- :func:`agent_plane_client._sse._parse_output_item` — a
  ``function_call`` whose ``name`` is
  :data:`RESERVED_APPROVAL_TOOL_NAME` parses to
  :class:`ApprovalRequest` (not :class:`ToolCall`).
- :func:`agent_plane_client._responses._handle_approval_request`
  — calls the registered hook, PATCHes the verdict as
  ``{"approved": bool}``, and fail-closes when no hook is
  registered.
- REPL ``_make_approval_prompt`` — renders the approval
  request and returns ``True`` / ``False`` based on the
  user's y/n answer.

The approval path through :func:`_handle_polling_tool_calls`
(the non-streaming fallback) is also exercised to confirm it
never routes ``request_approval`` into a user-supplied
:class:`ToolHandler`.

Real HTTP is stubbed via a minimal ``_FakeHttpClient`` — these
tests exercise the SDK's branching logic, not the server.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# The editable-install of ``agent_plane_client`` points at the
# sibling worktree. Load this worktree's copy under a distinct
# module name so we're actually testing the code we just
# edited. Same pattern the e2e suite uses via PYTHONPATH.
_SDK_ROOT = (
    Path(__file__).resolve().parents[2].parent / "sdks" / "python-client" / "agent_plane_client"
)


def _load_sdk_module(name: str) -> Any:
    """
    Load ``agent_plane_client.<name>`` from this worktree
    regardless of which ``agent_plane_client`` is resolved
    globally. Registering under ``_apc_under_test.<name>``
    so parent-package resolution works for helpers that
    reference sibling submodules.
    """
    parent_name = "_apc_under_test"
    if parent_name not in sys.modules:
        parent_spec = importlib.util.spec_from_file_location(
            parent_name,
            _SDK_ROOT / "__init__.py",
            submodule_search_locations=[str(_SDK_ROOT)],
        )
        assert parent_spec is not None and parent_spec.loader is not None
        parent_mod = importlib.util.module_from_spec(parent_spec)
        sys.modules[parent_name] = parent_mod
        parent_spec.loader.exec_module(parent_mod)
    full = f"{parent_name}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _SDK_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_events = _load_sdk_module("_events")
_sse = _load_sdk_module("_sse")
_tool_handler = _load_sdk_module("_tool_handler")
_responses = _load_sdk_module("_responses")

ApprovalRequest = _events.ApprovalRequest
ToolCall = _events.ToolCall
RESERVED_APPROVAL_TOOL_NAME = _events.RESERVED_APPROVAL_TOOL_NAME
ApprovalRequestCtx = _tool_handler.ApprovalRequestCtx
StreamHooks = _tool_handler.StreamHooks


class _FakeResponse:
    """Minimal httpx.Response stand-in for PATCH result checking."""

    def __init__(self, status_code: int = 200) -> None:
        """Initialize with the simulated status code."""
        self.status_code = status_code
        self.text = ""


class _FakeHttpClient:
    """
    Minimal async HTTP stub — records PATCHes without opening
    a socket.

    Only ``patch`` is exercised by the approval flow; other
    methods raise so an unexpected call fails loud rather
    than silently hitting the real server.
    """

    def __init__(self) -> None:
        """Initialize with no recorded calls."""
        self.patch_calls: list[dict[str, Any]] = []

    async def patch(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        """Record the PATCH and return a fake 200."""
        self.patch_calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(status_code=200)


# ── SSE parse: reserved-name carve-out ────────────────────


def test_sse_parses_request_approval_as_approval_request() -> None:
    """
    A ``function_call`` output item named ``request_approval``
    parses to :class:`ApprovalRequest` — never to
    :class:`ToolCall`. This prevents the streaming tool-handler
    path from treating the synthetic approval call as a real
    tool invocation.
    """
    raw = {
        "item": {
            "type": "function_call",
            "call_id": "call_abc",
            "name": RESERVED_APPROVAL_TOOL_NAME,
            "status": "action_required",
            "arguments": json.dumps(
                {
                    "phase": "tool_call",
                    "reason": "approve web search?",
                    "policy_name": "ask_search",
                    "content_preview": "q=classified",
                },
            ),
        },
    }
    event = _sse._parse_output_item(raw)
    assert isinstance(event, ApprovalRequest)
    assert event.call_id == "call_abc"
    assert event.phase == "tool_call"
    assert event.reason == "approve web search?"
    assert event.policy_name == "ask_search"
    assert event.content_preview == "q=classified"


def test_sse_parses_regular_function_call_as_toolcall() -> None:
    """
    A ``function_call`` with any non-reserved name still parses
    to :class:`ToolCall`. Regression guard: the carve-out must
    not leak into the regular tool-tunneling path.
    """
    raw = {
        "item": {
            "type": "function_call",
            "call_id": "call_real",
            "name": "Read",
            "status": "action_required",
            "arguments": json.dumps({"path": "x.py"}),
        },
    }
    event = _sse._parse_output_item(raw)
    assert isinstance(event, ToolCall)
    assert event.name == "Read"


def test_sse_approval_request_tolerates_missing_fields() -> None:
    """
    The server always emits all four approval fields today
    (POLICIES.md §7), but the parser defensively coerces
    missing / non-string values to empty string rather than
    raising. Defensive shape keeps a stray protocol skew from
    crashing the whole stream.
    """
    raw = {
        "item": {
            "type": "function_call",
            "call_id": "call_bare",
            "name": RESERVED_APPROVAL_TOOL_NAME,
            "arguments": "{}",
        },
    }
    event = _sse._parse_output_item(raw)
    assert isinstance(event, ApprovalRequest)
    assert event.reason == ""
    assert event.policy_name == ""
    assert event.phase == ""
    assert event.content_preview == ""


# ── _handle_approval_request: hook + PATCH wiring ─────────


@pytest.mark.asyncio
async def test_approval_hook_approved_patches_true_verdict() -> None:
    """
    A hook that returns ``True`` → SDK PATCHes
    ``{"approved": true}`` on the existing ``tool_results``
    contract. Verdict is serialized as JSON per POLICIES.md
    §13 (strict ``is True`` check on the server).
    """
    http = _FakeHttpClient()
    seen: list[ApprovalRequestCtx] = []

    async def _hook(ctx: ApprovalRequestCtx) -> bool:
        seen.append(ctx)
        return True

    hooks = StreamHooks(on_approval_request=_hook)
    event = ApprovalRequest(
        call_id="call_approved",
        reason="tainted",
        policy_name="deny_tainted",
        phase="tool_call",
        content_preview="args",
    )
    await _responses._handle_approval_request(
        http,  # type: ignore[arg-type]
        "http://localhost:8000",
        hooks,
        event,
        response_id="resp_1",
    )

    # The hook fired exactly once with the right context.
    assert len(seen) == 1
    assert seen[0].call_id == "call_approved"
    assert seen[0].policy_name == "deny_tainted"

    # The PATCH carried an "approved": true verdict.
    assert len(http.patch_calls) == 1
    call = http.patch_calls[0]
    assert call["url"] == "http://localhost:8000/v1/responses/resp_1"
    payload = call["json"]
    assert payload is not None
    tool_results = payload["tool_results"]
    assert len(tool_results) == 1
    assert tool_results[0]["call_id"] == "call_approved"
    # Verdict is a JSON string whose shape the server parses
    # with _parse_verdict — exact ``{"approved": true}``.
    verdict = json.loads(tool_results[0]["output"])
    assert verdict == {"approved": True}


@pytest.mark.asyncio
async def test_approval_hook_refused_patches_false_verdict() -> None:
    """
    Hook returns ``False`` → SDK PATCHes
    ``{"approved": false}``. Server treats this identically
    to timeout / cancel — the parked workflow wakes and the
    enforcement site short-circuits with a DENY sentinel.
    """
    http = _FakeHttpClient()

    async def _hook(ctx: ApprovalRequestCtx) -> bool:
        return False

    hooks = StreamHooks(on_approval_request=_hook)
    event = ApprovalRequest(
        call_id="call_refused",
        reason="",
        policy_name="p",
        phase="output",
        content_preview="",
    )
    await _responses._handle_approval_request(
        http,  # type: ignore[arg-type]
        "http://localhost:8000",
        hooks,
        event,
        response_id="resp_2",
    )
    verdict = json.loads(http.patch_calls[0]["json"]["tool_results"][0]["output"])
    assert verdict == {"approved": False}


@pytest.mark.asyncio
async def test_approval_without_hook_fails_closed() -> None:
    """
    No hook registered → SDK PATCHes ``{"approved": false}``.
    POLICIES.md §7.2: an unhandled ASK must DENY. Silently
    swallowing the ASK would stall the parked workflow until
    ``ask_timeout`` expired — refusing fail-closed is the
    right default.
    """
    http = _FakeHttpClient()
    hooks = StreamHooks()  # no on_approval_request
    event = ApprovalRequest(
        call_id="call_nohook",
        reason="",
        policy_name="p",
        phase="input",
        content_preview="",
    )
    await _responses._handle_approval_request(
        http,  # type: ignore[arg-type]
        "http://localhost:8000",
        hooks,
        event,
        response_id="resp_3",
    )
    verdict = json.loads(http.patch_calls[0]["json"]["tool_results"][0]["output"])
    assert verdict == {"approved": False}


@pytest.mark.asyncio
async def test_approval_hook_exception_fails_closed() -> None:
    """
    Hook raises → SDK catches, logs, PATCHes
    ``{"approved": false}``. A buggy approval handler must not
    crash the stream or stall the workflow; fail-closed keeps
    the invariant.
    """
    http = _FakeHttpClient()

    async def _hook(ctx: ApprovalRequestCtx) -> bool:
        raise RuntimeError("bug in handler")

    hooks = StreamHooks(on_approval_request=_hook)
    event = ApprovalRequest(
        call_id="call_bug",
        reason="",
        policy_name="p",
        phase="tool_call",
        content_preview="",
    )
    await _responses._handle_approval_request(
        http,  # type: ignore[arg-type]
        "http://localhost:8000",
        hooks,
        event,
        response_id="resp_4",
    )
    verdict = json.loads(http.patch_calls[0]["json"]["tool_results"][0]["output"])
    assert verdict == {"approved": False}


@pytest.mark.asyncio
async def test_approval_hook_accepts_sync_callable() -> None:
    """
    Hooks can be sync or async. A sync ``def hook(ctx) -> bool``
    must work too — the client awaits only when the return is
    awaitable.
    """
    http = _FakeHttpClient()

    def _hook(ctx: ApprovalRequestCtx) -> bool:
        return True

    hooks = StreamHooks(on_approval_request=_hook)
    event = ApprovalRequest(
        call_id="call_sync",
        reason="",
        policy_name="p",
        phase="output",
        content_preview="",
    )
    await _responses._handle_approval_request(
        http,  # type: ignore[arg-type]
        "http://localhost:8000",
        hooks,
        event,
        response_id="resp_5",
    )
    verdict = json.loads(http.patch_calls[0]["json"]["tool_results"][0]["output"])
    assert verdict == {"approved": True}


# ── Polling path: reserved-name carve-out ─────────────────


@pytest.mark.asyncio
async def test_polling_path_refuses_request_approval_without_handler() -> None:
    """
    The polling path fallback (non-streaming clients) has no
    approval hook surface. The reserved name must not be
    routed into a user ``ToolHandler.execute`` — it's not a
    real tool. Instead, the polling path PATCHes
    ``{"approved": false}`` directly, preserving the
    fail-closed default.
    """
    http = _FakeHttpClient()
    # Build a ResponsesNamespace against the fake http.
    ns = _responses.ResponsesNamespace(http, "http://localhost:8000")  # type: ignore[arg-type]

    class _Response:
        """Minimal Response stand-in with an output carrying a request_approval call."""

        output = [
            {
                "type": "function_call",
                "name": RESERVED_APPROVAL_TOOL_NAME,
                "call_id": "call_poll",
                "status": "action_required",
                "arguments": "{}",
            },
        ]

    class _ExplodingHandler:
        """ToolHandler that must never be invoked for the reserved name."""

        schemas: list[dict[str, Any]] = []

        def execute(self, info: Any) -> str:
            raise AssertionError(
                "request_approval must not be routed into a ToolHandler",
            )

    handler = _ExplodingHandler()
    await ns._handle_polling_tool_calls(
        "resp_poll",
        _Response(),  # type: ignore[arg-type]
        handler,  # type: ignore[arg-type]
    )

    assert len(http.patch_calls) == 1
    verdict = json.loads(http.patch_calls[0]["json"]["tool_results"][0]["output"])
    assert verdict == {"approved": False}


# ── REPL approval prompt wiring ───────────────────────────
#
# The REPL's approval flow doesn't call ``input()`` — that
# path fought prompt_toolkit's ``patch_stdout`` (characters
# vanishing mid-type, auto-delete jank). Instead, the hook
# creates an :class:`asyncio.Future` that the main input
# loop resolves when the user types ``y`` / ``n`` at the
# pinned prompt. These tests drive the future directly;
# the main-loop wiring in :func:`run_repl` is exercised
# via the e2e harness.


def _load_repl_module() -> Any:
    """
    Reload ``agent_plane.repl._repl`` so these tests see the
    edited source. Multiple tests in this file touch the
    module; a stale import cache would silently test the old
    API.
    """
    import importlib

    import agent_plane.repl._repl as repl_mod

    importlib.reload(repl_mod)
    return repl_mod


class _FakeHost:
    """TerminalHost stub — records everything the hook prints."""

    def __init__(self) -> None:
        """Initialize with an empty output log."""
        self.outputs: list[Any] = []

    def output(self, item: Any) -> None:
        """Record the item."""
        self.outputs.append(item)


class _FakeFmt:
    """Formatter stub — the hook reads style names off it."""

    warning = "yellow"
    muted = "dim"
    accent = "cyan"


@pytest.mark.asyncio
async def test_repl_approval_hook_renders_and_awaits_future() -> None:
    """
    The hook writes the approval preview to the host, creates
    a pending future on the shared :class:`_ApprovalState`,
    and awaits it. It must NOT touch stdin — previously
    calling :func:`input` inside a thread fought
    ``patch_stdout``.

    We drive the future manually to assert the shape without
    spinning up a full REPL.
    """
    repl_mod = _load_repl_module()
    host = _FakeHost()
    state = repl_mod._ApprovalState()
    prompt_fn = repl_mod._make_approval_prompt(host, _FakeFmt(), state)
    ctx = ApprovalRequestCtx(
        call_id="c1",
        reason="needs review",
        policy_name="gatekeeper",
        phase="tool_call",
        content_preview='{"tool": "search"}',
        response_id="r1",
    )
    # Kick off the hook; don't await yet.
    task = asyncio.create_task(prompt_fn(ctx))
    # Give the event loop a turn so the hook renders and
    # registers the pending future.
    await asyncio.sleep(0)
    assert state.pending is True
    assert host.outputs, "approval hook rendered nothing"

    # Resolve via the same path the main input loop takes.
    resolved = state.resolve_verdict(repl_mod._ApprovalVerdict.APPROVE_ONCE)
    assert resolved is True
    result = await task
    assert result is True


@pytest.mark.asyncio
async def test_repl_approval_state_resolve_on_refuse() -> None:
    """
    Resolving the future with REFUSE must yield ``False``
    from the hook — the fail-closed path for POLICIES.md §13.
    """
    repl_mod = _load_repl_module()
    state = repl_mod._ApprovalState()
    prompt_fn = repl_mod._make_approval_prompt(_FakeHost(), _FakeFmt(), state)
    ctx = ApprovalRequestCtx(
        call_id="c2",
        reason="",
        policy_name="p",
        phase="output",
        content_preview="",
        response_id="r2",
    )
    task = asyncio.create_task(prompt_fn(ctx))
    await asyncio.sleep(0)
    state.resolve_verdict(repl_mod._ApprovalVerdict.REFUSE)
    assert await task is False


@pytest.mark.asyncio
async def test_repl_approval_state_cancel_refuses_closed() -> None:
    """
    Cancelling an in-flight approval (user ^C during stream)
    must resolve the future to ``False``. Leaking an
    unresolved future would stall the next ASK forever.
    """
    repl_mod = _load_repl_module()
    state = repl_mod._ApprovalState()
    prompt_fn = repl_mod._make_approval_prompt(_FakeHost(), _FakeFmt(), state)
    ctx = ApprovalRequestCtx(
        call_id="c3",
        reason="",
        policy_name="",
        phase="input",
        content_preview="",
        response_id="r3",
    )
    task = asyncio.create_task(prompt_fn(ctx))
    await asyncio.sleep(0)
    state.cancel()
    assert await task is False
    # And cancel() clears state so no future is left pending.
    assert state.pending is False


def test_repl_approval_state_replaces_stale_future() -> None:
    """
    If a second ASK arrives while a prior one is still
    pending (defense-in-depth — the server should only park
    one at a time, but bugs happen), the prior future is
    resolved fail-closed and a fresh one is installed.
    """
    repl_mod = _load_repl_module()
    # Run in a loop to exercise the future primitives.

    async def _body() -> None:
        state = repl_mod._ApprovalState()
        first = state.begin("p1", "input")
        second = state.begin("p1", "input")
        # Old future was resolved False so the first ASK's
        # hook wakes with a refusal (never leaks).
        assert first.done() and first.result() is False
        # New future is still open, waiting for the verdict.
        assert not second.done()
        state.resolve_verdict(repl_mod._ApprovalVerdict.APPROVE_ONCE)
        assert second.result() is True

    asyncio.run(_body())


# ── Three-way verdict parser ─────────────────────────────
#
# The parser is the precise seam between user keystrokes and
# the approval state. It must: (a) accept the short forms
# users reach for (``y``, ``a``, ``n``), (b) disambiguate
# ``a`` as ALWAYS (not as a random non-``y`` character that
# falls through to refuse), and (c) fail-closed on anything
# outside the vocabulary.


@pytest.mark.parametrize(
    "text,expected",
    [
        # APPROVE_ONCE
        ("y", "APPROVE_ONCE"),
        ("Y", "APPROVE_ONCE"),
        ("yes", "APPROVE_ONCE"),
        ("YES", "APPROVE_ONCE"),
        ("approve", "APPROVE_ONCE"),
        ("ok", "APPROVE_ONCE"),
        (" y ", "APPROVE_ONCE"),
        # APPROVE_ALWAYS
        ("a", "APPROVE_ALWAYS"),
        ("A", "APPROVE_ALWAYS"),
        ("always", "APPROVE_ALWAYS"),
        ("ALWAYS", "APPROVE_ALWAYS"),
        ("approve always", "APPROVE_ALWAYS"),
        (" a ", "APPROVE_ALWAYS"),
        # REFUSE
        ("", "REFUSE"),
        ("n", "REFUSE"),
        ("no", "REFUSE"),
        ("anything else", "REFUSE"),
        ("yolo", "REFUSE"),  # near-miss — explicit refusal
    ],
)
def test_repl_parse_approval_input(text: str, expected: str) -> None:
    """
    Three-way verdict parser matches Claude Code muscle memory
    (``y`` / ``a`` / ``n``) and fails closed on anything
    outside the vocabulary. The enum comparison guards against
    regressions that'd silently demote APPROVE_ALWAYS to
    APPROVE_ONCE (or vice-versa).
    """
    repl_mod = _load_repl_module()
    verdict = repl_mod._parse_approval_input(text)
    assert verdict.name == expected


# ── Session auto-approve cache ────────────────────────────


@pytest.mark.asyncio
async def test_repl_approval_always_caches_and_auto_approves() -> None:
    """
    End-to-end for the "approve always" path.

    First ASK: user types "a" (mapped to APPROVE_ALWAYS). The
    state caches ``(policy_name, phase)`` and the hook returns
    ``True`` for this one. Host received the full
    approval-required banner.

    Second ASK for the same pair: hook checks the cache FIRST,
    returns True immediately, and prints a muted
    ``auto-approved`` audit line. Critically: no
    ``⚠ approval required`` banner is rendered — the whole
    point of caching is zero UI friction once you've said yes.
    """
    repl_mod = _load_repl_module()
    host = _FakeHost()
    state = repl_mod._ApprovalState()
    prompt_fn = repl_mod._make_approval_prompt(host, _FakeFmt(), state)

    # First ASK — prompts, user says "always".
    ctx1 = ApprovalRequestCtx(
        call_id="c1",
        reason="",
        policy_name="always_ask_on_input",
        phase="input",
        content_preview="hello",
        response_id="r1",
    )
    task1 = asyncio.create_task(prompt_fn(ctx1))
    await asyncio.sleep(0)
    assert state.pending is True
    # Find the banner in the first-ASK outputs.
    first_texts = [getattr(o, "plain", str(o)) for o in host.outputs]
    assert any("approval required" in t for t in first_texts), "First ASK must render the banner"
    outputs_before_always = len(host.outputs)
    state.resolve_verdict(repl_mod._ApprovalVerdict.APPROVE_ALWAYS)
    assert await task1 is True
    # Cache now has the pair.
    assert state.is_pre_approved("always_ask_on_input", "input")

    # Second ASK — same pair. Must auto-approve without
    # rendering the banner.
    ctx2 = ApprovalRequestCtx(
        call_id="c2",
        reason="",
        policy_name="always_ask_on_input",
        phase="input",
        content_preview="follow-up",
        response_id="r2",
    )
    task2 = asyncio.create_task(prompt_fn(ctx2))
    await asyncio.sleep(0)
    # Future NEVER gets created because the hook short-circuits.
    assert state.pending is False
    assert await task2 is True

    # Outputs added by ASK#2: ONLY the muted auto-approve
    # audit line — no banner, no policy/reason/preview lines.
    # Banner would be `approval required`, which must not
    # appear for the second ASK.
    second_ask_outputs = host.outputs[outputs_before_always:]
    second_ask_texts = [getattr(o, "plain", str(o)) for o in second_ask_outputs]
    assert all("approval required" not in t for t in second_ask_texts), (
        f"Second ASK rendered a banner despite auto-approval cache:\n{second_ask_texts}"
    )
    assert any("auto-approved" in t for t in second_ask_texts), (
        f"Auto-approve path must print an audit line:\n{second_ask_texts}"
    )


@pytest.mark.asyncio
async def test_repl_approval_always_is_scoped_to_policy_and_phase() -> None:
    """
    The cache key is ``(policy_name, phase)`` — a different
    policy OR a different phase still prompts. Granularity
    prevents a blanket "always" from accidentally approving a
    different gate the user never consented to.
    """
    repl_mod = _load_repl_module()
    state = repl_mod._ApprovalState()
    # User said "always" for policy_a at INPUT.
    state.remember_always("policy_a", "input")

    assert state.is_pre_approved("policy_a", "input") is True
    # Different policy — still prompts.
    assert state.is_pre_approved("policy_b", "input") is False
    # Same policy, different phase — still prompts.
    assert state.is_pre_approved("policy_a", "tool_call") is False


def test_repl_approval_once_does_not_populate_cache() -> None:
    """
    APPROVE_ONCE must leave the cache empty. Otherwise the
    next ASK would silently auto-approve, which is NOT what
    the user asked for — "once" means once.
    """
    repl_mod = _load_repl_module()

    async def _body() -> None:
        state = repl_mod._ApprovalState()
        state.begin("policy_x", "input")
        state.resolve_verdict(repl_mod._ApprovalVerdict.APPROVE_ONCE)
        assert state.is_pre_approved("policy_x", "input") is False

    asyncio.run(_body())


def test_repl_approval_refuse_does_not_populate_cache() -> None:
    """
    REFUSE must also leave the cache untouched. Caching a
    refusal would make the next ASK silently fail without the
    user getting a chance to reconsider.
    """
    repl_mod = _load_repl_module()

    async def _body() -> None:
        state = repl_mod._ApprovalState()
        state.begin("policy_x", "input")
        state.resolve_verdict(repl_mod._ApprovalVerdict.REFUSE)
        assert state.is_pre_approved("policy_x", "input") is False

    asyncio.run(_body())
