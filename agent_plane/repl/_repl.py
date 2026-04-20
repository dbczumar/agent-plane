"""Rich-based REPL for agent-plane — built on the UI SDK framework.

The public API is ``run_repl(client, agent_name, tool_handler)``.
"""

from __future__ import annotations

import asyncio
import enum
import os
import pathlib
from typing import Any

from agent_plane_client import (
    AgentPlaneClient,
    ApprovalRequestCtx,
    BlockStream,
    ResponseEndBlock,
    ResponseStartBlock,
    StreamHooks,
    ToolHandler,
    pipe,
    skip_intermediate_ends,
)
from agent_plane_ui_sdk import (
    RichBlockFormatter,
    TerminalHost,
)
from rich.text import Text


class TimedFormatter(RichBlockFormatter):  # type: ignore[misc]
    """Shows final elapsed time after response completes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._start_time: float | None = None

    def format_response_start(self, block: ResponseStartBlock) -> list[Any]:
        self._start_time = block.ctx.timestamp
        return super().format_response_start(block)

    def format_response_end(self, block: ResponseEndBlock) -> list[Any]:
        items = super().format_response_end(block)
        if self._start_time is not None:
            elapsed = block.ctx.timestamp - self._start_time
            items.append(Text.from_markup(f"   [{self.muted}]{elapsed:.1f}s[/{self.muted}]"))
            self._start_time = None
        return items


class _ApprovalVerdict(enum.Enum):
    """
    How the user answered a policy approval prompt.

    Three-way rather than boolean so the REPL can distinguish
    "approve just this one" from "approve and stop asking for
    the rest of this session". Mirrors the Claude Code model
    (y / A / n); same muscle memory transfers.

    - ``APPROVE_ONCE`` — allow this one request only.
    - ``APPROVE_ALWAYS`` — allow this request AND remember the
      decision for the rest of the REPL session. Future asks
      from the same policy at the same phase auto-approve
      without prompting.
    - ``REFUSE`` — refuse this request. Fail-closed default
      per POLICIES.md §13; anything not explicitly approve is
      a refusal.
    """

    APPROVE_ONCE = "approve_once"
    APPROVE_ALWAYS = "approve_always"
    REFUSE = "refuse"


# Input-token vocabulary for each verdict, case-insensitive.
# Both short and long forms accepted so muscle memory from
# other tools (aider's "y", claude-code's "yes") carries over.
# Anything outside these sets is a REFUSE — fail-closed per
# POLICIES.md §13.
_APPROVE_ONCE_TOKENS: frozenset[str] = frozenset({"y", "yes", "approve", "ok"})
_APPROVE_ALWAYS_TOKENS: frozenset[str] = frozenset(
    {"a", "always", "yes always", "approve always"},
)


def _parse_approval_input(text: str) -> _ApprovalVerdict:
    """
    Classify a line of user input as one of the three verdicts.

    Case-insensitive, whitespace-stripped. ALWAYS tokens are
    checked before ONCE tokens so the lone letter ``a`` is
    treated as "always" rather than ambiguously falling
    through to the refuse default.

    :param text: Raw user input from the main REPL prompt.
    :returns: The parsed verdict.
    """
    normalized = text.strip().lower()
    if normalized in _APPROVE_ALWAYS_TOKENS:
        return _ApprovalVerdict.APPROVE_ALWAYS
    if normalized in _APPROVE_ONCE_TOKENS:
        return _ApprovalVerdict.APPROVE_ONCE
    return _ApprovalVerdict.REFUSE


class _ApprovalState:
    """
    Per-REPL holder for pending approvals and the session
    auto-approve cache.

    Owning an object (rather than module globals) keeps
    multiple REPL sessions in the same process isolated —
    tests can spin up two :func:`run_repl` invocations and
    their state doesn't collide.

    Two pieces of state:

    1. The currently-pending approval :class:`asyncio.Future`
       (``None`` when no ASK is in flight). The hook creates
       it via :meth:`begin`; the main input loop resolves it
       via :meth:`resolve`. Using a future avoids the stdin /
       ``patch_stdout`` fight that a direct ``input()`` call
       produced.
    2. The session auto-approve cache: a set of
       ``(policy_name, phase)`` pairs the user said "always"
       to. Future ASKs matching one of these entries skip the
       prompt and auto-approve. Scoped to this REPL run —
       restart wipes the cache.
    """

    def __init__(self) -> None:
        """Start with no pending approval and an empty cache."""
        self._future: asyncio.Future[bool] | None = None
        # Current ASK's identity — captured on ``begin`` so
        # ``resolve_verdict`` can stash the pair on an
        # APPROVE_ALWAYS without the caller having to re-pass
        # ctx fields.
        self._current_policy: str | None = None
        self._current_phase: str | None = None
        # (policy_name, phase) → "approve always" cache.
        # ``phase`` comes from the server as a string
        # (``"input"``, ``"tool_call"``, ...) so storing the
        # pair as-is avoids any re-parsing overhead.
        self._always: set[tuple[str, str]] = set()

    @property
    def pending(self) -> bool:
        """:returns: ``True`` iff an approval is awaiting a verdict."""
        return self._future is not None and not self._future.done()

    def is_pre_approved(self, policy_name: str, phase: str) -> bool:
        """
        Look up an earlier "always" decision.

        Called by the approval hook BEFORE rendering anything —
        a pre-approved ASK must produce no UI noise. The cache
        key is specifically ``(policy_name, phase)``; different
        policies or different phases still prompt even if the
        user approved a related one.

        :param policy_name: Deciding policy's name from the
            :class:`ApprovalRequestCtx`.
        :param phase: Phase string from the ctx (``"input"`` /
            ``"tool_call"`` / etc.).
        :returns: ``True`` iff the user previously answered
            "always" for this policy+phase pair.
        """
        return (policy_name, phase) in self._always

    def remember_always(self, policy_name: str, phase: str) -> None:
        """
        Cache an "approve always" decision for the rest of the
        session.

        Idempotent — adding a duplicate entry is a no-op. The
        cache is NEVER persisted to disk; closing ``ap chat``
        clears it, so the next session starts from a clean
        slate. That matches what users expect from
        session-scoped approvals in other tools.

        :param policy_name: Deciding policy's name.
        :param phase: Phase string.
        """
        self._always.add((policy_name, phase))

    def begin(self, policy_name: str, phase: str) -> asyncio.Future[bool]:
        """
        Start a new approval — create the future the hook awaits.

        Records the identity of the ASK so
        :meth:`resolve_verdict` can cache an "always" decision
        against the right ``(policy_name, phase)`` pair
        without the caller having to re-pass them.

        If a previous approval's future is still open (the user
        never answered before a new ASK arrived), refuse the
        old one fail-closed and replace it. In practice the
        server only has one parked workflow per REPL at a
        time, so this is defense-in-depth.

        :param policy_name: Deciding policy's name from the
            :class:`ApprovalRequestCtx`.
        :param phase: Phase string from the ctx.
        :returns: The future to await. Resolves to ``True`` on
            approve (one or always) and ``False`` on refuse.
        """
        if self._future is not None and not self._future.done():
            self._future.set_result(False)
        self._current_policy = policy_name
        self._current_phase = phase
        self._future = asyncio.get_running_loop().create_future()
        return self._future

    def resolve_verdict(self, verdict: _ApprovalVerdict) -> bool:
        """
        Resolve a pending approval with a three-way verdict.

        On :attr:`_ApprovalVerdict.APPROVE_ALWAYS`, caches
        ``(current_policy, current_phase)`` so subsequent
        ASKs for that pair auto-approve without prompting.
        On any other verdict, the cache is untouched.

        :param verdict: The user's answer.
        :returns: ``True`` iff a pending approval existed and
            was resolved. ``False`` when there was nothing to
            resolve (the caller should route input normally).
        """
        if self._future is None or self._future.done():
            return False
        approved = verdict != _ApprovalVerdict.REFUSE
        if (
            verdict == _ApprovalVerdict.APPROVE_ALWAYS
            and self._current_policy is not None
            and self._current_phase is not None
        ):
            self.remember_always(self._current_policy, self._current_phase)
        self._future.set_result(approved)
        self._future = None
        self._current_policy = None
        self._current_phase = None
        return True

    def cancel(self) -> None:
        """
        Cancel any pending approval — refuse fail-closed.

        Called on REPL teardown or when the user ``/cancel``s
        an in-progress response to avoid leaking an unresolved
        future. Does NOT clear the "always" cache — that
        persists for the REPL session.
        """
        if self._future is not None and not self._future.done():
            self._future.set_result(False)
        self._future = None
        self._current_policy = None
        self._current_phase = None


def _make_approval_prompt(
    host: TerminalHost,
    fmt: RichBlockFormatter,
    state: _ApprovalState,
) -> Any:
    """
    Build the ``on_approval_request`` hook for the REPL.

    When the server emits a policy ASK (synthetic
    ``request_approval`` function_call), the SDK routes it to
    this hook. Two paths:

    - Pre-approved: the user previously said "always" for this
      ``(policy_name, phase)`` pair. Skip all UI, auto-approve.
      Print a short muted line so the transcript records that
      an auto-approve fired — silent auto-approval would be
      security-hostile (user forgets they once said "always").
    - Fresh ASK: render the preview, offer three options
      (``y`` / ``a`` / ``n``), await a future resolved by the
      main input loop.

    This hook does NOT touch stdin or call :func:`input` —
    under the REPL's active ``prompt_toolkit`` session, any
    direct stdin read fights ``patch_stdout`` and produces
    the "characters disappear / auto-delete" jank. Reusing
    the main input loop means typing the verdict works
    exactly like typing any other message. See POLICIES.md
    §7 + §15.10.

    :param host: The active :class:`TerminalHost` whose
        output channel we render the request on.
    :param fmt: Formatter whose accent / muted styles we
        reuse for visual consistency with the rest of the
        REPL.
    :param state: Shared :class:`_ApprovalState` that couples
        this hook to the main input loop and holds the
        session auto-approve cache.
    :returns: Async callable suitable for
        :attr:`StreamHooks.on_approval_request`.
    """

    async def _on_approval_request(ctx: ApprovalRequestCtx) -> bool:
        """
        Render the approval request and await the main loop's verdict.

        :param ctx: Parsed approval request carrying the
            reason, deciding policy, phase, and a truncated
            preview of the gated content.
        :returns: ``True`` on user approval (one or always);
            ``False`` otherwise.
        """
        if state.is_pre_approved(ctx.policy_name, ctx.phase):
            # Audit line — don't be silent when auto-approving,
            # the user might have forgotten they flipped it on.
            host.output(
                Text.from_markup(
                    f"   [{fmt.muted}]auto-approved · "
                    f"{ctx.policy_name} · {ctx.phase}[/{fmt.muted}]",
                ),
            )
            return True

        host.output(
            Text.from_markup(
                f"\n [{fmt.warning}]⚠ approval required · {ctx.phase}[/{fmt.warning}]",
            ),
        )
        host.output(
            Text.from_markup(
                f"   [{fmt.muted}]policy: {ctx.policy_name}[/{fmt.muted}]",
            ),
        )
        if ctx.reason:
            host.output(
                Text.from_markup(
                    f"   [{fmt.muted}]reason: {ctx.reason}[/{fmt.muted}]",
                ),
            )
        if ctx.content_preview:
            preview = ctx.content_preview
            if len(preview) > 200:
                preview = preview[:200] + "…"
            host.output(
                Text.from_markup(
                    f"   [{fmt.muted}]preview:[/{fmt.muted}] {preview}",
                ),
            )
        host.output(
            Text.from_markup(
                f"   [{fmt.accent}]y = approve once, "
                f"a = approve always (this session), "
                f"n = refuse[/{fmt.accent}]",
            ),
        )
        future = state.begin(ctx.policy_name, ctx.phase)
        return await future

    return _on_approval_request


async def run_repl(
    client: AgentPlaneClient,
    agent_name: str,
    tool_handler: ToolHandler | None,
    *,
    initial_message: str | None = None,
) -> None:
    """The entire REPL — using the framework.

    :param client: Connected AgentPlaneClient.
    :param agent_name: Agent name (used for API calls).
    :param tool_handler: Optional client-side tool handler.
    :param initial_message: If set, auto-send this message on startup
        (e.g. a greeting prompt for onboarding).
    """
    ui_name = agent_name.replace("-", " ").replace("_", " ")
    fmt = TimedFormatter(show_agent_labels=True)
    host = TerminalHost(model_name=ui_name)
    # Wire the policy-ASK seam into the session so any policy
    # in the agent's spec that returns ASK surfaces an inline
    # y/n prompt here. The hook lives on the session so every
    # turn in this REPL benefits — no per-call re-registration.
    # Shared state couples the hook (which awaits a future) to
    # the main input loop (which resolves it); reusing the
    # normal prompt_toolkit input path avoids the stdin /
    # patch_stdout fight that a direct input() call produced.
    approval_state = _ApprovalState()
    hooks = StreamHooks(on_approval_request=_make_approval_prompt(host, fmt, approval_state))
    session = client.session(model=agent_name, tool_handler=tool_handler, hooks=hooks)
    block_stream = BlockStream()
    is_streaming = False

    def show_help() -> None:
        from rich.text import Text

        lines = []
        for name, (desc, _) in COMMANDS.items():
            if name in ("/?", "/exit"):
                continue  # Skip aliases.
            lines.append(
                f"  [{fmt.accent}]{name}[/{fmt.accent}]  [{fmt.muted}]{desc}[/{fmt.muted}]"
            )
        host.output(Text.from_markup("\n".join(lines)))

    host.on_help = show_help

    async def on_input(text: str, attachments: list[Any] | None = None) -> None:
        nonlocal is_streaming

        # Pending policy approval: consume this input as the
        # verdict BEFORE slash-command / normal-send routing.
        # The hook is awaiting a future; resolving it wakes
        # the SSE stream. Echo the user's choice in dim so the
        # transcript makes sense on scrollback — otherwise a
        # bare "y" would look like an unrelated message.
        if approval_state.pending:
            verdict = _parse_approval_input(text)
            verdict_label = {
                _ApprovalVerdict.APPROVE_ONCE: "approved",
                _ApprovalVerdict.APPROVE_ALWAYS: "approved always (this session)",
                _ApprovalVerdict.REFUSE: "refused",
            }[verdict]
            host.output(
                Text.from_markup(
                    f"   [{fmt.muted}]› {verdict_label}[/{fmt.muted}]",
                ),
            )
            approval_state.resolve_verdict(verdict)
            return

        # Slash commands are short tokens like "/help", "/clear".
        # File paths like "/Users/foo/bar.jpg" start with "/" but
        # contain more path separators — don't treat those as commands.
        first_token = text.split()[0] if text.split() else ""
        if first_token.startswith("/") and "/" not in first_token[1:]:
            await handle_slash_command(text, session, client, host, fmt)
            return

        files = [a.path for a in attachments] if attachments else None
        filenames = [pathlib.Path(a.path).name for a in attachments] if attachments else None

        if is_streaming:
            # Show the message immediately in dimmed style so the
            # user knows it sent, then steer the agent.
            from rich.text import Text as RText

            host.output(RText.from_markup(f" [{fmt.muted}]❯ {text}[/{fmt.muted}]"))
            async for _ in session.send(text, files=files):
                pass  # Steer yields nothing if delivered.
            return

        host.output(fmt.user_message(text, attachments=filenames))
        host.start_timer()
        await asyncio.sleep(0)
        is_streaming = True
        try:
            stream = pipe(
                block_stream.stream(session, text, files=files),
                skip_intermediate_ends(),
            )
            from agent_plane_client import TextDone

            async for block in stream:
                if isinstance(block, TextDone) and block.has_code_blocks:
                    host.clear_streamed_text()
                for item in fmt.format(block):
                    host.output(item)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            # Escape key cancels this task. Tell the server to cancel
            # the in-progress response so the session state stays in
            # sync. Without this, _is_terminal stays False and the
            # next send() tries to steer a dead response.
            # shield() prevents the cancel() coroutine from being
            # re-cancelled by the propagating CancelledError.
            # Also refuse any pending approval fail-closed so the
            # hook's future doesn't leak waiting for a verdict
            # that will never come.
            approval_state.cancel()
            try:
                await asyncio.shield(session.cancel())
            except Exception:
                pass  # Best-effort — server may already have finished.
            from rich.text import Text as RText

            host.output(RText.from_markup(f"\n  [{fmt.muted}]cancelled[/{fmt.muted}]"))
            raise
        finally:
            is_streaming = False
            host.stop_timer()

    async with host:
        host.output(fmt.welcome(ui_name))

        from agent_plane_ui_sdk import StreamingText

        host.output(StreamingText(text="\n\n\n"))
        if initial_message:
            # Auto-send the initial message (e.g. onboarding greeting).
            asyncio.create_task(on_input(initial_message))
        await host.run(on_input)
    host.output(fmt.goodbye())


def _clear_screen() -> None:
    """Clear visible content by scrolling it off screen."""

    try:
        height = os.get_terminal_size().lines
    except (ValueError, OSError):
        height = 24
    print("\n" * height, end="", flush=True)


# ── Slash commands ───────────────────────────────────────

# Single registry: name → (help string, handler).
# Handlers take (arg, session, client, host, fmt).

COMMANDS: dict[str, tuple[str, Any]] = {}


def _cmd(name: str, help_text: str) -> Any:
    """Decorator to register a slash command."""

    def _register(fn: Any) -> Any:
        COMMANDS[name] = (help_text, fn)
        return fn

    return _register


@_cmd("/help", "Show this help")
async def _cmd_help(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    lines = []
    for name, (desc, _) in COMMANDS.items():
        if name in ("/?", "/exit"):
            continue  # Skip aliases.
        lines.append(f"  [{fmt.accent}]{name}[/{fmt.accent}]  [{fmt.muted}]{desc}[/{fmt.muted}]")
    host.output(Text.from_markup("\n".join(lines)))


COMMANDS["/?"] = COMMANDS["/help"]


@_cmd("/new", "Start a new conversation")
async def _cmd_new(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    session.reset()
    _clear_screen()
    host.output(fmt.welcome(session._model))
    host.output(Text.from_markup(f"\n  [{fmt.muted}]New conversation.[/{fmt.muted}]"))


@_cmd("/switch", "List or switch conversations")
async def _cmd_switch(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from datetime import datetime

    from rich.table import Table
    from rich.text import Text

    if not arg:
        convos = await client.conversations.list(limit=20)
        if convos:
            table = Table(title="Switch to…")
            table.add_column("#", style="bold " + fmt.accent)
            table.add_column("ID", style="dim")
            table.add_column("Title")
            table.add_column("Created", style="dim")
            for i, c in enumerate(convos, 1):
                when = datetime.fromtimestamp(c.created_at).strftime("%b %d %H:%M")
                table.add_row(str(i), c.id, c.title or "(untitled)", when)
            host.output(table)
            host.output(Text.from_markup(f"  [{fmt.muted}]/switch <id> to resume[/{fmt.muted}]"))
        else:
            host.output(Text.from_markup(f"  [{fmt.muted}]No conversations.[/{fmt.muted}]"))
    else:
        try:
            items = await client.conversations.list_items(arg, limit=100)
            last_response_id = None
            for item in reversed(items):
                rid = item.get("response_id")
                if isinstance(rid, str):
                    last_response_id = rid
                    break
            if last_response_id:
                session.reset()
                session.resume_from_response(last_response_id)
                # Clear screen and show recent history in consistent style.
                # ~5 welcome + 2 label + 2*recent messages.
                _clear_screen()
                host.output(fmt.welcome(session._model))
                host.output(
                    Text.from_markup(
                        f"  [{fmt.muted}]Resumed conversation {arg[:16]}…[/{fmt.muted}]\n"
                    )
                )
                recent = items[-6:] if len(items) > 6 else items
                for item in recent:
                    _render_history_item(item, host, fmt)
            else:
                host.output(Text.from_markup(f"  [{fmt.muted}]Empty conversation.[/{fmt.muted}]"))
        except Exception as exc:
            host.output(Text.from_markup(f"  [bold red]Error: {exc}[/]"))


@_cmd("/history", "Show current conversation history")
async def _cmd_history(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    if not session.current_response_id:
        host.output(Text.from_markup(f"  [{fmt.muted}]No active conversation.[/{fmt.muted}]"))
        return
    try:
        resp = await client.responses.get(session.current_response_id)
        if resp.conversation:
            items = await client.conversations.list_items(resp.conversation.id, limit=50)
            for item in items:
                _render_history_item(item, host, fmt)
        else:
            host.output(Text.from_markup(f"  [{fmt.muted}]No conversation.[/{fmt.muted}]"))
    except Exception as exc:
        host.output(Text.from_markup(f"  [bold red]Error: {exc}[/]"))


@_cmd("/agents", "List available agents")
async def _cmd_agents(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.table import Table
    from rich.text import Text

    agents = await client.agents.list()
    if agents:
        table = Table(title="Agents")
        table.add_column("Name", style="bold")
        table.add_column("ID", style="dim")
        for a in agents:
            table.add_row(a.name, a.id)
        host.output(table)
    else:
        host.output(Text.from_markup(f"  [{fmt.muted}]No agents.[/{fmt.muted}]"))


@_cmd("/cancel", "Cancel the current response")
async def _cmd_cancel(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    from rich.text import Text

    resp = await session.cancel()
    if resp:
        host.output(Text.from_markup(f"  [{fmt.warning}]Cancelled {resp.id}[/{fmt.warning}]"))


@_cmd("/quit", "Exit")
async def _cmd_quit(arg: str, session: Any, client: Any, host: Any, fmt: Any) -> None:
    raise EOFError


COMMANDS["/exit"] = COMMANDS["/quit"]


def _render_history_item(
    item: dict[str, Any],
    host: Any,
    fmt: Any = None,
) -> None:
    """Render a single conversation history item in consistent style."""
    from rich.text import Text

    if fmt is None:
        fmt = RichBlockFormatter()
    itype = item.get("type", "")
    if itype == "message":
        role = item.get("role", "")
        content = item.get("content", [])
        text_parts = []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("input_text", "output_text"):
                    text_parts.append(str(b.get("text", "")))
        text = " ".join(text_parts)
        if role == "user":
            host.output(fmt.user_message(text))
        elif role == "assistant":
            model = item.get("model", "")
            host.output(Text.from_markup(f" [{fmt.assistant}]◆ {model}[/{fmt.assistant}]"))
            # Show the text with proper indentation.
            preview = text[:300]
            if len(text) > 300:
                preview += "…"
            for line in preview.split("\n"):
                if line.strip():
                    host.output(Text.from_markup(f"   [{fmt.muted}]{line}[/{fmt.muted}]"))
    elif itype == "function_call":
        name = item.get("name", "?")
        host.output(Text.from_markup(f"   [{fmt.accent}]⏵ {name}[/{fmt.accent}]"))


async def handle_slash_command(
    line: str,
    session: Any,
    client: AgentPlaneClient,
    host: Any,
    fmt: Any,
) -> None:
    """Dispatch a slash command from the registry."""
    from rich.text import Text

    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    entry = COMMANDS.get(cmd)
    if entry:
        _, handler = entry
        await handler(arg, session, client, host, fmt)
    else:
        host.output(
            Text.from_markup(
                f"  [{fmt.muted}]Unknown command: {cmd} · /help for list[/{fmt.muted}]"
            )
        )
