"""Rich-based rendering for REPL stream events.

Design inspired by Claude Code, Aider, and OpenCode:
- Warm accent color (orange) for branding
- Semantic colors per role (cyan=user, green=assistant, dim=metadata)
- Rounded panels for tool calls with per-type accent colors
- Truncated output with "N more lines" footer
- Animated spinner with rotating status verbs
- Dim italic for thinking/reasoning with live preview
"""

from __future__ import annotations

import json

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# ── Color palette ────────────────────────────────────────
# Warm, Claude-inspired palette.

ACCENT = "#d87757"  # Warm orange — brand accent
USER = "bold cyan"
ASSISTANT = "bold green"
TOOL_CALL = "#5c9cf5"  # Blue
TOOL_RESULT = "dim"
REASONING = "dim italic #8a8a8a"
ERROR = "bold #ff6b80"
WARNING = "#ffa500"
SUCCESS = "#4eba65"
DIM = "dim"
MUTED = "#6a6a6a"

# ── Tool metadata registry ───────────────────────────────
# Data-driven tool config: color, whether output is code,
# which argument to display inline, and language hint.
# Adding a new tool = one dict entry, not scattered if-chains.

_TOOL_METADATA: dict[str, dict[str, object]] = {
    "Read": {
        "color": "#5c9cf5",
        "shows_code": True,
        "language": "text",
        "display_arg": "file_path",
    },
    "Write": {
        "color": "#4eba65",
        "shows_code": True,
        "language": "text",
        "display_arg": "file_path",
    },
    "Edit": {"color": "#d4a843", "shows_code": False, "display_arg": "file_path"},
    "Bash": {"color": "#ffa500", "shows_code": True, "language": "bash", "display_arg": "command"},
    "Glob": {"color": "#9d7cd8", "shows_code": False, "display_arg": "pattern"},
    "Grep": {"color": "#9d7cd8", "shows_code": True, "language": "text", "display_arg": "pattern"},
    "LSP": {"color": "#5c9cf5", "shows_code": True, "language": "text", "display_arg": "action"},
    "web_search": {"color": "#5c9cf5", "shows_code": False, "display_arg": "query"},
    "code_sandbox": {"color": "#ffa500", "shows_code": True, "language": "bash"},
    "spawn_sub_agents": {"color": "#d87757", "shows_code": False, "display_arg": "agents"},
    "check_sub_agents": {"color": "#d87757", "shows_code": False},
    "collect_sub_agents": {"color": "#d87757", "shows_code": False},
    "cancel_sub_agent": {"color": "#ff6b80", "shows_code": False},
}

# Spinner animation — rotating verbs for processing state.
SPINNER_VERBS = [
    "Thinking",
    "Processing",
    "Analyzing",
    "Reasoning",
    "Working",
    "Considering",
]

# Max lines/chars to show in tool result panels before truncation.
_MAX_RESULT_LINES = 30
_MAX_RESULT_CHARS = 4000

# Pygments lexer name → Rich Syntax language name.
_LEXER_NAME_MAP: dict[str, str] = {
    "python": "python",
    "python 3": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "bash": "bash",
    "shell": "bash",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "c++": "cpp",
    "ruby": "ruby",
    "yaml": "yaml",
    "json": "json",
    "toml": "toml",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "markdown": "markdown",
}


def _tool_color(name: str) -> str:
    """Get the accent color for a tool, with fallback for unknown tools."""
    meta = _TOOL_METADATA.get(name)
    if meta is not None:
        return str(meta.get("color", TOOL_CALL))
    return TOOL_CALL


# ── Prompt and user messages ─────────────────────────────


_MAX_USER_MSG_LINES = 4


def render_user_message(console: Console, text: str) -> None:
    """Render the user's message with gray background, truncated to 4 lines."""
    truncated = truncate_user_text(text)
    console.print()
    console.print(
        Text.from_markup(
            f" [{ACCENT}]❯[/{ACCENT}] [on #1a1a1a]{escape_markup(truncated)}[/on #1a1a1a]"
        )
    )


def render_steering_message(console: Console, text: str) -> None:
    """Render a steering message with gray background."""
    truncated = truncate_user_text(text)
    console.print(
        Text.from_markup(
            f" [{ACCENT}]❯[/{ACCENT}] [{DIM}](steering)[/{DIM}]"
            f" [on #1a1a1a]{escape_markup(truncated)}[/on #1a1a1a]"
        )
    )


def truncate_user_text(text: str) -> str:
    """Truncate user text to _MAX_USER_MSG_LINES lines."""
    lines = text.split("\n")
    if len(lines) <= _MAX_USER_MSG_LINES:
        return text
    kept = lines[:_MAX_USER_MSG_LINES]
    omitted = len(lines) - _MAX_USER_MSG_LINES
    return "\n".join(kept) + f"\n… {omitted} more lines"


# ── Tool calls and results ───────────────────────────────


def render_tool_call(console: Console, name: str, arguments: dict, agent_name: str) -> None:
    """Render a tool call as an inline status line."""
    color = _tool_color(name)
    args_str = _format_args_brief(name, arguments)
    prefix = f"[{MUTED}]{agent_name} → [/{MUTED}]" if "." in agent_name else ""
    console.print(
        Text.from_markup(f"   {prefix}[{color}]⏵ {name}[/{color}][{DIM}]({args_str})[/{DIM}]")
    )


def build_tool_result_panel(name: str, output: str) -> RenderableType:
    """Build a tool result panel renderable.

    Shared by both ``render_tool_result`` (direct printing) and
    ``StreamDisplay.add_tool_result`` (Live display accumulation).
    """
    color = _tool_color(name)
    lines = output.split("\n")
    total_lines = len(lines)

    display_output = output[:_MAX_RESULT_CHARS]
    display_lines = display_output.split("\n")

    if total_lines > _MAX_RESULT_LINES:
        visible = display_lines[:_MAX_RESULT_LINES]
        omitted = total_lines - _MAX_RESULT_LINES
        footer = Text.from_markup(f"[{MUTED}]… {omitted} more lines[/{MUTED}]")
    else:
        visible = display_lines
        footer = None

    first_line = lines[0][:80] if lines else ""

    # Try syntax highlighting for code-like output.
    renderable: RenderableType
    if _looks_like_code(output, name):
        lang = _guess_language(name, output)
        try:
            parts: list[RenderableType] = [
                Syntax(
                    "\n".join(visible),
                    lang,
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                ),
            ]
            if footer is not None:
                parts.append(footer)
            renderable = Group(*parts) if len(parts) > 1 else parts[0]
        except (ValueError, KeyError):
            body = "\n".join(visible)
            renderable = Text.from_markup(f"[{DIM}]{escape_markup(body)}[/{DIM}]")
            if footer is not None:
                renderable = Group(renderable, footer)
    else:
        body = "\n".join(visible)
        renderable = Text.from_markup(f"[{DIM}]{escape_markup(body)}[/{DIM}]")
        if footer is not None:
            renderable = Group(renderable, footer)

    return Padding(
        Panel(
            renderable,
            title=f"[{DIM}]{escape_markup(first_line)}[/{DIM}]",
            title_align="left",
            border_style=color,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        ),
        (0, 1, 0, 3),
    )


def render_tool_result(console: Console, name: str, output: str) -> None:
    """Render a tool result in a rounded panel with truncation."""
    console.print(build_tool_result_panel(name, output))


def render_native_tool(console: Console, tool_type: str, data: dict) -> None:
    """Render a provider-native tool call."""
    color = _tool_color(tool_type)
    label = _format_native_label(tool_type, data)
    console.print(Text.from_markup(f"   [{color}]⏵ {label}[/{color}]"))


# ── Reasoning / thinking ────────────────────────────────


def render_reasoning_start(console: Console) -> None:
    """Render reasoning start — just a dim indicator."""
    pass  # Handled by StreamDisplay's live rendering.


def render_reasoning_end(console: Console, reasoning: str, summary: str) -> None:
    """Render reasoning completion as a collapsible panel."""
    text = summary or reasoning
    if not text.strip():
        return

    lines = text.strip().split("\n")
    if len(lines) > 8:
        preview = "\n".join(lines[-8:])
        preview = f"[{MUTED}]… {len(lines) - 8} earlier lines[/{MUTED}]\n" + preview
    else:
        preview = "\n".join(lines)

    console.print(
        Padding(
            Panel(
                Text.from_markup(f"[{REASONING}]{escape_markup(preview)}[/{REASONING}]"),
                title=f"[{MUTED}]thinking[/{MUTED}]",
                title_align="left",
                border_style=MUTED,
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True,
            ),
            (0, 1, 0, 3),
        )
    )


# ── Message rendering ────────────────────────────────────


def render_message_text(console: Console, text: str) -> None:
    """Render final assistant message text as markdown."""
    console.print(Padding(Markdown(text, code_theme="monokai"), (0, 1, 0, 3)))
    console.print()


def render_assistant_header(console: Console) -> None:
    """Render the assistant response header."""
    console.print()
    console.print(Text.from_markup(f"  [{ASSISTANT}]◆[/{ASSISTANT}]"), end=" ")


# ── Builder functions (return renderables for rprint) ────


def _build_reasoning_panel(reasoning: str, summary: str) -> RenderableType:
    """Build a reasoning panel renderable."""
    text = summary or reasoning
    if not text.strip():
        return Text("")

    lines = text.strip().split("\n")
    if len(lines) > 8:
        preview = "\n".join(lines[-8:])
        preview = f"[{MUTED}]… {len(lines) - 8} earlier lines[/{MUTED}]\n" + preview
    else:
        preview = "\n".join(lines)

    return Padding(
        Panel(
            Text.from_markup(f"[{REASONING}]{escape_markup(preview)}[/{REASONING}]"),
            title=f"[{MUTED}]thinking[/{MUTED}]",
            title_align="left",
            border_style=MUTED,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        ),
        (0, 1, 0, 3),
    )


def _build_tool_call_line(name: str, arguments: dict, agent_name: str) -> str:
    """Build a tool call markup string."""
    color = _tool_color(name)
    args_str = _format_args_brief(name, arguments)
    prefix = f"[{MUTED}]{agent_name} → [/{MUTED}]" if "." in agent_name else ""
    return f"   {prefix}[{color}]⏵ {name}[/{color}][{DIM}]({args_str})[/{DIM}]"


def _build_native_tool_line(tool_type: str, data: dict) -> str:
    """Build a native tool call markup string."""
    color = _tool_color(tool_type)
    label = _format_native_label(tool_type, data)
    return f"   [{color}]⏵ {label}[/{color}]"


def _build_message_text(text: str) -> RenderableType:
    """Build a markdown renderable for the final message."""
    return Padding(Markdown(text, code_theme="monokai"), (0, 1, 0, 3))


# ── Status and metadata ──────────────────────────────────


def render_error(console: Console, message: str) -> None:
    """Render an error message."""
    console.print(
        Padding(
            Panel(
                Text.from_markup(f"[{ERROR}]{escape_markup(message)}[/{ERROR}]"),
                border_style="#ff6b80",
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True,
            ),
            (0, 1, 0, 3),
        )
    )


def render_retry(
    console: Console, source: str, attempt: int, max_attempts: int, delay: float
) -> None:
    """Render a retry notification."""
    console.print(
        Text.from_markup(
            f"   [{WARNING}]↻ retrying {source} "
            f"(attempt {attempt}/{max_attempts}, "
            f"waiting {delay:.1f}s)…[/{WARNING}]"
        )
    )


def render_status(console: Console, status: str, model: str) -> None:
    """Render a terminal status."""
    if status == "completed":
        return
    style_map = {
        "failed": ERROR,
        "incomplete": WARNING,
        "cancelled": WARNING,
    }
    style = style_map.get(status.split("(")[0].strip(), DIM)
    console.print(Text.from_markup(f"   [{style}]{status}[/{style}]"))


def render_file_output(console: Console, file_id: str, filename: str | None) -> None:
    """Render a file output notification."""
    name = filename or file_id
    console.print(Text.from_markup(f"   [{SUCCESS}]📎 {name}[/{SUCCESS}]"))


def render_sub_agent_spawned(console: Console, agents: list) -> None:
    """Render sub-agent spawn notification."""
    names = ", ".join(a.agent_name for a in agents)
    console.print(Text.from_markup(f"   [{ACCENT}]⑂ spawned: {names}[/{ACCENT}]"))


def render_compaction_start(console: Console) -> None:
    """Render compaction start."""
    console.print(Text.from_markup(f"   [{MUTED}]◐ compacting conversation…[/{MUTED}]"))


def render_welcome(console: Console, agent_name: str) -> None:
    """Render the welcome banner."""
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[{ACCENT}]agent-plane[/{ACCENT}]  [{DIM}]·[/{DIM}]  "
                f"[bold]{agent_name}[/bold]\n"
                f"[{DIM}]Type a message to chat · Ctrl+C to cancel · Ctrl+D to exit[/{DIM}]\n"
                f"[{DIM}]Commands: /new /cancel /agents /conversations"
                f" /history /attach /model /quit[/{DIM}]"
            ),
            box=box.ROUNDED,
            border_style=ACCENT,
            padding=(0, 1),
            expand=True,
        )
    )
    console.print()


def render_goodbye(console: Console) -> None:
    """Render exit message."""
    console.print(f"\n  [{DIM}]Goodbye.[/{DIM}]\n")


def render_server_starting(console: Console) -> None:
    """Render server startup message."""
    console.print(f"  [{DIM}]Starting server…[/{DIM}]")


def render_server_ready(console: Console, url: str) -> None:
    """Render server ready message."""
    console.print(f"  [{SUCCESS}]✓[/{SUCCESS}] [{DIM}]Server ready on {url}[/{DIM}]")


# ── Helpers ──────────────────────────────────────────────


def escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", "\\[").replace("]", "\\]")


def _format_args_brief(name: str, arguments: dict) -> str:
    """Format arguments for inline display.

    Uses the ``display_arg`` from the tool metadata registry to
    pick the most relevant argument. Falls back to compact JSON.
    """
    if not arguments:
        return ""

    # Check metadata for the preferred display argument.
    meta = _TOOL_METADATA.get(name, {})
    display_key = meta.get("display_arg")
    if display_key and display_key in arguments:
        value = arguments[display_key]
        # Special handling for list-valued display args (e.g., spawn_sub_agents.agents).
        if isinstance(value, list):
            names = [str(a.get("name", "?")) if isinstance(a, dict) else str(a) for a in value]
            return ", ".join(names)
        s = str(value)
        # For file paths, show just the filename.
        if display_key == "file_path" and "/" in s:
            s = s.rsplit("/", 1)[-1]
        return s[:80] + "…" if len(s) > 80 else s

    # Fallback: compact JSON.
    try:
        s = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(arguments)
    return s[:80] + "…" if len(s) > 80 else s


def _format_native_label(tool_type: str, data: dict) -> str:
    """Format a native tool label for display."""
    if tool_type == "web_search_call":
        action = data.get("action")
        if isinstance(action, dict):
            action_type = action.get("type", "")
            if action_type == "search":
                return f"web search: {str(action.get('query', ''))[:80]}"
            if action_type == "open_page":
                return f"web open: {str(action.get('url', ''))[:80]}"
            if action_type == "find_in_page":
                return "web find in page"
        return "web search"
    if tool_type == "mcp_call":
        name = data.get("name", "")
        return f"mcp: {name}" if name else "mcp call"
    return tool_type.replace("_", " ")


def _looks_like_code(output: str, tool_name: str) -> bool:
    """Check if tool output should be syntax-highlighted.

    Uses tool metadata first, then falls back to Pygments lexer
    analysis for unknown tools.
    """
    meta = _TOOL_METADATA.get(tool_name)
    if meta is not None:
        return meta.get("shows_code", False)
    # Fallback: use Pygments to analyze the output.
    return _pygments_can_lex(output)


def _guess_language(tool_name: str, output: str) -> str:
    """Detect the language for syntax highlighting.

    Uses Pygments' ``guess_lexer`` for reliable detection instead
    of fragile substring matching.
    """
    # Tool-specific overrides where we know the output format.
    meta = _TOOL_METADATA.get(tool_name)
    if meta is not None and "language" in meta:
        return meta["language"]

    # Use Pygments for detection.
    try:
        from pygments.lexers import guess_lexer

        lexer = guess_lexer(output[:2000])
        name = lexer.name.lower()
        return _LEXER_NAME_MAP.get(name, "text")
    except Exception:
        return "text"


def _pygments_can_lex(output: str) -> bool:
    """Check if Pygments can identify a non-text lexer for this output."""
    try:
        from pygments.lexers import TextLexer, guess_lexer

        lexer = guess_lexer(output[:1000])
        return not isinstance(lexer, TextLexer)
    except Exception:
        return False
