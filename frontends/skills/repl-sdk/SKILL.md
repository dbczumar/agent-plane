---
name: repl-sdk
description: Build terminal REPLs for agent-plane using the Python UI SDK. Use when the user asks about building a REPL, terminal chat, or CLI frontend for an agent.
user-invocable: false
---

# Building REPLs with the Agent Plane UI SDK

## Architecture

A REPL has three layers. See `reference.md` for the full API.

```
StreamRenderer      events → blocks        (state machine, no I/O)
RichBlockFormatter   blocks → Rich output   (panels, colors, markdown)
TerminalHost         prompt_toolkit I/O      (pinned input, background tasks)
```

## Minimal Working REPL

This is a complete, working REPL. Start here and customize.

```python
import asyncio
from agent_plane_ui_sdk import (
    AgentPlaneClient, LocalServer, StreamRenderer,
    pipe, skip_intermediate_ends,
)
from agent_plane_ui_sdk.terminal import RichBlockFormatter, TerminalHost

async def main():
    async with LocalServer(agent_path="./my-agent/") as server:
        client = server.client
        session = client.session(model="my-agent")
        renderer = StreamRenderer()
        fmt = RichBlockFormatter()
        host = TerminalHost(model_name="my agent")

        async def on_input(text: str) -> None:
            host.output(fmt.user_message(text))
            async for block in pipe(
                renderer.stream(session, text),
                skip_intermediate_ends(),
            ):
                for item in fmt.format(block):
                    host.output(item)
                await asyncio.sleep(0)  # Required: flush output.

        async with host:
            host.output(fmt.welcome("my agent"))
            await host.run(on_input)

asyncio.run(main())
```

## Adding Slash Commands

```python
COMMANDS: dict[str, tuple[str, object]] = {}

def cmd(name, help_text):
    def register(fn):
        COMMANDS[name] = (help_text, fn)
        return fn
    return register

@cmd("/new", "Start new conversation")
async def cmd_new(arg, session, client, host, fmt):
    session.reset()
    host.output(fmt.welcome(session._model))

@cmd("/help", "Show commands")
async def cmd_help(arg, session, client, host, fmt):
    from rich.text import Text
    for name, (desc, _) in COMMANDS.items():
        host.output(Text.from_markup(f"  {name}  [dim]{desc}[/]"))

# In the handler:
async def on_input(text):
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        entry = COMMANDS.get(parts[0])
        if entry:
            _, handler = entry
            await handler(parts[1] if len(parts) > 1 else "", session, client, host, fmt)
        return
    # ... normal message handling
```

## Adding Client-Side Tools

```python
from agent_plane_ui_sdk import ToolHandler, ToolCallInfo

handler = ToolHandler(
    schemas=[
        {"type": "function", "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {
                "file_path": {"type": "string"}
            }, "required": ["file_path"]},
        }},
    ],
    execute=lambda call: Path(call.arguments["file_path"]).read_text(),
)

session = client.session(model="coder", tool_handler=handler)
```

The SDK handles the tool loop automatically — stream → detect calls →
execute → POST results → stream again. The REPL sees one continuous
block stream.

## Adding an Elapsed Timer

```python
from agent_plane_ui_sdk import ResponseStartBlock, ResponseEndBlock

class TimedFormatter(RichBlockFormatter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_time = None

    def format_response_start(self, block):
        self._start_time = block.ctx.timestamp
        return super().format_response_start(block)

    def format_response_end(self, block):
        items = super().format_response_end(block)
        if self._start_time is not None:
            elapsed = block.ctx.timestamp - self._start_time
            from rich.text import Text
            items.append(Text.from_markup(f"   [dim]{elapsed:.1f}s[/]"))
            self._start_time = None
        return items

# Plus a live counter in the toolbar:
host.start_timer()   # before streaming
host.stop_timer()    # after streaming
```

## Adding Conversation Switching

```python
@cmd("/switch", "Switch conversation")
async def cmd_switch(arg, session, client, host, fmt):
    if not arg:
        convos = await client.conversations.list(limit=20)
        # ... show table
        return
    items = await client.conversations.list_items(arg, limit=100)
    last_id = None
    for item in reversed(items):
        rid = item.get("response_id")
        if isinstance(rid, str):
            last_id = rid
            break
    if last_id:
        session.reset()
        session.resume_from_response(last_id)
```

## Adding an Initial Message (Auto-Greet)

```python
async def run_repl(client, agent_name, tool_handler, *, initial_message=None):
    # ... setup ...
    async with host:
        host.output(fmt.welcome(ui_name))
        if initial_message:
            asyncio.create_task(on_input(initial_message))
        await host.run(on_input)
```

## Customizing Block Rendering

Override one method on the formatter:

```python
class MyFormatter(RichBlockFormatter):
    def format_tool_group(self, block):
        from rich.tree import Tree
        tree = Tree("Tools")
        for ex in block.executions:
            tree.add(f"{ex.name}({ex.args_summary})")
        return [tree]

    def format_reasoning(self, block):
        # Hide reasoning entirely:
        return []

fmt = MyFormatter(accent_color="#00aaff", code_theme="dracula")
```

## Hiding Block Types with Transforms

```python
from agent_plane_ui_sdk import pipe, skip_blocks, ReasoningBlock

stream = pipe(
    renderer.stream(session, text),
    skip_blocks(ReasoningBlock),         # No thinking panels
    skip_intermediate_ends(),            # Clean end events
)
```

## Key Gotchas

1. **`await asyncio.sleep(0)` after each block** — required so
   prompt_toolkit's stdout proxy flushes. Without it, output
   buffers until the handler finishes.

2. **`skip_intermediate_ends()`** — always use this. The tool loop
   emits `ResponseEndBlock` per iteration, not just at the end.

3. **Agent name display** — hyphens and underscores in config.yaml
   agent names are auto-replaced with spaces in the UI.

4. **Streaming text** — `TextChunk` blocks are buffered by
   `TerminalHost` and flushed as full terminal-width lines with
   indent and word-wrap. No partial-line writes.

5. **Tool result timing** — `ToolGroup` blocks arrive when results
   are guaranteed available, not on `ResponseCompleted`.

## Testing and QA

### Test Scenarios

Every REPL must handle ALL of these. Test each one individually
before calling the REPL "done."

| # | Scenario | What to verify | Agent to test with |
|---|---|---|---|
| 1 | **Simple text** | Text streams line-by-line, no truncation, proper indent | Any agent: "say hello" |
| 2 | **Long response** | 500+ char story streams smoothly, consistent line width | "tell me a 500 character story" |
| 3 | **Reasoning** | "thinking…" appears immediately, reasoning panel renders after | "what is 2+2? think carefully" |
| 4 | **Client-side tools** | Tool call + result display, tool loop iterates correctly | coder with --client-tools: "read README.md" |
| 5 | **Server-side tools** | Tool call shows as informational, result renders | archer: "search the web for RLHF" |
| 6 | **Sub-agents** | spawn_sub_agents + collect, tool calls attributed to sub-agent | archer: "research and fact-check X" |
| 7 | **Multiple tool iterations** | Tool loop runs 2-3 times before final text | coder: "glob *.py then read the first file" |
| 8 | **Code blocks** | Syntax-highlighted markdown re-render after streaming | "write a python hello world" |
| 9 | **Steering** | Type while streaming, message delivers as steer | Send "focus on X" while agent is responding |
| 10 | **Cancellation** | Escape cancels, agent stops, prompt returns | Press Escape during a long response |
| 11 | **Multi-turn** | Second message continues conversation, context preserved | "hello" → "what did I just say?" |
| 12 | **Slash commands** | /help, /new, /switch all work | Try each command |
| 13 | **Error handling** | Bad agent name, network drop, server crash | Use a nonexistent model name |
| 14 | **Empty input** | Enter with no text does nothing | Just press Enter |
| 15 | **Long input** | Multi-line or very long user message truncated in display | Paste a paragraph |

### Automated Test Loop

For each scenario, write a test that runs the REPL non-interactively
by calling `run_repl` internals directly (no prompt_toolkit):

```python
import asyncio
from agent_plane_ui_sdk import *

async def test_scenario(agent_path, prompt, expect_blocks):
    """Run one scenario and verify expected block types appear."""
    async with LocalServer(agent_path=agent_path) as server:
        client = server.client
        session = client.session(model=server_agent_name)
        renderer = StreamRenderer()
        blocks = []
        async for block in pipe(
            renderer.stream(session, prompt),
            skip_intermediate_ends(),
        ):
            blocks.append(block)
            print(f"  {type(block).__name__}", end="")
            if isinstance(block, TextChunk):
                print(f": {block.text[:40]!r}")
            elif isinstance(block, ToolGroup):
                print(f": {[e.name for e in block.executions]}")
            else:
                print()

        # Verify expected blocks appeared.
        block_types = {type(b).__name__ for b in blocks}
        for expected in expect_blocks:
            assert expected in block_types, f"Missing {expected}!"
        print(f"  PASSED ({len(blocks)} blocks)\n")

# Run all scenarios:
asyncio.run(test_scenario(
    "examples/agents/coder/",
    "Say hello in one sentence. No tools.",
    ["ResponseStartBlock", "TextChunk", "TextDone", "ResponseEndBlock"],
))
asyncio.run(test_scenario(
    "examples/agents/coder/",
    "What is 2+2? Think carefully. No tools.",
    ["ReasoningStartBlock", "ReasoningBlock", "TextChunk"],
))
```

### Agent-Driven QA Loop

When building a REPL, iterate through scenarios until ALL pass:

1. **Run through the scenario list above, one at a time.**
   Start with scenario 1 (simple text). Don't move to scenario 2
   until scenario 1 works perfectly.

2. **For each scenario:**
   a. Run the automated test to verify blocks flow correctly.
   b. Run the actual REPL interactively and verify the visual output.
   c. If either fails: diagnose, fix, retest BOTH.
   d. If both pass: move to the next scenario.

3. **After all scenarios pass individually, run a combined session:**
   Start the REPL, do scenarios 1, 4, 7, 9, 11, 12 in sequence
   without restarting. This tests state management across turns.

4. **Regression check:** After any code change, rerun scenarios
   1 (text), 4 (tools), 7 (multi-iteration), and 9 (steering).
   These are the most fragile.

### Common Failures and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| No output for 10+ seconds | Missing `await asyncio.sleep(0)` | Add it after each block in the handler |
| Text appears all at once | Streaming not flushing | Check `TerminalHost` text buffer logic |
| Garbled lines during steering | Partial-line writes | Only write complete lines through host.output |
| "thinking…" never appears | `ReasoningStartBlock` not handled | Add it to the formatter dispatch |
| Duplicate text | TextChunk streamed AND TextDone re-rendered | Only re-render on `has_code_blocks` |
| Tool results empty | ToolGroup emitted before results arrive | Emit on `ResponseCreated`/`TextDelta`, not `ResponseCompleted` |
| Missing terminal event | Server uses `response.TaskStatus.COMPLETED` | SSE parser must normalize enum event names |
| Double "cancelled" | Both `_cancel_stream` and `CancelledError` handler print | Only print in one place |
| Input bar disappears | Sequential approach (no background tasks) | Use `asyncio.create_task` + `patch_stdout` |
| ANSI codes visible | `patch_stdout` without `raw=True` | Use `patch_stdout(raw=True)` |

### Don't Stop Until It Works

**Do NOT claim the REPL works until you have personally verified
each scenario.** "The code looks right" is not verification.
Run it, watch it, test edge cases. The most common bugs are in
streaming timing, state transitions, and prompt_toolkit coordination
— none of which are visible from reading code.

If you can't run the REPL interactively (no API key, headless
environment), say so explicitly. Run the automated block-level tests
instead and note which scenarios couldn't be visually verified.

## Reference Implementation

The full REPL with all features is at `agent_plane/repl/_repl.py`.
It demonstrates: streaming, tool calls, reasoning, slash commands,
conversation switching, elapsed timer, initial message, F1 help.
