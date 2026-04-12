# Frontend SDK v2: Blocks with Context

## The Gap in v1

The v1 design (FRONTEND_SDK_LAYERS.md) has three clean layers:
StreamRenderer (events → blocks), BlockFormatter (blocks → display),
TerminalHost (terminal I/O). This works for 90% of cases. But:

1. **Blocks have no attribution.** A `ToolGroup` arrives, but which
   agent produced it? The root coder? The researcher sub-agent?
   The consumer can't route blocks to different panels.

2. **No way to transform the block stream.** "Hide reasoning" or
   "merge text across tool iterations" requires custom code in the
   consumer's handler. Common patterns should be composable.

3. **The formatter is context-blind.** It formats a `ToolGroup` the
   same way regardless of whether it came from the root agent or a
   sub-agent three levels deep. The consumer wants sub-agent blocks
   dimmed, indented, or in a separate panel.

4. **Conversation branching is unaddressed.** Forking from an earlier
   turn to explore a different path requires managing multiple
   sessions manually.

5. **Progressive rendering is impossible.** `TextChunk` is append-only.
   You can't re-render the accumulated text as markdown in-place
   without the host supporting "replace previous output."

---

## Core Idea: Every Block Carries Context

Instead of a flat stream of anonymous blocks, every block knows
**who** produced it, **when**, and **where** in the agent tree.

```python
@dataclass
class BlockContext:
    """Metadata attached to every render block."""
    agent: str          # "coder" or "coder.researcher"
    depth: int          # 0 = root, 1 = sub-agent, 2 = sub-sub, etc.
    turn: int           # conversation turn number (0-based)
    timestamp: float    # wall-clock time (monotonic)
```

Every block type inherits from a base with context:

```python
@dataclass
class RenderBlock:
    context: BlockContext

@dataclass
class TextChunk(RenderBlock):
    text: str

@dataclass
class ToolGroup(RenderBlock):
    executions: list[ToolExecution]

# etc. — all existing block types get context.
```

### Simple case: nothing changes

The consumer ignores context. The `match` still works:

```python
async for block in renderer.stream(session, text):
    match block:
        case TextChunk(text=t):
            print(t, end="", flush=True)
        case ToolGroup(executions=execs):
            for ex in execs:
                print(f"▸ {ex.name}")
```

Context is there but you don't touch it.

### Multi-agent routing

```python
async for block in renderer.stream(session, text):
    panel = block.context.agent  # "coder" or "coder.researcher"
    for item in fmt.format(block):
        host.output(item, panel=panel)
```

One extra line. The host routes output to the right panel. No
multi-stream subscription, no concurrent stream management. Just
metadata on blocks.

### Depth-aware formatting

```python
class MyFormatter(RichBlockFormatter):
    def format(self, block):
        items = super().format(block)
        if block.context.depth > 0:
            # Sub-agent output: dim with agent label
            label = f"[dim blue][{block.context.agent}][/]"
            return [Text.from_markup(label), *items]
        return items
```

The formatter sees the context and adjusts. No subclass needed
for the common case — `RichBlockFormatter` can have a
`show_agent_labels: bool` flag.

---

## Stream Transforms

Async generator functions that wrap the block stream to filter,
merge, or reshape it. No framework — just function composition.

### Built-in transforms

```python
from agent_plane_ui_sdk.transforms import (
    skip_blocks,
    skip_intermediate_ends,
    merge_text_across_iterations,
    only_agent,
    flatten_sub_agents,
)
```

**`skip_blocks(*types)`** — drop specific block types.

```python
from agent_plane_ui_sdk.transforms import skip_blocks

# Hide reasoning:
stream = skip_blocks(ReasoningBlock)(renderer.stream(session, text))
async for block in stream:
    ...
```

**`skip_intermediate_ends()`** — suppress `ResponseEndBlock` events
from tool loop iterations. Only yield the final one.

```python
stream = skip_intermediate_ends()(renderer.stream(session, text))
# Now you see exactly one ResponseEndBlock per turn.
```

**`merge_text_across_iterations()`** — when the tool loop runs
multiple iterations, each produces a separate `TextDone`. This
merges them into one.

```python
stream = merge_text_across_iterations()(renderer.stream(session, text))
# One TextDone with the full response text, not three fragments.
```

**`only_agent(name)`** — filter to blocks from a specific agent.

```python
# Show only the researcher's output:
stream = only_agent("coder.researcher")(renderer.stream(session, text))
```

**`flatten_sub_agents()`** — replace sub-agent tool groups (spawn +
collect) with inline block sequences from the sub-agents. Instead of
seeing `ToolGroup(spawn_sub_agents)`, you see the sub-agents' actual
reasoning, tool calls, and text — as if they were inline.

### Composition

Transforms compose via nesting:

```python
stream = (
    skip_intermediate_ends()(
        skip_blocks(ReasoningBlock)(
            renderer.stream(session, text)
        )
    )
)
```

Or with a helper for readability:

```python
stream = pipe(
    renderer.stream(session, text),
    skip_blocks(ReasoningBlock),
    skip_intermediate_ends(),
    merge_text_across_iterations(),
)
```

Where `pipe` is:

```python
def pipe(stream, *transforms):
    for t in transforms:
        stream = t(stream)
    return stream
```

### Writing custom transforms

A transform is any function that takes an `AsyncIterator[RenderBlock]`
and returns one:

```python
async def add_dividers(stream):
    """Add a visual divider between tool loop iterations."""
    prev_turn = -1
    async for block in stream:
        if isinstance(block, ResponseStartBlock) and block.context.turn > prev_turn:
            if prev_turn >= 0:
                yield DividerBlock()  # custom block type
            prev_turn = block.context.turn
        yield block
```

---

## Replace Semantics: Progressive Rendering

The formatter output model gains **replace** semantics. Instead of
only appending output, the formatter can say "replace the last thing
I rendered."

```python
@dataclass
class Append:
    """Normal output — add below previous output."""
    item: FormattedItem

@dataclass
class Replace:
    """Replace the last N outputs with new items."""
    items: list[FormattedItem]
    count: int = 1  # how many previous outputs to replace

Output = Append | Replace | StreamingText
```

**Why this matters**: progressive markdown rendering. Today,
`TextChunk` streams raw text and `TextDone` optionally re-renders
as markdown. But the re-render appears BELOW the raw text — you
can't replace what's already scrolled past.

With `Replace`, the formatter can re-render in place:

```python
class ProgressiveMarkdownFormatter(RichBlockFormatter):
    """Re-renders accumulated text as markdown on each chunk."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._accumulated = ""

    def format_text_chunk(self, block):
        self._accumulated += block.text
        # Replace the previous markdown render with an updated one.
        return [Replace(
            items=[Markdown(self._accumulated, code_theme=self.code_theme)],
            count=1,
        )]

    def format_text_done(self, block):
        # Final render — replace with the polished version.
        result = [Replace(
            items=[Markdown(block.full_text, code_theme=self.code_theme)],
            count=1,
        )]
        self._accumulated = ""
        return result
```

**How the host implements Replace**:

- **Terminal**: Use ANSI escape codes to move the cursor up, clear
  lines, and re-render. Works within the scrollback buffer. Limited
  to replacing recent output (can't go back 100 lines). Rich's
  `Live` does this internally.

- **Web (WebSocket)**: Send a `{"type": "replace", "count": 1, ...}`
  message. The browser replaces the last N elements in the chat div.

- **Simple fallback**: If the host doesn't support Replace, treat it
  as Append. The output doubles up (raw + rendered) but nothing
  breaks. Graceful degradation.

### What Replace enables

- **Progressive markdown** — re-render on each chunk (above)
- **Tool execution spinner** — show "⏵ Read ..." then replace with
  "✓ Read → result" when done
- **Live token counter** — update a status line in place
- **Collapsible sections** — "show more" that replaces a truncated
  panel with the full one

### Not required for v1

Replace adds host complexity. The simple REPL works without it.
Progressive markdown is a polish feature. The design supports it
but the first implementation can treat all output as Append.

---

## ConversationView: Branching

A separate abstraction above the renderer that manages conversation
trees. Not part of the block stream — it manages sessions.

```python
from agent_plane_ui_sdk import ConversationView

conv = ConversationView(client, model="coder", tool_handler=handler)

# Linear conversation:
stream1 = conv.send("explain this code")       # turn 0
stream2 = conv.send("now fix the bug")          # turn 1

# Fork from turn 0 (explore a different path):
branch = conv.fork(from_turn=0)
stream3 = branch.send("actually, explain the tests")  # turn 0 on branch

# Each stream is an AsyncIterator[RenderBlock]:
async for block in stream3:
    ...

# Navigate the tree:
conv.turns              # [Turn(0, "explain..."), Turn(1, "fix...")]
branch.turns            # [Turn(0, "actually, explain...")]
conv.branches           # [branch]
branch.parent           # conv
branch.fork_point       # 0
```

### What ConversationView owns

- **Session management** — creates/reuses sessions for each turn.
  Tracks `previous_response_id` per branch.
- **Turn history** — stores the user input and response ID for each
  turn. Enables forking.
- **Branch management** — maintains the tree of branches.
- **Renderer creation** — each `send()` returns a block stream from
  a fresh `StreamRenderer`.

### What ConversationView does NOT own

- **Rendering** — it returns block streams. The consumer renders.
- **Layout** — how branches are displayed is the consumer's choice.
- **Persistence** — it uses the server's conversation API. Local
  state is ephemeral.

### How branching works with the server

The server supports forking via `previous_response_id`. When you POST
a new response with `previous_response_id` pointing to a non-latest
response, the server creates a new conversation (copying history up
to the fork point).

`ConversationView.fork(from_turn=N)` creates a new branch whose
session has `previous_response_id` set to the response ID from turn
N. The server handles the conversation duplication.

---

## Formatter with Context Access

The v1 formatter is stateless — it takes a block and returns
renderables. But some rendering needs context:

- **File content** — `FileBlock` has `file_id` but not the bytes.
  To show an inline image, the formatter needs to download it.
- **Agent metadata** — to show the agent's description in a tooltip.
- **Cumulative cost** — to show running cost in the toolbar.

Solution: the formatter optionally receives a **context object** at
construction:

```python
@dataclass
class FormatterContext:
    """Optional context for rich formatting."""
    client: AgentPlaneClient | None = None
    session: Session | None = None

fmt = RichBlockFormatter(
    ctx=FormatterContext(client=client, session=session),
)
```

The base `format_*` methods don't use it. But subclasses can:

```python
class MyFormatter(RichBlockFormatter):
    async def format_file(self, block):
        if self.ctx and self.ctx.client:
            content = await self.ctx.client.files.get_content(block.file_id)
            return [render_image_in_terminal(content)]
        return super().format_file(block)
```

Note: this makes `format_file` async. The host needs to handle
async formatter methods. This is a trade-off — sync formatters are
simpler. Could also provide the client as a separate download helper
rather than making the formatter async.

---

## Putting It All Together

### Simple terminal REPL (~20 lines, same as v1)

```python
renderer = StreamRenderer()
fmt = RichBlockFormatter()

async def on_input(text):
    host.output(fmt.user_message(text))
    async for block in renderer.stream(session, text):
        for item in fmt.format(block):
            host.output(item)

async with TerminalHost(model_name="coder") as host:
    host.output(fmt.welcome("coder"))
    await host.run(on_input)
```

### Multi-agent terminal with sub-agent labels (~25 lines)

```python
fmt = RichBlockFormatter(show_agent_labels=True)

async def on_input(text):
    host.output(fmt.user_message(text))
    stream = pipe(
        renderer.stream(session, text),
        skip_intermediate_ends(),
    )
    async for block in stream:
        for item in fmt.format(block):
            host.output(item)
```

The formatter auto-prefixes sub-agent blocks with `[researcher]`
because `show_agent_labels=True` reads `block.context.depth`.

### Web UI with WebSocket (~20 lines)

```python
renderer = StreamRenderer()

async def ws_handler(websocket, text):
    stream = pipe(
        renderer.stream(session, text),
        skip_intermediate_ends(),
        merge_text_across_iterations(),
    )
    async for block in stream:
        msg = block_to_json(block)
        msg["agent"] = block.context.agent
        msg["depth"] = block.context.depth
        await websocket.send_json(msg)
```

### Branching UI (~30 lines)

```python
conv = ConversationView(client, model="coder", tool_handler=handler)

async def on_input(text):
    if text == "/fork":
        branch = conv.fork(from_turn=current_turn - 1)
        host.output(fmt.format_info(f"Forked from turn {current_turn - 1}"))
        return
    stream = conv.send(text)
    async for block in stream:
        for item in fmt.format(block):
            host.output(item)
```

### Test harness (~10 lines)

```python
renderer = StreamRenderer()
stream = pipe(
    renderer.stream(session, "list files"),
    skip_intermediate_ends(),
    skip_blocks(ReasoningBlock),
)
blocks = [b async for b in stream]

assert isinstance(blocks[0], ResponseStartBlock)
tools = [b for b in blocks if isinstance(b, ToolGroup)]
assert tools[0].executions[0].name == "Glob"
assert tools[0].context.agent == "coder"
```

---

## What This Design Makes Possible

| Capability | How |
|---|---|
| Multi-agent panels | Route by `block.context.agent` |
| Sub-agent labels | Formatter reads `block.context.depth` |
| Hide reasoning | `skip_blocks(ReasoningBlock)` transform |
| Merge fragmented text | `merge_text_across_iterations()` transform |
| Clean test assertions | `skip_intermediate_ends()` + collect blocks |
| Conversation branching | `ConversationView.fork(from_turn=N)` |
| Progressive markdown | `Replace` output + `ProgressiveMarkdownFormatter` |
| Custom code rendering | Override `format_text_done()` in formatter |
| File preview (images) | Formatter with `FormatterContext(client=...)` |
| Tool execution spinner | `Replace` output in `format_tool_group()` |
| Cost tracking | Read `ResponseEndBlock.response.usage` in handler |
| Custom block types | Write a transform that yields custom blocks |
| Agent-specific formatting | `match block.context.agent` in formatter |

## What This Design Cannot Do

| Limitation | Why | Escape hatch |
|---|---|---|
| Live sub-agent streaming | Server doesn't expose sub-agent SSE streams | Raw events + future server API |
| Reorder blocks across full response | Blocks stream in arrival order | `buffer_until_done()` transform + post-hoc sort |
| Change the flushing algorithm | StreamRenderer owns it | Raw events + custom state machine |
| Split-pane terminal layout | TerminalHost is scrollback-only | Use Textual instead |
| Block types the renderer doesn't know | Renderer has a fixed taxonomy | Custom transform that yields new types |

---

## Implementation Order

1. **BlockContext on all blocks** — add `context: BlockContext` field,
   populate in StreamRenderer. Non-breaking: default context is empty.

2. **Built-in transforms** — `skip_blocks`, `skip_intermediate_ends`,
   `merge_text_across_iterations`, `pipe`. Pure functions, no deps.

3. **Formatter context-awareness** — `RichBlockFormatter` reads
   `block.context.depth` for indentation and agent labels.

4. **ConversationView** — separate module. Manages session tree.

5. **Replace semantics** — `Replace` output type, terminal host
   cursor management. Polish feature, not blocking.

6. **FormatterContext** — async formatter methods for file download.
   Nice-to-have, not blocking.
