# REPL SDK API Reference

## Block Types

Every block inherits from `RenderBlock` with `ctx: BlockContext`.

### BlockContext

```python
@dataclass
class BlockContext:
    agent: str = ""       # "coder" or "coder.researcher"
    depth: int = 0        # 0 = root, 1 = sub-agent
    turn: int = 0         # Conversation turn number
    timestamp: float      # time.monotonic()
```

### Blocks

| Block | Fields |
|---|---|
| `ResponseStartBlock` | `model: str`, `response_id: str` |
| `ReasoningStartBlock` | — |
| `ReasoningBlock` | `reasoning_text: str`, `summary_text: str` |
| `ToolGroup` | `executions: list[ToolExecution]`, `iteration: int` |
| `NativeToolBlock` | `tool_type: str`, `label: str`, `data: dict` |
| `TextChunk` | `text: str` |
| `TextDone` | `full_text: str`, `has_code_blocks: bool` |
| `ErrorBlock` | `message: str`, `source: str` |
| `RetryBlock` | `source: str`, `attempt: int`, `max_attempts: int`, `delay_seconds: float` |
| `CompactionBlock` | — |
| `FileBlock` | `file_id: str`, `filename: str \| None` |
| `ResponseEndBlock` | `status: str`, `response: Response \| None` |

### ToolExecution

```python
@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, object]
    args_summary: str        # Pre-formatted for display
    call_id: str
    agent_name: str
    executed_by: str         # "client" or "server"
    output: str | None
```

## StreamRenderer

```python
renderer = StreamRenderer(text_flush_threshold=30)

async for block in renderer.stream(session, input, *, files=None):
    ...  # AnyBlock instances
```

- `text_flush_threshold` — min chars before word-boundary flush (default 30)
- `session` — `Session` from `client.session()`
- `input` — user text or content block list
- `files` — optional file paths to upload and attach

## Transforms

```python
from agent_plane_ui_sdk import pipe, skip_blocks, skip_intermediate_ends, merge_text_across_iterations, only_agent

# Compose:
stream = pipe(source, transform1, transform2, ...)

# Built-in:
skip_blocks(*types)              # Drop specific block types
skip_intermediate_ends()         # One ResponseEndBlock per turn
merge_text_across_iterations()   # Merge TextDone across tool loops
only_agent(name)                 # Filter to one agent
```

Custom transform — any `async def` that takes and yields `AsyncIterator[AnyBlock]`.

## RichBlockFormatter

```python
fmt = RichBlockFormatter(
    accent_color="#d87757",
    code_theme="monokai",
    max_result_lines=30,
    show_agent_labels=False,
)

items: list[FormattedItem] = fmt.format(block)
# FormattedItem = RenderableType | StreamingText
```

### Override Points

All return `list[FormattedItem]`:

| Method | Block type |
|---|---|
| `format_response_start(block)` | `ResponseStartBlock` |
| `format_reasoning_start(block)` | `ReasoningStartBlock` |
| `format_reasoning(block)` | `ReasoningBlock` |
| `format_text_chunk(block)` | `TextChunk` |
| `format_text_done(block)` | `TextDone` |
| `format_tool_group(block)` | `ToolGroup` |
| `format_native_tool(block)` | `NativeToolBlock` |
| `format_error(block)` | `ErrorBlock` |
| `format_retry(block)` | `RetryBlock` |
| `format_compaction(block)` | `CompactionBlock` |
| `format_file(block)` | `FileBlock` |
| `format_response_end(block)` | `ResponseEndBlock` |

### Non-Block Helpers

| Method | Returns |
|---|---|
| `fmt.welcome(model)` | Welcome banner panel |
| `fmt.user_message(text)` | User message with accent marker |
| `fmt.goodbye()` | Goodbye message |

### StreamingText

```python
@dataclass
class StreamingText:
    text: str
```

Marker type. `TerminalHost.output()` buffers these and flushes as
full terminal-width lines with word-wrap and indent.

## TerminalHost

```python
host = TerminalHost(
    prompt_marker="❯",
    accent_color="#d87757",
    history_file="~/.agent-plane-history",
    model_name="coder",
)
```

### Properties and Methods

| Member | Type | Description |
|---|---|---|
| `await host.run(handler)` | method | Input loop. Handler runs as background task. |
| `host.output(item)` | method | Display above prompt. Handles `StreamingText` and Rich renderables. |
| `host.start_timer()` | method | Start elapsed counter in toolbar. |
| `host.stop_timer()` | method | Stop elapsed counter. |
| `host.cancel()` | method | Cancel running tasks. Escape key calls this. |
| `host.is_busy` | property | True if handler task running. |
| `host.text_indent` | attribute | Indent for streamed text (default `"   "`). |
| `host.on_help` | attribute | F1 key callback (`Callable[[], None] \| None`). |
| `build_prompt()` | method | Override for custom prompt layout. Returns `FormattedText`. |
| `build_toolbar()` | method | Override for custom toolbar. Returns `FormattedText`. |

### Streaming Text Behavior

`StreamingText` items are buffered in `host._text_buffer`:
- **Newline flush**: splits on `\n`, wraps each line via `textwrap.fill`
- **Width flush**: when buffer exceeds `terminal_width - indent_width`, wraps at word boundary
- **Transition flush**: remaining buffer printed when a non-streaming item arrives
- **Unicode**: `wcwidth` for CJK/emoji display width

## Session

```python
session = client.session(model="coder", tool_handler=handler)

async for event in session.send(input, *, files=None):
    ...  # StreamEvent instances (raw events, before renderer)

await session.cancel()       # Cancel in-progress response
session.reset()              # New conversation
session.resume_from_response(response_id)  # Switch conversation
session.current_response_id  # Latest response ID
session.is_streaming          # True if response in progress
```

`session.send()` auto-steers if a response is in progress, or
starts a new turn if terminal. The caller doesn't decide.

## Client

```python
async with AgentPlaneClient(base_url="http://localhost:8080") as client:
    # Agents
    agent = await client.agents.create(bundle_path="./my-agent/", replace=True)
    agents = await client.agents.list()

    # Conversations
    convos = await client.conversations.list()
    items = await client.conversations.list_items(conv_id)

    # Responses (low-level — prefer session + renderer)
    response = await client.responses.create(model="coder", input="hello")
    async for event in client.responses.stream(model="coder", input="hello"):
        ...

    # Files
    file = await client.files.upload("./data.csv")
```

## LocalServer

```python
async with LocalServer(agent_path="./my-agent/") as server:
    client = server.client       # Pre-configured AgentPlaneClient
    print(server.base_url)       # http://127.0.0.1:PORT
```

Starts a temporary server with SQLite, deploys the agent, shuts down
on exit. The agent path can be a directory (tars automatically) or a
`.tar.gz` file.
