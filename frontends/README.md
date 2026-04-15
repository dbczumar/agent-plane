# Frontends

Tools for building user interfaces on top of agent-plane.

## Structure

```
frontends/
  sdks/
    python/                     pip-installable SDK
      pyproject.toml            pip install -e frontends/sdks/python
      agent_plane_ui_sdk/       import agent_plane_ui_sdk
        terminal/               Rich + prompt_toolkit components
  repl/                         Standalone REPL (uses the SDK)
```

Claude Code skills for frontend development live in the top-level
`skills/` directory (e.g. `skills/repl-sdk/`).

## Python UI SDK

The SDK provides three layers for building frontends:

1. **StreamRenderer** — consumes the agent's SSE event stream and
   emits semantic blocks (`TextChunk`, `ToolGroup`, `ReasoningBlock`,
   etc.). Pure state machine, no I/O, no terminal dependencies.

2. **RichBlockFormatter** — converts blocks to Rich renderables.
   Subclass and override one method to customize any block type.

3. **TerminalHost** — manages prompt_toolkit: pinned input bar,
   background task streaming, Escape to cancel, persistent history.

Non-terminal frontends (web, Slack, tests) use only layer 1.

### Install

```bash
pip install -e frontends/sdks/python
```

### Minimal REPL

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

        async def on_input(text):
            host.output(fmt.user_message(text))
            async for block in pipe(
                renderer.stream(session, text),
                skip_intermediate_ends(),
            ):
                for item in fmt.format(block):
                    host.output(item)
                await asyncio.sleep(0)

        async with host:
            host.output(fmt.welcome("my agent"))
            await host.run(on_input)

asyncio.run(main())
```

### Minimal Web UI

```python
from agent_plane_ui_sdk import StreamRenderer, TextChunk, ToolGroup, ResponseEndBlock, pipe, skip_intermediate_ends

async def handle(websocket, session, text):
    renderer = StreamRenderer()
    async for block in pipe(renderer.stream(session, text), skip_intermediate_ends()):
        match block:
            case TextChunk(text=t):
                await websocket.send_json({"type": "text", "chunk": t})
            case ToolGroup(executions=execs):
                await websocket.send_json({"type": "tools", "data": [
                    {"name": e.name, "output": e.output} for e in execs
                ]})
            case ResponseEndBlock(status=s):
                await websocket.send_json({"type": "done", "status": s})
```

### Customization

Override one formatter method:

```python
class MyFormatter(RichBlockFormatter):
    def format_tool_group(self, block):
        from rich.tree import Tree
        tree = Tree("Tools")
        for ex in block.executions:
            tree.add(f"{ex.name} → {(ex.output or '')[:50]}")
        return [tree]
```

Use transforms to reshape the block stream:

```python
from agent_plane_ui_sdk import pipe, skip_blocks, ReasoningBlock

stream = pipe(
    renderer.stream(session, text),
    skip_blocks(ReasoningBlock),  # Hide thinking
)
```

## Reference Implementation

The built-in REPL at `agent_plane/repl/` demonstrates all features:
streaming, tool calls, reasoning, slash commands, conversation
switching, elapsed timer. See `agent_plane/repl/_repl.py`.

## Design Documents

- `designs/CLIENT_AND_REPL.md` — client library + REPL design
- `designs/STREAM_RENDERER.md` — block renderer design
- `designs/FRONTEND_SDK_LAYERS.md` — three-layer architecture
- `designs/FRONTEND_SDK_V2.md` — blocks with context, transforms
