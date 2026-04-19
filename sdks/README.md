# SDKs

Python packages for integrating with agent-plane.

## Structure

```
sdks/
  python-client/           # Headless HTTP/SSE client
    pyproject.toml
    agent_plane_client/    # import agent_plane_client
  frontend/                # Terminal UI layer (Rich + prompt_toolkit)
    pyproject.toml
    agent_plane_ui_sdk/    # import agent_plane_ui_sdk
      terminal/
```

Claude Code skills for SDK development live in the top-level `skills/`
directory (e.g. `skills/repl-sdk/`).

## `agent_plane_client` — the headless client

Pure HTTP/SSE client. No Rich, no prompt_toolkit, no terminal
dependencies. Use this for:

- Scripts that invoke an agent and collect output.
- Web frontends, Slack bots, test harnesses — anything non-terminal.
- As the foundation layer for `agent_plane_ui_sdk` below.

Three levels of abstraction are available:

1. **Raw events** — `session.send()` yields typed wire events
   (`ResponseCreated`, `TextDelta`, `ToolCallDone`, etc.). 1:1 with SSE.
2. **Semantic blocks** — `BlockStream` folds events into higher-level
   units (`TextChunk`, `ToolGroup`, `ReasoningBlock`, …). Frameworks
   consuming these don't need to reimplement the stream state machine.
3. **Composable transforms** — `pipe`, `skip_blocks`,
   `skip_intermediate_ends`, `merge_text_across_iterations`, `only_agent`.

### Install

```bash
pip install -e sdks/python-client
```

### Minimal invocation

```python
import asyncio
from agent_plane_client import AgentPlaneClient

async def main():
    async with AgentPlaneClient(base_url="http://localhost:8080") as client:
        session = client.session(model="archer")
        async for event in session.send("hello"):
            print(event)

asyncio.run(main())
```

### Using semantic blocks (web, Slack, or any custom UI)

```python
from agent_plane_client import (
    BlockStream, TextChunk, ToolGroup, ResponseEndBlock,
    pipe, skip_intermediate_ends,
)

async def handle(websocket, session, text):
    block_stream = BlockStream()
    async for block in pipe(
        block_stream.stream(session, text),
        skip_intermediate_ends(),
    ):
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

## `agent_plane_ui_sdk` — the terminal frontend

Thin layer on top of `agent_plane_client` for building terminal REPLs.
Provides:

- **RichBlockFormatter** — converts `StreamBlock` values to Rich
  renderables. Subclass and override one method to customize.
- **TerminalHost** — manages prompt_toolkit: pinned input bar,
  background streaming, Escape to cancel, persistent history.

### Install

```bash
pip install -e sdks/frontend
```

(Pulls in `agent-plane-client` as a dependency.)

### Minimal REPL

```python
import asyncio
from agent_plane_client import (
    AgentPlaneClient, LocalServer, BlockStream,
    pipe, skip_intermediate_ends,
)
from agent_plane_ui_sdk import RichBlockFormatter, TerminalHost

async def main():
    async with LocalServer(agent_path="./my-agent/") as server:
        client = server.client
        session = client.session(model="my-agent")
        block_stream = BlockStream()
        fmt = RichBlockFormatter()
        host = TerminalHost(model_name="my agent")

        async def on_input(text):
            host.output(fmt.user_message(text))
            async for block in pipe(
                block_stream.stream(session, text),
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
from agent_plane_client import pipe, skip_blocks, ReasoningBlock

stream = pipe(
    block_stream.stream(session, text),
    skip_blocks(ReasoningBlock),  # Hide thinking
)
```

## Reference Implementation

The built-in REPL at `agent_plane/repl/` demonstrates all features:
streaming, tool calls, reasoning, slash commands, conversation
switching, elapsed timer. See `agent_plane/repl/_repl.py`.

## Design Documents

- `designs/CLIENT_AND_REPL.md` — client library + REPL design
- `designs/STREAM_RENDERER.md` — block stream design
- `designs/FRONTEND_SDK_LAYERS.md` — three-layer architecture
- `designs/FRONTEND_SDK_V2.md` — blocks with context, transforms
