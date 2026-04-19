#!/usr/bin/env python
"""Quickstart tour of ``agent_plane_client`` — the headless Python SDK.

Spins up a temporary agent-plane server, deploys the ``archer``
research agent, and runs five demos that each highlight one layer of
the SDK:

  1. One-shot invocation       — send a prompt, print the answer.
  2. Raw event stream          — typed SSE events as they arrive.
  3. Semantic blocks           — higher-level units via BlockStream.
  4. Multi-turn session        — previous_response_id handled for you.
  5. Client-side tool          — register a local function; the agent
                                 calls it, your code runs it, the
                                 result flows back automatically.

Run from the agent-plane repo root::

    OPENAI_API_KEY=sk-... python examples/clients/python/quickstart.py

Each demo is a self-contained coroutine; delete the ones you don't
need and keep whichever fits your use case.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

from agent_plane_client import (
    AgentPlaneClient,
    AgentPlaneError,
    BlockStream,
    LocalServer,
    ReasoningBlock,
    ResponseEndBlock,
    ResponseStartBlock,
    TextDone,
    ToolCallInfo,
    ToolGroup,
    ToolHandler,
    ToolResultBlock,
    pipe,
    skip_intermediate_ends,
)

# Raw event types live under the private ``_events`` module. Most
# callers will prefer ``BlockStream`` (demo 3) over raw events, but
# demo 2 shows what's underneath for advanced use cases.
from agent_plane_client._events import MessageDone, TextDelta

AGENT_PATH = "examples/agents/archer/"
MODEL = "archer"


# ── Demo 1: One-shot invocation ──────────────────────────────────────────


async def demo_one_shot(client: AgentPlaneClient) -> None:
    """Fire and collect: send a prompt, pull the final answer, print."""
    print("\n─── Demo 1: one-shot invocation ───")
    session = client.session(model=MODEL)
    final_text = ""
    async for event in session.send("Say hi in one short sentence. No tools."):
        if isinstance(event, MessageDone):
            for item in event.content:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    final_text += str(item.get("text", ""))
    print(f"  response_id = {session.current_response_id}")
    print(f"  answer      = {final_text!r}")


# ── Demo 2: Raw event stream ─────────────────────────────────────────────


async def demo_raw_events(client: AgentPlaneClient) -> None:
    """Iterate the typed wire events — 1:1 with server-side SSE frames.

    Useful when you're building a frontend that needs full control
    over event ordering (web UI, Slack bot, custom state machine).
    Otherwise prefer demo 3's BlockStream.
    """
    print("\n─── Demo 2: raw event stream ───")
    session = client.session(model=MODEL)
    counts: dict[str, int] = {}
    streamed = ""
    async for event in session.send("Count from one to three. No tools."):
        name = type(event).__name__
        counts[name] = counts.get(name, 0) + 1
        if isinstance(event, TextDelta):
            streamed += event.delta
    print(f"  event histogram = {counts}")
    print(f"  streamed text   = {streamed!r}")


# ── Demo 3: Semantic blocks via BlockStream ──────────────────────────────


async def demo_blocks(client: AgentPlaneClient) -> None:
    """BlockStream folds raw events into semantic units you can pattern-match.

    This is the recommended layer for building UIs: you get typed
    blocks for text, tool calls, reasoning, lifecycle, etc. without
    reimplementing the stream state machine.
    """
    print("\n─── Demo 3: semantic blocks (BlockStream) ───")
    session = client.session(model=MODEL)
    block_stream = BlockStream()
    async for block in pipe(
        block_stream.stream(session, "Give a one-word greeting. No tools."),
        skip_intermediate_ends(),  # Tool loops emit one End per iteration;
        # this leaves only the final one.
    ):
        match block:
            case ResponseStartBlock(model=m, response_id=rid):
                print(f"  ▸ start      model={m} id={rid}")
            case ReasoningBlock(summary_text=s):
                snippet = s.replace("\n", " ")[:60]
                print(f"  ▸ reasoning  {snippet!r}")
            case TextDone(full_text=t):
                print(f"  ▸ text       {t!r}")
            case ToolGroup(executions=execs):
                for e in execs:
                    out = (e.output or "")[:60].replace("\n", " ")
                    print(f"  ▸ tool       {e.name}({e.args_summary}) → {out!r}")
            case ResponseEndBlock(status=s):
                print(f"  ▸ end        status={s}")


# ── Demo 4: Multi-turn session ───────────────────────────────────────────


async def demo_multi_turn(client: AgentPlaneClient) -> None:
    """One Session object, multiple turns. The SDK threads response IDs."""
    print("\n─── Demo 4: multi-turn session ───")
    session = client.session(model=MODEL)
    block_stream = BlockStream()

    async def turn(prompt: str) -> str:
        text = ""
        async for block in pipe(block_stream.stream(session, prompt), skip_intermediate_ends()):
            if isinstance(block, TextDone):
                text = block.full_text
        return text

    first = await turn("My name is Corey. Reply 'Hi Corey' and nothing else. No tools.")
    print(f"  turn 1: {first!r}  (response_id={session.current_response_id})")
    second = await turn("What did I say my name was? One word only, no tools.")
    print(f"  turn 2: {second!r}  (response_id={session.current_response_id})")


# ── Demo 5: Client-side tool ─────────────────────────────────────────────


async def demo_client_tool(client: AgentPlaneClient) -> None:
    """Register a local function as a tool; let the agent invoke it.

    The tool schema (OpenAI function-calling format) is sent to the
    server at session start. When the agent decides to call it, the
    SDK runs ``execute`` locally and posts the result back, so from
    the agent's perspective it's just another tool.
    """
    print("\n─── Demo 5: client-side tool ───")

    def get_current_time(call: ToolCallInfo) -> str:
        """Return the current UTC time as a JSON-encoded ISO-8601 string."""
        now = datetime.now(timezone.utc).isoformat()
        return json.dumps({"now": now})

    handler = ToolHandler(
        schemas=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Return the current UTC time as ISO-8601.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        execute=get_current_time,
    )

    session = client.session(model=MODEL, tool_handler=handler)
    block_stream = BlockStream()
    async for block in pipe(
        block_stream.stream(
            session,
            "What is the current UTC time? Call the get_current_time tool and report the result.",
        ),
        skip_intermediate_ends(),
    ):
        if isinstance(block, ToolGroup):
            for e in block.executions:
                print(f"  ▸ tool call  {e.name}({e.args_summary})")
        elif isinstance(block, ToolResultBlock):
            out = block.output[:80].replace("\n", " ")
            print(f"  ▸ tool result {block.name} → {out}")
        elif isinstance(block, TextDone):
            print(f"  ▸ answer     {block.full_text!r}")


# ── Driver ───────────────────────────────────────────────────────────────


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (the archer agent uses OpenAI).")

    print(f"Spinning up a local agent-plane server + deploying '{MODEL}'...")
    async with LocalServer(agent_path=AGENT_PATH) as server:
        print(f"  base_url = {server.base_url}\n")
        try:
            await demo_one_shot(server.client)
            await demo_raw_events(server.client)
            await demo_blocks(server.client)
            await demo_multi_turn(server.client)
            await demo_client_tool(server.client)
        except AgentPlaneError as e:
            print(f"\nSDK error: {e}")
            raise
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
