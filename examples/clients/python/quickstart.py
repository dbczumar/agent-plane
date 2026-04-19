#!/usr/bin/env python
"""Quickstart for ``agent_plane_client`` — the Python client SDK.

Spins up a temporary agent-plane server, deploys the ``archer``
research agent, and demonstrates the patterns most apps need:

  1. query()                  — ask, get text back.
  2. query(stream=True)       — stream text chunks as they arrive.
  3. multi-turn session       — conversation state handled for you.
  4. client-side tools        — register a local function the agent
                                can call; your code runs it.
  5. file attachments         — attach a local file to the prompt.
  6. (advanced) BlockStream   — pointer for when text isn't enough.

Run from the agent-plane repo root::

    OPENAI_API_KEY=sk-... python examples/clients/python/quickstart.py
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import tempfile
from datetime import datetime, timezone

from agent_plane_client import (
    AgentPlaneClient,
    LocalServer,
    tool,
)

AGENT_PATH = "examples/agents/archer/"
MODEL = "archer"


# ── Tool definitions ────────────────────────────────────────────────────


# @tool requires module-level functions (so the same decorator can be
# used for bundled agent tools that get re-imported in a subprocess).
# Put your client-side tools at module scope.
@tool
def get_current_time() -> dict[str, str]:
    """Return the current UTC time as ISO-8601."""
    return {"now": datetime.now(timezone.utc).isoformat()}


# ── 1. Non-streaming query ──────────────────────────────────────────────


async def demo_query(client: AgentPlaneClient) -> None:
    """Send a prompt, get the final text back as a string."""
    print("\n─── 1. query() — blocking, returns str ───")
    text = await client.query(
        model=MODEL,
        input="Say hi in one short sentence. No tools.",
    )
    print(f"  answer: {text!r}")


# ── 2. Streaming query ─────────────────────────────────────────────────


async def demo_streaming(client: AgentPlaneClient) -> None:
    """Stream text chunks as they arrive."""
    print("\n─── 2. query(stream=True) — AsyncIterator[str] ───")
    stream = await client.query(
        model=MODEL,
        input="Count from one to five, one number per line. No tools.",
        stream=True,
    )
    print("  stream: ", end="", flush=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


# ── 3. Multi-turn session ───────────────────────────────────────────────


async def demo_multi_turn(client: AgentPlaneClient) -> None:
    """Use a Session for multi-turn conversations — response IDs are
    threaded automatically."""
    print("\n─── 3. Session — multi-turn ───")
    session = client.session(model=MODEL)
    a = await session.query("My name is Corey. Reply 'Hi Corey' and nothing else. No tools.")
    print(f"  turn 1: {a!r}")
    b = await session.query("What name did I just tell you? One word, no tools.")
    print(f"  turn 2: {b!r}")


# ── 4. Client-side tools ────────────────────────────────────────────────


async def demo_client_tools(client: AgentPlaneClient) -> None:
    """Register a local @tool-decorated function; the agent calls it."""
    print("\n─── 4. Client-side tool (@tool) ───")

    # The schema is derived automatically from the function's type
    # hints and docstring — no hand-rolled JSON. Tools work the same
    # with streaming and non-streaming; the SDK runs the tool loop
    # under the hood.
    text = await client.query(
        model=MODEL,
        input="What's the current UTC time? Call get_current_time and report it.",
        tools=[get_current_time],
    )
    print(f"  answer: {text!r}")


# ── 5. File attachment (multi-modal input) ──────────────────────────────


async def demo_file_attachment(client: AgentPlaneClient) -> None:
    """Attach a local file to the prompt. Works with text files, images,
    PDFs, etc. — the server handles upload + content routing."""
    print("\n─── 5. File attachment ───")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("The secret word is 'rhubarb'.\n")
        tmp_path = pathlib.Path(f.name)

    try:
        text = await client.query(
            model=MODEL,
            input="What secret word is in the attached file? Reply with just the word, no tools.",
            files=[str(tmp_path)],
        )
        print(f"  answer: {text!r}")
    finally:
        tmp_path.unlink()


# ── 6. Advanced: BlockStream for tool/reasoning display ─────────────────


async def demo_blockstream_pointer() -> None:
    """If you're building a UI that shows tool calls, reasoning, lifecycle,
    etc., you want ``BlockStream`` — see ``sdks/README.md`` or the repl-sdk
    skill. ``query()`` is the happy path for app code that just needs text."""
    print("\n─── 6. BlockStream (advanced — not run here) ───")
    print("  For UIs that display tool calls / reasoning / lifecycle,")
    print("  use `from agent_plane_client import BlockStream`.")
    print("  See `sdks/README.md` and the `repl-sdk` skill.")


# ── Driver ───────────────────────────────────────────────────────────────


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (the archer agent uses OpenAI).")

    print(f"Spinning up a local agent-plane server + deploying '{MODEL}'...")
    async with LocalServer(agent_path=AGENT_PATH) as server:
        print(f"  base_url = {server.base_url}\n")
        await demo_query(server.client)
        await demo_streaming(server.client)
        await demo_multi_turn(server.client)
        await demo_client_tools(server.client)
        await demo_file_attachment(server.client)
        await demo_blockstream_pointer()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
