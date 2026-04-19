#!/usr/bin/env python
"""Quickstart for ``agent_plane_client`` — the Python client SDK.

Spins up a temporary agent-plane server, deploys the ``archer``
research agent, and demonstrates the patterns most apps need:

  1. query()                  — ask, get text (+ any files) back.
  2. query(stream=True)       — stream text chunks as they arrive.
  3. multi-turn session       — conversation state handled for you.
  4. client-side tools        — register a local function the agent
                                can call; your code runs it.
  5. text file attachment     — attach a local text file to the prompt.
  6. image attachment         — attach a PNG; archer describes it.
  7. agent-produced files     — agent writes a file; we download it.
  8. (advanced) BlockStream   — pointer for when text isn't enough.

Run from the agent-plane repo root::

    OPENAI_API_KEY=sk-... python examples/clients/python/quickstart.py
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import struct
import tempfile
import zlib
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
    """Send a prompt, get a QueryResult with .text and .files."""
    print("\n─── 1. query() — blocking, returns QueryResult ───")
    result = await client.query(
        model=MODEL,
        input="Say hi in one short sentence. No tools.",
    )
    print(f"  answer: {result.text!r}")
    print(f"  files:  {result.files}")


# ── 2. Streaming query ─────────────────────────────────────────────────


async def demo_streaming(client: AgentPlaneClient) -> None:
    """Stream text chunks as they arrive. After iteration, .files is populated."""
    print("\n─── 2. query(stream=True) — QueryStream ───")
    stream = await client.query(
        model=MODEL,
        input="Count from one to five, one number per line. No tools.",
        stream=True,
    )
    print("  stream: ", end="", flush=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    print(f"  files:  {stream.files}")


# ── 3. Multi-turn session ───────────────────────────────────────────────


async def demo_multi_turn(client: AgentPlaneClient) -> None:
    """Use a Session for multi-turn conversations — response IDs are
    threaded automatically."""
    print("\n─── 3. Session — multi-turn ───")
    session = client.session(model=MODEL)
    a = await session.query("My name is Corey. Reply 'Hi Corey' and nothing else. No tools.")
    print(f"  turn 1: {a.text!r}")
    b = await session.query("What name did I just tell you? One word, no tools.")
    print(f"  turn 2: {b.text!r}")


# ── 4. Client-side tools ────────────────────────────────────────────────


async def demo_client_tools(client: AgentPlaneClient) -> None:
    """Register a local @tool-decorated function; the agent calls it."""
    print("\n─── 4. Client-side tool (@tool) ───")

    # The schema is derived automatically from the function's type
    # hints and docstring — no hand-rolled JSON. Tools work the same
    # with streaming and non-streaming; the SDK runs the tool loop
    # under the hood.
    result = await client.query(
        model=MODEL,
        input="What's the current UTC time? Call get_current_time and report it.",
        tools=[get_current_time],
    )
    print(f"  answer: {result.text!r}")


# ── 5. Text file attachment ─────────────────────────────────────────────


async def demo_text_file_attachment(client: AgentPlaneClient) -> None:
    """Attach a text file to the prompt. The server uploads + routes content."""
    print("\n─── 5. Text file attachment ───")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("The secret word is 'rhubarb'.\n")
        tmp_path = pathlib.Path(f.name)

    try:
        result = await client.query(
            model=MODEL,
            input="What secret word is in the attached file? Reply with just the word, no tools.",
            files=[str(tmp_path)],
        )
        print(f"  answer: {result.text!r}")
    finally:
        tmp_path.unlink()


# ── 6. Image attachment (vision) ────────────────────────────────────────


def _synthesize_split_png(
    width: int = 200,
    height: int = 200,
    color_a: tuple[int, int, int] = (220, 60, 60),
    color_b: tuple[int, int, int] = (60, 90, 220),
) -> bytes:
    """Generate a PNG split diagonally between ``color_a`` and ``color_b``.

    Uses only the stdlib (``struct`` + ``zlib``) so the quickstart
    has no binary fixtures to commit. Pixels where ``x + y < width``
    get ``color_a``; the rest get ``color_b``.
    """
    rows: list[bytes] = []
    for y in range(height):
        row = b"\x00"  # PNG per-row filter byte (0 = None)
        for x in range(width):
            row += bytes(color_a) if x + y < width else bytes(color_b)
        rows.append(row)
    raw = b"".join(rows)

    def _chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), 0s for filter/compression/interlace.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


async def demo_image_attachment(client: AgentPlaneClient) -> None:
    """Attach a PNG image. The SDK detects the mimetype and sends it as
    an ``input_image`` block; the vision-capable agent describes it."""
    print("\n─── 6. Image attachment (vision) ───")
    tmp_path = pathlib.Path(tempfile.mktemp(suffix=".png"))
    tmp_path.write_bytes(_synthesize_split_png())
    try:
        result = await client.query(
            model=MODEL,
            input=(
                "What two colors are in this image and how are they "
                "positioned? One sentence, no tools."
            ),
            files=[str(tmp_path)],
        )
        print(f"  answer: {result.text!r}")
    finally:
        tmp_path.unlink()


# ── 7. Agent-produced files ─────────────────────────────────────────────


async def demo_agent_produced_files(client: AgentPlaneClient) -> None:
    """Ask the agent to create a file; download it via result.files."""
    print("\n─── 7. Agent-produced files ───")
    session = client.session(model=MODEL)
    result = await session.query(
        "Write the exact text 'produced-by-archer' to a file called "
        "greeting.txt using code_sandbox, then call upload_file on "
        "'greeting.txt' so I can download it. Reply 'done' and nothing else."
    )
    print(f"  answer:       {result.text!r}")
    print(f"  files.count:  {len(result.files)}")

    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="ap-quickstart-"))
    for f in result.files:
        dest = await client.files.download(f.id, out_dir / f.filename)
        print(f"  downloaded:   {dest} ({dest.read_bytes()!r})")


# ── 8. Advanced: BlockStream for tool/reasoning display ─────────────────


async def demo_blockstream_pointer() -> None:
    """If you're building a UI that shows tool calls, reasoning, lifecycle,
    etc., you want ``BlockStream`` — see ``sdks/README.md`` or the repl-sdk
    skill. ``query()`` is the happy path for app code that just needs text."""
    print("\n─── 8. BlockStream (advanced — not run here) ───")
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
        await demo_text_file_attachment(server.client)
        await demo_image_attachment(server.client)
        await demo_agent_produced_files(server.client)
        await demo_blockstream_pointer()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
