"""Integration tests — BlockStream against a real agent-plane server.

These require an LLM API key. Run with:
    pytest tests/frontends/sdk/test_integration.py --llm-api-key $LLM_API_KEY -v

Skipped automatically if no API key is provided.
"""

from __future__ import annotations

import os

import pytest
from agent_plane_client import (
    BlockStream,
    LocalServer,
    TextDone,
    pipe,
    skip_intermediate_ends,
    tool,
)

# Skip all tests if no API key.
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture()
def api_key() -> str:
    return os.environ["OPENAI_API_KEY"]


@pytest.mark.asyncio()
async def test_simple_streaming(api_key: str) -> None:
    """Basic text response streams correctly through the block stream."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        client = server.client
        session = client.session(model="coder")
        block_stream = BlockStream()

        blocks = []
        async for block in pipe(
            block_stream.stream(session, "Say hello in one word. No tools."),
            skip_intermediate_ends(),
        ):
            blocks.append(block)

        types = {type(b).__name__ for b in blocks}
        assert "ResponseStartBlock" in types
        assert "ResponseEndBlock" in types

        # Should have at least some text.
        text_dones = [b for b in blocks if isinstance(b, TextDone)]
        assert len(text_dones) >= 1
        assert len(text_dones[0].full_text) > 0


@pytest.mark.asyncio()
async def test_reasoning_appears(api_key: str) -> None:
    """Reasoning-enabled agent produces reasoning blocks."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        client = server.client
        session = client.session(model="coder")
        block_stream = BlockStream()

        blocks = []
        async for block in pipe(
            block_stream.stream(session, "What is 2+2? Think carefully. No tools."),
            skip_intermediate_ends(),
        ):
            blocks.append(block)

        types = {type(b).__name__ for b in blocks}
        # Coder agent has reasoning_effort: medium.
        assert "ReasoningStartBlock" in types or "ReasoningBlock" in types


@pytest.mark.asyncio()
async def test_block_context_populated(api_key: str) -> None:
    """Blocks carry correct agent name in context."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        client = server.client
        session = client.session(model="coder")
        block_stream = BlockStream()

        blocks = []
        async for block in pipe(
            block_stream.stream(session, "Say hi. No tools."),
            skip_intermediate_ends(),
        ):
            blocks.append(block)

        for block in blocks:
            assert block.ctx.agent == "coder", (
                f"{type(block).__name__} has agent={block.ctx.agent!r}"
            )


@pytest.mark.asyncio()
async def test_multi_turn_session(api_key: str) -> None:
    """Session tracks previous_response_id across turns."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        client = server.client
        session = client.session(model="coder")
        block_stream = BlockStream()

        # Turn 1.
        blocks1 = []
        async for block in pipe(
            block_stream.stream(session, "Say hello. No tools."),
            skip_intermediate_ends(),
        ):
            blocks1.append(block)

        assert session.current_response_id is not None
        first_id = session.current_response_id

        # Turn 2 — should continue the conversation.
        blocks2 = []
        async for block in pipe(
            block_stream.stream(session, "Say goodbye. No tools."),
            skip_intermediate_ends(),
        ):
            blocks2.append(block)

        assert session.current_response_id is not None
        assert session.current_response_id != first_id


# ── query() — the convenience API typical apps use ─────────────────────


@pytest.mark.asyncio()
async def test_client_query_returns_text(api_key: str) -> None:
    """client.query() returns the agent's response as a string."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        text = await server.client.query(
            model="coder",
            input="Reply with exactly the word 'pong'. No tools.",
        )

    # Exact substring — proves the response text flowed all the way
    # through BlockStream → TextDone → query() and the agent actually
    # followed the instruction. A failure here means either the
    # pipeline dropped the text or the prompt wasn't respected.
    assert "pong" in text.lower(), f"Expected 'pong' in response, got {text!r}"


@pytest.mark.asyncio()
async def test_client_query_streaming_yields_chunks(api_key: str) -> None:
    """client.query(stream=True) yields text chunks; joined == full text."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        stream = await server.client.query(
            model="coder",
            input=("Write exactly this text and nothing else: 'alpha beta gamma'. No tools."),
            stream=True,
        )
        chunks = [c async for c in stream]

    joined = "".join(chunks)

    # At least one chunk means the iterator actually yielded — a
    # broken wrapper could return an empty iterator and silently pass
    # the content check below.
    assert len(chunks) >= 1, f"Expected at least one chunk, got {chunks}"

    # Content check — the joined text must contain the exact tokens
    # the agent was asked to emit. If this fails, either the stream
    # skipped text chunks (wrapper bug) or the prompt wasn't followed
    # (flaky LLM — rerun).
    assert all(w in joined.lower() for w in ("alpha", "beta", "gamma")), (
        f"Expected 'alpha', 'beta', 'gamma' in streamed text, got {joined!r}"
    )


@pytest.mark.asyncio()
async def test_session_query_multi_turn(api_key: str) -> None:
    """session.query() across turns — conversation state is threaded."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        session = server.client.session(model="coder")
        first = await session.query(
            "My name is Corey. Reply 'Hi Corey' and nothing else. No tools."
        )
        second = await session.query(
            "What name did I tell you? Reply with just the name, no tools."
        )

    # Turn 1 should contain the greeting.
    assert "corey" in first.lower(), f"Turn 1 response: {first!r}"

    # Turn 2 should echo the name from turn 1 — proves the session
    # actually threaded previous_response_id. If the session didn't,
    # the agent would have no memory of "Corey" and turn 2 would say
    # "I don't know" or similar.
    assert "corey" in second.lower(), (
        f"Turn 2 should recall 'Corey' from turn 1 but got {second!r}. "
        f"If this fails, session.query didn't forward previous_response_id."
    )


_secret_fruit_calls = 0


@tool
def _get_secret_fruit() -> dict[str, str]:
    """Return the secret fruit."""
    # Nested tool closures aren't allowed by @tool (module-level
    # requirement); use a module-level counter instead.
    global _secret_fruit_calls
    _secret_fruit_calls += 1
    return {"answer": "banana"}


@pytest.mark.asyncio()
async def test_query_with_client_tool(api_key: str) -> None:
    """client.query(tools=[@tool fn]) runs the tool loop transparently."""
    global _secret_fruit_calls
    _secret_fruit_calls = 0

    async with LocalServer(agent_path="examples/agents/coder/") as server:
        text = await server.client.query(
            model="coder",
            input=(
                "Call the _get_secret_fruit tool and report the 'answer' "
                "field from the result. Reply with just the word."
            ),
            tools=[_get_secret_fruit],
        )
    call_count = _secret_fruit_calls

    # Tool must have been invoked exactly once — proves query(tools=)
    # actually triggered the tool loop (client-side tool ran).
    # If 0, the LLM never called the tool; if >1, the loop ran extra
    # iterations unexpectedly.
    assert call_count == 1, (
        f"Expected exactly 1 tool invocation, got {call_count}. "
        f"If 0, the LLM didn't emit the tool call; if >1, the tool "
        f"loop re-iterated."
    )

    # Final text must contain the tool's answer — proves the result
    # flowed back to the LLM and the LLM incorporated it into its reply.
    assert "banana" in text.lower(), (
        f"Expected 'banana' (from tool result) in final text, got {text!r}"
    )
