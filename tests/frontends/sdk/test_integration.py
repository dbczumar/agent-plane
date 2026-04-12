"""Integration tests — StreamRenderer against a real agent-plane server.

These require an LLM API key. Run with:
    pytest tests/frontends/sdk/test_integration.py --llm-api-key $(cat /tmp/mykey) -v

Skipped automatically if no API key is provided.
"""

from __future__ import annotations

import os

import pytest
from agent_plane_ui_sdk import (
    LocalServer,
    StreamRenderer,
    TextDone,
    pipe,
    skip_intermediate_ends,
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
    """Basic text response streams correctly through renderer."""
    async with LocalServer(agent_path="examples/agents/coder/") as server:
        client = server.client
        session = client.session(model="coder")
        renderer = StreamRenderer()

        blocks = []
        async for block in pipe(
            renderer.stream(session, "Say hello in one word. No tools."),
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
        renderer = StreamRenderer()

        blocks = []
        async for block in pipe(
            renderer.stream(session, "What is 2+2? Think carefully. No tools."),
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
        renderer = StreamRenderer()

        blocks = []
        async for block in pipe(
            renderer.stream(session, "Say hi. No tools."),
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
        renderer = StreamRenderer()

        # Turn 1.
        blocks1 = []
        async for block in pipe(
            renderer.stream(session, "Say hello. No tools."),
            skip_intermediate_ends(),
        ):
            blocks1.append(block)

        assert session.current_response_id is not None
        first_id = session.current_response_id

        # Turn 2 — should continue the conversation.
        blocks2 = []
        async for block in pipe(
            renderer.stream(session, "Say goodbye. No tools."),
            skip_intermediate_ends(),
        ):
            blocks2.append(block)

        assert session.current_response_id is not None
        assert session.current_response_id != first_id
