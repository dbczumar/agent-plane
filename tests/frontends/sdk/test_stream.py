"""Unit tests for BlockStream — mock events → blocks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from agent_plane_client._blocks import (
    ReasoningBlock,
    ReasoningChunk,
    ReasoningStartBlock,
    TextChunk,
    TextDone,
    ToolGroup,
)
from agent_plane_client._events import (
    MessageDone,
    ReasoningDelta,
    ReasoningStarted,
    ReasoningSummaryDelta,
    ResponseCompleted,
    ResponseCreated,
    ResponseInProgress,
    TextDelta,
    ToolCall,
    ToolResult,
)
from agent_plane_client._stream import BlockStream
from agent_plane_client._types import Response


def _make_response(
    response_id: str = "resp_1",
    status: str = "completed",
    model: str = "test-agent",
) -> Response:
    """Create a minimal Response for testing."""
    return Response(
        id=response_id,
        status=status,
        model=model,
    )


class FakeSession:
    """Fake session that yields pre-defined events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def send(
        self,
        input: Any,
        *,
        files: Any = None,
    ) -> AsyncIterator[Any]:
        for event in self._events:
            yield event


@pytest.fixture()
def block_stream() -> BlockStream:
    return BlockStream(text_flush_threshold=10)


@pytest.mark.asyncio()
async def test_simple_text_response(block_stream: BlockStream) -> None:
    """Simple text → ResponseStart, TextChunks, TextDone, ResponseEnd."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            ResponseInProgress(response=_make_response(status="in_progress")),
            TextDelta(delta="Hello "),
            TextDelta(delta="world!"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    assert "ResponseStartBlock" in types
    assert "TextDone" in types
    assert "ResponseEndBlock" in types

    # Verify TextDone has the full text.
    text_done = next(b for b in blocks if isinstance(b, TextDone))
    assert text_done.full_text == "Hello world!"
    assert not text_done.has_code_blocks


@pytest.mark.asyncio()
async def test_text_with_code_blocks(block_stream: BlockStream) -> None:
    """Code fences in text → has_code_blocks=True."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            TextDelta(delta="```python\nprint('hi')\n```"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    text_done = next(b for b in blocks if isinstance(b, TextDone))
    assert text_done.has_code_blocks


@pytest.mark.asyncio()
async def test_reasoning_streams_chunks_live(block_stream: BlockStream) -> None:
    """
    Reasoning deltas must surface as :class:`ReasoningChunk` blocks
    while reasoning is in progress so the TUI can render live
    progress (e.g. Codex commands) instead of waiting until the
    section ends to dump a single panel.

    Contract: when chunks fire, the trailing :class:`ReasoningBlock`
    is suppressed — emitting both would make renderers show the same
    text twice (once streaming, once as a panel).
    """
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            ReasoningStarted(),
            ReasoningDelta(delta="Let me think...\n"),
            ReasoningSummaryDelta(delta="Summary here\n"),
            TextDelta(delta="Answer"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    # Start indicator and at least one streamed chunk.
    assert "ReasoningStartBlock" in types
    chunk_texts = [b.text for b in blocks if isinstance(b, ReasoningChunk)]
    assert chunk_texts, (
        f"No ReasoningChunk emitted — reasoning would be invisible "
        f"during the section. Got: {types}"
    )
    # Both delta sources must reach the consumer (the executor maps
    # Codex events to ReasoningSummaryDelta; LLM-native reasoning
    # comes through ReasoningDelta). Concatenated chunk text must
    # contain content from both.
    joined = "".join(chunk_texts)
    assert "Let me think" in joined, (
        f"ReasoningDelta payload missing from chunks. Joined: {joined!r}"
    )
    assert "Summary here" in joined, (
        f"ReasoningSummaryDelta payload missing from chunks. Joined: {joined!r}"
    )

    # ReasoningBlock must be suppressed — chunks already covered it.
    assert "ReasoningBlock" not in types, (
        f"ReasoningBlock leaked alongside chunks; renderers would "
        f"show the same text twice. Got: {types}"
    )


@pytest.mark.asyncio()
async def test_reasoning_started_without_deltas_emits_block(
    block_stream: BlockStream,
) -> None:
    """
    Edge case: ``ReasoningStarted`` arrives but no deltas follow
    before the section closes. With no chunks to stream, the
    :class:`ReasoningBlock` must still fire so non-streaming
    renderers know reasoning happened (even if empty).
    """
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            ReasoningStarted(),
            # No deltas — straight to text.
            TextDelta(delta="Direct answer"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    assert "ReasoningStartBlock" in types
    assert "ReasoningChunk" not in types
    # Block fires because no chunks did.
    assert "ReasoningBlock" in types
    block = next(b for b in blocks if isinstance(b, ReasoningBlock))
    assert block.reasoning_text == ""
    assert block.summary_text == ""


@pytest.mark.asyncio()
async def test_reasoning_delta_without_started_emits_implicit_start(
    block_stream: BlockStream,
) -> None:
    """
    Codex events arrive as bridged ``ReasoningSummaryDelta`` with
    no preceding ``ReasoningStarted`` (the executor maps directly
    from ``codex/event`` to deltas). The block stream must
    synthesize a :class:`ReasoningStartBlock` on the first delta
    so the formatter still gets its "thinking…" anchor.
    """
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            # No ReasoningStarted — straight into a delta.
            ReasoningSummaryDelta(delta="$ ls /tmp\n"),
            TextDelta(delta="Result"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]

    start_idx = next(
        (i for i, b in enumerate(blocks) if isinstance(b, ReasoningStartBlock)),
        None,
    )
    chunk_idx = next(
        (i for i, b in enumerate(blocks) if isinstance(b, ReasoningChunk)),
        None,
    )
    assert start_idx is not None, (
        "Implicit ReasoningStartBlock missing — Codex-bridged deltas "
        "would arrive without a section header in the TUI."
    )
    assert chunk_idx is not None, "ReasoningChunk missing for the bridged delta."
    assert start_idx < chunk_idx, "Start block must precede the first chunk."


@pytest.mark.asyncio()
async def test_tool_group_with_results(block_stream: BlockStream) -> None:
    """ToolCall + ToolResult + next ResponseCreated → ToolGroup with output."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response(response_id="resp_1")),
            ToolCall(
                name="Read",
                arguments={"file_path": "/tmp/f"},
                call_id="c1",
                status="completed",
                agent_name="coder",
            ),
            ResponseCompleted(response=_make_response(response_id="resp_1")),
            # Client SDK yields ToolResult between iterations:
            ToolResult(call_id="c1", output="file content"),
            # Next iteration:
            ResponseCreated(response=_make_response(response_id="resp_2")),
            TextDelta(delta="Done"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response(response_id="resp_2")),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    tool_groups = [b for b in blocks if isinstance(b, ToolGroup)]

    # First ToolGroup: emitted immediately with output=None (call line).
    assert len(tool_groups) >= 1
    assert tool_groups[0].executions[0].name == "Read"

    # ToolResultBlock: emitted when result arrives.
    from agent_plane_client._blocks import ToolResultBlock

    results = [b for b in blocks if isinstance(b, ToolResultBlock)]
    assert len(results) == 1
    assert results[0].output == "file content"


@pytest.mark.asyncio()
async def test_block_context_agent_name(block_stream: BlockStream) -> None:
    """Blocks carry the agent name from the response."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response(model="my-agent")),
            TextDelta(delta="hi"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response(model="my-agent")),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]

    for block in blocks:
        assert block.ctx.agent == "my-agent"


@pytest.mark.asyncio()
async def test_text_chunk_flushing(block_stream: BlockStream) -> None:
    """Text chunks flush on newlines and word boundaries."""
    # block_stream has threshold=10
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            TextDelta(delta="short\nline two is longer than threshold characters"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    chunks = [b for b in blocks if isinstance(b, TextChunk)]

    # At least one chunk from the newline split.
    assert len(chunks) >= 1
    # First chunk should be "short\n" (from the newline).
    assert chunks[0].text == "short\n"


@pytest.mark.asyncio()
async def test_empty_response(block_stream: BlockStream) -> None:
    """Response with no text or tools → just start + end blocks."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in block_stream.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    assert types == ["ResponseStartBlock", "ResponseEndBlock"]
