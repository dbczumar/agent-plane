"""Unit tests for StreamRenderer — mock events → blocks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from agent_plane_ui_sdk._blocks import (
    ReasoningBlock,
    TextChunk,
    TextDone,
    ToolGroup,
)
from agent_plane_ui_sdk._events import (
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
from agent_plane_ui_sdk._renderer import StreamRenderer
from agent_plane_ui_sdk._types import Response


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
def renderer() -> StreamRenderer:
    return StreamRenderer(text_flush_threshold=10)


@pytest.mark.asyncio()
async def test_simple_text_response(renderer: StreamRenderer) -> None:
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

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    assert "ResponseStartBlock" in types
    assert "TextDone" in types
    assert "ResponseEndBlock" in types

    # Verify TextDone has the full text.
    text_done = next(b for b in blocks if isinstance(b, TextDone))
    assert text_done.full_text == "Hello world!"
    assert not text_done.has_code_blocks


@pytest.mark.asyncio()
async def test_text_with_code_blocks(renderer: StreamRenderer) -> None:
    """Code fences in text → has_code_blocks=True."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            TextDelta(delta="```python\nprint('hi')\n```"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]
    text_done = next(b for b in blocks if isinstance(b, TextDone))
    assert text_done.has_code_blocks


@pytest.mark.asyncio()
async def test_reasoning_block(renderer: StreamRenderer) -> None:
    """Reasoning events → ReasoningStartBlock + ReasoningBlock."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            ReasoningStarted(),
            ReasoningDelta(delta="Let me think..."),
            ReasoningSummaryDelta(delta="Summary here"),
            TextDelta(delta="Answer"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    assert "ReasoningStartBlock" in types
    assert "ReasoningBlock" in types

    reasoning = next(b for b in blocks if isinstance(b, ReasoningBlock))
    assert reasoning.reasoning_text == "Let me think..."
    assert reasoning.summary_text == "Summary here"


@pytest.mark.asyncio()
async def test_tool_group_with_results(renderer: StreamRenderer) -> None:
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

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]
    tool_groups = [b for b in blocks if isinstance(b, ToolGroup)]

    # First ToolGroup: emitted immediately with output=None (call line).
    assert len(tool_groups) >= 1
    assert tool_groups[0].executions[0].name == "Read"

    # ToolResultBlock: emitted when result arrives.
    from agent_plane_ui_sdk._blocks import ToolResultBlock

    results = [b for b in blocks if isinstance(b, ToolResultBlock)]
    assert len(results) == 1
    assert results[0].output == "file content"


@pytest.mark.asyncio()
async def test_block_context_agent_name(renderer: StreamRenderer) -> None:
    """Blocks carry the agent name from the response."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response(model="my-agent")),
            TextDelta(delta="hi"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response(model="my-agent")),
        ]
    )

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]

    for block in blocks:
        assert block.ctx.agent == "my-agent"


@pytest.mark.asyncio()
async def test_text_chunk_flushing(renderer: StreamRenderer) -> None:
    """Text chunks flush on newlines and word boundaries."""
    # renderer has threshold=10
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            TextDelta(delta="short\nline two is longer than threshold characters"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]
    chunks = [b for b in blocks if isinstance(b, TextChunk)]

    # At least one chunk from the newline split.
    assert len(chunks) >= 1
    # First chunk should be "short\n" (from the newline).
    assert chunks[0].text == "short\n"


@pytest.mark.asyncio()
async def test_empty_response(renderer: StreamRenderer) -> None:
    """Response with no text or tools → just start + end blocks."""
    session = FakeSession(
        [
            ResponseCreated(response=_make_response()),
            ResponseCompleted(response=_make_response()),
        ]
    )

    blocks = [b async for b in renderer.stream(session, "test")]  # type: ignore[arg-type]
    types = [type(b).__name__ for b in blocks]

    assert types == ["ResponseStartBlock", "ResponseEndBlock"]
