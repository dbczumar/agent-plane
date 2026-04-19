"""Unit tests for Session.query / Session.query(stream=True).

These tests exercise the convenience wrappers by mocking
``Session.send()`` directly. The real event → block folding
path is covered separately in ``test_stream.py``; here we only
verify the wrapping behavior (collect → str, stream → chunks).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from agent_plane_client._events import (
    MessageDone,
    ResponseCompleted,
    ResponseCreated,
    ResponseInProgress,
    StreamEvent,
    TextDelta,
)
from agent_plane_client._session import Session
from agent_plane_client._types import Response


def _make_response(
    response_id: str = "resp_1",
    status: str = "completed",
    model: str = "test-agent",
) -> Response:
    """Minimal Response for synthesizing SSE events in tests."""
    return Response(id=response_id, status=status, model=model)


class _ScriptedSession(Session):
    """A Session subclass whose ``send()`` replays a fixed event list.

    ``Session.query()`` dispatches to ``self._collect_text`` /
    ``self._stream_text``, which in turn use ``BlockStream`` to fold
    events from ``self.send(...)``. We subclass Session so those
    helpers are actually present; we override ``__init__`` to avoid
    needing a real client, and override ``send`` to replay a script.

    :param events: Pre-baked events to yield on every ``send()`` call.
    """

    def __init__(self, events: list[StreamEvent]) -> None:
        # Deliberately skip Session.__init__ — we don't need a client
        # for these tests, and faking one would be more work than it's
        # worth. query() only reads self._collect_text / self._stream_text
        # (inherited) and the overridden self.send (below).
        self._events = events

    async def send(  # type: ignore[override]
        self,
        input: Any,
        *,
        files: Any = None,
        instructions: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


# ── query() — non-streaming returns final text ──────────────────────────


@pytest.mark.asyncio()
async def test_query_returns_final_text_simple() -> None:
    """A single text response → query() returns the joined text."""
    session = _ScriptedSession(
        events=[
            ResponseCreated(response=_make_response()),
            ResponseInProgress(response=_make_response(status="in_progress")),
            TextDelta(delta="Hello "),
            TextDelta(delta="world"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    # Use the real Session.query, bound to our scripted session instance.
    # Session.query doesn't touch any other Session attribute, so a
    # duck-typed object with send() is sufficient.
    text = await session.query("hi")

    # Exact content check — proves the TextDelta values traversed the
    # BlockStream folding + TextDone accumulation inside query().
    # If this returns "" or a MagicMock, the wrapper didn't collect
    # the block's full_text correctly.
    assert text == "Hello world"


@pytest.mark.asyncio()
async def test_query_empty_response_returns_empty_string() -> None:
    """No text events → query() returns ''. Must not raise or return None."""
    session = _ScriptedSession(
        events=[
            ResponseCreated(response=_make_response()),
            ResponseInProgress(response=_make_response(status="in_progress")),
            # No TextDelta / MessageDone — e.g. a cancelled turn.
            ResponseCompleted(response=_make_response()),
        ]
    )
    text = await session.query("hi")

    # Empty string is the contract, not None — keeps the return type
    # stable (`str`) regardless of whether the agent produced text.
    assert text == ""


# ── query(stream=True) — yields text chunks as they arrive ──────────────


@pytest.mark.asyncio()
async def test_query_stream_yields_text_chunks() -> None:
    """stream=True → AsyncIterator[str] of text, in order."""
    session = _ScriptedSession(
        events=[
            ResponseCreated(response=_make_response()),
            ResponseInProgress(response=_make_response(status="in_progress")),
            TextDelta(delta="Hello "),
            TextDelta(delta="world"),
            MessageDone(content=[]),
            ResponseCompleted(response=_make_response()),
        ]
    )

    result = await session.query("hi", stream=True)
    chunks = [c async for c in result]

    # Concatenating all yielded chunks must equal the full text.
    # A wrong result here means either TextChunk blocks weren't
    # passed through, or their ``text`` field was not extracted.
    assert "".join(chunks) == "Hello world"

    # At least one chunk. Non-empty proof the stream actually yielded
    # (a broken wrapper could return an empty iterator and still
    # satisfy the join-equals-expected check above when that expected
    # value is also "").
    assert len(chunks) >= 1
