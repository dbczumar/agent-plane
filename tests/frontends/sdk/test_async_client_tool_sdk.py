"""
Phase 5 part D — minimal SDK changes for async client tools.

Covers two narrow surface areas that lift the SDK to recognize
the Phase 5 protocol:

1. ``build_tool_handler`` emits ``"synchronous": false`` on the
   wire schema for ``@tool(synchronous=False)``-decorated
   functions, so the server's ``parse_client_side_tool_spec``
   sees the opt-out and routes the call through the
   async-dispatch path instead of parking.
2. The SSE parser converts the ``response.client_task.cancel``
   event the server emits during parent-cancel propagation into
   a typed :class:`ClientTaskCancel` event so consumers can
   handle local-task cancellation.

The full client-side async-dispatch lifecycle (asyncio task
tracking, automatic ``async_tool_results`` PATCH, cancel-on-SSE
handling, 1-hour cap) is intentionally out of scope here — see
``tests/_adherence/phase5.md`` D6 for the deferred work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from agent_plane_client._events import ClientTaskCancel, ToolCall, ToolResult
from agent_plane_client._sse import parse_sse_stream
from agent_plane_client.tools import build_tool_handler, tool

# ── @tool / build_tool_handler ────────────────────────────────


@tool
async def _sync_echo(text: str) -> str:
    """Echo back the input verbatim.

    Args:
        text: Text to echo.
    """
    return text


@tool(synchronous=False)
async def _async_long_compute(n: int) -> dict[str, int]:
    """Pretend to do long-running work that returns asynchronously.

    Args:
        n: Iteration count.
    """
    return {"n": n, "result": n * 2}


def test_build_tool_handler_omits_synchronous_for_sync_tools() -> None:
    """
    Default-synchronous tools must NOT carry ``"synchronous"``
    on the wire — the field is an explicit Phase 5 opt-out, not
    a redundant default. If the schema serialized
    ``synchronous: true`` for every tool, the wire shape would
    diverge from what existing OpenResponses-compatible clients
    expect.
    """
    handler = build_tool_handler([_sync_echo])
    assert len(handler.schemas) == 1
    schema = handler.schemas[0]
    # No top-level synchronous field — defaults take over server-side.
    assert "synchronous" not in schema, (
        f"Sync tools must not emit 'synchronous' on the wire; got schema={schema!r}"
    )
    # Sanity: function name and structure unchanged.
    assert schema["type"] == "function"
    fn_field = schema["function"]
    assert isinstance(fn_field, dict)
    assert fn_field["name"] == "_sync_echo"


def test_build_tool_handler_emits_synchronous_false_for_async_tools() -> None:
    """
    ``@tool(synchronous=False)`` must surface in the emitted
    schema as a top-level ``"synchronous": false`` so the
    server's ``parse_client_side_tool_spec`` (which reads
    ``raw["synchronous"]`` at the top level, not inside
    ``function``) routes the call through the async-dispatch
    path. If the field is missing or placed inside ``function``,
    the server would default to the legacy parking path and the
    LLM would block on the call instead of receiving the
    ``{task_id, kind: "client_tool"}`` handle.
    """
    handler = build_tool_handler([_async_long_compute])
    assert len(handler.schemas) == 1
    schema = handler.schemas[0]
    # Top-level (NOT inside function) per server contract.
    assert schema.get("synchronous") is False, (
        f"Async tools must emit 'synchronous: false' at the "
        f"schema's top level; got schema={schema!r}"
    )
    fn_field = schema["function"]
    assert isinstance(fn_field, dict)
    assert "synchronous" not in fn_field, (
        "synchronous must NOT live inside 'function' — server parser reads it from the outer dict."
    )


def test_build_tool_handler_mixes_sync_and_async() -> None:
    """
    A mixed handler must keep each tool's schema independent —
    flipping one tool's ``synchronous`` must not pollute the
    other. Catches a future regression where build_tool_handler
    might (incorrectly) reuse a shared dict and propagate one
    tool's flag onto every other entry.
    """
    handler = build_tool_handler([_sync_echo, _async_long_compute])
    by_name = {
        s["function"]["name"]: s  # type: ignore[index]
        for s in handler.schemas
    }
    assert "synchronous" not in by_name["_sync_echo"], (
        "_sync_echo must remain free of the synchronous field."
    )
    assert by_name["_async_long_compute"].get("synchronous") is False, (
        "_async_long_compute must keep its synchronous=False flag."
    )


# ── SSE parser: response.client_task.cancel ─────────────────


async def _bytes(*frames: bytes) -> AsyncIterator[bytes]:
    """
    Yield each frame as a discrete chunk (mimics httpx streaming).

    :param frames: One or more raw SSE byte frames to feed into
        :func:`parse_sse_stream`. Each frame is yielded as its
        own chunk so the parser sees realistic
        ``aiter_bytes()`` boundaries instead of one giant
        concatenated buffer.
    :yields: Each frame's bytes verbatim, in argument order.
    """
    for frame in frames:
        yield frame


@pytest.mark.asyncio
async def test_sse_parser_emits_client_task_cancel() -> None:
    """
    The server emits ``response.client_task.cancel`` when an
    in-flight ``kind="client_tool"`` task is cancelled (direct
    or via parent-cancel propagation). The SDK must surface this
    as a typed :class:`ClientTaskCancel` so consumers can cancel
    their local asyncio task. If the parser silently drops the
    event, the client would keep running the cancelled tool body
    indefinitely and waste compute / hold resources.
    """
    frame = (
        b"event: response.client_task.cancel\n"
        b'data: {"task_id": "task_abc123", "type": "response.client_task.cancel"}\n'
        b"\n"
    )

    events = []
    async for event in parse_sse_stream(_bytes(frame)):
        events.append(event)

    # Exactly one event — the parser must not produce duplicates.
    assert len(events) == 1, (
        f"Expected exactly one ClientTaskCancel event from the SSE "
        f"frame; got {len(events)}: {events!r}"
    )
    assert isinstance(events[0], ClientTaskCancel), (
        f"Expected ClientTaskCancel, got {type(events[0]).__name__}"
    )
    assert events[0].task_id == "task_abc123"


@pytest.mark.asyncio
async def test_sse_parser_drops_client_task_cancel_without_task_id() -> None:
    """
    A malformed ``response.client_task.cancel`` (no ``task_id``
    or empty string) must be dropped — emitting a
    :class:`ClientTaskCancel` with an empty ``task_id`` would
    cause the consumer to no-op silently or, worse, cancel the
    wrong local task if the consumer falls back on positional
    matching. Catches a server-side regression where the cancel
    payload loses its task_id.
    """
    frame = (
        b"event: response.client_task.cancel\n"
        b'data: {"type": "response.client_task.cancel"}\n'  # no task_id
        b"\n"
    )

    events = []
    async for event in parse_sse_stream(_bytes(frame)):
        events.append(event)

    assert events == [], f"Malformed cancel frame must be dropped; got {events!r}"


# ── Sanity: existing event shapes still parse ──────────────


@pytest.mark.asyncio
async def test_sse_parser_unchanged_for_function_call_output() -> None:
    """
    The async-dispatch protocol piggybacks on the existing
    ``response.output_item.done`` event for ``function_call``
    and ``function_call_output`` items — the handle JSON
    arrives as the ``output`` field on a normal FCO. This test
    proves Phase 5's parser additions did not regress that
    path: a typical async-dispatch sequence (function_call →
    function_call_output with handle JSON) must still produce
    :class:`ToolCall` + :class:`ToolResult` events.
    """
    frames = (
        # function_call (the LLM's call to the async tool)
        (
            b"event: response.output_item.done\n"
            b'data: {"item": {"type": "function_call", '
            b'"name": "_async_long_compute", '
            b'"arguments": "{\\"n\\": 5}", '
            b'"call_id": "call_abc", '
            b'"status": "completed", '
            b'"model": "test-agent"}}\n'
            b"\n"
        ),
        # function_call_output (the handle JSON the server emits inline)
        (
            b"event: response.output_item.done\n"
            b'data: {"item": {"type": "function_call_output", '
            b'"call_id": "call_abc", '
            b'"output": "{\\"task_id\\": \\"task_xyz\\", '
            b'\\"kind\\": \\"client_tool\\"}"}}\n'
            b"\n"
        ),
    )

    events = []
    async for event in parse_sse_stream(_bytes(*frames)):
        events.append(event)

    assert len(events) == 2, f"Expected ToolCall + ToolResult; got {len(events)}: {events!r}"
    call_event = events[0]
    result_event = events[1]
    assert isinstance(call_event, ToolCall)
    assert call_event.name == "_async_long_compute"
    assert call_event.call_id == "call_abc"
    assert call_event.arguments == {"n": 5}

    assert isinstance(result_event, ToolResult)
    assert result_event.call_id == "call_abc"
    # The handle JSON arrives verbatim as the FCO output —
    # downstream consumers parse it to extract task_id.
    assert "task_xyz" in result_event.output
    assert "client_tool" in result_event.output
