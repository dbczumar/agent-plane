"""
Phase 5 part D — minimal SDK changes for async client tools.

Covers two narrow surface areas that lift the SDK to recognize
the Phase 5 protocol:

1. ``build_tool_handler`` injects a per-call ``synchronous``
   boolean into ``parameters.properties`` for
   ``@tool(synchronous=False)``-decorated functions. Spec-
   compliant — ``properties`` is exactly where the OpenAI tool
   schema puts argument schemas, so the LLM sees the choice as
   a normal optional argument and the server reads
   ``arguments.synchronous`` per-call to route between sync
   parking and async dispatch.
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


@tool(synchronous=False)
async def _async_with_collision(synchronous: bool, payload: str) -> str:
    """Bad tool — its real ``synchronous`` arg collides with the routing hint.

    Used by the collision-rejection test below; the SDK must
    refuse to build a handler for this since injecting our
    routing hint would shadow the author's argument.

    Args:
        synchronous: Real arg whose name collides with the
            SDK's per-call async-dispatch hint.
        payload: Real payload string.
    """
    _ = synchronous  # avoid unused-arg lint; real test uses the metadata
    return payload


def _properties_of(schema: dict[str, object]) -> dict[str, object]:
    """
    Extract ``parameters.properties`` from a tool schema.

    Centralized so tests fail with a clear error if the schema
    shape ever changes.

    :param schema: One entry from
        :attr:`ToolHandler.schemas` — an OpenAI function tool
        wrapper.
    :returns: The properties dict (may be empty).
    """
    fn = schema["function"]
    assert isinstance(fn, dict)
    parameters = fn["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters.get("properties", {})
    assert isinstance(properties, dict)
    return properties


def test_build_tool_handler_omits_synchronous_for_sync_tools() -> None:
    """
    Default-synchronous tools must NOT inject a ``synchronous``
    property into the parameters schema — the LLM has nothing
    to set, and the server's per-call check
    (:func:`_wants_async_dispatch`) returns ``False`` so the
    tool keeps the legacy sync parking path.

    Catches a regression where the SDK injects the property
    universally and accidentally surfaces async dispatch on
    every tool.
    """
    handler = build_tool_handler([_sync_echo])
    assert len(handler.schemas) == 1
    schema = handler.schemas[0]
    # No top-level synchronous field — the v1 wire-shape extension
    # is gone entirely under v2.
    assert "synchronous" not in schema, (
        f"Sync tools must not emit a top-level 'synchronous' "
        f"field (the v1 extension is removed); got schema={schema!r}"
    )
    # And no synchronous *property* either — sync tools opt out
    # of even surfacing the choice to the LLM.
    properties = _properties_of(schema)
    assert "synchronous" not in properties, (
        f"Sync tools must not inject 'synchronous' into "
        f"parameters.properties; got properties={properties!r}"
    )
    # Sanity: function name and structure unchanged.
    assert schema["type"] == "function"
    fn_field = schema["function"]
    assert isinstance(fn_field, dict)
    assert fn_field["name"] == "_sync_echo"


def test_build_tool_handler_injects_synchronous_property_for_async_tools() -> None:
    """
    ``@tool(synchronous=False)`` must inject a ``synchronous``
    boolean into ``parameters.properties`` so the LLM sees the
    per-call async-dispatch choice as a real argument. The
    server's :func:`_wants_async_dispatch` then routes calls
    where ``arguments.synchronous == False`` through the
    async-dispatch path; calls that omit or set ``True`` go to
    the sync (parking) path.

    If the property is missing, the LLM has no way to express
    async intent and every call would fall back to sync — the
    tool author's ``synchronous=False`` declaration would be
    silently void.
    """
    handler = build_tool_handler([_async_long_compute])
    assert len(handler.schemas) == 1
    schema = handler.schemas[0]
    # Top-level extension is gone in v2.
    assert "synchronous" not in schema, (
        f"v1's top-level 'synchronous' wire extension must NOT appear in v2; got schema={schema!r}"
    )
    properties = _properties_of(schema)
    assert "synchronous" in properties, (
        f"@tool(synchronous=False) must inject 'synchronous' "
        f"into parameters.properties; got properties={properties!r}"
    )
    sync_prop = properties["synchronous"]
    assert isinstance(sync_prop, dict)
    assert sync_prop["type"] == "boolean", (
        f"Injected 'synchronous' must be a JSON Schema boolean; got {sync_prop!r}"
    )
    assert "description" in sync_prop and sync_prop["description"], (
        f"Injected 'synchronous' must carry a description so the "
        f"LLM understands when to set it; got {sync_prop!r}"
    )
    # The author's real arg is preserved alongside the injected one.
    assert "n" in properties, (
        f"Original tool args must survive injection; got properties={properties!r}"
    )


def test_build_tool_handler_mixes_sync_and_async() -> None:
    """
    A mixed handler must keep each tool's schema independent —
    injecting ``synchronous`` into one tool's properties must
    not leak into the other's. Catches a regression where the
    helper mutates shared metadata in place.
    """
    handler = build_tool_handler([_sync_echo, _async_long_compute])
    by_name = {
        s["function"]["name"]: s  # type: ignore[index]
        for s in handler.schemas
    }
    sync_props = _properties_of(by_name["_sync_echo"])
    async_props = _properties_of(by_name["_async_long_compute"])
    assert "synchronous" not in sync_props, (
        "_sync_echo must remain free of the synchronous property."
    )
    assert "synchronous" in async_props, "_async_long_compute must keep its synchronous property."


def test_build_tool_handler_does_not_mutate_metadata_schema() -> None:
    """
    The injection helper must deep-copy the metadata's
    ``json_schema`` — building the same handler twice (or
    sharing a metadata object across handlers) must not produce
    nested ``synchronous`` keys or otherwise corrupt the cached
    schema. Catches a copy-by-reference regression.
    """
    from agent_plane_client.tools._decorator import (
        TOOL_MARKER_ATTR,
        ToolMetadata,
    )

    meta_before: ToolMetadata = getattr(_async_long_compute, TOOL_MARKER_ATTR)
    original_properties = dict(meta_before.json_schema.get("properties", {}))
    assert "synchronous" not in original_properties, (
        "Pre-test sanity: metadata's pristine schema must not carry the injected property."
    )

    build_tool_handler([_async_long_compute])
    build_tool_handler([_async_long_compute])

    meta_after: ToolMetadata = getattr(_async_long_compute, TOOL_MARKER_ATTR)
    after_properties = meta_after.json_schema.get("properties", {})
    assert "synchronous" not in after_properties, (
        f"Building the handler must not mutate the metadata's "
        f"json_schema. After two builds, properties={after_properties!r}"
    )


def test_build_tool_handler_rejects_synchronous_param_collision() -> None:
    """
    If a ``@tool(synchronous=False)`` function declares a real
    ``synchronous`` parameter on its signature, injecting our
    routing-hint property would silently shadow the author's
    argument — and the server's per-call check would conflate
    the LLM's intent with whatever value the author meant.
    The SDK must refuse to build the handler.
    """
    with pytest.raises(ValueError, match="cannot be combined"):
        build_tool_handler([_async_with_collision])


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
