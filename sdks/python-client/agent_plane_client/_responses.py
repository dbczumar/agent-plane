"""Responses namespace — streaming, tool loop, polling, cancel, steer."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import traceback
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._errors import ToolCallDenied, raise_for_status
from ._events import (
    ClientTaskCancel,
    CompactionInProgress,
    ErrorEvent,
    MessageDone,
    NativeToolCall,
    OutputFileDone,
    ReasoningDelta,
    ReasoningStarted,
    ReasoningSummaryDelta,
    ResponseCancelled,
    ResponseCompleted,
    ResponseCreated,
    ResponseFailed,
    ResponseIncomplete,
    RetryEvent,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolResult,
)
from ._sse import parse_sse_stream
from ._tool_handler import (
    CompactionStartCtx,
    FileOutputCtx,
    MessageEndCtx,
    MessageStartCtx,
    NativeToolCallCtx,
    ReasoningEndCtx,
    ResponseEndCtx,
    ResponseStartCtx,
    RetryCtx,
    ServerErrorCtx,
    StreamHooks,
    ToolCallEndCtx,
    ToolCallInfo,
    ToolCallStartCtx,
    ToolHandler,
    ToolResultInfo,
    ToolResultsReadyCtx,
)
from ._tool_handler import (
    ReasoningStartCtx as ReasoningStartHookCtx,
)
from ._types import Response

_log = logging.getLogger("agent_plane_client.responses")

# Terminal statuses — the response won't change further.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "incomplete", "cancelled"})

# D6 — async client-tool dispatch.
#
# Server-side hard cap on a single client_tool task's lifetime
# (matches background_tool_workflow._CLIENT_TOOL_MAX_LIFETIME_S).
# After this elapses the holder workflow times out, sends a
# ``failed`` drain payload, and any later PATCH from the client
# is dropped. We mirror the cap on the SDK side so a tool body
# that's stuck (deadlock, lost upstream, etc.) doesn't pin the
# asyncio task forever — TimeoutError surfaces as a
# ``status="failed"`` PATCH that races the server's own timeout
# (whichever lands first wins under G3 first-write-wins).
_ASYNC_CLIENT_TOOL_MAX_LIFETIME_S = 3600.0

# Maximum traceback-line budget for a failure PATCH's
# ``error.traceback`` field. Mirrors the server's
# ``truncate_traceback`` budget so a single failure can't blow
# the parent's LLM context. Lines beyond this are summarized
# with a ``[... N more lines truncated]`` marker.
_FAILURE_TRACEBACK_LINE_BUDGET = 30


@dataclass
class _AsyncToolState:
    """
    Per-call dispatch state for a ``synchronous: false`` client
    tool the SDK is running locally.

    The SDK creates one of these the moment it sees a
    :class:`ToolCall` for an async tool, spawns the tool body
    on an :class:`asyncio.Task`, and parks the task on the
    ``task_id_event`` until the matching handle FCO arrives in
    the same SSE stream (or a later one). Once the body
    completes, it PATCHes ``async_tool_results`` for ``task_id``;
    if a :class:`ClientTaskCancel` SSE event arrives first, the
    asyncio task is cancelled and a ``status="cancelled"`` PATCH
    is attempted before the task unwinds.

    :param call_id: The LLM-assigned ``call_id`` from the
        ``function_call`` event. Used to match this state to
        the later ``function_call_output`` (handle FCO) event
        that carries the server-issued ``task_id``.
    :param task_id: Populated from the handle FCO's ``output``
        JSON (``{"task_id": "...", "kind": "client_tool", ...}``).
        ``None`` until the handle arrives — the body waits on
        ``task_id_event`` before PATCHing.
    :param task_id_event: Set when ``task_id`` becomes available.
        The body's ``await`` on this event is the synchronization
        point that prevents PATCHing before the server has even
        created the task row.
    :param asyncio_task: The :class:`asyncio.Task` running the
        tool body. ``None`` only during the brief window between
        state construction and ``asyncio.create_task`` (kept
        ``None``-able so a future refactor can construct state
        before scheduling).
    """

    call_id: str
    task_id: str | None = None
    task_id_event: asyncio.Event = field(default_factory=asyncio.Event)
    asyncio_task: asyncio.Task[None] | None = None


async def _call_hook(hook: Any, ctx: Any) -> Any:
    """Call a hook (sync or async) and return its result."""
    if hook is None:
        return None
    result = hook(ctx)
    if inspect.isawaitable(result):
        return await result
    return result


class ResponsesNamespace:
    """Methods for ``/v1/responses`` endpoints."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, object]],
        background: bool = False,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, object]] | None = None,
        reasoning: dict[str, str] | None = None,
    ) -> Response:
        """Create a response (blocking, non-streaming).

        :param model: Agent name.
        :param input: User text or content block list.
        :param background: If True, returns immediately (poll via get()).
        :param instructions: Per-request system instructions.
        :param previous_response_id: Prior response for multi-turn.
        :param tools: Client-specified tool schemas.
        :param reasoning: Reasoning config, e.g. ``{"effort": "high"}``.
        :returns: The response object.
        """
        body = _build_body(
            model=model,
            input=input,
            stream=False,
            background=background,
            instructions=instructions,
            previous_response_id=previous_response_id,
            tools=tools,
            reasoning=reasoning,
        )
        resp = await self._http.post(f"{self._base}/v1/responses", json=body)
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Response.from_dict(resp.json())

    async def stream(
        self,
        *,
        model: str,
        input: str | list[dict[str, object]],
        background: bool = False,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tool_handler: ToolHandler | None = None,
        hooks: StreamHooks | None = None,
        reasoning: dict[str, str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response, optionally running the tool loop.

        If ``tool_handler`` is provided, the client runs the full tool
        execution loop: stream -> detect tool calls -> execute via handler
        -> POST results -> stream again. The consumer sees one continuous
        sequence of events.

        If ``tool_handler`` is None, yields raw events for a single server
        response.

        :param model: Agent name.
        :param input: User text or content block list.
        :param tool_handler: Optional client-side tool execution config.
        :param hooks: Optional lifecycle hooks.
        """
        if hooks is None:
            hooks = StreamHooks()

        current_input: str | list[dict[str, object]] = input
        current_prev_id = previous_response_id
        iteration = 0
        # D6: per-call_id state for ``synchronous: false`` client
        # tools the SDK is dispatching locally. Persists across
        # the outer ``while True`` so a tool dispatched in one
        # iteration's stream can deliver via PATCH before — or
        # well after — the next iteration's stream opens.
        async_tool_state: dict[str, _AsyncToolState] = {}

        while True:
            tools = tool_handler.schemas if tool_handler is not None else None
            pending_client_calls: list[ToolCall] = []
            completed_call_ids: set[str] = set()
            current_response_id: str | None = None
            in_reasoning = False
            reasoning_text = ""
            summary_text = ""
            message_started = False

            body = _build_body(
                model=model,
                input=current_input,
                stream=True,
                background=background,
                instructions=instructions,
                previous_response_id=current_prev_id,
                tools=tools,
                reasoning=reasoning,
            )

            async with self._http.stream("POST", f"{self._base}/v1/responses", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise_for_status(
                        resp.status_code, resp.json() if resp.status_code < 500 else resp.text
                    )

                async for event in parse_sse_stream(resp.aiter_bytes()):
                    # ── Fire hooks and collect state ──────────

                    if isinstance(event, ResponseCreated):
                        current_response_id = event.response.id
                        await _call_hook(
                            hooks.on_response_start, ResponseStartCtx(response=event.response)
                        )

                    elif isinstance(event, ReasoningStarted):
                        in_reasoning = True
                        reasoning_text = ""
                        summary_text = ""
                        await _call_hook(hooks.on_reasoning_start, ReasoningStartHookCtx())

                    elif isinstance(event, ReasoningDelta):
                        reasoning_text += event.delta

                    elif isinstance(event, ReasoningSummaryDelta):
                        summary_text += event.delta

                    elif isinstance(event, TextDelta):
                        if in_reasoning:
                            in_reasoning = False
                            await _call_hook(
                                hooks.on_reasoning_end,
                                ReasoningEndCtx(
                                    reasoning_text=reasoning_text, summary_text=summary_text
                                ),
                            )
                        if not message_started:
                            message_started = True
                            await _call_hook(
                                hooks.on_message_start,
                                MessageStartCtx(response_id=current_response_id or ""),
                            )

                    elif isinstance(event, CompactionInProgress):
                        await _call_hook(hooks.on_compaction_start, CompactionStartCtx())

                    elif isinstance(event, ToolCall):
                        is_client_side = (
                            tool_handler is not None and event.call_id not in completed_call_ids
                        )
                        executed_by = "client" if is_client_side else "server"
                        await _call_hook(
                            hooks.on_tool_call_start,
                            ToolCallStartCtx(
                                name=event.name,
                                arguments=event.arguments,
                                call_id=event.call_id,
                                agent_name=event.agent_name,
                                executed_by=executed_by,
                            ),
                        )
                        # Tunneled sub-agent calls: execute immediately
                        # and PATCH back while the stream continues.
                        if event.status == "action_required" and tool_handler is not None:
                            completed_call_ids.add(event.call_id)
                            asyncio.ensure_future(
                                _execute_and_patch(
                                    self._http,
                                    self._base,
                                    tool_handler,
                                    hooks,
                                    event,
                                    current_response_id or "",
                                    iteration,
                                )
                            )
                        elif is_client_side:
                            # D6: detect ``synchronous: false`` tools
                            # and dispatch them locally as
                            # asyncio.Tasks. The matching FCO with
                            # the handle's task_id will arrive in
                            # this same stream and unblock the
                            # task's PATCH.
                            if tool_handler is not None and _is_async_tool_call(
                                event, tool_handler
                            ):
                                completed_call_ids.add(event.call_id)
                                state = _AsyncToolState(call_id=event.call_id)
                                async_tool_state[event.call_id] = state
                                state.asyncio_task = asyncio.create_task(
                                    _run_async_tool_body(
                                        self._http,
                                        self._base,
                                        tool_handler,
                                        hooks,
                                        state,
                                        event,
                                        current_response_id or "",
                                        iteration,
                                    )
                                )
                            else:
                                pending_client_calls.append(event)

                    elif isinstance(event, ToolResult):
                        completed_call_ids.add(event.call_id)
                        # D6: if this FCO is the handle for an
                        # async dispatch we just spawned, capture
                        # the task_id so the body's PATCH can
                        # address it. The body is parked on
                        # state.task_id_event; setting the event
                        # unblocks it.
                        state = async_tool_state.get(event.call_id)
                        if state is not None and state.task_id is None:
                            task_id = _parse_handle_task_id(event.output)
                            if task_id is not None:
                                state.task_id = task_id
                                state.task_id_event.set()
                        await _call_hook(
                            hooks.on_tool_call_end,
                            ToolCallEndCtx(
                                name="",
                                call_id=event.call_id,
                                agent_name="",
                                output=event.output,
                            ),
                        )

                    elif isinstance(event, ClientTaskCancel):
                        # D6: server is telling us to stop the
                        # local body. Look up by task_id and cancel
                        # the asyncio.Task — the body's
                        # ``except CancelledError`` will PATCH
                        # ``status="cancelled"`` and re-raise.
                        for s in async_tool_state.values():
                            if s.task_id == event.task_id and s.asyncio_task is not None:
                                s.asyncio_task.cancel()
                                break

                    elif isinstance(event, NativeToolCall):
                        await _call_hook(
                            hooks.on_native_tool_call,
                            NativeToolCallCtx(tool_type=event.tool_type, data=event.data),
                        )

                    elif isinstance(event, MessageDone):
                        if in_reasoning:
                            in_reasoning = False
                            await _call_hook(
                                hooks.on_reasoning_end,
                                ReasoningEndCtx(
                                    reasoning_text=reasoning_text, summary_text=summary_text
                                ),
                            )
                        await _call_hook(
                            hooks.on_message_end,
                            MessageEndCtx(content=event.content),
                        )
                        message_started = False

                    elif isinstance(event, OutputFileDone):
                        await _call_hook(
                            hooks.on_file_output,
                            FileOutputCtx(
                                file_id=event.file_id,
                                filename=event.filename,
                                content_type=event.content_type,
                            ),
                        )

                    elif isinstance(event, RetryEvent):
                        await _call_hook(
                            hooks.on_retry,
                            RetryCtx(
                                source=event.source,
                                tool_name=event.tool_name,
                                attempt=event.attempt,
                                max_attempts=event.max_attempts,
                                delay_seconds=event.delay_seconds,
                                error=event.error,
                            ),
                        )

                    elif isinstance(event, ErrorEvent):
                        await _call_hook(
                            hooks.on_server_error,
                            ServerErrorCtx(
                                source=event.source,
                                tool_name=event.tool_name,
                                error=event.error,
                            ),
                        )

                    elif isinstance(
                        event,
                        ResponseCompleted
                        | ResponseFailed
                        | ResponseIncomplete
                        | ResponseCancelled,
                    ):
                        if in_reasoning:
                            in_reasoning = False
                            await _call_hook(
                                hooks.on_reasoning_end,
                                ReasoningEndCtx(
                                    reasoning_text=reasoning_text, summary_text=summary_text
                                ),
                            )
                        status = event.response.status
                        await _call_hook(
                            hooks.on_response_end,
                            ResponseEndCtx(response=event.response, status=status),
                        )

                    # ── Yield event to consumer ──────────────
                    yield event

            # ── Post-stream: tool loop ───────────────────
            if current_response_id is not None:
                current_prev_id = current_response_id

            # Filter out calls that already have server-side results.
            pending_client_calls = [
                tc for tc in pending_client_calls if tc.call_id not in completed_call_ids
            ]

            if not pending_client_calls or tool_handler is None:
                break

            # Execute client-side tools and build results.
            results: list[dict[str, object]] = []
            result_infos: list[ToolResultInfo] = []

            for tc in pending_client_calls:
                call_info = ToolCallInfo(
                    name=tc.name,
                    arguments=tc.arguments,
                    call_id=tc.call_id,
                    agent_name=tc.agent_name,
                    response_id=current_response_id or "",
                    iteration=iteration,
                )
                try:
                    output = tool_handler.execute(call_info)
                    if inspect.isawaitable(output):
                        output = await output
                except ToolCallDenied as exc:
                    output = str(exc)

                await _call_hook(
                    hooks.on_tool_call_end,
                    ToolCallEndCtx(
                        name=tc.name,
                        call_id=tc.call_id,
                        agent_name=tc.agent_name,
                        output=output,
                    ),
                )
                yield ToolResult(call_id=tc.call_id, output=output)

                results.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": output,
                    }
                )
                result_infos.append(
                    ToolResultInfo(
                        call_id=tc.call_id,
                        name=tc.name,
                        output=output,
                        agent_name=tc.agent_name,
                    )
                )

            await _call_hook(
                hooks.on_tool_results_ready,
                ToolResultsReadyCtx(results=result_infos, iteration=iteration),
            )

            current_input = results
            iteration += 1

    async def get(self, response_id: str) -> Response:
        """Get a response by ID (poll for status).

        :param response_id: The response/task ID.
        :returns: Current response state.
        """
        resp = await self._http.get(f"{self._base}/v1/responses/{response_id}")
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Response.from_dict(resp.json())

    async def poll(
        self,
        response_id: str,
        *,
        interval: float = 0.5,
        tool_handler: ToolHandler | None = None,
    ) -> Response:
        """Poll a background response until it reaches a terminal status.

        If ``tool_handler`` is provided, tunneled tool calls
        (``status: "action_required"``) are executed and PATCHed back.

        :param response_id: The response/task ID.
        :param interval: Seconds between polls.
        :param tool_handler: Optional client-side tool handler.
        :returns: The terminal response.
        """
        while True:
            response = await self.get(response_id)
            if response.status in _TERMINAL_STATUSES:
                return response
            # Check for tunneled tool calls needing client execution.
            if tool_handler is not None:
                await self._handle_polling_tool_calls(response_id, response, tool_handler)
            await asyncio.sleep(interval)

    async def _handle_polling_tool_calls(
        self,
        response_id: str,
        response: Response,
        tool_handler: ToolHandler,
    ) -> None:
        """Execute action_required tool calls found during polling."""
        action_required = [
            item
            for item in response.output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("status") == "action_required"
        ]
        if not action_required:
            return

        tool_results = []
        for fc in action_required:
            name = str(fc.get("name", ""))
            call_id = str(fc.get("call_id", ""))
            args_str = str(fc.get("arguments", "{}"))
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}

            call_info = ToolCallInfo(
                name=name,
                arguments=arguments,
                call_id=call_id,
                agent_name=str(fc.get("model", "")),
                response_id=response_id,
                iteration=0,
            )
            output = tool_handler.execute(call_info)
            if inspect.isawaitable(output):
                output = await output
            tool_results.append({"call_id": call_id, "output": output})

        if tool_results:
            await self._patch_tool_results(response_id, tool_results)

    async def _patch_tool_results(
        self,
        response_id: str,
        tool_results: list[dict[str, str]],
    ) -> None:
        """PATCH tool results back to the server."""
        resp = await self._http.patch(
            f"{self._base}/v1/responses/{response_id}",
            json={"tool_results": tool_results},
            timeout=60.0,
        )
        if resp.status_code not in (200, 404, 409):
            _log.warning(
                "PATCH tool results failed (%d): %s",
                resp.status_code,
                resp.text[:200],
            )

    async def steer(
        self,
        response_id: str,
        input: str,
        *,
        model: str,
    ) -> Response:
        """Send a steering message to an in-progress response.

        :param response_id: The in-progress response ID.
        :param input: Steering text.
        :param model: Agent name.
        :returns: The response (same ID if delivered, new ID if agent finished).
        """
        body: dict[str, object] = {
            "model": model,
            "input": input,
            "previous_response_id": response_id,
            "stream": False,
            "background": True,
        }
        resp = await self._http.post(
            f"{self._base}/v1/responses",
            json=body,
            timeout=120.0,
        )
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Response.from_dict(resp.json())

    async def cancel(self, response_id: str) -> Response:
        """Cancel an in-progress response.

        :param response_id: The response ID to cancel.
        :returns: The cancelled response.
        """
        resp = await self._http.post(
            f"{self._base}/v1/responses/{response_id}/cancel",
            timeout=10.0,
        )
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Response.from_dict(resp.json())

    async def delete(self, response_id: str) -> None:
        """Delete a response.

        :param response_id: The response ID to delete.
        """
        resp = await self._http.delete(
            f"{self._base}/v1/responses/{response_id}",
        )
        if resp.status_code >= 400:
            data = resp.json() if resp.status_code < 500 else resp.text
            raise_for_status(resp.status_code, data)


# ── Helpers ──────────────────────────────────────────────


def _build_body(
    *,
    model: str,
    input: str | list[dict[str, object]],
    stream: bool,
    background: bool,
    instructions: str | None,
    previous_response_id: str | None,
    tools: list[dict[str, object]] | None,
    reasoning: dict[str, str] | None,
) -> dict[str, object]:
    """Build the request body for POST /v1/responses."""
    body: dict[str, object] = {
        "model": model,
        "input": input,
        "stream": stream,
    }
    if background:
        body["background"] = True
    if instructions is not None:
        body["instructions"] = instructions
    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id
    if tools is not None:
        body["tools"] = tools
    if reasoning is not None:
        body["reasoning"] = reasoning
    return body


async def _execute_and_patch(
    http: httpx.AsyncClient,
    base_url: str,
    tool_handler: ToolHandler,
    hooks: StreamHooks,
    tool_call: ToolCall,
    root_response_id: str,
    iteration: int,
) -> None:
    """Execute a tunneled sub-agent tool call and PATCH the result back.

    Runs in the background via ``asyncio.ensure_future`` while the
    SSE stream continues.
    """
    call_info = ToolCallInfo(
        name=tool_call.name,
        arguments=tool_call.arguments,
        call_id=tool_call.call_id,
        agent_name=tool_call.agent_name,
        response_id=root_response_id,
        iteration=iteration,
    )
    try:
        output = tool_handler.execute(call_info)
        if inspect.isawaitable(output):
            output = await output
    except ToolCallDenied as exc:
        output = str(exc)
    except Exception:
        _log.exception("Error executing tunneled tool call %s", tool_call.name)
        output = f"Error executing tool: {tool_call.name}"

    await _call_hook(
        hooks.on_tool_call_end,
        ToolCallEndCtx(
            name=tool_call.name,
            call_id=tool_call.call_id,
            agent_name=tool_call.agent_name,
            output=output,
        ),
    )

    try:
        resp = await http.patch(
            f"{base_url}/v1/responses/{root_response_id}",
            json={"tool_results": [{"call_id": tool_call.call_id, "output": output}]},
            timeout=60.0,
        )
        if resp.status_code not in (200, 404, 409):
            _log.warning(
                "PATCH failed for call_id %s: %s",
                tool_call.call_id,
                resp.text[:200],
            )
    except Exception:
        _log.exception("Error PATCHing tool result for call_id %s", tool_call.call_id)


# ── D6: async client-tool dispatch ────────────────────────────


def _is_async_tool_call(
    tool_call: ToolCall,
    tool_handler: ToolHandler,
) -> bool:
    """
    Return ``True`` iff this ``function_call`` is for a tool the
    handler exposed with ``synchronous: false`` (i.e. with a
    ``synchronous`` boolean inside ``parameters.properties``).

    The check is structural — same gate the server uses
    (``_wants_async_dispatch`` requires the schema to declare
    the property). If the SDK builds a handler without the
    property and the LLM hallucinates ``synchronous: false``
    in args, the server still routes sync; we mirror that
    decision here so the SDK doesn't double-track the call.

    :param tool_call: The :class:`ToolCall` event from the SSE
        stream.
    :param tool_handler: The :class:`ToolHandler` whose schemas
        were sent on the request. ``None`` is impossible at the
        call sites (gated above).
    :returns: ``True`` iff the matching schema's
        ``parameters.properties`` declares ``synchronous``.
        Any structural deviation falls through as ``False``.
    """
    for schema in tool_handler.schemas:
        fn = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != tool_call.name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            return False
        properties = params.get("properties")
        if not isinstance(properties, dict):
            return False
        return "synchronous" in properties
    return False


def _parse_handle_task_id(output: str) -> str | None:
    """
    Parse a handle FCO's ``output`` field and extract its task_id.

    The server emits async-tool dispatch handles as a
    ``function_call_output`` whose ``output`` is JSON of the
    form ``{"task_id": "...", "kind": "client_tool", ...}``.
    Returns the task_id only when both the JSON parses and the
    ``kind`` is ``"client_tool"`` — otherwise the FCO is for a
    different kind (sub_agent handle, sync result, etc.) and
    the SDK should not pair it with an async dispatch state.

    :param output: The ``output`` field of a
        :class:`ToolResult` event. Usually JSON but may be free
        text for non-async results.
    :returns: The handle's ``task_id`` when it's a valid
        ``client_tool`` handle; ``None`` otherwise.
    """
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("kind") != "client_tool":
        return None
    task_id = parsed.get("task_id")
    return task_id if isinstance(task_id, str) and task_id else None


def _truncate_traceback_lines(tb_text: str) -> str:
    """
    Cap a traceback at :data:`_FAILURE_TRACEBACK_LINE_BUDGET` lines.

    Mirrors the server's truncation budget so failure PATCHes
    don't push the LLM's context budget around. Keeps the head
    (exception type + deepest frames — the most diagnostic part)
    and appends a ``[... N more lines truncated]`` marker.

    :param tb_text: Raw multi-line traceback text from
        :func:`traceback.format_exc`.
    :returns: Either the original text (when within budget) or
        the head plus a truncation marker.
    """
    lines = tb_text.splitlines()
    if len(lines) <= _FAILURE_TRACEBACK_LINE_BUDGET:
        return tb_text
    head = "\n".join(lines[:_FAILURE_TRACEBACK_LINE_BUDGET])
    return f"{head}\n[... {len(lines) - _FAILURE_TRACEBACK_LINE_BUDGET} more lines truncated]"


async def _patch_async_tool_result(
    http: httpx.AsyncClient,
    base_url: str,
    root_response_id: str,
    task_id: str,
    status: str,
    output: str | None,
    error: dict[str, str] | None,
) -> None:
    """
    PATCH ``async_tool_results`` to the server.

    Idempotent on the server side — a second PATCH after the
    task is terminal hits the G3 first-write-wins no-op path.
    HTTP errors are logged but swallowed; losing the PATCH
    leaves the server's holder workflow waiting on its 1h
    ``DBOS.recv`` timeout, which still terminates the task.

    :param http: SDK HTTP client.
    :param base_url: Server base URL,
        e.g. ``"http://localhost:18501"``.
    :param root_response_id: The response id the SSE stream is
        open on. Server's PATCH route accepts the task_id in
        the body and looks up the right ``client_tool`` row;
        the URL's response_id only needs to be a valid response
        the caller has access to.
    :param task_id: The server-issued ``client_tool`` task id
        from the handle FCO.
    :param status: Terminal status — one of ``"completed"``,
        ``"failed"``, ``"cancelled"``.
    :param output: Tool output for ``"completed"``. ``None``
        otherwise.
    :param error: For ``"failed"`` only — a dict with
        ``message`` and (optionally) ``traceback`` keys.
    """
    body: dict[str, object] = {
        "task_id": task_id,
        "status": status,
    }
    if output is not None:
        body["output"] = output
    if error is not None:
        body["error"] = error
    try:
        resp = await http.patch(
            f"{base_url}/v1/responses/{root_response_id}",
            json={"async_tool_results": [body]},
            timeout=60.0,
        )
        if resp.status_code not in (200, 404, 409):
            _log.warning(
                "PATCH async_tool_results failed for task_id %s: %s",
                task_id,
                resp.text[:200],
            )
    except Exception:
        _log.exception(
            "Error PATCHing async_tool_results for task_id %s",
            task_id,
        )


async def _run_async_tool_body(
    http: httpx.AsyncClient,
    base_url: str,
    tool_handler: ToolHandler,
    hooks: StreamHooks,
    state: _AsyncToolState,
    tool_call: ToolCall,
    root_response_id: str,
    iteration: int,
) -> None:
    """
    Execute one async client tool's body and PATCH the result back.

    Lifecycle:

    1. Wait (up to 1 h) for the matching handle FCO to set
       ``state.task_id_event`` so we know what task to PATCH.
    2. Run the tool body via ``tool_handler.execute``, wrapped in
       :func:`asyncio.wait_for` with the lifetime cap so a
       deadlocked body can't pin the asyncio task forever.
    3. PATCH ``async_tool_results`` with the appropriate
       ``status`` (``completed`` / ``failed`` / ``cancelled``).
    4. Fire the ``on_tool_call_end`` hook so consumers see the
       outcome alongside the in-stream events.

    Cancellation is honored at every ``await`` point — when the
    SSE stream surfaces a :class:`ClientTaskCancel` for our
    ``task_id``, the outer code calls ``state.asyncio_task.cancel()``
    and the ``except asyncio.CancelledError`` block here PATCHes
    a ``cancelled`` status (best-effort) before re-raising.

    :param http: SDK HTTP client (shared with the stream).
    :param base_url: Server base URL.
    :param tool_handler: Handler that built the schemas
        (provides ``execute`` for the body).
    :param hooks: Stream hooks for ``on_tool_call_end``.
    :param state: Per-call state with the
        ``task_id_event`` synchronization gate.
    :param tool_call: The originating ``function_call`` event.
    :param root_response_id: The response id to PATCH against.
    :param iteration: The stream-loop iteration that emitted
        the call (passed through to hooks).
    """
    call_info = ToolCallInfo(
        name=tool_call.name,
        arguments=tool_call.arguments,
        call_id=tool_call.call_id,
        agent_name=tool_call.agent_name,
        response_id=root_response_id,
        iteration=iteration,
    )

    status: str
    output: str | None = None
    error: dict[str, str] | None = None

    try:
        # Wait for the handle FCO to populate state.task_id.
        # The server emits the handle synchronously after the
        # function_call, so this typically returns within
        # milliseconds — the timeout is only a backstop for a
        # broken stream that never delivers it.
        await asyncio.wait_for(
            state.task_id_event.wait(),
            timeout=_ASYNC_CLIENT_TOOL_MAX_LIFETIME_S,
        )

        # Now run the tool body. Cap at 1h so a hung body
        # surfaces as a clean ``failed`` PATCH rather than
        # leaking the asyncio task.
        body_coro = tool_handler.execute(call_info)
        if inspect.isawaitable(body_coro):
            raw_output = await asyncio.wait_for(
                body_coro, timeout=_ASYNC_CLIENT_TOOL_MAX_LIFETIME_S
            )
        else:
            raw_output = body_coro
        output = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, default=str)
        status = "completed"
    except asyncio.CancelledError:
        # Outer code (ClientTaskCancel handler) cancelled us.
        # Best-effort PATCH the cancelled status, then re-raise
        # so DBOS / asyncio see the cancellation.
        status = "cancelled"
        if state.task_id is not None:
            await _patch_async_tool_result(
                http,
                base_url,
                root_response_id,
                state.task_id,
                status=status,
                output=None,
                error=None,
            )
        raise
    except TimeoutError:
        status = "failed"
        error = {
            "message": (
                f"async client tool {tool_call.name!r} exceeded the "
                f"{_ASYNC_CLIENT_TOOL_MAX_LIFETIME_S}s SDK lifetime cap"
            ),
        }
    except ToolCallDenied as exc:
        status = "failed"
        error = {"message": str(exc)}
    except Exception as exc:
        _log.exception("async client tool %s raised", tool_call.name)
        status = "failed"
        error = {
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": _truncate_traceback_lines(traceback.format_exc()),
        }

    await _call_hook(
        hooks.on_tool_call_end,
        ToolCallEndCtx(
            name=tool_call.name,
            call_id=tool_call.call_id,
            agent_name=tool_call.agent_name,
            output=output if output is not None else (error or {}).get("message", ""),
        ),
    )

    # Final PATCH for non-cancellation paths (cancellation path
    # already PATCHed before re-raising). If state.task_id is
    # still ``None`` here the handle FCO never arrived and the
    # 1h wait timed out — surface as a failed PATCH would be
    # nice but we have nothing to address it to. Log and drop.
    if state.task_id is None:
        _log.warning(
            "async client tool %s never received a task_id from the "
            "handle FCO (timed out at %ss); dropping PATCH",
            tool_call.name,
            _ASYNC_CLIENT_TOOL_MAX_LIFETIME_S,
        )
        return
    await _patch_async_tool_result(
        http,
        base_url,
        root_response_id,
        state.task_id,
        status=status,
        output=output,
        error=error,
    )
