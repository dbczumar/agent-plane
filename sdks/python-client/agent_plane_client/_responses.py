"""Responses namespace — streaming, tool loop, polling, cancel, steer."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._errors import ToolCallDenied, raise_for_status
from ._events import (
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
                            pending_client_calls.append(event)

                    elif isinstance(event, ToolResult):
                        completed_call_ids.add(event.call_id)
                        await _call_hook(
                            hooks.on_tool_call_end,
                            ToolCallEndCtx(
                                name="",
                                call_id=event.call_id,
                                agent_name="",
                                output=event.output,
                            ),
                        )

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
