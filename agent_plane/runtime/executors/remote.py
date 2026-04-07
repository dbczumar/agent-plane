"""RemoteExecutor: delegate to a remote agent service over HTTP.

The remote service manages its own agent loop, tools, prompt,
and session state. Agent-plane sends messages, observes the SSE
event stream, and persists events for durability and relay.

Communicates via the ``POST /v1/turns`` REST protocol defined in
``designs/EXECUTOR_CONTRACT_FINAL.md``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from typing_extensions import Self

from agent_plane.runtime.executors.base import (
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    ReasoningChunk,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    TurnComplete,
)
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig

_logger = logging.getLogger(__name__)


class RemoteExecutor(Executor):
    """
    Executor that delegates to a remote agent service over HTTP.

    The remote service manages its own agent loop, tools, prompt,
    and session state. Agent-plane sends messages, observes the SSE
    event stream, and persists events for durability and relay.

    Communicates via the ``POST /v1/turns`` REST protocol defined in
    ``designs/EXECUTOR_CONTRACT_FINAL.md``.

    :param endpoint: URL of the remote turn endpoint, e.g.
        ``"http://localhost:8000/v1/turns"``.
    :param request_timeout: Per-HTTP-call timeout in seconds,
        e.g. ``300``.
    """

    def __init__(
        self,
        endpoint: str,
        request_timeout: int = 300,
    ) -> None:
        self._endpoint = endpoint
        self._request_timeout = request_timeout

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from the agent spec's executor config.

        :param spec: Agent spec with ``executor.endpoint`` set.
        :returns: Configured RemoteExecutor.
        """
        assert spec.executor.endpoint is not None
        return cls(
            endpoint=spec.executor.endpoint,
            request_timeout=spec.executor.request_timeout or 300,
        )

    def max_context_tokens(self) -> int | None:
        """
        Remote service manages its own context window.

        :returns: None — workflow skips compaction and @step.
        """
        return None

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        POST to the remote service and consume the SSE stream.

        On 404 (session not found), retries once with full
        conversation history so the remote can rebuild its session.

        Uses httpx streaming so events are yielded in real-time
        as the remote service produces them.

        :param messages: Conversation history as input items.
        :param tools: Ignored — remote defines its own tools.
        :param system_prompt: Ignored — remote defines its prompt.
        :param llm_config: Ignored — remote defines its config.
        :param context: Agent-plane capabilities and identifiers.
        """
        import httpx

        new_messages = _extract_new_messages(messages)
        body: dict[str, Any] = {
            "conversation_id": context.conversation_id,
            "new_messages": new_messages,
        }
        headers = {"Accept": "text/event-stream"}
        timeout = httpx.Timeout(
            connect=30.0,
            read=float(self._request_timeout),
            write=30.0,
            pool=30.0,
        )

        with httpx.Client(timeout=timeout) as client:
            try:
                # First attempt — normal turn.
                with client.stream(
                    "POST",
                    self._endpoint,
                    json=body,
                    headers=headers,
                ) as response:
                    if response.status_code == 404:
                        # Consume and discard the 404 body so the
                        # connection is released for the retry.
                        response.read()
                    elif response.status_code != 200:
                        yield ExecutorError(
                            message=(f"Remote executor returned {response.status_code}"),
                            code="remote_error",
                        )
                        return
                    else:
                        yield from _consume_remote_sse_stream(
                            response,
                        )
                        return
            except Exception as exc:
                yield ExecutorError(
                    message=(f"Cannot connect to remote executor at {self._endpoint}: {exc}"),
                    code="connection_error",
                )
                return

            # 404 recovery — resend with full history.
            body["history"] = _messages_to_history(messages)
            try:
                with client.stream(
                    "POST",
                    self._endpoint,
                    json=body,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        yield ExecutorError(
                            message=(f"Remote executor returned {response.status_code}"),
                            code="remote_error",
                        )
                        return
                    yield from _consume_remote_sse_stream(
                        response,
                    )
            except Exception as exc:
                yield ExecutorError(
                    message=(f"Cannot connect to remote executor at {self._endpoint}: {exc}"),
                    code="connection_error",
                )
                return


def _extract_new_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract the most recent user message(s) for a remote turn.

    Takes the trailing sequence of non-assistant messages (user +
    tool results) from the conversation history. These are the
    "new" messages the remote hasn't seen yet.

    :param messages: Full Responses API input items.
    :returns: The trailing new messages.
    """
    # Walk backwards to find the last assistant message boundary.
    new: list[dict[str, Any]] = []
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "assistant":
            break
        new.append(msg)
    new.reverse()
    return new if new else messages[-1:]


def _messages_to_history(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert Responses API input items to the simplified history
    format for session recovery.

    :param messages: Full Responses API input items.
    :returns: Simplified history with role/content/tool_calls.
    """
    # For now, pass through as-is. The remote service is
    # responsible for interpreting the format.
    return list(messages)


def _consume_remote_sse_stream(
    response: Any,
) -> Iterator[ExecutorEvent]:
    """
    Parse SSE data lines from a streaming httpx response.

    Uses ``iter_lines()`` on a streaming response so events
    are yielded in real-time as the remote produces them.
    Heartbeat events are consumed silently (keepalive only).
    ``turn_complete`` and ``error`` are terminal.

    :param response: An httpx streaming response (from
        ``client.stream()`` context manager).
    """
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        evt_type = payload.get("type", "")

        if evt_type == "text_chunk":
            yield TextChunk(text=payload["text"])

        elif evt_type == "reasoning_chunk":
            yield ReasoningChunk(
                delta=payload.get("delta", ""),
                event_type=payload.get("event_type", "reasoning_text"),
            )

        elif evt_type == "tool_call_requested":
            yield ToolCallRequested(
                call_id=payload["call_id"],
                name=payload["name"],
                arguments=payload["arguments"],
            )

        elif evt_type == "tool_call_observed":
            yield ToolCallObserved(
                call_id=payload["call_id"],
                name=payload["name"],
                arguments=payload["arguments"],
                result=payload["result"],
                status=payload["status"],
                duration_ms=payload["duration_ms"],
            )

        elif evt_type == "turn_complete":
            yield TurnComplete(text=payload.get("text"))
            return

        elif evt_type == "heartbeat":
            continue

        elif evt_type == "error":
            yield ExecutorError(
                message=payload["message"],
                code=payload.get("code"),
            )
            return
