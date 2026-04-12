"""Session helper — tracks conversation state for interactive use."""

from __future__ import annotations

import mimetypes
import pathlib
from collections.abc import AsyncIterator

# Import at the type level to avoid circular imports at runtime.
from typing import TYPE_CHECKING

from ._events import (
    ResponseCancelled,
    ResponseCompleted,
    ResponseCreated,
    ResponseFailed,
    ResponseIncomplete,
    StreamEvent,
)
from ._tool_handler import StreamHooks, ToolHandler
from ._types import Response

if TYPE_CHECKING:
    from ._client import AgentPlaneClient


_TERMINAL_STATUSES = frozenset({"completed", "failed", "incomplete", "cancelled"})


class Session:
    """Tracks conversation state for interactive use.

    Holds the model name, last response ID, and whether the current
    response is still running. ``send()`` automatically steers if
    a response is in progress, or starts a new turn if the response
    is terminal.

    :param client: The underlying :class:`AgentPlaneClient`.
    :param model: Agent name to use for requests.
    :param tool_handler: Optional client-side tool execution config.
    :param hooks: Optional lifecycle hooks.
    """

    def __init__(
        self,
        client: AgentPlaneClient,
        model: str,
        tool_handler: ToolHandler | None = None,
        hooks: StreamHooks | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._tool_handler = tool_handler
        self._hooks = hooks
        self._previous_response_id: str | None = None
        self._current_response_id: str | None = None
        self._is_terminal: bool = True

    @property
    def model(self) -> str:
        """The agent name for this session."""
        return self._model

    @property
    def current_response_id(self) -> str | None:
        """The most recent response ID, or None if no messages sent."""
        return self._current_response_id

    @property
    def is_streaming(self) -> bool:
        """True if a response is currently in progress."""
        return not self._is_terminal

    async def send(
        self,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a message — auto-steers if a response is in progress.

        Always returns an async iterator. The caller always does
        ``async for event in session.send(text): ...`` regardless
        of whether it steered or started a new turn.

        Three cases:

        1. **Response in progress, steer delivered**: The server
           accepted the message into the running agent's inbox.
           Yields nothing — the existing stream (from the original
           ``send()`` call) will surface the agent's reaction.

        2. **Response in progress, agent already finished**: The
           server created a new response instead of steering.
           Yields the full event stream for that new response.

        3. **No response in progress**: Starts a new turn. Yields
           the full event stream.

        :param input: User text or content block list.
        :param files: Optional file paths to upload and attach.
        :param instructions: Per-request system instructions.
        :yields: Stream events.
        """
        if files:
            input = await self._build_input_with_files(input, files)

        # Auto-steer if response is in progress.
        if not self._is_terminal and self._current_response_id is not None:
            steer_resp = await self._client.responses.steer(
                self._current_response_id,
                input if isinstance(input, str) else str(input),
                model=self._model,
            )
            if steer_resp.id == self._current_response_id:
                # Case 1: steering delivered. Nothing to yield.
                return
                yield  # noqa: RUF058 - makes this an async generator

            # Case 2: agent finished — server created a new response.
            # Stream it like a normal turn. The input was already
            # included in the steer POST, so the new response has it.
            async for event in self._stream_and_track(steer_resp.id, instructions):
                yield event
            return

        # Case 3: no response in progress — new turn.
        async for event in self._stream_and_track(
            None,
            instructions,
            input=input,
        ):
            yield event

    async def _stream_and_track(
        self,
        previous_response_id: str | None,
        instructions: str | None,
        *,
        input: str | list[dict[str, object]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response and track session state.

        If ``input`` is None, uses an empty string (for steer-fallback
        where the input was already sent via the steer POST). If
        ``previous_response_id`` is None, uses the session's stored ID.
        """
        self._is_terminal = False
        prev_id = (
            previous_response_id
            if previous_response_id is not None
            else self._previous_response_id
        )
        actual_input: str | list[dict[str, object]] = input if input is not None else ""

        async for event in self._client.responses.stream(
            model=self._model,
            input=actual_input,
            previous_response_id=prev_id,
            tool_handler=self._tool_handler,
            hooks=self._hooks,
            instructions=instructions,
        ):
            if isinstance(event, ResponseCreated):
                self._current_response_id = event.response.id

            if isinstance(
                event, (ResponseCompleted, ResponseFailed, ResponseIncomplete, ResponseCancelled)
            ):
                self._is_terminal = True
                self._previous_response_id = event.response.id
                self._current_response_id = event.response.id

            yield event

    async def cancel(self) -> Response | None:
        """Cancel the current in-progress response.

        :returns: The cancelled response, or None if no response is active.
        """
        if self._current_response_id is None:
            return None
        response = await self._client.responses.cancel(self._current_response_id)
        self._is_terminal = True
        self._previous_response_id = response.id
        return response

    def reset(self) -> None:
        """Reset the session — start a new conversation."""
        self._previous_response_id = None
        self._current_response_id = None
        self._is_terminal = True

    def resume_from_response(self, response_id: str) -> None:
        """Resume conversation from a specific response ID."""
        self._previous_response_id = response_id
        self._current_response_id = response_id
        self._is_terminal = True

    async def _build_input_with_files(
        self,
        text: str | list[dict[str, object]],
        file_paths: list[str],
    ) -> list[dict[str, object]]:
        """Upload files and build content blocks."""
        blocks: list[dict[str, object]] = []

        # Add text block.
        if isinstance(text, str) and text:
            blocks.append({"type": "input_text", "text": text})
        elif isinstance(text, list):
            blocks.extend(text)

        # Upload and add file blocks.
        for path in file_paths:
            uploaded = await self._client.files.upload(path)
            content_type = mimetypes.guess_type(path)[0]
            if content_type and content_type.startswith("image/"):
                blocks.append({"type": "input_image", "file_id": uploaded.id})
            else:
                blocks.append(
                    {
                        "type": "input_file",
                        "file_id": uploaded.id,
                        "filename": pathlib.Path(path).name,
                    }
                )

        return blocks
