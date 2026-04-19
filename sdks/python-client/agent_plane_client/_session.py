"""Session helper — tracks conversation state for interactive use."""

from __future__ import annotations

import contextlib
import mimetypes
import pathlib
from collections.abc import AsyncIterator, Callable, Iterator

# Import at the type level to avoid circular imports at runtime.
from typing import TYPE_CHECKING, Any, Literal, overload

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
                event, ResponseCompleted | ResponseFailed | ResponseIncomplete | ResponseCancelled
            ):
                self._is_terminal = True
                self._previous_response_id = event.response.id
                self._current_response_id = event.response.id

            yield event

    @overload
    async def query(
        self,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None = ...,
        tools: list[Callable[..., Any]] | None = ...,
        stream: Literal[False] = ...,
    ) -> str: ...

    @overload
    async def query(
        self,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None = ...,
        tools: list[Callable[..., Any]] | None = ...,
        stream: Literal[True],
    ) -> AsyncIterator[str]: ...

    async def query(
        self,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None = None,
        tools: list[Callable[..., Any]] | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Send a prompt and get text back.

        Default returns the final assistant text as a string::

            text = await session.query("hello")

        With ``stream=True`` returns an async iterator over text
        chunks in order::

            it = await session.query("hello", stream=True)
            async for chunk in it:
                print(chunk, end="", flush=True)

        Client-side tools can be passed per-call via ``tools=``, or
        configured session-wide via ``client.session(tool_handler=...)``.
        If this turn's ``tools=`` is given, it OVERRIDES any session
        handler for this call only.

        :param input: User text or a list of content-block dicts,
            e.g. ``"hello"`` or
            ``[{"type": "input_text", "text": "hi"}]``.
        :param files: Optional list of local file paths to upload and
            attach to the turn, e.g. ``["./data.csv"]``.
        :param tools: Optional list of ``@tool``-decorated functions
            the agent may call on this turn. Overrides the session's
            configured ``tool_handler`` for this call only.
        :param stream: If True, return an ``AsyncIterator[str]`` that
            yields text chunks as they arrive. If False (default),
            return the final text after the response completes.
        :returns: The assistant's final text (``stream=False``) or an
            async iterator of text chunks (``stream=True``). Empty
            string / empty iterator if the agent produced no text.
        :raises AgentPlaneError: If the response ends in an error.
        """
        if stream:
            return self._stream_text(input, files=files, tools=tools)
        return await self._collect_text(input, files=files, tools=tools)

    async def _collect_text(
        self,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None,
        tools: list[Callable[..., Any]] | None,
    ) -> str:
        """Run a turn and return the final assistant text."""
        # Local imports avoid a circular dep at module load — _stream
        # imports _session under TYPE_CHECKING.
        from ._blocks import TextDone
        from ._stream import BlockStream
        from ._transforms import merge_text_across_iterations, pipe, skip_intermediate_ends

        with _per_call_tool_override(self, tools):
            block_stream = BlockStream()
            final_text = ""
            async for block in pipe(
                block_stream.stream(self, input, files=files),
                # Merges per-iteration text into one TextDone per response,
                # so we don't truncate to just the last iteration's text.
                merge_text_across_iterations(),
                skip_intermediate_ends(),
            ):
                if isinstance(block, TextDone):
                    final_text = block.full_text
            return final_text

    async def _stream_text(
        self,
        input: str | list[dict[str, object]],
        *,
        files: list[str] | None,
        tools: list[Callable[..., Any]] | None,
    ) -> AsyncIterator[str]:
        """Run a turn and yield text chunks as they arrive."""
        from ._blocks import TextChunk
        from ._stream import BlockStream
        from ._transforms import pipe, skip_intermediate_ends

        with _per_call_tool_override(self, tools):
            block_stream = BlockStream()
            async for block in pipe(
                block_stream.stream(self, input, files=files),
                skip_intermediate_ends(),
            ):
                if isinstance(block, TextChunk):
                    yield block.text

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


@contextlib.contextmanager
def _per_call_tool_override(
    session: Session,
    tools: list[Callable[..., Any]] | None,
) -> Iterator[None]:
    """Temporarily override ``session._tool_handler`` for one call.

    If ``tools`` is ``None``, the session's configured handler is
    used unchanged. Otherwise a handler is built from the decorated
    functions and swapped in for the duration of the ``with`` block;
    the original is restored on exit, even on exception.

    :param session: The session whose ``_tool_handler`` to override.
    :param tools: List of ``@tool``-decorated functions, or ``None``
        to leave the session's handler in place.
    """
    if tools is None:
        yield
        return
    from .tools import build_tool_handler

    previous = session._tool_handler
    session._tool_handler = build_tool_handler(tools)
    try:
        yield
    finally:
        session._tool_handler = previous
