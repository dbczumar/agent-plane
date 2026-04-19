"""AgentPlaneClient — the top-level client tying all namespaces together."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, overload

import httpx

from ._agents import AgentsNamespace
from ._conversations import ConversationsNamespace
from ._files import FilesNamespace
from ._responses import ResponsesNamespace
from ._session import Session
from ._tool_handler import StreamHooks, ToolHandler


class AgentPlaneClient:
    """Typed Python client for the agent-plane server API.

    One-shot text::

        async with AgentPlaneClient(base_url="http://localhost:8080") as client:
            text = await client.query(model="archer", input="hello")

    Streaming text::

        stream = await client.query(model="archer", input="hi", stream=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)

    Multi-turn conversation::

        session = client.session(model="archer")
        await session.query("hello")
        await session.query("what did I just say?")

    For access to raw events or semantic blocks (tool-call display,
    reasoning, lifecycle), drop to :attr:`responses` or
    :class:`BlockStream`.

    :param base_url: Server base URL, e.g. ``"http://localhost:8080"``.
    :param headers: Extra headers sent on every request (e.g. auth).
    :param timeout: Default timeout for HTTP requests in seconds.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Long read timeout for SSE streams (tool execution can
        # pause the stream for minutes).
        sse_timeout = httpx.Timeout(
            connect=30.0,
            read=600.0,
            write=30.0,
            pool=30.0,
        )
        self._http = httpx.AsyncClient(
            headers=headers or {},
            timeout=sse_timeout,
        )

        self.agents = AgentsNamespace(self._http, self._base_url)
        self.files = FilesNamespace(self._http, self._base_url)
        self.conversations = ConversationsNamespace(self._http, self._base_url)
        self.responses = ResponsesNamespace(self._http, self._base_url)

    def session(
        self,
        model: str,
        *,
        tool_handler: ToolHandler | None = None,
        hooks: StreamHooks | None = None,
    ) -> Session:
        """Create a conversation session.

        A session tracks ``previous_response_id`` automatically.
        ``send()`` auto-steers if a response is in progress, or
        starts a new turn if the response is terminal.

        :param model: Agent name.
        :param tool_handler: Optional client-side tool execution config.
        :param hooks: Optional lifecycle hooks.
        :returns: A new :class:`Session`.
        """
        return Session(
            client=self,
            model=model,
            tool_handler=tool_handler,
            hooks=hooks,
        )

    @overload
    async def query(
        self,
        *,
        model: str,
        input: str | list[dict[str, object]],
        tools: list[Callable[..., Any]] | None = ...,
        tool_handler: ToolHandler | None = ...,
        files: list[str] | None = ...,
        stream: Literal[False] = ...,
    ) -> str: ...

    @overload
    async def query(
        self,
        *,
        model: str,
        input: str | list[dict[str, object]],
        tools: list[Callable[..., Any]] | None = ...,
        tool_handler: ToolHandler | None = ...,
        files: list[str] | None = ...,
        stream: Literal[True],
    ) -> AsyncIterator[str]: ...

    async def query(
        self,
        *,
        model: str,
        input: str | list[dict[str, object]],
        tools: list[Callable[..., Any]] | None = None,
        tool_handler: ToolHandler | None = None,
        files: list[str] | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """One-shot invocation: send a prompt, get text back.

        Default returns the final text::

            text = await client.query(model="archer", input="hi")

        With ``stream=True`` returns an async iterator over text
        chunks::

            it = await client.query(model="archer", input="hi", stream=True)
            async for chunk in it:
                print(chunk, end="", flush=True)

        With client-side tools, pass ``@tool``-decorated functions::

            from agent_plane_client import tool

            @tool
            def get_time() -> str:
                '''Return the current time.'''
                return datetime.now().isoformat()

            text = await client.query(
                model="archer", input="what time?", tools=[get_time],
            )

        Creates a single-turn session internally. For multi-turn
        conversations, call :meth:`session` and use its ``query()``.

        :param model: Agent name, e.g. ``"archer"``.
        :param input: User text or a list of content-block dicts.
        :param tools: List of ``@tool``-decorated Python functions
            the agent may call. Mutually exclusive with ``tool_handler``.
        :param tool_handler: Low-level escape hatch — a pre-built
            :class:`ToolHandler` with custom schemas/dispatch. Most
            callers should use ``tools=`` instead.
        :param files: Optional list of local file paths to attach.
        :param stream: If True, return an ``AsyncIterator[str]``.
            If False (default), return the final text as a string.
        :returns: Final text (``stream=False``) or iterator of text
            chunks (``stream=True``).
        :raises ValueError: If both ``tools`` and ``tool_handler``
            are provided.
        """
        handler = _resolve_tool_handler(tools=tools, tool_handler=tool_handler)
        session = self.session(model=model, tool_handler=handler)
        if stream:
            return await session.query(input, files=files, stream=True)
        return await session.query(input, files=files)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> AgentPlaneClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def _resolve_tool_handler(
    *,
    tools: list[Callable[..., Any]] | None,
    tool_handler: ToolHandler | None,
) -> ToolHandler | None:
    """Pick one of ``tools=`` or ``tool_handler=``; reject both.

    :param tools: High-level list of ``@tool``-decorated functions.
    :param tool_handler: Low-level pre-built handler.
    :returns: The handler to use, or ``None`` if neither was given.
    :raises ValueError: If both were provided.
    """
    if tools is not None and tool_handler is not None:
        raise ValueError(
            "Pass either `tools=[...]` or `tool_handler=...`, not both. "
            "`tools=` is the high-level API (auto-builds a handler from "
            "@tool-decorated functions); `tool_handler=` is the low-level "
            "escape hatch."
        )
    if tools is not None:
        # Local import keeps the dep inside the tools subpackage.
        from .tools import build_tool_handler

        return build_tool_handler(tools)
    return tool_handler
