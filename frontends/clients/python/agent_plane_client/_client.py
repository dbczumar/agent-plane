"""AgentPlaneClient — the top-level client tying all namespaces together."""

from __future__ import annotations

import httpx

from ._agents import AgentsNamespace
from ._conversations import ConversationsNamespace
from ._files import FilesNamespace
from ._responses import ResponsesNamespace
from ._session import Session
from ._tool_handler import StreamHooks, ToolHandler


class AgentPlaneClient:
    """Typed Python client for the agent-plane server API.

    Usage::

        client = AgentPlaneClient(base_url="http://localhost:8080")
        agents = await client.agents.list()
        async for event in client.responses.stream(model="archer", input="hello"):
            ...
        await client.close()

    Or as a context manager::

        async with AgentPlaneClient(base_url="http://localhost:8080") as client:
            ...

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

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> AgentPlaneClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
