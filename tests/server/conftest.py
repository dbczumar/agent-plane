"""Shared fixtures for server integration tests.

Uses real SqlAlchemyTaskStore + real DBOS workflow with a
ControllableMockClient that replaces the LLM. The mock auto-completes
by default so existing tests pass without modification. For concurrency
tests, use MockCall.block_until / MockCall.release to create
deterministic race windows.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from agent_plane.runtime import init as init_runtime
from agent_plane.runtime.agent_cache import AgentCache
from agent_plane.runtime.durability import destroy_dbos
from agent_plane.server.app import create_app
from agent_plane.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from agent_plane.stores.artifact_store.local import LocalArtifactStore
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from agent_plane.stores.task_store.sqlalchemy_store import SqlAlchemyTaskStore
from llms.types import (
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

# ── Controllable mock LLM ─────────────────────────────


@dataclass
class MockCall:
    """
    A single configured LLM call with optional synchronization
    gates.

    :param text: The assistant response text, e.g.
        ``"Hello from mock"``.
    :param block_before_response: If set, the mock blocks on this
        event before producing any output. Call ``release()`` from
        the test to unblock.
    :param call_event: Set by the mock when this call is entered.
        Tests can ``call_event.wait()`` to know the LLM was called.
    :param stream_tokens: If ``True``, yield individual text delta
        events before the completed event. If ``False``, yield only
        the completed event.
    """

    text: str = "Hello from the test agent."
    block_before_response: threading.Event | None = None
    call_event: threading.Event = field(default_factory=threading.Event)
    stream_tokens: bool = False

    def release(self) -> None:
        """
        Unblock a call that is waiting on ``block_before_response``.
        """
        if self.block_before_response is not None:
            self.block_before_response.set()


def _build_completed_event(text: str) -> ResponseCompletedEvent:
    """
    Build a ``ResponseCompletedEvent`` with a single text output.

    :param text: The assistant response text.
    :returns: A completed event with real ``llms.types`` dataclasses.
    """
    return ResponseCompletedEvent(
        response=Response(
            output=[MessageOutput(content=[OutputText(text=text)])],
            model="test-model",
        ),
    )


class ControllableMockClient:
    """
    Mock LLM client with per-call synchronization gates.

    Replaces ``_get_llm_client()`` in ``workflow.py``. Each call to
    ``responses.create()`` consumes the next ``MockCall`` from the
    queue. If the queue is exhausted, uses a default auto-completing
    call.

    Usage::

        client = ControllableMockClient()
        # First LLM call blocks until released
        call_1 = client.add_call(text="First", block=True)
        # ... start workflow ...
        call_1.call_event.wait()  # know the LLM was called
        # ... inject steering message ...
        call_1.release()  # unblock

    :param default_text: Text for auto-generated default calls when
        the queue is empty, e.g. ``"Hello from the test agent."``.
    """

    def __init__(self, default_text: str = "Hello from the test agent.") -> None:
        self._calls: list[MockCall] = []
        self._call_index = 0
        self._lock = threading.Lock()
        self._default_text = default_text
        self.responses = _MockResponsesNamespace(self)

    def add_call(
        self,
        text: str | None = None,
        block: bool = False,
        stream_tokens: bool = False,
    ) -> MockCall:
        """
        Enqueue a configured call.

        :param text: Response text. Defaults to ``default_text``.
        :param block: If ``True``, the call blocks until
            ``MockCall.release()`` is called.
        :param stream_tokens: If ``True``, emit text delta events
            before the completed event.
        :returns: The ``MockCall`` for synchronization.
        """
        call = MockCall(
            text=text or self._default_text,
            block_before_response=threading.Event() if block else None,
            stream_tokens=stream_tokens,
        )
        self._calls.append(call)
        return call

    def _next_call(self) -> MockCall:
        """
        Return the next MockCall, or a default if queue exhausted.

        :returns: The next ``MockCall`` to execute.
        """
        with self._lock:
            if self._call_index < len(self._calls):
                call = self._calls[self._call_index]
                self._call_index += 1
                return call
            # Default: auto-complete immediately
            return MockCall(text=self._default_text)

    def release_all(self) -> None:
        """
        Release every blocked call so DBOS workflow threads can exit.

        Called during fixture teardown to prevent the event loop from
        hanging on shutdown.
        """
        for call in self._calls:
            call.release()

    @property
    def call_count(self) -> int:
        """
        Number of ``responses.create()`` invocations so far.

        :returns: The total call count.
        """
        with self._lock:
            return self._call_index


class _MockResponsesNamespace:
    """
    ``client.responses`` namespace that dispatches to
    ``ControllableMockClient``.

    :param client: The parent mock client.
    """

    def __init__(self, client: ControllableMockClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Response | Iterator[ResponseStreamEvent]:
        """
        Mock ``responses.create()``. Consumes the next MockCall,
        optionally blocking, then returns a Response or stream.

        :param kwargs: Ignored (accepts any Responses API kwargs).
        :returns: A ``Response`` if ``stream`` is falsy, or an
            iterator of ``ResponseStreamEvent`` if ``stream=True``.
        """
        call = self._client._next_call()
        # Signal that this call has been entered
        call.call_event.set()
        # Optionally block until the test releases us
        if call.block_before_response is not None:
            call.block_before_response.wait()

        stream = kwargs.get("stream", False)
        if stream:
            return self._stream(call)
        return _build_completed_event(call.text).response

    def _stream(self, call: MockCall) -> Iterator[ResponseStreamEvent]:
        """
        Yield streaming events for a call.

        :param call: The ``MockCall`` controlling this stream.
        :returns: An iterator of ``ResponseStreamEvent``.
        """
        if call.stream_tokens:
            # Yield individual word tokens as deltas
            for word in call.text.split():
                yield ResponseTextDeltaEvent(delta=word + " ")
        yield _build_completed_event(call.text)


# ── Fixtures ──────────────────────────────────────────


@pytest.fixture()
def mock_llm() -> Iterator[ControllableMockClient]:
    """
    A ``ControllableMockClient`` instance for the current test.

    Tests that need to control LLM timing should call
    ``mock_llm.add_call(block=True)`` before creating responses.

    On teardown, releases all blocked calls so DBOS workflow threads
    can exit cleanly and the event loop shuts down.
    """
    client = ControllableMockClient()
    yield client
    client.release_all()


@pytest.fixture()
def task_store(
    db_uri: str,
    tmp_path: Path,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[SqlAlchemyTaskStore]:
    """
    Real SqlAlchemyTaskStore with runtime initialized and mock LLM
    patched in.

    On teardown, releases all blocked mock calls and destroys the
    DBOS singleton so background threads exit before the event loop
    shuts down.
    """
    agent_store = SqlAlchemyAgentStore(db_uri)
    conversation_store = SqlAlchemyConversationStore(db_uri)
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    agent_cache = AgentCache(
        artifact_store=artifact_store,
        cache_dir=tmp_path / ".cache",
    )
    ts = SqlAlchemyTaskStore(db_uri)
    init_runtime(
        conversation_store=conversation_store,
        task_store=ts,
        agent_store=agent_store,
        agent_cache=agent_cache,
    )
    # Patch the LLM client so the real workflow uses our mock
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_llm_client",
        lambda: mock_llm,
    )
    yield ts


@pytest.fixture()
def app(task_store: SqlAlchemyTaskStore, db_uri: str, tmp_path: Path) -> FastAPI:
    """
    Build the FastAPI app with real stores and real workflow
    execution (mock LLM is patched in via task_store fixture).
    """
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        task_store=task_store,
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=LocalArtifactStore(str(tmp_path / "artifacts")),
    )


@pytest_asyncio.fixture()
async def client(
    app: FastAPI,
    mock_llm: ControllableMockClient,
) -> AsyncIterator[httpx.AsyncClient]:
    """
    Async HTTP client wired to the FastAPI app (no real server).

    On teardown, releases blocked mock calls and destroys DBOS
    before the event loop shuts down. This must happen in an async
    fixture because the pytest-asyncio runner closes the event loop
    immediately after async fixture teardown completes.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    # Release blocked mock calls so DBOS workflow threads can finish
    mock_llm.release_all()
    # Destroy DBOS to stop scheduler/queue/event-loop threads
    destroy_dbos()
