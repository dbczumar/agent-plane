"""Shared fixtures for server integration tests.

Uses real SqlAlchemyTaskStore + real DBOS workflow with a
ControllableMockClient that replaces the LLM. The mock auto-completes
by default so existing tests pass without modification. For concurrency
tests, use MockCall.block_until / MockCall.release to create
deterministic race windows.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from agent_plane.llms.types import (
    FunctionCallOutput,
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
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

# ── Controllable mock LLM ─────────────────────────────


@dataclass
class MockCall:
    """
    A single configured LLM call with optional synchronization
    gates.

    :param text: The assistant response text, e.g.
        ``"Hello from mock"``. Ignored when ``tool_calls`` is set.
    :param tool_calls: If set, the response contains function calls
        instead of text. Each dict must have ``"call_id"``,
        ``"name"``, and ``"arguments"`` keys, e.g.
        ``[{"call_id": "c1", "name": "grep", "arguments": "{}"}]``.
    :param block_before_response: If set, the mock awaits this
        event before producing any output. Call ``release()`` from
        the test to unblock.
    :param call_event: Set by the mock when this call is entered.
        Tests can ``await call_event.wait()`` to know the LLM was
        called.
    :param stream_tokens: If ``True``, yield individual text delta
        events before the completed event. If ``False``, yield only
        the completed event.
    :param exception: If set, ``create()`` raises this exception
        instead of returning a response. Used to simulate retryable
        LLM errors (e.g. ``httpx.HTTPStatusError`` with 429).
    :param tool_calls_fn: If set, called with the ``create()`` kwargs
        to produce ``tool_calls`` dynamically. Use when tool call
        arguments depend on runtime state (e.g. response_ids from a
        prior spawn). Takes precedence over static ``tool_calls``.
    :param received_kwargs: Populated by the mock when this call is
        consumed. Contains the kwargs passed to
        ``responses.create()`` so tests can inspect what the LLM
        received (e.g. ``input``, ``instructions``, ``model``).
        ``None`` until the call is executed.
    """

    text: str = "Hello from the test agent."
    tool_calls: list[dict[str, str]] | None = None
    # threading.Event (not asyncio.Event) so the test event loop
    # can ``set()`` cross-loop into DBOS's background event loop
    # where the workflow body runs. asyncio.Event's internal
    # futures are loop-bound — calling ``set()`` from loop A
    # never wakes a ``wait()`` parked on loop B (silent hang
    # under any block=True mock LLM scenario).
    block_before_response: threading.Event | None = None
    call_event: threading.Event = field(default_factory=threading.Event)
    stream_tokens: bool = False
    exception: Exception | None = None
    # Callable[[dict[str, Any]], list[dict[str, str]]] — generates
    # tool_calls dynamically from create() kwargs.
    tool_calls_fn: Any = None
    # Callable[[dict[str, Any]], Exception | None] — predicate
    # that conditionally raises based on inspecting the call's
    # kwargs. Returning None means "do not raise". Useful when
    # parent and sub-agent share the FIFO mock queue and only
    # one of them should fail (route by an input substring).
    exception_fn: Any = None
    # Populated by the mock when this call is consumed. Contains
    # the kwargs passed to responses.create() so tests can inspect
    # what the LLM received (e.g. the input/history).
    # Any: kwargs from responses.create() are heterogeneous.
    received_kwargs: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )

    async def wait_called(self, *, timeout: float = 10.0) -> None:
        """
        Asynchronously wait until this MockCall has been entered.

        Bridges the underlying sync ``threading.Event`` (chosen
        because the workflow body runs on DBOS's
        ``_background_event_loop`` while the test runs on
        pytest-asyncio's loop, and asyncio.Event doesn't sync
        cross-loop) into an awaitable the test can use.

        :param timeout: Max seconds to wait. ``TimeoutError`` is
            raised if exceeded — matches the prior behavior of
            ``asyncio.wait_for(call.call_event.wait(), timeout)``.
        """
        await asyncio.to_thread(self.call_event.wait, timeout)
        if not self.call_event.is_set():
            raise TimeoutError(
                f"MockCall.call_event not set within {timeout}s",
            )

    def release(self) -> None:
        """
        Unblock a call that is waiting on ``block_before_response``.
        """
        if self.block_before_response is not None:
            self.block_before_response.set()


def _build_completed_event(
    text: str,
    tool_calls: list[dict[str, str]] | None = None,
) -> ResponseCompletedEvent:
    """
    Build a ``ResponseCompletedEvent`` with text and/or tool calls.

    :param text: The assistant response text.
    :param tool_calls: Optional list of tool call dicts, each with
        ``"call_id"``, ``"name"``, and ``"arguments"`` keys, e.g.
        ``[{"call_id": "c1", "name": "grep", "arguments": "{}"}]``.
        When provided, function call outputs are included in the
        response alongside any text.
    :returns: A completed event with real ``llms.types`` dataclasses.
    """
    output: list[MessageOutput | FunctionCallOutput] = []
    if tool_calls:
        for tc in tool_calls:
            output.append(
                FunctionCallOutput(
                    call_id=tc["call_id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
            )
    else:
        output.append(MessageOutput(content=[OutputText(text=text)]))
    return ResponseCompletedEvent(
        response=Response(output=output, model="test-model"),
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
        await call_1.call_event.wait()  # know the LLM was called
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
        tool_calls: list[dict[str, str]] | None = None,
        tool_calls_fn: Any = None,
        exception: Exception | None = None,
        exception_fn: Any = None,
    ) -> MockCall:
        """
        Enqueue a configured call.

        :param text: Response text. Defaults to ``default_text``.
            Ignored when ``tool_calls`` is provided.
        :param block: If ``True``, the call blocks until
            ``MockCall.release()`` is called.
        :param stream_tokens: If ``True``, emit text delta events
            before the completed event.
        :param tool_calls: If provided, the response contains
            function calls instead of text. Each dict must have
            ``"call_id"``, ``"name"``, and ``"arguments"`` keys.
        :param tool_calls_fn: If provided, called with the
            ``create()`` kwargs to produce ``tool_calls``
            dynamically. Use when arguments depend on runtime
            state (e.g. response_ids from a prior spawn).
        :param exception: If provided, ``create()`` raises this
            instead of returning. Use with ``httpx.HTTPStatusError``
            to simulate retryable LLM errors.
        :returns: The ``MockCall`` for synchronization.
        """
        call = MockCall(
            text=text or self._default_text,
            tool_calls=tool_calls,
            tool_calls_fn=tool_calls_fn,
            block_before_response=threading.Event() if block else None,
            stream_tokens=stream_tokens,
            exception=exception,
            exception_fn=exception_fn,
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
        Release every blocked call so DBOS workflow tasks can exit.

        Called during fixture teardown to prevent the event loop from
        hanging on shutdown.
        """
        for call in self._calls:
            call.release()

    def get_call(self, index: int) -> MockCall:
        """
        Return a queued ``MockCall`` by index.

        Use this instead of accessing ``_calls`` directly so tests
        interact through a public interface.

        :param index: Zero-based index into the queued calls list,
            e.g. ``0`` for the first call.
        :returns: The ``MockCall`` at the given index.
        :raises IndexError: If *index* is out of range.
        """
        return self._calls[index]

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

    async def create(
        self,
        **kwargs: Any,
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """
        Mock ``responses.create()``. Consumes the next MockCall,
        optionally awaiting a gate, then returns a Response or
        stream.

        Async to match the real client's ``await create()``.

        :param kwargs: Responses API kwargs — captured on the
            ``MockCall.received_kwargs`` for test inspection.
        :returns: A ``Response`` if ``stream`` is falsy, or an
            async iterator of ``ResponseStreamEvent`` if
            ``stream=True``.
        """
        call = self._client._next_call()
        # Capture kwargs so tests can inspect what the LLM received
        call.received_kwargs = kwargs
        # Resolve dynamic tool_calls if a factory function is set.
        # Returns None to fall back to text (e.g. when the input
        # doesn't match the expected pattern for this call).
        if call.tool_calls_fn is not None:
            dynamic = call.tool_calls_fn(kwargs)
            if dynamic is not None:
                call.tool_calls = dynamic
        # Signal that this call has been entered. threading.Event
        # is thread-safe — set() from any loop wakes wait() in
        # any other loop, which matters because the workflow
        # body runs on DBOS's _background_event_loop while the
        # test runs on pytest-asyncio's loop.
        call.call_event.set()
        # Optionally block until the test releases us. Use
        # asyncio.to_thread to bridge a sync threading.Event
        # wait into the async event loop without parking the
        # loop — the offloaded thread blocks; the loop yields.
        if call.block_before_response is not None:
            await asyncio.to_thread(call.block_before_response.wait)
        # Raise configured exception (simulates retryable errors).
        # exception_fn fires only if its predicate decides this
        # specific kwargs payload should fail; useful for FIFO-
        # shared mocks where only one consumer (parent vs sub-
        # agent) should hit the failure path.
        if call.exception_fn is not None:
            dynamic_exc = call.exception_fn(kwargs)
            if dynamic_exc is not None:
                raise dynamic_exc
        if call.exception is not None:
            raise call.exception

        stream = kwargs.get("stream", False)
        if stream:
            return self._stream(call)
        return _build_completed_event(
            call.text,
            tool_calls=call.tool_calls,
        ).response

    async def _stream(
        self,
        call: MockCall,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """
        Yield streaming events for a call.

        :param call: The ``MockCall`` controlling this stream.
        """
        if call.stream_tokens and not call.tool_calls:
            # Yield individual word tokens as deltas
            for word in call.text.split():
                yield ResponseTextDeltaEvent(delta=word + " ")
        yield _build_completed_event(
            call.text,
            tool_calls=call.tool_calls,
        )


# ── Fixtures ──────────────────────────────────────────


@pytest.fixture()
def mock_llm() -> Iterator[ControllableMockClient]:
    """
    A ``ControllableMockClient`` instance for the current test.

    Tests that need to control LLM timing should call
    ``mock_llm.add_call(block=True)`` before creating responses.

    On teardown, releases all blocked calls so DBOS workflow tasks
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

    :param db_uri: SQLite connection URI from the ``db_uri`` fixture.
    :param tmp_path: Pytest temp directory for artifacts and cache.
    :param mock_llm: Controllable mock LLM client for the test.
    :param monkeypatch: Pytest fixture for patching the LLM client
        factory in the workflow module.
    """
    agent_store = SqlAlchemyAgentStore(db_uri)
    conversation_store = SqlAlchemyConversationStore(db_uri)
    file_store = SqlAlchemyFileStore(db_uri)
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
        file_store=file_store,
        artifact_store=artifact_store,
    )
    # Patch the LLM client in BOTH locations so the mock is used
    # everywhere:
    # - workflow._get_llm_client: checkpointed path (_call_llm,
    #   _call_llm_streaming)
    # - executors.default._get_llm_client: executor-managed path
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_llm_client",
        lambda: mock_llm,
    )
    monkeypatch.setattr(
        "agent_plane.runtime.executors.default._get_llm_client",
        lambda: mock_llm,
    )
    yield ts


@pytest.fixture()
def app(task_store: SqlAlchemyTaskStore, db_uri: str, tmp_path: Path) -> FastAPI:
    """
    Build the FastAPI app with real stores and real workflow
    execution (mock LLM is patched in via task_store fixture).
    """
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        task_store=task_store,
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=artifact_store,
        agent_cache=AgentCache(
            artifact_store=artifact_store,
            cache_dir=tmp_path / "cache",
        ),
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
