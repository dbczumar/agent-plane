"""Tests for agent_plane.runtime.workflow helpers.

Covers pagination, execution timeout, and tool call splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.entities import (
    CompactionData,
    ConversationItem,
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
    NewConversationItem,
)
from agent_plane.llms.errors import ContextWindowExceededError, PermanentLLMError
from agent_plane.runtime.caps import RuntimeCaps
from agent_plane.runtime.compaction import CompactionResult, SummaryMetadata, _CompactionState
from agent_plane.runtime.executor import DefaultExecutor
from agent_plane.runtime.workflow import (
    _cleanup_executor_storage,
    _load_initial_history,
    _maybe_persist_compaction_item,
    _persist_executor_storage,
    _proactive_compact_if_needed,
    _reactive_compact,
    _recover_spawn_state,
    _restore_executor_storage,
    _run_agent_loop,
    _split_tool_calls,
    _sync_history,
    _ToolCall,
    fetch_all_items,
)
from agent_plane.spec.types import (
    AgentSpec,
    CompactionConfig,
    ExecutorSpec,
    LLMConfig,
    RetryConfig,
    ToolsConfig,
)
from agent_plane.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from agent_plane.tools.client_specified import ClientSideToolSpec
from agent_plane.tools.manager import ToolManager


@pytest.fixture()
def conversation_store(db_uri: str) -> SqlAlchemyConversationStore:
    return SqlAlchemyConversationStore(db_uri)


def _make_user_message(index: int) -> NewConversationItem:
    """Build a simple user message item for testing."""
    return NewConversationItem(
        type="message",
        response_id="resp_001",
        data=MessageData(
            role="user",
            content=[{"type": "input_text", "text": f"msg {index}"}],
        ),
    )


# ── fetch_all_items ──────────────────────────────────


def test_fetch_all_items_empty_conversation(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    conv = conversation_store.create_conversation()
    items = fetch_all_items(conversation_store, conv.id)
    assert items == []


def test_fetch_all_items_single_page(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """Under the default limit of 100, all items come back in one page."""
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [_make_user_message(i) for i in range(5)],
    )
    items = fetch_all_items(conversation_store, conv.id)
    assert len(items) == 5


def test_fetch_all_items_paginates_beyond_limit(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    When a conversation has more items than the default page size
    (100), fetch_all_items must paginate through all pages.
    """
    conv = conversation_store.create_conversation()
    total = 150
    # Append in batches to keep individual appends manageable
    batch_size = 50
    for start in range(0, total, batch_size):
        conversation_store.append(
            conv.id,
            [_make_user_message(i) for i in range(start, start + batch_size)],
        )

    items = fetch_all_items(conversation_store, conv.id)
    assert len(items) == total

    # Verify ordering is preserved (ascending by position)
    texts = [item.data.content[0]["text"] for item in items]
    assert texts == [f"msg {i}" for i in range(total)]


def test_fetch_all_items_with_after_cursor(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    When given an after cursor, fetch_all_items only returns items
    after that cursor — and still paginates through all remaining pages.
    """
    conv = conversation_store.create_conversation()
    total = 150
    batch_size = 50
    for start in range(0, total, batch_size):
        conversation_store.append(
            conv.id,
            [_make_user_message(i) for i in range(start, start + batch_size)],
        )

    # Get the first page to grab a cursor from the middle
    first_page = conversation_store.list_items(conv.id, limit=50)
    cursor = first_page.last_id

    items = fetch_all_items(
        conversation_store,
        conv.id,
        after=cursor,
    )
    # Should get items 50..149 (the remaining 100)
    assert len(items) == 100

    texts = [item.data.content[0]["text"] for item in items]
    assert texts == [f"msg {i}" for i in range(50, total)]


def test_fetch_all_items_exactly_at_page_boundary(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    When item count equals the page size exactly, has_more is False
    and no extra page is fetched.
    """
    conv = conversation_store.create_conversation()
    conversation_store.append(
        conv.id,
        [_make_user_message(i) for i in range(100)],
    )
    items = fetch_all_items(conversation_store, conv.id)
    assert len(items) == 100


# ── Execution timeout enforcement ───────────────────


def _make_agent_spec(execution_timeout: int) -> AgentSpec:
    """
    Build a minimal AgentSpec with the given execution timeout.

    :param execution_timeout: Wall-clock timeout in seconds for
        the execution config, e.g. ``30``.
    :returns: An AgentSpec with an LLM config, tools config, and
        the specified execution timeout.
    """
    return AgentSpec(
        spec_version=1,
        name="timeout-test-agent",
        llm=LLMConfig(
            model="openai/gpt-4o",
            request_timeout=300,
            retry=RetryConfig(max_attempts=1),
        ),
        tools=ToolsConfig(),
        executor=ExecutorSpec(
            timeout=execution_timeout,
            max_iterations=1000,
        ),
    )


def _stub_tool_manager() -> MagicMock:
    """
    Create a MagicMock standing in for ToolManager.

    The agent loop calls ``tool_mgr.start()`` and
    ``tool_mgr.get_tool_schemas()``. This stub returns
    an empty tool list so the loop can proceed to the
    timeout check without needing real MCP connections.

    :returns: A MagicMock configured with ``start`` and
        ``get_tool_schemas`` methods.
    """
    mgr = MagicMock()
    mgr.start.return_value = None
    mgr.get_tool_schemas.return_value = []
    return mgr


def _stub_executor() -> DefaultExecutor:
    """
    Create a DefaultExecutor for timeout tests.

    The timeout tests exit the loop before reaching the executor
    call, so this executor is never invoked — it just satisfies
    the ``_run_agent_loop`` signature.

    :returns: A DefaultExecutor with minimal LLM config.
    """
    return DefaultExecutor(
        llm_config=LLMConfig(
            model="openai/gpt-4o",
            request_timeout=300,
            retry=RetryConfig(max_attempts=1),
        ),
    )


def _patch_agent_loop_deps(
    monkeypatch: pytest.MonkeyPatch,
    monotonic_values: list[float],
    caps: RuntimeCaps,
) -> list[dict[str, Any]]:
    """
    Monkeypatch all heavy dependencies of ``_run_agent_loop`` so
    the timeout logic can be tested in isolation.

    Patches: ``time.monotonic``, ``get_caps``, ``_write_output``,
    ``get_conversation_store``, ``get_task_store``, and
    ``fetch_all_items``.

    :param monkeypatch: The pytest monkeypatch fixture.
    :param monotonic_values: Sequence of floats returned by
        successive ``time.monotonic()`` calls, e.g.
        ``[0.0, 100.0]``.
    :param caps: The RuntimeCaps to return from ``get_caps()``.
    :returns: A mutable list that ``_write_output`` appends
        each emitted event dict to.
    """
    clock = iter(monotonic_values)
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.time.monotonic",
        lambda: next(clock),
    )

    monkeypatch.setattr(
        "agent_plane.runtime.workflow.get_caps",
        lambda: caps,
    )

    emitted_events: list[dict[str, Any]] = []

    def _capture_write(task_id: str, event: dict[str, Any]) -> None:
        """
        Capture events emitted by ``_write_output``.

        :param task_id: The task identifier (unused, captured
            for signature compatibility).
        :param event: The event dict to capture.
        """
        emitted_events.append(event)

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._write_output",
        _capture_write,
    )

    # Stub conversation store — the loop calls list_items
    # via fetch_all_items which we also stub
    mock_conv_store = MagicMock()
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.get_conversation_store",
        lambda: mock_conv_store,
    )

    # Stub task store — not reached in the timeout path
    mock_task_store = MagicMock()
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.get_task_store",
        lambda: mock_task_store,
    )

    # Return empty history so the loop starts with no items.
    # Both fetch_all_items (used inside tool/steering paths) and
    # _load_initial_history (used at loop start) are stubbed.
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.fetch_all_items",
        lambda store, conv_id, after=None: [],
    )
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._load_initial_history",
        lambda store, conv_id: [],
    )

    return emitted_events


def test_execution_timeout_resolution_takes_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The effective execution timeout is
    ``min(spec.executor.timeout, caps.execution_timeout)``.

    When the spec timeout (30s) is lower than the cap (7200s),
    the loop uses 30s. We verify by providing a monotonic clock
    that exceeds 30s but not 7200s on the first check, and
    confirming the loop terminates with execution_timeout.
    """
    spec = _make_agent_spec(execution_timeout=30)
    # Cap is much higher — spec timeout (30) should win
    caps = RuntimeCaps(execution_timeout=7200)
    # First call: start_time=0.0, second call: elapsed=31.0 (> 30)
    _patch_agent_loop_deps(
        monkeypatch,
        monotonic_values=[0.0, 31.0],
        caps=caps,
    )

    result = _run_agent_loop(
        task_id="task_timeout_min",
        conversation_id="conv_001",
        spec=spec,
        agent_name="timeout-test-agent",
        agent_id="ag_test",
        instructions=None,
        tool_mgr=_stub_tool_manager(),
        executor=_stub_executor(),
    )

    # The loop should have terminated due to timeout, not
    # max_iterations — confirming min(30, 7200) = 30 was used
    assert result.status == "incomplete", (
        "Expected 'incomplete' status when elapsed exceeds spec "
        "timeout, indicating the min(spec, cap) resolved to "
        "the spec value"
    )
    assert result.incomplete_details == {"reason": "execution_timeout"}, (
        "Expected execution_timeout reason, confirming the "
        "resolved timeout (not max_iterations) triggered the exit"
    )


def test_execution_timeout_terminates_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When ``time.monotonic()`` indicates the deadline is exceeded,
    the loop terminates with status='incomplete' and
    incomplete_details={'reason': 'execution_timeout'}.
    """
    spec = _make_agent_spec(execution_timeout=60)
    caps = RuntimeCaps(execution_timeout=60)
    # First call: start_time=0.0, second call: elapsed=60.0 (== timeout)
    _patch_agent_loop_deps(
        monkeypatch,
        monotonic_values=[0.0, 60.0],
        caps=caps,
    )

    result = _run_agent_loop(
        task_id="task_timeout_term",
        conversation_id="conv_002",
        spec=spec,
        agent_name="timeout-test-agent",
        agent_id="ag_test",
        instructions=None,
        tool_mgr=_stub_tool_manager(),
        executor=_stub_executor(),
    )

    # Status must be "incomplete" — not "completed" or "failed"
    assert result.status == "incomplete", (
        f"Expected 'incomplete' when elapsed >= execution_timeout; got '{result.status}'"
    )
    # incomplete_details must specify the timeout reason
    assert result.incomplete_details == {"reason": "execution_timeout"}, (
        f"Expected execution_timeout reason in incomplete_details; got {result.incomplete_details}"
    )
    # Output list must exist (even if empty) — not None
    assert isinstance(result.output, list), "Output must be a list, not None or other type"


def test_execution_timeout_emits_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When execution timeout fires, a ``response.error`` SSE event
    is emitted with ``source='execution'``,
    ``error.code='execution_timeout'``, and a message containing
    the timeout value.
    """
    spec = _make_agent_spec(execution_timeout=45)
    caps = RuntimeCaps(execution_timeout=100)
    # Resolved timeout = min(45, 100) = 45
    # First call: start_time=0.0, second call: elapsed=50.0 (> 45)
    emitted = _patch_agent_loop_deps(
        monkeypatch,
        monotonic_values=[0.0, 50.0],
        caps=caps,
    )

    _run_agent_loop(
        task_id="task_timeout_event",
        conversation_id="conv_003",
        spec=spec,
        agent_name="timeout-test-agent",
        agent_id="ag_test",
        instructions=None,
        tool_mgr=_stub_tool_manager(),
        executor=_stub_executor(),
    )

    # Exactly one error event should have been emitted
    assert len(emitted) == 1, f"Expected exactly 1 SSE event; got {len(emitted)}"
    event = emitted[0]

    # Verify event type identifies this as an error event
    assert event["type"] == "response.error", (
        "Event type must be 'response.error' for timeout events"
    )
    # Verify the event source is "execution" (not "llm" or "tool")
    assert event["source"] == "execution", (
        "Event source must be 'execution' for execution-level timeouts"
    )
    # Verify error code is "execution_timeout"
    assert event["error"]["code"] == "execution_timeout", (
        "Error code must be 'execution_timeout'; got '{}'".format(event["error"]["code"])
    )
    # Verify the message contains the resolved timeout value
    assert "45s" in event["error"]["message"], (
        "Error message must contain the resolved timeout '45s'; got '{}'".format(
            event["error"]["message"]
        )
    )
    # Verify detail is None (no additional detail for timeouts)
    assert event["error"]["detail"] is None, "Error detail must be None for execution timeouts"


def test_execution_timeout_preserves_prior_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Output items accumulated before the timeout fires are
    preserved in the returned ``_AgentLoopResult.output``.

    We simulate this by pre-populating the ``output_items`` list
    via the first LLM iteration producing output, then timing
    out on the second iteration. However, since we control the
    clock and the loop checks timeout at the TOP of each
    iteration, we instead directly test that a non-empty
    output_items list at timeout time is preserved.

    Approach: patch ``_sync_history`` and
    ``_call_llm_for_iteration_with_error_handling`` to simulate
    one successful iteration that appends to output_items, then
    time out on the second iteration.
    """
    spec = _make_agent_spec(execution_timeout=60)
    caps = RuntimeCaps(execution_timeout=60)

    # Clock: start=0.0, first check=10.0 (under 60, loop runs),
    # _sync_history call, LLM call, second check=70.0 (over 60,
    # timeout). We need enough monotonic values for all calls.
    _patch_agent_loop_deps(
        monkeypatch,
        # 0.0 = start_time
        # 10.0 = first iteration timeout check (< 60, proceed)
        # 70.0 = second iteration timeout check (>= 60, timeout)
        monotonic_values=[0.0, 10.0, 70.0],
        caps=caps,
    )

    # Stub _sync_history to be a no-op that returns last_seen
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._sync_history",
        lambda conv_store, conv_id, last_seen, history: last_seen,
    )

    # Track the output_items list reference so we can inject
    # a prior item into it during the LLM call
    prior_output_item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "partial"}],
    }

    def _fake_executor_turn(
        task_id: str,
        executor: Any,
        spec: AgentSpec,
        llm_config: LLMConfig,
        history: list[Any],
        instructions: str | None,
        tool_schemas: list[Any],
        compaction_state: _CompactionState,
        context: Any,
        content_cache: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Fake executor turn that simulates a tool-call response.

        We return a response with tool calls so the loop enters
        ``_handle_tool_calls``, which we also stub. The important
        thing is that ``output_items`` gets populated before the
        next iteration's timeout check.

        :param task_id: Task identifier (unused).
        :param executor: Executor (unused).
        :param spec: Agent spec (unused).
        :param llm_config: LLM config (unused).
        :param history: Conversation history (unused).
        :param instructions: Instructions (unused).
        :param tool_schemas: Tool schemas (unused).
        :param compaction_state: Per-execution compaction state (unused).
        :param context: Executor context (unused).
        :param content_cache: Per-task content cache (unused).
        :returns: An LLM response dict with empty tool calls (tool
            detection is controlled via the separate ``_has_tool_calls``
            stub).
        """
        return {"model": "fake", "text": None, "tool_calls": [], "native_tool_items": []}

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._executor_turn_with_compaction",
        _fake_executor_turn,
    )

    # _has_tool_calls returns True so loop enters tool handling
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._has_tool_calls",
        lambda resp: True,
    )

    def _fake_handle_tool_calls(
        task_id: str,
        conversation_id: str,
        llm_resp: Any,
        agent_name: str,
        agent_id: str,
        tools_config: ToolsConfig,
        history: list[Any],
        output_items: list[dict[str, Any]],
        conv_store: Any,
        tool_mgr: Any,
    ) -> str | None:
        """
        Fake tool call handler that injects a prior output item.

        This simulates output accumulated during the first
        iteration before the timeout fires on the second.

        :param task_id: Task identifier (unused).
        :param conversation_id: Conversation ID (unused).
        :param llm_resp: LLM response (unused).
        :param agent_name: Agent name (unused).
        :param agent_id: Agent ID (unused).
        :param tools_config: Tools config (unused).
        :param history: Conversation history (unused).
        :param output_items: Mutable output list — we append
            a prior item to simulate accumulated output.
        :param conv_store: Conversation store (unused).
        :param tool_mgr: Tool manager (unused).
        :returns: A last_seen cursor value.
        """
        output_items.append(prior_output_item)
        return "item_001"

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._handle_tool_calls",
        _fake_handle_tool_calls,
    )

    # Stub _sync_steered_after_tools to return last_seen unchanged
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._sync_steered_after_tools",
        lambda cs, cid, pre, post, hist: post,
    )

    result = _run_agent_loop(
        task_id="task_timeout_preserve",
        conversation_id="conv_004",
        spec=spec,
        agent_name="timeout-test-agent",
        agent_id="ag_test",
        instructions=None,
        tool_mgr=_stub_tool_manager(),
        executor=_stub_executor(),
    )

    # Status must be incomplete due to timeout
    assert result.status == "incomplete", "Expected 'incomplete' after timeout on second iteration"
    assert result.incomplete_details == {"reason": "execution_timeout"}, (
        "Expected execution_timeout reason after second iteration"
    )
    # The prior output item must be preserved in the result
    assert len(result.output) == 1, (
        f"Expected 1 prior output item preserved after timeout; got {len(result.output)}"
    )
    assert result.output[0] == prior_output_item, (
        "The preserved output item must match the item appended "
        "during the first iteration's tool handling"
    )


# ── _split_tool_calls ─────────────────────────────────


def _make_tool_manager(
    client_tool_names: list[str],
) -> ToolManager:
    """
    Build a ToolManager with only client-side tools registered.

    Uses a minimal AgentSpec with no skills or MCP servers, so
    the only registered tools are the client-specified ones.

    :param client_tool_names: Names for the client-side tools
        to register, e.g. ``["get_weather", "search"]``.
    :returns: A ToolManager with the specified client-side tools.
    """
    spec = AgentSpec(
        spec_version=1,
        name="split-test-agent",
        llm=LLMConfig(
            model="openai/gpt-4o",
            request_timeout=300,
            retry=RetryConfig(max_attempts=1),
        ),
        tools=ToolsConfig(),
        executor=ExecutorSpec(timeout=60, max_iterations=100),
    )
    client_specs = [
        ClientSideToolSpec(
            name=name,
            schema={
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Client tool: {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )
        for name in client_tool_names
    ]
    return ToolManager(
        spec=spec,
        client_tool_specs=client_specs,
    )


def _make_tool_call(name: str) -> _ToolCall:
    """
    Build a minimal _ToolCall for testing.

    :param name: The tool function name, e.g.
        ``"load_skill"`` or ``"get_weather"``.
    :returns: A _ToolCall with the given name and dummy
        call_id / arguments.
    """
    return _ToolCall(
        call_id=f"call_{name}",
        name=name,
        arguments="{}",
    )


def test_split_all_server_side() -> None:
    """
    When no tools are client-side, all calls land in
    ``server`` and ``has_client`` is False.
    """
    tool_mgr = _make_tool_manager(client_tool_names=[])
    tool_calls = [
        _make_tool_call("load_skill"),
        _make_tool_call("mcp_github"),
    ]

    split = _split_tool_calls(tool_calls, tool_mgr)

    # All tools are server-side — none are registered as client tools
    assert len(split.server) == 2, (
        f"Expected 2 server-side tools, got {len(split.server)}. "
        "If 0, is_client_side_tool is returning True for unregistered tools."
    )
    assert split.server[0].name == "load_skill"
    assert split.server[1].name == "mcp_github"
    # No client tools in the batch
    assert split.has_client is False, "has_client should be False when no client-side tools exist"


def test_split_all_client_side() -> None:
    """
    When all tools are client-side, ``server`` is empty
    and ``has_client`` is True.
    """
    tool_mgr = _make_tool_manager(
        client_tool_names=["get_weather", "search"],
    )
    tool_calls = [
        _make_tool_call("get_weather"),
        _make_tool_call("search"),
    ]

    split = _split_tool_calls(tool_calls, tool_mgr)

    # No server-side tools — both are client-side
    assert split.server == [], (
        f"Expected empty server list, got {len(split.server)} tools. "
        "If non-empty, is_client_side_tool failed to detect a registered client tool."
    )
    assert split.has_client is True, "has_client should be True when client-side tools are present"


def test_split_mixed_batch() -> None:
    """
    When a batch contains both server-side and client-side tools,
    only server-side tools appear in ``server`` and ``has_client``
    is True.

    This is the critical scenario: the old code used ``any()`` and
    skipped ALL tools when any client tool was present. The fix
    ensures server-side tools are separated for execution.
    """
    tool_mgr = _make_tool_manager(
        client_tool_names=["get_weather"],
    )
    tool_calls = [
        _make_tool_call("load_skill"),
        _make_tool_call("get_weather"),
        _make_tool_call("mcp_github"),
    ]

    split = _split_tool_calls(tool_calls, tool_mgr)

    # 2 server-side tools (load_skill, mcp_github), 1 client (get_weather)
    assert len(split.server) == 2, (
        f"Expected 2 server-side tools, got {len(split.server)}. "
        "If 0, the old any() bug is back — all tools are being skipped. "
        "If 3, get_weather was not detected as client-side."
    )
    server_names = [tc.name for tc in split.server]
    assert server_names == ["load_skill", "mcp_github"], (
        f"Server tools should be load_skill and mcp_github in order; got {server_names}"
    )
    assert split.has_client is True, (
        "has_client should be True — get_weather is a client-side tool"
    )


def test_split_preserves_tool_call_fields() -> None:
    """
    Server-side _ToolCall instances in the split preserve all
    fields (call_id, name, arguments) from the original.
    """
    tool_mgr = _make_tool_manager(client_tool_names=["search"])
    tc = _ToolCall(
        call_id="call_xyz789",
        name="load_skill",
        arguments='{"name": "summarize"}',
    )

    split = _split_tool_calls([tc], tool_mgr)

    # The server tool is the same object — fields intact
    assert split.server[0].call_id == "call_xyz789", "call_id must survive the split unchanged"
    assert split.server[0].name == "load_skill"
    assert split.server[0].arguments == '{"name": "summarize"}', (
        "arguments must survive the split unchanged"
    )


# ── _reactive_compact tiktoken validation ────────────────────────────────────


def test_reactive_compact_plausible_overflow_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When tiktoken estimate is within 30% of the provider-reported token count,
    _reactive_compact proceeds with compaction and returns compacted messages.
    """
    # count_tokens returns 140000 — 1.4% divergence from reported 142000 (within 30%).
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        lambda msgs, model: 140000,
    )
    compacted_messages = [{"role": "user", "content": "compacted summary"}]
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.compact",
        lambda messages, history, **kw: CompactionResult(
            messages=compacted_messages,
            summary_metadata=None,
        ),
    )

    class _RaisesIfCalled:
        """Stub LLM client — compact() is patched out so this must never be used."""

        class responses:
            """Responses namespace."""

            @staticmethod
            def create(**kwargs: Any) -> None:
                """Raise — compact() is patched so the client must not be reached."""
                raise AssertionError("_get_llm_client() result was accessed unexpectedly")

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_llm_client",
        lambda: _RaisesIfCalled(),
    )

    exc = ContextWindowExceededError(
        "Context window exceeded: 142000 tokens > 128000 max",
        code="context_length_exceeded",
        max_context_tokens=128000,
        actual_tokens=142000,
    )
    state = _CompactionState(
        context_window=None,
        last_summary=None,
        config=None,
        model="openai/gpt-4o",
    )
    messages = [{"role": "user", "content": "long conversation history"}]

    result = _reactive_compact(messages, [], 1000, exc, state, "task_001")

    # Compacted messages returned — proves compact() was called (not re-raised).
    assert result == compacted_messages, (
        f"Expected compacted messages to be returned, got: {result!r}. "
        "Failure means _reactive_compact raised PermanentLLMError instead of compacting."
    )
    # context_window cached from the overflow for proactive checks in future iterations.
    assert state.context_window == 128000, (
        f"Expected context_window cached as 128000, got: {state.context_window}. "
        "Failure means the discovered window was not saved for proactive checks."
    )


def test_reactive_compact_implausible_overflow_raises_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When tiktoken estimate diverges from provider-reported tokens by > 30%,
    _reactive_compact re-raises as PermanentLLMError to prevent a pointless
    compact-retry loop on a misclassified error.
    """
    # count_tokens returns 50000 — 65% divergence from reported 142000 (outside 30%).
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        lambda msgs, model: 50000,
    )

    exc = ContextWindowExceededError(
        "Context window exceeded: 142000 tokens > 128000 max",
        code="context_length_exceeded",
        max_context_tokens=128000,
        actual_tokens=142000,
    )
    state = _CompactionState(
        context_window=None,
        last_summary=None,
        config=None,
        model="openai/gpt-4o",
    )
    messages = [{"role": "user", "content": "test"}]

    with pytest.raises(PermanentLLMError) as exc_info:
        _reactive_compact(messages, [], 1000, exc, state, "task_001")

    # Must be PermanentLLMError, not ContextWindowExceededError, to prevent retry loop.
    assert exc_info.value.code == exc.code, (
        f"Expected code '{exc.code}' preserved on re-raise, got '{exc_info.value.code}'. "
        "Failure means the re-raised error lost its original error code."
    )
    # Compact must not have been called — there's nothing to monkeypatch but we verify
    # by checking that the state.context_window was cached before the validation check.
    assert state.context_window == 128000, (
        "context_window should be cached even when validation fails — "
        "the window was discovered from the error before the ratio check."
    )


# ── _proactive_compact_if_needed ──────────────────────────────────────────────


def test_proactive_compact_skips_when_context_window_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When ``context_window`` is None (no overflow seen yet),
    proactive compaction returns messages unchanged.
    """
    messages = [{"role": "user", "content": "hello"}]
    state = _CompactionState(
        context_window=None,
        last_summary=None,
        config=None,
        model="openai/gpt-4o",
    )

    result = _proactive_compact_if_needed(
        messages,
        [],
        100,
        state,
        "task_001",
    )

    # Messages returned unchanged — proactive path not taken because
    # context_window is None (no prior overflow to establish the window).
    assert result is messages, (
        "Expected original messages returned (identity check). "
        "If a copy was returned, proactive compaction ran when "
        "context_window is None — it should have skipped."
    )


def test_proactive_compact_skips_when_under_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When estimated tokens are below the trigger threshold,
    proactive compaction returns messages unchanged.
    """
    # 5000 tokens + 100 sys_tokens = 5100, well under 128000 * 0.8 = 102400
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        lambda msgs, model: 5000,
    )

    messages = [{"role": "user", "content": "short message"}]
    state = _CompactionState(
        context_window=128000,
        last_summary=None,
        config=CompactionConfig(trigger_threshold=0.8),
        model="openai/gpt-4o",
    )

    result = _proactive_compact_if_needed(
        messages,
        [],
        100,
        state,
        "task_001",
    )

    # 5000 + 100 = 5100 < 102400 threshold — no compaction needed.
    assert result is messages, (
        "Expected original messages returned unchanged. "
        "If compaction ran, the threshold check is broken — "
        "5100 tokens should be well under the 102400 budget."
    )


def test_proactive_compact_fires_when_over_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When estimated tokens exceed the trigger threshold,
    proactive compaction runs and returns compacted messages.
    """
    # 110000 + 1000 sys_tokens = 111000, over 128000 * 0.8 = 102400
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        lambda msgs, model: 110000,
    )
    compacted = [{"role": "user", "content": "compacted"}]
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.compact",
        lambda messages, history, **kw: CompactionResult(
            messages=compacted,
            summary_metadata=None,
        ),
    )

    class _RaisesIfCalled:
        """Stub — compact() is patched so LLM must not be reached."""

        class responses:
            """Responses namespace."""

            @staticmethod
            def create(**kwargs: Any) -> None:
                """Raise — compact() is patched."""
                raise AssertionError("LLM client reached unexpectedly")

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_llm_client",
        lambda: _RaisesIfCalled(),
    )

    messages = [{"role": "user", "content": "long history..."}]
    state = _CompactionState(
        context_window=128000,
        last_summary=None,
        config=CompactionConfig(trigger_threshold=0.8),
        model="openai/gpt-4o",
    )

    result = _proactive_compact_if_needed(
        messages,
        [],
        1000,
        state,
        "task_001",
    )

    # 110000 + 1000 = 111000 > 102400 — compact() must have fired.
    assert result == compacted, (
        f"Expected compacted messages, got {result!r}. "
        "If original messages returned, the threshold check "
        "didn't trigger compaction at 111000 tokens vs 102400 budget."
    )


def test_proactive_compact_updates_last_summary_on_layer2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When proactive compaction triggers Layer 2, the summary metadata
    is stored in ``compaction_state.last_summary`` for end-of-execution
    persistence.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        lambda msgs, model: 110000,
    )
    summary = SummaryMetadata(
        text="Proactive summary.",
        last_item_id="msg_099",
        model="openai/gpt-4o",
        token_count=200,
    )
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.compact",
        lambda messages, history, **kw: CompactionResult(
            messages=[{"role": "assistant", "content": "summary"}],
            summary_metadata=summary,
        ),
    )

    class _RaisesIfCalled:
        """Stub — compact() is patched so LLM must not be reached."""

        class responses:
            """Responses namespace."""

            @staticmethod
            def create(**kwargs: Any) -> None:
                """Raise — compact() is patched."""
                raise AssertionError("LLM client reached unexpectedly")

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._get_llm_client",
        lambda: _RaisesIfCalled(),
    )

    state = _CompactionState(
        context_window=128000,
        last_summary=None,
        config=CompactionConfig(trigger_threshold=0.8),
        model="openai/gpt-4o",
    )

    _proactive_compact_if_needed(
        [{"role": "user", "content": "long"}],
        [],
        1000,
        state,
        "task_001",
    )

    # last_summary must be populated so _maybe_persist_compaction_item
    # writes it at the end of execution.
    assert state.last_summary is summary, (
        "Expected last_summary to be set to the Layer 2 summary. "
        "If None, proactive compaction didn't propagate summary_metadata "
        "to the state — it would be lost and never persisted."
    )


def test_proactive_compact_uses_config_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Proactive compaction respects the configured trigger_threshold,
    not just the 0.8 default.
    """
    # With threshold=0.5: budget = 128000 * 0.5 = 64000
    # 60000 + 1000 = 61000 < 64000 → should NOT compact
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        lambda msgs, model: 60000,
    )

    messages = [{"role": "user", "content": "medium history"}]
    state = _CompactionState(
        context_window=128000,
        last_summary=None,
        config=CompactionConfig(trigger_threshold=0.5),
        model="openai/gpt-4o",
    )

    result = _proactive_compact_if_needed(
        messages,
        [],
        1000,
        state,
        "task_001",
    )

    # 61000 < 64000 — under the custom 50% threshold.
    assert result is messages, (
        "Expected no compaction with 50% threshold. "
        "61000 tokens is under the 64000 budget. "
        "If compacted, threshold config was ignored."
    )


# ── _load_initial_history ─────────────────────────────────────────────────────


def test_load_initial_history_no_compaction_loads_everything(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    With no compaction item, _load_initial_history loads the full conversation.
    """
    conv = conversation_store.create_conversation()
    for i in range(5):
        conversation_store.append(
            conv.id,
            [
                NewConversationItem(
                    type="message",
                    response_id=f"resp_{i:03d}",
                    data=MessageData(
                        role="user",
                        content=[{"type": "input_text", "text": f"msg {i}"}],
                    ),
                )
            ],
        )

    history = _load_initial_history(conversation_store, conv.id)

    # All 5 messages loaded — no compaction cursor to limit the query.
    assert len(history) == 5, (
        f"Expected 5 items loaded when no compaction exists, got {len(history)}. "
        "Failure means _load_initial_history incorrectly skipped items."
    )
    # No compaction items in the result — they are filtered out.
    assert all(item.type != "compaction" for item in history)


def test_load_initial_history_with_compaction_loads_summary_plus_recent(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    With a compaction item, _load_initial_history returns a synthetic summary
    pair and only the items AFTER last_item_id, not the full history.
    """
    conv = conversation_store.create_conversation()

    # Append 3 "old" items that will be covered by the summary
    old_items = conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_000",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": f"old msg {i}"}],
                ),
            )
            for i in range(3)
        ],
    )
    last_covered_id = old_items[-1].id  # last item covered by compaction

    # Append the compaction item (covers items 0..2)
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="compaction",
                response_id="task_001",
                data=CompactionData(
                    summary="User asked about X. Agent answered Y.",
                    last_item_id=last_covered_id,
                    model="openai/gpt-4o",
                    token_count=50,
                ),
            )
        ],
    )

    # Append 2 "recent" items after the compaction
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_001",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": f"recent msg {i}"}],
                ),
            )
            for i in range(2)
        ],
    )

    history = _load_initial_history(conversation_store, conv.id)

    # 2 synthetic items (user + assistant from compaction) + 2 recent items = 4 total.
    assert len(history) == 4, (
        f"Expected 4 items (2 synthetic + 2 recent), got {len(history)}. "
        "Failure means old items were loaded verbatim instead of replaced by summary."
    )
    # First item is the synthetic user summary request.
    assert history[0].type == "message"
    assert isinstance(history[0].data, MessageData)
    assert history[0].data.role == "user"
    assert "[This is an automatically generated summary" in history[0].data.content[0]["text"]
    # Second item is the synthetic assistant summary.
    assert history[1].data.role == "assistant"
    assert history[1].data.content[0]["text"] == "User asked about X. Agent answered Y."
    # Items 2 and 3 are the recent real messages.
    assert history[2].data.role == "user"
    assert history[2].data.content[0]["text"] == "recent msg 0"
    # No compaction items appear in the result — they are metadata only.
    assert all(item.type != "compaction" for item in history)


def test_load_initial_history_post_summary_items_not_lost(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    Items appended AFTER the compaction but before last_item_id are NOT lost —
    _load_initial_history uses last_item_id as the cursor, not the compaction item's position.
    """
    conv = conversation_store.create_conversation()

    # 5 "old" items
    old_items = conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_000",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": f"old {i}"}],
                ),
            )
            for i in range(5)
        ],
    )
    covered_id = old_items[2].id  # summary covers items 0..2 only

    # Append compaction covering only items 0..2 (items 3,4 are NOT covered)
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="compaction",
                response_id="task_001",
                data=CompactionData(
                    summary="Summary of items 0 to 2.",
                    last_item_id=covered_id,
                    model="openai/gpt-4o",
                    token_count=30,
                ),
            )
        ],
    )

    history = _load_initial_history(conversation_store, conv.id)

    # Expected: 2 synthetic items (user+assistant) + items 3 and 4 = 4 total.
    # Items 3 and 4 are AFTER covered_id so they must not be lost.
    assert len(history) == 4, (
        f"Expected 4 items (2 synthetic + 2 post-coverage items), got {len(history)}. "
        "Failure means items between last_item_id and compaction item were dropped."
    )
    # Items 3 and 4 must be present verbatim.
    assert history[2].data.content[0]["text"] == "old 3"
    assert history[3].data.content[0]["text"] == "old 4"


# ── _maybe_persist_compaction_item ───────────────────────────────────────────


def test_maybe_persist_compaction_item_idempotent(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    Calling _maybe_persist_compaction_item twice with the same task_id
    results in exactly one compaction item (idempotent for crash-recovery).
    """
    conv = conversation_store.create_conversation()
    summary = SummaryMetadata(
        text="First summary.",
        last_item_id="msg_abc",
        model="openai/gpt-4o",
        token_count=25,
    )

    _maybe_persist_compaction_item(summary, "task_001", conv.id, conversation_store)
    # Simulate crash-recovery replay: call again with the same task_id.
    _maybe_persist_compaction_item(summary, "task_001", conv.id, conversation_store)

    items = conversation_store.list_items(conv.id, type="compaction")
    # Only one compaction item must exist — the second call was a no-op.
    assert len(items.data) == 1, (
        f"Expected exactly 1 compaction item after idempotent double-persist, "
        f"got {len(items.data)}. Failure means the dedup check is missing or broken."
    )
    item = items.data[0]
    # Full structure must be preserved through persist round-trip.
    assert item.data.summary == "First summary.", f"summary text mismatch: {item.data.summary!r}"
    assert item.data.last_item_id == "msg_abc", (
        f"last_item_id mismatch: {item.data.last_item_id!r}"
    )
    assert item.data.model == "openai/gpt-4o", f"model mismatch: {item.data.model!r}"
    # response_id is on ConversationItem, not CompactionData — it's the task_id.
    # The dedup guard in _maybe_persist_compaction_item checks this field.
    assert item.response_id == "task_001", (
        f"response_id={item.response_id!r} must equal task_id 'task_001'. "
        "If different, the dedup guard would not block re-append on crash replay."
    )


def test_maybe_persist_compaction_item_different_task_ids_produce_separate_items(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    Two executions with different task IDs each produce their own compaction item.
    """
    conv = conversation_store.create_conversation()
    summary1 = SummaryMetadata(
        text="Summary from task 1.",
        last_item_id="msg_001",
        model="openai/gpt-4o",
        token_count=20,
    )
    summary2 = SummaryMetadata(
        text="Summary from task 2.",
        last_item_id="msg_002",
        model="openai/gpt-4o",
        token_count=25,
    )

    _maybe_persist_compaction_item(summary1, "task_001", conv.id, conversation_store)
    _maybe_persist_compaction_item(summary2, "task_002", conv.id, conversation_store)

    items = conversation_store.list_items(conv.id, type="compaction")
    # Two compaction items — one per execution.
    assert len(items.data) == 2, (
        f"Expected 2 compaction items (one per task), got {len(items.data)}. "
        "Failure means different task IDs were incorrectly treated as duplicates."
    )
    # response_id is on ConversationItem — use it to map items by task_id.
    by_response_id = {i.response_id: i for i in items.data}
    # Each task_id produces a distinct compaction item with the correct summary.
    assert by_response_id["task_001"].data.summary == "Summary from task 1.", (
        f"task_001 summary mismatch: {by_response_id.get('task_001')}"
    )
    assert by_response_id["task_001"].data.last_item_id == "msg_001", (
        "task_001 last_item_id mismatch"
    )
    assert by_response_id["task_002"].data.summary == "Summary from task 2.", (
        f"task_002 summary mismatch: {by_response_id.get('task_002')}"
    )
    assert by_response_id["task_002"].data.last_item_id == "msg_002", (
        "task_002 last_item_id mismatch"
    )


# ── _sync_history compaction filter ──────────────────────────────────────────


def test_sync_history_filters_compaction_items(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    """
    ``_sync_history`` must exclude ``type=compaction`` items from
    the history list while still advancing ``last_seen`` past them.

    If the filter were removed, compaction items would be appended
    to history and eventually passed to the LLM prompt builder,
    which would silently skip them — but the history list should
    never contain items the prompt builder ignores.

    :param conversation_store: Real SQLAlchemy store for the test.
    """
    conv = conversation_store.create_conversation()

    # Append a user message, then a compaction item, then another
    # user message. This simulates: execution 1 produces a
    # compaction item, then execution 2 appends a new user message.
    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_001",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "msg before compact"}],
                ),
            ),
        ],
    )
    first_items = conversation_store.list_items(conv.id, limit=1)
    first_id = first_items.data[0].id

    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="compaction",
                response_id="resp_001",
                data=CompactionData(
                    summary="Summary of prior context.",
                    last_item_id=first_id,
                    model="openai/gpt-4o",
                    token_count=50,
                ),
            ),
        ],
    )

    conversation_store.append(
        conv.id,
        [
            NewConversationItem(
                type="message",
                response_id="resp_002",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": "msg after compact"}],
                ),
            ),
        ],
    )

    # Start with last_seen pointing at the first message.
    # _sync_history should pick up the compaction item AND the
    # second message, but only add the message to history.
    history: list[Any] = []
    new_last_seen = _sync_history(
        conversation_store,
        conv.id,
        first_id,
        history,
    )

    # History must contain ONLY the second user message — the
    # compaction item must be filtered out.
    assert len(history) == 1, (
        f"Expected 1 item in history (the post-compaction message), "
        f"got {len(history)}. "
        f"Types: {[h.type for h in history]}. "
        f"If 2, the compaction item was not filtered out."
    )
    assert history[0].type == "message", (
        f"Expected message type, got {history[0].type!r}. If 'compaction', the filter is broken."
    )
    assert history[0].data.content[0]["text"] == "msg after compact", (
        "Expected the post-compaction user message."
    )

    # last_seen must advance past the compaction item so it's
    # not re-fetched on the next sync.
    assert new_last_seen is not None, "last_seen must be advanced."
    assert new_last_seen != first_id, (
        "last_seen must advance past the compaction item, not stay at the first message."
    )


# ── _recover_spawn_state ──────────────────────────────────


def _make_conversation_item(
    item_id: str,
    item_type: str,
    data: MessageData | FunctionCallData | FunctionCallOutputData,
) -> ConversationItem:
    """
    Build a ConversationItem with minimal required fields.

    :param item_id: Store-assigned item ID, e.g. ``"msg_001"``.
    :param item_type: Item type, e.g. ``"function_call"``.
    :param data: The typed data payload.
    :returns: A ConversationItem ready for use in tests.
    """
    return ConversationItem(
        id=item_id,
        type=item_type,
        status="completed",
        response_id="resp_test",
        created_at=1000,
        data=data,
    )


def test_recover_spawn_state_from_spawn_output() -> None:
    """
    ``_recover_spawn_state`` extracts spawned IDs from
    ``spawn_sub_agents`` function_call_output items.

    A function_call with name ``spawn_sub_agents`` paired with
    a function_call_output whose JSON contains ``response_ids``
    populates ``spawned_ids``.

    If the spawn output is absent or malformed, spawned_ids
    should be empty — the function must not crash.
    """
    history = [
        # function_call: spawn_sub_agents
        _make_conversation_item(
            "fc_1",
            "function_call",
            FunctionCallData(
                agent="parent",
                name="spawn_sub_agents",
                arguments='{"agents": [{"name": "worker"}]}',
                call_id="call_spawn_1",
            ),
        ),
        # function_call_output: spawn result with response_ids
        _make_conversation_item(
            "fco_1",
            "function_call_output",
            FunctionCallOutputData(
                call_id="call_spawn_1",
                output='{"response_ids": ["resp_sub_A", "resp_sub_B"]}',
            ),
        ),
    ]

    spawned, collected = _recover_spawn_state(history)

    # 2 sub-agent IDs from the spawn output.
    assert spawned == {"resp_sub_A", "resp_sub_B"}, (
        f"Expected spawned IDs from spawn output, got {spawned}. "
        f"If empty, the function failed to parse the "
        f"spawn_sub_agents output JSON."
    )
    # No collect calls → nothing collected.
    assert collected == set(), f"Expected no collected IDs, got {collected}."


def test_recover_spawn_state_from_check_output() -> None:
    """
    ``_recover_spawn_state`` extracts collected IDs from
    ``check_sub_agents`` function_call_output items.

    A function_call with name ``check_sub_agents`` paired with
    a function_call_output whose JSON contains ``results`` with
    terminal statuses populates ``collected_ids``.
    """
    history = [
        # spawn call + output
        _make_conversation_item(
            "fc_1",
            "function_call",
            FunctionCallData(
                agent="parent",
                name="spawn_sub_agents",
                arguments='{"agents": [{"name": "w1"}]}',
                call_id="call_spawn",
            ),
        ),
        _make_conversation_item(
            "fco_1",
            "function_call_output",
            FunctionCallOutputData(
                call_id="call_spawn",
                output='{"response_ids": ["resp_sub_1"]}',
            ),
        ),
        # check call + output with terminal status
        _make_conversation_item(
            "fc_2",
            "function_call",
            FunctionCallData(
                agent="parent",
                name="check_sub_agents",
                arguments='{"response_ids": ["resp_sub_1"]}',
                call_id="call_check",
            ),
        ),
        _make_conversation_item(
            "fco_2",
            "function_call_output",
            FunctionCallOutputData(
                call_id="call_check",
                output=('{"results": [{"response_id": "resp_sub_1", "status": "completed"}]}'),
            ),
        ),
    ]

    spawned, collected = _recover_spawn_state(history)

    assert spawned == {"resp_sub_1"}, f"Expected 1 spawned ID, got {spawned}."
    # check_sub_agents reported "completed" → collected.
    assert collected == {"resp_sub_1"}, (
        f"Expected resp_sub_1 in collected (terminal status), "
        f"got {collected}. If empty, the function failed to "
        f"parse the check_sub_agents output."
    )


def test_recover_spawn_state_from_auto_collect_message() -> None:
    """
    ``_recover_spawn_state`` extracts collected IDs from
    auto-collected system messages.

    Auto-collect persists results as user messages with prefix
    ``[System: auto-collected sub-agent results]`` followed by
    a JSON line. The function parses these to populate
    ``collected_ids``.
    """
    auto_collect_text = (
        '[System: auto-collected sub-agent results]\n{"results": [{"response_id": "resp_auto_1"}]}'
    )
    history = [
        # spawn call + output
        _make_conversation_item(
            "fc_1",
            "function_call",
            FunctionCallData(
                agent="parent",
                name="spawn_sub_agents",
                arguments='{"agents": [{"name": "w"}]}',
                call_id="call_sp",
            ),
        ),
        _make_conversation_item(
            "fco_1",
            "function_call_output",
            FunctionCallOutputData(
                call_id="call_sp",
                output='{"response_ids": ["resp_auto_1"]}',
            ),
        ),
        # Auto-collected system message
        _make_conversation_item(
            "msg_auto",
            "message",
            MessageData(
                role="user",
                content=[{"type": "input_text", "text": auto_collect_text}],
            ),
        ),
    ]

    spawned, collected = _recover_spawn_state(history)

    assert spawned == {"resp_auto_1"}, f"Expected 1 spawned ID, got {spawned}."
    # Auto-collect message → collected.
    assert collected == {"resp_auto_1"}, (
        f"Expected resp_auto_1 collected via auto-collect "
        f"message, got {collected}. If empty, the function "
        f"failed to parse the system message prefix."
    )


def test_recover_spawn_state_partial_collection() -> None:
    """
    ``_recover_spawn_state`` correctly handles partial
    collection: some sub-agents collected, others still running.

    This is the common case when steering interrupts
    auto-collect before all sub-agents finish.
    """
    history = [
        # spawn: 2 sub-agents
        _make_conversation_item(
            "fc_1",
            "function_call",
            FunctionCallData(
                agent="parent",
                name="spawn_sub_agents",
                arguments='{"agents": [{"name": "a"}, {"name": "b"}]}',
                call_id="call_sp",
            ),
        ),
        _make_conversation_item(
            "fco_1",
            "function_call_output",
            FunctionCallOutputData(
                call_id="call_sp",
                output='{"response_ids": ["resp_A", "resp_B"]}',
            ),
        ),
        # Only resp_A was auto-collected (steering interrupted).
        _make_conversation_item(
            "msg_auto",
            "message",
            MessageData(
                role="user",
                content=[
                    {
                        "type": "input_text",
                        "text": (
                            "[System: auto-collected sub-agent results]\n"
                            '{"results": [{"response_id": "resp_A"}]}'
                        ),
                    }
                ],
            ),
        ),
    ]

    spawned, collected = _recover_spawn_state(history)

    # Both spawned.
    assert spawned == {"resp_A", "resp_B"}, f"Expected both sub-agent IDs spawned, got {spawned}."
    # Only resp_A collected — resp_B is still uncollected.
    # This triggers a second auto-collect cycle in the agent loop.
    assert collected == {"resp_A"}, (
        f"Expected only resp_A collected (partial), got "
        f"{collected}. If both are collected, the function "
        f"is incorrectly marking uncollected sub-agents."
    )


def test_recover_spawn_state_empty_history() -> None:
    """
    ``_recover_spawn_state`` returns empty sets for history
    with no spawn-related items.
    """
    history = [
        _make_conversation_item(
            "msg_1",
            "message",
            MessageData(
                role="user",
                content=[{"type": "input_text", "text": "Hello"}],
            ),
        ),
    ]

    spawned, collected = _recover_spawn_state(history)

    assert spawned == set(), f"Expected no spawned IDs for non-spawn history, got {spawned}."
    assert collected == set(), f"Expected no collected IDs for non-spawn history, got {collected}."


# ── Executor storage ──────────────────────────────────────


def _populate_executor_dir(base: Path) -> Path:
    """
    Create a storage dir with nested files for round-trip tests.

    :param base: Parent directory to create source dir in.
    :returns: Path to the populated source directory.
    """
    source = base / "source"
    source.mkdir()
    (source / "session.json").write_text('{"turn": 1}')
    claude_dir = source / ".claude"
    claude_dir.mkdir()
    (claude_dir / "transcript.txt").write_text("Hello world")
    return source


def test_executor_storage_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Persist and restore executor storage via the artifact store.

    The restored files must match the originals, including
    nested directories like ``.claude/``.
    """
    from agent_plane.stores.artifact_store.local import (
        LocalArtifactStore,
    )

    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.get_artifact_store",
        lambda: store,
    )

    source = _populate_executor_dir(tmp_path)
    conv_id = "conv_test_123"
    _persist_executor_storage(conv_id, source)
    restored = _restore_executor_storage(conv_id)

    assert (restored / "session.json").read_text() == '{"turn": 1}', (
        "session.json content lost during round-trip."
    )
    assert (restored / ".claude" / "transcript.txt").read_text() == "Hello world", (
        ".claude/transcript.txt lost — nested dirs must be preserved."
    )

    _cleanup_executor_storage(restored)
    assert not restored.exists(), "cleanup should remove the temp directory."


def test_executor_storage_fresh_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    First conversation turn — no snapshot in artifact store.

    ``_restore_executor_storage`` must return an empty temp
    directory without raising.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.get_artifact_store",
        lambda: None,
    )

    storage_dir = _restore_executor_storage("conv_new")

    # Directory exists and is empty.
    assert storage_dir.exists()
    assert list(storage_dir.iterdir()) == [], "Fresh storage dir should be empty."

    _cleanup_executor_storage(storage_dir)
    assert not storage_dir.exists(), "cleanup should remove the temp directory."


def test_executor_storage_skip_persist_when_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Empty storage directory — ``_persist_executor_storage`` must
    skip the upload to avoid storing empty tarballs.
    """
    from agent_plane.stores.artifact_store.local import (
        LocalArtifactStore,
    )

    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.get_artifact_store",
        lambda: store,
    )

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    _persist_executor_storage("conv_empty", empty_dir)

    # No artifact should have been created.
    assert not store.exists("executor_storage/conv_empty.tar.gz"), (
        "Empty directories should not produce an artifact."
    )
