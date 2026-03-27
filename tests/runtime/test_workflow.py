"""Tests for agent_plane.runtime.workflow helpers: pagination, execution timeout, tool call splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_plane.entities import MessageData, NewConversationItem
from agent_plane.runtime.caps import RuntimeCaps
from agent_plane.runtime.workflow import (
    _run_agent_loop,
    _split_tool_calls,
    _ToolCall,
    fetch_all_items,
)
from agent_plane.spec.types import (
    AgentSpec,
    ExecutionConfig,
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
            timeout=300,
            retry=RetryConfig(max_attempts=1),
        ),
        tools=ToolsConfig(),
        execution=ExecutionConfig(
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

    # Return empty history so the loop starts with no items
    monkeypatch.setattr(
        "agent_plane.runtime.workflow.fetch_all_items",
        lambda store, conv_id, after=None: [],
    )

    return emitted_events


def test_execution_timeout_resolution_takes_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The effective execution timeout is
    ``min(spec.execution.timeout, caps.execution_timeout)``.

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
        instructions=None,
        tool_mgr=_stub_tool_manager(),
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
        instructions=None,
        tool_mgr=_stub_tool_manager(),
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
        instructions=None,
        tool_mgr=_stub_tool_manager(),
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

    def _fake_llm_call(
        task_id: str,
        spec: AgentSpec,
        llm_config: LLMConfig,
        history: list[Any],
        instructions: str | None,
        tool_schemas: list[Any],
    ) -> MagicMock:
        """
        Fake LLM call that simulates a tool-call response.

        We return a response with tool calls so the loop enters
        ``_handle_tool_calls``, which we also stub. The important
        thing is that ``output_items`` gets populated before the
        next iteration's timeout check.

        :param task_id: Task identifier (unused).
        :param spec: Agent spec (unused).
        :param llm_config: LLM config (unused).
        :param history: Conversation history (unused).
        :param instructions: Instructions (unused).
        :param tool_schemas: Tool schemas (unused).
        :returns: A MagicMock LLM response with tool calls.
        """
        resp = MagicMock()
        # Signal that this response has tool calls
        resp.output = [MagicMock()]
        return resp

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._call_llm_for_iteration_with_error_handling",
        _fake_llm_call,
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
        instructions=None,
        tool_mgr=_stub_tool_manager(),
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
            timeout=300,
            retry=RetryConfig(max_attempts=1),
        ),
        tools=ToolsConfig(),
        execution=ExecutionConfig(timeout=60, max_iterations=100),
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
        work_dir=Path("/tmp/test"),
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
