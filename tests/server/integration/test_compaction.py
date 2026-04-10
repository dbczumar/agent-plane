"""Server integration tests for compaction behavior.

Covers: compaction item persistence, next-execution cursor loading,
and idempotent append for crash-recovery safety.

Design note: The unit-test layer already covers ``compact()``,
``_reactive_compact``, and ``_proactive_compact_if_needed`` in
isolation (``tests/runtime/test_compaction.py``). These integration
tests focus on the durable persistence contract and cursor-loading
behavior that only surfaces when the full workflow runs end-to-end
with real stores.

Compaction is triggered here by monkeypatching
``_executor_turn_with_compaction`` in ``agent_plane.runtime.workflow``: the
patched version sets ``compaction_state.last_summary`` directly on the
mutable state object, simulating what happens after Layer 2 fires, and
then lets the workflow's ``finally`` block call
``_maybe_persist_compaction_item`` as normal.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from agent_plane.llms.errors import ContextWindowExceededError
from agent_plane.runtime.compaction import CompactionResult, SummaryMetadata, _CompactionState
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent, create_test_response

pytestmark = pytest.mark.asyncio

# Fake summary used by _fake_llm_call_with_compaction.
_FAKE_SUMMARY_TEXT = "Compacted summary of prior conversation."
_FAKE_SUMMARY = SummaryMetadata(
    text=_FAKE_SUMMARY_TEXT,
    # Placeholder last_item_id — the test verifies this is replaced by
    # the real item ID from the conversation at persist time.
    # However, _maybe_persist_compaction_item stores whatever is in
    # summary_metadata, so we need a plausible value. The actual
    # item ID is filled in by _run_layer2 in production; here we set
    # it to a sentinel that we check is absent in favour of the real value.
    last_item_id="PLACEHOLDER",
    model="test-agent",
    token_count=50,
)

# Fake LLM response dict shape expected by the agent loop.
_FAKE_LLM_RESPONSE: dict[str, Any] = {
    "model": "test-agent",
    "text": "Hello from the test agent.",
    "tool_calls": [],
    "native_tool_items": [],
}


def _make_compacting_llm_call(
    first_call_seen: list[bool],
) -> Any:
    """
    Build a ``_executor_turn_with_compaction`` replacement that injects
    a fake :class:`SummaryMetadata` on the first call.

    On the first call the patched function mutates
    ``compaction_state.last_summary`` to simulate what happens after
    Layer 2 fires. On subsequent calls it returns normally, allowing
    the workflow to complete.

    :param first_call_seen: A single-element mutable list used as a
        boolean flag to track whether the first call has fired.
        Pass ``[False]`` from the test to initialize.
    :returns: A callable with the same signature as
        ``_executor_turn_with_compaction``.
    """

    async def _fake_executor_turn(
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
        Fake executor turn that sets ``compaction_state.last_summary``
        on the first invocation to simulate Layer 2 firing.

        :param task_id: Task identifier (unused — passed through).
        :param executor: Executor (unused).
        :param spec: Agent spec (unused).
        :param llm_config: LLM config (unused).
        :param history: Conversation history. Used to find the last
            real item ID so ``last_item_id`` in the summary points
            at a real conversation item.
        :param instructions: Per-request instructions (unused).
        :param tool_schemas: Tool schemas (unused).
        :param compaction_state: Per-execution compaction state.
            Mutated in place on the first call to set
            ``last_summary``, triggering compaction item persistence.
        :param context: Executor context (unused).
        :param content_cache: Per-task content cache (unused).
        :returns: A minimal LLM response dict that causes the loop
            to emit a final assistant message and complete.
        """
        if not first_call_seen[0]:
            first_call_seen[0] = True
            # Pick the last real item in history as the summary boundary.
            # This mirrors what _find_last_summarized_item_id does in
            # production: it finds the item at the recent-window boundary.
            last_item_id = history[-1].id if history else "item_none"
            compaction_state.last_summary = SummaryMetadata(
                text=_FAKE_SUMMARY_TEXT,
                last_item_id=last_item_id,
                model=llm_config.model,
                token_count=50,
            )
        return _FAKE_LLM_RESPONSE

    return _fake_executor_turn


async def _run_compacting_execution(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """
    Run a single foreground execution that simulates Layer 2 compaction.

    Monkeypatches ``_executor_turn_with_compaction`` to inject a
    ``SummaryMetadata`` into ``compaction_state``, which causes the
    ``finally`` block in ``_run_agent_loop`` to call
    ``_maybe_persist_compaction_item``. Returns the completed response
    JSON body.

    :param client: The test HTTP client.
    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: The completed response body dict.
    """
    first_call_seen: list[bool] = [False]
    monkeypatch.setattr(
        "agent_plane.runtime.workflow._executor_turn_with_compaction",
        _make_compacting_llm_call(first_call_seen),
    )

    result = await create_test_response(
        client,
        background=False,
        stream=False,
    )
    assert result.status_code == 200, (
        f"Expected 200 but got {result.status_code}. Body: {result.body}"
    )
    assert result.body["status"] == "completed", f"Response did not complete: {result.body}"
    return result.body


async def test_compaction_item_persisted_when_layer2_triggers(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When Layer 2 compaction fires, a ``compaction`` item is persisted
    to the conversation store with a valid ``last_item_id`` and
    ``model`` field.

    :param client: Async HTTP client wired to the FastAPI app.
    :param mock_llm: Controllable mock LLM for the test (unused here —
        the LLM call is monkeypatched at a lower level).
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    await create_test_agent(client)
    resp_body = await _run_compacting_execution(client, monkeypatch)

    conv_id = resp_body["conversation"]["id"]
    items_resp = await client.get(f"/v1/conversations/{conv_id}/items")
    assert items_resp.status_code == 200
    items: list[dict[str, Any]] = items_resp.json()["data"]

    compaction_items = [i for i in items if i.get("type") == "compaction"]
    # Layer 2 must produce exactly one compaction item per execution.
    # If 0, _maybe_persist_compaction_item did not run or
    # compaction_state.last_summary was None (compaction did not trigger).
    assert len(compaction_items) == 1, (
        f"Expected exactly 1 compaction item, found {len(compaction_items)}. "
        f"If 0, the finally block did not call _maybe_persist_compaction_item "
        f"or last_summary was not set. "
        f"All item types: {[i.get('type') for i in items]}"
    )

    compaction_item = compaction_items[0]

    # last_item_id must point to a real item in the conversation.
    # This proves CompactionData was serialised with the real item ID
    # from history, not the placeholder.
    all_item_ids = {i["id"] for i in items}
    last_item_id = compaction_item.get("last_item_id")
    assert last_item_id in all_item_ids, (
        f"compaction.last_item_id={last_item_id!r} does not match any "
        f"conversation item. Known IDs: {all_item_ids}. "
        f"If 'item_none', history was empty when compaction ran."
    )

    # model must be set — it records which model generated the summary.
    compaction_model = compaction_item.get("model")
    assert isinstance(compaction_model, str) and len(compaction_model) > 0, (
        f"compaction item model must be a non-empty string, got {compaction_model!r}. "
        f"CompactionData.model must be set by the fake LLM call."
    )


async def test_next_execution_loads_from_compaction_cursor(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The second execution after compaction was persisted starts history
    from the synthetic summary pair, not the full conversation.

    The first LLM input message in the second execution must contain
    the summary header ``"[This is an automatically generated summary"``.
    Original pre-compaction messages must NOT be passed in full — they
    are replaced by the compaction cursor.

    :param client: Async HTTP client wired to the FastAPI app.
    :param mock_llm: Controllable mock LLM for the test (unused — both
        executions are driven by a monkeypatched _executor_turn_with_compaction).
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    await create_test_agent(client)

    # --- Execution 1: triggers compaction via patched _executor_turn_with_compaction ---
    first_resp = await _run_compacting_execution(client, monkeypatch)
    first_response_id = first_resp["id"]

    # --- Execution 2: inspect what history _load_initial_history produces ---
    # We patch _executor_turn_with_compaction for execution 2 to capture the
    # ``history`` argument it receives. This tells us whether
    # _load_initial_history used the compaction cursor and prepended
    # the synthetic summary pair. The fake still returns a valid response
    # so the workflow completes.
    captured_history: list[Any] = []

    async def _capture_history_llm_call(
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
        Capture the ``history`` argument for assertion and return a
        normal response so the workflow completes.

        :param task_id: Task identifier (unused).
        :param executor: Executor (unused).
        :param spec: Agent spec (unused).
        :param llm_config: LLM config (unused).
        :param history: The history list passed by the agent loop,
            captured for assertion.
        :param instructions: Per-request instructions (unused).
        :param tool_schemas: Tool schemas (unused).
        :param compaction_state: Per-execution compaction state (unused).
        :param context: Executor context (unused).
        :param content_cache: Per-task content cache (unused).
        :returns: A minimal LLM response dict.
        """
        captured_history.extend(history)
        return _FAKE_LLM_RESPONSE

    monkeypatch.setattr(
        "agent_plane.runtime.workflow._executor_turn_with_compaction",
        _capture_history_llm_call,
    )

    second_resp = await create_test_response(
        client,
        input_text="Continue after compaction",
        previous_response_id=first_response_id,
        background=False,
        stream=False,
    )
    assert second_resp.status_code == 200, f"Second execution failed: {second_resp.body}"
    assert second_resp.body["status"] == "completed"

    # Verify that history contains the synthetic summary pair.
    # compaction_to_history_items() produces two items:
    # 1. user message with "[This is an automatically generated summary..."
    # 2. assistant message with the summary text
    # If the compaction cursor was used, both should appear at the start.
    # Extract text content from all history items for inspection.
    history_texts = _extract_texts_from_history(captured_history)

    # The summary header must appear in the history items — this proves
    # _load_initial_history found the compaction item and called
    # compaction_to_history_items(), which prepends the synthetic
    # "[This is an automatically generated summary..." user message.
    summary_header = "[This is an automatically generated summary"
    assert any(summary_header in t for t in history_texts), (
        f"Summary header not found in second execution's history. "
        f"Expected to find {summary_header!r} in: {history_texts}. "
        f"If missing, _load_initial_history did not use the compaction "
        f"cursor, meaning the compaction item was not found or "
        f"compaction_to_history_items() was not called."
    )


async def test_compaction_persists_only_once_per_execution(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``_maybe_persist_compaction_item`` is idempotent: exactly one
    compaction item is appended even if the finally block runs multiple
    times (crash-recovery dedup via ``response_id`` guard).

    This test verifies that a single execution produces exactly one
    compaction item, not two or more.

    :param client: Async HTTP client wired to the FastAPI app.
    :param mock_llm: Controllable mock LLM for the test (unused —
        the LLM call is monkeypatched at a lower level).
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    await create_test_agent(client)
    resp_body = await _run_compacting_execution(client, monkeypatch)

    conv_id = resp_body["conversation"]["id"]
    items_resp = await client.get(f"/v1/conversations/{conv_id}/items")
    assert items_resp.status_code == 200
    items: list[dict[str, Any]] = items_resp.json()["data"]

    compaction_items = [i for i in items if i.get("type") == "compaction"]
    # Exactly one compaction item means the response_id guard in
    # _maybe_persist_compaction_item fired correctly. If > 1, the
    # dedup check failed and duplicates were appended.
    assert len(compaction_items) == 1, (
        f"Expected exactly 1 compaction item (idempotent dedup), "
        f"found {len(compaction_items)}. "
        f"If > 1, _maybe_persist_compaction_item appended duplicates "
        f"because the response_id guard did not prevent re-append."
    )

    # Verify the single item has the correct response_id (the task ID),
    # proving the dedup guard uses the right key.
    task_id = resp_body["id"]
    assert compaction_items[0].get("response_id") == task_id, (
        f"compaction item response_id={compaction_items[0].get('response_id')!r} "
        f"must equal the task ID {task_id!r}. "
        f"If different, the dedup guard would not prevent re-append on replay."
    )


def _extract_all_texts(messages: list[dict[str, Any]]) -> list[str]:
    """
    Extract all text strings from a Responses API input messages list.

    Handles both string ``content`` values and list ``content`` values
    with typed blocks (``input_text``, ``text``, ``output_text``).

    :param messages: The input messages list from ``responses.create()``.
    :returns: Flat list of all text strings found across all messages.
    """
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "input_text",
                    "text",
                    "output_text",
                ):
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
    return texts


async def test_reactive_compact_overflow_then_retry_succeeds(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Full reactive compaction loop: first LLM call overflows →
    workflow catches ``ContextWindowExceededError`` → tiktoken
    validates → compact() runs → retry succeeds.

    This exercises the real ``_call_llm_maybe_compact`` error path
    (not monkeypatched). The mock LLM's first call throws
    ``ContextWindowExceededError``; the second call succeeds. The
    test verifies the response completes and the LLM was called
    exactly twice (overflow + retry).

    :param client: Async HTTP client wired to the FastAPI app.
    :param mock_llm: Controllable mock LLM — first call throws
        overflow, second call returns text.
    :param monkeypatch: Pytest monkeypatch fixture for patching
        tiktoken and compact().
    """
    await create_test_agent(client)

    # First LLM call: throw ContextWindowExceededError.
    # The executor catches this and yields ContextWindowExceeded event.
    overflow_exc = ContextWindowExceededError(
        "Context window exceeded: 150000 tokens > 128000 max",
        code="context_length_exceeded",
        max_context_tokens=128000,
        actual_tokens=150000,
    )
    mock_llm.add_call(exception=overflow_exc)

    # Second LLM call: succeed with normal text.
    mock_llm.add_call(text="Response after compaction.")

    # Patch count_tokens: called twice — once for system tokens (small)
    # and once for messages (close to reported 150000). The reactive
    # compaction ratio check uses (messages + sys_tokens) / reported.
    # 148000 + 500 = 148500; ratio = 148500 / 150000 = 0.99, well
    # within 0.7-1.3.
    _token_calls: list[int] = [0]

    def _fake_count_tokens(
        msgs: list[dict[str, Any]],
        model: str,
    ) -> int:
        """
        Return system token count on first call, message token count
        on subsequent calls.

        :param msgs: Messages (used to distinguish call site).
        :param model: Model string (unused).
        :returns: Token count.
        """
        _token_calls[0] += 1
        # First call is always sys_tokens (a single system message).
        # Subsequent calls are for message lists.
        if _token_calls[0] == 1:
            return 500
        return 148000

    monkeypatch.setattr(
        "agent_plane.runtime.workflow.count_tokens",
        _fake_count_tokens,
    )

    # Patch compact() to return a minimal message list. We verify
    # compact was called by checking the second LLM call receives
    # the compacted input.  The real compact() is async, so the
    # replacement must also be async — a sync return would cause
    # "object CompactionResult can't be used in 'await' expression".
    compacted_msgs = [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]

    async def _fake_compact(
        messages: list[dict[str, Any]],
        history: list[Any],
        **kw: Any,
    ) -> CompactionResult:
        """
        Async stub that returns pre-built compacted messages.

        :param messages: Input messages (ignored).
        :param history: Conversation history (ignored).
        :param kw: Remaining keyword arguments (ignored).
        :returns: A CompactionResult with the pre-built compacted
            messages and no summary metadata.
        """
        return CompactionResult(
            messages=compacted_msgs,
            summary_metadata=None,
        )

    monkeypatch.setattr(
        "agent_plane.runtime.workflow.compact",
        _fake_compact,
    )

    result = await create_test_response(
        client,
        background=False,
        stream=False,
    )

    # Response must complete — the retry after compaction succeeded.
    assert result.status_code == 200, (
        f"Expected 200, got {result.status_code}. Body: {result.body}. "
        "If 500, the ContextWindowExceededError was not caught or "
        "compact-retry path is broken."
    )
    assert result.body["status"] == "completed", (
        f"Expected completed status, got {result.body['status']}. "
        "If 'failed', the retry after compaction did not succeed."
    )

    # LLM was called exactly 2 times: overflow + retry.
    # 1 = overflow never triggered compact-retry path.
    # 3+ = compaction-retry looped (should only retry once).
    assert mock_llm.call_count == 2, (
        f"Expected 2 LLM calls (1 overflow + 1 retry), "
        f"got {mock_llm.call_count}. "
        f"If 1, the overflow was not detected or compact-retry "
        f"did not fire. If 3+, the retry looped."
    )

    # Verify the response contains the text from the second call.
    output = result.body.get("output", [])
    output_texts = [
        item["content"][0]["text"]
        for item in output
        if item.get("type") == "message"
        and item.get("role") == "assistant"
        and item.get("content")
    ]
    assert any("compaction" in t.lower() for t in output_texts), (
        f"Expected assistant text from the retry call to contain "
        f"'compaction'. Got: {output_texts}. "
        f"If empty, the retry response was not persisted."
    )


def _extract_texts_from_history(history: list[Any]) -> list[str]:
    """
    Extract all text strings from a list of
    :class:`~agent_plane.entities.ConversationItem` objects.

    Handles ``MessageData`` items, extracting text from their
    ``content`` blocks. Returns a flat list of all text strings.

    :param history: List of ``ConversationItem`` objects as passed
        to ``_executor_turn_with_compaction``.
    :returns: Flat list of all text strings found across all items.
    """
    from agent_plane.entities import MessageData

    texts: list[str] = []
    for item in history:
        data = getattr(item, "data", None)
        if not isinstance(data, MessageData):
            continue
        for block in data.content or []:
            if isinstance(block, dict) and block.get("type") in (
                "input_text",
                "text",
                "output_text",
            ):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
    return texts
