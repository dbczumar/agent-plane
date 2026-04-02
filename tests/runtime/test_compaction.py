"""Unit tests for agent_plane.runtime.compaction."""

from __future__ import annotations

from typing import Any

import pytest

from agent_plane.entities import (
    CompactionData,
    ConversationItem,
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
)
from agent_plane.llms.errors import RetryableLLMError
from agent_plane.llms.types import MessageOutput, OutputText, Response
from agent_plane.runtime.compaction import (
    _BINARY_CONTENT_CLEARED,
    _TOOL_RESULT_CLEARED,
    compact,
    compaction_to_history_items,
    count_tokens,
    summarize_history,
)
from agent_plane.spec.types import CompactionConfig

# ---------------------------------------------------------------------------
# LLM client stubs
# ---------------------------------------------------------------------------


class _RaisesIfCalled:
    """
    LLM client stub that fails the test if ``responses.create()`` is ever
    called.

    Use this for ``compact()`` calls where Layer 2 must NOT fire. If the
    production code unexpectedly reaches ``summarize_history``, the
    ``AssertionError`` surfaces immediately rather than silently succeeding
    via a ``MagicMock``.
    """

    class responses:
        """Namespace mirroring the real client's ``responses`` attribute."""

        @staticmethod
        def create(**kwargs: Any) -> None:
            """
            Raise if called — Layer 2 must not have fired.

            :param kwargs: Forwarded kwargs from the real API call.
            :raises AssertionError: Always.
            """
            raise AssertionError(
                "llm_client.responses.create() was called unexpectedly. "
                "Layer 2 must not fire in this test — check that count_tokens "
                "is mocked below budget or that summarize_history is patched."
            )


class _ReturnsTextClient:
    """
    LLM client stub that returns a real ``Response`` containing a fixed text.

    Use this for ``summarize_history`` tests where a real LLM response is
    needed but the test must not hit the network.

    :param text: The assistant text the stub will return, e.g.
        ``"Summary of earlier conversation context."``.
    :param model: The model name to embed in the returned ``Response``, e.g.
        ``"openai/gpt-4o"``.
    """

    def __init__(self, text: str, model: str = "test-model") -> None:
        self._text = text
        self._model = model
        self.call_count = 0

    class _Responses:
        """
        Inner namespace mirroring ``client.responses``.

        :param outer: The enclosing ``_ReturnsTextClient`` instance.
        """

        def __init__(self, outer: _ReturnsTextClient) -> None:
            self._outer = outer

        def create(self, **kwargs: Any) -> Response:
            """
            Return a real ``Response`` with the configured text.

            :param kwargs: Forwarded kwargs from the real API call.
            :returns: A ``Response`` wrapping the configured text.
            """
            self._outer.call_count += 1
            return Response(
                output=[MessageOutput(content=[OutputText(text=self._outer._text)])],
                model=self._outer._model,
            )

    @property
    def responses(self) -> _ReturnsTextClient._Responses:
        """
        Return the ``responses`` namespace for this stub client.

        :returns: The ``_Responses`` inner instance.
        """
        return self._Responses(self)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_conv_item(
    item_id: str,
    item_type: str,
    data: Any,
    response_id: str = "resp_001",
) -> ConversationItem:
    """
    Build a ConversationItem for testing.

    :param item_id: Unique identifier for the item.
    :param item_type: Type string, e.g. "message", "function_call".
    :param data: The item payload (MessageData, FunctionCallData, etc.).
    :param response_id: Response/task identifier to associate with the item.
    """
    return ConversationItem(
        id=item_id,
        type=item_type,
        status="completed",
        response_id=response_id,
        created_at=1000,
        data=data,
    )


def _user_msg(item_id: str, text: str = "User message") -> ConversationItem:
    """
    Build a user-role ConversationItem with a single input_text block.

    :param item_id: Unique identifier for the item.
    :param text: Text content of the user message.
    """
    return _make_conv_item(
        item_id,
        "message",
        MessageData(role="user", content=[{"type": "input_text", "text": text}]),
    )


def _assistant_msg(item_id: str, text: str = "Assistant response") -> ConversationItem:
    """
    Build an assistant-role ConversationItem with a single output_text block.

    :param item_id: Unique identifier for the item.
    :param text: Text content of the assistant response.
    """
    return _make_conv_item(
        item_id,
        "message",
        MessageData(
            role="assistant",
            content=[{"type": "output_text", "text": text}],
            agent="test-model",
        ),
    )


def _fc_item(item_id: str, call_id: str = "call_abc") -> ConversationItem:
    """
    Build a function_call ConversationItem.

    :param item_id: Unique identifier for the item.
    :param call_id: Tool call identifier, e.g. "call_abc".
    """
    return _make_conv_item(
        item_id,
        "function_call",
        FunctionCallData(
            agent="test-model",
            name="my_tool",
            arguments="{}",
            call_id=call_id,
        ),
    )


def _fco_item(
    item_id: str,
    call_id: str = "call_abc",
    output: str = "tool result",
) -> ConversationItem:
    """
    Build a function_call_output ConversationItem.

    :param item_id: Unique identifier for the item.
    :param call_id: Tool call identifier matching the originating function_call.
    :param output: The tool output string.
    """
    return _make_conv_item(
        item_id,
        "function_call_output",
        FunctionCallOutputData(call_id=call_id, output=output),
    )


def _user_msg_dict(text: str = "User message") -> dict[str, Any]:
    """
    Build a user-role message dict for the messages list.

    :param text: Text content of the user message.
    """
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant_msg_dict(text: str = "Assistant response") -> dict[str, Any]:
    """
    Build an assistant-role message dict for the messages list.

    :param text: Text content of the assistant response.
    """
    return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _fc_dict(call_id: str = "call_abc", name: str = "my_tool") -> dict[str, Any]:
    """
    Build a function_call dict for the messages list.

    :param call_id: Tool call identifier.
    :param name: Name of the tool being called.
    """
    return {"type": "function_call", "id": call_id, "name": name, "arguments": "{}"}


def _fco_dict(call_id: str = "call_abc", output: str = "tool result") -> dict[str, Any]:
    """
    Build a function_call_output dict for the messages list.

    :param call_id: Tool call identifier matching the originating function_call.
    :param output: The tool output string.
    """
    return {"type": "function_call_output", "call_id": call_id, "output": output}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_compaction_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch DBOS-dependent side effects so compaction unit tests don't
    need DBOS initialized.

    Patches:
    - ``_emit_compaction_event`` → no-op (avoids write_stream/live_publish DBOS calls)
    """
    monkeypatch.setattr(
        "agent_plane.runtime.compaction._emit_compaction_event",
        lambda task_id: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_compaction_under_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Layer 1 always runs but returns early if token count is within budget."""
    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", lambda msgs, model: 50)
    messages = [_user_msg_dict("hi"), _assistant_msg_dict("hello")]
    history = [_user_msg("msg_001", "hi"), _assistant_msg("msg_002", "hello")]

    result = compact(
        messages,
        history,
        config=None,
        context_window=100000,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # _RaisesIfCalled: Layer 2 must not fire (budget met after Layer 1).
        # If summarize_history() is unexpectedly called, the test fails immediately.
        llm_client=_RaisesIfCalled(),
    )

    # Layer 1 always applies clearing, but since budget is met, returns early.
    # summary_metadata=None proves Layer 2 (summarization) never fired.
    assert result.summary_metadata is None
    # Messages content preserved — no tool result bodies were replaced.
    assert result.messages[0]["content"][0]["text"] == "hi"
    assert result.messages[1]["content"][0]["text"] == "hello"


def test_layer1_clears_tool_results_outside_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Layer 1 replaces function_call_output bodies outside the recent window
    with _TOOL_RESULT_CLEARED, while preserving bodies inside the window.

    ``recent_window=2`` counting backward through [u3,fc3,fco3,a3] and [u2,fc2,fco2,a2]:
    - i=11: a3 → groups=1; i=9: fc3 → groups=2 ≥ 2 → boundary=9.
    - Items 0..8 outside window (eligible for clearing).
    - Items 9..11 inside window (protected: fc3, fco3, a3).
    """
    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", lambda msgs, model: 50)

    history = [
        _user_msg("msg_u1", "iter1"),
        _fc_item("msg_fc1", "c1"),
        _fco_item("msg_fco1", "c1"),
        _assistant_msg("msg_a1"),
        _user_msg("msg_u2", "iter2"),
        _fc_item("msg_fc2", "c2"),
        _fco_item("msg_fco2", "c2"),
        _assistant_msg("msg_a2"),
        _user_msg("msg_u3", "iter3"),
        _fc_item("msg_fc3", "c3"),
        _fco_item("msg_fco3", "c3"),
        _assistant_msg("msg_a3"),
    ]
    messages = [
        _user_msg_dict("iter1"),
        _fc_dict("c1"),
        _fco_dict("c1", "tool result iter1"),
        _assistant_msg_dict(),
        _user_msg_dict("iter2"),
        _fc_dict("c2"),
        _fco_dict("c2", "tool result iter2"),
        _assistant_msg_dict(),
        _user_msg_dict("iter3"),
        _fc_dict("c3"),
        _fco_dict("c3", "tool result iter3"),
        _assistant_msg_dict(),
    ]

    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=2),
        context_window=100000,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # _RaisesIfCalled: token count is within budget so Layer 2 must not
        # fire. Fails loudly if summarize_history() is unexpectedly reached.
        llm_client=_RaisesIfCalled(),
    )

    # fco at index 2 (iter1, outside window) must be cleared.
    assert result.messages[2]["output"] == _TOOL_RESULT_CLEARED, (
        f"Expected iter1 tool result to be cleared (outside window), "
        f"got: {result.messages[2]['output']!r}"
    )
    # fco at index 6 (iter2, outside window) must be cleared.
    assert result.messages[6]["output"] == _TOOL_RESULT_CLEARED, (
        f"Expected iter2 tool result to be cleared (outside window), "
        f"got: {result.messages[6]['output']!r}"
    )
    # fco at index 10 (iter3 — inside window, boundary=9 so index 10 ≥ 9) must be preserved.
    assert result.messages[10]["output"] == "tool result iter3", (
        f"Expected iter3 tool result to be preserved (inside window, boundary=9), "
        f"got: {result.messages[10]['output']!r}"
    )
    # summary_metadata=None confirms only Layer 1 fired (Layer 2 not triggered).
    assert result.summary_metadata is None


def test_layer1_never_touches_user_message_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Layer 1 (tool result clearing) must never modify user message text content,
    even for messages outside the recent window.
    """
    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", lambda msgs, model: 50)

    history = [
        _user_msg("msg_u1", "Important user text outside window"),
        _fc_item("msg_fc1", "c1"),
        _fco_item("msg_fco1", "c1"),
        _assistant_msg("msg_a1"),
        _user_msg("msg_u2", "Another user message inside window"),
        _assistant_msg("msg_a2"),
    ]
    messages = [
        _user_msg_dict("Important user text outside window"),
        _fc_dict("c1"),
        _fco_dict("c1", "tool output"),
        _assistant_msg_dict(),
        _user_msg_dict("Another user message inside window"),
        _assistant_msg_dict(),
    ]

    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=1),
        context_window=100000,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # _RaisesIfCalled: budget is met after Layer 1 so Layer 2 must not
        # fire. Fails loudly if summarize_history() is unexpectedly reached.
        llm_client=_RaisesIfCalled(),
    )

    # User text at index 0 (outside window) must be preserved.
    # Failure here means Layer 1 modified non-tool-result content.
    assert result.messages[0]["content"][0]["text"] == "Important user text outside window"
    # User text at index 4 (inside window) must also be preserved.
    assert result.messages[4]["content"][0]["text"] == "Another user message inside window"


def test_layer1_clears_binary_content_and_preserves_file_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Layer 1 clears image/file block data outside the recent window,
    preserves file_id, and leaves text blocks in the same message untouched.
    """
    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", lambda msgs, model: 50)

    # User message with image block (outside window) + text block
    image_msg = {
        "role": "user",
        "content": [
            {"type": "image", "data": "base64IMAGEDATA==", "file_id": "file_abc123"},
            {"type": "text", "text": "Please describe this image"},
        ],
    }
    history = [
        _user_msg("msg_u1", "user with image"),
        _assistant_msg("msg_a1"),  # boundary (recent_window=1)
    ]
    messages = [image_msg, _assistant_msg_dict()]

    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=1),
        context_window=100000,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # _RaisesIfCalled: budget is met after Layer 1 so Layer 2 must not
        # fire. Fails loudly if summarize_history() is unexpectedly reached.
        llm_client=_RaisesIfCalled(),
    )

    image_block = result.messages[0]["content"][0]
    text_block = result.messages[0]["content"][1]

    # Image data must be cleared — the binary payload was replaced.
    assert image_block["data"] == _BINARY_CONTENT_CLEARED, (
        f"Expected image data to be cleared, got: {image_block['data']!r}"
    )
    # file_id must be preserved so the agent can re-fetch the image.
    assert image_block["file_id"] == "file_abc123", (
        f"Expected file_id 'file_abc123' preserved, got: {image_block['file_id']!r}"
    )
    # Text block in the same message must be untouched.
    assert text_block["text"] == "Please describe this image"


def test_layer1_binary_content_inside_window_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary content inside the recent window must not be cleared by Layer 1."""
    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", lambda msgs, model: 50)

    image_msg_outside = {
        "role": "user",
        "content": [{"type": "image", "data": "OLD_DATA==", "file_id": "file_old"}],
    }
    image_msg_inside = {
        "role": "user",
        "content": [{"type": "image", "data": "NEW_DATA==", "file_id": "file_new"}],
    }
    # With recent_window=2 and history [u1, a1, u2, a2]:
    # i=3: a2 → groups=1; i=1: a1 → groups=2 ≥ 2 → boundary=1.
    # Items 0 outside window (image_msg_outside, messages[0]).
    # Items 1..3 inside window (a1, image_msg_inside at msg index 2, a2).
    history = [
        _user_msg("msg_u1"),
        _assistant_msg("msg_a1"),
        _user_msg("msg_u2"),
        _assistant_msg("msg_a2"),
    ]
    messages = [image_msg_outside, _assistant_msg_dict(), image_msg_inside, _assistant_msg_dict()]

    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=2),
        context_window=100000,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # _RaisesIfCalled: budget is met after Layer 1 so Layer 2 must not fire.
        llm_client=_RaisesIfCalled(),
    )

    # The image OUTSIDE the window (index 0 < boundary=1) should be cleared.
    assert result.messages[0]["content"][0]["data"] == _BINARY_CONTENT_CLEARED
    # The image INSIDE the window (index 2 ≥ boundary=1) must be untouched.
    assert result.messages[2]["content"][0]["data"] == "NEW_DATA==", (
        "Image inside recent window must not be cleared by Layer 1."
    )


@pytest.mark.parametrize(
    ("recent_window", "outside_fco_idx", "protected_fco_idx"),
    [
        # recent_window=2: boundary at index 17 (fc17). fco18 protected; fco14 outside.
        (2, 14, 18),
        # recent_window=3: boundary at index 15 (a15). fco18 protected; fco10 outside.
        (3, 10, 18),
        # recent_window=4: boundary at index 13 (fc13). fco18 protected; fco6 outside.
        (4, 6, 18),
    ],
    ids=["window-2", "window-3", "window-4"],
)
def test_recent_window_boundary_parametrized(
    monkeypatch: pytest.MonkeyPatch,
    recent_window: int,
    outside_fco_idx: int,
    protected_fco_idx: int,
) -> None:
    """
    Items inside the recent window must never be modified;
    items outside must have their tool result bodies cleared.
    """
    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", lambda msgs, model: 50)

    history = []
    messages = []
    for i in range(5):
        call_id = f"c{i}"
        history.extend(
            [
                _user_msg(f"msg_u{i}"),
                _fc_item(f"msg_fc{i}", call_id),
                _fco_item(f"msg_fco{i}", call_id, f"output_iter_{i}"),
                _assistant_msg(f"msg_a{i}"),
            ]
        )
        messages.extend(
            [
                _user_msg_dict(),
                _fc_dict(call_id),
                _fco_dict(call_id, f"output_iter_{i}"),
                _assistant_msg_dict(),
            ]
        )

    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=recent_window),
        context_window=100000,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # _RaisesIfCalled: count_tokens is mocked below budget, so Layer 2
        # must not fire. Fails loudly if summarize_history() is called.
        llm_client=_RaisesIfCalled(),
    )

    outside_output = result.messages[outside_fco_idx]["output"]
    protected_output = result.messages[protected_fco_idx]["output"]

    # Tool result outside the recent window must be cleared.
    assert outside_output == _TOOL_RESULT_CLEARED, (
        f"fco at index {outside_fco_idx} should be cleared (outside window={recent_window}), "
        f"got: {outside_output!r}"
    )
    # Tool result inside the recent window must be preserved.
    assert protected_output != _TOOL_RESULT_CLEARED, (
        f"fco at index {protected_fco_idx} should be preserved (inside window={recent_window}), "
        f"but was cleared"
    )


def test_layer2_triggers_when_layer1_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When Layer 1 alone is insufficient (token count still above budget),
    Layer 2 (LLM summarization) is triggered.
    """
    call_counts = [0]

    def mock_count_tokens(msgs: list[dict[str, Any]], model: str) -> int:
        """
        Return above-budget on the first call to force Layer 2, then below-budget
        for all subsequent calls so Layer 2 can succeed.
        """
        call_counts[0] += 1
        # First call (after Layer 1): above budget → trigger Layer 2
        # Second call (inside _run_layer2): check if to_summarize too large
        # Third call (after Layer 2): summary + recent fits budget
        if call_counts[0] == 1:
            return 10001  # above budget=10000 → Layer 2 needed
        return 50  # all subsequent calls: below budget

    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", mock_count_tokens)
    monkeypatch.setattr(
        "agent_plane.runtime.compaction.summarize_history",
        lambda msgs, llm_client, model: {
            "text": "Summary of earlier conversation",
            "token_count": 50,
        },
    )

    # 2 iterations; recent_window=1 → boundary at index 7 (last assistant)
    history = [
        _user_msg("msg_u1"),
        _fc_item("msg_fc1", "c1"),
        _fco_item("msg_fco1", "c1"),
        _assistant_msg("msg_a1"),
        _user_msg("msg_u2"),
        _fc_item("msg_fc2", "c2"),
        _fco_item("msg_fco2", "c2"),
        _assistant_msg("msg_a2"),
    ]
    messages = [
        _user_msg_dict(),
        _fc_dict("c1"),
        _fco_dict("c1"),
        _assistant_msg_dict(),
        _user_msg_dict(),
        _fc_dict("c2"),
        _fco_dict("c2"),
        _assistant_msg_dict(),
    ]

    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=1),
        context_window=12500,  # budget = int(12500*0.8) = 10000
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # summarize_history is monkeypatched above so llm_client is never used.
        # _RaisesIfCalled still catches any accidental bypass of the patch.
        llm_client=_RaisesIfCalled(),
    )

    # summary_metadata being set proves Layer 2 fired successfully.
    assert result.summary_metadata is not None, (
        "Layer 2 should have triggered and set summary_metadata, "
        "but it is None — check that mock count_tokens returns > budget on first call."
    )
    # The summary text must match what summarize_history returned.
    assert result.summary_metadata.text == "Summary of earlier conversation"
    # last_item_id must point to a real history item before the boundary.
    # boundary=7 (last assistant) → last summarized item = msg_fco2 at index 6 (non-synthetic).
    # Actually: _find_last_summarized_item_id looks for last non-synthetic item before boundary.
    # With recent_window=1, boundary=7, last item before boundary is msg_fco2 at index 6.
    # Wait - boundary=7 means items 7+ are protected. Items 0..6 are summarized.
    # _find_last_summarized_item_id(history, boundary=7) → history[6] = msg_fco2.
    assert result.summary_metadata.last_item_id == "msg_fco2", (
        f"last_item_id should point to the last item before the boundary, "
        f"got: {result.summary_metadata.last_item_id!r}"
    )
    # The compacted messages should start with the synthetic summary pair
    # (user + assistant messages from _summary_to_messages).
    assert result.messages[0]["role"] == "user"
    assert "automatically generated summary" in result.messages[0]["content"]
    assert result.messages[1]["role"] == "assistant"
    assert result.messages[1]["content"] == "Summary of earlier conversation"


def test_layer2_failure_falls_back_to_layer3(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When Layer 2 summarization fails, compact() falls back to Layer 3
    (truncation) without raising. summary_metadata is None.
    """
    # First 2 calls above budget (trigger Layer 2); subsequent calls below budget
    # so Layer 3 stops truncating after one pass (not emptying the list).
    call_idx = [0]

    def mock_count_tokens(msgs: list[dict[str, Any]], model: str) -> int:
        """
        Return above-budget on the first 2 calls to trigger Layer 2,
        then below-budget so Layer 3 terminates with remaining messages.
        """
        call_idx[0] += 1
        return 10001 if call_idx[0] <= 2 else 50

    monkeypatch.setattr("agent_plane.runtime.compaction.count_tokens", mock_count_tokens)

    def _raise_retryable(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Raise RetryableLLMError to simulate an unavailable LLM."""
        raise RetryableLLMError("LLM unavailable", code="503")

    monkeypatch.setattr(
        "agent_plane.runtime.compaction.summarize_history",
        _raise_retryable,
    )

    history = [
        _user_msg("msg_u1"),
        _assistant_msg("msg_a1"),
        _user_msg("msg_u2"),
        _assistant_msg("msg_a2"),
    ]
    messages = [
        _user_msg_dict("first"),
        _assistant_msg_dict(),
        _user_msg_dict("second"),
        _assistant_msg_dict(),
    ]

    # Must not raise even though summarize_history fails.
    result = compact(
        messages,
        history,
        config=CompactionConfig(trigger_threshold=0.8, recent_window=1),
        context_window=12500,
        system_token_budget=0,
        model="openai/gpt-4o",
        task_id="task_001",
        # summarize_history is monkeypatched to raise before reaching llm_client.
        # _RaisesIfCalled catches any accidental bypass of the monkeypatch.
        llm_client=_RaisesIfCalled(),
    )

    # summary_metadata=None proves Layer 2 failed (not persisted).
    assert result.summary_metadata is None, (
        "summary_metadata must be None when Layer 2 summarization fails — "
        "it is only set on successful summarization."
    )
    # Some messages must have been returned (Layer 3 truncated, not emptied).
    assert len(result.messages) > 0


def test_summarize_history_returns_text_and_token_count() -> None:
    """summarize_history calls the LLM and returns text + token_count > 0."""
    summary_text = "Summary of earlier conversation context."
    stub_llm = _ReturnsTextClient(text=summary_text, model="openai/gpt-4o")

    messages = [{"role": "user", "content": "prior conversation"}]
    result = summarize_history(messages, stub_llm, "openai/gpt-4o")

    # The "text" field must match what the LLM returned.
    assert result["text"] == summary_text, (
        f"Expected summary text from LLM response, got: {result['text']!r}"
    )
    # token_count must be positive — proves count_tokens ran on the text.
    assert result["token_count"] > 0, (
        "token_count must be > 0; failure means count_tokens wasn't called or returned 0."
    )
    # The LLM must have been called exactly once.
    assert stub_llm.call_count == 1, (
        f"Expected 1 LLM call, got {stub_llm.call_count}. "
        "Failure means summarize_history called the LLM more than once or not at all."
    )


def test_summarize_history_recursive_prompt_includes_continuation_prefix() -> None:
    """
    When history starts with a prior summary, the summarization prompt
    includes a 'Incorporate it' continuation instruction.
    """
    # Prior summary header that triggers recursive detection
    prior_summary_header = (
        "[This is an automatically generated summary of the prior conversation "
        "context. The original messages are available but not included in this "
        "prompt for brevity.]\n\n"
        "Please provide a summary of our conversation so far."
    )
    messages = [
        {"role": "user", "content": prior_summary_header},
        {"role": "assistant", "content": "Earlier we discussed X and Y."},
        {"role": "user", "content": "Now let's continue with Z."},
    ]

    captured_instructions: list[str] = []
    mock_resp = Response(
        output=[MessageOutput(content=[OutputText(text="Combined summary.")])],
        model="openai/gpt-4o",
    )

    class _CapturingClient:
        class responses:
            @staticmethod
            def create(**kwargs: Any) -> Response:
                """Capture the instructions kwarg and return the mock response."""
                captured_instructions.append(kwargs.get("instructions", ""))
                return mock_resp

    result = summarize_history(messages, _CapturingClient(), "openai/gpt-4o")

    assert len(captured_instructions) == 1
    # The continuation prefix must be present when history starts with a prior summary.
    assert "Incorporate it into your new summary" in captured_instructions[0], (
        "Recursive summarization prompt must include the 'Incorporate it' instruction; "
        "failure means _build_summarization_prompt did not detect the prior summary header."
    )
    assert result["text"] == "Combined summary."


def test_compaction_to_history_items_produces_valid_pair() -> None:
    """
    compaction_to_history_items() produces a user+assistant synthetic pair
    for inclusion at the start of conversation history.
    """
    compaction_item = ConversationItem(
        id="cmp_abc123",
        type="compaction",
        status="completed",
        response_id="task_001",
        created_at=1000,
        data=CompactionData(
            summary="The user asked to analyze the dataset. The agent loaded data.csv.",
            last_item_id="msg_xyz999",
            model="openai/gpt-4o",
            token_count=42,
        ),
    )

    result = compaction_to_history_items(compaction_item)

    # Must return exactly 2 items: synthetic user + assistant.
    assert len(result) == 2, (
        f"Expected exactly 2 items (user + assistant), got {len(result)}. "
        "Failure means compaction_to_history_items changed its output shape."
    )
    user_item = result[0]
    assistant_item = result[1]

    # Both items must be message type for history processing.
    assert user_item.type == "message"
    assert assistant_item.type == "message"

    # User item must have role=user.
    assert isinstance(user_item.data, MessageData)
    assert user_item.data.role == "user"

    # User content must contain the summary marker prefix so the LLM
    # understands this is synthetic context, not a real prior message.
    user_text = user_item.data.content[0]["text"]
    assert "[This is an automatically generated summary" in user_text, (
        "User content must contain the summary marker prefix — "
        "failure means the synthetic header was changed or removed."
    )

    # Assistant item must have the summary text verbatim.
    assert isinstance(assistant_item.data, MessageData)
    assert assistant_item.data.role == "assistant"
    assistant_text = assistant_item.data.content[0]["text"]
    assert assistant_text == "The user asked to analyze the dataset. The agent loaded data.csv.", (
        f"Assistant content must equal the CompactionData.summary, got: {assistant_text!r}"
    )

    # IDs must be derived from the compaction item ID.
    assert user_item.id == "cmp_abc123_user"
    assert assistant_item.id == "cmp_abc123_assistant"


def test_count_tokens_returns_positive_integer() -> None:
    """count_tokens returns a positive integer for non-empty messages."""
    messages = [{"role": "user", "content": "Hello world, this is a test message."}]
    result = count_tokens(messages, "openai/gpt-4o")
    # Must be a positive integer; failure means tiktoken encoding failed.
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_unknown_model_falls_back() -> None:
    """Unknown model falls back to cl100k_base encoding without raising."""
    messages = [{"role": "user", "content": "test"}]
    # Should not raise even for completely unknown model names.
    result = count_tokens(messages, "unknown/totally-fake-model-xyz")
    assert result > 0
