"""Unit tests for RichBlockFormatter."""

from __future__ import annotations

from agent_plane_ui_sdk._blocks import (
    CompactionBlock,
    ErrorBlock,
    FileBlock,
    ReasoningBlock,
    ReasoningStartBlock,
    ResponseEndBlock,
    ResponseStartBlock,
    RetryBlock,
    TextChunk,
    TextDone,
    ToolExecution,
    ToolGroup,
)
from agent_plane_ui_sdk.terminal._formatter import (
    RichBlockFormatter,
    StreamingText,
)


def test_format_response_start() -> None:
    """ResponseStartBlock → Text with model name."""
    fmt = RichBlockFormatter()
    items = fmt.format(ResponseStartBlock(model="coder", response_id="r1"))
    assert len(items) == 1
    # Should contain the model name in the rendered text.
    text_str = str(items[0])
    assert "coder" in text_str


def test_format_text_chunk_returns_streaming_text() -> None:
    """TextChunk → StreamingText marker."""
    fmt = RichBlockFormatter()
    items = fmt.format(TextChunk(text="hello "))
    assert len(items) == 1
    assert isinstance(items[0], StreamingText)
    assert items[0].text == "hello "


def test_format_text_done_no_code_blocks() -> None:
    """TextDone without code blocks → empty (streamed text is enough)."""
    fmt = RichBlockFormatter()
    items = fmt.format(TextDone(full_text="just text", has_code_blocks=False))
    assert items == []


def test_format_text_done_with_code_blocks() -> None:
    """TextDone with code blocks → Markdown renderable."""
    fmt = RichBlockFormatter()
    items = fmt.format(
        TextDone(
            full_text="```python\nprint('hi')\n```",
            has_code_blocks=True,
        )
    )
    assert len(items) == 1
    # Should be a Rich renderable (Padding wrapping Markdown).
    assert not isinstance(items[0], StreamingText)


def test_format_tool_group() -> None:
    """ToolGroup → tool call line + result panel per execution."""
    fmt = RichBlockFormatter()
    group = ToolGroup(
        executions=[
            ToolExecution(
                name="Read",
                arguments={"file_path": "/tmp/f"},
                args_summary="f",
                call_id="c1",
                agent_name="coder",
                executed_by="client",
                output="content",
            ),
        ]
    )
    items = fmt.format(group)
    # At least 2 items: tool call line + result panel.
    assert len(items) >= 2


def test_format_tool_group_no_output() -> None:
    """ToolGroup with no output → only the tool call line."""
    fmt = RichBlockFormatter()
    group = ToolGroup(
        executions=[
            ToolExecution(
                name="Glob",
                arguments={"pattern": "*"},
                args_summary="*",
                call_id="c1",
                agent_name="coder",
                executed_by="server",
                output=None,
            ),
        ]
    )
    items = fmt.format(group)
    # Only the tool call line, no result panel.
    assert len(items) == 1


def test_format_reasoning_start() -> None:
    """ReasoningStartBlock → thinking indicator."""
    fmt = RichBlockFormatter()
    items = fmt.format(ReasoningStartBlock())
    assert len(items) == 1
    assert "thinking" in str(items[0]).lower()


def test_format_reasoning_with_text() -> None:
    """ReasoningBlock with text → panel."""
    fmt = RichBlockFormatter()
    items = fmt.format(
        ReasoningBlock(
            reasoning_text="deep thoughts here",
            summary_text="",
        )
    )
    assert len(items) == 1  # The panel.


def test_format_reasoning_empty() -> None:
    """ReasoningBlock with no text → nothing."""
    fmt = RichBlockFormatter()
    items = fmt.format(ReasoningBlock(reasoning_text="", summary_text=""))
    assert items == []


def test_format_error() -> None:
    """ErrorBlock → error panel."""
    fmt = RichBlockFormatter()
    items = fmt.format(ErrorBlock(message="something broke", source="llm"))
    assert len(items) == 1


def test_format_retry() -> None:
    """RetryBlock → retry indicator."""
    fmt = RichBlockFormatter()
    items = fmt.format(
        RetryBlock(
            source="tool",
            attempt=2,
            max_attempts=3,
            delay_seconds=1.5,
        )
    )
    assert len(items) == 1
    assert "retrying" in str(items[0]).lower()


def test_format_compaction() -> None:
    """CompactionBlock → compacting indicator."""
    fmt = RichBlockFormatter()
    items = fmt.format(CompactionBlock())
    assert len(items) == 1
    assert "compacting" in str(items[0]).lower()


def test_format_file() -> None:
    """FileBlock → file indicator."""
    fmt = RichBlockFormatter()
    items = fmt.format(FileBlock(file_id="f1", filename="photo.png"))
    assert len(items) == 1
    assert "photo.png" in str(items[0])


def test_format_response_end_completed() -> None:
    """Completed response → nothing (no status message)."""
    fmt = RichBlockFormatter()
    items = fmt.format(ResponseEndBlock(status="completed"))
    assert items == []


def test_format_response_end_failed() -> None:
    """Failed response → status message."""
    fmt = RichBlockFormatter()
    items = fmt.format(ResponseEndBlock(status="failed"))
    assert len(items) == 1
    assert "failed" in str(items[0]).lower()


def test_show_agent_labels_for_sub_agents() -> None:
    """show_agent_labels=True adds agent name for sub-agent blocks."""
    from agent_plane_ui_sdk._blocks import BlockContext

    fmt = RichBlockFormatter(show_agent_labels=True)
    block = TextChunk(
        text="sub-agent text",
        ctx=BlockContext(agent="coder.researcher", depth=1),
    )
    items = fmt.format(block)
    # Should have the agent label + the streaming text.
    assert len(items) == 2
    # Rich Text.plain gives the visible text; markup has the agent name.
    label_text = items[0].plain if hasattr(items[0], "plain") else str(items[0])
    assert "researcher" in label_text or "coder.researcher" in repr(items[0])


def test_show_agent_labels_not_for_root() -> None:
    """show_agent_labels=True doesn't add label for root agent."""
    from agent_plane_ui_sdk._blocks import BlockContext

    fmt = RichBlockFormatter(show_agent_labels=True)
    block = TextChunk(
        text="root text",
        ctx=BlockContext(agent="coder", depth=0),
    )
    items = fmt.format(block)
    # Just the streaming text, no label.
    assert len(items) == 1


def test_custom_accent_color() -> None:
    """Custom accent color is used."""
    fmt = RichBlockFormatter(accent_color="#ff0000")
    assert fmt.accent == "#ff0000"


def test_welcome_message() -> None:
    """welcome() returns a renderable with the model name."""
    fmt = RichBlockFormatter()
    item = fmt.welcome("my-agent")
    assert item is not None


def test_user_message() -> None:
    """user_message() returns a renderable."""
    fmt = RichBlockFormatter()
    item = fmt.user_message("hello world")
    assert item is not None


def test_user_message_truncation() -> None:
    """Long user messages are truncated to 4 lines."""
    fmt = RichBlockFormatter()
    long_text = "line1\nline2\nline3\nline4\nline5\nline6"
    item = fmt.user_message(long_text)
    rendered = str(item)
    assert "more lines" in rendered


def test_subclass_override() -> None:
    """Subclassing and overriding one method works."""

    class CustomFormatter(RichBlockFormatter):
        def format_error(self, block: ErrorBlock) -> list:  # type: ignore[override]
            return [StreamingText(text=f"CUSTOM ERROR: {block.message}")]

    fmt = CustomFormatter()
    items = fmt.format(ErrorBlock(message="test error", source="llm"))
    assert len(items) == 1
    assert isinstance(items[0], StreamingText)
    assert "CUSTOM ERROR" in items[0].text
