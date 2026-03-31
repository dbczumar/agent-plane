"""Tests for the ACP adapter (input building, session mgmt, SSE parsing)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from integrations.toad.adapter import (
    _build_input,
    _convert_resource_block,
    _convert_resource_link,
    _map_conversations_to_sessions,
    _replay_item,
    _unix_to_iso8601,
    create_adapter,
)


@pytest_asyncio.fixture
async def mock_client() -> AsyncIterator[httpx.AsyncClient]:
    """Async client with a no-op transport for unit tests.

    Used where _build_input needs a client but no actual HTTP
    calls are made (text-only prompts).
    """
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
        base_url="http://test",
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_build_input_text_blocks(
    mock_client: httpx.AsyncClient,
) -> None:
    """Text prompt blocks become input_text content blocks."""
    prompt = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
    ]
    result = await _build_input(mock_client, prompt)
    assert len(result) == 2
    assert result[0] == {"type": "input_text", "text": "Hello"}
    assert result[1] == {"type": "input_text", "text": "World"}


@pytest.mark.asyncio
async def test_build_input_string_fallback(
    mock_client: httpx.AsyncClient,
) -> None:
    """Non-list prompt falls back to single input_text block."""
    result = await _build_input(mock_client, "just a string")
    assert len(result) == 1
    assert result[0] == {
        "type": "input_text",
        "text": "just a string",
    }


@pytest.mark.asyncio
async def test_build_input_empty_list(
    mock_client: httpx.AsyncClient,
) -> None:
    """Empty prompt list produces no blocks."""
    result = await _build_input(mock_client, [])
    assert result == []


@pytest.mark.asyncio
async def test_build_input_image_upload() -> None:
    """Image blocks upload to /v1/files and become input_image."""
    import base64

    image_data = base64.b64encode(b"fake-png-data").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a mock file ID for the upload."""
        return httpx.Response(
            200,
            json={"id": "file_img_123"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
    ) as client:
        result = await _build_input(
            client,
            [{"type": "image", "data": image_data}],
        )
    assert len(result) == 1
    assert result[0] == {
        "type": "input_image",
        "file_id": "file_img_123",
    }


def test_convert_resource_block() -> None:
    """Resource blocks become input_text with file header."""
    block = {
        "type": "resource",
        "resource": {
            "uri": "file:///src/main.py",
            "text": "print('hello')",
        },
    }
    result = _convert_resource_block(block)
    assert result["type"] == "input_text"
    assert "file:///src/main.py" in result["text"]
    assert "print('hello')" in result["text"]


def test_convert_resource_link_non_file_uri() -> None:
    """Non-file:// resource links become text references."""
    block = {"uri": "https://example.com/doc.md"}
    result = _convert_resource_link(block)
    assert result["type"] == "input_text"
    assert "https://example.com/doc.md" in str(result["text"])


def test_convert_resource_link_file_uri(tmp_path: object) -> None:
    """file:// resource links read and embed the file content."""
    import pathlib

    assert isinstance(tmp_path, pathlib.Path)
    test_file = tmp_path / "sample.txt"
    test_file.write_text("file content here")
    block = {"uri": f"file://{test_file}"}
    result = _convert_resource_link(block)
    assert result["type"] == "input_text"
    assert "file content here" in str(result["text"])


@pytest.mark.asyncio
async def test_create_adapter_registers_expected_methods() -> None:
    """create_adapter wires up all expected ACP method handlers."""
    rpc = create_adapter(
        server_url="http://localhost:18400",
        agent_name="test-agent",
    )
    expected_methods = {
        "initialize",
        "session/new",
        "session/prompt",
        "session/cancel",
        "session/list",
        "session/load",
        "fs/read_text_file",
        "fs/write_text_file",
    }
    assert expected_methods.issubset(set(rpc.handlers.keys()))


@pytest.mark.asyncio
async def test_initialize_advertises_full_capabilities() -> None:
    """initialize reports loadSession, list, image, embeddedContext."""
    rpc = create_adapter(
        server_url="http://localhost:18400",
        agent_name="test-agent",
    )
    result = await rpc.handlers["initialize"]({
        "protocolVersion": 1,
        "clientCapabilities": {},
        "clientInfo": {"name": "test", "version": "0.0.1"},
    })
    assert result["protocolVersion"] == 1
    caps = result["agentCapabilities"]
    assert caps["loadSession"] is True
    assert caps["sessionCapabilities"]["list"] is True
    assert caps["promptCapabilities"]["image"] is True
    assert caps["promptCapabilities"]["embeddedContext"] is True
    assert "image/png" in caps["promptCapabilities"][
        "supportedMediaTypes"
    ]


@pytest.mark.asyncio
async def test_session_new_returns_session_id() -> None:
    """session/new handler returns a session ID string."""
    rpc = create_adapter(
        server_url="http://localhost:18400",
        agent_name="test-agent",
    )
    result = await rpc.handlers["session/new"]({
        "cwd": "/tmp/test",
        "mcpServers": [],
    })
    assert isinstance(result, dict)
    session_id = result["sessionId"]
    assert isinstance(session_id, str)
    assert len(session_id) == 16


@pytest.mark.asyncio
async def test_session_prompt_unknown_session_raises() -> None:
    """session/prompt with unknown session ID raises ValueError."""
    rpc = create_adapter(
        server_url="http://localhost:18400",
        agent_name="test-agent",
    )
    with pytest.raises(ValueError, match="Unknown session"):
        await rpc.handlers["session/prompt"]({
            "sessionId": "nonexistent",
            "prompt": [{"type": "text", "text": "hi"}],
        })


@pytest.mark.asyncio
async def test_read_file_handler(tmp_path: object) -> None:
    """fs/read_text_file reads a real file from disk."""
    import pathlib

    assert isinstance(tmp_path, pathlib.Path)
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello from test")

    rpc = create_adapter(
        server_url="http://localhost:18400",
        agent_name="test-agent",
    )
    result = await rpc.handlers["fs/read_text_file"]({
        "sessionId": "ignored",
        "path": str(test_file),
    })
    assert isinstance(result, dict)
    assert result["content"] == "hello from test"


@pytest.mark.asyncio
async def test_write_file_handler(tmp_path: object) -> None:
    """fs/write_text_file writes content to disk."""
    import pathlib

    assert isinstance(tmp_path, pathlib.Path)
    test_file = tmp_path / "output.txt"

    rpc = create_adapter(
        server_url="http://localhost:18400",
        agent_name="test-agent",
    )
    result = await rpc.handlers["fs/write_text_file"]({
        "sessionId": "ignored",
        "path": str(test_file),
        "content": "written by test",
    })
    assert result == {}
    assert test_file.read_text() == "written by test"


def test_map_conversations_to_sessions() -> None:
    """Conversations list maps to ACP session format."""
    body = {
        "data": [
            {
                "id": "conv_1",
                "title": "My Chat",
                "created_at": 1711800000,
            },
            {
                "id": "conv_2",
                "title": None,
                "created_at": 1711800100,
            },
        ],
        "has_more": False,
    }
    sessions = _map_conversations_to_sessions(body)
    assert len(sessions) == 2
    assert sessions[0]["sessionId"] == "conv_1"
    assert sessions[0]["title"] == "My Chat"
    # None title falls back to "Untitled"
    assert sessions[1]["title"] == "Untitled"


def test_unix_to_iso8601() -> None:
    """Unix timestamp converts to ISO 8601 with UTC timezone."""
    result = _unix_to_iso8601(1711800000)
    assert "2024-03-30" in result
    assert "+00:00" in result


def test_unix_to_iso8601_none() -> None:
    """None timestamp returns None."""
    assert _unix_to_iso8601(None) is None


def test_replay_user_message_item() -> None:
    """User message items replay as user_message_chunk."""
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello"}],
    }
    updates = _replay_item(item)
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "user_message_chunk"
    assert updates[0]["content"]["text"] == "Hello"


def test_replay_assistant_message_item() -> None:
    """Assistant message items replay as agent_message_chunk."""
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hi there"}],
    }
    updates = _replay_item(item)
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_message_chunk"
    assert updates[0]["content"]["text"] == "Hi there"


def test_replay_function_call_item() -> None:
    """Function call items replay as tool_call with completed status."""
    item = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "search.web",
        "arguments": '{"q": "test"}',
    }
    updates = _replay_item(item)
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "tool_call"
    assert updates[0]["toolCallId"] == "call_1"
    # Replayed tool calls show as completed (historical)
    assert updates[0]["status"] == "completed"


def test_replay_function_call_output_item() -> None:
    """Function call output items replay as tool_call_update."""
    item = {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "search results here",
    }
    updates = _replay_item(item)
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "tool_call_update"
    assert updates[0]["content"]["text"] == "search results here"


def test_replay_reasoning_item() -> None:
    """Reasoning items replay as agent_thought_chunk."""
    item = {
        "type": "reasoning",
        "summary": [{"type": "text", "text": "I should search..."}],
    }
    updates = _replay_item(item)
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_thought_chunk"
    assert updates[0]["content"]["text"] == "I should search..."


def test_replay_unknown_item_type() -> None:
    """Unknown item types produce no updates."""
    item = {"type": "some_future_type", "data": {}}
    assert _replay_item(item) == []
