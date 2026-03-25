"""Tests for llms.adapters.openai — payload building and SSE parsing."""

import json

from llms.adapters.openai import OpenAICompatibleAdapter, _parse_sse_line

# ── Payload building ─────────────────────────────────────


def test_basic_payload_structure() -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        api_key_env=None,
    )
    payload = adapter._build_payload(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-5.4",
        tools=None,
        stream=False,
        extra={},
    )
    assert payload["model"] == "gpt-5.4"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert "tools" not in payload
    assert "stream" not in payload


def test_tools_included_when_provided() -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        api_key_env=None,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        }
    ]
    payload = adapter._build_payload(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-5.4",
        tools=tools,
        stream=False,
        extra={},
    )
    assert payload["tools"] == tools


def test_stream_options_added_for_streaming() -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        api_key_env=None,
    )
    payload = adapter._build_payload(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-5.4",
        tools=None,
        stream=True,
        extra={},
    )
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_extra_kwargs_merged_into_payload() -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1",
        api_key_env=None,
    )
    payload = adapter._build_payload(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-5.4",
        tools=None,
        stream=False,
        extra={"temperature": 0.5, "top_p": 0.9},
    )
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9


def test_base_url_trailing_slash_stripped() -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="https://api.openai.com/v1/",
        api_key_env=None,
    )
    assert adapter._base_url == "https://api.openai.com/v1"


# ── Headers ──────────────────────────────────────────────


def test_headers_without_api_key() -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="https://localhost",
        api_key_env=None,
    )
    headers = adapter._build_headers()
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_headers_with_api_key(monkeypatch: object) -> None:
    # Use monkeypatch via pytest fixture type
    monkeypatch.setenv("TEST_API_KEY", "sk-test-123")  # type: ignore[union-attr]
    adapter = OpenAICompatibleAdapter(
        base_url="https://localhost",
        api_key_env="TEST_API_KEY",
    )
    headers = adapter._build_headers()
    assert headers["Authorization"] == "Bearer sk-test-123"


# ── SSE parsing ──────────────────────────────────────────


def test_parse_sse_data_line() -> None:
    data = {"id": "chatcmpl-1", "choices": []}
    line = f"data: {json.dumps(data)}"
    result = _parse_sse_line(line)
    assert result == data


def test_parse_sse_done_sentinel() -> None:
    assert _parse_sse_line("data: [DONE]") is None


def test_parse_sse_non_data_line() -> None:
    assert _parse_sse_line("event: message") is None
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": comment") is None
