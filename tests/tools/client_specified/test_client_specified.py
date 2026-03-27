"""Tests for agent_plane.tools.client_specified."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent_plane.tools.client_specified import (
    CallbackTool,
    CallbackToolSpec,
    parse_callback_tool_spec,
    parse_callback_tool_specs,
)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def minimal_raw_tool() -> dict[str, Any]:
    """
    The minimum valid raw tool dict: a function schema with a
    callback URL in the agent_plane extension key.

    :returns: A dict in OpenAI function tool format with agent_plane
        extension.
    """
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        "agent_plane": {
            "callback": {"url": "https://api.example.com/tools/get_weather"}
        },
    }


@pytest.fixture()
def raw_tool_with_headers() -> dict[str, Any]:
    """
    A raw tool dict that includes callback headers.

    :returns: A dict in OpenAI function tool format with agent_plane
        extension including Authorization header.
    """
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        "agent_plane": {
            "callback": {
                "url": "https://api.example.com/tools/search",
                "headers": {"Authorization": "Bearer tok_xyz"},
            }
        },
    }


@pytest.fixture()
def weather_spec() -> CallbackToolSpec:
    """
    A pre-built CallbackToolSpec for the get_weather tool.

    :returns: A :class:`CallbackToolSpec` with a callback URL and
        no headers.
    """
    return CallbackToolSpec(
        name="get_weather",
        schema={
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        callback_url="https://api.example.com/tools/get_weather",
        callback_headers={},
    )


# ── parse_callback_tool_spec ──────────────────────────────


def test_parse_minimal_tool(minimal_raw_tool: dict[str, Any]) -> None:
    """
    parse_callback_tool_spec returns a correctly populated
    CallbackToolSpec from a minimal valid raw dict.
    """
    spec = parse_callback_tool_spec(minimal_raw_tool)

    # Name extracted from function.name
    assert spec.name == "get_weather"
    # callback_url extracted from agent_plane.callback.url
    assert spec.callback_url == "https://api.example.com/tools/get_weather"
    # No headers means empty dict — not None
    assert spec.callback_headers == {}


def test_parse_strips_agent_plane_from_schema(minimal_raw_tool: dict[str, Any]) -> None:
    """
    The agent_plane key must not appear in the parsed schema — the LLM
    should not see internal callback metadata.

    A failure here would mean callback URLs and auth tokens are exposed
    to the LLM in the tool schema.
    """
    spec = parse_callback_tool_spec(minimal_raw_tool)

    # agent_plane key must be gone
    assert "agent_plane" not in spec.schema
    # Standard OpenAI fields must still be present
    assert spec.schema["type"] == "function"
    assert spec.schema["function"]["name"] == "get_weather"


def test_parse_tool_with_headers(raw_tool_with_headers: dict[str, Any]) -> None:
    """
    parse_callback_tool_spec captures callback headers from
    agent_plane.callback.headers.
    """
    spec = parse_callback_tool_spec(raw_tool_with_headers)

    assert spec.name == "search"
    assert spec.callback_headers == {"Authorization": "Bearer tok_xyz"}


@pytest.mark.parametrize(
    "bad_tool,expected_fragment",
    [
        # Wrong type field
        (
            {
                "type": "not_function",
                "function": {"name": "x"},
                "agent_plane": {"callback": {"url": "https://x.com"}},
            },
            "type 'function'",
        ),
        # Missing function object entirely
        (
            {"type": "function", "agent_plane": {"callback": {"url": "https://x.com"}}},
            "missing 'function'",
        ),
        # Missing function.name
        (
            {
                "type": "function",
                "function": {"description": "no name here"},
                "agent_plane": {"callback": {"url": "https://x.com"}},
            },
            "missing function.name",
        ),
        # Missing agent_plane key entirely
        (
            {"type": "function", "function": {"name": "x"}},
            "missing 'agent_plane'",
        ),
        # Missing agent_plane.callback
        (
            {"type": "function", "function": {"name": "x"}, "agent_plane": {}},
            "missing 'agent_plane.callback'",
        ),
        # Missing callback URL
        (
            {
                "type": "function",
                "function": {"name": "x"},
                "agent_plane": {"callback": {}},
            },
            "missing 'agent_plane.callback.url'",
        ),
        # Non-string header value
        (
            {
                "type": "function",
                "function": {"name": "x"},
                "agent_plane": {
                    "callback": {
                        "url": "https://x.com",
                        "headers": {"X-Count": 42},
                    }
                },
            },
            "string keys to string values",
        ),
    ],
)
def test_parse_raises_on_malformed(
    bad_tool: dict[str, Any],
    expected_fragment: str,
) -> None:
    """
    parse_callback_tool_spec raises ValueError with a descriptive
    message for each class of malformed input.

    A failure (no exception raised, or wrong exception type) would
    mean malformed client tools are silently accepted, leading to
    runtime errors deep inside the agent loop.
    """
    with pytest.raises(ValueError, match=expected_fragment):
        parse_callback_tool_spec(bad_tool)


def test_parse_callback_tool_specs_empty() -> None:
    """
    parse_callback_tool_specs returns an empty list for empty input.
    """
    assert parse_callback_tool_specs([]) == []


def test_parse_callback_tool_specs_multiple(
    minimal_raw_tool: dict[str, Any],
    raw_tool_with_headers: dict[str, Any],
) -> None:
    """
    parse_callback_tool_specs parses every tool in the list and
    returns them in order.
    """
    specs = parse_callback_tool_specs([minimal_raw_tool, raw_tool_with_headers])

    # Two tools parsed in order
    assert len(specs) == 2, (
        f"Expected 2 specs (one per raw tool), got {len(specs)}. "
        "If 0 or 1, parse_callback_tool_specs short-circuited."
    )
    assert specs[0].name == "get_weather"
    assert specs[1].name == "search"


# ── CallbackTool.get_schema ───────────────────────────────


def test_get_schema_returns_spec_schema(weather_spec: CallbackToolSpec) -> None:
    """
    CallbackTool.get_schema returns exactly the schema stored in the
    spec — the LLM sees only standard OpenAI format, no agent_plane key.
    """
    tool = CallbackTool(weather_spec)

    schema = tool.get_schema()

    assert schema is weather_spec.schema
    assert "agent_plane" not in schema
    assert schema["function"]["name"] == "get_weather"


def test_name_property(weather_spec: CallbackToolSpec) -> None:
    """
    CallbackTool.name returns the tool name from the spec.
    """
    tool = CallbackTool(weather_spec)

    assert tool.name == "get_weather"


# ── CallbackTool.invoke ───────────────────────────────────


def test_invoke_posts_to_callback_url(
    weather_spec: CallbackToolSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CallbackTool.invoke makes a POST to callback_url with the correct
    JSON body and returns the response text.

    If this test fails after a refactor, check that invoke still uses
    httpx.post and passes name + arguments in the body.
    """
    mock_response = MagicMock()
    mock_response.text = "Sunny, 72°F"
    mock_response.raise_for_status = MagicMock()

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        return mock_response

    monkeypatch.setattr("agent_plane.tools.client_specified.httpx.post", fake_post)

    tool = CallbackTool(weather_spec)
    result = tool.invoke('{"city": "San Francisco"}')

    # POST to the callback URL
    assert captured["url"] == "https://api.example.com/tools/get_weather"
    # Body carries name and arguments
    assert captured["json"] == {
        "name": "get_weather",
        "arguments": '{"city": "San Francisco"}',
    }
    # Response text returned as tool result
    assert result == "Sunny, 72°F"


def test_invoke_sends_callback_headers(
    raw_tool_with_headers: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CallbackTool.invoke forwards callback_headers to the HTTP request.

    A failure means auth tokens in callback headers are silently dropped,
    causing 401s at the callback server.
    """
    spec = parse_callback_tool_spec(raw_tool_with_headers)

    mock_response = MagicMock()
    mock_response.text = "result"
    mock_response.raise_for_status = MagicMock()
    captured_headers: dict[str, str] = {}

    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        captured_headers.update(kwargs.get("headers", {}))
        return mock_response

    monkeypatch.setattr("agent_plane.tools.client_specified.httpx.post", fake_post)

    tool = CallbackTool(spec)
    tool.invoke("{}")

    assert captured_headers.get("Authorization") == "Bearer tok_xyz"


def test_invoke_returns_error_string_on_http_status_error(
    weather_spec: CallbackToolSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the callback returns a non-2xx status, invoke returns an
    error string (not raises) so the LLM can handle the failure
    gracefully within the agent loop.
    """
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        raise httpx.HTTPStatusError(
            "503",
            request=MagicMock(),
            response=mock_response,
        )

    monkeypatch.setattr("agent_plane.tools.client_specified.httpx.post", fake_post)

    tool = CallbackTool(weather_spec)
    result = tool.invoke("{}")

    # Error string returned — not raised
    assert result.startswith("Error:")
    assert "503" in result


def test_invoke_returns_error_string_on_connection_error(
    weather_spec: CallbackToolSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the HTTP connection fails entirely, invoke returns an error
    string so the agent loop can continue and report the failure to
    the LLM rather than crashing the workflow.
    """
    def fake_post(url: str, **kwargs: Any) -> MagicMock:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("agent_plane.tools.client_specified.httpx.post", fake_post)

    tool = CallbackTool(weather_spec)
    result = tool.invoke("{}")

    assert result.startswith("Error:")
    assert "connection refused" in result
