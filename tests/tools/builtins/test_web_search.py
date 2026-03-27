"""Tests for built-in web search tools."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_plane.tools.builtins import get_builtin_tool
from agent_plane.tools.builtins.web_search_google import (
    WebSearchGoogleTool,
)
from agent_plane.tools.builtins.web_search_openai import (
    WebSearchOpenAITool,
)
from agent_plane.tools.builtins.web_search_perplexity import (
    WebSearchPerplexityTool,
)


# ── Registry ─────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("web_search_openai", WebSearchOpenAITool),
        ("web_search_google", WebSearchGoogleTool),
        ("web_search_perplexity", WebSearchPerplexityTool),
    ],
)
def test_get_builtin_tool_returns_correct_type(
    name: str,
    expected_type: type,
) -> None:
    """
    ``get_builtin_tool`` returns the correct tool class for
    each registered name.
    """
    tool = get_builtin_tool(name)
    assert isinstance(tool, expected_type), (
        f"Expected {expected_type.__name__} for {name!r}, "
        f"got {type(tool).__name__}."
    )


def test_get_builtin_tool_unknown_returns_none() -> None:
    """
    ``get_builtin_tool`` returns ``None`` for unregistered names.
    """
    assert get_builtin_tool("nonexistent") is None


# ── OpenAI passthrough ───────────────────────────────


def test_openai_schema_is_passthrough() -> None:
    """
    OpenAI web search schema is ``{"type": "web_search_preview"}``,
    not a function schema.
    """
    tool = WebSearchOpenAITool()
    schema = tool.get_schema()
    # Passthrough schema — no "function" key.
    assert schema == {"type": "web_search_preview"}, (
        f"Expected passthrough schema, got {schema}. "
        f"If it has a 'function' key, it was incorrectly "
        f"implemented as a function tool."
    )


def test_openai_invoke_raises() -> None:
    """
    OpenAI tool invoke() raises RuntimeError since execution
    is handled by OpenAI server-side.
    """
    tool = WebSearchOpenAITool()
    with pytest.raises(RuntimeError, match="passthrough"):
        tool.invoke("{}")


def test_openai_name() -> None:
    """Tool name matches the config.yaml builtin name."""
    assert WebSearchOpenAITool().name == "web_search_openai"


# ── Google Custom Search ─────────────────────────────


def test_google_schema_is_function() -> None:
    """
    Google web search has a standard function schema with
    a ``query`` parameter.
    """
    tool = WebSearchGoogleTool()
    schema = tool.get_schema()
    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "web_search_google"
    # query is required.
    assert "query" in func["parameters"]["required"]


def test_google_invoke_missing_env_vars() -> None:
    """
    Google tool returns a clear error when env vars are missing.
    """
    tool = WebSearchGoogleTool()
    with patch.dict("os.environ", {}, clear=True):
        result = tool.invoke(json.dumps({"query": "test"}))
    assert "GOOGLE_SEARCH_API_KEY" in result, (
        f"Expected env var error message, got: {result!r}"
    )


def test_google_invoke_missing_query() -> None:
    """
    Google tool returns error when query param is missing.
    """
    tool = WebSearchGoogleTool()
    result = tool.invoke(json.dumps({}))
    # Exact error message from invoke() when query is absent.
    assert result == "Error: 'query' parameter is required", (
        f"Expected specific 'query' required error, got: {result!r}"
    )


def test_google_invoke_formats_results() -> None:
    """
    Google tool invoke() with a mocked HTTP response returns
    numbered results with title, link, and snippet.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "items": [
            {
                "title": "Python Docs",
                "link": "https://docs.python.org",
                "snippet": "Welcome to Python.",
            },
            {
                "title": "PEP 8",
                "link": "https://peps.python.org/pep-0008/",
                "snippet": "Style guide.",
            },
        ],
    }

    tool = WebSearchGoogleTool()
    env = {
        "GOOGLE_SEARCH_API_KEY": "fake-key",
        "GOOGLE_SEARCH_ENGINE_ID": "fake-engine",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch("agent_plane.tools.builtins.web_search_google.httpx.get")
        as mock_get,
    ):
        mock_get.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "python"}))

    # Both results present with correct numbering — proves the
    # formatting pipeline ran end-to-end through invoke().
    assert "1. Python Docs" in result
    assert "2. PEP 8" in result
    assert "https://docs.python.org" in result
    assert "Style guide." in result


def test_google_invoke_empty_results() -> None:
    """
    Google tool returns 'No results found.' when the API
    returns no items.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {"items": []}

    tool = WebSearchGoogleTool()
    env = {
        "GOOGLE_SEARCH_API_KEY": "fake-key",
        "GOOGLE_SEARCH_ENGINE_ID": "fake-engine",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch("agent_plane.tools.builtins.web_search_google.httpx.get")
        as mock_get,
    ):
        mock_get.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "python"}))

    assert result == "No results found."


# ── Perplexity ───────────────────────────────────────


def test_perplexity_schema_is_function() -> None:
    """
    Perplexity web search has a standard function schema with
    a ``query`` parameter.
    """
    tool = WebSearchPerplexityTool()
    schema = tool.get_schema()
    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "web_search_perplexity"
    assert "query" in func["parameters"]["required"]


def test_perplexity_invoke_missing_env_var() -> None:
    """
    Perplexity tool returns a clear error when API key is missing.
    """
    tool = WebSearchPerplexityTool()
    with patch.dict("os.environ", {}, clear=True):
        result = tool.invoke(json.dumps({"query": "test"}))
    assert "PERPLEXITY_API_KEY" in result, (
        f"Expected env var error message, got: {result!r}"
    )


def test_perplexity_invoke_missing_query() -> None:
    """
    Perplexity tool returns error when query param is missing.
    """
    tool = WebSearchPerplexityTool()
    result = tool.invoke(json.dumps({}))
    # Exact error message from invoke() when query is absent.
    assert result == "Error: 'query' parameter is required", (
        f"Expected specific 'query' required error, got: {result!r}"
    )


def test_perplexity_invoke_with_citations() -> None:
    """
    Perplexity tool invoke() with a mocked HTTP response returns
    the answer text with numbered citations.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Python is a programming language.",
                },
            },
        ],
        "citations": [
            "https://python.org",
            "https://wikipedia.org/wiki/Python",
        ],
    }

    tool = WebSearchPerplexityTool()
    with (
        patch.dict(
            "os.environ",
            {"PERPLEXITY_API_KEY": "fake-key"},
            clear=True,
        ),
        patch(
            "agent_plane.tools.builtins.web_search_perplexity.httpx.post"
        ) as mock_post,
    ):
        mock_post.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "what is python"}))

    # Answer content traversed the full pipeline from mock → invoke → format.
    assert "Python is a programming language." in result
    # Citations are numbered and present.
    assert "[1] https://python.org" in result
    assert "[2] https://wikipedia.org/wiki/Python" in result


def test_perplexity_invoke_no_citations() -> None:
    """
    Perplexity tool invoke() works when no citations are returned.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [
            {"message": {"content": "Just an answer."}},
        ],
    }

    tool = WebSearchPerplexityTool()
    with (
        patch.dict(
            "os.environ",
            {"PERPLEXITY_API_KEY": "fake-key"},
            clear=True,
        ),
        patch(
            "agent_plane.tools.builtins.web_search_perplexity.httpx.post"
        ) as mock_post,
    ):
        mock_post.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "simple question"}))

    assert result == "Just an answer."
    # No "Sources:" section when citations are absent.
    assert "Sources:" not in result


def test_perplexity_invoke_empty_response() -> None:
    """
    Perplexity tool returns 'No answer returned.' when the
    API returns empty choices.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": []}

    tool = WebSearchPerplexityTool()
    with (
        patch.dict(
            "os.environ",
            {"PERPLEXITY_API_KEY": "fake-key"},
            clear=True,
        ),
        patch(
            "agent_plane.tools.builtins.web_search_perplexity.httpx.post"
        ) as mock_post,
    ):
        mock_post.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "test"}))

    assert result == "No answer returned."
