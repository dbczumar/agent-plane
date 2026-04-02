"""Tests for built-in web search tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_plane.tools.base import ToolContext
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
        f"Expected {expected_type.__name__} for {name!r}, got {type(tool).__name__}."
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


def test_openai_invoke_raises(tool_ctx: ToolContext) -> None:
    """
    OpenAI tool invoke() raises RuntimeError since execution
    is handled by OpenAI server-side.
    """
    tool = WebSearchOpenAITool()
    with pytest.raises(RuntimeError, match="passthrough"):
        tool.invoke("{}", tool_ctx)


def test_openai_name() -> None:
    """Tool name matches the config.yaml builtin name."""
    assert WebSearchOpenAITool.name() == "web_search_openai"


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


def test_google_invoke_missing_keys(tool_ctx: ToolContext) -> None:
    """
    Google tool returns a clear error when neither spec config
    nor env vars provide the required keys.
    """
    tool = WebSearchGoogleTool()
    with patch.dict("os.environ", {}, clear=True):
        result = tool.invoke(json.dumps({"query": "test"}), tool_ctx)
    # Error message mentions both spec config and env var paths.
    assert "api_key" in result, f"Expected key error message, got: {result!r}"
    assert "engine_id" in result, f"Expected engine_id in error, got: {result!r}"


def test_google_invoke_missing_query(tool_ctx: ToolContext) -> None:
    """
    Google tool returns error when query param is missing.
    """
    tool = WebSearchGoogleTool()
    result = tool.invoke(json.dumps({}), tool_ctx)
    # Exact error message from invoke() when query is absent.
    assert result == "Error: 'query' parameter is required", (
        f"Expected specific 'query' required error, got: {result!r}"
    )


def test_google_invoke_formats_results(tool_ctx: ToolContext) -> None:
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
        patch("agent_plane.tools.builtins.web_search_google.httpx.get") as mock_get,
    ):
        mock_get.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "python"}), tool_ctx)

    # Both results present with correct numbering — proves the
    # formatting pipeline ran end-to-end through invoke().
    assert "1. Python Docs" in result
    assert "2. PEP 8" in result
    assert "https://docs.python.org" in result
    assert "Style guide." in result


def test_google_invoke_empty_results(tool_ctx: ToolContext) -> None:
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
        patch("agent_plane.tools.builtins.web_search_google.httpx.get") as mock_get,
    ):
        mock_get.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "python"}), tool_ctx)

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


def test_perplexity_invoke_missing_key(tool_ctx: ToolContext) -> None:
    """
    Perplexity tool returns a clear error when neither spec
    config nor env var provides the API key.
    """
    tool = WebSearchPerplexityTool()
    with patch.dict("os.environ", {}, clear=True):
        result = tool.invoke(json.dumps({"query": "test"}), tool_ctx)
    # Error message mentions both spec config and env var paths.
    assert "api_key" in result, f"Expected key error message, got: {result!r}"
    assert "PERPLEXITY_API_KEY" in result, f"Expected env var name in error, got: {result!r}"


def test_perplexity_invoke_missing_query(tool_ctx: ToolContext) -> None:
    """
    Perplexity tool returns error when query param is missing.
    """
    tool = WebSearchPerplexityTool()
    result = tool.invoke(json.dumps({}), tool_ctx)
    # Exact error message from invoke() when query is absent.
    assert result == "Error: 'query' parameter is required", (
        f"Expected specific 'query' required error, got: {result!r}"
    )


def test_perplexity_invoke_with_citations(tool_ctx: ToolContext) -> None:
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
        patch("agent_plane.tools.builtins.web_search_perplexity.httpx.post") as mock_post,
    ):
        mock_post.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "what is python"}), tool_ctx)

    # Answer content traversed the full pipeline from mock → invoke → format.
    assert "Python is a programming language." in result
    # Citations are numbered and present.
    assert "[1] https://python.org" in result
    assert "[2] https://wikipedia.org/wiki/Python" in result


def test_perplexity_invoke_no_citations(tool_ctx: ToolContext) -> None:
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
        patch("agent_plane.tools.builtins.web_search_perplexity.httpx.post") as mock_post,
    ):
        mock_post.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "simple question"}), tool_ctx)

    assert result == "Just an answer."
    # No "Sources:" section when citations are absent.
    assert "Sources:" not in result


def test_perplexity_invoke_empty_response(tool_ctx: ToolContext) -> None:
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
        patch("agent_plane.tools.builtins.web_search_perplexity.httpx.post") as mock_post,
    ):
        mock_post.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "test"}), tool_ctx)

    assert result == "No answer returned."


# ── Spec-level config ───────────────────────────────


def test_google_uses_spec_config_over_env(tool_ctx: ToolContext) -> None:
    """
    Google tool prefers api_key/engine_id from spec config
    over environment variables.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {"items": []}

    # Spec config provides keys; env has different values.
    tool = WebSearchGoogleTool(
        config={
            "api_key": "spec-key",
            "engine_id": "spec-engine",
        }
    )
    env = {
        "GOOGLE_SEARCH_API_KEY": "env-key",
        "GOOGLE_SEARCH_ENGINE_ID": "env-engine",
    }
    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "agent_plane.tools.builtins.web_search_google.httpx.get",
        ) as mock_get,
    ):
        mock_get.return_value = fake_response
        tool.invoke(json.dumps({"query": "test"}), tool_ctx)

    # Verify the HTTP call used spec config keys, not env keys.
    call_kwargs = mock_get.call_args
    params = call_kwargs.kwargs["params"]
    assert params["key"] == "spec-key", f"Expected spec config api_key, got {params['key']!r}"
    assert params["cx"] == "spec-engine", f"Expected spec config engine_id, got {params['cx']!r}"


def test_perplexity_uses_spec_config_over_env(tool_ctx: ToolContext) -> None:
    """
    Perplexity tool prefers api_key from spec config over
    environment variable.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "answer"}}],
    }

    tool = WebSearchPerplexityTool(config={"api_key": "spec-pplx"})
    with (
        patch.dict(
            "os.environ",
            {"PERPLEXITY_API_KEY": "env-pplx"},
            clear=True,
        ),
        patch(
            "agent_plane.tools.builtins.web_search_perplexity.httpx.post",
        ) as mock_post,
    ):
        mock_post.return_value = fake_response
        tool.invoke(json.dumps({"query": "test"}), tool_ctx)

    # Verify the HTTP call used spec config key, not env key.
    call_kwargs = mock_post.call_args
    headers = call_kwargs.kwargs["headers"]
    assert headers["Authorization"] == "Bearer spec-pplx", (
        f"Expected spec config api_key in header, got {headers['Authorization']!r}"
    )


def test_get_builtin_tool_passes_config(tool_ctx: ToolContext) -> None:
    """
    ``get_builtin_tool`` passes the config dict through to
    the tool constructor.
    """
    config = {"api_key": "test-key", "engine_id": "test-engine"}
    tool = get_builtin_tool("web_search_google", config=config)
    assert isinstance(tool, WebSearchGoogleTool)
    # Config is stored and accessible for invoke().
    # Verify by checking it doesn't error with missing env vars
    # when config provides the keys.
    fake_response = MagicMock()
    fake_response.json.return_value = {"items": []}
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "agent_plane.tools.builtins.web_search_google.httpx.get",
        ) as mock_get,
    ):
        mock_get.return_value = fake_response
        result = tool.invoke(json.dumps({"query": "test"}), tool_ctx)
    # Should succeed (no error) because config provided the keys.
    assert result == "No results found."
