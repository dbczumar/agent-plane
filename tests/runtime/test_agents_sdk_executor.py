"""Tests for agent_plane.runtime.executors.agents_sdk.

Tests monkeypatch the ``agents`` module to avoid requiring a real
``openai-agents`` installation. Mock event objects use simple
dataclasses with the same attributes the executor reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_plane.runtime.executors.agents_sdk import (
    AgentsSdkExecutor,
    _build_model_settings,
    _build_openai_client,
    _extract_codex_tools,
    _has_web_search,
    _make_function_tool,
    _map_event,
    _messages_to_input,
)
from agent_plane.runtime.executors.base import (
    ExecutorContext,
    ReasoningChunk,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
)
from agent_plane.spec.types import (
    AgentSpec,
    BuiltinToolConfig,
    ExecutorSpec,
    LLMConfig,
)

# ── Mock SDK types ───────────────────────────────────────


@dataclass
class _MockRawItem:
    """
    Mock for the SDK's raw tool call item.

    :param call_id: Tool call identifier.
    :param name: Tool name.
    :param arguments: JSON-encoded arguments string.
    """

    call_id: str = "call_abc"
    name: str = "test_tool"
    arguments: str = '{"key": "value"}'


@dataclass
class _MockToolItem:
    """
    Mock for the SDK's tool call run item.

    :param raw_item: The raw tool call data.
    :param output: The tool's output string.
    """

    raw_item: _MockRawItem = field(
        default_factory=_MockRawItem,
    )
    output: str = "tool result"


@dataclass
class _MockStreamEvent:
    """
    Mock for the SDK's ``StreamEvent``.

    :param type: Event type, e.g. ``"raw_response_event"``
        or ``"run_item_stream_event"``.
    :param data: Raw event data (for ``raw_response_event``).
    :param name: Event name (for ``run_item_stream_event``),
        e.g. ``"tool_called"``.
    :param item: The run item (for ``run_item_stream_event``).
    """

    type: str = "raw_response_event"
    data: Any = None
    name: str | None = None
    item: Any = None


@dataclass
class _MockTextDelta:
    """
    Mock for ``ResponseTextDeltaEvent``.

    :param delta: The text fragment.
    """

    delta: str = "Hello"


@dataclass
class _MockReasoningSummaryDelta:
    """
    Mock for ``ResponseReasoningSummaryTextDeltaEvent``.

    :param delta: The reasoning text fragment.
    """

    delta: str = "thinking..."


# Give it the class name the executor checks.
_MockReasoningSummaryDelta.__name__ = "ResponseReasoningSummaryTextDeltaEvent"
_MockReasoningSummaryDelta.__qualname__ = "ResponseReasoningSummaryTextDeltaEvent"


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture()
def executor_context() -> ExecutorContext:
    """
    Minimal executor context for tests.

    :returns: An ``ExecutorContext`` with async stub call_tool.
    """

    async def _stub_call_tool(
        _req: ToolCallRequested,
    ) -> ToolResult:
        """
        Stub async call_tool.

        :param _req: Ignored.
        :returns: A fixed successful result.
        """
        return ToolResult(
            content="stub output",
            status="success",
        )

    return ExecutorContext(
        task_id="task_test_123",
        conversation_id="conv_test_456",
        storage_dir=Path("/tmp/test-storage"),
        call_tool=_stub_call_tool,
    )


def _minimal_spec(
    model: str = "gpt-5.4",
    builtins: list[BuiltinToolConfig] | None = None,
    connection: dict[str, str] | None = None,
) -> AgentSpec:
    """
    Build a minimal AgentSpec for agents_sdk executor tests.

    :param model: The LLM model identifier.
    :param builtins: Builtin tool configs, or ``None`` for
        empty.
    :param connection: LLM connection overrides, or ``None``.
    :returns: A configured AgentSpec.
    """
    return AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="agents_sdk"),
        llm=LLMConfig(model=model, connection=connection),
        tools=AgentSpec(
            spec_version=1,
        ).tools
        if builtins is None
        else AgentSpec(
            spec_version=1,
        ).tools,
    )


# ── _extract_codex_tools ─────────────────────────────────


def test_extract_codex_tools_strips_prefix() -> None:
    """
    ``codex:``-prefixed builtins are extracted with prefix
    stripped.
    """
    spec = AgentSpec(
        spec_version=1,
        tools=AgentSpec(spec_version=1).tools,
    )
    spec.tools.builtins = [
        BuiltinToolConfig(name="codex:Shell"),
        BuiltinToolConfig(name="codex:ApplyPatch"),
        BuiltinToolConfig(name="web_search"),
    ]
    result = _extract_codex_tools(spec)
    assert result == ["Shell", "ApplyPatch"]


def test_extract_codex_tools_empty_when_no_prefix() -> None:
    """
    No ``codex:`` tools → empty list.
    """
    spec = AgentSpec(
        spec_version=1,
        tools=AgentSpec(spec_version=1).tools,
    )
    spec.tools.builtins = [
        BuiltinToolConfig(name="web_search"),
    ]
    result = _extract_codex_tools(spec)
    assert result == []


# ── _has_web_search ──────────────────────────────────────


def test_has_web_search_true() -> None:
    """
    Returns True when ``web_search`` is present.
    """
    builtins = [
        BuiltinToolConfig(name="web_search"),
    ]
    assert _has_web_search(builtins) is True


def test_has_web_search_false() -> None:
    """
    Returns False when ``web_search`` is absent.
    """
    builtins = [
        BuiltinToolConfig(name="codex:Shell"),
    ]
    assert _has_web_search(builtins) is False


# ── _build_model_settings ────────────────────────────────


def test_build_model_settings_basic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Default LLMConfig produces a ModelSettings with no extras.
    """
    # Mock the agents module for ModelSettings.
    _patch_agents_module(monkeypatch)

    config = LLMConfig(model="gpt-5.4")
    settings = _build_model_settings(config)
    assert settings.temperature is None
    assert settings.max_tokens is None


def test_build_model_settings_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``reasoning_effort`` maps to ``ModelSettings.reasoning``.
    """
    _patch_agents_module(monkeypatch)
    _patch_reasoning_type(monkeypatch)

    config = LLMConfig(
        model="o3",
        extra={"reasoning_effort": "high"},
    )
    settings = _build_model_settings(config)
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "high"
    # "detailed" enables reasoning summary streaming.
    assert settings.reasoning.summary == "detailed"


def test_build_model_settings_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``temperature`` maps to ``ModelSettings.temperature``.
    """
    _patch_agents_module(monkeypatch)

    config = LLMConfig(
        model="gpt-5.4",
        extra={"temperature": 0.7},
    )
    settings = _build_model_settings(config)
    assert settings.temperature == 0.7


def test_build_model_settings_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``max_completion_tokens`` maps to ``ModelSettings.max_tokens``.
    """
    _patch_agents_module(monkeypatch)

    config = LLMConfig(
        model="gpt-5.4",
        extra={"max_completion_tokens": 4096},
    )
    settings = _build_model_settings(config)
    assert settings.max_tokens == 4096


def test_build_model_settings_extra_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unknown keys pass through to ``extra_body``.
    """
    _patch_agents_module(monkeypatch)

    config = LLMConfig(
        model="gpt-5.4",
        extra={"custom_param": "value"},
    )
    settings = _build_model_settings(config)
    assert settings.extra_body == {"custom_param": "value"}


# ── _build_openai_client ─────────────────────────────────


def test_build_openai_client_with_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Connection dict → AsyncOpenAI client is created.
    """
    created_args: dict[str, Any] = {}

    class _MockAsyncOpenAI:
        """
        Mock AsyncOpenAI that captures constructor args.

        :param kwargs: Captured constructor kwargs.
        """

        def __init__(self, **kwargs: Any) -> None:
            created_args.update(kwargs)

    monkeypatch.setattr(
        "agent_plane.runtime.executors.agents_sdk.AsyncOpenAI",
        _MockAsyncOpenAI,
        raising=False,
    )
    # Patch the import path.
    import sys
    import types

    mock_openai = types.ModuleType("openai")
    mock_openai.AsyncOpenAI = _MockAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    result = _build_openai_client(
        {"api_key": "sk-test", "base_url": "https://api.test"},
    )
    assert result is not None
    assert created_args["api_key"] == "sk-test"
    assert created_args["base_url"] == "https://api.test"


def test_build_openai_client_none_uses_env() -> None:
    """
    ``None`` connection → returns ``None`` (SDK reads env).
    """
    result = _build_openai_client(None)
    assert result is None


# ── _map_event ───────────────────────────────────────────


def test_map_event_text_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``ResponseTextDeltaEvent`` maps to ``TextChunk``.
    """
    _patch_response_types(monkeypatch)

    event = _MockStreamEvent(
        type="raw_response_event",
        data=_MockTextDelta(delta="Hello world"),
    )
    result = _map_event(event)
    assert isinstance(result, TextChunk)
    assert result.text == "Hello world"


def test_map_event_reasoning_summary_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reasoning summary delta maps to ``ReasoningChunk``.
    """
    _patch_response_types(monkeypatch)

    event = _MockStreamEvent(
        type="raw_response_event",
        data=_MockReasoningSummaryDelta(delta="Let me think"),
    )
    result = _map_event(event)
    assert isinstance(result, ReasoningChunk)
    assert result.delta == "Let me think"
    assert result.event_type == "reasoning_summary"


def test_map_event_tool_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``tool_called`` run item maps to ``ToolCallObserved``.
    """
    _patch_response_types(monkeypatch)

    raw = _MockRawItem(
        call_id="call_123",
        name="Read",
        arguments='{"file_path": "/tmp/test"}',
    )
    item = _MockToolItem(raw_item=raw, output="file contents")
    event = _MockStreamEvent(
        type="run_item_stream_event",
        name="tool_called",
        item=item,
    )
    result = _map_event(event)
    assert isinstance(result, ToolCallObserved)
    assert result.call_id == "call_123"
    assert result.name == "Read"
    assert result.arguments == {"file_path": "/tmp/test"}
    assert result.result == "file contents"
    assert result.status == "success"


def test_map_event_ignores_unknown_raw_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unknown raw event subtypes → ``None`` (no crash).
    """
    _patch_response_types(monkeypatch)

    @dataclass
    class _UnknownEvent:
        """Mock unknown event type."""

        pass

    event = _MockStreamEvent(
        type="raw_response_event",
        data=_UnknownEvent(),
    )
    result = _map_event(event)
    assert result is None


def test_map_event_ignores_unknown_run_item_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unknown run item event names → ``None`` (no crash).
    """
    _patch_response_types(monkeypatch)

    event = _MockStreamEvent(
        type="run_item_stream_event",
        name="future_event_type",
        item=None,
    )
    result = _map_event(event)
    assert result is None


def test_map_event_ignores_unknown_top_level_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unknown top-level event type → ``None`` (no crash).
    """
    _patch_response_types(monkeypatch)

    event = _MockStreamEvent(type="agent_updated")
    result = _map_event(event)
    assert result is None


# ── from_spec ────────────────────────────────────────────


def test_from_spec_extracts_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``from_spec`` extracts ``spec.llm.model``.
    """
    _patch_ensure_sdk(monkeypatch)

    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="agents_sdk"),
        llm=LLMConfig(model="gpt-5.4"),
    )
    executor = AgentsSdkExecutor.from_spec(spec)
    assert executor._model == "gpt-5.4"


def test_from_spec_extracts_codex_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``from_spec`` extracts ``codex:``-prefixed tool names.
    """
    _patch_ensure_sdk(monkeypatch)

    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="agents_sdk"),
        llm=LLMConfig(model="gpt-5.4"),
    )
    spec.tools.builtins = [
        BuiltinToolConfig(name="codex:Shell"),
    ]
    executor = AgentsSdkExecutor.from_spec(spec)
    assert executor._codex_tools == ["Shell"]


def test_from_spec_extracts_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``from_spec`` passes through ``llm.connection``.
    """
    _patch_ensure_sdk(monkeypatch)

    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="agents_sdk"),
        llm=LLMConfig(
            model="gpt-5.4",
            connection={"api_key": "sk-test"},
        ),
    )
    executor = AgentsSdkExecutor.from_spec(spec)
    assert executor._connection == {"api_key": "sk-test"}


def test_from_spec_no_llm_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``from_spec`` with no LLM config raises ``AssertionError``.
    """
    _patch_ensure_sdk(monkeypatch)

    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="agents_sdk"),
    )
    with pytest.raises(AssertionError):
        AgentsSdkExecutor.from_spec(spec)


def test_max_context_tokens_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``max_context_tokens()`` returns ``None`` — SDK manages
    context.
    """
    _patch_ensure_sdk(monkeypatch)

    spec = AgentSpec(
        spec_version=1,
        executor=ExecutorSpec(type="agents_sdk"),
        llm=LLMConfig(model="gpt-5.4"),
    )
    executor = AgentsSdkExecutor.from_spec(spec)
    assert executor.max_context_tokens() is None


# ── _make_function_tool ──────────────────────────────────


@pytest.mark.asyncio
async def test_make_function_tool_calls_call_tool(
    monkeypatch: pytest.MonkeyPatch,
    executor_context: ExecutorContext,
) -> None:
    """
    The function tool wrapper calls ``context.call_tool()``
    and returns the result content.
    """
    _patch_function_tool(monkeypatch)

    schema = {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
            },
        },
    }
    tool = _make_function_tool(schema, executor_context)
    # Invoke the tool's on_invoke_tool callback.
    result = await tool.on_invoke_tool(
        None,
        json.dumps({"path": "/tmp/test"}),
    )
    # The stub call_tool returns "stub output".
    assert result == "stub output"


@pytest.mark.asyncio
async def test_make_function_tool_preserves_name(
    monkeypatch: pytest.MonkeyPatch,
    executor_context: ExecutorContext,
) -> None:
    """
    The function tool has the correct name and description.
    """
    _patch_function_tool(monkeypatch)

    schema = {
        "type": "function",
        "function": {
            "name": "custom_tool",
            "description": "Does custom things",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    tool = _make_function_tool(schema, executor_context)
    assert tool.name == "custom_tool"
    assert tool.description == "Does custom things"


# ── _messages_to_input ───────────────────────────────────


def test_messages_to_input_passthrough() -> None:
    """
    Messages are passed through unchanged.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = _messages_to_input(messages)
    assert result is messages


# ── Monkeypatch helpers ──────────────────────────────────


def _patch_agents_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Patch ``agents.ModelSettings`` with a simple mock.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    import types

    @dataclass
    class _MockModelSettings:
        """
        Mock ModelSettings.

        :param temperature: Temperature setting.
        :param top_p: Top-p setting.
        :param max_tokens: Max tokens setting.
        :param reasoning: Reasoning config.
        :param extra_body: Extra body params.
        """

        temperature: float | None = None
        top_p: float | None = None
        max_tokens: int | None = None
        reasoning: Any = None
        extra_body: dict[str, Any] | None = None

    mock_agents = types.ModuleType("agents")
    mock_agents.ModelSettings = _MockModelSettings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", mock_agents)


def _patch_reasoning_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Patch ``openai.types.shared.Reasoning`` with a mock.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    import types

    @dataclass
    class _MockReasoning:
        """
        Mock Reasoning type.

        :param effort: Reasoning effort level.
        :param summary: Summary mode.
        """

        effort: str = "medium"
        summary: str = "detailed"

    shared_mod = types.ModuleType("openai.types.shared")
    shared_mod.Reasoning = _MockReasoning  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "openai.types.shared",
        shared_mod,
    )
    # Ensure parent modules exist.
    if "openai" not in sys.modules:
        openai_mod = types.ModuleType("openai")
        monkeypatch.setitem(sys.modules, "openai", openai_mod)
    if "openai.types" not in sys.modules:
        types_mod = types.ModuleType("openai.types")
        monkeypatch.setitem(
            sys.modules,
            "openai.types",
            types_mod,
        )


def _patch_response_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Patch ``openai.types.responses.ResponseTextDeltaEvent``.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    import types

    resp_mod = types.ModuleType("openai.types.responses")
    resp_mod.ResponseTextDeltaEvent = _MockTextDelta  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "openai.types.responses",
        resp_mod,
    )
    # Ensure parent modules exist.
    if "openai" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "openai",
            types.ModuleType("openai"),
        )
    if "openai.types" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "openai.types",
            types.ModuleType("openai.types"),
        )


def _patch_ensure_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Patch ``_ensure_sdk`` to return a mock agents module.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import types

    mock_agents = types.ModuleType("agents")
    monkeypatch.setattr(
        "agent_plane.runtime.executors.agents_sdk._ensure_sdk",
        lambda: mock_agents,
    )


def _patch_function_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Patch ``agents.FunctionTool`` with a simple mock.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    import sys
    import types

    @dataclass
    class _MockFunctionTool:
        """
        Mock FunctionTool.

        :param name: Tool name.
        :param description: Tool description.
        :param params_json_schema: JSON schema for params.
        :param on_invoke_tool: Async callback.
        """

        name: str = ""
        description: str = ""
        params_json_schema: dict[str, Any] = field(
            default_factory=dict,
        )
        on_invoke_tool: Any = None

    mock_agents = types.ModuleType("agents")
    mock_agents.FunctionTool = _MockFunctionTool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", mock_agents)
