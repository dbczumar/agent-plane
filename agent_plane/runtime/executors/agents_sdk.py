"""AgentsSdkExecutor: run agents using the OpenAI Agents SDK.

Uses the ``openai-agents`` Python package to run an agent with the
OpenAI Agents SDK. The SDK manages its own agent loop (tool calls,
retries, context). This executor translates the SDK's streaming
events into agent-plane executor events.

Server-side and client-side tools are registered as SDK function
tools whose implementations call ``context.call_tool()``. The SDK
executes them during its internal agent loop.

Requirements::

    pip install openai-agents

Environment::

    OPENAI_API_KEY – API key for OpenAI (or set via llm.connection)
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from typing_extensions import Self

from agent_plane.runtime.executors.base import (
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    NativeToolOutput,
    ReasoningChunk,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
)
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import (
    BuiltinToolConfig,
    LLMConfig,
)

_logger = logging.getLogger(__name__)

# High ceiling for the SDK's internal turn limit. The workflow's
# max_iterations is the authoritative limit — this just prevents
# the SDK from running indefinitely if the workflow doesn't stop
# it first.
_SDK_MAX_TURNS = 200


def _ensure_sdk() -> Any:
    """
    Import and return the ``agents`` module.

    :returns: The ``agents`` module.
    :raises ImportError: If the package is not installed.
    """
    try:
        import agents

        return agents
    except ImportError as exc:
        raise ImportError(
            "AgentsSdkExecutor requires 'openai-agents'. "
            "Install with: pip install 'agent-plane[agents-sdk]'"
        ) from exc


def _extract_codex_tools(spec: AgentSpec) -> list[str]:
    """
    Extract ``codex:``-prefixed tool names from the agent spec.

    Strips the ``codex:`` prefix and returns bare tool names.

    :param spec: The agent spec.
    :returns: List of tool names, e.g. ``["Shell", "ApplyPatch"]``.
    """
    result: list[str] = []
    for builtin in spec.tools.builtins:
        name = builtin.name
        if name.startswith("codex:"):
            result.append(name[len("codex:") :])
    return result


def _has_web_search(
    builtins: list[BuiltinToolConfig],
) -> bool:
    """
    Check if ``web_search_openai`` is in the builtins list.

    :param builtins: The agent spec's builtin tool configs.
    :returns: True if ``web_search_openai`` is declared.
    """
    return any(b.name == "web_search_openai" for b in builtins)


def _build_model_settings(llm_config: LLMConfig) -> Any:
    """
    Map ``LLMConfig`` fields to the SDK's ``ModelSettings``.

    Extracts known keys from ``llm_config.extra`` and maps them
    to ``ModelSettings`` fields. Remaining keys are passed
    through via ``extra_body``.

    :param llm_config: The agent's LLM configuration.
    :returns: A ``ModelSettings`` instance.
    """
    from agents import ModelSettings

    extra = dict(llm_config.extra)
    reasoning_effort = extra.pop("reasoning_effort", None)

    settings = ModelSettings(
        temperature=extra.pop("temperature", None),
        top_p=extra.pop("top_p", None),
        max_tokens=extra.pop("max_completion_tokens", None),
    )
    if reasoning_effort:
        from openai.types.shared import Reasoning

        settings.reasoning = Reasoning(
            effort=reasoning_effort,
            # "detailed" enables reasoning summary streaming.
            summary="detailed",
        )
    if extra:
        settings.extra_body = extra
    return settings


def _build_openai_client(
    connection: dict[str, str] | None,
    timeout: int = 300,
    max_retries: int = 3,
) -> Any | None:
    """
    Build an ``AsyncOpenAI`` client from connection params.

    Returns ``None`` when no connection overrides are provided —
    the SDK reads ``OPENAI_API_KEY`` from the environment.

    :param connection: Per-provider connection overrides, e.g.
        ``{"api_key": "sk-...", "base_url": "https://..."}``.
        ``None`` uses environment defaults.
    :param timeout: Request timeout in seconds, e.g. ``300``.
    :param max_retries: Maximum retry attempts, e.g. ``3``.
    :returns: An ``AsyncOpenAI`` client, or ``None``.
    """
    if connection is None:
        return None
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=connection.get("api_key"),
        base_url=connection.get("base_url"),
        timeout=float(timeout),
        max_retries=max_retries,
    )


def _build_model(
    model_name: str,
    client: Any | None,
) -> Any:
    """
    Build the model parameter for the ``Agent`` constructor.

    When a custom client is provided, wraps it in
    ``OpenAIResponsesModel`` so the SDK uses it. Otherwise
    returns the model name string (SDK uses its default client).

    :param model_name: The model identifier, e.g. ``"gpt-5.4"``.
    :param client: An ``AsyncOpenAI`` client, or ``None``.
    :returns: A model name string or ``OpenAIResponsesModel``.
    """
    if client is None:
        return model_name
    from agents.models.openai_responses import (
        OpenAIResponsesModel,
    )

    return OpenAIResponsesModel(
        model=model_name,
        openai_client=client,
    )


def _make_function_tool(
    schema: dict[str, Any],
    context: ExecutorContext,
) -> Any:
    """
    Wrap an agent-plane tool schema as an Agents SDK function tool.

    The tool's implementation calls ``context.call_tool()`` which
    routes to server-side execution or client-side parking
    transparently.

    :param schema: OpenAI-format tool schema, e.g.
        ``{"type": "function", "function": {"name": "Read", ...}}``.
    :param context: Agent-plane executor context providing
        ``call_tool``.
    :returns: An Agents SDK ``FunctionTool`` instance.
    """
    from agents import FunctionTool

    func_spec = schema["function"]
    name = func_spec["name"]
    # description is optional in the OpenAI tool schema spec.
    # Empty string is the correct default — the SDK accepts it.
    desc = func_spec.get("description") or ""
    params_schema = func_spec.get(
        "parameters",
        {"type": "object", "properties": {}},
    )

    async def _on_invoke(
        ctx: Any,
        args_json: str,
    ) -> str:
        """
        Execute the tool via agent-plane's call_tool.

        :param ctx: The SDK's run context (unused).
        :param args_json: JSON-encoded tool arguments.
        :returns: The tool's output string.
        """
        arguments = json.loads(args_json)
        req = ToolCallRequested(
            call_id=f"call_{uuid4().hex[:12]}",
            name=name,
            arguments=arguments,
        )
        result: ToolResult = await context.call_tool(req)
        return result.content

    return FunctionTool(
        name=name,
        description=desc,
        params_json_schema=params_schema,
        on_invoke_tool=_on_invoke,
    )


def _messages_to_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert agent-plane messages to Agents SDK input format.

    The messages are already in Responses API format, which the
    SDK accepts directly.

    :param messages: Responses API input items.
    :returns: The same items (pass-through).
    """
    return messages


def _map_event(
    event: Any,
) -> ExecutorEvent | None:
    """
    Map an Agents SDK ``StreamEvent`` to an executor event.

    Returns ``None`` for event types that should be silently
    ignored (unknown types, informational events).

    :param event: A ``StreamEvent`` from
        ``result.stream_events()``.
    :returns: The mapped executor event, or ``None``.
    """
    if event.type == "raw_response_event":
        return _map_raw_response_event(event.data)
    if event.type == "run_item_stream_event":
        return _map_run_item_event(event)
    return None


def _map_raw_response_event(data: Any) -> ExecutorEvent | None:
    """
    Map a raw Responses API streaming event to an executor event.

    :param data: The raw event data from the SDK.
    :returns: The mapped executor event, or ``None``.
    """
    from openai.types.responses import ResponseTextDeltaEvent

    if isinstance(data, ResponseTextDeltaEvent):
        return TextChunk(text=data.delta)

    # Check for reasoning summary by class name to avoid
    # importing types that may not exist in all SDK versions.
    cls_name = type(data).__name__
    if cls_name == "ResponseReasoningSummaryTextDeltaEvent":
        return ReasoningChunk(
            delta=data.delta,
            event_type="reasoning_summary",
        )

    return None


def _map_run_item_event(event: Any) -> ExecutorEvent | None:
    """
    Map a run-item-level streaming event to an executor event.

    :param event: A ``RunItemStreamEvent`` from the SDK.
    :returns: The mapped executor event, or ``None``.
    """
    name = getattr(event, "name", None)

    if name == "tool_called":
        return _map_tool_called(event.item)

    if name == "tool_search_output_created":
        raw_item = getattr(event.item, "raw_item", None)
        if raw_item is not None:
            # Convert to dict if it has model_dump (Pydantic).
            if hasattr(raw_item, "model_dump"):
                return NativeToolOutput(
                    item=raw_item.model_dump(),
                )
            return NativeToolOutput(item=dict(raw_item))

    return None


def _map_tool_called(item: Any) -> ToolCallObserved | None:
    """
    Map a completed tool call item to ``ToolCallObserved``.

    :param item: The SDK's tool call run item.
    :returns: A ``ToolCallObserved`` event, or ``None`` if the
        item lacks the expected attributes.
    """
    raw = getattr(item, "raw_item", None)
    if raw is None:
        return None

    call_id = getattr(raw, "call_id", None) or str(
        uuid4().hex[:12],
    )
    name = getattr(raw, "name", "unknown")
    arguments_str = getattr(raw, "arguments", "{}")

    try:
        arguments = json.loads(arguments_str)
    except (json.JSONDecodeError, TypeError):
        arguments = {}

    output = getattr(item, "output", None)
    result_str = str(output) if output is not None else ""

    return ToolCallObserved(
        call_id=call_id,
        name=name,
        arguments=arguments,
        result=result_str,
        status="success",
        duration_ms=0.0,
    )


# ── Codex MCP integration (Layer 2) ─────────────────────


class _CodexSessionRewriter:
    """
    Tracks Codex ``threadId`` for session continuity.

    Filters ``codex-reply`` from tool discovery so the LLM
    only sees ``codex``. Rewrites subsequent ``codex`` calls
    to ``codex-reply`` with the stored ``threadId``.

    :param thread_id: The current Codex thread ID, or ``None``
        before the first call.
    """

    def __init__(self) -> None:
        self._thread_id: str | None = None

    def tool_filter(
        self,
        ctx: Any,
        tool: Any,
    ) -> bool:
        """
        Filter ``codex-reply`` from tool discovery.

        The LLM only sees ``codex``. Session continuity is
        handled transparently by ``rewrite_call``.

        :param ctx: The SDK's ``ToolFilterContext`` (unused).
        :param tool: The MCP tool definition.
        :returns: True to include, False to exclude.
        """
        name = getattr(tool, "name", "")
        return name != "codex-reply"

    def rewrite_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Rewrite ``codex`` → ``codex-reply`` if session exists.

        :param tool_name: The tool name from the SDK.
        :param arguments: The tool arguments dict.
        :returns: Possibly rewritten (tool_name, arguments).
        """
        if tool_name != "codex":
            return tool_name, arguments
        args = dict(arguments or {})
        if self._thread_id is None:
            # First call — inject defaults.
            args.setdefault("approval-policy", "never")
            args.setdefault("sandbox", "workspace-write")
            return "codex", args
        # Subsequent call — rewrite to codex-reply.
        return "codex-reply", {
            "prompt": args.get("prompt", ""),
            "threadId": self._thread_id,
        }

    def capture_thread_id(
        self,
        result: Any,
    ) -> None:
        """
        Extract ``threadId`` from a Codex tool result.

        :param result: The ``CallToolResult`` from the MCP call.
        """
        sc = getattr(result, "structuredContent", None)
        if isinstance(sc, dict):
            tid = sc.get("threadId")
            if tid is not None:
                self._thread_id = tid

    def clear_thread_id(self) -> None:
        """
        Clear stored ``threadId`` (e.g. after MCP restart).
        """
        self._thread_id = None


def _build_codex_mcp(
    codex_tools: list[str],
    rewriter: _CodexSessionRewriter,
    workspace: str,
) -> Any | None:
    """
    Build a Codex MCP server with session rewriting.

    Returns ``None`` if no Codex tools are declared or the
    ``codex`` binary is not on PATH. The returned server
    intercepts ``call_tool`` to rewrite ``codex`` calls to
    ``codex-reply`` with the stored ``threadId`` for session
    continuity.

    :param codex_tools: Codex tool names (e.g. ``["Shell"]``).
    :param rewriter: The session rewriter for this conversation.
    :param workspace: Working directory for Codex, e.g.
        ``"/tmp/storage/workspace"``.
    :returns: A ``_SessionAwareMcpServer`` instance, or ``None``.
    """
    if not codex_tools:
        return None
    if shutil.which("codex") is None:
        _logger.warning(
            "codex: tools declared but 'codex' binary not"
            " found on PATH — Codex tools will be"
            " unavailable",
        )
        return None

    return _SessionAwareMcpServer(
        rewriter=rewriter,
        params={
            "command": "codex",
            "args": ["mcp-server"],
            "cwd": workspace,
        },
        name="Codex CLI",
        # Long timeout — Codex sessions can be long-lived.
        client_session_timeout_seconds=360000,
        tool_filter=rewriter.tool_filter,
    )


class _SessionAwareMcpServer:
    """
    MCPServerStdio wrapper with Codex session rewriting.

    Delegates all attributes and methods to an inner
    ``MCPServerStdio`` via ``__getattr__``. Only ``call_tool``
    is intercepted for session rewriting.

    :param rewriter: The ``_CodexSessionRewriter`` for this
        conversation.
    :param kwargs: Forwarded to ``MCPServerStdio``.
    """

    def __init__(
        self,
        rewriter: _CodexSessionRewriter,
        **kwargs: Any,
    ) -> None:
        from agents.mcp import MCPServerStdio

        # Use object.__setattr__ to avoid triggering
        # __getattr__ during __init__.
        object.__setattr__(self, "_rewriter", rewriter)
        object.__setattr__(
            self,
            "_inner",
            MCPServerStdio(**kwargs),
        )

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the inner server.

        All methods and properties are forwarded except
        ``call_tool`` which is overridden explicitly.

        :param name: The attribute name.
        :returns: The attribute from the inner server.
        """
        return getattr(self._inner, name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """
        Intercept tool calls for Codex session rewriting.

        Rewrites ``codex`` → ``codex-reply`` with stored
        ``threadId`` when a session exists. Captures
        ``threadId`` from the result for future calls.

        :param tool_name: The MCP tool name, e.g.
            ``"codex"``.
        :param arguments: The tool arguments dict.
        :param meta: Optional metadata (forwarded).
        :returns: The ``CallToolResult``.
        """
        rewritten_name, rewritten_args = self._rewriter.rewrite_call(
            tool_name,
            arguments,
        )
        result = await self._inner.call_tool(
            rewritten_name,
            rewritten_args,
            meta,
        )
        self._rewriter.capture_thread_id(result)
        return result


# Per-conversation Codex session rewriters. Persisted across
# tasks so threadId survives between turns.
_codex_rewriters: dict[str, _CodexSessionRewriter] = {}


def _get_or_create_rewriter(
    conv_id: str,
) -> _CodexSessionRewriter:
    """
    Get or create a session rewriter for a conversation.

    :param conv_id: The conversation identifier, e.g.
        ``"conv_abc123"``.
    :returns: The rewriter for this conversation.
    """
    if conv_id not in _codex_rewriters:
        _codex_rewriters[conv_id] = _CodexSessionRewriter()
    return _codex_rewriters[conv_id]


class AgentsSdkExecutor(Executor):
    """
    Executor wrapping the OpenAI Agents SDK.

    The SDK manages the agent loop internally. Server-side and
    client-side tools are registered as function tools. The SDK
    streams events via ``Runner.run_streamed()``.

    :param model: The model identifier, e.g. ``"gpt-5.4"``.
    :param codex_tools: Codex tool names (stripped of ``codex:``
        prefix), e.g. ``["Shell", "ApplyPatch"]``. Empty for
        non-coding agents.
    :param builtins: Builtin tool configs from the agent spec.
    :param connection: Per-provider connection overrides, or
        ``None`` for environment defaults.
    :param request_timeout: Per-LLM-call timeout in seconds,
        e.g. ``300``.
    :param max_retries: Maximum retry attempts for the OpenAI
        client, e.g. ``3``.
    """

    def __init__(
        self,
        *,
        model: str,
        codex_tools: list[str],
        builtins: list[BuiltinToolConfig],
        connection: dict[str, str] | None = None,
        request_timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._codex_tools = codex_tools
        self._builtins = builtins
        self._connection = connection
        self._request_timeout = request_timeout
        self._max_retries = max_retries

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from the agent spec's config.

        Extracts model, ``codex:``-prefixed tools, connection
        overrides, and timeout/retry settings.

        :param spec: Agent spec with
            ``executor.type == "agents_sdk"``.
        :returns: Configured AgentsSdkExecutor.
        """
        _ensure_sdk()
        assert spec.llm is not None
        codex_tools = _extract_codex_tools(spec)
        return cls(
            model=spec.llm.model,
            codex_tools=codex_tools,
            builtins=spec.tools.builtins,
            connection=spec.llm.connection,
            request_timeout=spec.llm.request_timeout,
            max_retries=spec.llm.retry.max_attempts,
        )

    def max_context_tokens(self) -> int | None:
        """
        SDK manages its own context window.

        :returns: None — workflow skips compaction and @step.
        """
        return None

    def on_task_start(self, context: ExecutorContext) -> None:
        """
        Ensure workspace directory exists for Codex.

        :param context: Agent-plane capabilities and identifiers.
        """
        if self._codex_tools:
            workspace = context.storage_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

    def on_task_end(self, context: ExecutorContext) -> None:
        """
        No-op. Codex session rewriters persist in the module-level
        ``_codex_rewriters`` dict for session continuity.

        :param context: Agent-plane capabilities and identifiers.
        """

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Run one SDK turn as an async generator.

        Builds an ``Agent`` with function tools, calls
        ``Runner.run_streamed()``, and yields executor events.
        No queue bridge needed — the SDK is async-native.

        :param messages: Conversation history as Responses API
            input items.
        :param tools: Client-side and server-side tool schemas
            in OpenAI format.
        :param system_prompt: Assembled system instructions.
        :param llm_config: LLM configuration (model, extra,
            connection, timeout, retry).
        :param context: Agent-plane capabilities and
            identifiers.
        """
        agent = _build_agent(
            self,
            tools,
            system_prompt,
            llm_config,
            context,
        )
        input_items = _messages_to_input(messages)
        async for event in _stream_sdk_turn(
            agent,
            input_items,
        ):
            yield event


def _build_agent(
    executor: AgentsSdkExecutor,
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
) -> Any:
    """
    Construct an Agents SDK ``Agent`` from executor config.

    Attaches Codex MCP server when ``codex:`` tools are
    declared. The MCP server provides Shell and ApplyPatch
    capabilities with session continuity via
    ``_CodexSessionRewriter``.

    :param executor: The executor instance.
    :param tools: Tool schemas in OpenAI format.
    :param system_prompt: System instructions.
    :param llm_config: LLM configuration.
    :param context: Agent-plane executor context.
    :returns: A configured ``Agent`` instance.
    """
    sdk = _ensure_sdk()

    # Only wrap function-type tools. Passthrough tools
    # (e.g. web_search_openai with type="web_search_preview")
    # are handled via hosted_tools, not function tools.
    function_tools = [
        _make_function_tool(schema, context)
        for schema in tools
        if schema.get("type") == "function"
    ]
    hosted_tools: list[Any] = []
    if _has_web_search(executor._builtins):
        hosted_tools.append(sdk.WebSearchTool())

    # Codex MCP for coding tools (Shell, ApplyPatch).
    mcp_servers: list[Any] = []
    if executor._codex_tools:
        rewriter = _get_or_create_rewriter(
            context.conversation_id,
        )
        workspace = str(
            context.storage_dir / "workspace",
        )
        codex_mcp = _build_codex_mcp(
            executor._codex_tools,
            rewriter,
            workspace,
        )
        if codex_mcp is not None:
            mcp_servers.append(codex_mcp)

    model_settings = _build_model_settings(llm_config)
    client = _build_openai_client(
        executor._connection,
        executor._request_timeout,
        executor._max_retries,
    )
    model = _build_model(llm_config.model, client)

    return sdk.Agent(
        name="agent",
        instructions=system_prompt,
        model=model,
        model_settings=model_settings,
        tools=[*function_tools, *hosted_tools],
        mcp_servers=mcp_servers,
    )


async def _stream_sdk_turn(
    agent: Any,
    input_items: list[dict[str, Any]],
) -> AsyncIterator[ExecutorEvent]:
    """
    Run ``Runner.run_streamed()`` and yield executor events.

    Handles ``MaxTurnsExceeded`` and generic exceptions,
    converting them to executor event types.

    :param agent: The configured Agents SDK ``Agent``.
    :param input_items: Responses API input items.
    """
    sdk = _ensure_sdk()
    try:
        result = sdk.Runner.run_streamed(
            agent,
            input=input_items,
            max_turns=_SDK_MAX_TURNS,
        )
        async for event in result.stream_events():
            mapped = _map_event(event)
            if mapped is not None:
                yield mapped
        yield TurnComplete(text=result.final_output)
    except Exception as exc:
        cls_name = type(exc).__name__
        if cls_name == "MaxTurnsExceeded":
            yield TurnComplete(text=None)
        else:
            _logger.error("Agents SDK error: %s", exc)
            yield ExecutorError(
                message=f"Agents SDK error: {exc}",
                code=cls_name,
            )
