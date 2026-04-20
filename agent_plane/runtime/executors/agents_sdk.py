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

import asyncio
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, Union
from uuid import uuid4

from mcp.types import ServerNotification
from pydantic import BaseModel, ConfigDict, Field, RootModel
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


_otel_instrumentor_initialized = False


def _ensure_otel_instrumentor() -> None:
    """
    Install the OTel instrumentation for the OpenAI Agents SDK.

    The SDK has its own proprietary tracing system; this bridge
    converts those traces into OTel spans that nest under the
    agent-plane ``invoke_agent`` span. Called once per process at
    executor creation.

    No-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset (telemetry
    disabled) or when the optional
    ``opentelemetry-instrumentation-openai-agents-v2`` package is
    not installed.
    """
    global _otel_instrumentor_initialized
    if _otel_instrumentor_initialized:
        return
    _otel_instrumentor_initialized = True

    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    # Content capture forwards through to the instrumentor's own
    # env var so message bodies end up in gen_ai.input.messages /
    # gen_ai.output.messages span attributes.
    if os.environ.get("AGENT_PLANE_OTEL_CAPTURE_CONTENT", "").strip().lower() in (
        "true",
        "1",
        "yes",
    ):
        os.environ.setdefault(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
            "span_and_event",
        )

    try:
        from opentelemetry.instrumentation.openai_agents import (
            OpenAIAgentsInstrumentor,
        )
    except ImportError:
        _logger.info(
            "OpenAIAgentsInstrumentor not installed; "
            "install 'agent-plane[agents-sdk]' to enable "
            "OTel tracing for the OpenAI Agents SDK executor."
        )
        return

    try:
        OpenAIAgentsInstrumentor().instrument()
    except Exception:
        _logger.exception(
            "failed to install OpenAIAgentsInstrumentor; "
            "OpenAI Agents SDK spans will not appear in traces"
        )


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
    Check if ``web_search`` is in the builtins list.

    :param builtins: The agent spec's builtin tool configs.
    :returns: True if ``web_search`` is declared.
    """
    return any(b.name == "web_search" for b in builtins)


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

    if name == "tool_output":
        return _map_tool_output(event.item)

    if name == "tool_called":
        # Only emit if the item already has output populated
        # (true for function tools, false for MCP tools like
        # codex where the result arrives in a later
        # "tool_output" event).
        item_output = getattr(event.item, "output", None)
        if item_output is not None:
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


def _map_tool_output(item: Any) -> ToolCallObserved | None:
    """
    Map a tool output item to ``ToolCallObserved``.

    The ``tool_output`` event fires AFTER the tool executes and
    carries the result. This is the preferred source for
    ``ToolCallObserved`` because it includes the actual output.
    For MCP tools (e.g. Codex), the ``tool_called`` event's
    output is always empty — the real result only appears here.

    :param item: The SDK's ``ToolCallOutputItem``.
    :returns: A ``ToolCallObserved`` event, or ``None``.
    """
    raw = getattr(item, "raw_item", None)
    if raw is None:
        return None

    # raw is a FunctionCallOutput dict or similar.
    if isinstance(raw, dict):
        call_id = raw.get("call_id", str(uuid4().hex[:12]))
        output = raw.get("output", "")
    else:
        call_id = getattr(raw, "call_id", None) or str(
            uuid4().hex[:12],
        )
        output = getattr(raw, "output", "")

    # The item-level output is the coerced string form.
    item_output = getattr(item, "output", None)
    result_str = str(item_output) if item_output is not None else str(output) if output else ""

    return ToolCallObserved(
        call_id=call_id,
        name="codex",
        arguments={},
        result=result_str,
        status="success",
        duration_ms=0.0,
    )


def _map_tool_called(item: Any) -> ToolCallObserved | None:
    """
    Map a tool call invocation to ``ToolCallObserved``.

    The ``tool_called`` event fires when a tool is invoked but
    BEFORE the result is available. For function tools, ``output``
    is populated; for MCP tools (e.g. Codex), it is empty — the
    result arrives later in a ``tool_output`` event.

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


class _CodexEventMsg(BaseModel):
    """
    The ``params.msg`` payload inside a Codex ``codex/event``
    notification. Codex uses ``msg.type`` as a discriminator for
    event kinds (``reasoning``, ``exec_command_begin``,
    ``agent_message_delta``, etc.). All other fields are
    kind-specific and accepted permissively.
    """

    model_config = ConfigDict(extra="allow")
    type: str = ""


class _CodexEventParams(BaseModel):
    """
    The ``params`` block of a Codex ``codex/event`` notification.
    """

    model_config = ConfigDict(extra="allow")
    msg: _CodexEventMsg = Field(default_factory=_CodexEventMsg)


class _CodexEventNotification(BaseModel):
    """
    Pydantic model for Codex's vendor ``codex/event`` MCP
    notification. The MCP spec does not define this method, so
    the Python SDK's default ``ServerNotification`` union rejects
    it. :class:`_PermissiveServerNotification` adds this variant
    so the notification reaches the session's message handler
    instead of being logged-and-dropped.
    """

    model_config = ConfigDict(extra="allow")
    method: Literal["codex/event"]
    params: _CodexEventParams


# Extend the default server-notification union with the Codex
# vendor method. ``_CodexEventNotification`` is listed first so
# Pydantic matches it before trying the spec variants.
_PermissiveRoot = Union[  # noqa: UP007 — RootModel needs typing.Union at runtime
    _CodexEventNotification,
    ServerNotification.model_fields["root"].annotation,  # type: ignore[misc]
]


class _PermissiveServerNotification(RootModel[_PermissiveRoot]):
    """
    Server-notification union that also accepts Codex's vendor
    ``codex/event`` method. Used as the session's
    ``_receive_notification_type`` so Codex streaming events
    pass Pydantic validation and reach the message handler.
    """


_CODEX_REASONING_MSG_TYPES: frozenset[str] = frozenset(
    {
        # The model's planning/reasoning text — the thing the
        # user explicitly wants to see mid-turn.
        "reasoning",
        "agent_reasoning",
        # Shell commands being executed inside Codex — visible
        # signal of progress that helps explain the wait.
        "exec_command_begin",
    },
)


class _CodexSessionRewriter:
    """
    Tracks Codex ``threadId`` for within-turn session continuity.

    Filters ``codex-reply`` from tool discovery so the LLM
    only sees ``codex``. Rewrites subsequent ``codex`` calls
    to ``codex-reply`` with the stored ``threadId``. Scoped
    to a single turn's ``codex mcp-server`` subprocess: a
    fresh instance is constructed in :func:`_build_agent`
    each turn because the subprocess is respawned per turn.
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


def _build_codex_home(
    workspace: str,
    connection: dict[str, str] | None,
) -> str:
    """
    Create an isolated ``CODEX_HOME`` for the Codex subprocess.

    The Codex CLI reads auth from ``$CODEX_HOME/auth.json``.
    Using a per-workspace home avoids mutating the global
    ``~/.codex/`` directory, which is critical for multi-tenant
    servers where different agents may use different API keys.

    :param workspace: The agent's workspace directory, e.g.
        ``"/tmp/storage/workspace"``.
    :param connection: Per-provider connection overrides from
        the agent spec, e.g. ``{"api_key": "sk-..."}``.
    :returns: Path to the isolated CODEX_HOME directory.
    """
    codex_home = Path(workspace) / ".codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)

    api_key: str | None = None
    if connection is not None:
        api_key = connection.get("api_key")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        auth_path = codex_home / "auth.json"
        auth_path.write_text(
            json.dumps(
                {"auth_mode": "apikey", "OPENAI_API_KEY": api_key},
            ),
        )
    else:
        _logger.warning(
            "codex: no API key found in agent config or environment — codex auth.json not written",
        )
    return str(codex_home)


def _build_codex_mcp(
    codex_tools: list[str],
    rewriter: _CodexSessionRewriter,
    workspace: str,
    connection: dict[str, str] | None = None,
) -> Any | None:
    """
    Build a Codex MCP server with session rewriting.

    Returns ``None`` if no Codex tools are declared or the
    ``codex`` binary is not on PATH. The returned server
    intercepts ``call_tool`` to rewrite ``codex`` calls to
    ``codex-reply`` with the stored ``threadId`` for session
    continuity.

    Each agent gets an isolated ``CODEX_HOME`` under its
    workspace so that different agents on a multi-tenant
    server don't share or overwrite each other's auth.

    :param codex_tools: Codex tool names (e.g. ``["Shell"]``).
    :param rewriter: The session rewriter for this conversation.
    :param workspace: Working directory for Codex, e.g.
        ``"/tmp/storage/workspace"``.
    :param connection: Per-provider connection overrides, e.g.
        ``{"api_key": "sk-..."}``. Used to provision auth
        for the Codex subprocess.
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

    codex_home = _build_codex_home(workspace, connection)

    return _make_session_aware_mcp_server(
        rewriter=rewriter,
        params={
            "command": "codex",
            "args": ["mcp-server"],
            "cwd": workspace,
            # CODEX_HOME isolates auth per-agent. The MCP
            # default env provides HOME/PATH/SHELL; we add
            # CODEX_HOME so the subprocess reads its own
            # auth.json instead of the global ~/.codex/.
            "env": {"CODEX_HOME": codex_home},
        },
        name="Codex CLI",
        # Long timeout — Codex sessions can be long-lived.
        client_session_timeout_seconds=360000,
        # Static filter — block codex-reply so the LLM only
        # sees codex. Session rewriting in call_tool handles
        # the rest. Dynamic (callable) filters require
        # run_context/agent which aren't available at
        # construction time.
        tool_filter={"blocked_names": ["codex-reply"]},
    )


def _make_session_aware_mcp_server(
    rewriter: _CodexSessionRewriter,
    **kwargs: Any,
) -> Any:
    """
    Create an ``MCPServerStdio`` subclass instance that

    - intercepts ``call_tool`` for Codex session rewriting, and
    - captures Codex's vendor ``codex/event`` notification
      stream onto ``self.codex_events`` (an ``asyncio.Queue``),
      so the executor can interleave Codex reasoning/progress
      with the SDK's event stream.

    Uses runtime subclassing so the instance passes all
    ``isinstance`` checks the SDK performs internally.

    :param rewriter: The ``_CodexSessionRewriter`` for this turn.
    :param kwargs: Forwarded to ``MCPServerStdio``. Must not
        include ``message_handler`` — this factory wires its own.
    :returns: A session-aware MCP server instance.
    """
    from agents.mcp import MCPServerStdio

    class _SessionAware(MCPServerStdio):
        """
        ``MCPServerStdio`` subclass that rewrites ``codex`` →
        ``codex-reply`` and captures ``codex/event`` notifications.
        """

        def __init__(self, **inner_kwargs: Any) -> None:
            # Queue carrying parsed ``_CodexEventMsg`` payloads —
            # consumed by the executor in ``_stream_sdk_turn``.
            # Unbounded: Codex can fire hundreds of events per
            # tool call; dropping them would defeat the purpose.
            self.codex_events: asyncio.Queue[_CodexEventMsg] = asyncio.Queue()
            super().__init__(
                message_handler=self._handle_message,
                **inner_kwargs,
            )

        async def _handle_message(self, message: Any) -> None:
            """
            Forward ``codex/event`` notifications onto
            :attr:`codex_events`. Non-Codex messages are a no-op
            (the default handler is also a no-op).
            """
            root = getattr(message, "root", None)
            if isinstance(root, _CodexEventNotification):
                await self.codex_events.put(root.params.msg)

        async def connect(self) -> None:
            """
            Connect and replace the session's
            ``_receive_notification_type`` with the permissive
            union so Pydantic accepts ``codex/event``. Without
            this swap, ``BaseSession._receive_loop`` logs a
            ``ValidationError`` on every event and never calls
            ``message_handler``.
            """
            await super().connect()
            if self.session is not None:
                self.session._receive_notification_type = _PermissiveServerNotification

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None,
            meta: dict[str, Any] | None = None,
        ) -> Any:
            """
            Rewrite ``codex`` → ``codex-reply`` and capture
            ``threadId``.

            :param tool_name: The MCP tool name.
            :param arguments: The tool arguments dict.
            :param meta: Optional metadata (forwarded).
            :returns: The ``CallToolResult``.
            """
            name, args = rewriter.rewrite_call(
                tool_name,
                arguments,
            )
            result = await super().call_tool(
                name,
                args,
                meta,
            )
            rewriter.capture_thread_id(result)
            return result

    return _SessionAware(**kwargs)


# Rewriter lifetime is scoped to a single turn's MCP server:
# ``MCPServerManager`` spawns a fresh ``codex mcp-server`` process
# per turn and kills it at turn end, so the ``threadId`` captured
# from turn N is invalid (``Session not found``) when replayed
# against the fresh subprocess in turn N+1. Within a turn, multiple
# codex calls still chain correctly because the subprocess and
# rewriter share the same lifetime.


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
        _ensure_otel_instrumentor()
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
        No-op. Codex session state is scoped to the per-turn MCP
        server — a fresh rewriter is built in :func:`_build_agent`
        for each turn's subprocess.

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
        MCP servers (Codex) are connected via
        ``MCPServerManager`` for proper lifecycle management.

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
        agent, mcp_servers = _build_agent(
            self,
            tools,
            system_prompt,
            llm_config,
            context,
        )
        input_items = _messages_to_input(messages)

        if mcp_servers:
            # MCP servers must be connected before the SDK
            # can use them. MCPServerManager handles the
            # connect/cleanup lifecycle.
            from agents.mcp import MCPServerManager

            async with MCPServerManager(
                mcp_servers,
            ) as mgr:
                # Replace agent's mcp_servers with the
                # connected ones from the manager.
                agent.mcp_servers = mgr.active_servers
                codex_events = _find_codex_event_queue(mgr.active_servers)
                async for event in _stream_sdk_turn(
                    agent,
                    input_items,
                    codex_events=codex_events,
                ):
                    yield event
        else:
            async for event in _stream_sdk_turn(
                agent,
                input_items,
            ):
                yield event


def _find_codex_event_queue(
    active_servers: list[Any],
) -> asyncio.Queue[_CodexEventMsg] | None:
    """
    Return the first connected MCP server's ``codex_events``
    queue, if any. Returns ``None`` when no Codex MCP is
    attached to this turn (e.g. an agent with only
    ``web_search``).

    :param active_servers: The ``MCPServerManager.active_servers``
        list after ``__aenter__``.
    """
    for server in active_servers:
        queue = getattr(server, "codex_events", None)
        if isinstance(queue, asyncio.Queue):
            return queue
    return None


def _build_agent(
    executor: AgentsSdkExecutor,
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
) -> tuple[Any, list[Any]]:
    """
    Construct an Agents SDK ``Agent`` from executor config.

    Returns the agent and its MCP servers separately so the
    caller can manage MCP lifecycle via ``MCPServerManager``.

    :param executor: The executor instance.
    :param tools: Tool schemas in OpenAI format.
    :param system_prompt: System instructions.
    :param llm_config: LLM configuration.
    :param context: Agent-plane executor context.
    :returns: A tuple of (agent, mcp_servers). The caller
        must connect MCP servers before running the agent.
    """
    sdk = _ensure_sdk()

    # Only wrap function-type tools. Passthrough tools
    # (e.g. web_search with type="web_search_preview")
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
        rewriter = _CodexSessionRewriter()
        workspace = str(
            context.storage_dir / "workspace",
        )
        codex_mcp = _build_codex_mcp(
            executor._codex_tools,
            rewriter,
            workspace,
            connection=executor._connection,
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

    agent = sdk.Agent(
        name="agent",
        instructions=system_prompt,
        model=model,
        model_settings=model_settings,
        tools=[*function_tools, *hosted_tools],
        # MCP servers are passed here but must be connected
        # by the caller via MCPServerManager before running.
        mcp_servers=mcp_servers,
    )
    return agent, mcp_servers


async def _stream_sdk_turn(
    agent: Any,
    input_items: list[dict[str, Any]],
    codex_events: asyncio.Queue[_CodexEventMsg] | None = None,
) -> AsyncIterator[ExecutorEvent]:
    """
    Run ``Runner.run_streamed()`` and yield executor events,
    interleaving Codex reasoning / progress events from
    ``codex_events`` when provided.

    Handles ``MaxTurnsExceeded`` and generic exceptions,
    converting them to executor event types.

    :param agent: The configured Agents SDK ``Agent``.
    :param input_items: Responses API input items.
    :param codex_events: Per-turn Codex event queue populated by
        ``_SessionAware._handle_message``. When present,
        ``codex/event`` payloads (reasoning text, command begin)
        are surfaced as :class:`ReasoningChunk` events so the
        TUI can render live progress during the tool call
        window. ``None`` for agents without Codex tools.
    """
    sdk = _ensure_sdk()
    try:
        result = sdk.Runner.run_streamed(
            agent,
            input=input_items,
            max_turns=_SDK_MAX_TURNS,
        )
        async for event in _merge_sdk_and_codex_events(
            result.stream_events(),
            codex_events,
        ):
            yield event
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


async def _merge_sdk_and_codex_events(
    sdk_stream: AsyncIterator[Any],
    codex_events: asyncio.Queue[_CodexEventMsg] | None,
) -> AsyncIterator[ExecutorEvent]:
    """
    Interleave openai-agents SDK events with Codex
    ``codex/event`` notifications.

    Keeps one outstanding ``__anext__`` on the SDK stream and
    one outstanding ``get()`` on the Codex queue, yielding
    whichever completes first. On SDK exhaustion, cancels the
    pending Codex task and returns. A one-line race between
    the two sources; no polling.

    :param sdk_stream: The SDK's async event iterator.
    :param codex_events: Codex notification queue, or ``None``
        when the agent has no Codex MCP attached.
    """
    if codex_events is None:
        async for event in sdk_stream:
            mapped = _map_event(event)
            if mapped is not None:
                yield mapped
        return

    sdk_iter = sdk_stream.__aiter__()
    sdk_task: asyncio.Task[Any] | None = asyncio.ensure_future(sdk_iter.__anext__())
    codex_task: asyncio.Task[_CodexEventMsg] | None = asyncio.ensure_future(
        codex_events.get(),
    )
    try:
        while sdk_task is not None:
            pending: set[asyncio.Task[Any]] = {sdk_task}
            if codex_task is not None:
                pending.add(codex_task)
            done, _ = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is sdk_task:
                    try:
                        sdk_event = task.result()
                    except StopAsyncIteration:
                        sdk_task = None
                        continue
                    mapped = _map_event(sdk_event)
                    if mapped is not None:
                        yield mapped
                    sdk_task = asyncio.ensure_future(sdk_iter.__anext__())
                else:
                    msg: _CodexEventMsg = task.result()
                    mapped_codex = _codex_msg_to_executor_event(msg)
                    if mapped_codex is not None:
                        yield mapped_codex
                    codex_task = asyncio.ensure_future(codex_events.get())
    finally:
        if codex_task is not None and not codex_task.done():
            codex_task.cancel()
        if sdk_task is not None and not sdk_task.done():
            sdk_task.cancel()


def _codex_msg_to_executor_event(msg: _CodexEventMsg) -> ExecutorEvent | None:
    """
    Map a Codex ``codex/event`` ``msg`` payload to an executor
    event, or ``None`` to drop.

    Only a small subset is surfaced — the types that actually
    tell the user what Codex is doing right now:

    - ``reasoning`` / ``agent_reasoning`` → :class:`ReasoningChunk`
      with the reasoning text. This is the main thing the user
      asked to see.
    - ``exec_command_begin`` → :class:`ReasoningChunk` containing
      the command string as ``$ <cmd>``, so long shell calls
      don't sit silently.

    All other Codex events (lifecycle, token_count, internal
    item bookkeeping) are dropped — they'd be noise in the TUI.

    :param msg: The parsed ``msg`` block from a ``codex/event``
        notification.
    """
    if msg.type not in _CODEX_REASONING_MSG_TYPES:
        return None
    raw = msg.model_dump()
    if msg.type == "exec_command_begin":
        command = raw.get("command")
        if isinstance(command, list):
            rendered = " ".join(str(c) for c in command)
        elif isinstance(command, str):
            rendered = command
        else:
            return None
        return ReasoningChunk(
            delta=f"$ {rendered}\n",
            event_type="reasoning_summary",
        )
    # ``reasoning`` / ``agent_reasoning`` — text body lives in one
    # of a handful of field names depending on Codex build.
    text = raw.get("text") or raw.get("delta") or raw.get("content") or ""
    if not isinstance(text, str) or not text:
        return None
    return ReasoningChunk(
        delta=text,
        event_type="reasoning_summary",
    )
