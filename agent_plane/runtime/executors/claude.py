"""ClaudeAgentsExecutor: run agents using the Claude Agent SDK.

Uses the ``claude-agent-sdk`` Python package to run Claude Code as the
underlying agent harness. The SDK manages its own internal agent loop
(tool calls, retries, context). This executor translates the SDK message
stream into agent-plane executor events.

Server-side tools (Bash, Read, Edit, etc.) are configured via
``allowed_tools`` — the SDK executes them with its built-in
implementations. Client-side tools are bridged through an in-process
MCP server backed by ``context.await_tool_output``, which parks the
call and blocks until the client delivers a result.

Requirements::

    pip install claude-agent-sdk

Environment::

    ANTHROPIC_API_KEY – API key for Claude
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from typing_extensions import Self

from agent_plane.runtime.executors.base import (
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
)
from agent_plane.spec import AgentSpec
from agent_plane.spec.types import LLMConfig

_logger = logging.getLogger(__name__)

# Claude Code built-in tools available via allowed_tools.
_BUILTIN_TOOLS = ["Bash", "Read", "Edit", "Write", "Glob", "Grep"]


def _ensure_sdk() -> Any:
    """
    Import and return the ``claude_agent_sdk`` module.

    :returns: The ``claude_agent_sdk`` module.
    :raises ImportError: If the package is not installed.
    """
    try:
        import claude_agent_sdk

        return claude_agent_sdk
    except ImportError as exc:
        raise ImportError(
            "ClaudeAgentsExecutor requires 'claude-agent-sdk'. "
            "Install with: pip install claude-agent-sdk"
        ) from exc


@dataclass
class _CallbackHolder:
    """
    Mutable holder for the current task's ``await_tool_output`` callback.

    Shared between the long-lived MCP server handlers and per-task setup.
    Updated to the current task's callback before each ``client.query()``
    call; cleared to ``None`` after the query completes. MCP handlers read
    it at call time (not at creation time), so they always route to the
    current task's parking infrastructure.

    :param callback: The current task's ``await_tool_output`` function,
        or ``None`` between tasks (no active query running).
    """

    callback: Callable[[ToolCallRequested], ToolResult] | None = None


@dataclass
class _ClientState:
    """
    Per-conversation SDK client state.

    Does not own the event loop — the ``_ClientRegistry`` manages
    loop lifecycle separately so that ``_get_or_create_client`` (which
    builds this) doesn't need to know about threading.

    :param client: The connected ``ClaudeSDKClient`` instance.
    :param model: Model name the client was created with, or ``None``
        if using the SDK default.
    :param callback_holder: Updated each task so MCP tool handlers
        route to the current task's ``await_tool_output``.
    :param last_used: ``time.monotonic()`` timestamp of the last
        ``on_task_end`` call. Used for TTL-based eviction.
    """

    client: Any
    model: str | None
    callback_holder: _CallbackHolder
    last_used: float


class _ClientRegistry:
    """
    Process-lifetime registry of SDK clients keyed by conversation ID.

    Each conversation gets its own event loop and background thread,
    isolating conversations from each other. A slow SDK call in one
    conversation does not block others.

    SDK subprocesses persist across tasks so bash session state (cwd,
    env vars, running processes) survives between conversation turns.

    DBOS serializes tasks per ``conversation_id`` (one task at a time),
    so each key is mutated by only one thread at a time. The internal
    lock guards cross-conversation operations such as eviction iterating
    the dict while another conversation registers a new entry.

    :param ttl_seconds: Idle time after which a client is disconnected
        and evicted, e.g. ``3600.0``.
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        # Per-conversation event loops. Created by get_or_create_loop
        # on first task, persist until eviction.
        self._loops: dict[
            str,
            tuple[asyncio.AbstractEventLoop, threading.Thread],
        ] = {}
        # SDK client state, set once the client connects.
        self._clients: dict[str, _ClientState] = {}

    def get(self, conv_id: str) -> _ClientState | None:
        """
        Return the state for ``conv_id``, or ``None`` if absent.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :returns: Existing ``_ClientState``, or ``None``.
        """
        return self._clients.get(conv_id)

    def get_or_create_loop(
        self,
        conv_id: str,
    ) -> asyncio.AbstractEventLoop:
        """
        Return the event loop for ``conv_id``, creating one if needed.

        If no ``_ClientState`` exists yet (first task), a new loop and
        background thread are spun up and a placeholder state is NOT
        created — the loop is returned directly, and ``_async_turn``
        will call ``register`` once the SDK client is connected.

        If a ``_ClientState`` already exists, its loop is returned.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :returns: The per-conversation asyncio event loop.
        """
        existing = self._loops.get(conv_id)
        if existing is not None:
            return existing[0]
        # First task for this conversation — create a dedicated loop.
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            daemon=True,
            name=f"sdk-loop-{conv_id}",
        )
        thread.start()
        self._loops[conv_id] = (loop, thread)
        return loop

    def register(self, conv_id: str, state: _ClientState) -> None:
        """
        Store a newly connected client state.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :param state: The ``_ClientState`` to register.
        """
        with self._lock:
            self._clients[conv_id] = state

    def remove_client(self, conv_id: str) -> _ClientState | None:
        """
        Remove client state only, keeping the event loop alive.

        Used on error from within the conversation's event loop —
        stopping the loop from inside it would deadlock. The loop
        stays alive for the current turn to finish; it will be
        reused if a new client connects, or cleaned up by eviction.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :returns: The removed ``_ClientState``, or ``None``.
        """
        with self._lock:
            return self._clients.pop(conv_id, None)

    def remove(self, conv_id: str) -> _ClientState | None:
        """
        Remove client state AND stop the conversation's event loop.

        Must NOT be called from within the conversation's loop.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :returns: The removed ``_ClientState``, or ``None``.
        """
        with self._lock:
            state = self._clients.pop(conv_id, None)
            loop_entry = self._loops.pop(conv_id, None)
        if loop_entry is not None:
            loop, thread = loop_entry
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)
        return state

    def touch(self, conv_id: str) -> None:
        """
        Update the ``last_used`` timestamp for ``conv_id``.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        """
        state = self._clients.get(conv_id)
        if state is not None:
            state.last_used = time.monotonic()

    def __contains__(self, conv_id: str) -> bool:
        """
        Return True if a client is registered for ``conv_id``.

        :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
        :returns: True if present.
        """
        return conv_id in self._clients

    def evict_stale(self) -> None:
        """
        Disconnect and remove clients idle longer than ``ttl_seconds``.

        Stops each evicted conversation's event loop and joins its
        background thread.
        """
        now = time.monotonic()
        with self._lock:
            stale = [
                conv_id
                for conv_id, state in self._clients.items()
                if now - state.last_used > self._ttl
            ]
            evicted: list[tuple[_ClientState, asyncio.AbstractEventLoop, threading.Thread]] = []
            for conv_id in stale:
                state = self._clients.pop(conv_id)
                loop_entry = self._loops.pop(conv_id, None)
                if loop_entry is not None:
                    evicted.append((state, loop_entry[0], loop_entry[1]))
        for state, loop, thread in evicted:
            _logger.debug(
                "evicting stale SDK client (idle %.0fs)",
                now - state.last_used,
            )
            _disconnect_in_loop(state.client, loop)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)


# Process-lifetime singleton — persists across executor instances.
_client_registry = _ClientRegistry()


@dataclass
class _PendingToolCall:
    """
    Metadata for an in-flight tool call being tracked through the SDK stream.

    :param name: Tool name, e.g. ``"Bash"``.
    :param start_time: Monotonic timestamp when the tool call started.
    """

    name: str
    start_time: float


@dataclass
class _StreamState:
    """
    Mutable state accumulated while consuming the SDK event stream.

    :param response_text: Accumulated response text from text deltas.
        ``None`` until the first text delta arrives.
    :param got_stream_events: Whether any ``StreamEvent`` messages
        have been received (controls dedup with ``AssistantMessage``).
    :param pending: In-flight tool calls keyed by ``tool_use_id``.
    :param args_buffers: Accumulated ``input_json_delta`` fragments
        per ``tool_use_id``.
    """

    response_text: str | None = None
    got_stream_events: bool = False
    pending: dict[str, _PendingToolCall] = dataclass_field(
        default_factory=dict,
    )
    args_buffers: dict[str, list[str]] = dataclass_field(
        default_factory=dict,
    )


def _extract_latest_user_message(
    messages: list[dict[str, Any]],
) -> str | None:
    """
    Extract the latest user message text from a message list.

    :param messages: Responses API input items.
    :returns: The text of the last user message, or ``None`` if none.
    """
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "input_text"
                ]
                return "\n".join(parts)
    return None


def _build_history_prompt(
    messages: list[dict[str, Any]],
) -> str:
    """
    Serialize full message history into a single text prompt.

    Used for crash recovery when no SDK session state is available.
    The SDK sees the prior conversation as context in the first
    prompt.

    Skips ``function_call`` and ``function_call_output`` items —
    these represent tool calls the executor ran internally on a
    prior turn. Including them as empty ``"user: "`` lines would
    confuse the model.

    :param messages: Responses API input items (full history).
    :returns: A multi-line text prompt summarizing the conversation.
    """
    lines = ["Conversation so far:"]
    for msg in messages:
        # Skip executor-observed tool call items — they don't
        # have role/content and would produce empty "user: " lines.
        msg_type = msg.get("type")
        if msg_type in ("function_call", "function_call_output"):
            continue
        # After filtering function_call/function_call_output, all
        # remaining items are user/assistant messages with role+content.
        role = str(msg["role"])
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{role}: {content}")
    lines.append("")
    lines.append("Respond to the latest user message, using the conversation above as context.")
    return "\n".join(lines)


def _has_session_transcript(storage_dir: Path) -> bool:
    """
    Check whether the storage dir contains a Claude session transcript.

    If ``.claude/`` exists inside the storage dir, the SDK can use
    ``--continue`` to resume from the restored session state.

    :param storage_dir: The scoped persistent directory.
    :returns: True if session transcript files exist.
    """
    claude_dir = storage_dir / ".claude"
    return claude_dir.is_dir() and any(claude_dir.iterdir())


class ClaudeAgentsExecutor(Executor):
    """
    Executor that wraps the Claude Agent SDK.

    The SDK runs Claude Code's full agent loop internally. Server-side
    tools execute via the SDK's built-in handlers. Client-side tools
    are routed through an in-process MCP server backed by
    ``context.await_tool_output``.

    Maintains a persistent ``ClaudeSDKClient`` per conversation across
    ``run_turn()`` calls — the SDK subprocess keeps context in memory
    between turns.

    :param allowed_tools: Server-side built-in tool names, e.g.
        ``["Bash", "Read", "Edit"]``.
    :param model: Optional model override, e.g.
        ``"claude-sonnet-4-20250514"``.
    """

    def __init__(
        self,
        *,
        allowed_tools: list[str],
        model: str | None = None,
    ) -> None:
        self._allowed_tools = allowed_tools
        self._model = model

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from the agent spec's tool list and model.

        Extracts ``claude:``-prefixed tools from the spec's tool
        config and strips the prefix to get allowed tool names.

        :param spec: Agent spec with ``executor.type == "claude_sdk"``.
        :returns: Configured ClaudeAgentsExecutor.
        """
        allowed = _extract_claude_tools(spec)
        model = spec.llm.model if spec.llm else None
        return cls(allowed_tools=allowed, model=model)

    def max_context_tokens(self) -> int | None:
        """
        SDK manages its own context window.

        :returns: None — workflow skips compaction and @step.
        """
        return None

    def on_task_end(self, context: ExecutorContext) -> None:
        """
        Mark the conversation's SDK client as idle and evict stale clients.

        Does NOT disconnect the live client — the subprocess keeps running
        so bash session state (cwd, env vars, running processes) persists
        into the next task. Stale clients (idle > the registry TTL)
        are disconnected here to bound resource usage.

        :param context: Agent-plane capabilities and identifiers.
        """
        _client_registry.touch(context.conversation_id)
        _client_registry.evict_stale()

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        Run one SDK turn. Bridges async SDK stream into sync iterator.

        Server-side tools: SDK executes via built-in handlers.
        Client-side tools: routed through an in-process MCP server
        backed by ``context.await_tool_output``.

        :param messages: Used to build prompt for fresh/recovery
            sessions. Ignored for continuing sessions (SDK has context
            in memory).
        :param tools: Client-side tool schemas — used to build the
            MCP server so the SDK can call these tools.
        :param system_prompt: System instructions for the SDK.
        :param llm_config: LLM configuration (model override).
        :param context: Agent-plane capabilities and identifiers.
        """
        event_queue: queue.Queue[ExecutorEvent | None] = queue.Queue(
            maxsize=256,
        )
        # Each conversation has its own event loop so conversations
        # are isolated — a slow SDK call in one can't starve others.
        loop = _client_registry.get_or_create_loop(
            context.conversation_id,
        )
        asyncio.run_coroutine_threadsafe(
            _async_turn(
                executor=self,
                messages=messages,
                tools=tools,
                system_prompt=system_prompt,
                llm_config=llm_config,
                context=context,
                event_queue=event_queue,
            ),
            loop,
        )
        while True:
            event = event_queue.get()
            if event is None:
                break
            yield event


# ── Async implementation (runs in the bridge thread) ──────


async def _async_turn(
    executor: ClaudeAgentsExecutor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Async implementation of one SDK turn.

    Runs in the registry's shared event loop. Builds the prompt,
    configures the SDK client (on first task), sends the query, and
    maps the SDK stream events into executor events pushed to the
    queue. Puts ``None`` on ``event_queue`` when done.

    :param executor: The executor instance.
    :param messages: Responses API input items.
    :param tools: Client-side tool schemas.
    :param system_prompt: System instructions.
    :param llm_config: LLM configuration.
    :param context: Agent-plane capabilities and identifiers.
    :param event_queue: Queue for bridging events to the sync caller.
    """
    sdk = _ensure_sdk()
    conv_id = context.conversation_id
    existing = _client_registry.get(conv_id)
    is_continuing = existing is not None

    prompt = _build_prompt(messages, context.storage_dir, is_continuing)
    if prompt is None:
        event_queue.put(TurnComplete(text=None))
        event_queue.put(None)
        return

    if existing is not None:
        client = existing.client
        holder = existing.callback_holder
    else:
        holder = _CallbackHolder()
        mcp_server = _build_client_tool_mcp_server(sdk, tools, holder)
        options = _build_sdk_options(
            sdk,
            executor,
            system_prompt,
            llm_config,
            context,
            mcp_server,
            tools,
        )
        client = await _get_or_create_client(
            sdk,
            conv_id,
            options,
            llm_config,
            holder,
        )

    # Point the MCP server's callback at this task before querying.
    # Safe: runs in a single event loop thread. The MCP handler reads
    # holder.callback only during client.query(), so the sequence
    # set → query → clear is strictly sequential.
    holder.callback = context.await_tool_output
    try:
        await client.query(prompt, session_id=conv_id)
        await _consume_sdk_stream(sdk, client, event_queue)
    except Exception as exc:
        await _close_client_async(conv_id)
        event_queue.put(ExecutorError(message=f"Claude SDK error: {exc}"))
    finally:
        holder.callback = None
        event_queue.put(None)


def _build_prompt(
    messages: list[dict[str, Any]],
    storage_dir: Path,
    is_continuing: bool,
) -> str | None:
    """
    Build the prompt string for the SDK.

    For continuing sessions (client already in memory), send only the
    latest user message. For restored sessions with a transcript, send
    only the latest message (the SDK replays its own transcript). For
    fresh or crash-recovery sessions, build a full history prompt.

    :param messages: Responses API input items.
    :param storage_dir: The scoped persistent directory.
    :param is_continuing: Whether an SDK client already exists for
        this conversation.
    :returns: The prompt string, or ``None`` if no user message found.
    """
    if is_continuing:
        return _extract_latest_user_message(messages)
    if _has_session_transcript(storage_dir):
        return _extract_latest_user_message(messages)
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) <= 1:
        return _extract_latest_user_message(messages)
    return _build_history_prompt(messages)


def _build_client_tool_mcp_server(
    sdk: Any,
    tools: list[dict[str, Any]],
    callback_holder: _CallbackHolder,
) -> Any | None:
    """
    Build an in-process MCP server for client-side tools.

    Each client-side tool becomes an MCP tool whose handler reads the
    current task's callback from ``callback_holder`` at call time.
    Because the holder is updated before each ``client.query()`` and
    cleared after, tool calls always route to the active task's parking
    infrastructure even when the SDK client is reused across tasks.

    :param sdk: The ``claude_agent_sdk`` module.
    :param tools: Client-side tool schemas (OpenAI format).
    :param callback_holder: Mutable holder for the current task's
        ``await_tool_output`` callback.
    :returns: An MCP server object, or ``None`` if no client tools.
    """
    if not tools:
        return None

    mcp_tools = []
    for schema in tools:
        tool_name = schema.get("name", "")
        tool_desc = schema.get("description", "")
        # OpenAI tool schemas always include "parameters"; the empty
        # object fallback matches the SDK's expectation for no-arg tools.
        tool_params = schema.get(
            "parameters",
            {
                "type": "object",
                "properties": {},
            },
        )

        handler = _make_client_tool_handler(tool_name, callback_holder)
        decorated = sdk.tool(tool_name, tool_desc, tool_params)(handler)
        mcp_tools.append(decorated)

    return sdk.create_sdk_mcp_server(
        name="agent_plane",
        version="1.0.0",
        tools=mcp_tools,
    )


def _make_client_tool_handler(
    tool_name: str,
    callback_holder: _CallbackHolder,
) -> Any:
    """
    Create an async MCP tool handler for a client-side tool.

    The handler reads ``callback_holder.callback`` at call time, not at
    creation time. This allows the same MCP server (and SDK client) to be
    reused across tasks: the caller updates ``callback_holder.callback``
    before each ``client.query()`` and clears it after.

    :param tool_name: The tool name, e.g. ``"Read"``.
    :param callback_holder: Mutable holder for the current task's
        ``await_tool_output`` callback.
    :returns: An async callable for the SDK's ``@tool`` decorator.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        """
        MCP handler that parks a client-side tool call.

        :param args: Tool arguments from the SDK.
        :returns: MCP-format result dict.
        """
        cb = callback_holder.callback
        if cb is None:
            raise RuntimeError(
                f"No active task callback for tool '{tool_name}'."
                " Tool called outside of an active query."
            )
        call = ToolCallRequested(
            call_id=f"sdk_{tool_name}_{int(time.monotonic() * 1000)}",
            name=tool_name,
            arguments=args,
        )
        loop = asyncio.get_running_loop()
        # await_tool_output is sync (blocking) — run in thread
        result: ToolResult = await loop.run_in_executor(None, cb, call)
        response: dict[str, Any] = {
            "content": [{"type": "text", "text": result.content}],
        }
        if result.status == "error":
            response["isError"] = True
        return response

    return handler


def _build_sdk_options(
    sdk: Any,
    executor: ClaudeAgentsExecutor,
    system_prompt: str,
    llm_config: LLMConfig,
    context: ExecutorContext,
    mcp_server: Any | None,
    tools: list[dict[str, Any]],
) -> Any:
    """
    Build ``ClaudeAgentOptions`` for the SDK client.

    :param sdk: The ``claude_agent_sdk`` module.
    :param executor: The executor instance.
    :param system_prompt: System instructions.
    :param llm_config: LLM configuration (model override).
    :param context: Agent-plane capabilities.
    :param mcp_server: MCP server for client-side tools, or ``None``.
    :param tools: Client-side tool schemas (for ``allowed_tools``).
    :returns: Configured ``ClaudeAgentOptions``.
    """
    mcp_servers: dict[str, Any] = {}
    if mcp_server is not None:
        mcp_servers["agent_plane"] = mcp_server

    allowed_tools = list(executor._allowed_tools)
    for schema in tools:
        tool_name = schema.get("name", "")
        allowed_tools.append(f"mcp__agent_plane__{tool_name}")

    # Unset CLAUDECODE to prevent "nested session" error from the SDK.
    env = {"CLAUDECODE": ""}

    model = executor._model or llm_config.model

    options = sdk.ClaudeAgentOptions(
        tools=list(executor._allowed_tools),
        system_prompt=system_prompt or None,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        env=env,
        include_partial_messages=True,
        # Disable SDK session persistence — agent-plane manages
        # session state via storage_dir and the artifact store.
        extra_args={"no-session-persistence": None},
        cwd=str(context.storage_dir),
    )

    cli_path = shutil.which("claude")
    if cli_path:
        options.cli_path = cli_path

    if model:
        options.model = model

    return options


async def _get_or_create_client(
    sdk: Any,
    conv_id: str,
    options: Any,
    llm_config: LLMConfig,
    holder: _CallbackHolder,
) -> Any:
    """
    Create and register a new SDK client for a conversation.

    Only called on first task for a given ``conv_id`` — subsequent tasks
    find the existing entry in ``_client_registry`` and skip this function.

    :param sdk: The ``claude_agent_sdk`` module.
    :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
    :param options: ``ClaudeAgentOptions`` for client creation.
    :param llm_config: LLM configuration (for model tracking).
    :param holder: The callback holder to store with the client state.
    :returns: The connected ``ClaudeSDKClient`` instance.
    """
    client = sdk.ClaudeSDKClient(options)
    await client.connect()
    _client_registry.register(
        conv_id,
        _ClientState(
            client=client,
            model=llm_config.model,
            callback_holder=holder,
            last_used=time.monotonic(),
        ),
    )
    return client


async def _consume_sdk_stream(
    sdk: Any,
    client: Any,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Consume the SDK message stream and push executor events.

    Maps SDK event types to agent-plane executor events:

    - ``StreamEvent`` / ``text_delta`` → ``TextChunk``
    - ``StreamEvent`` / ``content_block_start`` (tool_use) → buffer
    - ``UserMessage`` / ``ToolResultBlock`` → ``ToolCallObserved``
    - ``ResultMessage`` → capture final text
    - ``SystemMessage`` / ``api_retry`` with 401/403/404 → ``ExecutorError``

    :param sdk: The ``claude_agent_sdk`` module.
    :param client: The connected ``ClaudeSDKClient``.
    :param event_queue: Queue for pushing executor events.
    """
    from claude_agent_sdk.types import (
        StreamEvent as _StreamEvent,
    )

    state = _StreamState()
    message_stream = client.receive_response()
    try:
        async for message in message_stream:
            if isinstance(message, _StreamEvent):
                state.got_stream_events = True
                _handle_stream_event(message.event, state, event_queue)

            elif isinstance(message, sdk.AssistantMessage):
                if not state.got_stream_events:
                    _handle_assistant_message(
                        sdk,
                        message,
                        state,
                        event_queue,
                    )

            elif isinstance(message, sdk.UserMessage):
                _handle_tool_results(sdk, message, state, event_queue)

            elif isinstance(message, sdk.ResultMessage):
                if state.response_text is None and message.result:
                    state.response_text = message.result

            elif isinstance(message, sdk.SystemMessage):
                error = _check_terminal_error(message)
                if error is not None:
                    event_queue.put(error)
                    return
    finally:
        aclose = getattr(message_stream, "aclose", None)
        if aclose is not None:
            await aclose()

    event_queue.put(TurnComplete(text=state.response_text or None))


def _handle_stream_event(
    evt: dict[str, Any],
    state: _StreamState,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Handle a single SDK ``StreamEvent``.

    :param evt: The raw event dict from the SDK stream.
    :param state: Mutable stream state for tracking tools and text.
    :param event_queue: Queue for pushing executor events.
    """
    evt_type = evt.get("type", "")

    if evt_type == "content_block_start":
        block = evt.get("content_block", {})
        if block.get("type") == "tool_use":
            tool_id = block.get("id", "")
            tool_name = block.get("name", "unknown")
            state.pending[tool_id] = _PendingToolCall(
                name=tool_name,
                start_time=time.monotonic(),
            )
            state.args_buffers[tool_id] = []

    elif evt_type == "content_block_delta":
        _handle_content_delta(evt, state, event_queue)


def _handle_content_delta(
    evt: dict[str, Any],
    state: _StreamState,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Handle a ``content_block_delta`` event (text or tool args).

    :param evt: The raw event dict from the SDK stream.
    :param state: Mutable stream state.
    :param event_queue: Queue for pushing executor events.
    """
    delta = evt.get("delta", {})
    delta_type = delta.get("type", "")
    if delta_type == "text_delta":
        text = delta.get("text", "")
        if text:
            if state.response_text is None:
                state.response_text = text
            else:
                state.response_text += text
            event_queue.put(TextChunk(text=text))
    elif delta_type == "input_json_delta":
        partial = delta.get("partial_json", "")
        if partial:
            _buffer_args(state, partial)


def _buffer_args(state: _StreamState, partial: str) -> None:
    """
    Buffer ``input_json_delta`` for the most recently started tool.

    The SDK doesn't include ``tool_use_id`` on delta events, so we
    match by finding the last pending tool that has a buffer.

    :param state: Mutable stream state.
    :param partial: The JSON fragment to buffer.
    """
    for tool_id in reversed(list(state.pending.keys())):
        if tool_id in state.args_buffers:
            state.args_buffers[tool_id].append(partial)
            return


def _handle_assistant_message(
    sdk: Any,
    message: Any,
    state: _StreamState,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Handle an ``AssistantMessage`` when no ``StreamEvent`` was received.

    Falls back to extracting text and tool calls from the full message.

    :param sdk: The ``claude_agent_sdk`` module.
    :param message: The ``AssistantMessage`` from the SDK.
    :param state: Mutable stream state.
    :param event_queue: Queue for pushing executor events.
    """
    for block in message.content:
        if isinstance(block, sdk.TextBlock):
            event_queue.put(TextChunk(text=block.text))
        elif isinstance(block, sdk.ToolUseBlock):
            state.pending[block.id] = _PendingToolCall(
                name=block.name,
                start_time=time.monotonic(),
            )
            state.args_buffers[block.id] = [json.dumps(block.input)]


def _handle_tool_results(
    sdk: Any,
    message: Any,
    state: _StreamState,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Handle a ``UserMessage`` containing ``ToolResultBlock`` entries.

    Matches each result to its pending request and emits
    ``ToolCallObserved``.

    :param sdk: The ``claude_agent_sdk`` module.
    :param message: The ``UserMessage`` from the SDK.
    :param state: Mutable stream state.
    :param event_queue: Queue for pushing executor events.
    """
    content = message.content
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, sdk.ToolResultBlock):
            _emit_tool_call_observed(block, state, event_queue)


def _emit_tool_call_observed(
    block: Any,
    state: _StreamState,
    event_queue: queue.Queue[ExecutorEvent | None],
) -> None:
    """
    Build and emit a ``ToolCallObserved`` event from a ``ToolResultBlock``.

    :param block: The ``ToolResultBlock`` from the SDK.
    :param state: Mutable stream state (pending tools and args buffers).
    :param event_queue: Queue for pushing executor events.
    """
    tool_id = block.tool_use_id
    pending_call = state.pending.pop(tool_id, None)
    if pending_call is not None:
        name = pending_call.name
        start = pending_call.start_time
    else:
        _logger.warning(
            "ToolResultBlock for unknown tool_use_id %s — tool call tracking may be out of sync",
            tool_id,
        )
        name = "unknown"
        start = time.monotonic()
    duration_ms = (time.monotonic() - start) * 1000

    arguments = _reconstruct_tool_args(
        state.args_buffers.pop(tool_id, []),
    )
    result_text = _extract_tool_result_text(block.content)

    event_queue.put(
        ToolCallObserved(
            call_id=tool_id,
            name=name,
            arguments=arguments,
            result=result_text,
            status="error" if block.is_error else "success",
            duration_ms=duration_ms,
        )
    )


def _reconstruct_tool_args(fragments: list[str]) -> dict[str, Any]:
    """
    Reconstruct tool arguments from buffered JSON fragments.

    :param fragments: Accumulated ``input_json_delta`` strings.
    :returns: Parsed arguments dict, or empty dict on failure.
    """
    args_str = "".join(fragments)
    if not args_str:
        return {}
    try:
        parsed: dict[str, Any] = json.loads(args_str)
        return parsed
    except json.JSONDecodeError:
        return {}


def _extract_tool_result_text(result_content: Any) -> str:
    """
    Extract text from a ``ToolResultBlock.content``.

    The content can be a plain string, a list of content blocks
    (each with a ``.text`` attribute), or another type.

    :param result_content: The raw content from the tool result block.
    :returns: Extracted text.
    """
    if isinstance(result_content, str):
        return result_content
    if isinstance(result_content, list):
        parts = [part.text for part in result_content if hasattr(part, "text")]
        return "\n".join(parts) if parts else str(result_content)
    return str(result_content)


def _check_terminal_error(message: Any) -> ExecutorError | None:
    """
    Check a ``SystemMessage`` for terminal errors (auth, not found).

    :param message: The ``SystemMessage`` from the SDK.
    :returns: An ``ExecutorError`` if the error is terminal, else ``None``.
    """
    if message.subtype != "api_retry":
        return None
    data = message.data
    status = data.get("error_status")
    error = data.get("error", "unknown_error")

    if status in {401, 403}:
        return ExecutorError(
            message=(
                f"Claude SDK auth failed ({error}, status={status}). Check ANTHROPIC_API_KEY."
            ),
            code="auth_failed",
        )
    if status == 404:
        return ExecutorError(
            message=(
                f"Claude SDK endpoint not found ({error},"
                f" status={status}). "
                "Check model name and API configuration."
            ),
            code="not_found",
        )
    return None


async def _close_client_async(conv_id: str) -> None:
    """
    Disconnect and remove the SDK client for a conversation.

    Called on error so a broken subprocess doesn't persist into the
    next task.

    :param conv_id: Conversation identifier, e.g. ``"conv_abc123"``.
    """
    # remove_client (not remove) — we're running inside this
    # conversation's event loop; stopping it here would deadlock.
    state = _client_registry.remove_client(conv_id)
    if state is not None:
        try:
            await state.client.disconnect()
        except Exception:
            _logger.warning(
                "Failed to disconnect Claude SDK client for %s",
                conv_id,
                exc_info=True,
            )


def _disconnect_in_loop(
    client: Any,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Disconnect an SDK client by submitting to the shared event loop.

    Used during TTL eviction (called from a workflow thread, not the
    event loop thread). Does not stop the loop — it's shared across
    all conversations.

    :param client: The ``ClaudeSDKClient`` to disconnect.
    :param loop: The shared event loop to submit to.
    """
    try:
        future = asyncio.run_coroutine_threadsafe(
            client.disconnect(),
            loop,
        )
        future.result(timeout=5.0)
    except Exception:
        _logger.warning(
            "Failed to disconnect Claude SDK client",
            exc_info=True,
        )


def _extract_claude_tools(spec: AgentSpec) -> list[str]:
    """
    Extract ``claude:``-prefixed tool names from the agent spec.

    Strips the ``claude:`` prefix and returns bare tool names
    suitable for ``allowed_tools`` on ``ClaudeAgentOptions``.

    :param spec: The agent spec.
    :returns: List of tool names, e.g. ``["Bash", "Read", "Edit"]``.
    """
    result: list[str] = []
    for builtin in spec.tools.builtins:
        name = builtin.name
        if name.startswith("claude:"):
            result.append(name[len("claude:") :])
    return result
