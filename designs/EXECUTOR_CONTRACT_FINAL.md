# Executor Contract (Final)

## Context

Agent-plane's `_run_agent_loop` today has the LLM call, tool dispatch, compaction,
and steering all coupled together in one function. This makes it impossible to swap
in a different agent runtime (Claude SDK, OpenAI Agents SDK, remote agent services)
without forking the loop.

The goal is an `Executor` ABC that decouples the "call LLM + run tools" concern from
the "persist, stream, compact, steer" concerns. The outer loop stays exactly as it is;
the executor is the only thing that varies.

Three executor types:

| Executor | What it wraps | Tools | Compaction |
|----------|--------------|-------|------------|
| `DefaultExecutor` | litellm (existing path) | Workflow executes via @step | Workflow owns |
| `ClaudeSDKExecutor` | Claude Agent SDK subprocess | SDK executes built-in; client-side via `await_tool_output` | SDK owns |
| `RemoteExecutor` | Any HTTP agent service | Remote service executes; can request workflow execution | Remote service owns |

---

## Event Types

```python
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass
class TextChunk:
    """
    A streamed text token from the model.

    :param text: The incremental text fragment, e.g. ``"Hello"``.
    """
    text: str


@dataclass
class ToolCallRequested:
    """
    The executor wants the workflow to execute a tool.

    The workflow executes the tool via ``_call_tool()`` (@step), appends
    a tool_result message, and calls ``run_turn()`` again. Used by
    external executors (DefaultExecutor) and remote executors that
    need agent-plane to execute or park a tool.

    :param call_id: Identifier for this call, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"web_search"``.
    :param arguments: Parsed tool arguments dict.
    """
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallObserved:
    """
    The executor ran a tool internally. Workflow just persists and streams.

    Emitted by internal executors (Claude SDK) after each tool the
    harness executed autonomously. Both the call and the result are
    bundled — the workflow never executes the tool itself.

    :param call_id: Identifier, e.g. ``"call_abc123"``.
    :param name: Tool name, e.g. ``"Bash"``.
    :param arguments: Parsed tool arguments dict.
    :param result: The tool's output string.
    :param status: ``"success"`` | ``"error"`` | ``"blocked"``.
    :param duration_ms: Wall-clock time the tool took, e.g. ``342.1``.
    """
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    status: str
    duration_ms: float


@dataclass
class TurnComplete:
    """
    The executor has finished its turn.

    :param text: The assistant's text response, or ``None`` if the turn
        ended with tool calls only (external executor).
    """
    text: str | None


@dataclass
class ContextWindowExceeded:
    """
    The executor hit a context window overflow.

    The workflow compacts messages and retries ``run_turn()``.

    :param max_tokens: The model's context window size, e.g. ``128000``.
    :param actual_tokens: The prompt size that triggered overflow,
        e.g. ``131072``.
    """
    max_tokens: int
    actual_tokens: int


@dataclass
class ExecutorError:
    """
    An unrecoverable executor failure. No retry.

    :param message: Human-readable error description.
    :param code: Machine-readable error code, e.g. ``"auth_failed"``.
    """
    message: str
    code: str | None = None


@dataclass
class ReasoningChunk:
    """
    A streamed reasoning token from the model.

    Gated by ``reasoning_effort`` in the LLM config. Emitted for
    full reasoning tokens, reasoning summaries, and the reasoning
    started signal.

    :param delta: The incremental reasoning text, e.g. ``"Let me think"``.
        Empty string for ``"started"`` events.
    :param event_type: One of ``"reasoning_text"``,
        ``"reasoning_summary"``, or ``"reasoning_started"``.
    """
    delta: str
    event_type: str


@dataclass
class NativeToolOutput:
    """
    A provider-native tool output (e.g. ``web_search_call`` result).

    Not dispatched locally — flows through to the client as-is.
    The workflow persists and streams the raw item dict.

    :param item: The raw output dict from the provider, e.g.
        ``{"type": "web_search_call", "id": "ws_1", ...}``.
    """
    item: dict[str, Any]


@dataclass
class ToolResult:
    """
    Result of a tool call executed via ``await_tool_output``.

    :param content: The tool's output string, e.g. ``"file contents..."``.
    :param status: ``"success"`` or ``"error"``.
    """
    content: str
    status: str


ExecutorEvent: TypeAlias = (
    TextChunk
    | ReasoningChunk
    | NativeToolOutput
    | ToolCallRequested
    | ToolCallObserved
    | TurnComplete
    | ContextWindowExceeded
    | ExecutorError
)
```

---

## ExecutorContext

```python
from pathlib import Path
from collections.abc import Callable


@dataclass
class ExecutorContext:
    """
    Capabilities and identifiers agent-plane provides to executors.

    Constructed by the workflow once per turn and passed to ``run_turn()``
    and lifecycle hooks. Extensible — new capabilities are added as
    fields, no signature changes needed.

    :param task_id: Current task identifier, e.g. ``"task_abc123"``.
    :param conversation_id: Current conversation identifier,
        e.g. ``"conv_abc123"``.
    :param storage_dir: Scoped persistent directory for this
        conversation. Contents survive across tasks — the workflow
        manages artifact store I/O. The executor reads/writes
        freely within this directory.
    :param await_tool_output: Submit a tool call for client-side
        execution. Blocks until the client returns a result. Used by
        executors that run tools internally but need to delegate
        client-side tools (e.g. human-in-the-loop, IDE actions)
        through agent-plane's parking infrastructure.
    """
    task_id: str
    conversation_id: str
    storage_dir: Path
    await_tool_output: Callable[[ToolCallRequested], ToolResult]
```

---

## Executor ABC

```python
import abc
from collections.abc import Iterator
from typing import Self


class Executor(abc.ABC):
    """
    Abstract base for agent executors.

    Subclasses wrap a specific LLM backend or agent harness. The workflow
    calls ``run_turn()`` and consumes the event stream uniformly —
    no branching on executor type.

    Construction is standardized via ``from_spec()``. Each subclass
    extracts what it needs from the AgentSpec.
    """

    @classmethod
    @abc.abstractmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Construct an executor from an agent spec.

        Each subclass extracts the fields it needs. The workflow calls
        this once at startup — the returned executor is reused across
        turns.

        :param spec: The parsed AgentSpec with a non-None llm field.
        :returns: A configured executor instance.
        """
        ...

    @abc.abstractmethod
    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        Run one executor turn and yield events.

        For external executors (DefaultExecutor): one LLM call. Yields
        TextChunks and ToolCallRequested events. The workflow executes
        tools and calls run_turn() again with results.

        For internal executors (ClaudeSDKExecutor): the full agent loop.
        Yields TextChunks, ToolCallObserved events, and a terminal
        TurnComplete or ExecutorError.

        For remote executors (RemoteExecutor): proxies to an HTTP
        endpoint. Can yield either ToolCallRequested (workflow executes)
        or ToolCallObserved (already executed), depending on the remote
        service.

        :param messages: Conversation history as Responses API input items.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: Assembled system instructions string.
        :param config: Model and sampling configuration.
        :param context: Capabilities and identifiers from agent-plane.
        """
        ...

    def on_task_start(self, context: ExecutorContext) -> None:
        """
        Called once at task start, after storage_dir has been restored.

        The workflow restores the scoped persistent directory from the
        artifact store before calling this hook. The executor can read
        prior session state from ``context.storage_dir`` immediately.

        :param context: Capabilities and identifiers from agent-plane.
        """

    def on_task_end(self, context: ExecutorContext) -> None:
        """
        Called once at task end (in a finally block). Writes to
        ``context.storage_dir`` will be persisted by the workflow
        after this returns.

        :param context: Same context from on_task_start.
        """

    def max_context_tokens(self) -> int | None:
        """
        Context window limit, or None if managed internally.

        When an int: the workflow owns compaction (proactive via tiktoken
        estimate, reactive via ContextWindowExceeded event).

        When None: the executor owns compaction internally. The workflow
        skips both proactive and reactive compaction.

        :returns: Token limit (e.g. ``128000``) or None.
        """
        return None
```

---

## ExecutorConfig

```python
@dataclass
class ExecutorConfig:
    """
    Model and sampling parameters for one run_turn() call.

    :param model: The litellm model identifier, e.g.
        ``"databricks/databricks-claude-sonnet-4"``.
    :param temperature: Sampling temperature, e.g. ``0.0``.
    :param max_tokens: Max output tokens, e.g. ``4096``.
    :param timeout: Per-call timeout in seconds, e.g. ``120``.
    :param retry: Retry policy for transient failures.
    :param reasoning: Optional reasoning config dict, e.g.
        ``{"effort": "high"}``.
    :param connection: Optional provider connection override.
    """
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    retry: RetryConfig
    reasoning: dict[str, str] | None = None
    connection: str | None = None
```

---

## Workflow Integration

### Checkpointed turn (@step wrapper)

The @step wrapper eagerly consumes the executor's event stream. During
live execution, streaming events (TextChunk, ReasoningChunk,
NativeToolOutput) are emitted to the live SSE connection via
``_write_output`` as they arrive — preserving real-time streaming.
All events are collected into a serializable list for DBOS caching.

On crash replay, the @step returns the cached event list without
calling the executor. ``_write_output`` is not called (the SSE
connection broke on crash — the client reconnects via GET and gets
persisted conversation items).

```python
# ── SSE event type mapping ─────────────────────────────────
_REASONING_SSE_TYPES: dict[str, str] = {
    "reasoning_text": "response.reasoning_text.delta",
    "reasoning_summary": "response.reasoning_summary_text.delta",
    "reasoning_started": "response.reasoning.started",
}


@step()
def _checkpointed_turn(
    task_id: str,
    executor: Executor,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system_prompt: str,
    config: ExecutorConfig,
    context: ExecutorContext,
) -> list[dict[str, Any]]:
    """
    Run one executor turn inside a DBOS checkpoint.

    Streaming events are emitted to live SSE as they arrive.
    All events are serialized and returned for DBOS caching.

    :param task_id: Task identifier for SSE routing.
    :param executor: The executor to run.
    :param messages: Conversation history as input items.
    :param tools: OpenAI-format tool schemas.
    :param system_prompt: Assembled system instructions.
    :param config: Model and sampling configuration.
    :param context: Agent-plane capabilities and identifiers.
    :returns: Serialized event list (DBOS-cached on replay).
    """
    events: list[dict[str, Any]] = []
    for event in executor.run_turn(
        messages, tools, system_prompt, config, context,
    ):
        if isinstance(event, TextChunk):
            _write_output(task_id, {
                "type": "response.output_text.delta",
                "delta": event.text,
            })
        elif isinstance(event, ReasoningChunk):
            sse_type = _REASONING_SSE_TYPES[event.event_type]
            payload: dict[str, Any] = {"type": sse_type}
            if event.delta:
                payload["delta"] = event.delta
            _write_output(task_id, payload)
        elif isinstance(event, NativeToolOutput):
            _write_output(task_id, {
                "type": "response.output_item.done",
                "item": event.item,
            })
        events.append(_event_to_dict(event))
    return events
```

### Loop body

The loop structure, all helper calls, and all durability logic are
unchanged. The only change: ``_call_llm_maybe_compact`` is replaced
by the executor turn, with compaction handled around it.

```python
def _run_agent_loop(
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    agent_name: str,
    agent_id: str,
    instructions: str | None,
    tool_mgr: ToolManager,
    executor: Executor,
    reasoning: dict[str, str] | None = None,
) -> _AgentLoopResult:
    ...
    storage_dir = _restore_executor_storage(conversation_id)
    context = ExecutorContext(
        task_id=task_id,
        conversation_id=conversation_id,
        storage_dir=storage_dir,
        await_tool_output=_build_await_tool_output(task_id, conversation_id),
    )
    executor.on_task_start(context)
    try:
        for iteration in range(max_iterations):
            # unchanged: timeout check, _sync_history
            ...

            # Build messages — unchanged
            sys_instructions = build_instructions(spec, instructions, tool_schemas)
            messages = history_to_input_items(resolved_history)
            messages = _proactive_compact_if_needed(
                messages, history, sys_tokens, compaction_state, task_id
            )

            # Run executor turn — replaces _call_llm_maybe_compact.
            # Workflow-managed executors (max_context_tokens != None)
            # go through @step for DBOS durability + live SSE.
            # Executor-managed (SDK, Remote) iterate directly.
            if executor.max_context_tokens() is not None:
                raw_events = _checkpointed_turn(
                    task_id, executor, messages, tool_schemas,
                    sys_instructions, config, context,
                )
                turn_events = [_dict_to_event(e) for e in raw_events]
            else:
                turn_events = executor.run_turn(
                    messages, tool_schemas, sys_instructions,
                    config, context,
                )

            tool_calls_this_turn: list[ToolCallRequested] = []
            turn_complete: TurnComplete | None = None

            for event in turn_events:
                if isinstance(event, TextChunk):
                    # Only reached for non-checkpointed executors.
                    # Checkpointed ones emitted via _write_output
                    # inside the @step.
                    _write_output(task_id, {
                        "type": "response.output_text.delta",
                        "delta": event.text,
                    })

                elif isinstance(event, ReasoningChunk):
                    # Non-checkpointed path only.
                    sse_type = _REASONING_SSE_TYPES[event.event_type]
                    payload: dict[str, Any] = {"type": sse_type}
                    if event.delta:
                        payload["delta"] = event.delta
                    _write_output(task_id, payload)

                elif isinstance(event, NativeToolOutput):
                    # Non-checkpointed path only.
                    _write_output(task_id, {
                        "type": "response.output_item.done",
                        "item": event.item,
                    })

                elif isinstance(event, ToolCallRequested):
                    _persist_and_stream(task_id, conv_store, conversation_id,
                        [_build_function_call_item(task_id, agent_name, event)],
                        output_items)
                    tool_calls_this_turn.append(event)

                elif isinstance(event, ToolCallObserved):
                    _persist_and_stream(task_id, conv_store, conversation_id,
                        [_build_function_call_item(task_id, agent_name, event),
                         _build_function_call_output_item(task_id, event)],
                        output_items)
                    _track_spawn_collect(output_items, spawned_ids, collected_ids)

                elif isinstance(event, ContextWindowExceeded):
                    messages = _reactive_compact(
                        messages, history, sys_tokens,
                        event.max_tokens, event.actual_tokens,
                        compaction_state, task_id)
                    break  # retry with compacted messages

                elif isinstance(event, TurnComplete):
                    turn_complete = event
                    break

                elif isinstance(event, ExecutorError):
                    _emit_llm_error_event(task_id, event)
                    return _AgentLoopResult(status="failed", ...)

            if turn_complete is None:
                continue  # compaction retry

            # From here: identical to today
            if not tool_calls_this_turn:
                ...  # auto-collect, steering handshake — unchanged
            else:
                ...  # tool execution, client-side split — unchanged

    finally:
        executor.on_task_end(context)
        _persist_executor_storage(conversation_id, storage_dir)
        if compaction_state.last_summary is not None:
            _maybe_persist_compaction_item(...)
```

---

## Implementation 1: DefaultExecutor

Wraps the existing `_invoke_llm_streaming()`. Thin shim — all logic that
currently lives in `_call_llm_maybe_compact()` is split: compaction stays
in the workflow; only the raw LLM call moves into the executor.

```python
class DefaultExecutor(Executor):
    """
    Executor backed by litellm. Does not handle tools internally.

    Yields ToolCallRequested for each tool call in the LLM response.
    The workflow executes tools via @step and re-invokes run_turn().

    :param config: Default model and sampling config.
    """

    def __init__(self, config: ExecutorConfig) -> None:
        self._config = config

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from agent spec's LLM config.

        :param spec: Agent spec with llm config.
        :returns: Configured DefaultExecutor.
        """
        assert spec.llm is not None
        config = ExecutorConfig(
            model=spec.llm.model,
            temperature=spec.llm.temperature,
            max_tokens=spec.llm.max_tokens,
            timeout=spec.llm.timeout,
            retry=spec.llm.retry,
            reasoning=None,
            connection=spec.llm.connection,
        )
        return cls(config=config)

    def max_context_tokens(self) -> int | None:
        """
        Return the model's known context window limit.

        The workflow uses this for proactive compaction.

        :returns: Token limit from the model registry.
        """
        return _get_model_context_window(self._config.model)

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        One LLM call via litellm.

        :param messages: Pre-compacted conversation history.
        :param tools: OpenAI-format tool schemas.
        :param system_prompt: System instructions.
        :param config: Model and sampling config.
        :param context: Agent-plane capabilities and identifiers.
        """
        try:
            llm_resp = _invoke_llm_streaming(
                context.task_id, messages, system_prompt,
                config.model, tools, config.connection,
                config.timeout, config.retry,
            )
        except ContextWindowExceededError as exc:
            yield ContextWindowExceeded(
                max_tokens=exc.max_context_tokens,
                actual_tokens=exc.actual_tokens,
            )
            return
        except (RetryableLLMError, PermanentLLMError) as exc:
            yield ExecutorError(message=str(exc), code=exc.code)
            return

        text = _get_text_content(llm_resp)
        if text:
            yield TextChunk(text=text)

        for tc in _get_tool_calls(llm_resp):
            yield ToolCallRequested(
                call_id=tc.call_id,
                name=tc.name,
                arguments=json.loads(tc.arguments),
            )

        yield TurnComplete(
            text=text if not _get_tool_calls(llm_resp) else None,
        )
```

---

## Implementation 2: ClaudeSDKExecutor

Wraps the Claude Agent SDK. The SDK runs its own internal agent loop
(LLM → tool → LLM → ...). Agent-plane observes and persists.

**Tool handling**: Server-side tools (Bash, Read, Edit, etc.) are passed
as `allowed_tools` — the SDK executes them with its built-in
implementations. Client-side tools use `context.await_tool_output` —
the executor builds an MCP server that wraps this callable, so the SDK
sees client-side tools as MCP tools. The executor has no parking logic;
`await_tool_output` handles parking transparently.

**Session state**: The SDK subprocess maintains context in memory across
`run_turn()` calls. Between tasks, `.claude/` session state is
persisted to `storage_dir` (workflow manages artifact store I/O).

**Compaction**: The SDK manages its own context window.
`max_context_tokens()` returns None — workflow skips compaction.

```python
class ClaudeSDKExecutor(Executor):
    """
    Executor that wraps the Claude Agent SDK.

    :param allowed_tools: Server-side built-in tool names,
        e.g. ``["Bash", "Read", "Edit", "Write", "Glob", "Grep"]``.
    :param model: Optional model override.
    """

    def __init__(
        self,
        *,
        allowed_tools: list[str],
        model: str | None = None,
    ) -> None:
        self._allowed_tools = allowed_tools
        self._model = model
        self._clients: dict[str, ClaudeSDKClient] = {}

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from agent spec's tool list and model.

        :param spec: Agent spec with tools and optional model override.
        :returns: Configured ClaudeSDKExecutor.
        """
        allowed = _extract_claude_tools(spec.tools)
        model = spec.llm.model if spec.llm else None
        return cls(allowed_tools=allowed, model=model)

    def on_task_start(self, context: ExecutorContext) -> None:
        """
        No-op. Storage dir is accessed via context in run_turn.

        :param context: Agent-plane capabilities and identifiers.
        """

    def on_task_end(self, context: ExecutorContext) -> None:
        """
        Disconnect SDK client. Session state in storage_dir is
        persisted by the workflow.

        :param context: Agent-plane capabilities and identifiers.
        """
        client = self._clients.pop(context.conversation_id, None)
        if client is not None:
            asyncio.run(client.disconnect())

    def max_context_tokens(self) -> int | None:
        """
        SDK manages its own context window.

        :returns: None — workflow skips compaction.
        """
        return None

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        Run one SDK turn. Bridges async SDK stream into sync iterator.

        Server-side tools: SDK executes via built-in handlers.
        Client-side tools: routed through an in-process MCP server
        backed by ``context.await_tool_output``.

        :param messages: Used to build prompt for fresh/recovery sessions.
            Ignored for continuing sessions (SDK has context in memory).
        :param tools: Client-side tool schemas — used to build the MCP
            server config so the SDK knows these tools exist.
        :param system_prompt: System instructions for the SDK.
        :param config: Model and sampling config.
        :param context: Agent-plane capabilities and identifiers.
        """
        q: queue.Queue[ExecutorEvent | None] = queue.Queue(maxsize=256)

        def _run():
            asyncio.run(self._async_turn(
                messages, tools, system_prompt, config, context, q,
            ))
            q.put(None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        while True:
            event = q.get()
            if event is None:
                break
            yield event
        thread.join()

    async def _async_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        context: ExecutorContext,
        q: queue.Queue[ExecutorEvent | None],
    ) -> None:
        """
        Async implementation. Sends prompt to SDK, maps stream events
        to ExecutorEvents, pushes into queue.

        :param messages: Conversation history.
        :param tools: Client-side tool schemas.
        :param system_prompt: System instructions.
        :param config: Model and sampling config.
        :param context: Agent-plane capabilities and identifiers.
        :param q: Queue for bridging events to sync caller.
        """
        sdk = _ensure_sdk()
        conv_id = context.conversation_id
        storage_dir = context.storage_dir
        is_continuing = conv_id in self._clients

        # --- Build prompt ---
        if is_continuing:
            prompt = _extract_latest_user_message(messages)
        elif storage_dir and _has_transcript(storage_dir):
            prompt = _extract_latest_user_message(messages)
        else:
            prompt = _build_prompt_from_history(messages)

        if not prompt:
            q.put(TurnComplete(text=None))
            return

        # --- Build MCP server backed by await_tool_output ---
        # Each client-side tool becomes an MCP tool whose handler
        # calls context.await_tool_output(), which parks the call
        # and blocks until the client returns a result.
        mcp_server = _build_client_tool_mcp_server(
            tools, context.await_tool_output,
        )

        # --- Configure SDK client ---
        client_tool_names = [t["name"] for t in tools]
        allowed_tools = list(self._allowed_tools)
        for name in client_tool_names:
            allowed_tools.append(f"mcp__agent_plane__{name}")

        options = sdk.ClaudeAgentOptions(
            tools=list(self._allowed_tools),
            system_prompt=system_prompt,
            mcp_servers={
                "agent_plane": mcp_server,
            },
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            env={"CLAUDECODE": ""},
            cwd=str(storage_dir),
        )
        if self._model or config.model:
            options.model = self._model or config.model

        client = await self._get_or_create_client(
            sdk, conv_id, options,
        )

        # --- Consume SDK stream ---
        pending: dict[str, tuple[str, dict[str, Any], float]] = {}
        got_stream_events = False
        response_text = ""

        try:
            await client.query(prompt, session_id=conv_id)
            async for message in client.receive_response():
                if isinstance(message, sdk.StreamEvent):
                    got_stream_events = True
                    evt = message.event
                    evt_type = evt.get("type", "")

                    if evt_type == "content_block_start":
                        block = evt.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "unknown")
                            pending[tool_id] = (
                                tool_name, {}, time.monotonic(),
                            )

                    elif evt_type == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                response_text += text
                                q.put(TextChunk(text=text))
                        elif delta.get("type") == "input_json_delta":
                            # Buffer tool arguments
                            partial = delta.get("partial_json", "")
                            # Accumulate per tool_id (omitted for brevity)

                elif isinstance(message, sdk.UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, sdk.ToolResultBlock):
                                entry = pending.pop(
                                    block.tool_use_id,
                                    ("unknown", {}, time.monotonic()),
                                )
                                name, args, start = entry
                                duration_ms = (
                                    (time.monotonic() - start) * 1000
                                )
                                q.put(ToolCallObserved(
                                    call_id=block.tool_use_id,
                                    name=name,
                                    arguments=args,
                                    result=_extract_text(block.content),
                                    status=(
                                        "error"
                                        if block.is_error
                                        else "success"
                                    ),
                                    duration_ms=duration_ms,
                                ))

                elif isinstance(message, sdk.AssistantMessage):
                    if not got_stream_events:
                        for block in message.content:
                            if isinstance(block, sdk.TextBlock):
                                response_text += block.text
                                q.put(TextChunk(text=block.text))
                            elif isinstance(block, sdk.ToolUseBlock):
                                pending[block.id] = (
                                    block.name,
                                    block.input,
                                    time.monotonic(),
                                )

                elif isinstance(message, sdk.ResultMessage):
                    if not response_text and message.result:
                        response_text = message.result

                elif isinstance(message, sdk.SystemMessage):
                    if message.subtype == "api_retry":
                        status = message.data.get("error_status")
                        if status in {401, 403}:
                            q.put(ExecutorError(
                                message=(
                                    "Claude SDK auth failed: "
                                    f"{message.data.get('error')}"
                                ),
                                code="auth_failed",
                            ))
                            return

            q.put(TurnComplete(text=response_text or None))

        except Exception as exc:
            await self._close_client(conv_id)
            q.put(ExecutorError(
                message=f"Claude SDK error: {exc}",
            ))

    async def _get_or_create_client(
        self,
        sdk: Any,
        conversation_id: str,
        options: Any,
    ) -> Any:
        """
        Return persistent client for this conversation, creating if needed.

        :param sdk: The claude_agent_sdk module.
        :param conversation_id: Conversation identifier.
        :param options: ClaudeAgentOptions for client creation.
        :returns: ClaudeSDKClient instance.
        """
        client = self._clients.get(conversation_id)
        if client is None:
            client = sdk.ClaudeSDKClient(options)
            await client.connect()
            self._clients[conversation_id] = client
        return client

    async def _close_client(self, conversation_id: str) -> None:
        """
        Disconnect and remove the client for a conversation.

        :param conversation_id: Conversation identifier.
        """
        client = self._clients.pop(conversation_id, None)
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass
```

---

## Implementation 3: RemoteExecutor

Delegates to a remote agent service over HTTP. The remote service
manages its own agent loop, tools, prompt, and session state. Any
framework (OmniAgents, custom, any language) can implement the endpoint.

### REST protocol: `POST /v1/turns`

#### Normal request

```http
POST /v1/turns HTTP/1.1
Content-Type: application/json
Accept: text/event-stream

{
  "conversation_id": "conv_abc123",
  "new_messages": [
    {"role": "user", "content": "What files are in the repo?"}
  ]
}
```

#### Normal response (SSE stream)

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"type": "text_chunk", "text": "Let me check..."}
data: {"type": "tool_call_observed", "call_id": "c1", "name": "Bash", "arguments": {"command": "ls"}, "result": "README.md\nsrc/", "status": "success", "duration_ms": 42.0}
data: {"type": "text_chunk", "text": "I found README.md and src/."}
data: {"type": "turn_complete", "text": "I found README.md and src/."}
```

#### Heartbeat (long-running tool execution)

Remote services MUST send periodic heartbeat events at least every 30
seconds while processing (internal tool execution, long inference, etc.).
Agent-plane uses the absence of heartbeats to detect dead connections.

```http
data: {"type": "heartbeat"}
```

#### Session not found (remote service crashed/restarted)

```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{"error": "session_not_found", "conversation_id": "conv_abc123"}
```

#### Recovery request (agent-plane retries with full history)

```http
POST /v1/turns HTTP/1.1
Content-Type: application/json
Accept: text/event-stream

{
  "conversation_id": "conv_abc123",
  "new_messages": [
    {"role": "user", "content": "Now show me the README"}
  ],
  "history": [
    {"role": "user", "content": "What files are in the repo?"},
    {"role": "assistant", "content": "I found README.md and src/.",
     "tool_calls": [
       {"call_id": "c1", "name": "Bash", "arguments": {"command": "ls"}}
     ]},
    {"role": "tool", "call_id": "c1", "name": "Bash",
     "content": "README.md\nsrc/", "status": "success"}
  ]
}
```

Remote service rebuilds session from `history`, processes `new_messages`.

#### Request schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversation_id` | string | yes | Keys the remote session |
| `new_messages` | array | yes | New message(s) to process |
| `history` | array | no | Full prior conversation. Sent only on recovery after 404 |

#### Message format (used in `new_messages` and `history`)

| role | Fields | Description |
|------|--------|-------------|
| `user` | `content` (string) | User message |
| `assistant` | `content` (string), `tool_calls` (array, optional) | Assistant response. `tool_calls` is a list of `{call_id, name, arguments}` for any tool invocations the assistant made. |
| `tool` | `call_id`, `name`, `content` (string), `status` (`"success"` \| `"error"`) | Tool result. `call_id` matches the originating tool call. |

#### SSE event types

| type | Fields | Semantics |
|------|--------|-----------|
| `text_chunk` | `text` | Streaming text delta |
| `tool_call_requested` | `call_id`, `name`, `arguments` | Remote agent wants workflow to execute (see below) |
| `tool_call_observed` | `call_id`, `name`, `arguments`, `result`, `status`, `duration_ms` | Already executed internally |
| `turn_complete` | `text` (nullable) | Terminal — turn finished |
| `heartbeat` | *(none)* | Keepalive — remote is still working |
| `error` | `message`, `code` (nullable) | Terminal — unrecoverable failure |

**Parallel tool requests**: the remote service MAY emit multiple
`tool_call_requested` events before `turn_complete`. Agent-plane
collects all of them, executes (or parks) each, then sends all
results back in one `POST /v1/turns` request:

```json
{
  "conversation_id": "conv_abc123",
  "new_messages": [
    {"role": "tool", "call_id": "c1", "name": "Read",
     "content": "file contents...", "status": "success"},
    {"role": "tool", "call_id": "c2", "name": "Bash",
     "content": "error: command not found", "status": "error"}
  ]
}
```

The `turn_complete` event after `tool_call_requested` events signals
that the batch is complete — agent-plane executes the collected tools
and POSTs results.

#### Cancellation

Agent-plane closing the SSE connection (HTTP client disconnect) is the
cancellation signal. Remote services SHOULD treat a closed connection
as "stop work on this turn." The conversation remains valid — a
subsequent `POST /v1/turns` continues from the last completed state.

#### Recovery handshake

```
Agent-plane                          Remote service
    |                                      |
    |  POST {conversation_id,              |
    |        new_messages}                 |
    |------------------------------------->|
    |                                      |
    |         [has session] 200 + SSE      |
    |<-------------------------------------|
    |                                      |

    --- OR ---

    |  POST {conversation_id,              |
    |        new_messages}                 |
    |------------------------------------->|
    |                                      |
    |         [no session] 404             |
    |<-------------------------------------|
    |                                      |
    |  POST {conversation_id,              |
    |        new_messages, history}         |
    |------------------------------------->|
    |                                      |
    |         [rebuilt] 200 + SSE          |
    |<-------------------------------------|
```

### Python implementation

```python
class RemoteExecutor(Executor):
    """
    Executor that delegates to a remote agent service over HTTP.

    The remote service manages its own agent loop, tools, prompt, and
    session state. Agent-plane sends messages, observes the event stream,
    and persists events for durability and SSE relay.

    :param endpoint: URL of the remote ``POST /v1/turns`` endpoint,
        e.g. ``"https://my-agent:8000/v1/turns"``.
    :param timeout: HTTP request timeout in seconds, e.g. ``300``.
    :param headers: Optional HTTP headers (auth, etc.),
        e.g. ``{"Authorization": "Bearer ..."}``.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        timeout: int = 300,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._timeout = timeout
        self._headers = headers or {}

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> Self:
        """
        Build from agent spec's endpoint and headers.

        :param spec: Agent spec with llm.endpoint and optional headers.
        :returns: Configured RemoteExecutor.
        """
        assert spec.llm is not None
        assert spec.llm.endpoint is not None
        return cls(
            endpoint=spec.llm.endpoint,
            timeout=spec.llm.timeout,
            headers=spec.llm.headers,
        )

    def max_context_tokens(self) -> int | None:
        """
        Remote service manages its own context.

        :returns: None.
        """
        return None

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig,
        context: ExecutorContext,
    ) -> Iterator[ExecutorEvent]:
        """
        POST to the remote service and consume the SSE event stream.

        On 404 (session not found), retries with full history from
        conv_store so the remote service can rebuild.

        :param messages: Full conversation history from conv_store.
        :param tools: Ignored — remote service defines its own tools.
        :param system_prompt: Ignored — remote service defines its prompt.
        :param config: Ignored — remote service defines its config.
        :param context: Agent-plane capabilities and identifiers.
        """
        new_messages = _extract_new_messages(messages)
        body: dict[str, Any] = {
            "conversation_id": context.conversation_id,
            "new_messages": new_messages,
        }

        response = self._post(body)

        if response.status_code == 404:
            body["history"] = _messages_to_history(messages)
            response = self._post(body)

        if response.status_code != 200:
            yield ExecutorError(
                message=(
                    f"Remote executor returned {response.status_code}"
                ),
                code="remote_error",
            )
            return

        yield from self._consume_sse(response)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        """
        POST to the remote endpoint with streaming.

        :param body: JSON request body.
        :returns: Streaming HTTP response.
        """
        return httpx.post(
            self._endpoint,
            json=body,
            headers={
                "Accept": "text/event-stream",
                **self._headers,
            },
            timeout=self._timeout,
        )

    def _consume_sse(
        self, response: httpx.Response,
    ) -> Iterator[ExecutorEvent]:
        """
        Parse SSE data lines into ExecutorEvents.

        Heartbeat events are consumed silently (keepalive only).
        Multiple tool_call_requested events may appear before
        turn_complete — all are yielded for the workflow to batch.

        :param response: Streaming HTTP response.
        :yields: ExecutorEvent instances.
        """
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            evt = payload["type"]

            if evt == "text_chunk":
                yield TextChunk(text=payload["text"])

            elif evt == "tool_call_requested":
                yield ToolCallRequested(
                    call_id=payload["call_id"],
                    name=payload["name"],
                    arguments=payload["arguments"],
                )

            elif evt == "tool_call_observed":
                yield ToolCallObserved(
                    call_id=payload["call_id"],
                    name=payload["name"],
                    arguments=payload["arguments"],
                    result=payload["result"],
                    status=payload["status"],
                    duration_ms=payload["duration_ms"],
                )

            elif evt == "turn_complete":
                yield TurnComplete(text=payload.get("text"))
                return

            elif evt == "heartbeat":
                # Keepalive — no event to yield
                continue

            elif evt == "error":
                yield ExecutorError(
                    message=payload["message"],
                    code=payload.get("code"),
                )
                return
```

---

## Executor Construction

```python
EXECUTOR_TYPES: dict[str, type[Executor]] = {
    "default": DefaultExecutor,
    "claude_sdk": ClaudeSDKExecutor,
    "remote": RemoteExecutor,
}


def _create_executor(spec: AgentSpec) -> Executor:
    """
    Construct the executor for the given agent spec.

    Dispatches to the appropriate subclass's ``from_spec()`` based
    on the ``executor`` field in the LLM config.

    :param spec: The parsed AgentSpec with a non-None llm field.
    :returns: A concrete Executor instance.
    """
    assert spec.llm is not None
    executor_type = spec.llm.executor or "default"
    cls = EXECUTOR_TYPES[executor_type]
    return cls.from_spec(spec)
```

---

## Agent Spec Changes

`LLMConfig` gains:

```python
@dataclass
class LLMConfig:
    ...
    executor: str | None = None    # "claude_sdk", "remote", or None (default)
    endpoint: str | None = None    # URL for remote executor
    headers: dict[str, str] | None = None  # Auth headers for remote executor
```

Existing specs with no `executor` field use `DefaultExecutor` unchanged.

---

## Deploying an OmniAgents Agent on Agent-Plane

OmniAgents agents are deployed as remote services. The agent runs in its
own process with its own session management, tools, and policies.
Agent-plane connects via `RemoteExecutor`.

### Step 1: Add a `/v1/turns` endpoint to the OmniAgents server

A thin adapter on the existing `create_app()` server that speaks the
executor SSE protocol:

```python
# omniagents side: add to server.py
@app.route("/v1/turns", methods=["POST"])
async def handle_turn(request):
    body = await request.json()
    conversation_id = body["conversation_id"]
    new_messages = body["new_messages"]
    history = body.get("history")

    session = manager.get_session(conversation_id)
    if session is None and history is None:
        return JSONResponse(
            {"error": "session_not_found",
             "conversation_id": conversation_id},
            status_code=404,
        )
    if session is None:
        session = manager.create_session_from_history(
            conversation_id, history,
        )

    for msg in new_messages:
        if msg["role"] == "user":
            await session.send(msg["content"])
        elif msg["role"] == "tool":
            await session.deliver_tool_result(
                msg["call_id"], msg["content"],
                # status lets the agent know if the tool errored
                status=msg.get("status", "success"),
            )

    async def event_stream():
        async for event in session.stream_turn():
            if isinstance(event, TextChunk):
                yield _sse({"type": "text_chunk",
                            "text": event.text})
            elif isinstance(event, ToolCallRequest):
                yield _sse({"type": "tool_call_observed",
                            "call_id": _next_id(),
                            "name": event.name,
                            "arguments": event.args,
                            ...})
            elif isinstance(event, ToolCallComplete):
                # Pair with request — emit tool_call_observed
                ...
            elif isinstance(event, TurnComplete):
                yield _sse({"type": "turn_complete",
                            "text": event.response})
            # Heartbeats are sent automatically by the framework's
            # SSE middleware every 30s during long tool executions

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
```

### Step 2: Deploy the OmniAgents service

```bash
# Standard OmniAgents server with the /v1/turns endpoint
omniagents serve my_agent.yaml --port 8000
```

### Step 3: Register with agent-plane

```yaml
# agent-plane agent spec
name: my-omni-agent
llm:
  executor: remote
  endpoint: https://my-agent-service:8000/v1/turns
execution:
  timeout: 300
```

The OmniAgents agent's prompt, tools, policies, and executor are defined
in `my_agent.yaml` — agent-plane doesn't need to know about them.
Agent-plane provides durability (DBOS + conv_store), REST API
(`POST /v1/responses`), SSE streaming, and steering.

---

## Durability

The durability story is unchanged because all durable operations stay in
the workflow:

- **conv_store writes** happen in the event consumer loop — each
  `ToolCallRequested` and `ToolCallObserved` event triggers
  `_persist_and_stream` before the loop continues.

- **`_call_tool()` @step** is still called from the workflow for
  `ToolCallRequested` events. DBOS caches and skips on crash recovery.

- **`storage_dir` in `finally`**: artifact store persistence for SDK
  executors runs in the same `finally` block as compaction persistence.

- **RemoteExecutor crash recovery**: agent-plane re-invokes via DBOS,
  sends `conversation_id` to remote service. If remote has session →
  continues. If not → 404 → retry with history from conv_store.

- **ClaudeSDKExecutor crash recovery**: DBOS re-invokes, workflow
  restores `storage_dir`, SDK uses `--continue` if transcript exists
  or falls back to prompt reconstruction from conv_store.

The one durability tradeoff: `ToolCallObserved` (internal/remote
executors) is not @step-wrapped. On crash mid-turn, the last in-flight
tool may re-execute. Acceptable — built-in tools are idempotent.

---

## Test Plan

1. **DefaultExecutor text response**: TextChunk → TurnComplete → persist.
2. **DefaultExecutor tool call**: ToolCallRequested → @step → re-invoke.
3. **DefaultExecutor context overflow**: ContextWindowExceeded → compact → retry.
4. **DefaultExecutor client-side tool**: ToolCallRequested → park/complete.
5. **ClaudeSDKExecutor internal tool**: ToolCallObserved → persist only.
6. **ClaudeSDKExecutor client-side tool**: tool routes through
   `await_tool_output` → parks → result returns to SDK.
7. **ClaudeSDKExecutor storage_dir lifecycle**: on_task_start reads,
   on_task_end writes, workflow persists to artifact store.
8. **RemoteExecutor normal flow**: POST → SSE → events persisted.
9. **RemoteExecutor recovery**: 404 → retry with history (including
   tool calls and results with status) → SSE.
10. **RemoteExecutor tool_call_requested**: remote requests tool → workflow
    executes → POST again with result including status.
11. **RemoteExecutor parallel tool requests**: multiple tool_call_requested
    events before turn_complete → workflow batches and executes all →
    POST results back in single request.
12. **RemoteExecutor heartbeat**: long-running turn sends heartbeats →
    agent-plane stays connected. Missing heartbeats → connection timeout.
13. **RemoteExecutor cancellation**: agent-plane closes SSE connection →
    remote stops work → next POST continues from last completed state.
14. **RemoteExecutor tool error propagation**: tool result with
    `status: "error"` delivered to remote → remote sees the error.
15. **Steering**: TurnComplete → persist-first-then-check → steered messages
    trigger next run_turn(). Works for all executor types.
16. **Crash recovery (Default)**: DBOS re-invokes, @step cached.
17. **Crash recovery (SDK)**: DBOS re-invokes, storage_dir restored.
18. **Crash recovery (Remote)**: DBOS re-invokes, 404 → history retry.
19. **Backward compatibility**: no `executor` field → DefaultExecutor.
20. **from_spec construction**: each executor type constructs correctly
    from AgentSpec via standardized `from_spec()` classmethod.

____

QUESTIONS FOR LATER:

- For claude code, how does steering work with multiple replicas and storage dirs? How does the other replica find the same storage dir? Do we call on_task_start() again? 
