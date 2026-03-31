"""ACP adapter bridging Toad <-> agent-plane server.

This module implements the ACP (Agent Client Protocol) handlers that
Toad expects, translating each call into the corresponding agent-plane
HTTP API request and streaming SSE events back as ACP notifications.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from integrations.toad.events import EventTranslator
from integrations.toad.jsonrpc import Server
from integrations.toad.mcp_client import (
    McpConnection,
    parse_mcp_server_params,
)

log = logging.getLogger(__name__)

# ACP protocol version we advertise to Toad.
_PROTOCOL_VERSION = 1


@dataclass
class SessionState:
    """Per-session state tracking conversation continuity.

    :param cwd: The working directory Toad opened the session in.
    :param conversation_id: The agent-plane conversation ID for this
        session, populated after the first prompt completes.
    :param translator: Event translator that tracks the last
        ``response_id`` for multi-turn conversation threading.
    :param mcp_connections: Active MCP server connections for this
        session, keyed by server name.
    :param tool_schemas: OpenAI-format tool schemas discovered from
        MCP servers, passed in the ``tools`` field of each request.
    """

    cwd: str
    conversation_id: str | None = None
    translator: EventTranslator = field(default_factory=EventTranslator)
    mcp_connections: list[McpConnection] = field(default_factory=list)
    tool_schemas: list[dict[str, object]] = field(default_factory=list)
    # Lookup from tool name to the MCP connection that owns it
    _tool_to_connection: dict[str, McpConnection] = field(default_factory=dict)


def create_adapter(
    server_url: str,
    agent_name: str,
) -> Server:
    """Build a JSONRPC :class:`Server` with all ACP handlers wired up.

    :param server_url: Base URL of the agent-plane server, e.g.
        ``"http://localhost:18400"``.
    :param agent_name: The agent name (``model`` field in
        ``/v1/responses``), e.g. ``"archer"``.
    :returns: A fully configured :class:`Server` ready to
        :meth:`Server.run`.
    """
    rpc = Server()
    sessions: dict[str, SessionState] = {}
    client = httpx.AsyncClient(
        base_url=server_url,
        timeout=httpx.Timeout(
            connect=10.0,
            # Read timeout must be long — agent responses can
            # take minutes for complex tasks.
            read=600.0,
            write=10.0,
            pool=10.0,
        ),
    )

    @rpc.method("initialize")
    async def handle_initialize(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Respond to Toad's handshake with our capabilities.

        :param params: ``{"protocolVersion": int,
            "clientCapabilities": {...}, "clientInfo": {...}}``.
        :returns: Protocol version and agent capabilities.
        """
        return {
            "protocolVersion": _PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": True,
                "sessionCapabilities": {"list": True},
                "promptCapabilities": {
                    "image": True,
                    "embeddedContext": True,
                    "supportedMediaTypes": [
                        "text/plain",
                        "image/png",
                        "image/jpeg",
                        "image/gif",
                        "image/webp",
                    ],
                },
            },
        }

    @rpc.method("session/new")
    async def handle_session_new(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Create a new chat session.

        Connects to any MCP servers Toad provides, discovers
        their tools, and stores schemas for use in prompts.

        :param params: ``{"cwd": str, "mcpServers": [...]}``.
        :returns: ``{"sessionId": str}``
        """
        import uuid

        session_id = uuid.uuid4().hex[:16]
        # cwd is required by ACP session/new spec
        cwd = str(params["cwd"])
        session = SessionState(cwd=cwd)
        # Connect to MCP servers and discover tools
        mcp_servers_raw = params.get("mcpServers", [])
        if isinstance(mcp_servers_raw, list):
            await _connect_mcp_servers(session, mcp_servers_raw, cwd)
        sessions[session_id] = session
        return {"sessionId": session_id}

    @rpc.method("session/prompt")
    async def handle_session_prompt(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Send a user prompt and stream the agent response.

        Calls ``POST /v1/responses`` with ``stream: true``, parses
        the SSE stream, translates events to ACP
        ``session/update`` notifications, and returns when the
        stream ends. If the LLM invokes client-side tools, those
        are executed via MCP, results PATCHed back, and the loop
        continues.

        :param params: ``{"sessionId": str, "prompt":
            [{"type": "text", "text": str}, ...]}``.
        :returns: ``{"stopReason": str}`` reflecting how the
            response terminated.
        """
        session_id = str(params.get("sessionId", ""))
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        session.translator.reset_for_prompt()
        input_blocks = await _build_input(client, params.get("prompt", []))

        payload: dict[str, object] = {
            "model": agent_name,
            "input": input_blocks,
            "stream": True,
            "store": True,
        }
        if session.tool_schemas:
            payload["tools"] = session.tool_schemas
        prev_id = session.translator.last_response_id
        if prev_id is not None:
            payload["previous_response_id"] = prev_id

        await _prompt_with_tool_loop(client, rpc, session_id, session, agent_name, payload)
        stop_reason = session.translator.stop_reason or "end_turn"
        return {"stopReason": stop_reason}

    @rpc.method("session/cancel")
    async def handle_session_cancel(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Cancel an in-progress response.

        :param params: ``{"sessionId": str}``.
        :returns: Empty dict.
        """
        session_id = str(params.get("sessionId", ""))
        session = sessions.get(session_id)
        if session is None:
            return {}
        resp_id = session.translator.last_response_id
        if resp_id is not None:
            try:
                await client.post(f"/v1/responses/{resp_id}/cancel")
            except httpx.HTTPError:
                log.warning("Failed to cancel response %s", resp_id)
        return {}

    @rpc.method("fs/read_text_file")
    async def handle_read_file(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Read a local file (Toad file-system delegation).

        Agent-plane agents execute server-side so this reads from
        the local filesystem where Toad is running.

        :param params: ``{"sessionId": str, "path": str}``.
        :returns: ``{"content": str}``
        """
        import pathlib

        # path is required by ACP fs/read_text_file spec
        path = pathlib.Path(str(params["path"]))
        try:
            content = path.read_text()
        except OSError as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc
        return {"content": content}

    @rpc.method("fs/write_text_file")
    async def handle_write_file(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Write a local file (Toad file-system delegation).

        :param params: ``{"sessionId": str, "path": str,
            "content": str}``.
        :returns: Empty dict on success.
        """
        import pathlib

        # path and content are required by ACP fs/write_text_file spec
        path = pathlib.Path(str(params["path"]))
        content = str(params["content"])
        try:
            path.write_text(content)
        except OSError as exc:
            raise ValueError(f"Cannot write {path}: {exc}") from exc
        return {}

    @rpc.method("session/list")
    async def handle_session_list(
        params: dict[str, object],
    ) -> dict[str, object]:
        """List saved conversations from agent-plane.

        :param params: ``{"cursor": str | None, "limit": int}``.
        :returns: ``{"sessions": [...], "nextCursor": str | None}``
        """
        limit = int(params.get("limit", 20))
        cursor = params.get("cursor")
        query: dict[str, object] = {
            "limit": limit,
            "order": "desc",
        }
        if cursor is not None:
            query["after"] = str(cursor)
        resp = await client.get("/v1/conversations", params=query)
        resp.raise_for_status()
        body = resp.json()
        sessions_list = _map_conversations_to_sessions(body)
        next_cursor = body["last_id"] if body.get("has_more") else None
        return {
            "sessions": sessions_list,
            "nextCursor": next_cursor,
        }

    @rpc.method("session/load")
    async def handle_session_load(
        params: dict[str, object],
    ) -> dict[str, object]:
        """Load a saved conversation and replay items as ACP events.

        :param params: ``{"sessionId": str}``.
        :returns: ``{"sessionId": str}`` with a new local
            session wired to the loaded conversation.
        """
        conv_id = str(params["sessionId"])
        import uuid

        session_id = uuid.uuid4().hex[:16]
        items = await _fetch_all_items(client, conv_id)
        translator = EventTranslator()
        # Track the last response_id from loaded items
        for item in items:
            resp_id = item.get("response_id")
            if resp_id is not None:
                translator.last_response_id = str(resp_id)
        translator.last_conversation_id = conv_id
        sessions[session_id] = SessionState(
            cwd="/",
            conversation_id=conv_id,
            translator=translator,
        )
        for item in items:
            updates = _replay_item(item)
            for update in updates:
                await rpc.notify(
                    "session/update",
                    {"sessionId": session_id, "update": update},
                )
        return {"sessionId": session_id}

    return rpc


async def _build_input(
    client: httpx.AsyncClient,
    prompt: object,
) -> list[dict[str, object]]:
    """Convert ACP prompt content blocks to agent-plane input blocks.

    Handles text, image (uploaded via ``/v1/files``), resource
    (inline content), and resource link (local file read) blocks.

    :param client: The httpx async client for image uploads.
    :param prompt: The raw ``prompt`` value from JSONRPC params,
        expected to be a list of dicts.
    :returns: A list of agent-plane input content blocks, e.g.
        ``[{"type": "input_text", "text": "Hello"}]``.
    """
    if not isinstance(prompt, list):
        return [{"type": "input_text", "text": str(prompt)}]
    blocks: list[dict[str, object]] = []
    for item in prompt:
        if not isinstance(item, dict):
            continue
        block_type = item.get("type")
        if block_type == "text":
            blocks.append({"type": "input_text", "text": str(item["text"])})
        elif block_type == "image":
            file_id = await _upload_image(client, item)
            if file_id is not None:
                blocks.append({"type": "input_image", "file_id": file_id})
        elif block_type == "resource":
            blocks.append(_convert_resource_block(item))
        elif block_type == "resource_link":
            blocks.append(_convert_resource_link(item))
    return blocks


async def _upload_image(
    client: httpx.AsyncClient,
    block: dict[str, object],
) -> str | None:
    """Upload an image block to agent-plane's ``/v1/files``.

    Reads the image data from the block's ``data`` field (base64)
    or ``url`` field and uploads it.

    :param client: The httpx async client.
    :param block: The ACP image content block with ``data`` or
        ``url`` field.
    :returns: The file ID from the upload response, or ``None``
        if the upload fails.
    """
    import base64

    data = block.get("data")
    if data is not None:
        # Base64-encoded image data
        image_bytes = base64.b64decode(str(data))
        media_type = str(block.get("mimeType", "image/png"))
        resp = await client.post(
            "/v1/files",
            files={
                "file": ("image", image_bytes, media_type),
            },
            data={"purpose": "user_data"},
        )
        if resp.status_code == 200:
            return str(resp.json().get("id"))
    return None


def _convert_resource_block(
    block: dict[str, object],
) -> dict[str, object]:
    """Convert an ACP resource block to an input_text block.

    Resource blocks carry inline content from embedded context
    (e.g. file snippets pasted by the user).

    :param block: The ACP resource block with ``resource``
        containing ``uri`` and ``text`` or ``blob``.
    :returns: An ``input_text`` block with the resource content.
    """
    resource = block.get("resource")
    if not isinstance(resource, dict):
        resource = {}
    # uri is required per ACP resource block spec
    uri = str(resource.get("uri", "unknown"))
    # Content is in "text" for text resources, "blob" for binary
    text = str(resource.get("text") or resource.get("blob") or "")
    return {
        "type": "input_text",
        "text": f"--- file: {uri} ---\n{text}",
    }


def _convert_resource_link(
    block: dict[str, object],
) -> dict[str, object]:
    """Convert an ACP resource_link block to an input_text block.

    Reads local ``file://`` URIs; for other schemes, includes the
    URI as a text reference.

    :param block: The ACP resource_link block with ``uri`` field.
    :returns: An ``input_text`` block with file content or URI
        reference.
    """
    # uri is required per ACP resource_link spec
    uri = str(block.get("uri", "unknown"))
    if uri.startswith("file://"):
        import pathlib

        path = pathlib.Path(uri[len("file://") :])
        try:
            content = path.read_text()
            return {
                "type": "input_text",
                "text": f"--- file: {uri} ---\n{content}",
            }
        except OSError:
            log.warning("Cannot read resource link: %s", uri)
    return {
        "type": "input_text",
        "text": f"[resource: {uri}]",
    }


def _map_conversations_to_sessions(
    body: dict[str, object],
) -> list[dict[str, object]]:
    """Map agent-plane conversation list response to ACP sessions.

    :param body: The JSON response from ``GET /v1/conversations``.
    :returns: List of ACP session dicts with ``sessionId``,
        ``title``, ``cwd``, and ``updatedAt`` fields.
    """
    data = body.get("data", [])
    if not isinstance(data, list):
        return []
    sessions_list: list[dict[str, object]] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        sessions_list.append(
            {
                "sessionId": str(conv["id"]),
                "cwd": "/",
                # ACP requires a non-null title string; "Untitled" is
                # the Toad convention for unnamed conversations.
                "title": conv.get("title") or "Untitled",
                "updatedAt": _unix_to_iso8601(conv.get("created_at")),
            }
        )
    return sessions_list


def _unix_to_iso8601(timestamp: object) -> str | None:
    """Convert a Unix epoch timestamp to ISO 8601 string.

    :param timestamp: Unix epoch seconds (int or float), or
        ``None``.
    :returns: ISO 8601 formatted string, e.g.
        ``"2026-03-30T12:00:00+00:00"``, or ``None`` if
        *timestamp* is ``None``.
    """
    if timestamp is None:
        return None
    dt = datetime.fromtimestamp(float(str(timestamp)), tz=timezone.utc)
    return dt.isoformat()


async def _fetch_all_items(
    client: httpx.AsyncClient,
    conversation_id: str,
) -> list[dict[str, object]]:
    """Fetch all items from a conversation, paginating as needed.

    :param client: The httpx async client.
    :param conversation_id: The agent-plane conversation ID.
    :returns: All conversation items in chronological order.
    """
    items: list[dict[str, object]] = []
    after: str | None = None
    while True:
        params: dict[str, object] = {
            "order": "asc",
            "limit": 100,
        }
        if after is not None:
            params["after"] = after
        resp = await client.get(
            f"/v1/conversations/{conversation_id}/items",
            params=params,
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data", [])
        if not isinstance(data, list):
            break
        for item in data:
            if isinstance(item, dict):
                items.append(item)
        if not body.get("has_more"):
            break
        after = body.get("last_id")
        if after is None:
            break
    return items


def _replay_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Convert a conversation item to ACP session/update dicts.

    Maps each item type to the appropriate ACP notification:
    user message -> ``user_message_chunk``, assistant message ->
    ``agent_message_chunk``, function_call -> ``tool_call``,
    function_call_output -> ``tool_call_update``, reasoning ->
    ``agent_thought_chunk``.

    :param item: A conversation item dict from the API.
    :returns: List of ACP update dicts (usually one).
    """
    item_type = item.get("type")
    if item_type == "message":
        return _replay_message_item(item)
    if item_type == "function_call":
        return _replay_function_call_item(item)
    if item_type == "function_call_output":
        return _replay_function_call_output_item(item)
    if item_type == "reasoning":
        return _replay_reasoning_item(item)
    return []


def _replay_message_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Replay a message item as ACP user/agent message chunks.

    :param item: The message item dict with ``role`` and
        ``content`` fields.
    :returns: List of ACP update dicts.
    """
    role = item.get("role")
    content = item.get("content", [])
    update_type = "user_message_chunk" if role == "user" else "agent_message_chunk"
    updates: list[dict[str, object]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                # "text" may be absent on non-text content blocks
                text = str(block.get("text") or "")
                if text:
                    updates.append(
                        {
                            "sessionUpdate": update_type,
                            "content": {"type": "text", "text": text},
                        }
                    )
    return updates


def _replay_function_call_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Replay a function_call item as an ACP tool_call.

    :param item: The function_call item dict with ``call_id``,
        ``name``, and ``arguments`` fields.
    :returns: Single ``tool_call`` update.
    """
    # call_id, name, arguments are required per API.md item schema
    call_id = str(item["call_id"])
    name = str(item["name"])
    arguments = str(item["arguments"])
    return [
        {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": name,
            "status": "completed",
            "tool": {"name": name, "parameters": arguments},
        }
    ]


def _replay_function_call_output_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Replay a function_call_output item as an ACP tool_call_update.

    :param item: The function_call_output item dict with
        ``call_id`` and ``output`` fields.
    :returns: Single ``tool_call_update`` update.
    """
    # call_id, output are required per API.md item schema
    call_id = str(item["call_id"])
    output = str(item["output"])
    return [
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "completed",
            "content": {"type": "text", "text": output},
        }
    ]


def _replay_reasoning_item(
    item: dict[str, object],
) -> list[dict[str, object]]:
    """Replay a reasoning item as an ACP agent_thought_chunk.

    :param item: The reasoning item dict with ``summary`` field.
    :returns: Single ``agent_thought_chunk`` update, or empty if
        no summary text.
    """
    summary = item.get("summary", [])
    updates: list[dict[str, object]] = []
    if isinstance(summary, list):
        for block in summary:
            if isinstance(block, dict):
                # "text" may be absent on non-text content blocks
                text = str(block.get("text") or "")
                if text:
                    updates.append(
                        {
                            "sessionUpdate": "agent_thought_chunk",
                            "content": {"type": "text", "text": text},
                        }
                    )
    return updates


async def _stream_response(
    client: httpx.AsyncClient,
    rpc: Server,
    session_id: str,
    translator: EventTranslator,
    payload: dict[str, object],
) -> None:
    """POST to ``/v1/responses`` and stream SSE back as ACP updates.

    Parses the SSE ``event:``/``data:`` lines, translates each
    event via *translator*, and sends resulting ACP
    ``session/update`` notifications through *rpc*.

    :param client: The httpx async client pointed at agent-plane.
    :param rpc: The JSONRPC server for sending notifications.
    :param session_id: ACP session ID for the notification params.
    :param translator: Stateful event translator that accumulates
        tool call state and tracks the last response ID.
    :param payload: The JSON body for ``POST /v1/responses``.
    """
    async with client.stream("POST", "/v1/responses", json=payload) as resp:
        resp.raise_for_status()
        event_type: str | None = None
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                raw = line[len("data:") :].strip()
                if raw == "[DONE]":
                    break
                if event_type is None:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                updates = translator.translate(event_type, data)
                for update in updates:
                    await rpc.notify(
                        "session/update",
                        {
                            "sessionId": session_id,
                            "update": update,
                        },
                    )
                # Reset for next event pair
                event_type = None


async def _prompt_with_tool_loop(
    client: httpx.AsyncClient,
    rpc: Server,
    session_id: str,
    session: SessionState,
    agent_name: str,
    payload: dict[str, object],
) -> None:
    """Stream a response and execute client-side tools in a loop.

    After each streamed response, checks for pending client-side
    tool calls. If any exist, executes them via MCP, PATCHes
    results back, and sends a follow-up ``POST /v1/responses``
    to continue the agent loop.

    :param client: The httpx async client pointed at agent-plane.
    :param rpc: The JSONRPC server for sending notifications.
    :param session_id: ACP session ID.
    :param session: The session state with MCP connections.
    :param agent_name: Agent model name for follow-up requests.
    :param payload: Initial JSON body for ``POST /v1/responses``.
    """
    # Cap iterations to prevent runaway tool loops
    max_iterations = 50
    for _ in range(max_iterations):
        session.translator.reset_for_prompt()
        await _stream_response(client, rpc, session_id, session.translator, payload)
        _sync_session_from_translator(session)
        pending = session.translator.pending_client_tool_calls
        if not pending:
            break
        await _execute_and_patch_tools(client, rpc, session_id, session, pending)
        # Continue with a follow-up request
        payload = _build_continuation_payload(agent_name, session)


def _sync_session_from_translator(session: SessionState) -> None:
    """Copy translator state back to session after a stream.

    :param session: The session to update.
    """
    if session.translator.last_conversation_id is not None:
        session.conversation_id = session.translator.last_conversation_id


def _build_continuation_payload(
    agent_name: str,
    session: SessionState,
) -> dict[str, object]:
    """Build a follow-up POST /v1/responses payload after tool execution.

    :param agent_name: Agent model name.
    :param session: Session with updated response_id and tools.
    :returns: A payload dict for the next ``POST /v1/responses``.
    """
    payload: dict[str, object] = {
        "model": agent_name,
        "input": [],
        "stream": True,
        "store": True,
    }
    if session.tool_schemas:
        payload["tools"] = session.tool_schemas
    prev_id = session.translator.last_response_id
    if prev_id is not None:
        payload["previous_response_id"] = prev_id
    return payload


async def _execute_and_patch_tools(
    client: httpx.AsyncClient,
    rpc: Server,
    session_id: str,
    session: SessionState,
    pending: list[dict[str, str]],
) -> None:
    """Execute pending client-side tool calls via MCP and PATCH results.

    For each pending tool call, finds the MCP connection that owns
    it, invokes the tool, sends ``tool_call_update`` ACP
    notifications, and PATCHes all results back to agent-plane.

    :param client: The httpx async client.
    :param rpc: The JSONRPC server for notifications.
    :param session_id: ACP session ID.
    :param session: The session with MCP connections.
    :param pending: List of dicts with ``call_id``, ``name``,
        ``arguments`` for each pending call.
    """
    tool_results: list[dict[str, str]] = []
    for call in pending:
        call_id = call["call_id"]
        name = call["name"]
        arguments_str = call["arguments"]
        output = await _execute_single_tool(session, name, arguments_str)
        tool_results.append({"call_id": call_id, "output": output})
        # Notify Toad that the tool call completed
        await rpc.notify(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "completed",
                    "content": {
                        "type": "text",
                        "text": output,
                    },
                },
            },
        )
    # PATCH results back to agent-plane
    resp_id = session.translator.last_response_id
    if resp_id is not None and tool_results:
        await client.patch(
            f"/v1/responses/{resp_id}",
            json={"tool_results": tool_results},
        )


async def _execute_single_tool(
    session: SessionState,
    name: str,
    arguments_str: str,
) -> str:
    """Execute one tool call via the session's MCP connections.

    :param session: Session with ``_tool_to_connection`` lookup.
    :param name: Tool name to invoke.
    :param arguments_str: JSON-encoded arguments string.
    :returns: Tool output string, or error message if execution
        fails.
    """
    connection = session._tool_to_connection.get(name)
    if connection is None:
        return f"[Error] Tool {name!r} not found in MCP servers"
    try:
        arguments = json.loads(arguments_str)
    except json.JSONDecodeError:
        arguments = {}
    try:
        return await connection.call_tool(name, arguments)
    except Exception as exc:
        log.warning("MCP tool %s failed: %s", name, exc)
        return f"[Error] {exc}"


async def _connect_mcp_servers(
    session: SessionState,
    mcp_servers_raw: list[object],
    cwd: str,
) -> None:
    """Connect to MCP servers and discover their tools.

    :param session: Session to populate with connections and
        tool schemas.
    :param mcp_servers_raw: Raw MCP server dicts from ACP
        ``session/new`` params.
    :param cwd: Working directory for subprocess-based servers.
    """
    for raw in mcp_servers_raw:
        if not isinstance(raw, dict):
            continue
        server_name = str(raw.get("name", "unnamed"))
        try:
            params = parse_mcp_server_params(raw, cwd=cwd)
            conn = McpConnection(
                server_params=params,
                server_name=server_name,
            )
            tools = await conn.connect()
            session.mcp_connections.append(conn)
            for tool in tools:
                session.tool_schemas.append(tool.schema)
                session._tool_to_connection[tool.name] = conn
            log.info(
                "Connected to MCP server %s: %d tools",
                server_name,
                len(tools),
            )
        except Exception as exc:
            log.warning(
                "Failed to connect MCP server %s: %s",
                server_name,
                exc,
            )
