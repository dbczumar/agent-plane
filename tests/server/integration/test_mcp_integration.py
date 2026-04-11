"""Integration tests for MCP tool execution through the full pipeline.

Starts a real FastMCP HTTP (SSE) server with filesystem tools, creates
an agent whose MCP config points to the server, then runs the full
workflow pipeline with a mock LLM that calls the MCP tools. Verifies
that real tool outputs (directory listings, file contents) appear in
the persisted conversation items.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
import yaml
from mcp.server.fastmcp import FastMCP

from agent_plane.tools.mcp import clear_discovery_cache
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response

pytestmark = [pytest.mark.asyncio]


# ── Agent name used to link create_test_agent ↔ create_test_response ──

_AGENT_NAME = "mcp-test-agent"


# ── In-process MCP server ────────────────────────────────


def _create_mcp_server(root_dir: Path) -> FastMCP:
    """
    Build a FastMCP server with filesystem tools rooted at ``root_dir``.

    :param root_dir: Absolute path that the tools treat as the
        filesystem root, e.g. ``Path("/tmp/pytest-xyz/mcp_root")``.
    :returns: A configured ``FastMCP`` instance (not yet running).
    """
    mcp = FastMCP("test-filesystem")

    @mcp.tool()
    def list_directory(path: str = ".") -> str:
        """
        List files and directories at the given path.

        :param path: Relative path from the root directory.
        """
        resolved = (root_dir / path).resolve()
        if not resolved.is_relative_to(root_dir):
            raise ValueError(f"Path escapes root: {path}")
        if not resolved.is_dir():
            raise ValueError(f"Not a directory: {path}")
        entries: list[str] = []
        for entry in sorted(resolved.iterdir()):
            name = entry.name
            if entry.is_dir():
                name += "/"
            entries.append(name)
        return "\n".join(entries) if entries else "(empty directory)"

    @mcp.tool()
    def read_file(path: str) -> str:
        """
        Read the contents of a text file.

        :param path: Relative path to the file.
        """
        resolved = (root_dir / path).resolve()
        if not resolved.is_relative_to(root_dir):
            raise ValueError(f"Path escapes root: {path}")
        if not resolved.is_file():
            raise ValueError(f"Not a file: {path}")
        return resolved.read_text()

    return mcp


def _find_free_port() -> int:
    """
    Find an available TCP port on localhost.

    :returns: An unused port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Bundle builder ────────────────────────────────────────


def _add_text(tf: tarfile.TarFile, name: str, content: str) -> None:
    """
    Add a text file to a tarball.

    :param tf: The open TarFile to add to.
    :param name: Archive member name, e.g.
        ``"tools/mcp/filesystem.yaml"``.
    :param content: The text content of the file.
    """
    data = content.encode()
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _build_mcp_agent_bundle(mcp_url: str) -> bytes:
    """
    Build an agent bundle (tar.gz) with MCP config pointing to
    a running HTTP MCP server.

    :param mcp_url: The SSE endpoint URL of the running MCP
        server, e.g. ``"http://127.0.0.1:54321/sse"``.
    :returns: Raw tar.gz bytes ready for upload.
    """
    # str | int | dict: top-level YAML config mixes strings, ints, and nested dicts.
    config: dict[str, str | int | dict[str, str]] = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {"model": _AGENT_NAME},
    }

    mcp_yaml: dict[str, str] = {
        "name": "filesystem",
        "description": "Read-only filesystem tools for testing.",
        "transport": "http",
        "url": mcp_url,
    }

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_text(tf, "config.yaml", yaml.dump(config))
        _add_text(tf, "tools/mcp/filesystem.yaml", yaml.dump(mcp_yaml))
    return buf.getvalue()


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def mcp_work_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory with known files for the MCP
    filesystem server to serve.

    :returns: Path to the work directory containing test files.
    """
    work = tmp_path / "mcp_root"
    work.mkdir()
    (work / "hello.txt").write_text("Hello from MCP!")
    sub = work / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("Nested content.")
    return work


@pytest.fixture()
def mcp_server_url(mcp_work_dir: Path) -> Iterator[str]:
    """
    Start a real FastMCP HTTP server on a random port and yield
    its SSE endpoint URL. Shuts down on teardown.

    :returns: The SSE URL, e.g. ``"http://127.0.0.1:54321/sse"``.
    """
    port = _find_free_port()
    mcp = _create_mcp_server(mcp_work_dir)
    app = mcp.sse_app()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)

    # Run uvicorn in a daemon thread so it doesn't block the test
    thread = threading.Thread(
        target=server.run,
        daemon=True,
    )
    thread.start()

    # Wait for the server to be ready (uvicorn sets started flag).
    # Short sleep avoids spinning the CPU while the server boots.
    import time

    while not server.started:
        time.sleep(0.01)

    yield f"http://127.0.0.1:{port}/sse"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _clear_mcp_cache() -> None:
    """
    Clear the MCP discovery cache before each test so stale
    entries from other tests don't interfere.
    """
    clear_discovery_cache()


# ── Helpers ───────────────────────────────────────────────


async def _create_mcp_agent(
    client: httpx.AsyncClient,
    mcp_url: str,
) -> dict[str, Any]:
    """
    Upload an agent bundle with MCP config pointing to a running
    HTTP server.

    :param client: HTTP client for the test server.
    :param mcp_url: SSE endpoint URL of the MCP server, e.g.
        ``"http://127.0.0.1:54321/sse"``.
    :returns: The agent creation response JSON.
    """
    bundle = _build_mcp_agent_bundle(mcp_url)
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"Agent creation failed: {resp.status_code} {resp.text}"
    return resp.json()


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status.

    :param client: HTTP client for the server.
    :param response_id: The response/task ID to poll, e.g.
        ``"resp_abc123"``.
    :returns: The response JSON dict once completed or failed.
    :raises AssertionError: If the response doesn't complete
        within the polling window.
    """
    for _ in range(200):
        resp = await client.get(f"/v1/responses/{response_id}")
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.1)
    raise AssertionError(f"Response {response_id} did not reach terminal status")


async def _get_items(
    client: httpx.AsyncClient,
    conv_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all conversation items.

    :param client: HTTP client for the server.
    :param conv_id: The conversation ID, e.g. ``"conv_abc123"``.
    :returns: List of item dicts sorted by position.
    """
    resp = await client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    assert resp.status_code == 200
    # Any: API returns heterogeneous item dicts
    result: list[dict[str, Any]] = resp.json()["data"]
    return result


# ── Tests ─────────────────────────────────────────────────


async def test_mcp_list_directory_returns_real_files(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    mcp_server_url: str,
) -> None:
    """
    MCP ``list_directory`` tool returns real directory contents
    through the full workflow pipeline.

    The LLM is mocked to call ``list_directory``, then produce a
    final text response. The function_call_output in the
    conversation should contain the actual file listing from the
    temporary directory.
    """
    await _create_mcp_agent(client, mcp_server_url)

    # LLM call 1: return a tool call to list_directory
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_mcp_list_1",
                "name": "list_directory",
                "arguments": json.dumps({"path": "."}),
            },
        ],
    )
    # LLM call 2: after tool result, return final text
    mock_llm.add_call(text="Listed the directory.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="List the files",
        background=True,
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    items = await _get_items(client, conv_id)

    # 4 items: user message, function_call, function_call_output, assistant.
    # If fewer, the tool call didn't execute; if more, extra LLM rounds fired.
    assert len(items) == 4, (
        f"Expected 4 items [user, fc, fco, assistant], got {len(items)}: "
        f"{[i['type'] for i in items]}"
    )

    # User message — proves the input was persisted
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"

    # function_call for list_directory — proves the mock LLM's tool
    # call was routed through the workflow and persisted
    assert items[1]["type"] == "function_call"
    assert items[1]["name"] == "list_directory"
    assert items[1]["call_id"] == "call_mcp_list_1"

    # function_call_output with REAL filesystem data — proves the MCP
    # server was actually called and returned live directory contents.
    # If "hello.txt" is missing, the MCP connection or tool execution failed.
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_mcp_list_1"
    output = items[2]["output"]
    assert "hello.txt" in output, f"Expected 'hello.txt' in MCP output, got: {output!r}"
    assert "subdir/" in output, f"Expected 'subdir/' in MCP output, got: {output!r}"

    # Final assistant message — proves the second LLM call completed
    assert items[3]["type"] == "message"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "Listed the directory."


async def test_mcp_read_file_returns_real_content(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    mcp_server_url: str,
) -> None:
    """
    MCP ``read_file`` tool returns real file contents through
    the full workflow pipeline.

    The LLM is mocked to call ``read_file`` on a known test
    file. The function_call_output should contain the exact
    content written to the file by the fixture.
    """
    await _create_mcp_agent(client, mcp_server_url)

    # LLM call 1: return a tool call to read_file
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_mcp_read_1",
                "name": "read_file",
                "arguments": json.dumps({"path": "hello.txt"}),
            },
        ],
    )
    # LLM call 2: after tool result, return final text
    mock_llm.add_call(text="Read the file.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="Read hello.txt",
        background=True,
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    items = await _get_items(client, conv_id)

    # 4 items: user, function_call, function_call_output, assistant
    assert len(items) == 4, (
        f"Expected 4 items [user, fc, fco, assistant], got {len(items)}: "
        f"{[i['type'] for i in items]}"
    )

    # function_call_output must contain the EXACT file content written
    # by the fixture. If it doesn't match, the MCP server read the
    # wrong file or the content was corrupted in transit.
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_mcp_read_1"
    assert items[2]["output"] == "Hello from MCP!", (
        f"Expected exact file content, got: {items[2]['output']!r}"
    )


async def test_mcp_multi_tool_round_trip(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    mcp_server_url: str,
) -> None:
    """
    Two sequential MCP tool calls (list_directory then read_file)
    both produce real results in the correct conversation order.
    """
    await _create_mcp_agent(client, mcp_server_url)

    # Round 1: list_directory
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_mcp_multi_1",
                "name": "list_directory",
                "arguments": json.dumps({"path": "."}),
            },
        ],
    )
    # Round 2: read_file (after seeing list_directory results)
    mock_llm.add_call(
        tool_calls=[
            {
                "call_id": "call_mcp_multi_2",
                "name": "read_file",
                "arguments": json.dumps({"path": "subdir/nested.txt"}),
            },
        ],
    )
    # Round 3: final text
    mock_llm.add_call(text="Done with both tools.")

    result = await create_test_response(
        client,
        model=_AGENT_NAME,
        input_text="List then read",
        background=True,
    )
    assert result.status_code == 200
    response_id = result.body["id"]
    conv_id = result.body["conversation"]["id"]

    body = await _wait_for_completion(client, response_id)
    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    items = await _get_items(client, conv_id)

    # 6 items: user, fc1, fco1, fc2, fco2, assistant.
    # If fewer, one of the tool calls was skipped; if more, extra rounds fired.
    assert len(items) == 6, f"Expected 6 items, got {len(items)}: {[i['type'] for i in items]}"

    # Round 1: list_directory — real directory listing from the MCP server.
    # "hello.txt" proves the list_directory tool executed against the fixture dir.
    assert items[1]["type"] == "function_call"
    assert items[1]["name"] == "list_directory"
    assert items[2]["type"] == "function_call_output"
    assert "hello.txt" in items[2]["output"], (
        f"Expected 'hello.txt' in list_directory output: {items[2]['output']!r}"
    )

    # Round 2: read_file — exact content of the nested file.
    # "Nested content." is the exact text written by the mcp_work_dir fixture.
    assert items[3]["type"] == "function_call"
    assert items[3]["name"] == "read_file"
    assert items[4]["type"] == "function_call_output"
    assert items[4]["output"] == "Nested content.", (
        f"Expected 'Nested content.', got: {items[4]['output']!r}"
    )

    # Final text — proves the third LLM call completed and was persisted
    assert items[5]["type"] == "message"
    assert items[5]["role"] == "assistant"
    assert items[5]["content"][0]["text"] == "Done with both tools."
