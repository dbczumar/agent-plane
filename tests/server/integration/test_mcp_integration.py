"""Integration tests for MCP tool execution through the full pipeline.

Starts a real MCP filesystem server (stdio transport), mocks only the
LLM to return tool calls targeting MCP-discovered tools, and verifies
the full workflow: discovery → tool invocation → real results in the
conversation history.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from agent_plane.tools.mcp import clear_discovery_cache
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_response

pytestmark = pytest.mark.asyncio

# ── Agent name used to link create_test_agent ↔ create_test_response ──

_AGENT_NAME = "mcp-test-agent"


# ── Bundle builder ────────────────────────────────────────


def _build_mcp_agent_bundle(work_dir: Path) -> bytes:
    """
    Build an agent bundle (tar.gz) that includes a real MCP
    filesystem server.

    The bundle contains:

    - ``config.yaml`` — minimal agent spec with LLM config
    - ``tools/mcp/filesystem.yaml`` — MCP server declaration
    - ``tools/mcp/filesystem_server.py`` — FastMCP server script

    The filesystem server's working directory is ``work_dir``,
    which is passed via the ``MCP_ROOT`` environment variable
    so the server roots its operations there.

    :param work_dir: Absolute path to the directory the MCP
        filesystem server will serve, e.g. ``"/tmp/pytest-xyz/work"``.
    :returns: Raw tar.gz bytes ready for upload.
    """
    config = {
        "spec_version": 1,
        "name": _AGENT_NAME,
        "llm": {"model": _AGENT_NAME},
    }

    mcp_yaml = {
        "name": "filesystem",
        "description": "Read-only filesystem tools for testing.",
        "transport": "stdio",
        "command": "python",
        "args": ["tools/mcp/filesystem_server.py"],
        "env": {"MCP_ROOT": str(work_dir)},
    }

    server_py = _MCP_SERVER_SOURCE

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_text(tf, "config.yaml", yaml.dump(config))
        _add_text(tf, "tools/mcp/filesystem.yaml", yaml.dump(mcp_yaml))
        _add_text(tf, "tools/mcp/filesystem_server.py", server_py)
    return buf.getvalue()


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


# ── Embedded MCP server source ────────────────────────────
# Inlined so the test is self-contained. Uses MCP_ROOT env var
# to control the root directory (set in the MCP YAML config).

_MCP_SERVER_SOURCE = '''\
"""Minimal filesystem MCP server for integration tests."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem")

_ROOT = Path(os.environ.get("MCP_ROOT", os.getcwd())).resolve()


def _safe_resolve(path: str) -> Path:
    """
    Resolve a path and verify it is within the root.

    :param path: Relative or absolute path string.
    :returns: The resolved absolute path.
    :raises ValueError: If the path escapes the root.
    """
    resolved = (_ROOT / path).resolve()
    if not resolved.is_relative_to(_ROOT):
        raise ValueError(f"Path {path!r} is outside the root: {_ROOT}")
    return resolved


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """
    List files and directories at the given path.

    :param path: Relative path from the root directory.
    """
    resolved = _safe_resolve(path)
    if not resolved.is_dir():
        raise ValueError(f"Not a directory: {path}")
    entries: list[str] = []
    for entry in sorted(resolved.iterdir()):
        name = entry.name
        if entry.is_dir():
            name += "/"
        entries.append(name)
    return "\\n".join(entries) if entries else "(empty directory)"


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the contents of a text file.

    :param path: Relative path to the file.
    """
    resolved = _safe_resolve(path)
    if not resolved.is_file():
        raise ValueError(f"Not a file: {path}")
    return resolved.read_text()


if __name__ == "__main__":
    mcp.run()
'''


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
    mcp_work_dir: Path,
) -> dict[str, Any]:
    """
    Upload an agent bundle with a real MCP filesystem server.

    :param client: HTTP client for the test server.
    :param mcp_work_dir: Root directory for the filesystem server.
    :returns: The agent creation response JSON.
    """
    bundle = _build_mcp_agent_bundle(mcp_work_dir)
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201
    return resp.json()


async def _wait_for_completion(
    client: httpx.AsyncClient,
    response_id: str,
) -> dict[str, Any]:
    """
    Poll until a response reaches a terminal status.

    Uses a short sleep between polls because MCP server startup
    adds latency compared to skill-only workflows.

    :param client: HTTP client for the server.
    :param response_id: The response/task ID to poll.
    :returns: The response JSON dict once completed or failed.
    :raises AssertionError: If the response doesn't complete
        within the polling window.
    """
    import asyncio

    for _ in range(100):
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
    :param conv_id: The conversation ID.
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
    mcp_work_dir: Path,
) -> None:
    """
    MCP ``list_directory`` tool returns real directory contents
    through the full workflow pipeline.

    The LLM is mocked to call ``list_directory``, then produce a
    final text response. The function_call_output in the
    conversation should contain the actual file listing from the
    temporary directory.
    """
    await _create_mcp_agent(client, mcp_work_dir)

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
        f"Expected completed, got {body['status']}. "
        f"Error: {body.get('error')}"
    )

    items = await _get_items(client, conv_id)

    # Expected: [user, function_call, function_call_output, assistant]
    assert len(items) == 4, (
        f"Expected 4 items [user, fc, fco, assistant], got {len(items)}: "
        f"{[i['type'] for i in items]}"
    )

    # Verify user message
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"

    # Verify function_call for list_directory
    assert items[1]["type"] == "function_call"
    assert items[1]["name"] == "list_directory"
    assert items[1]["call_id"] == "call_mcp_list_1"

    # Verify function_call_output contains REAL filesystem data
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_mcp_list_1"
    output = items[2]["output"]
    assert "hello.txt" in output, (
        f"Expected 'hello.txt' in MCP output, got: {output!r}"
    )
    assert "subdir/" in output, (
        f"Expected 'subdir/' in MCP output, got: {output!r}"
    )

    # Verify final assistant message
    assert items[3]["type"] == "message"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"][0]["text"] == "Listed the directory."


async def test_mcp_read_file_returns_real_content(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    mcp_work_dir: Path,
) -> None:
    """
    MCP ``read_file`` tool returns real file contents through
    the full workflow pipeline.

    The LLM is mocked to call ``read_file`` on a known test
    file. The function_call_output should contain the exact
    content written to the file by the fixture.
    """
    await _create_mcp_agent(client, mcp_work_dir)

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
        f"Expected completed, got {body['status']}. "
        f"Error: {body.get('error')}"
    )

    items = await _get_items(client, conv_id)

    assert len(items) == 4, (
        f"Expected 4 items [user, fc, fco, assistant], got {len(items)}: "
        f"{[i['type'] for i in items]}"
    )

    # Verify function_call_output contains the exact file content
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_mcp_read_1"
    assert items[2]["output"] == "Hello from MCP!", (
        f"Expected exact file content, got: {items[2]['output']!r}"
    )


async def test_mcp_multi_tool_round_trip(
    client: httpx.AsyncClient,
    mock_llm: ControllableMockClient,
    mcp_work_dir: Path,
) -> None:
    """
    Two sequential MCP tool calls (list_directory then read_file)
    both produce real results in the correct conversation order.
    """
    await _create_mcp_agent(client, mcp_work_dir)

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
        f"Expected completed, got {body['status']}. "
        f"Error: {body.get('error')}"
    )

    items = await _get_items(client, conv_id)

    # Expected: user, fc1, fco1, fc2, fco2, assistant
    assert len(items) == 6, (
        f"Expected 6 items, got {len(items)}: "
        f"{[i['type'] for i in items]}"
    )

    # Round 1: list_directory
    assert items[1]["type"] == "function_call"
    assert items[1]["name"] == "list_directory"
    assert items[2]["type"] == "function_call_output"
    assert "hello.txt" in items[2]["output"]

    # Round 2: read_file
    assert items[3]["type"] == "function_call"
    assert items[3]["name"] == "read_file"
    assert items[4]["type"] == "function_call_output"
    assert items[4]["output"] == "Nested content."

    # Final text
    assert items[5]["type"] == "message"
    assert items[5]["role"] == "assistant"
    assert items[5]["content"][0]["text"] == "Done with both tools."
