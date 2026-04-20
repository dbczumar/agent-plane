"""Implementation of the ``ap chat`` command.

Opens the REPL to chat with an agent. Supports two modes:

- **Local mode** (target is a path): starts a temporary server with
  the agent pre-registered, then opens the REPL connected to it.
- **Remote mode** (target is a URL): connects to an existing server
  and opens the REPL. The user picks which agent to talk to.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
import httpx
import yaml
from agent_plane_client import AgentPlaneClient, ToolCallInfo, ToolHandler


def run_chat(target: str, client_tools: str | None) -> None:
    """
    Main entry point for ``ap chat``.

    :param target: Path to an agent directory/bundle, or a server URL.
    :param client_tools: Optional client-side tool set name.
    """
    # Client-side tools are a CLI/TUI convenience (e.g. shell access
    # for coding agents). They don't affect agent behavior — the spec
    # is self-contained.
    tool_handler = _load_tool_handler(client_tools) if client_tools else None

    if _is_url(target):
        _chat_remote(target, tool_handler)
    else:
        _chat_local(target, tool_handler)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def _is_url(target: str) -> bool:
    """
    Check if the target looks like a URL.

    :param target: The target string.
    :returns: True if it starts with http:// or https://.
    """
    return target.startswith("http://") or target.startswith("https://")


# ---------------------------------------------------------------------------
# Remote mode
# ---------------------------------------------------------------------------


def _chat_remote(server_url: str, tool_handler: ToolHandler | None) -> None:
    """
    Connect to a remote server and open the REPL.

    Lists available agents and lets the user pick one.

    :param server_url: The server URL.
    :param tool_handler: Optional client-side tool handler.
    """
    base_url = server_url.rstrip("/")
    agent_name = _pick_agent(base_url)

    click.echo(f"\n  Chatting with [bold]{agent_name}[/bold] on {base_url}\n")

    _run_repl(base_url, agent_name, tool_handler)


def _pick_agent(base_url: str) -> str:
    """
    List agents on the server and let the user pick one.

    If only one agent is available, selects it automatically.

    :param base_url: Server base URL.
    :returns: The chosen agent name.
    """
    resp = httpx.get(f"{base_url}/api/agents", timeout=10.0)
    resp.raise_for_status()
    agents = resp.json()["data"]

    if not agents:
        raise click.ClickException("No agents found on the server.")

    if len(agents) == 1:
        name = str(agents[0]["name"])
        click.echo(f"\n  Agent: {name}")
        return name

    click.echo("\n  Available agents:\n")
    for i, agent in enumerate(agents, 1):
        click.echo(f"    {i}. {agent['name']}")

    while True:
        raw = str(click.prompt("\n  Agent", default="1"))
        try:
            choice = int(raw)
            if 1 <= choice <= len(agents):
                return str(agents[choice - 1]["name"])
        except ValueError:
            # Try matching by name.
            for agent in agents:
                if agent["name"] == raw.strip():
                    return str(agent["name"])
        click.echo(f"  Enter a number between 1 and {len(agents)}.")


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------


def _chat_local(agent_path: str, tool_handler: ToolHandler | None) -> None:
    """
    Start a local server with the agent and open the REPL.

    :param agent_path: Path to the agent directory or bundle.
    :param tool_handler: Optional client-side tool handler.
    """
    path = Path(agent_path)
    if not path.exists():
        raise click.ClickException(f"Agent path not found: {agent_path}")

    agent_name = _extract_agent_name(path)
    port = _find_free_port()
    server_proc = _start_local_server(path, port)

    try:
        _wait_for_server(port, server_proc)
        click.echo(f"  Agent: {agent_name}\n")
        _run_repl(f"http://127.0.0.1:{port}", agent_name, tool_handler)
    finally:
        _stop_server(server_proc)


def _extract_agent_name(agent_path: Path) -> str:
    """
    Read the agent name from config.yaml for display in the REPL.

    Falls back to the directory name when the spec omits ``name``
    (which is optional per AGENTSPEC.md for root agents). This is
    only used as a display label — the server performs full spec
    validation independently.

    :param agent_path: Path to the agent directory.
    :returns: The agent name for REPL display.
    """
    config_path = agent_path / "config.yaml"
    if not config_path.exists():
        # No config yet — the server will fail with a clear error;
        # use the directory name as a placeholder for the startup message.
        return agent_path.name

    config = yaml.safe_load(config_path.read_text())
    # ``name`` is optional in AGENTSPEC.md — directory name is the
    # standard fallback for display purposes.
    return config.get("name") or agent_path.name


def _find_free_port() -> int:
    """
    Find a free TCP port.

    :returns: An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_local_server(agent_path: Path, port: int) -> subprocess.Popen[bytes]:
    """
    Launch a temporary agent plane server.

    :param agent_path: Path to the agent directory.
    :param port: Port to listen on.
    :returns: The server subprocess.
    """
    tmpdir = tempfile.mkdtemp(prefix="ap-chat-")
    click.echo(f"\n  Starting server on port {port}...")

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_plane.cli",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-uri",
            f"sqlite:///{tmpdir}/chat.db",
            "--artifact-location",
            f"{tmpdir}/artifacts",
            "--agent",
            str(agent_path),
        ],
        # Inherit env so the spec parser can expand ${VAR} references
        # in connection blocks (e.g. ${OPENAI_API_KEY}). The subprocess
        # resolves these at spec-parse time, not at runtime.
        env={**os.environ},
        # Discard stdout/stderr. A PIPE here would deadlock once the
        # kernel's ~64 KB pipe buffer fills (e.g. under the Codex MCP
        # notification firehose) because nothing drains it.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_server(
    port: int, server_proc: subprocess.Popen[bytes], timeout: float = 15.0
) -> None:
    """
    Poll until the server responds.

    :param port: The server port.
    :param server_proc: The server subprocess, used to detect early exit.
    :param timeout: Max seconds to wait.
    :raises click.ClickException: If the server doesn't start.
    """
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_proc.poll() is not None:
            _raise_server_failed(server_proc)
        try:
            resp = httpx.get(f"{base_url}/v1/conversations", timeout=2.0)
            if resp.status_code in (200, 404):
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    _raise_server_failed(server_proc)


def _raise_server_failed(server_proc: subprocess.Popen[bytes]) -> None:
    """
    Raise a descriptive error for a failed server startup.

    :param server_proc: The server subprocess.
    :raises click.ClickException: Always.
    """
    raise click.ClickException(
        "Server failed to start. Re-run the agent_plane server directly to see its output:\n"
        f"  {' '.join(server_proc.args) if isinstance(server_proc.args, list) else server_proc.args}"
    )


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    """
    Gracefully stop the server subprocess.

    :param proc: The server subprocess.
    """
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# REPL launch
# ---------------------------------------------------------------------------


def _run_repl(
    base_url: str,
    agent_name: str,
    tool_handler: ToolHandler | None,
) -> None:
    """
    Open the REPL connected to the server.

    :param base_url: Server base URL.
    :param agent_name: Agent name to chat with.
    :param tool_handler: Optional client-side tool handler.
    """
    from agent_plane.repl import run_repl

    async def _main() -> None:
        async with AgentPlaneClient(base_url=base_url) as client:
            await run_repl(client, agent_name, tool_handler)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Client-side tools
# ---------------------------------------------------------------------------


def _load_tool_handler(name: str) -> ToolHandler:
    """
    Load a client-side tool set by name and wrap it as a ToolHandler.

    Prefers the modern ``@tool``-decorated functions (exposed
    by the tool set as ``_TOOL_FNS``) so the SDK's D6 lifecycle
    can detect ``synchronous: false`` properties on the wire
    schema. Falls back to the legacy ``TOOLS`` + ``execute_tool``
    surface for tool sets that haven't migrated yet — same
    behavior as before, just constructed manually rather than
    via ``build_tool_handler``.

    :param name: Tool set name, e.g. ``"coding"``.
    :returns: A ToolHandler with schemas and execute function.
    :raises click.ClickException: If the tool set is not found.
    """
    try:
        from agent_plane.client_tools import get_tool_set

        tool_set = get_tool_set(name)
    except (ImportError, SystemExit):
        raise click.ClickException(f"Tool set {name!r} not found. Available: coding, async_demo")

    fns = getattr(tool_set, "_TOOL_FNS", None)
    if fns is not None:
        # Modern path: @tool-decorated functions. The SDK's
        # build_tool_handler derives schemas from type hints +
        # docstrings, strips ``synchronous`` routing hints
        # before invoking the user fn, and bridges sync vs
        # async ``execute`` correctly.
        from agent_plane_client.tools import build_tool_handler

        return build_tool_handler(fns)

    # Legacy path: hand-written TOOLS dict + sync
    # execute_tool dispatcher.
    def execute(call: ToolCallInfo) -> str:
        """
        Execute a client-side tool call (legacy sync path).

        :param call: The tool call info with name and arguments.
        :returns: The tool result string.
        """
        return str(tool_set.execute_tool(call.name, call.arguments))

    return ToolHandler(schemas=tool_set.TOOLS, execute=execute)
