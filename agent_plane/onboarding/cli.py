"""
Implementation of the ``ap create`` command.

Handles interactive provider selection, prepares the onboarding
assistant with the chosen model/credentials, boots a temporary
server, and launches the terminal frontend for the onboarding session.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import click
import httpx
import yaml

if TYPE_CHECKING:
    from rich.console import Console

from agent_plane.onboarding.provider_selection import (
    ProviderSelection,
    resolve_provider_from_model,
    select_provider_interactive,
)


def run_create(
    message: str | None,
    model: str | None,
    allow_shell_access: bool,
) -> None:
    """
    Main entry point for ``ap create``.

    Orchestrates provider selection, onboarding assistant preparation,
    server startup, and frontend launch.

    :param message: Optional message for non-interactive mode. When
        provided, the onboarding assistant runs with this as its
        initial prompt and exits when done.
    :param model: Optional model in litellm format
        (``provider/model_name``). Required for non-interactive mode.
        Skips provider selection in interactive mode.
    :param allow_shell_access: Whether to give the onboarding
        assistant full shell access.
    """
    if message is None:
        allow_shell_access = _show_welcome_and_prompt(allow_shell_access)

    selection = _resolve_selection(message, model)

    from rich.console import Console

    Console().print(f"\n  [bold]Onboarding model:[/bold] {selection.model}")

    agent_dir = _prepare_onboarding_agent(selection, allow_shell_access)
    try:
        _run_with_server(agent_dir, message, allow_shell_access)
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)


def _show_welcome_and_prompt(allow_shell_access: bool) -> bool:
    """
    Show the welcome banner and prompt for shell access if needed.

    :param allow_shell_access: Whether shell access was already
        granted via CLI flag.
    :returns: The resolved shell access setting.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print()
    console.print(
        Panel(
            "[bold]Agent Plane Onboarding[/bold]\n\n"
            "You're about to launch an [cyan]onboarding assistant[/cyan] — "
            "an AI agent that will help you create a new agent plane agent "
            "through an interactive conversation.\n\n"
            "First, choose an access mode, then pick a model provider "
            "and model to power the assistant.",
            border_style="blue",
            padding=(1, 2),
        )
    )

    if not allow_shell_access:
        return _prompt_shell_access(console)
    return allow_shell_access


def _prompt_shell_access(console: Console) -> bool:
    """
    Ask the user whether to enable full shell access.

    :param console: Rich console for styled output.
    :returns: ``True`` if the user grants shell access.
    """
    console.print()
    console.print(
        "  [bold]• Sandbox mode[/bold] (default) — restricted to an "
        "isolated workspace. Creates the agent in a sandbox, then "
        "exports it to your chosen path.\n"
    )
    console.print(
        "  [bold]• Shell access[/bold] — full access to your filesystem, "
        "shell commands, and network. Can read your existing code and "
        "write the generated agent directly to disk."
    )
    answer = click.prompt(
        "\n  Allow shell access? (y/n)",
        default="n",
        show_default=True,
    )
    return answer.strip().lower() in ("y", "yes")


def _resolve_selection(
    message: str | None,
    model: str | None,
) -> ProviderSelection:
    """
    Determine provider selection from CLI arguments.

    :param message: Non-interactive message, or ``None`` for interactive.
    :param model: Model flag value, or ``None``.
    :returns: The resolved :class:`ProviderSelection`.
    :raises click.ClickException: If non-interactive mode lacks ``--model``.
    """
    if message is not None and model is None:
        raise click.ClickException(
            "Non-interactive mode requires --model. "
            'Example: ap create "build a research agent" '
            "--model anthropic/claude-sonnet-4-20250514"
        )

    if model is not None:
        return resolve_provider_from_model(model)
    return select_provider_interactive()


def _run_with_server(
    agent_dir: str,
    message: str | None,
    allow_shell_access: bool,
) -> None:
    """
    Boot a temp server, run the onboarding session, then shut down.

    :param agent_dir: Path to the prepared onboarding assistant directory.
    :param message: Non-interactive message, or ``None`` for interactive.
    :param allow_shell_access: Whether to enable full shell tools.
    """
    port = _find_free_port()
    server_proc = _start_server(agent_dir, port)
    try:
        _wait_for_server(server_proc, port)
        agent_id = _get_agent_id(port, "onboarding-buddy")
        if message is not None:
            _run_non_interactive(port, agent_id, message, allow_shell_access)
        else:
            _run_interactive(port, allow_shell_access)
    finally:
        _stop_server(server_proc)


# ---------------------------------------------------------------------------
# Agent preparation
# ---------------------------------------------------------------------------


def _prepare_onboarding_agent(
    selection: ProviderSelection,
    allow_shell_access: bool,
) -> str:
    """
    Copy the built-in onboarding assistant to a temp dir with resolved config.

    Replaces the placeholder model/key in config.yaml with the user's
    selection. When shell access is disabled, adds ``code_sandbox``
    and ``export_agent`` to the agent's builtin tools so it can work
    in a sandbox and export the result.

    :param selection: The provider/model/credentials chosen by the user.
    :param allow_shell_access: Whether full shell access is enabled.
    :returns: Path to the temporary agent directory.
    """
    source = Path(__file__).parent / "agent"
    tmpdir = tempfile.mkdtemp(prefix="ap-onboarding-")
    dest = Path(tmpdir) / "agent"
    shutil.copytree(str(source), str(dest))

    _rewrite_agent_config(dest, selection, allow_shell_access)
    return str(dest)


def _rewrite_agent_config(
    dest: Path,
    selection: ProviderSelection,
    allow_shell_access: bool,
) -> None:
    """
    Rewrite config.yaml with the user's model, credentials, and tools.

    :param dest: Path to the copied agent directory.
    :param selection: The provider/model/credentials chosen by the user.
    :param allow_shell_access: Whether full shell access is enabled.
        When False, adds code_sandbox and export_agent to builtins.
    """
    config_path = dest / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["llm"]["model"] = selection.model
    config["llm"]["connection"] = dict(selection.credentials)

    if not allow_shell_access:
        # Add sandbox tools so the assistant can create files in the
        # sandbox and export the finished agent to the user's path.
        builtins = config.get("tools", {}).get("builtins", [])
        builtins.extend(["code_sandbox", "export_agent"])
        config.setdefault("tools", {})["builtins"] = builtins

    config_path.write_text(yaml.dump(config, default_flow_style=False))


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """
    Find a free TCP port by binding to port 0.

    :returns: An available port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_server(agent_dir: str, port: int) -> subprocess.Popen[bytes]:
    """
    Launch a temporary agent plane server with the onboarding assistant.

    :param agent_dir: Path to the prepared onboarding assistant directory.
    :param port: Port to listen on.
    :returns: The server subprocess.
    """
    tmpdir = tempfile.mkdtemp(prefix="ap-create-db-")
    db_uri = f"sqlite:///{tmpdir}/onboarding.db"
    art_loc = f"{tmpdir}/artifacts"

    click.echo("\nLaunching onboarding assistant...")

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
            db_uri,
            "--artifact-location",
            art_loc,
            "--agent",
            agent_dir,
        ],
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _wait_for_server(
    proc: subprocess.Popen[bytes],
    port: int,
    timeout: float = 15.0,
) -> None:
    """
    Poll until the server responds or timeout.

    :param proc: The server subprocess.
    :param port: The server port.
    :param timeout: Max seconds to wait.
    :raises click.ClickException: If the server fails to start.
    """
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise click.ClickException(
                f"Server exited with code {proc.returncode}.\n{out[-3000:]}"
            )
        try:
            resp = httpx.get(f"{base_url}/v1/conversations", timeout=2.0)
            if resp.status_code in (200, 404):
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise click.ClickException(f"Server did not start within {timeout}s")


def _get_agent_id(port: int, agent_name: str) -> str:
    """
    Look up the agent ID by name from the running server.

    :param port: The server port.
    :param agent_name: The agent name to find, e.g. ``"onboarding-buddy"``.
    :returns: The agent's ID string.
    :raises click.ClickException: If the agent is not found.
    """
    base_url = f"http://127.0.0.1:{port}"
    resp = httpx.get(f"{base_url}/api/agents", timeout=10.0)
    resp.raise_for_status()
    for agent in resp.json()["data"]:
        if agent["name"] == agent_name:
            return str(agent["id"])
    raise click.ClickException(
        f"Onboarding assistant '{agent_name}' not found after server startup."
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
# Frontend launch
# ---------------------------------------------------------------------------


def _run_interactive(
    port: int,
    allow_shell_access: bool,
) -> None:
    """
    Launch the terminal TUI for an interactive onboarding session.

    :param port: The server port.
    :param allow_shell_access: Whether to pass ``--tools coding``
        for full shell access.
    """
    import asyncio

    from agent_plane_client import AgentPlaneClient

    from agent_plane.repl import run_repl

    tool_handler = None
    if allow_shell_access:
        tool_handler = _load_coding_tool_handler()

    async def _run() -> None:
        async with AgentPlaneClient(base_url=f"http://127.0.0.1:{port}") as client:
            await run_repl(
                client,
                "onboarding-buddy",
                tool_handler,
                initial_message="Introduce yourself and help me get started.",
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def _run_non_interactive(
    port: int,
    agent_id: str,
    message: str,
    allow_shell_access: bool,
) -> None:
    """
    Run the onboarding assistant non-interactively with a single message.

    Sends the message, prints text output, handles tool calls if
    shell access is enabled, and loops until done.

    :param port: The server port.
    :param agent_id: The onboarding assistant's ID.
    :param message: The user's message.
    :param allow_shell_access: Whether to enable full shell tools.
    """
    tool_set = _load_coding_tools() if allow_shell_access else None
    _non_interactive_loop(
        port=port,
        tool_set=tool_set,
        initial_input=message,
    )


def _non_interactive_loop(
    port: int,
    tool_set: ModuleType | None,
    initial_input: str | list[dict[str, Any]],
) -> None:
    """
    Send messages and handle tool calls until the assistant is done.

    :param port: The server port.
    :param tool_set: Loaded coder tool set module, or ``None``.
    :param initial_input: The first input — a plain string for the
        initial user message, or a list of function_call_output dicts
        for tool result continuations.
    """
    base_url = f"http://127.0.0.1:{port}"
    tools = tool_set.TOOLS if tool_set is not None else []
    previous_response_id: str | None = None
    pending_input: str | list[dict[str, Any]] = initial_input

    while True:
        data = _send_request(base_url, pending_input, tools, previous_response_id)
        previous_response_id = data["id"]
        _print_text_output(data)
        tool_calls = _extract_tool_calls(data)

        if not tool_calls or tool_set is None:
            break
        pending_input = _execute_tool_calls(tool_calls, tool_set)


def _send_request(
    base_url: str,
    pending_input: str | list[dict[str, Any]],
    tools: list[dict[str, Any]],
    previous_response_id: str | None,
) -> dict[str, Any]:
    """
    Send a request to the agent plane server.

    :param base_url: Server base URL.
    :param pending_input: Input — a plain string for the first message,
        or a list of function_call_output dicts for continuations.
    :param tools: Tool schemas to include, or empty list.
    :param previous_response_id: Previous response ID for continuation.
    :returns: Parsed JSON response dict.
    """
    body: dict[str, str | bool | list[dict[str, Any]]] = {
        "model": "onboarding-buddy",
        "input": pending_input,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    if previous_response_id:
        body["previous_response_id"] = previous_response_id

    resp = httpx.post(f"{base_url}/v1/responses", json=body, timeout=300.0)
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def _print_text_output(data: dict[str, Any]) -> None:
    """
    Print text content from a response to stdout.

    :param data: Parsed response JSON.
    """
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    click.echo(content["text"])


def _extract_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract pending function_call items from a response.

    :param data: Parsed response JSON.
    :returns: List of function_call dicts with ``call_id``, ``name``,
        ``arguments``.
    """
    return [item for item in data.get("output", []) if item.get("type") == "function_call"]


def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_set: ModuleType,
) -> list[dict[str, Any]]:
    """
    Execute tool calls locally and return function_call_output items.

    :param tool_calls: List of function_call dicts from the response.
    :param tool_set: The loaded coder tool set module.
    :returns: List of function_call_output dicts to send back.
    """
    outputs: list[dict[str, Any]] = []
    for call in tool_calls:
        args = json.loads(call["arguments"])
        result = tool_set.execute_tool(call["name"], args)
        # 20000 char limit matches the terminal frontend's truncation.
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": str(result)[:20000],
            }
        )
    return outputs


def _load_coding_tool_handler() -> Any:
    """Load the coder tool set as a ToolHandler for client-side tools."""

    from agent_plane_client import ToolCallInfo, ToolHandler

    from agent_plane.client_tools import get_tool_set

    tool_set = get_tool_set("coding")

    def execute(call: ToolCallInfo) -> str:
        """Execute a client-side tool call.

        :param call: Tool call info with name and arguments.
        :returns: The tool result string.
        """
        return str(tool_set.execute_tool(call.name, call.arguments))

    return ToolHandler(schemas=tool_set.TOOLS, execute=execute)


def _load_coding_tools() -> ModuleType | None:
    """
    Load the coder tool set for full shell access.

    :returns: The coder tool set module with ``TOOLS`` and
        ``execute_tool`` attributes, or ``None`` if the tool set
        package is not available (e.g. running from an installed
        package without the examples/ directory).
    """
    try:
        from agent_plane.client_tools import get_tool_set

        return get_tool_set("coding")
    except ImportError:
        return None
