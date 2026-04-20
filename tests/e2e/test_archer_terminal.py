"""End-to-end test for terminal_run package installation and isolation.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_archer_terminal.py \\
        --llm-api-key $LLM_API_KEY -v

Exercises:
- terminal_run tool executing pip and npm install commands
- Installed packages usable in subsequent terminal_run calls
- Package directories isolated to the conversation's workspace
- Sandbox blocks reads of files outside the workspace
"""

from __future__ import annotations

import os
import tempfile

import httpx

from tests.e2e.conftest import poll_until_terminal


def _get_output_items(
    body: dict,
    item_type: str,
    name: str | None = None,
) -> list[dict]:
    """
    Filter output items by type and optional tool name.

    :param body: The response body from GET /v1/responses/{id}.
    :param item_type: Item type to filter, e.g. ``"function_call"``.
    :param name: Optional tool name filter.
    :returns: Matching items.
    """
    items = body.get("output", [])
    filtered = [i for i in items if i.get("type") == item_type]
    if name is not None:
        filtered = [i for i in filtered if i.get("name") == name]
    return filtered


def test_pip_install_and_use(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Ask the archer agent to pip install a package and use it.
    The package should install into the workspace and be importable
    in a subsequent terminal_run call within the same conversation.

    **What breaks if wrong**:
    - srt blocks pypi.org: pip install fails
    - PIP_TARGET not set: installs to system (or fails in sandbox)
    - PYTHONPATH not set: import fails despite install
    - Workspace not persistent: package gone on next turn
    - terminal_run shell not persistent across turns
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "Use the terminal_run tool to run this exact command: pip install cowsay",
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=120)
    assert final["status"] == "completed", f"Turn 1 failed: {final.get('error')}"

    fc_items = _get_output_items(final, "function_call", "terminal_run")
    assert len(fc_items) >= 1, (
        f"Expected terminal_run call, got: "
        f"{[i.get('name') for i in _get_output_items(final, 'function_call')]}"
    )

    fco_items = _get_output_items(final, "function_call_output")
    install_outputs = [i for i in fco_items if i.get("call_id") == fc_items[0].get("call_id")]
    assert len(install_outputs) == 1
    install_output = install_outputs[0].get("output", "")
    assert "Error" not in install_output or "Successfully installed" in install_output, (
        f"pip install may have failed: {install_output[:200]}"
    )

    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use terminal_run to run this exact command: "
                "python -c \"import cowsay; print(cowsay.get_output_string('cow', 'e2e test'))\""
            ),
            "previous_response_id": response_id,
            "background": True,
        },
    )
    resp2.raise_for_status()
    response_id_2 = resp2.json()["id"]
    final2 = poll_until_terminal(http_client, response_id_2, timeout=120)
    assert final2["status"] == "completed", f"Turn 2 failed: {final2.get('error')}"

    fco_items_2 = _get_output_items(final2, "function_call_output")
    has_cow_output = any("e2e test" in (i.get("output") or "") for i in fco_items_2)
    assert has_cow_output, (
        f"Expected cowsay output with 'e2e test' in turn 2, "
        f"got: {[i.get('output', '')[:100] for i in fco_items_2]}"
    )


def test_npm_install_and_use(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Ask the archer agent to npm install a package and use it.

    **What breaks if wrong**:
    - srt blocks registry.npmjs.org: npm install fails
    - npm_config_prefix not set: installs globally (fails in sandbox)
    - NODE_PATH not set: require() fails
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use the terminal_run tool to run these two commands "
                "in a single invocation:\n"
                "npm install cowsay && "
                "npx cowsay 'npm works'"
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=120)
    assert final["status"] == "completed", f"npm test failed: {final.get('error')}"

    fco_items = _get_output_items(final, "function_call_output")
    has_npm_output = any("npm works" in (i.get("output") or "") for i in fco_items)
    assert has_npm_output, (
        f"Expected cowsay output with 'npm works', "
        f"got: {[i.get('output', '')[:100] for i in fco_items]}"
    )


def test_packages_isolated_across_conversations(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    Packages installed in one conversation are NOT visible in a
    different conversation. Each conversation gets its own workspace.

    **What breaks if wrong**:
    - PIP_TARGET points to a shared directory
    - Workspace not scoped to conversation
    """
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": "Use terminal_run to run: pip install cowsay",
            "background": True,
        },
    )
    resp1.raise_for_status()
    final1 = poll_until_terminal(
        http_client,
        resp1.json()["id"],
        timeout=120,
    )
    assert final1["status"] == "completed"

    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use terminal_run to run this exact command: "
                'python -c "import cowsay" '
                "and tell me if it succeeded or failed"
            ),
            "background": True,
        },
    )
    resp2.raise_for_status()
    final2 = poll_until_terminal(
        http_client,
        resp2.json()["id"],
        timeout=120,
    )
    assert final2["status"] == "completed"

    fco_items = _get_output_items(final2, "function_call_output")
    has_import_error = any(
        "ModuleNotFoundError" in (i.get("output") or "")
        or "No module named" in (i.get("output") or "")
        for i in fco_items
    )
    assert has_import_error, (
        f"Expected ModuleNotFoundError in conversation 2 (isolation), "
        f"got: {[i.get('output', '')[:100] for i in fco_items]}"
    )


def test_sandbox_blocks_reads_outside_workspace(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    The sandbox must block reading files outside the workspace.

    Creates a sentinel file in /tmp with unique content, then asks
    the agent to ``cat`` it. The sandbox's srt filesystem policy
    should deny the read — the sentinel content must NOT appear in
    the tool output.

    **What breaks if wrong**:
    - denyRead is empty: srt allows all reads, sentinel leaks
    - allowRead too broad: reads outside workspace succeed
    """
    sentinel = f"SANDBOX_LEAK_{os.getpid()}"
    fd, sentinel_path = tempfile.mkstemp(
        prefix="ap_sandbox_read_test_",
        dir="/tmp",
    )
    try:
        os.write(fd, sentinel.encode())
        os.close(fd)

        resp = http_client.post(
            "/v1/responses",
            json={
                "model": archer_agent,
                "input": (
                    f"Use the terminal_run tool to run this exact command: cat {sentinel_path}"
                ),
                "background": True,
            },
        )
        resp.raise_for_status()
        response_id = resp.json()["id"]
        final = poll_until_terminal(
            http_client,
            response_id,
            timeout=120,
        )
        assert final["status"] == "completed", f"Task failed: {final.get('error')}"

        fco_items = _get_output_items(final, "function_call_output")
        all_tool_output = " ".join(i.get("output", "") for i in fco_items)

        assert sentinel not in all_tool_output, (
            f"SECURITY: sandbox allowed reading {sentinel_path} "
            f"outside workspace. Tool output contained the "
            f"sentinel '{sentinel}'. The srt denyRead policy is "
            f"not blocking reads outside the workspace."
        )
    finally:
        try:
            os.unlink(sentinel_path)
        except OSError:
            pass


def test_sandbox_allows_network_access(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    The sandbox must allow unrestricted network access.

    Asks the agent to ``curl`` example.com and checks that the
    HTTP status code appears in the output — proving the request
    reached the server and got a response.

    **What breaks if wrong**:
    - srt network proxy active: curl fails with connection error
    - allowedDomains restrictive: only package registries work
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Use the terminal_run tool to run this exact "
                "command: curl -sk https://example.com -o /dev/null "
                "-w '%{http_code}' --connect-timeout 10"
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(
        http_client,
        response_id,
        timeout=120,
    )
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    fco_items = _get_output_items(final, "function_call_output")
    all_tool_output = " ".join(i.get("output", "") for i in fco_items)
    assert "200" in all_tool_output, (
        f"Expected HTTP 200 from example.com proving network "
        f"access works inside sandbox. "
        f"Got: {all_tool_output[:200]}. "
        f"The srt network proxy may still be active."
    )
