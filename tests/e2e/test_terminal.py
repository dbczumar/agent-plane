"""End-to-end tests for the persistent-terminal builtins.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_terminal.py --llm-api-key $LLM_API_KEY -v

Exercises the three terminal tools through a real LLM on a dedicated
test agent (examples/agents/terminal_test/) that has only these
builtins — so the LLM cannot pick a different shell-family tool and
leave ambiguity in the test.

Primary property under test: **shell state persists across turns**.
This is the defining difference between the new terminal tool and
the retiring code_sandbox, and the whole point of Phase 1 of the
persistent-terminal rollout.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _get_output_items(
    body: dict[str, Any],
    item_type: str,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter response.output by item type and optional tool name.

    :param body: The response body from GET /v1/responses/{id}.
    :param item_type: Item type to filter, e.g. ``"function_call"``.
    :param name: Optional tool name filter, e.g. ``"terminal_run"``.
    :returns: Matching items in original order.
    """
    items = body.get("output", [])
    filtered = [i for i in items if i.get("type") == item_type]
    if name is not None:
        filtered = [i for i in filtered if i.get("name") == name]
    return filtered


def test_cwd_persists_across_turns(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """`cd` in turn 1 persists to turn 2 — the defining property.

    Turn 1: ``cd /tmp``.
    Turn 2: ``pwd``. Must report ``/tmp``.

    If this fails, shells are being recreated per turn (bug in
    registry/manager scoping) or the tool is routing to different
    managers. Breakage-mode clear: output of ``pwd`` would be the
    workspace root, not ``/tmp``.
    """
    # Turn 1: cd into a known dir. Prompt the LLM explicitly by
    # tool name — soft prompts like "run X" occasionally result in
    # the model describing the action instead of invoking the tool.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Call the terminal_run tool with command='cd /tmp'. "
                "Do not describe — actually invoke the tool."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    response_id = resp1.json()["id"]
    final1 = poll_until_terminal(http_client, response_id, timeout=120)
    assert final1["status"] == "completed", f"Turn 1 failed: {final1.get('error')}"

    # Verify terminal_run was the tool used.
    fc_items = _get_output_items(final1, "function_call", "terminal_run")
    assert len(fc_items) >= 1, (
        f"Expected terminal_run call in turn 1, got tools: "
        f"{[i.get('name') for i in _get_output_items(final1, 'function_call')]}; "
        f"full output: {final1.get('output')!r}"
    )

    # Turn 2: pwd — state from turn 1 must persist.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": "Now run `pwd`. Tell me only what directory you're in.",
            "previous_response_id": response_id,
            "background": True,
        },
    )
    resp2.raise_for_status()
    response_id_2 = resp2.json()["id"]
    final2 = poll_until_terminal(http_client, response_id_2, timeout=120)
    assert final2["status"] == "completed", f"Turn 2 failed: {final2.get('error')}"

    # Verify turn 2 actually invoked terminal_run (not just answered
    # from memory). If the LLM skipped the tool call, "all_outputs"
    # below would be empty and the /tmp assertion would misleadingly
    # fail as a "persistence" failure rather than a tool-invocation
    # failure.
    fc_items_2 = _get_output_items(final2, "function_call", "terminal_run")
    assert len(fc_items_2) >= 1, (
        f"Expected terminal_run call in turn 2, got: "
        f"{[i.get('name') for i in _get_output_items(final2, 'function_call')]}"
    )
    # And the tool's output contains /tmp — proves pwd saw the cwd
    # set in turn 1's cd, which only works if shells persist.
    fco_items_2 = _get_output_items(final2, "function_call_output")
    all_outputs = " ".join(i.get("output") or "" for i in fco_items_2)
    assert "/tmp" in all_outputs, (
        f"Expected /tmp in terminal_run output (cwd should have "
        f"persisted from turn 1), got: {all_outputs[:300]}"
    )


def test_env_var_persists_across_turns(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """`export` in turn 1 persists to turn 2 (same axis as cwd).

    A second proof of shell-state persistence via a different
    mechanism — if cwd breaks but env works (or vice versa), we'd
    know to look at a specific subsystem.
    """
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": "Run `export AP_E2E_TOKEN=banana` using the terminal.",
            "background": True,
        },
    )
    resp1.raise_for_status()
    response_id = resp1.json()["id"]
    final1 = poll_until_terminal(http_client, response_id, timeout=120)
    assert final1["status"] == "completed", f"Turn 1 failed: {final1.get('error')}"

    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": "Now run `echo $AP_E2E_TOKEN`. Tell me what it prints.",
            "previous_response_id": response_id,
            "background": True,
        },
    )
    resp2.raise_for_status()
    response_id_2 = resp2.json()["id"]
    final2 = poll_until_terminal(http_client, response_id_2, timeout=120)
    assert final2["status"] == "completed", f"Turn 2 failed: {final2.get('error')}"

    # Same belt-and-suspenders check as in the cwd test: verify
    # turn 2 actually called terminal_run, so a missing 'banana' is
    # interpretable as a persistence failure rather than a
    # tool-invocation failure.
    fc_items_2 = _get_output_items(final2, "function_call", "terminal_run")
    assert len(fc_items_2) >= 1, (
        f"Expected terminal_run call in turn 2, got: "
        f"{[i.get('name') for i in _get_output_items(final2, 'function_call')]}"
    )
    fco_items = _get_output_items(final2, "function_call_output")
    all_outputs = " ".join(i.get("output") or "" for i in fco_items)
    assert "banana" in all_outputs, (
        f"Expected 'banana' in terminal_run output (env var should "
        f"have persisted from turn 1), got: {all_outputs[:300]}"
    )
