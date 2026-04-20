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

import json
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


def _parse_terminal_run_stdout(fco_items: list[dict[str, Any]]) -> list[str]:
    """Extract ``stdout`` fields from a list of function_call_output items.

    ``terminal_run`` returns a JSON string like
    ``{"stdout": "...", "exit_code": 0, "status": "completed",
    "shell": "default"}``. Tests routinely want just the stdout
    strings to substring-match against.

    Fails loud if no output parses — silently returning an empty list
    would let tests pass with vacuous assertions when the tool
    dispatch is actually broken (no outputs arrived, or all outputs
    are from some other tool).

    :param fco_items: Items from ``_get_output_items(body,
        "function_call_output")``.
    :returns: The stdout field from each parseable terminal_run output.
    :raises AssertionError: If no terminal_run outputs were parseable.
    """
    stdouts: list[str] = []
    for i in fco_items:
        raw = i.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and "stdout" in parsed:
            stdouts.append(parsed.get("stdout") or "")
    assert stdouts, (
        f"No terminal_run outputs were parseable. Got "
        f"{len(fco_items)} function_call_output items; none had a "
        f"'stdout' field. Either no terminal_run was invoked, or the "
        f"outputs are all from a different tool. Raw items: "
        f"{[i.get('output', '')[:120] for i in fco_items]}"
    )
    return stdouts


# ── Named multi-shell parallelism ────────────────────────────────


def test_named_shells_are_isolated(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """Two named shells in one conversation have independent state.

    Turn 1: set ``$FLAVOR=vanilla`` in shell ``dev``.
    Turn 2: set ``$FLAVOR=chocolate`` in shell ``test``.
    Turn 3: read ``$FLAVOR`` from each. They must differ.

    If shells are not actually isolated — e.g. the manager hands out
    the same subprocess for different names, or the name routing
    collapses to a single shell — both reads would show the same
    value and this test fails. Covers §6.1's "parallelism" claim
    and the TerminalManager's shell-name map.

    Two separate writes instead of one combined turn because the LLM
    sometimes optimizes "set A=x and B=y" into a single call on a
    single shell; making it two explicit turns forces independent
    tool invocations against the two shells.
    """
    # Turn 1: set in 'dev'.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": ("Use terminal_run with shell='dev' to run: export FLAVOR=vanilla"),
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    final1 = poll_until_terminal(http_client, rid1, timeout=120)
    assert final1["status"] == "completed", f"Turn 1 failed: {final1.get('error')}"

    # Turn 2: set in 'test' — DIFFERENT shell.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": ("Use terminal_run with shell='test' to run: export FLAVOR=chocolate"),
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    final2 = poll_until_terminal(http_client, rid2, timeout=120)
    assert final2["status"] == "completed", f"Turn 2 failed: {final2.get('error')}"

    # Turn 3: read from both shells, back to back.
    resp3 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Make TWO terminal_run calls: first with shell='dev' "
                "and command='echo DEV_FLAVOR=$FLAVOR', then with "
                "shell='test' and command='echo TEST_FLAVOR=$FLAVOR'. "
                "Both calls are required."
            ),
            "previous_response_id": rid2,
            "background": True,
        },
    )
    resp3.raise_for_status()
    rid3 = resp3.json()["id"]
    final3 = poll_until_terminal(http_client, rid3, timeout=120)
    assert final3["status"] == "completed", f"Turn 3 failed: {final3.get('error')}"

    # The LLM should have called terminal_run twice in turn 3 — once
    # per shell. If it collapsed to one call, we can't prove
    # isolation (both shells' state never contrasts).
    fc_items = _get_output_items(final3, "function_call", "terminal_run")
    assert len(fc_items) >= 2, (
        f"Expected 2 terminal_run calls in turn 3 (one per shell), "
        f"got {len(fc_items)}: {[i.get('arguments') for i in fc_items]}"
    )

    fco_items = _get_output_items(final3, "function_call_output")
    stdouts = _parse_terminal_run_stdout(fco_items)
    combined = " ".join(stdouts)
    # The whole point: dev saw vanilla, test saw chocolate. If dev
    # sees chocolate (or test sees vanilla), the shells are not
    # isolated — one shell is being aliased to the other.
    assert "DEV_FLAVOR=vanilla" in combined, (
        f"Expected 'DEV_FLAVOR=vanilla' in outputs — shell 'dev' lost "
        f"its state or got cross-contaminated. stdouts: {stdouts}"
    )
    assert "TEST_FLAVOR=chocolate" in combined, (
        f"Expected 'TEST_FLAVOR=chocolate' in outputs — shell 'test' "
        f"lost its state or got cross-contaminated. stdouts: {stdouts}"
    )


# ── terminal_list + terminal_close round-trip ────────────────────


def test_terminal_list_and_close_round_trip(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """`terminal_list` enumerates shells; `terminal_close` evicts them.

    Turn 1: create two shells ('main' and 'dev') via terminal_run.
    Turn 2: call terminal_list — both names must appear.
    Turn 3: call terminal_close on 'main'.
    Turn 4: call terminal_list — only 'dev' remains.

    This is the **only** e2e that exercises ``terminal_list`` and
    ``terminal_close``. Failure modes it catches:
    - terminal_list returns empty (registry lookup broken for
      conversations that have active shells).
    - terminal_close silently no-ops (shell persists after close).
    - Close-then-list race drops the wrong shell.
    """
    # Turn 1: populate two shells.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Make TWO terminal_run calls: first with shell='main' "
                "command='echo init', then with shell='dev' "
                "command='echo init'. Both required."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    final1 = poll_until_terminal(http_client, rid1, timeout=120)
    assert final1["status"] == "completed"

    # Turn 2: list shells.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Call the terminal_list tool (no arguments). "
                "Then tell me exactly which shells are open."
            ),
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    final2 = poll_until_terminal(http_client, rid2, timeout=120)
    assert final2["status"] == "completed"

    # Confirm the agent used terminal_list and that its output
    # contained both shell names. Parse the terminal_list JSON
    # directly (not substring-match the whole response) so a coincidental
    # "main" or "dev" appearing elsewhere — e.g. in the LLM's own text
    # message or in a stdout field from a stray terminal_run — doesn't
    # satisfy the assertion.
    list_calls = _get_output_items(final2, "function_call", "terminal_list")
    assert len(list_calls) >= 1, "Expected terminal_list to be invoked."
    list_call_ids = {c.get("call_id") for c in list_calls}
    list_json_outputs = [
        i.get("output") or ""
        for i in _get_output_items(final2, "function_call_output")
        if i.get("call_id") in list_call_ids
    ]
    assert list_json_outputs, (
        "No function_call_output items matched the terminal_list call_ids. "
        "The tool call was issued but never got an output — workflow bug."
    )
    shell_sets = []
    for raw in list_json_outputs:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise AssertionError(
                f"terminal_list output is not valid JSON: {raw!r} ({exc})"
            ) from exc
        shells_val = parsed.get("shells")
        assert isinstance(shells_val, list), (
            f"terminal_list output missing 'shells' list: got {parsed!r}"
        )
        shell_sets.append(set(shells_val))
    all_shells_seen: set[str] = set().union(*shell_sets)
    # Both 'main' and 'dev' must appear in the structured shells list
    # — not anywhere in the raw bytes. This rules out the false-positive
    # where 'main' shows up in some other field's value.
    assert {"main", "dev"}.issubset(all_shells_seen), (
        f"terminal_list should show both 'main' and 'dev', got shells={all_shells_seen!r}"
    )

    # Turn 3: close 'main'.
    resp3 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": ("Call terminal_close with shell='main'. That's it."),
            "previous_response_id": rid2,
            "background": True,
        },
    )
    resp3.raise_for_status()
    rid3 = resp3.json()["id"]
    final3 = poll_until_terminal(http_client, rid3, timeout=120)
    assert final3["status"] == "completed"
    close_calls = _get_output_items(final3, "function_call", "terminal_close")
    assert len(close_calls) >= 1, "Expected terminal_close to be invoked."

    # Turn 4: list again — 'main' must be gone, 'dev' must remain.
    resp4 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Call terminal_list (no arguments). Tell me exactly which shells are open now."
            ),
            "previous_response_id": rid3,
            "background": True,
        },
    )
    resp4.raise_for_status()
    rid4 = resp4.json()["id"]
    final4 = poll_until_terminal(http_client, rid4, timeout=120)
    assert final4["status"] == "completed"

    # Parse the post-close terminal_list output as structured JSON
    # — same approach as the earlier assertion in this test. Guarantees
    # we're checking the actual registry state, not any substring that
    # could match for unrelated reasons.
    list_calls_4 = _get_output_items(final4, "function_call", "terminal_list")
    assert len(list_calls_4) >= 1, "Turn 4 didn't invoke terminal_list."
    list_call_ids_4 = {c.get("call_id") for c in list_calls_4}
    list_json_after = [
        i.get("output") or ""
        for i in _get_output_items(final4, "function_call_output")
        if i.get("call_id") in list_call_ids_4
    ]
    assert list_json_after, "terminal_list produced no outputs in turn 4."
    shells_after: set[str] = set()
    for raw in list_json_after:
        parsed = json.loads(raw)
        shells_val = parsed.get("shells")
        assert isinstance(shells_val, list), f"terminal_list output missing 'shells': {parsed!r}"
        shells_after.update(shells_val)
    # 'dev' should still be present (we didn't close it).
    assert "dev" in shells_after, (
        f"Expected 'dev' still in terminal_list after closing 'main', got shells={shells_after!r}"
    )
    # 'main' must be gone — this is the whole point of close.
    assert "main" not in shells_after, (
        f"Expected 'main' to be GONE from terminal_list after close, "
        f"but the registry still reports shells={shells_after!r}. "
        f"terminal_close is not evicting shells from the manager."
    )


# ── Close wipes state (re-create under same name yields fresh shell) ──


def test_close_then_recreate_yields_fresh_shell(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """`terminal_close` + re-use of same name → new shell (no state leak).

    Turn 1: set env var in 'scratch'.
    Turn 2: close 'scratch'.
    Turn 3: in a new 'scratch', read the env var — must be empty.

    If closed shells don't actually die (e.g. a reference leak or an
    eviction bug), the env var would survive and this test fails.
    This complements the unit test ``test_dead_shell_replaced_on_next_use``
    with LLM-driven coverage.
    """
    # Turn 1: set a sentinel env var.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Use terminal_run with shell='scratch' to run: export SHOULD_VANISH=yes_im_here"
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    assert poll_until_terminal(http_client, rid1, timeout=120)["status"] == "completed"

    # Turn 2: close the shell.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": "Call terminal_close with shell='scratch'.",
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    assert poll_until_terminal(http_client, rid2, timeout=120)["status"] == "completed"

    # Turn 3: use a new 'scratch' — env var must be gone.
    resp3 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": ("Use terminal_run with shell='scratch' to run: echo VAR=$SHOULD_VANISH"),
            "previous_response_id": rid2,
            "background": True,
        },
    )
    resp3.raise_for_status()
    rid3 = resp3.json()["id"]
    final3 = poll_until_terminal(http_client, rid3, timeout=120)
    assert final3["status"] == "completed"

    fco = _get_output_items(final3, "function_call_output")
    stdouts = _parse_terminal_run_stdout(fco)
    combined = " ".join(stdouts)
    # With a fresh shell, $SHOULD_VANISH is unset → echo prints
    # 'VAR=' (empty). If the old shell was reused (close bug),
    # 'VAR=yes_im_here' appears and the test fails with a clear
    # message pointing at the close path.
    assert "yes_im_here" not in combined, (
        f"Closed shell's env var leaked into the new shell. "
        f"terminal_close is not actually killing the shell. "
        f"stdouts: {stdouts}"
    )
    assert "VAR=" in combined, (
        f"Expected 'VAR=' in fresh shell output (env var should be "
        f"unset after close). stdouts: {stdouts}"
    )


# ── Timeout kills command, shell survives ────────────────────────


def test_timeout_kills_long_command_but_shell_survives(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """``timeout_ms`` kills a stuck command; the shell stays usable.

    Turn 1: ``sleep 30`` with ``timeout_ms=500``. The tool output
    must have ``status="killed"`` and a non-zero exit code (128+SIGINT=130).
    Turn 2: a normal command in the same shell must succeed —
    proves the timeout killed the command, not the shell.

    Complements unit tests ``test_timeout_kills_running_command``
    and ``test_shell_survives_timeout_kill`` with a full-stack
    LLM-driven run.
    """
    import json

    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Use terminal_run with command='sleep 30' and "
                "timeout_ms=500. I want to see the timeout fire."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    rid1 = resp1.json()["id"]
    final1 = poll_until_terminal(http_client, rid1, timeout=120)
    assert final1["status"] == "completed", (
        f"Task failed (not the tool — the task): {final1.get('error')}"
    )

    # The tool's JSON should report status=killed.
    fco1 = _get_output_items(final1, "function_call_output")
    assert len(fco1) >= 1
    tool_outputs = [json.loads(i["output"]) for i in fco1 if i.get("output")]
    assert any(o.get("status") == "killed" for o in tool_outputs), (
        f"Expected a terminal_run output with status='killed' "
        f"(timeout_ms=500 against sleep 30), got: "
        f"{[o.get('status') for o in tool_outputs]}"
    )

    # Turn 2: confirm shell still works.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": "Use terminal_run to run: echo still-alive",
            "previous_response_id": rid1,
            "background": True,
        },
    )
    resp2.raise_for_status()
    rid2 = resp2.json()["id"]
    final2 = poll_until_terminal(http_client, rid2, timeout=120)
    assert final2["status"] == "completed"
    fco2 = _get_output_items(final2, "function_call_output")
    stdouts = _parse_terminal_run_stdout(fco2)
    combined = " ".join(stdouts)
    # If the timeout killed the shell (instead of just the command),
    # the next command's status would be "shell_crashed" and stdout
    # would be empty. This assertion fails clearly in that case.
    assert "still-alive" in combined, (
        f"Shell should have survived the timeout kill, but second "
        f"command's output is missing 'still-alive'. stdouts: {stdouts}"
    )


# ── Large-output truncation + disk persistence ───────────────────


def test_large_output_is_truncated_with_disk_path(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """A command producing >30 KB output is truncated with a disk-file marker.

    The agent runs a command that generates ~50 KB of text. The
    tool's stdout must contain the head+tail marker and a path to
    the full disk log (§6.7 Layer 2 + Layer 3).

    Failure modes this catches:
    - Ring buffer overflow loses output silently (no marker).
    - Inline cap not enforced → 50 KB leaks into the LLM's context.
    - Disk log write fails quietly → agent has no recovery path.
    """
    import json

    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Use terminal_run to run: python3 -c 'import sys; sys.stdout.write(\"x\" * 50000)'"
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    rid = resp.json()["id"]
    final = poll_until_terminal(http_client, rid, timeout=120)
    assert final["status"] == "completed"

    fco = _get_output_items(final, "function_call_output")
    tool_outputs = [json.loads(i["output"]) for i in fco if i.get("output")]
    terminal_outputs = [o for o in tool_outputs if "stdout" in o]
    assert terminal_outputs, "Expected at least one terminal_run output."

    out = terminal_outputs[0]
    stdout = out.get("stdout", "")
    # Inline cap: returned stdout must be well under 50 KB (the head
    # + tail slices sum to 20 KB + a short marker).
    assert len(stdout) < 25_000, (
        f"Inline stdout exceeds inline cap (got {len(stdout)} chars). "
        f"Either the 30 KB cap isn't firing, or the truncation logic "
        f"leaks more than head+tail."
    )
    # Truncation marker present.
    assert "truncated" in stdout, f"Expected 'truncated' marker in stdout, got: {stdout[:300]}"
    # Disk log path surfaced, so the agent can ``cat`` the full output.
    assert ".agent_plane/terminal/" in stdout, (
        f"Expected disk-log path in stdout, got: {stdout[:300]}"
    )
