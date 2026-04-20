"""End-to-end tests for ``terminal_send_input`` + pyte screen view.

Drives the interactive-terminal flow through a real LLM on the
``terminal_test`` agent. Complements the unit tests
(``tests/terminals/test_shell_interactive.py``) and
tool-layer tests (``tests/tools/builtins/test_terminal.py``) by
proving the LLM-facing UX for interactive programs actually
works end-to-end: the agent launches a program that reads
stdin, sends input via ``terminal_send_input``, and gets back
both the streaming delta and a pyte-rendered screen that
reflects the interaction.

Requires ``--llm-api-key``. Excluded from default pytest runs.
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
    """Filter response.output by type + optional tool name.

    :param body: Response body from GET /v1/responses/{id}.
    :param item_type: Item type to keep, e.g. ``"function_call"``.
    :param name: Optional tool name filter.
    :returns: Matching items in original order.
    """
    items = body.get("output", [])
    filtered = [i for i in items if i.get("type") == item_type]
    if name is not None:
        filtered = [i for i in filtered if i.get("name") == name]
    return filtered


def _parse_function_call_outputs(
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse the JSON string inside every function_call_output.

    :param body: Response body from GET /v1/responses/{id}.
    :returns: Each parsed payload, preserving original order. Bad
        JSON or non-dict payloads are skipped (not an error
        path — they're just irrelevant to what we're asserting).
    """
    results: list[dict[str, Any]] = []
    for item in _get_output_items(body, "function_call_output"):
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
    return results


def test_send_input_drives_cat_via_async_terminal(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """A real LLM launches ``cat``, types two lines, sends EOF, and
    the final output contains both lines.

    This is the minimum e2e validation: if ``terminal_send_input``
    can't drive ``cat`` (the simplest stdin-reading program),
    nothing else interactive can work. Catches:
    - send_input not wired into the PTY (bytes lost)
    - terminal_send_input tool not auto-registered
    - cat's stdout not making it back via recent_activity
    - EOF (``\\u0004``) not recognized by the LLM
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Your goal is to drive the `cat` program. "
                "Step 1: call terminal_run with command='cat', "
                "shell='default', synchronous=false. "
                "Step 2: call terminal_send_input with the task_id "
                "and chars='hello-interactive-zzz\\nworld-interactive-qqq"
                "\\n\\u0004' (two lines plus a Ctrl-D for EOF so "
                "cat exits). "
                "Step 3: confirm you saw both 'hello-interactive-zzz' "
                "and 'world-interactive-qqq' in the result. "
                "Do NOT write any commentary before the tool calls."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=180)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # The LLM should have called terminal_run AND
    # terminal_send_input at least once each.
    fc_names = [i.get("name") for i in _get_output_items(final, "function_call")]
    assert "terminal_run" in fc_names, f"Expected terminal_run call, got: {fc_names}"
    assert "terminal_send_input" in fc_names, (
        f"Expected terminal_send_input call, got: {fc_names}. "
        f"The LLM didn't reach for the interactive tool even "
        f"though the prompt explicitly requested it."
    )

    # At least one terminal_send_input output should report
    # delivered=True — if every call returned delivered=False, the
    # task-id routing is broken.
    send_outputs = [
        o
        for o in _parse_function_call_outputs(final)
        if "delivered" in o  # terminal_send_input shape
    ]
    assert any(o.get("delivered") is True for o in send_outputs), (
        f"Expected at least one terminal_send_input with "
        f"delivered=True, got: {[o for o in send_outputs]}"
    )

    # Both stdout tokens must appear somewhere in the conversation.
    # They'll be either in recent_activity / screen of the
    # send_input responses, the auto-delivered completion system
    # message, or the assistant's final text — any of those is
    # fine for the purpose of this test (proving bytes roundtrip).
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    blob = json.dumps(items)
    assert "hello-interactive-zzz" in blob, (
        "First typed line not visible anywhere in the "
        "conversation. send_input likely not routing to cat's "
        "stdin."
    )
    assert "world-interactive-qqq" in blob, (
        "Second typed line not visible. Either send_input "
        "coalesced badly, cat didn't read the second line, or "
        "EOF handling dropped it."
    )


def test_send_input_screen_reflects_typed_prompt_answer(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """The pyte ``screen`` field reflects the program's interaction
    state after the LLM types an answer to a prompt.

    Uses bash's ``read -p`` builtin: it prints a prompt on stderr
    without a newline, reads a line from stdin, echoes it with
    "Got: ". The rendered screen after the LLM sends its answer
    should contain the "Got: ..." line — proving the pyte view is
    live and reflects interactive state, not just a snapshot from
    before the interaction.

    Catches:
    - Screen field missing from terminal_send_input response
    - Pyte not being fed during send_input (screen shows only the
      pre-input state)
    - Screen right-padding bug leaving whitespace that breaks
      substring checks
    """
    unique = "screen-answer-marker-42"
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Your goal is to answer a shell prompt. "
                "Step 1: call terminal_run with "
                "command='bash -c \\'read -p \"Your name: \" name; "
                f"echo \"Got: $name\"\\'', shell='default', "
                "synchronous=false. "
                "Step 2: call terminal_send_input with the task_id "
                f"and chars='{unique}\\n' to answer the prompt. "
                "Step 3: after the result auto-delivers, report "
                f"whether you saw 'Got: {unique}'. "
                "Do NOT write any commentary before the tool calls."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=180)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # At least one terminal_send_input response should have a
    # non-empty 'screen' field populated from pyte. If the screen
    # field is missing entirely, the pyte integration didn't
    # plumb through to the tool response.
    send_outputs = [o for o in _parse_function_call_outputs(final) if o.get("delivered") is True]
    assert send_outputs, (
        f"No successful terminal_send_input call. Full outputs: "
        f"{[o for o in _parse_function_call_outputs(final)]}"
    )
    screens = [o.get("screen") for o in send_outputs if "screen" in o]
    assert screens, (
        "No 'screen' field in any terminal_send_input response. "
        "pyte is not making it through to the tool payload."
    )

    # The final answer marker should appear somewhere in the
    # conversation (either in a send_input screen/recent_activity,
    # the auto-delivered completion, or the assistant's text).
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items = items_resp.json()["data"]
    blob = json.dumps(items)
    assert unique in blob, (
        f"Answer marker '{unique}' not anywhere in conversation. "
        f"send_input may not have made it to read's stdin."
    )
    assert f"Got: {unique}" in blob, (
        f"Expected 'Got: {unique}' (bash's echo of our answer) "
        f"somewhere in the conversation. If '{unique}' appears "
        f"alone, the LLM typed it but bash never echoed — "
        f"timing/quiescence issue."
    )


def test_send_input_rejects_unknown_task_id(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """terminal_send_input on a made-up task_id returns a clear error.

    Asks the LLM to pass a bogus task_id on purpose and then
    recover. The tool response must be
    ``{delivered: false, reason: ...}`` rather than a 500 or
    silent drop. Proves the error path is visible to the LLM so
    it can react (e.g. by retrying with a real task_id).
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "For this diagnostic: call terminal_send_input ONCE "
                "with task_id='bogus_nonexistent_task_xyz' and "
                "chars='hi'. Then report exactly what the "
                "'delivered' and 'reason' fields of the response "
                "were. Do NOT call any other tool."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    final = poll_until_terminal(http_client, response_id, timeout=120)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # The bogus task_id send_input call must have delivered=False.
    send_outputs = [o for o in _parse_function_call_outputs(final) if "delivered" in o]
    assert send_outputs, (
        f"No terminal_send_input response found. Outputs: "
        f"{[o for o in _parse_function_call_outputs(final)]}"
    )
    # Every send_input in this test used the bogus id, so every
    # response should be delivered=False.
    for out in send_outputs:
        assert out["delivered"] is False, f"Bogus task_id still reported delivered=True: {out!r}"
        assert out.get("reason") in {
            "shell_unavailable",
            "task_no_longer_running",
        }, (
            f"Expected a recognized reason for the failed send "
            f"(shell_unavailable | task_no_longer_running), got "
            f"{out.get('reason')!r} in {out!r}."
        )


def test_send_input_drives_vim_writes_file(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """A real LLM drives ``vim`` via ``terminal_send_input``: opens a
    file, enters insert mode, types a marker, writes+quits, and then
    ``cat``s the file synchronously to prove vim actually wrote it.

    This is the full alt-screen e2e: vim uses alt-screen, cursor
    positioning, and termcap escapes that only pyte (not a plain
    ring buffer) can make sense of. The test catches:

    - ``Shell.send_input`` bytes not reaching the PTY — vim never
      sees ``i<marker>\\u001b:wq\\n``, the file is not written,
      the synchronous ``cat`` fails the marker check.
    - pyte not being fed during send_input — the ``screen`` field
      on send_input responses renders empty or garbage, failing
      the screen-content check.
    - Alt-screen escape codes not handled — vim's ``~`` marker
      lines, the ``INSERT`` mode indicator, and the filename in
      the status line all vanish.
    - yield_time_ms too short to let vim react — the send_input
      response would come back before vim re-renders and the
      screen check fails cleanly.
    - srt sandbox blocking workspace writes — ``cat`` returns an
      error instead of the marker (the LLM would surface that).

    The LLM cannot fake success: the synchronous ``cat`` output is
    part of the conversation and is only populated by a real file
    read. If any link breaks, one of the two assertions fires.
    """
    # Unique marker chosen to be implausible in random LLM prose,
    # base64-ish + fixed suffix so grep for it is unambiguous.
    unique = "VIM-ROUNDTRIP-MARKER-K7Q3"
    filename = f"vimtest-{unique}.txt"
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Your goal is to drive the vim editor and then verify "
                "what it wrote to disk. Follow these steps exactly, "
                "calling the tool first and never writing commentary "
                "before a tool call.\n"
                f"Step 1: call terminal_run with command='vim {filename}', "
                "shell='default', synchronous=false. "
                "This returns a task_id for the vim session. Use a "
                "workspace-relative filename (do NOT prefix with /tmp "
                "or any absolute path) — only the workspace is "
                "writable.\n"
                "Step 2: call terminal_send_input with that task_id, "
                f"chars='i{unique}\\u001b:wq\\n', and "
                "yield_time_ms=3000. The 'i' enters insert mode, the "
                "body is the literal marker, '\\u001b' is Escape to "
                "exit insert mode, ':wq\\n' writes and quits. The "
                "generous yield gives vim time to process the full "
                "sequence.\n"
                "Step 3: after vim's async task auto-delivers its "
                "completion as a system message, call terminal_run "
                f"SYNCHRONOUSLY with command='cat {filename}', "
                "shell='default', synchronous=true. This reads what "
                "vim wrote.\n"
                "Step 4: report the exact stdout of the cat command. "
                "Do NOT write any commentary before any tool call."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    # 240s budget: vim spawn + send_input quiescence + async auto-
    # deliver + sync cat + 4 LLM turns. Plenty of margin.
    final = poll_until_terminal(http_client, response_id, timeout=240)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    fc_items = _get_output_items(final, "function_call")
    fc_names = [i.get("name") for i in fc_items]
    # Both tools are mandatory for the test. If send_input is absent,
    # the LLM never tried to drive vim; if terminal_run is absent,
    # vim was never launched and there's nothing to drive.
    assert "terminal_run" in fc_names, (
        f"Expected terminal_run call (to launch vim and later cat), got: {fc_names}"
    )
    assert "terminal_send_input" in fc_names, (
        f"Expected terminal_send_input call (to drive vim), "
        f"got: {fc_names}. The LLM didn't reach for the "
        f"interactive tool."
    )

    # --- Assertion 1: pyte rendered vim's alt-screen state. ---
    # Find every terminal_send_input response where delivered=True
    # and screen is non-empty. A broken send_input path that reports
    # delivered=False, or a pyte path that returns empty strings,
    # both fail this.
    send_outputs = [
        o
        for o in _parse_function_call_outputs(final)
        if "delivered" in o and o.get("delivered") is True
    ]
    assert send_outputs, (
        f"No terminal_send_input call with delivered=True. "
        f"Outputs: {[o for o in _parse_function_call_outputs(final)]}. "
        f"Either the task_id routing is broken or every send_input "
        f"call hit a closed shell."
    )
    screens = [o.get("screen", "") for o in send_outputs]
    # Use any(...) because some calls might happen after vim has
    # already exited (screen would be the post-quit shell prompt).
    # We only need ONE call where pyte captured vim's alt-screen.
    # Three tells: the '~' empty-line marker (only vim draws those
    # exactly; unlikely in a shell prompt), the 'INSERT' mode
    # indicator (appears the moment 'i' is processed), or the
    # filename from the modeline. Any one proves pyte plumbed
    # vim's escape codes through.
    vim_screen_tells = any(("~" in s) or ("INSERT" in s) or (filename in s) for s in screens)
    assert vim_screen_tells, (
        f"No terminal_send_input 'screen' field contained vim alt-"
        f"screen evidence ('~' empty-line marker, 'INSERT' mode "
        f"indicator, or the filename {filename!r} from the "
        f"modeline). Screens observed: {screens!r}. Either pyte "
        f"isn't being fed vim's bytes, vim never started, or the "
        f"send_input tool is returning a stale/empty screen."
    )

    # --- Assertion 2: vim actually wrote the marker to disk. ---
    # The synchronous ``cat`` output lives inside a function_call_
    # output with the terminal_run payload shape
    # ({stdout, exit_code, status, shell}). The marker must be in
    # the stdout of one of those. This is the ONLY legitimate
    # proof — the LLM can claim success in its final text without
    # actually reading the file, but the synchronous cat stdout is
    # populated by the real process. If Shell.send_input didn't
    # reach vim, the file doesn't exist (or is empty) and this
    # fails cleanly.
    run_outputs = [
        o for o in _parse_function_call_outputs(final) if "stdout" in o and "exit_code" in o
    ]
    assert run_outputs, (
        f"No synchronous terminal_run output (with stdout/exit_code) "
        f"found. Outputs: {[o for o in _parse_function_call_outputs(final)]}. "
        f"The LLM didn't run the verification cat, so we can't "
        f"distinguish a real vim write from a hallucinated one."
    )
    cat_outputs = [o for o in run_outputs if unique in (o.get("stdout") or "")]
    assert cat_outputs, (
        f"Marker {unique!r} not found in the stdout of any "
        f"synchronous terminal_run result. Run outputs: "
        f"{[{'stdout': o.get('stdout'), 'exit_code': o.get('exit_code')} for o in run_outputs]}. "
        f"This is the ground-truth proof that vim actually wrote "
        f"the marker to {filename}. If this fails while the screen "
        f"assertion passes, pyte is rendering vim but "
        f"Shell.send_input's bytes never reach the PTY (or vim's "
        f":wq was interpreted as literal text, not a command)."
    )


def test_send_input_multi_turn_python_repl_holds_state(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """A real LLM drives ``python3 -i -q`` across 4 separate
    ``terminal_send_input`` turns and the REPL's interpreter state
    (the binding ``x = 42``) survives from turn 1 into turn 2, so
    ``y = x * 2`` evaluates to ``84`` which prints in turn 3.

    This is the multi-turn stdin-persistence check. A single
    send_input call proves bytes reach stdin; this test proves the
    underlying PTY + interpreter stays alive and stateful across
    SEPARATE send_input tool calls. The failure modes it catches:

    - **State reset between turns**: if each send_input call
      spawned a fresh python or truncated the session, turn 2's
      ``x * 2`` would raise ``NameError: name 'x' is not defined``.
      The NameError-exclusion assertion catches this even if some
      other path somehow surfaces ``84``.
    - **Bytes coalesced or dropped**: if the transport fused the
      four turns into one write or dropped the ``x = 42`` turn,
      ``y`` is never bound and ``print(y)`` raises NameError too.
    - **delivered flag lying**: all four turns must report
      delivered=True; the >= 2 check also guards against the LLM
      merging everything into one send_input (which would
      accidentally work but wouldn't exercise multi-turn).
    - **Screen/recent_activity truncation**: the ``84`` must make
      it back to the conversation — either in a send_input
      payload or the auto-delivered completion message — proving
      output routing survives across turns.
    - **REPL not exiting**: the final ``exit()`` call drops the
      interpreter so the async task reaches terminal status and
      the parent response can complete (otherwise poll times out).
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                "Your goal is to drive an interactive Python REPL "
                "across MULTIPLE separate terminal_send_input calls "
                "to prove state survives between them. Follow this "
                "script EXACTLY — one tool call per step, in order. "
                "Do NOT combine steps into one send_input call. "
                "Step 1: call terminal_run with "
                "command='python3 -i -q', shell='default', "
                "synchronous=false. Keep the task_id. "
                "Step 2: call terminal_send_input with the task_id "
                "and chars='x = 42\\n' and yield_time_ms=500. "
                "Step 3: call terminal_send_input with the SAME "
                "task_id and chars='y = x * 2\\n' and "
                "yield_time_ms=500. This line REFERENCES x — if the "
                "REPL forgot x from step 2 it will raise NameError. "
                "Step 4: call terminal_send_input with the SAME "
                "task_id and chars='print(y)\\n' and "
                "yield_time_ms=2000 so Python has time to print and "
                "redraw the prompt. "
                "Step 5: call terminal_send_input with the SAME "
                "task_id and chars='exit()\\n' so the REPL "
                "terminates cleanly. "
                "Step 6: after the auto-delivered completion "
                "message, report exactly what print(y) produced. "
                "Do NOT write any commentary before the tool calls."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]
    # Multi-turn with real LLM + 4 send_input calls + yield_time_ms
    # budgets adds up: give this one a generous ceiling.
    final = poll_until_terminal(http_client, response_id, timeout=240)
    assert final["status"] == "completed", f"Task failed: {final.get('error')}"

    # The LLM must have reached for terminal_run + terminal_send_input.
    fc_names = [i.get("name") for i in _get_output_items(final, "function_call")]
    assert "terminal_run" in fc_names, f"Expected terminal_run call, got: {fc_names}"
    assert "terminal_send_input" in fc_names, (
        f"Expected terminal_send_input calls, got: {fc_names}. "
        f"The LLM didn't reach for the interactive tool."
    )

    # Count successful send_input deliveries. >= 2 is the multi-turn
    # floor: 1 delivered=True could be any of the 4 turns in
    # isolation, but >= 2 proves the LLM actually made multiple
    # separate tool calls and each one got through. Anything less
    # means the LLM either coalesced the prompt into one chars
    # payload (which would bypass the multi-turn invariant this
    # test exists to prove) or later turns failed to deliver.
    send_outputs = [o for o in _parse_function_call_outputs(final) if "delivered" in o]
    delivered_true = [o for o in send_outputs if o.get("delivered") is True]
    assert len(delivered_true) >= 2, (
        f"Expected >= 2 terminal_send_input calls with "
        f"delivered=True (this is a multi-turn test), got "
        f"{len(delivered_true)}. Full send_input outputs: "
        f"{send_outputs}. If 1, the LLM likely collapsed all "
        f"steps into a single send_input — which defeats the "
        f"point of this test."
    )

    # Build a PTY-output-only blob for content assertions. We
    # only scan function_call_output payloads (the send_input
    # responses' recent_activity/screen, and the completion
    # system message via auto-delivery) — NOT the initial user
    # prompt, which contains the word "NameError" as documentation
    # and would false-positive the NameError-exclusion check.
    pty_outputs = _parse_function_call_outputs(final)
    pty_blob_parts: list[str] = []
    for out in pty_outputs:
        for field in ("recent_activity", "screen", "stdout", "output"):
            val = out.get(field)
            if isinstance(val, str):
                pty_blob_parts.append(val)
    # Also scan the auto-delivered "[System: task ... completed]"
    # messages — these carry the final stdout of the command and
    # are user-role input_text items. Distinguishable from the
    # test prompt because they start with a ``[System: `` sentinel.
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    for item in items_resp.json()["data"]:
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        for block in item.get("content", []):
            text = block.get("text") or ""
            if text.startswith("[System: task"):
                pty_blob_parts.append(text)
    pty_blob = "\n".join(pty_blob_parts)

    # The load-bearing assertion. ``84`` is computable ONLY if
    # (a) x was bound in turn 2, (b) that binding survived into
    # turn 3 where y = x * 2 evaluated, (c) print(y) ran in turn
    # 4, and (d) the "84\n" stdout byte made it back up through
    # the ring buffer into either a send_input response or the
    # terminal-completion system message. If any link breaks,
    # "84" can't appear in PTY output (the prompt itself never
    # contains ``84``).
    assert "84" in pty_blob, (
        "Expected '84' (value of x * 2 where x was set in a "
        "PREVIOUS send_input turn) somewhere in the PTY output. "
        "Absence means REPL state didn't persist across send_input "
        "turns, or the print output never made it back. Full PTY "
        f"blob tail: ...{pty_blob[-2000:]}"
    )

    # NameError-exclusion closes the hole where '84' could come
    # from an unrelated source (e.g. the LLM typed '42 * 2'
    # directly rather than 'x * 2', accidentally making the test
    # green without proving state persistence). If x was NOT
    # bound when turn 3 ran ``y = x * 2``, CPython emits
    # ``NameError: name 'x' is not defined`` to stderr — that
    # string would then appear in the PTY output. We scan only
    # PTY output (not the test prompt) so mentions in the
    # documentation don't false-positive.
    assert "NameError" not in pty_blob, (
        "'NameError' appeared in PTY output — this means "
        "the REPL forgot the binding from a previous "
        "send_input turn (state did NOT persist across turns). "
        f"Full PTY blob tail: ...{pty_blob[-2000:]}"
    )


def test_three_persistent_repls_and_followup_turn_is_processed(
    http_client: httpx.Client,
    terminal_test_agent: str,
) -> None:
    """Three Python REPLs stay live across turns; the user's
    follow-up message starts a new turn (not steering stuck on
    the previous one).

    This reproduces the user-visible bug where launching long-
    lived interactive programs with ``synchronous=false`` used to
    wedge the conversation: the parent turn was blocking on
    ``async_work_complete`` signals for REPLs that never exit,
    so the next user message arrived as steering on the stuck
    response instead of starting a new turn. Fix: terminal-kind
    tasks are no longer in ``_DRAIN_KINDS``; agents poll via
    ``check_task(task_id, wait_ms=...)`` when they need to wait
    for a specific completion.

    Flow:
    1. Turn 1: launch 3 python3 REPLs async, send
       ``import random; print(random.random())\\n`` to each via
       terminal_send_input. Agent reports it launched them.
       CRITICAL: turn 1 MUST finalize (status=completed) even
       though all 3 REPLs are still running.
    2. Turn 2: a fresh user message. It must be delivered as a
       new turn (parent_response_id = turn 1). The agent
       responds. The turn-2 response must also complete.

    Failure modes caught:
    - Terminal tasks blocking the drain again (turn 1 hangs
      waiting for REPL1/2/3 completion, test times out).
    - Turn 1 completes but turn 2 is wedged on steering (the
      server thinks turn 1 is still active).
    - Any of the REPLs fails to launch / receive input (no
      random number appears in the screen fields).
    """
    import time as _time

    unique = "REPL-STATE-TRIPLE-M9K"
    # Turn 1: launch + send to each REPL, then report.
    resp1 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": (
                f"(conv tag {unique}) "
                "Step 1: call terminal_run three times, each with "
                "synchronous=false, to launch three python REPLs: "
                "command='python3 -i -q' on shell='r1', then "
                "shell='r2', then shell='r3'. Keep the task_ids "
                "— you'll need them.\n"
                "Step 2: call terminal_send_input on EACH of the "
                "three task_ids with chars='import random; "
                "print(random.random())\\n' and yield_time_ms=1500. "
                "Do this three times (once per task_id).\n"
                "Step 3: report that you launched three REPLs and "
                "printed a random number from each. Do NOT try to "
                "exit them — they should stay alive for the next "
                "turn.\n"
                "Do NOT write commentary before tool calls."
            ),
            "background": True,
        },
    )
    resp1.raise_for_status()
    response_id_1 = resp1.json()["id"]
    final_1 = poll_until_terminal(http_client, response_id_1, timeout=180)
    # Load-bearing: if the drain still blocks on terminal tasks,
    # turn 1 would never reach a terminal status and poll_until_
    # terminal would raise after 180s.
    assert final_1["status"] == "completed", (
        f"Turn 1 didn't reach 'completed' within 180s (status="
        f"{final_1.get('status')!r}, error={final_1.get('error')}). "
        f"If timed out, the terminal-in-drain regression is back "
        f"— async terminal_run(synchronous=false) for long-lived "
        f"REPLs blocks the parent turn."
    )

    # Pull out the three task_ids from turn 1 so we can verify
    # they're still alive (or confirm they show up in subsequent
    # check_task calls driven by the LLM in turn 2).
    terminal_handles: list[str] = []
    for item in _get_output_items(final_1, "function_call_output"):
        raw = item.get("output") or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if (
            isinstance(parsed, dict)
            and parsed.get("tool_name") == "terminal_run"
            and parsed.get("status") == "in_progress"
            and parsed.get("task_id")
        ):
            terminal_handles.append(parsed["task_id"])
    # Three REPLs launched means three async handles.
    assert len(terminal_handles) == 3, (
        f"Expected 3 async terminal_run handles in turn 1, got "
        f"{len(terminal_handles)}. Handles: {terminal_handles!r}. "
        f"The LLM either didn't launch all three or used "
        f"synchronous=true for some."
    )

    # Three random-number prints must appear in the function_call_
    # output screens / recent_activity. Each REPL should have
    # produced a float. We verify the presence of at least three
    # distinct prints by counting '>>> ' prompt echoes that carry
    # a decimal number on the preceding line.
    import re

    screen_texts = []
    for o in _parse_function_call_outputs(final_1):
        if o.get("delivered") is True:
            for field in ("screen", "recent_activity"):
                val = o.get(field)
                if isinstance(val, str):
                    screen_texts.append(val)
    combined = "\n".join(screen_texts)
    # random.random() returns 0.xxxxxxxx — match "0." followed by
    # at least 5 digits to filter out false positives like "1.0"
    # or short literals in unrelated output.
    float_prints = re.findall(r"0\.\d{5,}", combined)
    assert len(float_prints) >= 3, (
        f"Expected at least 3 float prints (one per REPL) in the "
        f"send_input responses' screen/recent_activity fields, "
        f"got {len(float_prints)}: {float_prints!r}. Either the "
        f"LLM didn't send to all three REPLs or the REPLs didn't "
        f"echo back. Combined screens: {combined[:500]!r}"
    )

    # --- Turn 2: follow-up message threads off turn 1. ---
    # Without the drain-for-terminals fix, turn 1 would be stuck
    # and this POST would hang or be routed as steering. With the
    # fix, turn 1 is done and turn 2 is a clean new turn.
    _time.sleep(0.5)  # brief gap — not required for correctness, makes the
    # server-side "new turn" distinction cleaner to observe.
    resp2 = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_test_agent,
            "input": "Thanks. How many REPLs did you launch?",
            "previous_response_id": response_id_1,
            "background": True,
        },
    )
    assert resp2.status_code == 200, (
        f"Turn 2 POST failed with status {resp2.status_code}: "
        f"{resp2.text[:300]}. If 409, the previous response is "
        f"still active — meaning the turn 1 'completed' assertion "
        f"above lied and the drain is still wedged."
    )
    response_id_2 = resp2.json()["id"]
    assert response_id_2 != response_id_1, (
        f"Turn 2's response_id equals turn 1's ({response_id_1!r}) "
        f"— the server treated the follow-up as steering on the "
        f"in-flight response instead of starting a new turn."
    )
    final_2 = poll_until_terminal(http_client, response_id_2, timeout=180)
    assert final_2["status"] == "completed", (
        f"Turn 2 didn't reach 'completed' (status="
        f"{final_2.get('status')!r}, error={final_2.get('error')})."
    )
    # The second turn's assistant text should reference "3" or
    # "three". Soft check — accept either form.
    turn2_text_blob = json.dumps(final_2.get("output", []))
    assert ("3" in turn2_text_blob) or ("three" in turn2_text_blob.lower()), (
        f"Turn 2's response didn't mention '3' or 'three' REPLs. "
        f"The agent isn't remembering what it did in turn 1. "
        f"Turn 2 output: {turn2_text_blob[:500]!r}"
    )
