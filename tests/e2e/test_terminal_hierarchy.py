"""End-to-end test for a two-level sub-agent hierarchy that drives
persistent terminals all the way down.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_terminal_hierarchy.py \\
        --llm-api-key $LLM_API_KEY -v

This is the only e2e test that proves the *full* chain:

    user ── turn ──▶ terminal_supervisor
                         │
                         ├─ spawn_sub_agent(type="worker", name="w1")
                         │     └─ worker w1
                         │          ├─ terminal_run(rA, async) ─▶ MARKER_UID=0.<...>_END
                         │          └─ terminal_run(rB, async) ─▶ MARKER_UID=0.<...>_END
                         │
                         └─ spawn_sub_agent(type="worker", name="w2")
                               └─ worker w2
                                    ├─ terminal_run(rA, async) ─▶ MARKER_UID=0.<...>_END
                                    └─ terminal_run(rB, async) ─▶ MARKER_UID=0.<...>_END

A real LLM drives every step. Every link in that chain must be
healthy for the final supervisor response to contain four
distinct sentinel floats — no single component can be faked.
"""

from __future__ import annotations

import json
import re

import httpx

from tests.e2e.conftest import poll_until_terminal
from tests.e2e.helpers import (
    final_assistant_text,
    get_output_items,
    parse_function_call_outputs,
)

# ─── Test ────────────────────────────────────────────────────


def test_supervisor_spawns_two_workers_each_drives_two_terminals(
    http_client: httpx.Client,
    terminal_supervisor_agent: str,
) -> None:
    """Supervisor spawns two worker sub-agents; each worker launches
    two async Python REPLs and reports a unique marker from each
    back to the supervisor; the supervisor aggregates all four.

    Uses ``random.random()`` formatted as a 12-digit float in the
    ``MARKER_UID=<float>_END`` sentinel rather than ``os.getpid()``
    — srt's per-shell PID-namespace isolation makes every REPL
    see itself as the same low PID (observed: ``4`` every time),
    so PIDs are useless as a uniqueness key under sandboxing.

    The final aggregated response must contain FOUR distinct
    sentinel floats somewhere in the assistant text or in the
    function_call_output / conversation-item blobs that reached
    the supervisor's context. Each REPL is a separate python3
    process so each ``random.random()`` yields a fresh float;
    four distinct floats = four separate processes actually ran.

    End-to-end failure modes this one test catches (the chain is
    long — any broken link loses at least one sentinel):

    - ``spawn_sub_agent`` not registered on the supervisor or its
      schema broken — zero ``spawn_sub_agent`` function_calls, the
      spawn-count assertion fires first and explains why.
    - Sub-agent-type resolution broken (``type="worker"`` not
      found) — spawn returns an error payload, the worker never
      runs, no sentinels reach the supervisor.
    - ``send_to_sub_agent`` not returning the sub-task_id / not
      starting a new task on the worker's conversation — the
      worker never receives the "launch two REPLs" instruction,
      so zero sentinels come back.
    - ``terminal_run(synchronous=false)`` broken inside the worker
      — the REPLs never spawn, ``terminal_send_input`` has nothing
      to drive, no sentinels.
    - ``terminal_send_input`` bytes not reaching the PTY inside a
      sub-agent conversation — the REPL never runs the
      ``random.random()`` line, no output returns.
    - ``check_task``/auto-delivery on the worker side broken —
      REPL output never routes back to the worker LLM, so the
      worker's final reply lacks sentinels.
    - Worker-to-supervisor completion signal (async_work_complete
      drain) broken — supervisor times out waiting for workers and
      the outer poll_until_terminal fails at 420s.
    - Supervisor aggregation happening before both workers return
      — fewer than 4 sentinels reach the final supervisor reply.

    Any one of those breaks this test. That's the point — a single
    integration proof for the whole hierarchy.

    :param http_client: HTTP client pointed at the live e2e server.
    :param terminal_supervisor_agent: Name of the uploaded
        supervisor agent (fixture returns ``"terminal_supervisor"``).
    """
    # The prompt is imperative and numbered so the LLM has minimal
    # freedom to re-order or skip steps. It is written for the
    # SUPERVISOR — the supervisor then paraphrases into
    # spawn_sub_agent(input=...) payloads for the workers.
    user_prompt = (
        "You are coordinating two worker sub-agents that each run "
        "two Python REPLs. Follow these steps exactly. Do NOT write "
        "any commentary before the tool calls — call the tools "
        "first.\n"
        "Step 1: In THIS response, emit TWO spawn_sub_agent tool "
        "calls in parallel:\n"
        '  - spawn_sub_agent(type="worker", name="w1", input='
        '"Launch TWO async Python REPLs on shells named rA and rB '
        "(use terminal_run with synchronous=false, command="
        "'python3 -i -q', shell='rA' and then shell='rB'). Once "
        "both REPLs are up, use terminal_send_input on the first "
        "REPL's task_id with chars='import random; "
        'print("MARKER_UID="+format(random.random(),\\".12f\\")+"_END")\\n\' '
        "and wait_ms=2000, then do the same on the second REPL. "
        "Your final reply to me must contain BOTH "
        "MARKER_UID=<float>_END markers from the REPL outputs, "
        'copied VERBATIM (not paraphrased).")\n'
        '  - spawn_sub_agent(type="worker", name="w2", input='
        '"Launch TWO async Python REPLs on shells named rA and rB '
        "(use terminal_run with synchronous=false, command="
        "'python3 -i -q', shell='rA' and then shell='rB'). Once "
        "both REPLs are up, use terminal_send_input on the first "
        "REPL's task_id with chars='import random; "
        'print("MARKER_UID="+format(random.random(),\\".12f\\")+"_END")\\n\' '
        "and wait_ms=2000, then do the same on the second REPL. "
        "Your final reply to me must contain BOTH "
        "MARKER_UID=<float>_END markers from the REPL outputs, "
        'copied VERBATIM (not paraphrased).")\n'
        "Step 2: The two spawn_sub_agent results auto-deliver as "
        "'[System: task ... completed]' user messages with the "
        "worker's final text embedded. Wait for BOTH before "
        "replying.\n"
        "Step 3: In your final assistant message, list ALL FOUR "
        "PIDs that the workers reported (two from w1 plus two "
        "from w2). Quote each PID verbatim. Then say 'Collected 4 "
        "PIDs across 2 workers.'"
    )

    resp = http_client.post(
        "/v1/responses",
        json={
            "model": terminal_supervisor_agent,
            "input": user_prompt,
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    # 420s budget breakdown: supervisor plans + emits 2 spawn calls
    # (~1 LLM round), each worker then does ~4 LLM rounds
    # (launch rA, launch rB, send_input on rA, send_input on rB,
    # final reply) in parallel, worker completion auto-delivers,
    # supervisor aggregates (~1-2 LLM rounds). Real LLMs stall
    # occasionally so we give generous headroom — the sub-agent
    # e2e default is 240s for a single sub-agent, this test fans
    # out one level deeper.
    final = poll_until_terminal(http_client, response_id, timeout=420)
    assert final["status"] == "completed", (
        f"Supervisor turn did not complete: status="
        f"{final.get('status')!r}, error={final.get('error')!r}. "
        f"If the outer poll timed out, one of the workers almost "
        f"certainly never signaled async_work_complete (drain "
        f"regression) — check the worker's terminal tasks."
    )

    # ── Assertion 1: the supervisor actually emitted ≥ 2 spawn_sub_agent calls. ──
    # If zero, the LLM skipped delegation entirely and the PID
    # check below would fire with a confusing "no PIDs" message.
    # The floor is 2 (one per named worker); > 2 is fine — the LLM
    # may have retried after a name collision. < 2 means the
    # hierarchy was never built in the first place and no deeper
    # assertion is meaningful.
    spawn_calls = get_output_items(final, "function_call", name="spawn_sub_agent")
    assert len(spawn_calls) >= 2, (
        f"Expected at least 2 spawn_sub_agent function_calls "
        f"(one for w1 and one for w2), got {len(spawn_calls)}. "
        f"Tool calls emitted: "
        f"{[i.get('name') for i in get_output_items(final, 'function_call')]}. "
        f"If 0, spawn_sub_agent is not registered on the "
        f"supervisor or the LLM didn't pick it up. If 1, the "
        f"supervisor only dispatched one worker — the hierarchy "
        f"was never fan-out."
    )

    # ── Assertion 2: at least 4 distinct PIDs reach the supervisor's context. ──
    # Build a blob containing every string the supervisor could
    # have used to aggregate the answer: its own assistant text
    # AND every function_call_output it saw (the auto-delivered
    # worker completions live in the conversation as user
    # messages, but they're also surfaced to the supervisor's LLM
    # input — we pull the conversation items below to be
    # exhaustive).
    pty_blob_parts: list[str] = []

    # The supervisor's own final text (should contain all 4 PIDs
    # per Step 3 of the prompt).
    pty_blob_parts.append(final_assistant_text(final))

    # Every function_call_output payload string (spawn_sub_agent
    # returns a handle, but the result_preview / auto-delivery
    # carries the worker's final reply with the PIDs).
    for out in parse_function_call_outputs(final):
        pty_blob_parts.append(json.dumps(out))

    # The auto-delivered '[System: task ... completed]' user
    # messages in the conversation — these are how worker results
    # actually reach the supervisor's LLM context. Pull from the
    # conversation store directly so a buggy output-serialization
    # can't hide them from the test.
    #
    # IMPORTANT: we deliberately skip the ORIGINAL user-role prompt
    # text (which contains literal integers like 'wait_ms=1500')
    # because those would false-positive the PID regex below.
    # '[System: task ...]' auto-deliveries arrive as user-role
    # messages too, but their text starts with the '[System: task '
    # sentinel — that's our inclusion filter.
    conv_id = final["conversation"]["id"]
    items_resp = http_client.get(
        f"/v1/conversations/{conv_id}/items",
        params={"limit": 100},
    )
    items_resp.raise_for_status()
    for item in items_resp.json().get("data", []):
        if item.get("type") != "message":
            continue
        role = item.get("role")
        for block in item.get("content", []):
            text = block.get("text") or ""
            if not text:
                continue
            # Always include assistant text (supervisor reasoning /
            # final reply). For user messages, ONLY include auto-
            # delivered system messages — skip the human prompt so
            # its 'wait_ms=1500' etc. can't satisfy the PID regex.
            if role == "assistant":
                pty_blob_parts.append(text)
            elif role == "user" and text.startswith("[System: task"):
                pty_blob_parts.append(text)

    pty_blob = "\n".join(pty_blob_parts)

    # Match the exact MARKER_PID=<digits>_END sentinel the workers
    # were instructed to emit. Counting these (rather than loose
    # 4-6 digit integers) rules out false positives from unrelated
    # numbers the LLM might reference (wait_ms=2000, tool arg
    # counts, etc.) AND rules out hallucinated "PID: 4"-style
    # replies — the LLM cannot plausibly emit this exact sentinel
    # string without having seen it arrive from a real REPL stdout.
    # Extract just the digit portion so set-dedup counts distinct
    # PIDs.
    uid_matches = re.findall(r"MARKER_UID=([\d.]+)_END", pty_blob)
    uid_candidates = set(uid_matches)
    # We use ``random.random()`` rather than ``os.getpid()`` because
    # the srt sandbox wraps each shell in a fresh PID namespace —
    # every python3 REPL sees itself as a low PID (observed:
    # literally ``4`` every time), so PIDs are NOT unique across
    # shells. A fresh random float per process is unique by design
    # (14 significant digits → collision probability is vanishing).
    assert len(uid_candidates) >= 4, (
        f"Expected at least 4 distinct MARKER_UID=<float>_END "
        f"sentinels across the supervisor's final text and "
        f"conversation, got {len(uid_candidates)} distinct of "
        f"{len(uid_matches)} total: {sorted(uid_candidates)!r}. "
        f"Each REPL is a separate python3 process so each "
        f"random.random() call yields a unique float; 4 distinct "
        f"sentinels = 4 separate REPL processes actually ran "
        f"across 2 workers. Fewer means at least one link in the "
        f"chain (spawn_sub_agent / worker LLM / terminal_run "
        f"async / terminal_send_input / PTY stdout routing / "
        f"worker completion auto-delivery) broke. Blob tail: "
        f"...{pty_blob[-1500:]}"
    )

    # ── Assertion 3 (soft): supervisor's final text references 4 PIDs. ──
    # Not load-bearing — the PID count above is what proves the
    # plumbing. This just verifies the UX: the supervisor's final
    # assistant message mentions the count it was asked to
    # aggregate. Accept the digit '4', the word 'four', or (as a
    # last resort) the presence of at least four MARKER_PID
    # sentinels in the assistant text itself.
    final_text = final_assistant_text(final)
    text_uids = set(re.findall(r"MARKER_UID=([\d.]+)_END", final_text))
    mentions_count = "4" in final_text or "four" in final_text.lower() or len(text_uids) >= 4
    assert mentions_count, (
        f"Supervisor's final assistant text didn't clearly "
        f"aggregate four PIDs — expected '4', 'four', or four "
        f"PID-shaped integers in the reply. Got: "
        f"{final_text[:1000]!r}. The plumbing may still work "
        f"(see Assertion 2) but the UX — the supervisor "
        f"actually SUMMARISING the four PIDs to the user — is "
        f"not surfacing."
    )
