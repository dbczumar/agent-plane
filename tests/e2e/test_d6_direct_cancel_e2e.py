"""
E2E for D6 test plan #7: direct cancel propagates to the SDK.

Proves that when the LLM calls ``cancel_task`` on a running
async client_tool, the SDK's D6 lifecycle cancels the local
``asyncio.Task`` running the tool body — the body's
``except asyncio.CancelledError`` branch fires, and the body
never returns normally.

Without the SDK-side SSE handling:
- ``CancelTaskTool`` still emits ``response.client_task.cancel``
  (committed earlier),
- The SDK's ``stream()`` sees the event and calls
  ``state.asyncio_task.cancel()``,
- The running body (a blocking ``time.sleep`` inside the
  asyncio task) would otherwise run to its full duration.

Without the fix this test would time out / take the full
30 s sleep; with the fix the body's ``except`` fires within
a couple seconds of the cancel_task call.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_d6_direct_cancel_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from agent_plane_client import AgentPlaneClient
from agent_plane_client._events import (
    MessageDone,
    ResponseCompleted,
    ResponseFailed,
    ResponseIncomplete,
)
from agent_plane_client.tools import build_tool_handler, tool

from tests.e2e.conftest import upload_agent

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_FIXTURE = _FIXTURES_DIR / "d6-direct-cancel-test"

# The @tool body sleeps this long. Test must wall-clock
# complete in well under this — proving cancellation short-
# circuited the sleep.
_BODY_SLEEP_S = 30

# Upper bound on total wall-clock. LLM round-trip (~3s) +
# tool dispatch + cancel_task call + PATCH round-trip + LLM
# final response (~3s) ≈ 10s comfortable ceiling. Anything
# approaching _BODY_SLEEP_S means the body wasn't cancelled
# and the sleep ran to completion.
_MAX_WALL_CLOCK_S = 15.0


@pytest.fixture(scope="session")
def direct_cancel_test_agent(http_client: httpx.Client) -> str:
    """Upload the d6-direct-cancel-test fixture."""
    return upload_agent(http_client, _FIXTURE)


# Cancellation fingerprint. The body appends one of three
# outcomes per invocation so the assertions can tell:
#
# - "completed_normally" — body's sleep ran to end (bad).
# - "cancelled_mid_sleep" — asyncio.CancelledError raised
#   during the sleep (good — SDK cancelled the body).
# - "other_exception:<repr>" — something else went wrong.
#
# Module-level rather than fixture-scoped so the @tool fn
# can append without argument plumbing.
_body_outcomes: list[str] = []


@tool(synchronous=False)
async def slow_compute(seconds: int) -> str:
    """Sleep for ``seconds`` seconds, then return a marker.

    The body is async so ``asyncio.sleep`` (not ``time.sleep``)
    is the wait primitive — this is what makes cancellation
    observable: ``asyncio.sleep`` raises CancelledError the
    moment its surrounding Task is cancelled. A blocking
    ``time.sleep`` would finish its full duration no matter
    what the enclosing Task tried to do.

    Args:
        seconds: Sleep duration. The agent's AGENTS.md tells
            it to always pass 30, long enough that the test
            can observe cancellation without racing.
    """
    try:
        await asyncio.sleep(seconds)
        _body_outcomes.append("completed_normally")
        return f"slept-{seconds}"
    except asyncio.CancelledError:
        _body_outcomes.append("cancelled_mid_sleep")
        raise
    except BaseException as exc:  # noqa: BLE001 — body boundary
        _body_outcomes.append(f"other_exception:{exc!r}")
        raise


@pytest.mark.asyncio
async def test_sdk_cancels_local_body_on_llm_cancel_task(
    live_server: str,
    direct_cancel_test_agent: str,
) -> None:
    """
    When the LLM calls ``cancel_task`` on an in-flight async
    client tool, the SDK must cancel the local ``asyncio.Task``
    running the tool body — the body's ``except
    CancelledError`` branch fires, and the body never returns
    normally.

    Failure modes this test catches:

    - ``CancelTaskTool`` doesn't emit
      ``response.client_task.cancel`` — the SDK never sees
      the cancel signal, the body runs to full duration
      (30 s) and total wall-clock trips ``_MAX_WALL_CLOCK_S``.
    - SDK's ``stream()`` receives ``ClientTaskCancel`` but
      doesn't look up the right ``_AsyncToolState`` by
      ``task_id`` — body uncancelled, same symptom.
    - ``state.asyncio_task.cancel()`` fires but ``asyncio``
      can't find a cancellation point (e.g. body uses
      ``time.sleep``) — this test specifically uses
      ``asyncio.sleep`` so the cancellation is observable;
      fan-out covers the blocking-body case separately.
    """
    _body_outcomes.clear()
    handler = build_tool_handler([slow_compute])

    start = time.monotonic()
    async with AgentPlaneClient(base_url=live_server) as client:
        terminal_status: str | None = None
        final_text_chunks: list[str] = []

        async for event in client.responses.stream(
            model=direct_cancel_test_agent,
            input=(
                "Run the slow_compute+cancel_task sequence from your "
                "instructions. Don't skip cancel_task, and don't wait "
                "for slow_compute to finish."
            ),
            tool_handler=handler,
        ):
            if isinstance(event, MessageDone):
                for block in event.content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text") or ""
                        if isinstance(text, str):
                            final_text_chunks.append(text)
            elif isinstance(event, ResponseCompleted):
                terminal_status = "completed"
            elif isinstance(event, ResponseFailed):
                terminal_status = "failed"
                err = event.response.error
                if err is not None:
                    final_text_chunks.append(f"[FAILED] {err!r}")
            elif isinstance(event, ResponseIncomplete):
                terminal_status = "incomplete"

    elapsed = time.monotonic() - start

    assert terminal_status == "completed", (
        f"Direct-cancel flow should terminate cleanly; got "
        f"terminal_status={terminal_status!r}. "
        f"final_text_chunks={final_text_chunks!r}"
    )

    # Load-bearing: the body did NOT complete normally. If the
    # SDK had ignored the cancellation, the asyncio.sleep(30)
    # would have run to end and the outcome would be
    # "completed_normally".
    assert "completed_normally" not in _body_outcomes, (
        f"Body completed its full {_BODY_SLEEP_S}s sleep — SDK "
        f"did not propagate the LLM's cancel_task to the local "
        f"asyncio.Task. Outcomes: {_body_outcomes!r}"
    )

    # At least one cancellation observed. Could be more than
    # one if the LLM retries (shouldn't happen with our
    # AGENTS.md but guard against it).
    cancelled_count = _body_outcomes.count("cancelled_mid_sleep")
    assert cancelled_count >= 1, (
        f"Expected at least one slow_compute invocation to raise "
        f"asyncio.CancelledError during its sleep; got {_body_outcomes!r}. "
        f"If the body raised some other exception, the SDK's "
        f"cancel path is firing something unexpected."
    )

    # End-to-end wall-clock. The body's sleep is 30s; if
    # cancellation worked, total should be well under that.
    assert elapsed < _MAX_WALL_CLOCK_S, (
        f"Total stream duration {elapsed:.1f}s exceeds the "
        f"ceiling {_MAX_WALL_CLOCK_S}s. The body's sleep is "
        f"{_BODY_SLEEP_S}s; anything approaching that means "
        f"cancellation didn't fire and the sleep ran to the end."
    )
