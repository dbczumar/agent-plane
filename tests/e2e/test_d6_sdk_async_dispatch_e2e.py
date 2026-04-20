"""
E2E for Phase 5 D6 — SDK-side async client tool lifecycle.

Proves the python-client SDK now drives the full async path
end-to-end without any caller bookkeeping:

1. SDK exposes ``@tool(synchronous=False)`` functions on the
   wire schema with ``parameters.properties.synchronous``.
2. Real LLM calls one with ``synchronous: false``.
3. Server dispatches a ``kind="client_tool"`` task and emits
   the handle FCO inline.
4. SDK detects the async dispatch, spawns the tool body on
   an ``asyncio.Task``, and PATCHes ``async_tool_results``
   with the body's return value when it completes — without
   the caller writing any of the dispatch / handle-parsing /
   PATCH machinery.
5. Server's drain delivers ``[System: task X (client_tool)
   completed]\\n<body>`` to the parent's conversation.
6. LLM reads the system message and replies
   ``ANSWER:<body>``.
7. Test asserts the body text round-tripped through both the
   tool and the drain.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_d6_sdk_async_dispatch_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import asyncio
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
_FIXTURE = _FIXTURES_DIR / "d6-sdk-async-dispatch-test"

# Marker the @tool body returns. The agent's AGENTS.md instructs
# the LLM to echo the marker back via ``ANSWER:<body>`` after
# the drain delivers it as a system message — finding the
# marker in the LLM's final assistant text proves the entire
# loop closed: SDK dispatch → server drain → LLM reads system
# message → SDK PATCH → drain delivery.
_MARKER = "D6_SDK_ASYNC_LIFECYCLE_OK_77"


@pytest.fixture(scope="session")
def d6_test_agent(http_client: httpx.Client) -> str:
    """Upload the D6 E2E fixture."""
    return upload_agent(http_client, _FIXTURE)


# Tool body — the SDK runs this in an asyncio.Task when the
# server dispatches the call as ``synchronous: false``. Returns
# ``value`` verbatim so the LLM's ANSWER:<body> can be matched
# against the input marker.
@tool(synchronous=False)
async def compute(value: str) -> str:
    """Echo the input string back asynchronously.

    Args:
        value: Marker to echo. Test asserts this is what the
            LLM ultimately replies with.
    """
    # Tiny await so the body is visibly async (not just sync
    # masquerading) — proves the asyncio.Task path is what
    # ran, not an inline-execute fallback.
    await asyncio.sleep(0.05)
    return value


@pytest.mark.asyncio
async def test_sdk_async_client_tool_completes_round_trip(
    live_server: str,
    d6_test_agent: str,
) -> None:
    """
    Full SDK-driven async-client-tool lifecycle, end-to-end.

    Failure modes this test catches:

    - SDK doesn't detect the async call (``_is_async_tool_call``
      returns False) → call lands in ``pending_client_calls``,
      the legacy sync flow runs, the test deadlocks waiting for
      the LLM to ANSWER (it would never get a system message).
    - SDK doesn't capture ``task_id`` from the handle FCO →
      the body's ``state.task_id_event`` never fires → the
      body's ``await asyncio.wait_for(state.task_id_event...)``
      times out → no PATCH is sent → server's holder workflow
      runs out the 1h cap (test would hit its own timeout
      first).
    - SDK PATCHes the wrong status / wrong topic → server's
      drain never delivers a ``completed`` system message →
      LLM has nothing to ANSWER with.
    - Server-side audit-fix-#1 routing regresses → drain
      message lands on the wrong agent → LLM never sees it.
    """
    handler = build_tool_handler([compute])

    async with AgentPlaneClient(base_url=live_server) as client:
        # Drive the stream to completion. Collect events so the
        # test can assert on terminal status + the assistant
        # message body that contains the marker.
        final_text_chunks: list[str] = []
        terminal_status: str | None = None
        failure_diag: str | None = None

        async for event in client.responses.stream(
            model=d6_test_agent,
            input=f"Compute on the value {_MARKER!r}.",
            tool_handler=handler,
        ):
            if isinstance(event, MessageDone):
                # Last assistant message in a turn — the LLM
                # may emit several turns (one to call the tool,
                # one to ANSWER). We collect every assistant
                # message's text for the assertion.
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
                failure_diag = repr(err)[:600] if err is not None else "no error info"
            elif isinstance(event, ResponseIncomplete):
                terminal_status = "incomplete"

    assert terminal_status == "completed", (
        f"D6 lifecycle should complete cleanly; got "
        f"terminal_status={terminal_status!r}, "
        f"failure_diag={failure_diag!r}, "
        f"final_text_chunks={final_text_chunks!r}"
    )

    # The LLM's reply should contain the marker (per the agent
    # AGENTS.md instructions: ANSWER:<body> where <body> is the
    # system message body, which is the tool's return value,
    # which is the marker). Anywhere across the turn chain is
    # fine — the SDK's outer ``while True`` may have spanned
    # several iterations.
    joined = "\n".join(final_text_chunks)
    assert _MARKER in joined, (
        f"D6 lifecycle round-trip failed: marker {_MARKER!r} not "
        f"found in any assistant message text. "
        f"final_text_chunks={final_text_chunks!r}. "
        f"\nIf the test hangs / times out instead of failing here, "
        f"the SDK probably didn't spawn the body or didn't PATCH — "
        f"see _run_async_tool_body in agent_plane_client/_responses.py."
    )
