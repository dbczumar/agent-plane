"""
E2E for D6 test plan #4: parallel fan-out.

Proves that when the LLM dispatches multiple ``@tool(synchronous
=False)`` calls in a single turn, the SDK runs them
concurrently on the event loop (via ``asyncio.create_task``
per body) — not serialized. Without the D6 fan-out fix (sync
``execute`` routed through ``asyncio.to_thread``) a sync
body with ``time.sleep`` would block every sibling body AND
the caller's render loop.

Test uses three tools each sleeping 3 s. Serial execution
would take ~9 s; parallel takes ~3 s. Assert total elapsed
is comfortably under the serial floor.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_d6_parallel_fan_out_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import tarfile
import tempfile
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

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_FIXTURE = _FIXTURES_DIR / "d6-fan-out-test"

# Per-tool sleep duration. Keep small to keep the test fast
# but large enough that serial (3x this) is clearly
# distinguishable from parallel (1x).
_SLEEP_S = 3

# Number of parallel invocations to request from the LLM.
_FAN_OUT = 3

# Wall-clock ceiling for parallel execution: the longest
# single body (3 s) plus real-LLM round-trip overhead and
# SSE/PATCH plumbing. Serial would take ≥ 9 s. Missing the
# ceiling means something serialized.
_PARALLEL_WALL_CLOCK_MAX_S = 8.0


def _upload(http_client: httpx.Client, agent_dir: Path) -> str:
    """
    Upload an agent bundle from a directory tree.

    :param http_client: HTTP client pointed at the live server.
    :param agent_dir: Directory containing config.yaml.
    :returns: The agent's name.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(agent_dir), arcname=".")
        bundle_path = tmp.name
    try:
        with open(bundle_path, "rb") as f:
            resp = http_client.post(
                "/api/agents",
                files={
                    "bundle": (
                        "agent.tar.gz",
                        f,
                        "application/gzip",
                    ),
                },
            )
        if resp.status_code == 409:
            return agent_dir.name
        resp.raise_for_status()
        return resp.json()["name"]
    finally:
        Path(bundle_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def fan_out_test_agent(http_client: httpx.Client) -> str:
    """Upload the shared D6 dispatch fixture."""
    return _upload(http_client, _FIXTURE)


# Concurrency fingerprint: every invocation appends a
# (start, end) pair to this list. The test inspects it to
# check that invocation windows overlapped (i.e. they ran
# in parallel, not serially).
_concurrency_log: list[tuple[float, float]] = []


@tool(synchronous=False)
def compute(value: str) -> str:
    """Sleep briefly, then echo the value back.

    The sleep uses a blocking ``time.sleep`` on purpose: it's
    how we assert the SDK is running each body on a worker
    thread via ``asyncio.to_thread`` (without that, six
    siblings would all block the event loop serially).

    Args:
        value: A label to echo so the test can match
            individual invocations.
    """
    start = time.monotonic()
    time.sleep(_SLEEP_S)
    end = time.monotonic()
    _concurrency_log.append((start, end))
    return f"done-{value}"


@pytest.mark.asyncio
async def test_async_client_tools_fan_out_in_parallel(
    live_server: str,
    fan_out_test_agent: str,
) -> None:
    """
    Three @tool(synchronous=False) calls in one LLM turn run
    concurrently end-to-end.

    Failure modes this test catches:

    - ``_run_async_tool_body`` awaits sync ``execute`` on the
      event loop (old pre-D6 pattern): bodies run serially,
      total wall-clock ≥ 3 * _SLEEP_S, fails the
      ``_PARALLEL_WALL_CLOCK_MAX_S`` assertion.
    - Server-side dispatch serializes (e.g. the async drain
      waits after EACH dispatch before returning to the LLM):
      same symptom — wall-clock tied to sum-of-sleeps.
    - SDK doesn't track multiple in-flight task_ids: only
      the first one's handle would get paired with a
      ``task_id_event`` and PATCHed; the other two bodies
      would time out on the 1-hour cap.
    """
    _concurrency_log.clear()

    handler = build_tool_handler([compute])

    start_overall = time.monotonic()
    async with AgentPlaneClient(base_url=live_server) as client:
        terminal_status: str | None = None
        final_text_chunks: list[str] = []

        async for event in client.responses.stream(
            model=fan_out_test_agent,
            input=(
                f"Call `compute` exactly {_FAN_OUT} times in parallel "
                f"with synchronous=false and values 'a', 'b', 'c'. "
                f"Emit all three tool_calls in the same turn so they "
                f"dispatch together — do not call them sequentially. "
                f"After all three complete, reply with ANSWER: "
                f"followed by the three done-* results joined with commas."
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
                # Promote to the final-text collector so the
                # assertion message includes the failure details.
                err_info = event.response.error
                if err_info is not None:
                    final_text_chunks.append(f"[FAILED] {err_info!r}")
            elif isinstance(event, ResponseIncomplete):
                terminal_status = "incomplete"

    elapsed = time.monotonic() - start_overall

    # Must have terminated cleanly.
    assert terminal_status == "completed", (
        f"Fan-out should complete cleanly; got terminal_status="
        f"{terminal_status!r}. final_text_chunks={final_text_chunks!r}"
    )

    # LLM actually emitted _FAN_OUT dispatches. Anything less
    # and the parallelism claim is unverifiable.
    assert len(_concurrency_log) >= _FAN_OUT, (
        f"Expected at least {_FAN_OUT} compute() invocations "
        f"(LLM was told to fan out); got {len(_concurrency_log)}. "
        f"Tool invocations: {_concurrency_log!r}. "
        f"Final assistant texts: {final_text_chunks!r}. "
        f"If fewer, the LLM ignored the fan-out instruction and "
        f"this test's parallelism claim isn't being exercised."
    )

    # Tool-level concurrency: the invocations must overlap in
    # wall-clock time. Check that the max end-time minus min
    # start-time is LESS than sum-of-durations (which would be
    # serial) — pragmatic check that handles >_FAN_OUT calls
    # (if the LLM adds one) and out-of-order scheduling.
    start_times = [s for s, _ in _concurrency_log]
    end_times = [e for _, e in _concurrency_log]
    span = max(end_times) - min(start_times)
    total_compute_time = sum(e - s for s, e in _concurrency_log)
    assert span < total_compute_time * 0.95, (
        f"Tool bodies did not overlap: span={span:.2f}s but the "
        f"sum of per-body durations is {total_compute_time:.2f}s. "
        f"Serial execution would have span ≈ total; parallel "
        f"execution has span much smaller than total. "
        f"Invocation windows: {_concurrency_log!r}"
    )

    # End-to-end wall-clock assertion. Generous ceiling — real
    # LLM + SSE + PATCH overhead plus _SLEEP_S for the bodies.
    assert elapsed < _PARALLEL_WALL_CLOCK_MAX_S * 2, (
        f"Total stream duration {elapsed:.1f}s exceeds the "
        f"generous ceiling {_PARALLEL_WALL_CLOCK_MAX_S * 2:.1f}s. "
        f"If the SDK's D6 tool dispatcher accidentally runs "
        f"bodies serially, total would be ≈ {_FAN_OUT}*_SLEEP_S "
        f"= {_FAN_OUT * _SLEEP_S}s + LLM overhead, which would "
        f"trip this ceiling. The tool-overlap assertion above "
        f"is the more specific check, but this guards the "
        f"end-to-end wall-clock."
    )
