"""Test tools for the async-tool E2E suite.

Two ``@tool(synchronous=False)`` functions and one sync ``@tool`` so the
fixture agent can drive the dispatch + drain pipeline against a real LLM.
The async tools sleep briefly so the dispatch handle and the auto-
delivered result are observable as distinct events; the markers
returned are deliberately distinctive substrings so the assertion
``"FOO_MARKER_42" in final_text`` is unambiguous.
"""

from __future__ import annotations

import time

from agent_plane_client import tool


@tool(synchronous=False)
def delayed_echo(label: str) -> str:
    """
    Sleep briefly, then echo the label inside an unambiguous marker.

    The 2-second sleep makes the dispatch ↔ auto-delivery sequence
    observable end-to-end without slowing the test materially. The
    ``ECHO_FROM_ASYNC[...]`` wrapper is the substring the test
    asserts on — distinctive enough that paraphrasing or
    hallucination is detectable.

    Args:
        label: Text to echo back, e.g. ``"alpha"``.
    """
    time.sleep(2)
    return f"ECHO_FROM_ASYNC[{label}]"


@tool(synchronous=False)
def boom_async() -> str:
    """
    Always raise so the failure path of the async pipeline is exercised.

    The marker string in the exception message is asserted on by the
    test — proves the failure traceback survived truncation and the
    drain.

    Args:
        (no arguments)
    """
    raise RuntimeError("ASYNC_TOOL_BOOM_MARKER")


@tool
def count_chars(text: str) -> int:
    """
    Return the literal character count of ``text``.

    Synchronous on purpose so the mixed-tool E2E test can prove that
    the same agent runs both kinds in one turn.

    Args:
        text: Text to measure.
    """
    return len(text)
