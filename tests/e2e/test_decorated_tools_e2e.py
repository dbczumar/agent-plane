"""End-to-end tests for the @tool decorator on real LLM + real server.

Verifies the full pipeline:
- Agent image with ``@tool``-decorated functions in
  ``tools/python/*.py`` is loaded into a real ``ap server``.
- Real LLM calls each tool with arguments inferred from the
  derived schema.
- Tool runs in a subprocess; result returns through the runner
  and gets persisted in the conversation.
- Final LLM response references the literal output values.

These tests require an LLM API key and a working ``ap`` CLI on
PATH; they are excluded from the default ``pytest`` run via
``--ignore=tests/e2e``.

**TUI verification** (mandatory per CLAUDE.md before merge):

- archer's word_count: ``python examples/frontends/terminal.py
  examples/agents/archer/`` then ask "Count the words in this
  paragraph: <text>".
- decorator-signatures-test: ``python examples/frontends/terminal.py
  tests/_fixtures/agents/decorator-signatures-test/`` then ask
  "Greet Alice, format a record for Bob age 42, and compute
  with value 5".
"""

from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path

import httpx
import pytest

_DECORATOR_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "decorator-signatures-test"
)


@pytest.fixture(scope="session")
def decorator_signatures_agent(http_client: httpx.Client) -> str:
    """
    Upload the decorator-signatures-test fixture agent.

    :param http_client: HTTP client pointed at the live server.
    :returns: The agent's name (matches its config.yaml ``name``).
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(_DECORATOR_FIXTURE_DIR), arcname=".")
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
            # Already registered from a prior test run in the same session.
            return _DECORATOR_FIXTURE_DIR.name
        resp.raise_for_status()
        return resp.json()["name"]
    finally:
        Path(bundle_path).unlink(missing_ok=True)


def _create_response_blocking(
    http_client: httpx.Client,
    *,
    model: str,
    user_text: str,
    timeout_s: float = 120.0,
) -> dict:
    """
    POST a response, poll until terminal, return the final body.

    Uses the polling API rather than streaming SSE — simpler to
    consume in a test, even though it's a different code path
    than the TUI. Per CLAUDE.md, the TUI must be manually verified
    before merging this test (see module docstring).

    :param http_client: HTTP client pointed at the live server.
    :param model: Agent name to invoke.
    :param user_text: Plain-text input message for the agent.
    :param timeout_s: Max seconds to wait for the response to
        complete.
    :returns: The final response JSON.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": model,
            "input": user_text,
            "background": True,
            "store": True,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        get_resp = http_client.get(f"/v1/responses/{response_id}")
        get_resp.raise_for_status()
        body = get_resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(1.0)
    raise AssertionError(
        f"Response {response_id} did not complete within {timeout_s}s; "
        f"final status was {body.get('status')!r}."
    )


def _final_text(response_body: dict) -> str:
    """
    Extract the assistant's final text from a response.

    Walks ``output`` items, picks message items with role
    ``"assistant"``, and concatenates their ``output_text`` blocks.

    :param response_body: The response JSON returned from
        ``GET /v1/responses/{id}``.
    :returns: Concatenated assistant text. Empty string if no
        assistant message exists (which a passing test would
        catch via the content assertions).
    """
    parts: list[str] = []
    for item in response_body.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n\n".join(parts)


def test_archer_word_count_e2e(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    archer + migrated ``word_count`` produces a correct count.

    Phrase chosen so the count is unambiguous: 7 words.
    """
    body = _create_response_blocking(
        http_client,
        model=archer_agent,
        user_text=(
            "Use the word_count tool to count the words in "
            "exactly this phrase: 'one two three four five six seven'. "
            "Tell me the number."
        ),
    )
    assert body["status"] == "completed", (
        f"archer turn did not complete: status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    # The literal count must appear in the LLM's final response.
    # If "7" is missing, either word_count returned the wrong number
    # or the LLM didn't surface the result.
    assert "7" in final, f"Expected the count '7' in the final response, got: {final!r}"


def test_decorated_tools_varied_signatures_e2e(
    http_client: httpx.Client,
    decorator_signatures_agent: str,
) -> None:
    """
    The decorator-signatures-test agent calls all three tools and
    surfaces literal output from each.

    Exercises:
    - Primitive arg (greet name='Alice').
    - Pydantic BaseModel arg (format_record name='Bob' age=42).
    - Multiple primitives + Annotated description (compute value=5).
    """
    body = _create_response_blocking(
        http_client,
        model=decorator_signatures_agent,
        user_text=(
            "Call all three tools: "
            "greet with name='Alice', "
            "format_record with name='Bob' age=42 (no email), "
            "and compute with value=5 (use the default multiplier). "
            "Then in your final response include the literal output "
            "values from each tool so I can verify them."
        ),
    )
    assert body["status"] == "completed", (
        f"signatures-test turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    # Greet output: must contain "Alice" (literal name).
    assert "Alice" in final, (
        f"Final response missing 'Alice' from greet — either the tool "
        f"wasn't called or its result didn't surface. Got: {final!r}"
    )
    # format_record output: must contain "Bob" and "42".
    assert "Bob" in final, f"Missing 'Bob' from format_record. Got: {final!r}"
    assert "42" in final, f"Missing age '42' from format_record. Got: {final!r}"
    # compute output: must contain "10" (5 * 2 default multiplier).
    assert "10" in final, (
        f"Missing computed value '10' (5 * 2 default) — multiplier "
        f"default may not be honored, or compute wasn't called. "
        f"Got: {final!r}"
    )
