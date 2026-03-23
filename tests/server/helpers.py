"""Shared helper functions for server integration tests."""

from __future__ import annotations

from typing import Any

import httpx


async def create_test_agent(
    client: httpx.AsyncClient,
    name: str = "test-agent",
    description: str | None = None,
) -> dict[str, Any]:
    """Create an agent via the API and return the response JSON."""
    data: dict[str, str] = {"name": name}
    if description is not None:
        data["description"] = description
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", b"fake bundle", "application/gzip")},
        data=data,
    )
    assert resp.status_code == 201
    return resp.json()


async def create_test_response(
    client: httpx.AsyncClient,
    model: str = "test-agent",
    input_text: str = "Hello",
    background: bool = True,
    stream: bool = False,
    **kwargs: Any,
) -> tuple[int, dict[str, Any]]:
    """
    Create a response via the API and return (status_code, response JSON).

    Defaults to background=True so the endpoint returns immediately
    without blocking on task completion.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "background": background,
        "stream": stream,
        **kwargs,
    }
    resp = await client.post("/v1/responses", json=payload)
    return resp.status_code, resp.json()
