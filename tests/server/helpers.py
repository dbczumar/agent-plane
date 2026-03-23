"""Shared helper functions for server integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiResponse:
    """Result of an API call — avoids returning positional tuples."""

    status_code: int
    # Any: JSON response bodies are inherently heterogeneous dicts.
    body: dict[str, Any]


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
    instructions: str | None = None,
    previous_response_id: str | None = None,
    store: bool | None = None,
    conversation: dict[str, str] | None = None,
    reasoning: dict[str, str] | None = None,
) -> ApiResponse:
    """
    Create a response via the API and return an ApiResponse.

    Defaults to background=True so the endpoint returns immediately
    without blocking on task completion.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "background": background,
        "stream": stream,
    }
    if instructions is not None:
        payload["instructions"] = instructions
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    if store is not None:
        payload["store"] = store
    if conversation is not None:
        payload["conversation"] = conversation
    if reasoning is not None:
        payload["reasoning"] = reasoning
    resp = await client.post("/v1/responses", json=payload)
    return ApiResponse(status_code=resp.status_code, body=resp.json())
