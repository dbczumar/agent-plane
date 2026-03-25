"""Shared helper functions for server integration tests."""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from typing import Any

import httpx
import yaml


@dataclass
class ApiResponse:
    """Result of an API call — avoids returning positional tuples."""

    status_code: int
    # Any: JSON response bodies are inherently heterogeneous dicts.
    body: dict[str, Any]


def build_agent_bundle(
    name: str,
    description: str | None = None,
) -> bytes:
    """
    Build a minimal valid agent bundle (tar.gz) for testing.

    The bundle contains a single config.yaml with the given spec fields.
    """
    # Any: YAML config values are heterogeneous (str, int, etc.)
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": name,
        # LLM config is required for the real workflow to execute.
        # The model value must match the agent name used in
        # create_test_response(model=...).
        "llm": {"model": name},
    }
    if description is not None:
        config["description"] = description
    config_bytes = yaml.dump(config).encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


async def create_test_agent(
    client: httpx.AsyncClient,
    name: str = "test-agent",
    description: str | None = None,
) -> dict[str, Any]:
    """Create an agent via the API and return the response JSON."""
    bundle = build_agent_bundle(name=name, description=description)
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
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
