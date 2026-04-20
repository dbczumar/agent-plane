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
    sub_agents: list[dict[str, Any]] | None = None,
    max_iterations: int | None = None,
) -> bytes:
    """
    Build a minimal valid agent bundle (tar.gz) for testing.

    The bundle contains a single config.yaml with the given spec
    fields. When ``sub_agents`` is provided, each entry is added as
    ``agents/<name>/config.yaml`` and the parent's
    ``tools.agents`` list is populated.

    :param name: Agent name, e.g. ``"test-agent"``.
    :param description: Optional description.
    :param sub_agents: Optional list of sub-agent config dicts.
        Each must have at least a ``"name"`` key, e.g.
        ``[{"name": "researcher", "description": "..."}]``.
    :param max_iterations: Optional override for
        ``executor.max_iterations`` — useful for tests that want
        to force an ``incomplete`` terminal state after a known
        number of LLM turns. ``None`` uses the spec default.
    """
    # Any: YAML config values are heterogeneous (str, int, etc.)
    config: dict[str, Any] = {
        "spec_version": 1,
        "name": name,
        # LLM config is required for the real workflow to execute.
        # The model value must match the agent name used in
        # create_test_response(model=...).
        "llm": {
            "model": name,
            # api_key is required by spec validation; the workflow
            # uses the mock LLM client so it's never actually sent.
            "connection": {"api_key": "test-key"},
        },
    }
    if description is not None:
        config["description"] = description
    if max_iterations is not None:
        config["executor"] = {"max_iterations": max_iterations}
    if sub_agents:
        config["tools"] = {
            "agents": [sa["name"] for sa in sub_agents],
        }
    config_bytes = yaml.dump(config).encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tf.addfile(info, io.BytesIO(config_bytes))
        # Add sub-agent config files
        for sa in sub_agents or []:
            sa_config: dict[str, Any] = {
                "spec_version": 1,
                "name": sa["name"],
                "llm": {
                    "model": sa["name"],
                    "connection": {"api_key": "test-key"},
                },
            }
            if "description" in sa:
                sa_config["description"] = sa["description"]
            sa_bytes = yaml.dump(sa_config).encode()
            sa_info = tarfile.TarInfo(
                name=f"agents/{sa['name']}/config.yaml",
            )
            sa_info.size = len(sa_bytes)
            tf.addfile(sa_info, io.BytesIO(sa_bytes))
    return buf.getvalue()


async def create_test_agent(
    client: httpx.AsyncClient,
    name: str = "test-agent",
    description: str | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Create an agent via the API and return the response JSON."""
    bundle = build_agent_bundle(
        name=name,
        description=description,
        max_iterations=max_iterations,
    )
    resp = await client.post(
        "/api/agents",
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"create_test_agent failed: {resp.text}"
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
    tools: list[dict[str, Any]] | None = None,
) -> ApiResponse:
    """
    Create a response via the API and return an ApiResponse.

    Defaults to background=True so the endpoint returns immediately
    without blocking on task completion.

    :param tools: Optional list of client-side tool schemas in
        standard OpenAI function format, e.g.
        ``[{"type": "function", "function": {"name": "get_weather", ...}}]``.
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
    if tools is not None:
        payload["tools"] = tools
    resp = await client.post("/v1/responses", json=payload)
    return ApiResponse(status_code=resp.status_code, body=resp.json())
