"""
OpenAI and OpenAI-compatible provider adapter.

Handles OpenAI, Groq, DeepSeek, xAI, OpenRouter, and Ollama — any
provider that speaks the OpenAI Chat Completions API format.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx

from llms.adapters.base import BaseAdapter

# Timeout for non-streaming requests (seconds)
_REQUEST_TIMEOUT = 120

# Timeout for streaming connection (seconds)
_STREAM_TIMEOUT = 300


class OpenAICompatibleAdapter(BaseAdapter):
    """
    Adapter for providers using the OpenAI Chat Completions format.

    :param base_url: The provider's API base URL, e.g.
        ``"https://api.openai.com/v1"``.
    :param api_key_env: Environment variable name for the API key,
        e.g. ``"OPENAI_API_KEY"``. ``None`` if no auth needed.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None,
    ) -> None:
        # Normalize so f"{base_url}/chat/completions" never double-slashes
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env

    def _get_api_key(self) -> str | None:
        """
        Read the API key from the environment.

        :returns: The API key string, or ``None`` if no key is
            configured.
        """
        if self._api_key_env is None:
            return None
        return os.environ.get(self._api_key_env)

    def _build_headers(self) -> dict[str, str]:
        """
        Build HTTP headers for the request.

        :returns: Headers dict with Authorization if an API key is
            available.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key := self._get_api_key():
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build the Chat Completions request payload.

        :param messages: Chat Completions messages.
        :param model: Model name without provider prefix.
        :param tools: Tool schemas or ``None``.
        :param stream: Whether to enable streaming.
        :param extra: Additional kwargs (temperature, etc.).
        :returns: The request payload dict.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **extra,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
        return payload

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """
        Send a Chat Completions request to the provider.

        :param messages: Chat Completions messages.
        :param model: Model name, e.g. ``"gpt-5.4"``.
        :param tools: Tool schemas or ``None``.
        :param stream: Enable streaming.
        :param extra: Additional kwargs.
        :returns: Response dict or iterator of chunk dicts.
        """
        payload = self._build_payload(messages, model, tools, stream, extra)
        url = f"{self._base_url}/chat/completions"
        headers = self._build_headers()

        if stream:
            return self._stream_request(url, headers, payload)
        return self._send_request(url, headers, payload)

    def _send_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a non-streaming HTTP POST and return the JSON response.

        :param url: The full endpoint URL.
        :param headers: HTTP headers.
        :param payload: JSON payload.
        :returns: Parsed JSON response dict.
        :raises httpx.HTTPStatusError: On non-2xx status.
        """
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _stream_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """
        Send a streaming HTTP POST and yield parsed SSE data chunks.

        :param url: The full endpoint URL.
        :param headers: HTTP headers.
        :param payload: JSON payload with ``stream: true``.
        :returns: Iterator of parsed Chat Completions chunk dicts.
        """
        with httpx.Client(timeout=_STREAM_TIMEOUT) as client:
            with client.stream(
                "POST", url, headers=headers, json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    parsed = _parse_sse_line(line)
                    if parsed is not None:
                        yield parsed


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """
    Parse a single SSE line into a data dict.

    Ignores non-data lines (event:, id:, comments) and the
    ``[DONE]`` sentinel.

    :param line: A raw SSE line, e.g. ``"data: {\"id\": ...}"``.
    :returns: Parsed JSON dict, or ``None`` if the line should
        be skipped.
    """
    if not line.startswith("data: "):
        return None
    data = line[len("data: "):]
    if data.strip() == "[DONE]":
        return None
    return json.loads(data)
