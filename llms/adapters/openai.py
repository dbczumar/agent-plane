"""
OpenAI and OpenAI-compatible provider adapter.

Handles OpenAI, Groq, DeepSeek, xAI, OpenRouter, and Ollama — any
provider that speaks the OpenAI Chat Completions API format.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

from llms.adapters.base import BaseAdapter
from llms.types import (
    FunctionCallOutput,
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseReasoningStartedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    Usage,
)

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

    def _build_headers(
        self,
        api_key_override: str | None = None,
    ) -> dict[str, str]:
        """
        Build HTTP headers for the request.

        :param api_key_override: Explicit API key to use instead of
            the environment variable. ``None`` falls back to env.
        :returns: Headers dict with Authorization if an API key is
            available.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = api_key_override or self._get_api_key()
        if api_key:
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
        *,
        connection_params: dict[str, str] | None = None,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """
        Send a Chat Completions request to the provider.

        :param messages: Chat Completions messages.
        :param model: Model name, e.g. ``"gpt-5.4"``.
        :param tools: Tool schemas or ``None``.
        :param stream: Enable streaming.
        :param extra: Additional kwargs.
        :param connection_params: Per-call overrides. Supported keys:
            ``"api_key"``, ``"base_url"``.
        :returns: Response dict or iterator of chunk dicts.
        """
        params = connection_params or {}
        payload = self._build_payload(messages, model, tools, stream, extra)
        override_base = params.get("base_url")
        effective_base = override_base.rstrip("/") if override_base else self._base_url
        url = f"{effective_base}/chat/completions"
        headers = self._build_headers(api_key_override=params.get("api_key"))

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
            result: dict[str, Any] = resp.json()
            return result

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
            with client.stream("POST", url, headers=headers, json=payload) as resp:
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
    data = line[len("data: ") :]
    if data.strip() == "[DONE]":
        return None
    result: dict[str, Any] = json.loads(data)
    return result


def _parse_responses_output(
    output_items: list[dict[str, Any]],
) -> list[MessageOutput | FunctionCallOutput]:
    """
    Convert Responses API output items to ``llms.types`` output objects.

    Skips ``reasoning`` items — only ``message`` and ``function_call``
    items are returned.

    :param output_items: List of output item dicts from the Responses
        API response, e.g. the ``response.output`` list.
    :returns: List of :class:`MessageOutput` and/or
        :class:`FunctionCallOutput` instances.
    """
    output: list[MessageOutput | FunctionCallOutput] = []
    for item in output_items:
        if item.get("type") == "message":
            parts = [
                OutputText(text=p["text"])
                for p in item.get("content", [])
                if p.get("type") == "output_text" and p.get("text")
            ]
            if parts:
                output.append(MessageOutput(content=parts))
        elif item.get("type") == "function_call":
            output.append(
                FunctionCallOutput(
                    call_id=item["call_id"],
                    name=item["name"],
                    arguments=item["arguments"],
                )
            )
    return output


def _parse_responses_response(data: dict[str, Any]) -> Response:
    """
    Convert a Responses API response dict to a :class:`Response`.

    :param data: The full Responses API response JSON dict.
    :returns: A :class:`Response` with parsed output and usage.
    """
    output = _parse_responses_output(data.get("output", []))
    usage_data: dict[str, Any] = data.get("usage") or {}
    usage = (
        Usage(
            input_tokens=usage_data.get("input_tokens"),
            output_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )
        if usage_data
        else None
    )
    return Response(output=output, model=data.get("model", ""), usage=usage)


def _parse_responses_event(
    event_type: str,
    data: dict[str, Any],
) -> ResponseStreamEvent | None:
    """
    Convert a single Responses API SSE event to a
    :class:`ResponseStreamEvent`, or ``None`` if the event type
    is not handled.

    :param event_type: The SSE event name, e.g.
        ``"response.output_text.delta"``.
    :param data: The parsed JSON payload from the ``data:`` line.
    :returns: A streaming event dataclass, or ``None``.
    """
    if event_type == "response.output_text.delta":
        return ResponseTextDeltaEvent(delta=data["delta"])
    if event_type == "response.reasoning_summary_text.delta":
        return ResponseReasoningSummaryTextDeltaEvent(delta=data["delta"])
    if event_type == "response.reasoning_text.delta":
        return ResponseReasoningTextDeltaEvent(delta=data["delta"])
    if event_type == "response.output_item.added":
        if data.get("item", {}).get("type") == "reasoning":
            return ResponseReasoningStartedEvent()
    if event_type == "response.completed":
        return ResponseCompletedEvent(response=_parse_responses_response(data["response"]))
    return None


class OpenAIAdapter(OpenAICompatibleAdapter):
    """
    OpenAI-specific adapter that calls ``/v1/responses`` natively.

    Extends :class:`OpenAICompatibleAdapter` (which uses Chat
    Completions) by adding :meth:`responses_create` — a direct
    Responses API path that preserves reasoning token streaming events
    that Chat Completions does not expose.

    :param base_url: The OpenAI API base URL.
    :param api_key_env: Environment variable name for the API key.
    """

    def responses_create(
        self,
        *,
        input: list[dict[str, Any]],  # noqa: A002 — mirrors OpenAI SDK parameter name
        instructions: str | None,
        model: str,
        tools: list[dict[str, Any]] | None,
        reasoning: dict[str, str] | None,
        stream: bool,
        connection_params: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Response | Iterator[ResponseStreamEvent]:
        """
        Call the OpenAI Responses API (``/v1/responses``) directly.

        Used instead of Chat Completions so that reasoning token
        streaming events (``response.reasoning_summary_text.delta``,
        ``response.reasoning_text.delta``) flow through unmodified.

        :param input: Responses API input items.
        :param instructions: System instructions string, or ``None``.
        :param model: Model name without provider prefix, e.g.
            ``"o4-mini"``.
        :param tools: OpenAI-format tool schemas, or ``None``.
        :param reasoning: Reasoning config dict, e.g.
            ``{"effort": "high", "summary": "detailed"}``, or ``None``.
        :param stream: If ``True``, return an iterator of
            :class:`ResponseStreamEvent`. If ``False``, return a
            :class:`Response`.
        :param connection_params: Per-call overrides. Supported keys:
            ``"api_key"``, ``"base_url"``.
        :param kwargs: Additional API kwargs (temperature, etc.).
        :returns: A :class:`Response` or an iterator of
            :class:`ResponseStreamEvent`.
        """
        params = connection_params or {}
        payload: dict[str, Any] = {"model": model, "input": input, **kwargs}
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = tools
        if reasoning:
            payload["reasoning"] = reasoning
        if stream:
            payload["stream"] = True

        override_base = params.get("base_url")
        effective_base = override_base.rstrip("/") if override_base else self._base_url
        url = f"{effective_base}/responses"
        headers = self._build_headers(api_key_override=params.get("api_key"))

        if stream:
            return self._stream_responses(url, headers, payload)
        resp_data = self._send_request(url, headers, payload)
        return _parse_responses_response(resp_data)

    def _stream_responses(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Iterator[ResponseStreamEvent]:
        """
        Stream the Responses API and yield typed
        :class:`ResponseStreamEvent` instances.

        Parses SSE ``event:`` + ``data:`` pairs, mapping each to the
        appropriate event dataclass. Unknown event types are skipped.

        :param url: The ``/v1/responses`` endpoint URL.
        :param headers: HTTP headers including Authorization.
        :param payload: The request payload with ``stream: true``.
        :yields: :class:`ResponseStreamEvent` instances.
        """
        current_event: str | None = None
        buf = ""
        with httpx.Client(timeout=_STREAM_TIMEOUT) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.rstrip("\r")
                        if line.startswith("event: "):
                            current_event = line[7:]
                        elif line.startswith("data: ") and current_event:
                            data_str = line[6:]
                            if data_str.strip() != "[DONE]":
                                event = _parse_responses_event(current_event, json.loads(data_str))
                                if event is not None:
                                    yield event
                            current_event = None
