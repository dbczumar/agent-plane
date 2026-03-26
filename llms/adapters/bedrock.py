"""
AWS Bedrock Converse API adapter.

Translates Chat Completions format to/from the Bedrock Converse API.
Uses ``boto3`` (lazy import) for AWS authentication and HTTP.
Ported from MLflow AI Gateway's Bedrock provider.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from llms.adapters.base import BaseAdapter


class BedrockAdapter(BaseAdapter):
    """
    Adapter for AWS Bedrock using the Converse API.

    Auth is handled via standard AWS environment variables
    (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``AWS_DEFAULT_REGION``) or an IAM role.
    """

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """
        Get or create the ``bedrock-runtime`` boto3 client.

        :returns: A boto3 ``bedrock-runtime`` client.
        """
        if self._client is None:
            import boto3  # type: ignore[import-untyped]

            self._client = boto3.client("bedrock-runtime")
        return self._client

    def _create_client_from_params(
        self,
        params: dict[str, str],
    ) -> Any:
        """
        Create a fresh boto3 client from explicit connection params.

        :param params: Connection overrides. Supported keys:
            ``"aws_region"``, ``"aws_access_key_id"``,
            ``"aws_secret_access_key"``, ``"aws_session_token"``.
        :returns: A boto3 ``bedrock-runtime`` client.
        """
        import boto3

        boto_kwargs: dict[str, str] = {}
        if region := params.get("aws_region"):
            boto_kwargs["region_name"] = region
        if access_key := params.get("aws_access_key_id"):
            boto_kwargs["aws_access_key_id"] = access_key
        if secret_key := params.get("aws_secret_access_key"):
            boto_kwargs["aws_secret_access_key"] = secret_key
        if session_token := params.get("aws_session_token"):
            boto_kwargs["aws_session_token"] = session_token
        return boto3.client("bedrock-runtime", **boto_kwargs)

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
        Send a request via Bedrock Converse API.

        :param messages: Chat Completions format messages.
        :param model: Bedrock model ID, e.g.
            ``"anthropic.claude-3-sonnet-20240229-v1:0"``.
        :param tools: Tool schemas or ``None``.
        :param stream: Enable streaming.
        :param extra: Additional kwargs (temperature, etc.).
        :param connection_params: Per-call overrides. Supported keys:
            ``"aws_region"``, ``"aws_access_key_id"``,
            ``"aws_secret_access_key"``, ``"aws_session_token"``.
        :returns: Chat Completions response dict or chunk iterator.
        """
        converse_kwargs = _build_converse_kwargs(messages, model, tools, extra)
        if connection_params:
            client = self._create_client_from_params(connection_params)
        else:
            client = self._get_client()

        if stream:
            return _stream_converse(client, converse_kwargs)
        return _send_converse(client, converse_kwargs)


# ── Request translation ───────────────────────────────────


def _build_converse_kwargs(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """
    Build kwargs for the Bedrock Converse API call.

    :param messages: Chat Completions messages.
    :param model: Bedrock model ID.
    :param tools: OpenAI-format tool schemas or ``None``.
    :param extra: Additional kwargs.
    :returns: Kwargs dict for ``client.converse()``.
    """
    converse_messages, system_prompts = _messages_to_converse(messages)

    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": converse_messages,
    }
    if system_prompts:
        kwargs["system"] = system_prompts

    # Inference config
    inference_config: dict[str, Any] = {}
    if "temperature" in extra:
        inference_config["temperature"] = extra.pop("temperature")
    if "top_p" in extra:
        inference_config["topP"] = extra.pop("top_p")
    if max_tokens := extra.pop("max_tokens", None) or extra.pop("max_completion_tokens", None):
        inference_config["maxTokens"] = max_tokens
    if stop := extra.pop("stop", None):
        inference_config["stopSequences"] = stop if isinstance(stop, list) else [stop]
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    # Tools
    if tools:
        kwargs["toolConfig"] = {"tools": _convert_tools(tools)}

    return kwargs


def _messages_to_converse(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """
    Convert Chat Completions messages to Bedrock Converse format.

    :param messages: Chat Completions messages.
    :returns: Tuple of (converse_messages, system_prompts).
    """
    system_prompts: list[dict[str, Any]] = []
    converse_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            system_prompts.append({"text": msg["content"]})
        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if text := msg.get("content"):
                content_blocks.append({"text": text})
            for tc in msg.get("tool_calls") or []:
                func = tc["function"]
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tc["id"],
                            "name": func["name"],
                            "input": json.loads(func["arguments"]),
                        }
                    }
                )
            if content_blocks:
                converse_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
        elif role == "tool":
            converse_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": msg["tool_call_id"],
                                "content": [{"text": msg["content"]}],
                            }
                        }
                    ],
                }
            )
        else:
            converse_messages.append(
                {
                    "role": "user",
                    "content": [{"text": msg.get("content") or ""}],
                }
            )

    return converse_messages, system_prompts or None


def _convert_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert OpenAI tool schemas to Bedrock toolSpec format.

    :param tools: OpenAI-format tool definitions.
    :returns: Bedrock tool definitions.
    """
    bedrock_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        bedrock_tools.append(
            {
                "toolSpec": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "inputSchema": {"json": func.get("parameters", {})},
                }
            }
        )
    return bedrock_tools


# ── Response translation ──────────────────────────────────


def _converse_to_chat(
    response: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """
    Convert Bedrock Converse response to Chat Completions format.

    :param response: Bedrock Converse response dict.
    :param model: Model ID for the response.
    :returns: Chat Completions response dict.
    """
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                {
                    "id": tu["toolUseId"],
                    "type": "function",
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
            )

    stop_reason = response.get("stopReason", "end_turn")
    finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"

    usage = response.get("usage", {})

    return {
        "id": f"bedrock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ("\n".join(text_parts) if text_parts else None),
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("inputTokens"),
            "completion_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
        },
    }


# ── HTTP (via boto3) ──────────────────────────────────────


def _send_converse(
    client: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Send a non-streaming Converse request.

    :param client: boto3 ``bedrock-runtime`` client.
    :param kwargs: Converse API kwargs.
    :returns: Chat Completions response dict.
    """
    model = kwargs["modelId"]
    response = client.converse(**kwargs)
    return _converse_to_chat(response, model)


def _stream_converse(
    client: Any,
    kwargs: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """
    Send a streaming Converse request and yield Chat Completions chunks.

    :param client: boto3 ``bedrock-runtime`` client.
    :param kwargs: Converse API kwargs.
    :returns: Iterator of Chat Completions chunk dicts.
    """
    model = kwargs["modelId"]
    response = client.converse_stream(**kwargs)

    for event in response.get("stream", []):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if text := delta.get("text"):
                yield {
                    "id": f"bedrock-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
            finish = "tool_calls" if stop_reason == "tool_use" else "stop"
            yield {
                "id": f"bedrock-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish,
                    }
                ],
            }
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            if usage:
                yield {
                    "id": f"bedrock-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": usage.get("inputTokens"),
                        "completion_tokens": usage.get("outputTokens"),
                        "total_tokens": usage.get("totalTokens"),
                    },
                }
