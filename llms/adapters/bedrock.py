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

from llms.adapters._content import parse_data_uri
from llms.adapters.base import BaseAdapter

# Default connect timeout: 30s to establish TCP connection.
_BOTO_CONNECT_TIMEOUT = 30


def _boto_config(
    read_timeout: int,
    max_retries: int | None = None,
) -> Any:
    """
    Build a botocore ``Config`` with timeout and retry settings.

    :param read_timeout: Read timeout in seconds for the HTTP
        response, e.g. ``300``. Propagated from the agent spec's
        ``llm.timeout``.
    :param max_retries: Maximum retry attempts at the boto3 transport
        layer. ``None`` disables boto3-level retries (the workflow's
        ``execute_with_retry`` handles retries instead).
    :returns: A ``botocore.config.Config`` instance.
    """
    from botocore.config import Config  # type: ignore[import-untyped]

    retries = (
        {"max_attempts": max_retries, "mode": "adaptive"}
        if max_retries is not None
        else {"max_attempts": 0}
    )
    return Config(
        connect_timeout=_BOTO_CONNECT_TIMEOUT,
        read_timeout=read_timeout,
        retries=retries,
    )


class BedrockAdapter(BaseAdapter):
    """
    Adapter for AWS Bedrock using the Converse API.

    Auth is handled via standard AWS environment variables
    (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``AWS_DEFAULT_REGION``) or an IAM role.

    Unlike other adapters, boto3 clients are not cached because
    the read timeout is propagated from the agent spec and may
    differ between calls.
    """

    def _make_client(
        self,
        timeout: int,
        connection_params: dict[str, str] | None = None,
    ) -> Any:
        """
        Create a boto3 ``bedrock-runtime`` client.

        :param timeout: Read timeout in seconds, propagated from
            the agent spec's ``llm.timeout``, e.g. ``300``.
        :param connection_params: Optional overrides. Supported
            keys: ``"aws_region"``, ``"aws_access_key_id"``,
            ``"aws_secret_access_key"``, ``"aws_session_token"``.
        :returns: A boto3 ``bedrock-runtime`` client.
        """
        import boto3  # type: ignore[import-untyped]

        boto_kwargs: dict[str, str] = {}
        if connection_params:
            if region := connection_params.get("aws_region"):
                boto_kwargs["region_name"] = region
            if access_key := connection_params.get("aws_access_key_id"):
                boto_kwargs["aws_access_key_id"] = access_key
            if secret_key := connection_params.get("aws_secret_access_key"):
                boto_kwargs["aws_secret_access_key"] = secret_key
            if session_token := connection_params.get("aws_session_token"):
                boto_kwargs["aws_session_token"] = session_token
        return boto3.client(
            "bedrock-runtime",
            config=_boto_config(read_timeout=timeout),
            **boto_kwargs,
        )

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
        *,
        connection_params: dict[str, str] | None = None,
        timeout: int | None = None,
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
        :param timeout: Read timeout in seconds, propagated from
            the agent spec's ``llm.timeout``. ``None`` uses the
            module default (300s).
        :returns: Chat Completions response dict or chunk iterator.
        """
        converse_kwargs = _build_converse_kwargs(messages, model, tools, extra)
        # Default 300s matches the streaming default in other adapters.
        effective_timeout = timeout if timeout is not None else 300
        client = self._make_client(effective_timeout, connection_params)

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
        role = msg["role"]

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
            blocks = _content_to_converse_blocks(msg.get("content"))
            converse_messages.append({"role": "user", "content": blocks})

    return converse_messages, system_prompts or None


def _content_to_converse_blocks(
    content: list[dict[str, Any]] | str | None,
) -> list[dict[str, Any]]:
    """
    Convert Chat Completions content to Bedrock Converse content blocks.

    Handles string content (text-only), list content (multimodal),
    and ``None`` (empty blocks).

    :param content: Chat Completions content — string, list of
        content part dicts, or ``None``.
    :returns: Bedrock Converse content block list.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    return [_translate_part_to_converse(part) for part in content]


def _translate_part_to_converse(part: dict[str, Any]) -> dict[str, Any]:
    """
    Translate a single Chat Completions content part to Bedrock
    Converse format.

    - ``text`` → ``{"text": "..."}``
    - ``image_url`` with data URI → ``{"image": {"format": "...",
      "source": {"bytes": "..."}}}``
    - ``input_file`` with file_data → ``{"document": {"format": "...",
      "name": "...", "source": {"bytes": "..."}}}``
    - Unrecognized → rendered as text placeholder.

    :param part: A Chat Completions content part dict.
    :returns: A Bedrock Converse content block dict.
    """
    part_type = part.get("type")

    if part_type == "text":
        return {"text": part["text"]}

    if part_type == "image_url":
        image_url = part["image_url"]
        url = image_url["url"]
        parsed = parse_data_uri(url)
        if parsed is not None:
            # MIME types are always type/subtype per RFC 2045.
            fmt = parsed.media_type.split("/")[-1]
            return {
                "image": {
                    "format": fmt,
                    "source": {"bytes": parsed.data},
                },
            }
        # Bedrock does not support external URLs in image blocks.
        return {"text": f"[image: {url}]"}

    if part_type == "input_file":
        # application/octet-stream is the RFC 2046 default for
        # unknown binary content — safe fallback when file store
        # metadata lacks a content_type.
        media_type = part.get("content_type", "application/octet-stream")
        # MIME types are always type/subtype per RFC 2045.
        fmt = media_type.split("/")[-1]
        result: dict[str, Any] = {
            "document": {
                "format": fmt,
                "source": {"bytes": part["file_data"]},
            },
        }
        if filename := part.get("filename"):
            result["document"]["name"] = filename
        return result

    # Unrecognized part type — render as text placeholder.
    return {"text": f"[unsupported content: {part_type}]"}


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
        spec: dict[str, Any] = {
            "name": func["name"],
            "inputSchema": {"json": func.get("parameters", {})},
        }
        if desc := func.get("description"):
            spec["description"] = desc
        bedrock_tools.append({"toolSpec": spec})
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

    # Bedrock Converse API always returns stopReason; fail loud if missing.
    stop_reason = response["stopReason"]
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
            # Bedrock always includes stopReason in messageStop; fail loud if missing.
            stop_reason = event["messageStop"]["stopReason"]
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
