"""
Base adapter interface for LLM provider adapters.

Each adapter translates between Chat Completions format and the
provider's native API, and handles HTTP communication. All methods
are async — adapters use ``httpx.AsyncClient`` for non-blocking I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BaseAdapter(ABC):
    """
    Abstract base class for provider adapters.

    Subclasses implement :meth:`chat_completions` to send a request
    in the provider's native format and return a Chat Completions
    response dict (or async iterator of chunk dicts for streaming).
    """

    @abstractmethod
    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        extra: dict[str, Any],
        *,
        connection_params: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Send a chat completions request to the provider.

        :param messages: Chat Completions format messages, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        :param model: The model name (without provider prefix),
            e.g. ``"claude-sonnet-4-20250514"``.
        :param tools: OpenAI-format tool schemas, or ``None``.
        :param stream: If ``True``, return an async iterator of
            chunk dicts. If ``False``, return a single response
            dict.
        :param extra: Additional provider-specific kwargs, e.g.
            ``{"temperature": 0.7, "reasoning_effort": "high"}``.
        :param connection_params: Per-call connection overrides.
            Each adapter defines which keys it supports. ``None``
            means use the adapter's default credentials (env vars,
            etc.). Common keys by provider:

            - OpenAI-compatible: ``{"api_key": "...",
              "base_url": "..."}``
            - Anthropic: ``{"api_key": "..."}``
            - Databricks: ``{"api_key": "...",
              "base_url": "..."}``
            - Bedrock: ``{"aws_region": "...",
              "aws_access_key_id": "...",
              "aws_secret_access_key": "..."}``
            - Vertex: ``{"project": "...", "location": "..."}``
        :param timeout: Request timeout in seconds. ``None`` uses
            the adapter's default (120s non-streaming, 300s
            streaming).
        :returns: A Chat Completions response dict when
            ``stream=False``, or an async iterator of Chat
            Completions chunk dicts when ``stream=True``.
        """
        ...
