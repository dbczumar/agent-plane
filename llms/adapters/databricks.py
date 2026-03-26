"""
Databricks Model Serving adapter.

Extends the OpenAI-compatible adapter with Databricks-specific
authentication (PAT token). Ported from MLflow AI Gateway's
DatabricksProvider.

Connection config (``base_url`` and ``api_key``) must be provided
via ``connection_params`` at call time — typically from the
``connection:`` block in the agent spec's ``llm:`` config.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from llms.adapters.openai import OpenAICompatibleAdapter


class DatabricksAdapter(OpenAICompatibleAdapter):
    """
    Adapter for Databricks Model Serving.

    Requires ``connection_params`` with:
    - ``"base_url"``: Workspace serving URL, e.g.
      ``"https://my-workspace.databricks.com/serving-endpoints"``.
    - ``"api_key"``: Personal access token or OAuth token.

    These come from the ``connection:`` block in the agent spec's
    ``llm:`` config — not from environment variables.
    """

    def __init__(self) -> None:
        # No env var defaults — connection info comes from
        # connection_params at call time.
        super().__init__(
            base_url="",
            api_key_env=None,
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
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """
        Send a Chat Completions request to Databricks Model Serving.

        :param messages: Chat Completions format messages.
        :param model: Model name, e.g. ``"databricks-gpt-5-4"``.
        :param tools: Tool schemas or ``None``.
        :param stream: Enable streaming.
        :param extra: Additional kwargs.
        :param connection_params: Required. Must contain
            ``"base_url"`` and ``"api_key"``.
        :returns: Response dict or iterator of chunk dicts.
        :raises ValueError: If ``connection_params`` is missing or
            lacks ``"base_url"``.
        """
        if not connection_params or "base_url" not in connection_params:
            raise ValueError(
                "Databricks adapter requires 'base_url' in"
                " connection_params (from llm.connection config)"
            )
        return super().chat_completions(
            messages,
            model,
            tools,
            stream,
            extra,
            connection_params=connection_params,
        )
