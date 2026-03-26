"""
Databricks Model Serving adapter.

Extends the OpenAI-compatible adapter with Databricks-specific
authentication (PAT token). Ported from MLflow AI Gateway's
DatabricksProvider.
"""

from __future__ import annotations

import os

from llms.adapters.openai import OpenAICompatibleAdapter


class DatabricksAdapter(OpenAICompatibleAdapter):
    """
    Adapter for Databricks Model Serving.

    Config from environment:
    - ``DATABRICKS_HOST``: Workspace URL, e.g.
      ``"https://my-workspace.databricks.com"``.
    - ``DATABRICKS_TOKEN``: Personal access token.

    Both can be overridden per-call via ``connection_params``
    with keys ``"api_key"`` and ``"base_url"``.
    """

    def __init__(self) -> None:
        host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        base_url = f"{host}/serving-endpoints"
        super().__init__(
            base_url=base_url,
            api_key_env="DATABRICKS_TOKEN",
        )
