"""
Google Vertex AI adapter.

Uses the same Gemini payload format but with GCP auth (Application
Default Credentials or service account) and Vertex AI endpoints.
Ported from MLflow AI Gateway's VertexAIProvider.
"""

from __future__ import annotations

import os
from typing import Any

from llms.adapters.gemini import GeminiAdapter

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexAdapter(GeminiAdapter):
    """
    Adapter for Google Vertex AI.

    Inherits Gemini translation logic but uses Vertex AI endpoints
    and GCP OAuth authentication.

    Config from environment:
    - ``VERTEX_PROJECT``: GCP project ID (required).
    - ``VERTEX_LOCATION``: GCP region, defaults to ``"us-central1"``.
    - ``GOOGLE_APPLICATION_CREDENTIALS``: Path to service account
      JSON (optional, uses ADC if not set).
    """

    def __init__(self) -> None:
        self._cached_credentials: Any = None

    def _get_credentials(self) -> Any:
        """
        Get GCP credentials, refreshing if needed.

        :returns: A ``google.auth.credentials.Credentials`` object
            with a valid access token.
        """
        if (
            self._cached_credentials is not None
            and self._cached_credentials.valid
        ):
            return self._cached_credentials

        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=_DEFAULT_SCOPES)
        credentials.refresh(google.auth.transport.requests.Request())
        self._cached_credentials = credentials
        return credentials

    def _get_headers(self) -> dict[str, str]:
        """
        Build Vertex AI headers with OAuth bearer token.

        :returns: Headers dict with Authorization.
        """
        credentials = self._get_credentials()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.token}",
        }

    def _get_base_url(self) -> str:
        """
        Build the Vertex AI endpoint URL.

        :returns: The Vertex AI base URL for the configured
            project and location.
        """
        project = os.environ.get("VERTEX_PROJECT", "")
        location = os.environ.get("VERTEX_LOCATION", "us-central1")
        return (
            f"https://{location}-aiplatform.googleapis.com"
            f"/v1/projects/{project}"
            f"/locations/{location}"
            f"/publishers/google/models"
        )
