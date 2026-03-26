"""
Google Vertex AI adapter.

Uses the same Gemini payload format but with GCP auth (Application
Default Credentials or service account) and Vertex AI endpoints.
Ported from MLflow AI Gateway's VertexAIProvider.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from llms.adapters.gemini import GeminiAdapter

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexAdapter(GeminiAdapter):
    """
    Adapter for Google Vertex AI.

    Inherits Gemini translation logic but uses Vertex AI endpoints
    and GCP OAuth authentication.

    Config from environment (used as defaults):
    - ``VERTEX_PROJECT``: GCP project ID (required).
    - ``VERTEX_LOCATION``: GCP region, defaults to ``"us-central1"``.
    - ``GOOGLE_APPLICATION_CREDENTIALS``: Path to service account
      JSON (optional, uses ADC if not set).

    Per-call ``connection_params`` keys: ``"project"``,
    ``"location"``, or a full ``"base_url"`` override.
    """

    def __init__(self) -> None:
        self._cached_credentials: Any = None

    def _get_credentials(self) -> Any:
        """
        Get GCP credentials, refreshing if needed.

        :returns: A ``google.auth.credentials.Credentials`` object
            with a valid access token.
        """
        if self._cached_credentials is not None and self._cached_credentials.valid:
            return self._cached_credentials

        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=_DEFAULT_SCOPES)  # type: ignore[no-untyped-call]
        credentials.refresh(
            google.auth.transport.requests.Request()  # type: ignore[no-untyped-call]
        )
        self._cached_credentials = credentials
        return credentials

    def _get_headers(
        self,
        api_key_override: str | None = None,
    ) -> dict[str, str]:
        """
        Build Vertex AI headers with OAuth bearer token.

        :param api_key_override: Not used by Vertex AI (uses GCP
            OAuth). Accepted for interface compatibility.
        :returns: Headers dict with Authorization.
        """
        credentials = self._get_credentials()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.token}",
        }

    def _get_base_url(self) -> str:
        """
        Build the Vertex AI endpoint URL from environment variables.

        :returns: The Vertex AI base URL for the configured
            project and location.
        """
        project = os.environ.get("VERTEX_PROJECT", "")
        location = os.environ.get("VERTEX_LOCATION", "us-central1")
        return _build_vertex_url(project, location)

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
        Send a request to Vertex AI.

        :param messages: Chat Completions format messages.
        :param model: Model name, e.g. ``"gemini-2.5-pro"``.
        :param tools: Tool schemas or ``None``.
        :param stream: Enable streaming.
        :param extra: Additional kwargs.
        :param connection_params: Per-call overrides. Supported keys:
            ``"project"``, ``"location"`` (builds Vertex URL), or
            ``"base_url"`` (used directly).
        :returns: Chat Completions response dict or chunk iterator.
        """
        resolved_params = _resolve_vertex_params(connection_params)
        return super().chat_completions(
            messages,
            model,
            tools,
            stream,
            extra,
            connection_params=resolved_params,
        )


def _resolve_vertex_params(
    connection_params: dict[str, str] | None,
) -> dict[str, str] | None:
    """
    Convert Vertex-specific ``"project"``/``"location"`` keys into
    a ``"base_url"`` that the parent Gemini adapter understands.

    If ``connection_params`` already contains ``"base_url"``, it is
    passed through unchanged. If it contains ``"project"`` and/or
    ``"location"``, a Vertex URL is built from them.

    :param connection_params: Raw connection params from the caller.
    :returns: Params with ``"base_url"`` resolved, or ``None``.
    """
    if not connection_params:
        return None

    # If caller provided a full base_url, pass through as-is.
    if "base_url" in connection_params:
        return connection_params

    project = connection_params.get("project")
    location = connection_params.get("location")
    if project or location:
        resolved_project = project if project else os.environ.get("VERTEX_PROJECT", "")
        resolved_location = (
            location if location else os.environ.get("VERTEX_LOCATION", "us-central1")
        )
        return {
            **connection_params,
            "base_url": _build_vertex_url(resolved_project, resolved_location),
        }

    return connection_params


def _build_vertex_url(project: str, location: str) -> str:
    """
    Build the Vertex AI endpoint URL from project and location.

    :param project: GCP project ID.
    :param location: GCP region, e.g. ``"us-central1"``.
    :returns: The Vertex AI base URL.
    """
    return (
        f"https://{location}-aiplatform.googleapis.com"
        f"/v1/projects/{project}"
        f"/locations/{location}"
        f"/publishers/google/models"
    )
