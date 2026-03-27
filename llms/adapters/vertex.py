"""
Google Vertex AI adapter.

Uses the same Gemini payload format but with GCP auth (Application
Default Credentials or service account) and Vertex AI endpoints.
Ported from MLflow AI Gateway's VertexAIProvider.

Connection config (``project``, ``location``) must be provided via
``connection_params`` at call time — typically from the
``connection:`` block in the agent spec's ``llm:`` config.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agent_plane.errors import AgentPlaneError, ErrorCode
from llms.adapters.gemini import GeminiAdapter

_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Timeout for the GCP OAuth token refresh HTTP request.
# Tighter than the SDK default (120s) so a hanging metadata
# server doesn't block the LLM call too long.
_AUTH_REFRESH_TIMEOUT = 30


class VertexAdapter(GeminiAdapter):
    """
    Adapter for Google Vertex AI.

    Inherits Gemini translation logic but uses Vertex AI endpoints
    and GCP OAuth authentication.

    Requires ``connection_params`` with:
    - ``"project"``: GCP project ID, e.g. ``"my-gcp-project"``.
    - ``"location"``: GCP region, e.g. ``"us-central1"``.

    Or alternatively a full ``"base_url"`` override.

    These come from the ``connection:`` block in the agent spec's
    ``llm:`` config — not from environment variables.
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

        import functools

        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(scopes=_DEFAULT_SCOPES)  # type: ignore[no-untyped-call]
        request = google.auth.transport.requests.Request()  # type: ignore[no-untyped-call]
        # 30s timeout for the OAuth token refresh HTTP request.
        # The SDK default is 120s; we tighten it so a hanging
        # metadata server doesn't block the LLM call too long.
        credentials.refresh(functools.partial(request, timeout=_AUTH_REFRESH_TIMEOUT))
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
        Not used — Vertex AI requires ``connection_params``.

        :returns: Never returns.
        :raises AgentPlaneError: Always — Vertex requires connection_params
            with ``"project"`` and ``"location"``.
        """
        raise AgentPlaneError(
            "Vertex AI requires 'project' and 'location' in"
            " connection_params (from llm.connection config)",
            code=ErrorCode.INVALID_INPUT,
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
        Send a request to Vertex AI.

        :param messages: Chat Completions format messages.
        :param model: Model name, e.g. ``"gemini-2.5-pro"``.
        :param tools: Tool schemas or ``None``.
        :param stream: Enable streaming.
        :param extra: Additional kwargs.
        :param connection_params: Required. Must contain
            ``"project"`` + ``"location"`` or ``"base_url"``.
        :param timeout: Request timeout in seconds. ``None`` uses
            the module default.
        :returns: Chat Completions response dict or chunk iterator.
        :raises AgentPlaneError: If ``connection_params`` is missing or
            lacks required keys.
        """
        resolved_params = _resolve_vertex_params(connection_params)
        return super().chat_completions(
            messages,
            model,
            tools,
            stream,
            extra,
            connection_params=resolved_params,
            timeout=timeout,
        )


def _resolve_vertex_params(
    connection_params: dict[str, str] | None,
) -> dict[str, str]:
    """
    Convert Vertex-specific ``"project"``/``"location"`` keys into
    a ``"base_url"`` that the parent Gemini adapter understands.

    All connection info must come from ``connection_params`` — no
    environment variable fallbacks.

    :param connection_params: Raw connection params from the caller.
        Must contain ``"project"`` + ``"location"`` or ``"base_url"``.
    :returns: Params with ``"base_url"`` resolved.
    :raises AgentPlaneError: If params are missing or incomplete.
    """
    if not connection_params:
        raise AgentPlaneError(
            "Vertex AI requires connection_params with"
            " 'project' and 'location' (from llm.connection config)",
            code=ErrorCode.INVALID_INPUT,
        )

    # If caller provided a full base_url, pass through as-is.
    if "base_url" in connection_params:
        return connection_params

    project = connection_params.get("project")
    location = connection_params.get("location")
    if not project:
        raise AgentPlaneError(
            "Vertex AI requires 'project' in connection_params (from llm.connection config)",
            code=ErrorCode.INVALID_INPUT,
        )
    if not location:
        raise AgentPlaneError(
            "Vertex AI requires 'location' in connection_params (from llm.connection config)",
            code=ErrorCode.INVALID_INPUT,
        )
    return {
        **connection_params,
        "base_url": _build_vertex_url(project, location),
    }


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
