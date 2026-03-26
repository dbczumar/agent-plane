"""Tests for llms.adapters.vertex — connection_params resolution."""

import os

from llms.adapters.vertex import _build_vertex_url, _resolve_vertex_params


def test_resolve_returns_none_for_none() -> None:
    """
    ``None`` input produces ``None`` output — no overrides.
    """
    assert _resolve_vertex_params(None) is None


def test_resolve_passes_through_base_url() -> None:
    """
    If ``connection_params`` already has ``"base_url"``, pass through unchanged.
    """
    params = {"base_url": "https://custom.endpoint.com/v1"}
    assert _resolve_vertex_params(params) is params


def test_resolve_builds_url_from_project_and_location() -> None:
    """
    ``"project"`` and ``"location"`` are converted to a Vertex ``"base_url"``.
    """
    params = {"project": "my-proj", "location": "europe-west1"}
    result = _resolve_vertex_params(params)
    assert result is not None
    expected_url = _build_vertex_url("my-proj", "europe-west1")
    assert result["base_url"] == expected_url
    # Original keys are preserved
    assert result["project"] == "my-proj"
    assert result["location"] == "europe-west1"


def test_resolve_falls_back_to_env_for_missing_project(
    monkeypatch: object,
) -> None:
    """
    If only ``"location"`` is provided, ``"project"`` falls back to
    the ``VERTEX_PROJECT`` env var.
    """
    monkeypatch.setenv("VERTEX_PROJECT", "env-proj")  # type: ignore[union-attr]
    params = {"location": "asia-east1"}
    result = _resolve_vertex_params(params)
    assert result is not None
    assert result["base_url"] == _build_vertex_url("env-proj", "asia-east1")


def test_resolve_no_project_or_location_passes_through() -> None:
    """
    Params without ``"project"``, ``"location"``, or ``"base_url"``
    are passed through unchanged.
    """
    params = {"some_other_key": "value"}
    assert _resolve_vertex_params(params) is params


def test_build_vertex_url_structure() -> None:
    """
    The Vertex URL follows the expected GCP pattern.
    """
    url = _build_vertex_url("my-proj", "us-central1")
    assert url == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-proj"
        "/locations/us-central1"
        "/publishers/google/models"
    )
