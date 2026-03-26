"""Tests for llms.adapters.vertex — connection_params resolution."""

import pytest

from agent_plane.errors import AgentPlaneError
from llms.adapters.vertex import _build_vertex_url, _resolve_vertex_params


def test_resolve_raises_when_no_params() -> None:
    """
    ``None`` input raises ``AgentPlaneError`` — Vertex requires
    connection_params.
    """
    with pytest.raises(AgentPlaneError, match="requires connection_params"):
        _resolve_vertex_params(None)


def test_resolve_raises_when_empty_params() -> None:
    """
    Empty dict raises ``AgentPlaneError`` — Vertex requires
    connection_params with project/location or base_url.
    """
    with pytest.raises(AgentPlaneError, match="requires connection_params"):
        _resolve_vertex_params({})


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
    expected_url = _build_vertex_url("my-proj", "europe-west1")
    assert result["base_url"] == expected_url
    # Original keys are preserved
    assert result["project"] == "my-proj"
    assert result["location"] == "europe-west1"


def test_resolve_raises_when_project_missing() -> None:
    """
    AgentPlaneError when ``"location"`` is provided but ``"project"`` is not.
    No env var fallback.
    """
    params = {"location": "us-east1"}
    with pytest.raises(AgentPlaneError, match="requires 'project'"):
        _resolve_vertex_params(params)


def test_resolve_raises_when_location_missing() -> None:
    """
    AgentPlaneError when ``"project"`` is provided but ``"location"`` is not.
    No env var fallback.
    """
    params = {"project": "my-proj"}
    with pytest.raises(AgentPlaneError, match="requires 'location'"):
        _resolve_vertex_params(params)


def test_resolve_raises_when_no_recognized_keys() -> None:
    """
    Params without ``"project"``, ``"location"``, or ``"base_url"``
    raise AgentPlaneError — Vertex needs at least one of these.
    """
    params = {"some_other_key": "value"}
    with pytest.raises(AgentPlaneError, match="requires 'project'"):
        _resolve_vertex_params(params)


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
